#include <nanobench.h>

#include "safe-shm/cyclic_buffer.hpp"
#include "safe-shm/image.hpp"
#include "safe-shm/seqlock.hpp"
#include "safe-shm/stamped.hpp"
#include "safe-shm/time_series.hpp"

#include <cstring>
#include <fcntl.h>
#include <mqueue.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

// ─── Payload types ──────────────────────────────────────────────────────────

struct SmallPayload
{
    double values[8]; // 64 bytes
};

struct MediumPayload
{
    uint8_t data[4096]; // 4 KB
};

struct LargePayload
{
    uint8_t data[1024 * 1024]; // 1 MB
};

static_assert(sizeof(SmallPayload) == 64);
static_assert(sizeof(MediumPayload) == 4096);
static_assert(sizeof(LargePayload) == 1024 * 1024);

// ─── Benchmarks ─────────────────────────────────────────────────────────────

int main()
{
    // ── CyclicBuffer benchmarks ──────────────────────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("CyclicBuffer (insert + get_latest, lock-free)")
            .warmup(100)
            .minEpochIterations(500);

        {
            safe_shm::CyclicBuffer<SmallPayload, 64> buf("bench_cb_small");
            SmallPayload src{};
            std::fill(std::begin(src.values), std::end(src.values), 1.0);
            b.batch(sizeof(SmallPayload)).unit("byte").run(
                "CyclicBuffer 64 B", [&]
                {
                    buf.insert(src);
                    auto val = buf.get_latest();
                    ankerl::nanobench::doNotOptimizeAway(val);
                });
        }

        {
            safe_shm::CyclicBuffer<MediumPayload, 16> buf("bench_cb_med");
            MediumPayload src{};
            std::memset(src.data, 0xAA, sizeof(src.data));
            b.batch(sizeof(MediumPayload)).unit("byte").run(
                "CyclicBuffer 4 KB", [&]
                {
                    buf.insert(src);
                    auto val = buf.get_latest();
                    ankerl::nanobench::doNotOptimizeAway(val);
                });
        }

        {
            safe_shm::CyclicBuffer<LargePayload, 4> buf("bench_cb_large");
            LargePayload src{};
            std::memset(src.data, 0xBB, sizeof(src.data));
            b.batch(sizeof(LargePayload)).unit("byte").run(
                "CyclicBuffer 1 MB", [&]
                {
                    buf.insert(src);
                    auto val = buf.get_latest();
                    ankerl::nanobench::doNotOptimizeAway(val);
                });
        }
    }

    // ── CyclicBuffer FHD image ───────────────────────────────────────────
    {
        using Image = safe_shm::img::ImageFHD_RGB;
        ankerl::nanobench::Bench b;
        b.title("CyclicBuffer FHD RGB (~6 MB)")
            .warmup(10)
            .minEpochIterations(50);

        safe_shm::CyclicBuffer<Image, 4> buf("bench_cb_fhd");
        auto img = std::make_unique<Image>();
        img->timestamp = 1;
        img->frame_number = 1;
        std::fill(img->data.begin(), img->data.end(), 0xFF);

        b.batch(sizeof(Image)).unit("byte").run(
            "CyclicBuffer FHD RGB", [&]
            {
                buf.insert(*img);
                auto val = buf.get_latest();
                ankerl::nanobench::doNotOptimizeAway(val.timestamp);
            });
    }

    // ── Seqlock vs CyclicBuffer comparison ───────────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("Seqlock vs CyclicBuffer (64 B store+load)")
            .warmup(100)
            .minEpochIterations(500);

        {
            safe_shm::Seqlock<SmallPayload> sl("bench_cmp_seqlock");
            SmallPayload src{};
            std::fill(std::begin(src.values), std::end(src.values), 3.0);
            b.batch(sizeof(SmallPayload)).unit("byte").run(
                "Seqlock 64 B", [&]
                {
                    sl.store(src);
                    auto val = sl.load();
                    ankerl::nanobench::doNotOptimizeAway(val);
                });
        }

        {
            safe_shm::CyclicBuffer<SmallPayload, 64> buf("bench_cmp_cb");
            SmallPayload src{};
            std::fill(std::begin(src.values), std::end(src.values), 3.0);
            b.batch(sizeof(SmallPayload)).unit("byte").run(
                "CyclicBuffer 64 B", [&]
                {
                    buf.insert(src);
                    auto val = buf.get_latest();
                    ankerl::nanobench::doNotOptimizeAway(val);
                });
        }
    }

    // ── TimeSeries lookup benchmarks ─────────────────────────────────────
    {
        using S = safe_shm::Stamped<SmallPayload>;
        ankerl::nanobench::Bench b;
        b.title("TimeSeries lookup (64-element buffer)")
            .warmup(100)
            .minEpochIterations(500);

        // Pre-fill buffer
        safe_shm::CyclicBuffer<S, 64> buf("bench_ts_lookup");
        for (uint64_t i = 0; i < 64; ++i)
        {
            SmallPayload p{};
            p.values[0] = static_cast<double>(i);
            S s{i * 1000, i, p};
            buf.insert(s);
        }

        safe_shm::TimeSeries<S, 64> ts("bench_ts_lookup");

        b.batch(1).unit("op").run(
            "find_closest (binary search)", [&]
            {
                auto val = ts.find_closest(uint64_t{32500});
                ankerl::nanobench::doNotOptimizeAway(val);
            });

        b.batch(1).unit("op").run(
            "find_interpolation_pair", [&]
            {
                auto val = ts.find_interpolation_pair(uint64_t{32500});
                ankerl::nanobench::doNotOptimizeAway(val);
            });

        b.batch(1).unit("op").run(
            "get_latest_if_fresh", [&]
            {
                auto val = ts.get_latest_if_fresh(uint64_t{60000});
                ankerl::nanobench::doNotOptimizeAway(val);
            });
    }

    // ── POSIX message queue benchmark ────────────────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("POSIX message queue (mq_send + mq_receive)")
            .warmup(50)
            .minEpochIterations(200);

        auto run_mq = [&](char const *name, std::size_t size)
        {
            // Clean up
            mq_unlink("/bench_mq");

            struct mq_attr attr{};
            attr.mq_flags = 0;
            attr.mq_maxmsg = 1;
            attr.mq_msgsize = static_cast<long>(size);
            attr.mq_curmsgs = 0;

            mqd_t mq = mq_open("/bench_mq", O_CREAT | O_RDWR, 0666, &attr);
            if (mq == static_cast<mqd_t>(-1))
            {
                // mq may not be available, skip
                return;
            }

            auto src = std::make_unique<char[]>(size);
            auto dst = std::make_unique<char[]>(size);
            std::memset(src.get(), 0xAA, size);

            b.batch(size).unit("byte").run(name, [&]
                                           {
                mq_send(mq, src.get(), size, 0);
                unsigned prio;
                mq_receive(mq, dst.get(), size, &prio);
                ankerl::nanobench::doNotOptimizeAway(dst[0]); });

            mq_close(mq);
            mq_unlink("/bench_mq");
        };

        run_mq("mq 64 B", 64);
        run_mq("mq 4 KB", 4096);
    }

    // ── Unix domain socket benchmark ─────────────────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("Unix domain socket (socketpair send + recv)")
            .warmup(50)
            .minEpochIterations(200);

        auto run_uds = [&](char const *name, std::size_t size)
        {
            int fds[2];
            if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) < 0)
                return;

            // Set send buffer large enough
            int buf_size = static_cast<int>(std::max(size * 2, std::size_t{65536}));
            setsockopt(fds[0], SOL_SOCKET, SO_SNDBUF, &buf_size, sizeof(buf_size));
            setsockopt(fds[1], SOL_SOCKET, SO_RCVBUF, &buf_size, sizeof(buf_size));

            auto src = std::make_unique<char[]>(size);
            auto dst = std::make_unique<char[]>(size);
            std::memset(src.get(), 0xBB, size);

            b.batch(size).unit("byte").run(name, [&]
                                           {
                auto w = send(fds[0], src.get(), size, 0);
                std::size_t total = 0;
                while (total < size)
                {
                    auto r = recv(fds[1], dst.get() + total, size - total, 0);
                    if (r <= 0) break;
                    total += static_cast<std::size_t>(r);
                }
                ankerl::nanobench::doNotOptimizeAway(w);
                ankerl::nanobench::doNotOptimizeAway(dst[0]); });

            close(fds[0]);
            close(fds[1]);
        };

        run_uds("UDS 64 B", 64);
        run_uds("UDS 4 KB", 4096);
        // UDS 1 MB omitted — too slow for benchmarking (kernel copy overhead)
    }

    // ── pipe benchmark (for comparison table) ────────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("pipe (kernel IPC reference)")
            .warmup(50)
            .minEpochIterations(200);

        auto run_pipe = [&](char const *name, std::size_t size)
        {
            auto buf = std::make_unique<uint8_t[]>(size);
            auto recv_buf = std::make_unique<uint8_t[]>(size);
            std::memset(buf.get(), 0xEE, size);

            int fds[2];
            if (pipe(fds) < 0)
                return;
            fcntl(fds[0], F_SETPIPE_SZ, static_cast<int>(std::max(size, std::size_t{65536})));

            b.batch(size).unit("byte").run(name, [&]
                                           {
                auto w = write(fds[1], buf.get(), size);
                auto r = read(fds[0], recv_buf.get(), size);
                ankerl::nanobench::doNotOptimizeAway(w);
                ankerl::nanobench::doNotOptimizeAway(r); });

            close(fds[0]);
            close(fds[1]);
        };

        run_pipe("pipe 64 B", 64);
        run_pipe("pipe 4 KB", 4096);
        run_pipe("pipe 1 MB", 1024 * 1024);
    }

    // ── memcpy baseline (for comparison table) ───────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("memcpy baseline (local memory)")
            .warmup(100)
            .minEpochIterations(500);

        auto run = [&](char const *name, std::size_t size)
        {
            auto src = std::make_unique<uint8_t[]>(size);
            auto dst = std::make_unique<uint8_t[]>(size);
            std::memset(src.get(), 0xAA, size);

            b.batch(size).unit("byte").run(name, [&]
                                           { std::memcpy(dst.get(), src.get(), size);
                                             ankerl::nanobench::doNotOptimizeAway(dst[0]); });
        };

        run("memcpy 64 B", 64);
        run("memcpy 4 KB", 4096);
        run("memcpy 1 MB", 1024 * 1024);
    }

    return 0;
}
