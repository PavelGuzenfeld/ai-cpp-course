#include <nanobench.h>

#include "safe-shm/dblbuf_loader.hpp"
#include "safe-shm/image.hpp"
#include "safe-shm/image_shm.hpp"
#include "safe-shm/seqlock.hpp"
#include "safe-shm/shared_memory.hpp"
#include "safe-shm/storage.hpp"

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

// ─── Raw POSIX shm baseline ─────────────────────────────────────────────────

class RawPosixShm
{
public:
    explicit RawPosixShm(char const *name, std::size_t size)
        : name_(name), size_(size)
    {
        fd_ = shm_open(name, O_CREAT | O_RDWR, 0666);
        if (fd_ < 0)
            throw std::runtime_error("shm_open failed");
        if (ftruncate(fd_, static_cast<off_t>(size)) < 0)
            throw std::runtime_error("ftruncate failed");
        ptr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (ptr_ == MAP_FAILED)
            throw std::runtime_error("mmap failed");
    }

    ~RawPosixShm()
    {
        if (ptr_ && ptr_ != MAP_FAILED)
            munmap(ptr_, size_);
        if (fd_ >= 0)
            close(fd_);
        shm_unlink(name_.c_str());
    }

    void *get() noexcept { return ptr_; }

    RawPosixShm(RawPosixShm const &) = delete;
    RawPosixShm &operator=(RawPosixShm const &) = delete;

private:
    std::string name_;
    std::size_t size_;
    int fd_{-1};
    void *ptr_{nullptr};
};

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
    // ── memcpy baseline ─────────────────────────────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("memcpy baseline (local memory, no IPC)")
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
        run("memcpy 6 MB (FHD RGB)", sizeof(safe_shm::img::ImageFHD_RGB));
    }

    // ── raw POSIX shm ───────────────────────────────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("raw POSIX shm (mmap + memcpy, no sync)")
            .warmup(100)
            .minEpochIterations(500);

        auto run = [&](char const *name, std::size_t size)
        {
            RawPosixShm shm("/bench_raw_posix", size);
            auto src = std::make_unique<uint8_t[]>(size);
            std::memset(src.get(), 0xBB, size);

            b.batch(size).unit("byte").run(name, [&]
                                           { std::memcpy(shm.get(), src.get(), size);
                                             ankerl::nanobench::doNotOptimizeAway(
                                                 *static_cast<uint8_t *>(shm.get())); });
        };

        run("raw posix shm 64 B", 64);
        run("raw posix shm 4 KB", 4096);
        run("raw posix shm 1 MB", 1024 * 1024);
        run("raw posix shm 6 MB (FHD RGB)", sizeof(safe_shm::img::ImageFHD_RGB));
    }

    // ── SharedMemory<T> ─────────────────────────────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("SharedMemory<T> (safe-shm, no sync)")
            .warmup(100)
            .minEpochIterations(500);

        {
            safe_shm::SharedMemory<SmallPayload> shm("bench_shmem_small");
            SmallPayload src{};
            std::fill(std::begin(src.values), std::end(src.values), 3.14);
            b.batch(sizeof(SmallPayload)).unit("byte").run("SharedMemory 64 B", [&]
                                                           { shm.get() = src;
                                                             ankerl::nanobench::doNotOptimizeAway(shm.get()); });
        }
        {
            safe_shm::SharedMemory<MediumPayload> shm("bench_shmem_med");
            MediumPayload src{};
            std::memset(src.data, 0xCC, sizeof(src.data));
            b.batch(sizeof(MediumPayload)).unit("byte").run("SharedMemory 4 KB", [&]
                                                            { shm.get() = src;
                                                              ankerl::nanobench::doNotOptimizeAway(shm.get()); });
        }
        {
            safe_shm::SharedMemory<LargePayload> shm("bench_shmem_large");
            LargePayload src{};
            std::memset(src.data, 0xDD, sizeof(src.data));
            b.batch(sizeof(LargePayload)).unit("byte").run("SharedMemory 1 MB", [&]
                                                           { shm.get() = src;
                                                             ankerl::nanobench::doNotOptimizeAway(shm.get()); });
        }
    }

    // ── Storage + DblBufLoader ──────────────────────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("Storage + DblBufLoader (futex lock + double-buffer)")
            .warmup(50)
            .minEpochIterations(200);

        auto run = [&](char const *name, char const *shm_name, auto const &src)
        {
            using T = std::decay_t<decltype(src)>;
            safe_shm::Storage<T> storage(shm_name);
            safe_shm::DblBufLoader<T> loader(shm_name, [](std::string_view) {});

            b.batch(sizeof(T)).unit("byte").run(name, [&]
                                                {
                storage.store(src);
                auto snap = loader.load();
                loader.wait();
                ankerl::nanobench::doNotOptimizeAway(*snap); });
        };

        SmallPayload small{};
        std::fill(std::begin(small.values), std::end(small.values), 1.0);
        run("Storage+DblBufLoader 64 B", "bench_dblbuf_s", small);

        MediumPayload medium{};
        std::memset(medium.data, 0xAA, sizeof(medium.data));
        run("Storage+DblBufLoader 4 KB", "bench_dblbuf_m", medium);

        LargePayload large{};
        std::memset(large.data, 0xBB, sizeof(large.data));
        run("Storage+DblBufLoader 1 MB", "bench_dblbuf_l", large);
    }

    // ── DoubleBufferShm<T> ──────────────────────────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("DoubleBufferShm<T> (store + load cycle)")
            .warmup(50)
            .minEpochIterations(200);

        auto run = [&](char const *name, char const *shm_name, auto const &src)
        {
            using T = std::decay_t<decltype(src)>;
            safe_shm::DoubleBufferShm<T> shm(shm_name, [](std::string_view) {});

            b.batch(sizeof(T)).unit("byte").run(name, [&]
                                                {
                shm.store(src);
                auto snap = shm.load();
                shm.wait();
                ankerl::nanobench::doNotOptimizeAway(*snap); });
        };

        SmallPayload small{};
        std::fill(std::begin(small.values), std::end(small.values), 2.0);
        run("DoubleBufferShm 64 B", "bench_dbshm_s", small);

        MediumPayload medium{};
        std::memset(medium.data, 0xCC, sizeof(medium.data));
        run("DoubleBufferShm 4 KB", "bench_dbshm_m", medium);

        LargePayload large{};
        std::memset(large.data, 0xDD, sizeof(large.data));
        run("DoubleBufferShm 1 MB", "bench_dbshm_l", large);
    }

    // ── Seqlock (store + load cycle) ──────────────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("Seqlock (store + load, lock-free)")
            .warmup(100)
            .minEpochIterations(500);

        auto run = [&](char const *name, char const *shm_name, auto const &src)
        {
            using T = std::decay_t<decltype(src)>;
            safe_shm::Seqlock<T> sl(shm_name);

            b.batch(sizeof(T)).unit("byte").run(name, [&]
                                                {
                sl.store(src);
                auto val = sl.load();
                ankerl::nanobench::doNotOptimizeAway(val); });
        };

        SmallPayload small{};
        std::fill(std::begin(small.values), std::end(small.values), 3.0);
        run("Seqlock 64 B", "bench_seqlock_s", small);

        MediumPayload medium{};
        std::memset(medium.data, 0xEE, sizeof(medium.data));
        run("Seqlock 4 KB", "bench_seqlock_m", medium);

        LargePayload large{};
        std::memset(large.data, 0xFF, sizeof(large.data));
        run("Seqlock 1 MB", "bench_seqlock_l", large);
    }

    // ── pipe baseline ───────────────────────────────────────────────────
    {
        ankerl::nanobench::Bench b;
        b.title("pipe (kernel IPC baseline)")
            .warmup(50)
            .minEpochIterations(200);

        auto run = [&](char const *name, std::size_t size)
        {
            auto buf = std::make_unique<uint8_t[]>(size);
            auto recv = std::make_unique<uint8_t[]>(size);
            std::memset(buf.get(), 0xEE, size);

            int fds[2];
            if (pipe(fds) < 0)
                throw std::runtime_error("pipe failed");
            fcntl(fds[0], F_SETPIPE_SZ, static_cast<int>(std::max(size, std::size_t{65536})));

            b.batch(size).unit("byte").run(name, [&]
                                           {
                auto w = write(fds[1], buf.get(), size);
                auto r = read(fds[0], recv.get(), size);
                ankerl::nanobench::doNotOptimizeAway(w);
                ankerl::nanobench::doNotOptimizeAway(r); });

            close(fds[0]);
            close(fds[1]);
        };

        run("pipe 64 B", 64);
        run("pipe 4 KB", 4096);
        run("pipe 1 MB", 1024 * 1024);
    }

    // ── FHD image throughput ────────────────────────────────────────────
    {
        using Image = safe_shm::img::ImageFHD_RGB;
        ankerl::nanobench::Bench b;
        b.title("FHD RGB image (~6 MB) throughput")
            .warmup(10)
            .minEpochIterations(50);

        {
            safe_shm::DoubleBufferShm<Image> shm("bench_fhd_img", [](std::string_view) {});
            auto img = std::make_unique<Image>();
            img->timestamp = 1;
            img->frame_number = 1;
            std::fill(img->data.begin(), img->data.end(), 0xFF);

            b.batch(sizeof(Image)).unit("byte").run("DoubleBufferShm FHD RGB", [&]
                                                    {
                shm.store(*img);
                auto snap = shm.load();
                shm.wait();
                ankerl::nanobench::doNotOptimizeAway(snap->timestamp); });
        }

        {
            safe_shm::Seqlock<Image> sl("bench_fhd_seqlock");
            auto img = std::make_unique<Image>();
            img->timestamp = 3;
            img->frame_number = 3;
            std::fill(img->data.begin(), img->data.end(), 0xCC);

            b.batch(sizeof(Image)).unit("byte").run("Seqlock FHD RGB", [&]
                                                    {
                sl.store(*img);
                auto val = sl.load();
                ankerl::nanobench::doNotOptimizeAway(val.timestamp); });
        }

        {
            RawPosixShm raw("/bench_fhd_raw", sizeof(Image));
            auto img = std::make_unique<Image>();
            img->timestamp = 2;
            img->frame_number = 2;
            std::fill(img->data.begin(), img->data.end(), 0xAA);

            b.batch(sizeof(Image)).unit("byte").run("raw POSIX shm FHD RGB", [&]
                                                    { std::memcpy(raw.get(), img.get(), sizeof(Image));
                                                      ankerl::nanobench::doNotOptimizeAway(
                                                          *static_cast<uint8_t *>(raw.get())); });
        }
    }

    // ── cross-process latency ───────────────────────────────────────────
    {
        constexpr auto shm_name = "bench_xproc";
        ankerl::nanobench::Bench b;
        b.title("cross-process latency (fork + store + load)")
            .warmup(5)
            .minEpochIterations(20)
            .epochIterations(20);

        // Pre-create the shared memory
        {
            safe_shm::Storage<SmallPayload> storage(shm_name);
            SmallPayload init{};
            storage.store(init);
        }

        int iter = 0;
        b.batch(sizeof(SmallPayload)).unit("byte").run(
            "cross-process 64 B", [&]
            {
                pid_t pid = fork();
                if (pid == 0)
                {
                    safe_shm::Storage<SmallPayload> storage(shm_name);
                    SmallPayload data{};
                    data.values[0] = static_cast<double>(iter);
                    storage.store(data);
                    _exit(0);
                }

                int status;
                waitpid(pid, &status, 0);

                safe_shm::DblBufLoader<SmallPayload> loader(shm_name, [](std::string_view) {});
                auto snap = loader.load();
                loader.wait();
                ankerl::nanobench::doNotOptimizeAway(*snap);
                ++iter; });
    }

    return 0;
}
