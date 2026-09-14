#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/cyclic_buffer.hpp"
#include "safe-shm/dblbuf_loader.hpp"
#include "safe-shm/image.hpp"
#include "safe-shm/seqlock.hpp"
#include "safe-shm/shared_memory.hpp"
#include "safe-shm/shm_lifecycle.hpp"
#include "safe-shm/stamped.hpp"
#include "safe-shm/storage.hpp"
#include "safe-shm/time_series.hpp"
#include <sys/wait.h>
#include <thread>

struct Telemetry
{
    double lat;
    double lon;
    float alt;
    uint32_t seq;
};

using StampedTelem = safe_shm::Stamped<Telemetry>;

// ── Seqlock with Stamped<T> ────────────────────────────────────────────────

TEST_CASE("Seqlock<Stamped<T>> same-process round-trip")
{
    safe_shm::Seqlock<StampedTelem> sl("xproc_seqlock_stamped");

    auto s = safe_shm::stamp(Telemetry{32.0, -117.0, 150.5f, 1}, 42);
    sl.store(s);

    auto result = sl.load();
    CHECK(result.sequence == 42);
    CHECK(result.data.lat == doctest::Approx(32.0));
    CHECK(result.data.lon == doctest::Approx(-117.0));
    CHECK(result.data.alt == doctest::Approx(150.5f));
    CHECK(result.data.seq == 1);
    CHECK(result.timestamp_ns > 0);
}

TEST_CASE("SeqlockWriter/Reader<Stamped<T>> cross-process")
{
    constexpr auto name = "xproc_seqlock_stamped_xp";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::SeqlockWriter<StampedTelem> writer(name);
        auto s = safe_shm::stamp(Telemetry{51.5, -0.1, 30.0f, 7}, 99);
        writer.store(s);
        _exit(EXIT_SUCCESS);
    }

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    safe_shm::SeqlockReader<StampedTelem> reader(name);
    auto result = reader.load();
    CHECK(result.sequence == 99);
    CHECK(result.data.lat == doctest::Approx(51.5));
    CHECK(result.data.seq == 7);
}

// ── CyclicBuffer with Stamped<T> ──────────────────────────────────────────

TEST_CASE("CyclicBuffer<Stamped<T>> same-process")
{
    safe_shm::CyclicBuffer<StampedTelem, 16> buf("xproc_cb_stamped");

    for (uint64_t i = 0; i < 10; ++i)
    {
        auto s = safe_shm::stamp(
            Telemetry{32.0 + static_cast<double>(i), -117.0, 0.0f, static_cast<uint32_t>(i)}, i);
        buf.insert(s);
    }

    CHECK(buf.available() == 10);
    auto latest = buf.get_latest();
    CHECK(latest.sequence == 9);
    CHECK(latest.data.lat == doctest::Approx(41.0));
}

TEST_CASE("CyclicBufferWriter/Reader<Stamped<T>> cross-process")
{
    constexpr auto name = "xproc_cb_stamped_xp";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::CyclicBufferWriter<StampedTelem, 16> writer(name);
        for (uint64_t i = 0; i < 8; ++i)
        {
            StampedTelem s{i * 100, i, {static_cast<double>(i), 0.0, 0.0f, static_cast<uint32_t>(i)}};
            writer.insert(s);
        }
        _exit(EXIT_SUCCESS);
    }

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    safe_shm::CyclicBufferReader<StampedTelem, 16> reader(name);
    CHECK(reader.available() == 8);
    CHECK(reader.total_writes() == 8);

    auto latest = reader.get_latest();
    CHECK(latest.sequence == 7);
    CHECK(latest.timestamp_ns == 700);
}

// ── TimeSeries cross-process ───────────────────────────────────────────────

TEST_CASE("TimeSeries writer/reader cross-process")
{
    constexpr auto name = "xproc_ts_xp";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::CyclicBufferWriter<StampedTelem, 32> writer(name);
        for (uint64_t i = 0; i < 20; ++i)
        {
            StampedTelem s{i * 50, i, {static_cast<double>(i) * 0.5, 0.0, 0.0f, static_cast<uint32_t>(i)}};
            writer.insert(s);
        }
        _exit(EXIT_SUCCESS);
    }

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    safe_shm::TimeSeries<StampedTelem, 32> reader(name);
    CHECK(reader.available() == 20);

    // find_closest
    auto closest = reader.find_closest(uint64_t{225});
    REQUIRE(closest.has_value());
    CHECK((closest->timestamp_ns == 200 || closest->timestamp_ns == 250));

    // find_interpolation_pair
    auto interp = reader.find_interpolation_pair(uint64_t{225});
    REQUIRE(interp.has_value());
    CHECK(interp->before.timestamp_ns == 200);
    CHECK(interp->after.timestamp_ns == 250);
    CHECK(interp->alpha == doctest::Approx(0.5));

    // get_latest_if_fresh
    auto fresh = reader.get_latest_if_fresh(uint64_t{900});
    REQUIRE(fresh.has_value());
    CHECK(fresh->timestamp_ns == 950);

    auto stale = reader.get_latest_if_fresh(uint64_t{1000});
    CHECK_FALSE(stale.has_value());
}

// ── Storage + DblBufLoader with Stamped<T> ─────────────────────────────────

TEST_CASE("Storage/DblBufLoader<Stamped<T>> cross-process")
{
    constexpr auto name = "xproc_dblbuf_stamped";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::Storage<StampedTelem> storage(name);
        auto s = safe_shm::stamp(Telemetry{32.0, -117.0, 150.5f, 42}, 7);
        storage.store(s);
        _exit(EXIT_SUCCESS);
    }

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    safe_shm::DblBufLoader<StampedTelem> loader(name);
    auto snap = loader.load();
    loader.wait();

    CHECK(snap->sequence == 7);
    CHECK(snap->data.lat == doctest::Approx(32.0));
    CHECK(snap->data.seq == 42);
}

// ── Multi-reader scenario ──────────────────────────────────────────────────

TEST_CASE("CyclicBuffer 1 writer, 3 readers cross-process")
{
    constexpr auto name = "xproc_multi_reader";
    constexpr int NUM_ITEMS = 100;
    constexpr int NUM_READERS = 3;

    // Writer child
    pid_t writer_pid = fork();
    REQUIRE(writer_pid >= 0);

    if (writer_pid == 0)
    {
        safe_shm::CyclicBufferWriter<int, 256> writer(name);
        for (int i = 1; i <= NUM_ITEMS; ++i)
            writer.insert(i);
        _exit(EXIT_SUCCESS);
    }

    // Wait for writer to finish
    int status;
    waitpid(writer_pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    // Reader children
    for (int r = 0; r < NUM_READERS; ++r)
    {
        pid_t reader_pid = fork();
        REQUIRE(reader_pid >= 0);

        if (reader_pid == 0)
        {
            safe_shm::CyclicBufferReader<int, 256> reader(name);
            auto avail = reader.available();
            if (avail == 0)
                _exit(EXIT_FAILURE);
            auto latest = reader.get_latest();
            if (latest != NUM_ITEMS)
                _exit(EXIT_FAILURE);
            _exit(EXIT_SUCCESS);
        }

        waitpid(reader_pid, &status, 0);
        REQUIRE(WIFEXITED(status));
        CHECK(WEXITSTATUS(status) == 0);
    }
}

// ── Large payload: FHD images in CyclicBuffer ──────────────────────────────

TEST_CASE("CyclicBuffer<ImageFHD_RGB, 4> round-trip")
{
    using Image = safe_shm::img::ImageFHD_RGB;
    safe_shm::CyclicBuffer<Image, 4> buf("xproc_cb_image");

    auto img = std::make_unique<Image>();
    img->timestamp = 1000;
    img->frame_number = 1;
    std::fill(img->data.begin(), img->data.end(), 0xAB);

    buf.insert(*img);

    auto result = buf.get_latest();
    CHECK(result.timestamp == 1000);
    CHECK(result.frame_number == 1);
    CHECK(result.data[0] == 0xAB);
    CHECK(result.data[result.data.size() - 1] == 0xAB);
}

TEST_CASE("CyclicBuffer<ImageFHD_RGB> cross-process")
{
    using Image = safe_shm::img::ImageFHD_RGB;
    constexpr auto name = "xproc_cb_image_xp";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::CyclicBufferWriter<Image, 4> writer(name);
        Image img{};
        img.timestamp = 2000;
        img.frame_number = 2;
        std::fill(img.data.begin(), img.data.end(), 0xCD);
        writer.insert(img);
        _exit(EXIT_SUCCESS);
    }

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    safe_shm::CyclicBufferReader<Image, 4> reader(name);
    auto result = reader.get_latest();
    CHECK(result.timestamp == 2000);
    CHECK(result.frame_number == 2);
    CHECK(result.data[0] == 0xCD);
    CHECK(result.data[result.data.size() - 1] == 0xCD);
}

// ── OwnedShm lifecycle across fork ─────────────────────────────────────────

TEST_CASE("OwnedShm lifecycle cross-process")
{
    constexpr auto name = "xproc_owned_lifecycle";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::OwnedShm<int> owned(name);
        owned.data() = 999;
        owned.update_heartbeat();
        // Keep alive briefly for parent to read
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        _exit(EXIT_SUCCESS);
    }

    // Let child initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Parent reads the shared memory
    if (safe_shm::shm_exists(name))
    {
        safe_shm::SharedMemory<safe_shm::ShmHeader> header_reader(name);
        auto const &hdr = header_reader.get();
        CHECK(hdr.magic == safe_shm::SHM_HEADER_MAGIC);
        CHECK(safe_shm::is_writer_alive(hdr));
    }

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
}
