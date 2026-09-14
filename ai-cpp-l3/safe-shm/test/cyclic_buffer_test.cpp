#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/cyclic_buffer.hpp"
#include "safe-shm/stamped.hpp"
#include <array>
#include <sys/wait.h>
#include <thread>

struct Telemetry
{
    double lat;
    double lon;
    float alt;
    uint32_t seq;
};

TEST_CASE("CyclicBuffer basic insert and get_latest")
{
    safe_shm::CyclicBuffer<int, 8> buf("test_cb_basic");

    buf.insert(10);
    CHECK(buf.get_latest() == 10);
    CHECK(buf.available() == 1);
    CHECK(buf.total_writes() == 1);

    buf.insert(20);
    CHECK(buf.get_latest() == 20);
    CHECK(buf.available() == 2);
    CHECK(buf.total_writes() == 2);
}

TEST_CASE("CyclicBuffer get by reverse index")
{
    safe_shm::CyclicBuffer<int, 8> buf("test_cb_revindex");

    for (int i = 0; i < 5; ++i)
        buf.insert(i * 10);

    CHECK(buf.get(0) == 40); // latest
    CHECK(buf.get(1) == 30);
    CHECK(buf.get(2) == 20);
    CHECK(buf.get(3) == 10);
    CHECK(buf.get(4) == 0); // oldest
}

TEST_CASE("CyclicBuffer fills to capacity")
{
    safe_shm::CyclicBuffer<int, 4> buf("test_cb_fill");

    for (int i = 0; i < 4; ++i)
        buf.insert(i);

    CHECK(buf.available() == 4);
    CHECK(buf.get(0) == 3);
    CHECK(buf.get(3) == 0);
}

TEST_CASE("CyclicBuffer overflow wraps correctly")
{
    safe_shm::CyclicBuffer<int, 4> buf("test_cb_overflow");

    for (int i = 0; i < 10; ++i)
        buf.insert(i);

    CHECK(buf.available() == 4);
    CHECK(buf.total_writes() == 10);
    CHECK(buf.get(0) == 9); // latest
    CHECK(buf.get(1) == 8);
    CHECK(buf.get(2) == 7);
    CHECK(buf.get(3) == 6); // oldest surviving
}

TEST_CASE("CyclicBuffer try_get out-of-range returns nullopt")
{
    safe_shm::CyclicBuffer<int, 4> buf("test_cb_tryget");

    buf.insert(1);
    buf.insert(2);

    CHECK(buf.try_get(0).has_value());
    CHECK(buf.try_get(1).has_value());
    CHECK_FALSE(buf.try_get(2).has_value()); // only 2 elements
    CHECK_FALSE(buf.try_get(100).has_value());
}

TEST_CASE("CyclicBuffer with struct")
{
    safe_shm::CyclicBuffer<Telemetry, 8> buf("test_cb_struct");

    buf.insert({32.0, -117.0, 150.5f, 1});
    buf.insert({33.0, -118.0, 200.0f, 2});

    auto latest = buf.get_latest();
    CHECK(latest.lat == doctest::Approx(33.0));
    CHECK(latest.lon == doctest::Approx(-118.0));
    CHECK(latest.alt == doctest::Approx(200.0f));
    CHECK(latest.seq == 2);

    auto prev = buf.get(1);
    CHECK(prev.lat == doctest::Approx(32.0));
    CHECK(prev.seq == 1);
}

TEST_CASE("CyclicBuffer with Stamped<T>")
{
    using S = safe_shm::Stamped<Telemetry>;
    safe_shm::CyclicBuffer<S, 16> buf("test_cb_stamped");

    for (uint64_t i = 0; i < 10; ++i)
    {
        auto s = safe_shm::stamp(Telemetry{32.0 + static_cast<double>(i), -117.0, 0.0f, static_cast<uint32_t>(i)}, i);
        buf.insert(s);
    }

    CHECK(buf.available() == 10);
    auto latest = buf.get_latest();
    CHECK(latest.sequence == 9);
    CHECK(latest.data.lat == doctest::Approx(41.0));
}

TEST_CASE("CyclicBuffer large payload")
{
    using BigArray = std::array<uint8_t, 4096>;
    safe_shm::CyclicBuffer<BigArray, 4> buf("test_cb_large");

    BigArray src;
    src.fill(0xAB);
    buf.insert(src);

    auto result = buf.get_latest();
    for (auto byte : result)
        CHECK(byte == 0xAB);
}

TEST_CASE("CyclicBufferWriter + CyclicBufferReader cross-process")
{
    constexpr auto name = "test_cb_xproc";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::CyclicBufferWriter<Telemetry, 8> writer(name);
        for (uint32_t i = 0; i < 5; ++i)
            writer.insert({32.0 + i, -117.0, static_cast<float>(i) * 10.0f, i});
        _exit(EXIT_SUCCESS);
    }

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    safe_shm::CyclicBufferReader<Telemetry, 8> reader(name);
    CHECK(reader.available() == 5);
    CHECK(reader.total_writes() == 5);

    auto latest = reader.get_latest();
    CHECK(latest.lat == doctest::Approx(36.0));
    CHECK(latest.seq == 4);

    auto oldest = reader.get(4);
    CHECK(oldest.lat == doctest::Approx(32.0));
    CHECK(oldest.seq == 0);
}

TEST_CASE("CyclicBuffer concurrent writer and reader")
{
    constexpr auto name = "test_cb_conc";
    constexpr int WRITES = 10000;

    safe_shm::CyclicBuffer<int, 256> buf(name);

    std::jthread writer([&](std::stop_token)
                        {
        for (int i = 1; i <= WRITES; ++i)
            buf.insert(i); });

    int last = 0;
    int reads = 0;
    while (last < WRITES && reads < WRITES * 10)
    {
        if (buf.available() == 0)
            continue;
        auto val = buf.try_get(0);
        if (!val)
            continue;
        CHECK(*val >= 1);
        CHECK(*val <= WRITES);
        // Values must be monotonically non-decreasing (we only read latest)
        CHECK(*val >= last);
        last = *val;
        ++reads;
    }

    writer.join();
    CHECK(buf.get_latest() == WRITES);
}

TEST_CASE("CyclicBufferReader wait_for_write")
{
    constexpr auto name = "test_cb_blocking";

    safe_shm::CyclicBuffer<int, 8> buf(name);
    buf.insert(0);
    auto initial_writes = buf.total_writes();

    std::jthread writer([&]
                        {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        buf.insert(42); });

    safe_shm::CyclicBufferReader<int, 8> reader(name);
    auto result = reader.wait_for_write(initial_writes, uint64_t{500'000'000}); // 500ms
    REQUIRE(result.has_value());
    CHECK(*result > initial_writes);
    CHECK(reader.get_latest() == 42);

    writer.join();
}

TEST_CASE("CyclicBufferReader wait_for_write timeout")
{
    constexpr auto name = "test_cb_timeout";

    safe_shm::CyclicBuffer<int, 8> buf(name);
    buf.insert(0);
    auto writes = buf.total_writes();

    safe_shm::CyclicBufferReader<int, 8> reader(name);
    auto result = reader.wait_for_write(writes, uint64_t{50'000'000}); // 50ms
    CHECK_FALSE(result.has_value());
}

TEST_CASE("CyclicBuffer capacity is correct")
{
    CHECK(safe_shm::CyclicBuffer<int, 8>::capacity() == 8);
    CHECK(safe_shm::CyclicBuffer<int, 64>::capacity() == 64);
    CHECK(safe_shm::CyclicBufferReader<int, 16>::capacity() == 16);
}
