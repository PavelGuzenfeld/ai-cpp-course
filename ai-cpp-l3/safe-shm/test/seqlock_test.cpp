#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/seqlock.hpp"
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

TEST_CASE("Seqlock store and load")
{
    safe_shm::Seqlock<int> sl("test_seqlock_basic");
    sl.store(42);
    CHECK(sl.load() == 42);
    sl.store(99);
    CHECK(sl.load() == 99);
}

TEST_CASE("Seqlock with struct")
{
    safe_shm::Seqlock<Telemetry> sl("test_seqlock_struct");
    sl.store({32.0, -117.0, 150.5f, 1});
    auto t = sl.load();
    CHECK(t.lat == doctest::Approx(32.0));
    CHECK(t.lon == doctest::Approx(-117.0));
    CHECK(t.alt == doctest::Approx(150.5f));
    CHECK(t.seq == 1);
}

TEST_CASE("Seqlock large payload")
{
    using BigArray = std::array<uint8_t, 65536>;
    safe_shm::Seqlock<BigArray> sl("test_seqlock_large");

    BigArray src;
    src.fill(0xAB);
    sl.store(src);

    auto result = sl.load();
    for (auto byte : result)
        CHECK(byte == 0xAB);
}

TEST_CASE("SeqlockWriter + SeqlockReader cross-process")
{
    constexpr auto name = "test_seqlock_xproc";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::SeqlockWriter<Telemetry> writer(name);
        writer.store({51.5, -0.1, 30.0f, 7});
        _exit(EXIT_SUCCESS);
    }

    int status;
    waitpid(pid, &status, 0);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    safe_shm::SeqlockReader<Telemetry> reader(name);
    auto t = reader.load();
    CHECK(t.lat == doctest::Approx(51.5));
    CHECK(t.lon == doctest::Approx(-0.1));
    CHECK(t.alt == doctest::Approx(30.0f));
    CHECK(t.seq == 7);
}

TEST_CASE("Seqlock concurrent writer and reader")
{
    constexpr auto name = "test_seqlock_conc";
    constexpr int WRITES = 10000;

    safe_shm::Seqlock<int> sl(name);
    sl.store(0);

    std::jthread writer([&](std::stop_token)
                        {
        for (int i = 1; i <= WRITES; ++i)
            sl.store(i); });

    int last = -1;
    int reads = 0;
    while (last < WRITES && reads < WRITES * 10)
    {
        int val = sl.load();
        CHECK(val >= 0);
        CHECK(val <= WRITES);
        // Values must be monotonically non-decreasing
        CHECK(val >= last);
        last = val;
        ++reads;
    }

    writer.join();
    CHECK(sl.load() == WRITES);
}

TEST_CASE("Seqlock sequence counter increments correctly")
{
    safe_shm::Seqlock<int> sl("test_seqlock_seq");
    CHECK((sl.sequence() & 1u) == 0); // must be even (no write in progress)

    sl.store(1);
    auto s1 = sl.sequence();
    CHECK((s1 & 1u) == 0);

    sl.store(2);
    auto s2 = sl.sequence();
    CHECK(s2 > s1);
    CHECK((s2 & 1u) == 0);
}

TEST_CASE("SeqlockReader load_blocking waits for new data")
{
    constexpr auto name = "test_seqlock_blocking";

    // Pre-write initial value
    {
        safe_shm::SeqlockWriter<int> w(name);
        w.store(0);
    }

    safe_shm::SeqlockReader<int> reader(name);
    uint32_t last_seq = reader.sequence();
    CHECK(reader.load() == 0);

    // Writer in background
    std::jthread writer([&]
                        {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        safe_shm::SeqlockWriter<int> w(name);
        w.store(42); });

    auto val = reader.load_blocking(last_seq);
    REQUIRE(val.has_value());
    CHECK(*val == 42);
    writer.join();
}

TEST_CASE("load_blocking returns nullopt on timeout")
{
    constexpr auto name = "test_seqlock_timeout";

    safe_shm::Seqlock<int> sl(name);
    sl.store(0);

    uint32_t last_seq = sl.sequence();

    // No writer — should timeout
    auto result = sl.load_blocking(last_seq, uint64_t{50'000'000}); // 50 ms
    CHECK_FALSE(result.has_value());
}

TEST_CASE("load_blocking handles sequence wrap (!= instead of >)")
{
    constexpr auto name = "test_seqlock_wrap";

    safe_shm::Seqlock<int> sl(name);
    sl.store(100);

    // Simulate a high last_seq (as if counter had been running a long time)
    // The key point: even if last_seq is very high, != will detect the change
    uint32_t last_seq = 0xFFFF'FFFEu; // near max uint32_t
    auto result = sl.load_blocking(last_seq, uint64_t{100'000'000}); // 100ms timeout
    REQUIRE(result.has_value());
    CHECK(*result == 100);
}

TEST_CASE("Seqlock load_blocking works on combined class")
{
    constexpr auto name = "test_seqlock_combined_blocking";

    safe_shm::Seqlock<int> sl(name);
    sl.store(0);
    uint32_t last_seq = sl.sequence();

    std::jthread writer([&]
                        {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        sl.store(77); });

    auto val = sl.load_blocking(last_seq);
    REQUIRE(val.has_value());
    CHECK(*val == 77);
    writer.join();
}
