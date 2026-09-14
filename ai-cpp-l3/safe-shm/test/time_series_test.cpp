#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/time_series.hpp"
#include <sys/wait.h>

struct Telemetry
{
    double lat;
    double lon;
    float alt;
    uint32_t seq;
};

using StampedTelem = safe_shm::Stamped<Telemetry>;

// Custom key extractor for testing (extracts sequence number)
struct SeqKeyExtractor
{
    uint32_t operator()(Telemetry const &t) const noexcept { return t.seq; }
};

TEST_CASE("TimeSeries find_closest exact match")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_closest");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_closest");

    for (uint64_t i = 0; i < 10; ++i)
    {
        StampedTelem s{i * 100, i, {32.0 + static_cast<double>(i), -117.0, 0.0f, static_cast<uint32_t>(i)}};
        writer.insert(s);
    }

    auto result = reader.find_closest(uint64_t{500});
    REQUIRE(result.has_value());
    CHECK(result->timestamp_ns == 500);
    CHECK(result->data.seq == 5);
}

TEST_CASE("TimeSeries find_closest nearest")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_nearest");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_nearest");

    for (uint64_t i = 0; i < 10; ++i)
    {
        StampedTelem s{i * 100, i, {0.0, 0.0, 0.0f, static_cast<uint32_t>(i)}};
        writer.insert(s);
    }

    // 250 is between 200 and 300, closer to 300
    auto result = reader.find_closest(uint64_t{250});
    REQUIRE(result.has_value());
    // Should be either 200 or 300 (both within 50)
    CHECK((result->timestamp_ns == 200 || result->timestamp_ns == 300));
}

TEST_CASE("TimeSeries find_closest empty buffer")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_empty");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_empty");

    CHECK_FALSE(reader.find_closest(uint64_t{100}).has_value());
}

TEST_CASE("TimeSeries find_interpolation_pair normal case")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_interp");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_interp");

    for (uint64_t i = 0; i < 10; ++i)
    {
        StampedTelem s{i * 100, i, {static_cast<double>(i), 0.0, 0.0f, static_cast<uint32_t>(i)}};
        writer.insert(s);
    }

    // Target 350: between 300 (seq=3) and 400 (seq=4)
    auto result = reader.find_interpolation_pair(uint64_t{350});
    REQUIRE(result.has_value());
    CHECK(result->before.timestamp_ns == 300);
    CHECK(result->after.timestamp_ns == 400);
    CHECK(result->alpha == doctest::Approx(0.5));
}

TEST_CASE("TimeSeries find_interpolation_pair at 25%")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_interp25");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_interp25");

    for (uint64_t i = 0; i < 10; ++i)
    {
        StampedTelem s{i * 1000, i, {0.0, 0.0, 0.0f, static_cast<uint32_t>(i)}};
        writer.insert(s);
    }

    // Target 2250: between 2000 and 3000, 25% into the interval
    auto result = reader.find_interpolation_pair(uint64_t{2250});
    REQUIRE(result.has_value());
    CHECK(result->before.timestamp_ns == 2000);
    CHECK(result->after.timestamp_ns == 3000);
    CHECK(result->alpha == doctest::Approx(0.25));
}

TEST_CASE("TimeSeries find_interpolation_pair boundary exact match")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_interp_exact");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_interp_exact");

    for (uint64_t i = 0; i < 5; ++i)
    {
        StampedTelem s{i * 100, i, {0.0, 0.0, 0.0f, static_cast<uint32_t>(i)}};
        writer.insert(s);
    }

    // Exact match on newest
    auto result = reader.find_interpolation_pair(uint64_t{400});
    REQUIRE(result.has_value());
    CHECK(result->alpha == doctest::Approx(0.0));
}

TEST_CASE("TimeSeries find_interpolation_pair out of range")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_interp_oor");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_interp_oor");

    for (uint64_t i = 0; i < 5; ++i)
    {
        StampedTelem s{(i + 1) * 100, i, {0.0, 0.0, 0.0f, static_cast<uint32_t>(i)}};
        writer.insert(s);
    }

    // Before oldest
    CHECK_FALSE(reader.find_interpolation_pair(uint64_t{50}).has_value());
    // After newest
    CHECK_FALSE(reader.find_interpolation_pair(uint64_t{600}).has_value());
}

TEST_CASE("TimeSeries find_interpolation_pair too few elements")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_interp_few");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_interp_few");

    // Empty
    CHECK_FALSE(reader.find_interpolation_pair(uint64_t{100}).has_value());

    // Single element
    StampedTelem s{100, 0, {0.0, 0.0, 0.0f, 0}};
    writer.insert(s);
    CHECK_FALSE(reader.find_interpolation_pair(uint64_t{100}).has_value());
}

TEST_CASE("TimeSeries get_latest_if_fresh with fresh data")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_fresh");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_fresh");

    StampedTelem s{1000, 0, {32.0, -117.0, 150.5f, 1}};
    writer.insert(s);

    auto result = reader.get_latest_if_fresh(uint64_t{500});
    REQUIRE(result.has_value());
    CHECK(result->timestamp_ns == 1000);
}

TEST_CASE("TimeSeries get_latest_if_fresh with stale data")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_stale");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_stale");

    StampedTelem s{100, 0, {0.0, 0.0, 0.0f, 0}};
    writer.insert(s);

    CHECK_FALSE(reader.get_latest_if_fresh(uint64_t{500}).has_value());
}

TEST_CASE("TimeSeries with custom key extractor")
{
    safe_shm::CyclicBufferWriter<Telemetry, 16> writer("test_ts_custom_key");
    safe_shm::TimeSeries<Telemetry, 16, SeqKeyExtractor> reader("test_ts_custom_key");

    for (uint32_t i = 0; i < 10; ++i)
        writer.insert({static_cast<double>(i), 0.0, 0.0f, i * 10});

    // Find closest to seq=35 (should be 30 or 40)
    auto result = reader.find_closest(uint32_t{35});
    REQUIRE(result.has_value());
    CHECK((result->seq == 30 || result->seq == 40));
}

TEST_CASE("TimeSeries cross-process writer/reader")
{
    constexpr auto name = "test_ts_xproc";

    pid_t pid = fork();
    REQUIRE(pid >= 0);

    if (pid == 0)
    {
        safe_shm::CyclicBufferWriter<StampedTelem, 16> writer(name);
        for (uint64_t i = 0; i < 10; ++i)
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

    safe_shm::TimeSeries<StampedTelem, 16> reader(name);
    CHECK(reader.available() == 10);

    auto closest = reader.find_closest(uint64_t{450});
    REQUIRE(closest.has_value());
    CHECK((closest->timestamp_ns == 400 || closest->timestamp_ns == 500));

    auto interp = reader.find_interpolation_pair(uint64_t{450});
    REQUIRE(interp.has_value());
    CHECK(interp->before.timestamp_ns == 400);
    CHECK(interp->after.timestamp_ns == 500);
    CHECK(interp->alpha == doctest::Approx(0.5));
}

// ── Edge cases: out-of-bounds and unit confusion ──────────────────────

TEST_CASE("TimeSeries find_closest with max_distance rejects far targets")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_maxdist");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_maxdist");

    // Write data with timestamps 0..900 (step 100ns)
    for (uint64_t i = 0; i < 10; ++i)
    {
        StampedTelem s{i * 100, i, {0.0, 0.0, 0.0f, static_cast<uint32_t>(i)}};
        writer.insert(s);
    }

    // Without max_distance: always finds something
    auto result = reader.find_closest(uint64_t{999'999'999});
    REQUIRE(result.has_value());
    CHECK(result->timestamp_ns == 900); // closest to the far-future target

    // With max_distance=50: rejects targets far from data range
    CHECK_FALSE(reader.find_closest(uint64_t{999'999'999}, uint64_t{50}).has_value());

    // With max_distance=50: accepts targets within range
    auto close = reader.find_closest(uint64_t{520}, uint64_t{50});
    REQUIRE(close.has_value());
    CHECK(close->timestamp_ns == 500);

    // With max_distance=0: only exact matches
    auto exact = reader.find_closest(uint64_t{300}, uint64_t{0});
    REQUIRE(exact.has_value());
    CHECK(exact->timestamp_ns == 300);

    auto no_exact = reader.find_closest(uint64_t{301}, uint64_t{0});
    CHECK_FALSE(no_exact.has_value());
}

TEST_CASE("TimeSeries ns vs ms unit confusion detection")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_units");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_units");

    // Write data with realistic nanosecond timestamps (1-10 seconds in ns)
    for (uint64_t i = 1; i <= 10; ++i)
    {
        StampedTelem s{i * 1'000'000'000ULL, i, {0.0, 0.0, 0.0f, static_cast<uint32_t>(i)}};
        writer.insert(s);
    }

    // Simulating the bug: user queries with milliseconds (5000ms = 5s)
    // instead of nanoseconds (5'000'000'000ns)
    uint64_t target_ms = 5000;                    // WRONG: this is 5 microseconds in ns
    uint64_t target_ns = 5'000'000'000ULL;        // CORRECT

    // With max_distance set to 100ms (in ns), the ms query is caught
    // (closest data point is 1s = 1e9ns, distance ~1e9 >> 100ms)
    uint64_t tolerance_ns = 100'000'000ULL; // 100ms

    auto bad = reader.find_closest(target_ms, tolerance_ns);
    CHECK_FALSE(bad.has_value()); // Correctly rejected: 5000ns is far from 1-10s range

    auto good = reader.find_closest(target_ns, tolerance_ns);
    REQUIRE(good.has_value());
    CHECK(good->timestamp_ns == 5'000'000'000ULL);
}

TEST_CASE("TimeSeries far-future and far-past queries")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_farquery");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_farquery");

    for (uint64_t i = 0; i < 5; ++i)
    {
        StampedTelem s{1000 + i * 100, i, {0.0, 0.0, 0.0f, static_cast<uint32_t>(i)}};
        writer.insert(s);
    }

    // Far future without max_distance: returns newest element
    auto future = reader.find_closest(uint64_t{999'999'999'999ULL});
    REQUIRE(future.has_value());
    CHECK(future->timestamp_ns == 1400);

    // Far past without max_distance: returns oldest element
    auto past = reader.find_closest(uint64_t{0});
    REQUIRE(past.has_value());
    CHECK(past->timestamp_ns == 1000);

    // Far future with max_distance: rejected
    CHECK_FALSE(reader.find_closest(uint64_t{999'999'999'999ULL}, uint64_t{1000}).has_value());

    // Far past with max_distance: rejected
    CHECK_FALSE(reader.find_closest(uint64_t{0}, uint64_t{500}).has_value());

    // interpolation_pair out of range: already returns nullopt
    CHECK_FALSE(reader.find_interpolation_pair(uint64_t{0}).has_value());
    CHECK_FALSE(reader.find_interpolation_pair(uint64_t{999'999}).has_value());
}

TEST_CASE("TimeSeries get_latest_if_fresh detects unit mismatch")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_fresh_units");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_fresh_units");

    // Data at 5 seconds (nanoseconds)
    StampedTelem s{5'000'000'000ULL, 0, {0.0, 0.0, 0.0f, 0}};
    writer.insert(s);

    // Correct: check if fresh relative to 4 seconds (in ns)
    auto fresh = reader.get_latest_if_fresh(uint64_t{4'000'000'000ULL});
    REQUIRE(fresh.has_value());

    // Bug: min_key in ms instead of ns — 4000ms << 5'000'000'000ns
    // This would incorrectly report as "fresh" because 5e9 > 4000
    auto false_fresh = reader.get_latest_if_fresh(uint64_t{4000});
    CHECK(false_fresh.has_value()); // technically "correct" but misleading

    // Bug: min_key in seconds — 4s is 4, which << 5'000'000'000ns
    auto false_fresh2 = reader.get_latest_if_fresh(uint64_t{4});
    CHECK(false_fresh2.has_value()); // also "correct" but misleading
    // NOTE: These pass because comparison is numeric. The caller must
    // ensure consistent units. Use monotonic_now_ns() for freshness checks.
}

TEST_CASE("TimeSeries single-element edge cases")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_single");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_single");

    StampedTelem s{1000, 0, {0.0, 0.0, 0.0f, 0}};
    writer.insert(s);

    // find_closest with single element
    auto result = reader.find_closest(uint64_t{1000});
    REQUIRE(result.has_value());
    CHECK(result->timestamp_ns == 1000);

    // find_closest with max_distance on single element
    auto close = reader.find_closest(uint64_t{1050}, uint64_t{100});
    REQUIRE(close.has_value());
    CHECK(close->timestamp_ns == 1000);

    auto far = reader.find_closest(uint64_t{2000}, uint64_t{100});
    CHECK_FALSE(far.has_value());
}

TEST_CASE("TimeSeries delegates to reader correctly")
{
    safe_shm::CyclicBufferWriter<StampedTelem, 16> writer("test_ts_delegate");
    safe_shm::TimeSeries<StampedTelem, 16> reader("test_ts_delegate");

    for (uint64_t i = 0; i < 5; ++i)
    {
        StampedTelem s{i * 100, i, {0.0, 0.0, 0.0f, static_cast<uint32_t>(i)}};
        writer.insert(s);
    }

    CHECK(reader.available() == 5);
    CHECK(reader.total_writes() == 5);
    CHECK(reader.capacity() == 16);

    auto latest = reader.get_latest();
    CHECK(latest.sequence == 4);
}
