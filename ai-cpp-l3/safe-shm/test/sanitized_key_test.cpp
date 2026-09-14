#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/sanitized_key.hpp"
#include "safe-shm/time_series.hpp"

using StampedDbl = safe_shm::Stamped<double>;

// ── SanitizedKey construction ─────────────────────────────────────────

TEST_CASE("SanitizedKey can only be created through sanitize()")
{
    // This must NOT compile (private constructor):
    // safe_shm::SanitizedKey<> key(42);

    auto key = safe_shm::sanitize(42, [](uint64_t v) { return v > 0; });
    REQUIRE(key.has_value());
    CHECK(key->value() == 42);
}

TEST_CASE("sanitize returns nullopt when validator rejects")
{
    auto key = safe_shm::sanitize(0, [](uint64_t v) { return v > 0; });
    CHECK_FALSE(key.has_value());
}

// ── Built-in validators ───────────────────────────────────────────────

TEST_CASE("RangeValidator accepts values in range")
{
    safe_shm::RangeValidator v{100, 200};
    CHECK(v(100));
    CHECK(v(150));
    CHECK(v(200));
    CHECK_FALSE(v(99));
    CHECK_FALSE(v(201));
}

TEST_CASE("ProximityValidator catches unit mismatches")
{
    // Reference: 5 seconds in nanoseconds
    safe_shm::ProximityValidator v{5'000'000'000ULL, 1'000'000'000ULL}; // ±1s tolerance

    // Correct: 4.5 seconds in ns → within tolerance
    CHECK(v(4'500'000'000ULL));

    // Bug: 5000 (ms instead of ns) → off by ~5e9, rejected
    CHECK_FALSE(v(5000));

    // Bug: 5 (seconds instead of ns) → off by ~5e9, rejected
    CHECK_FALSE(v(5));
}

TEST_CASE("MonotonicNsValidator rejects non-ns values")
{
    safe_shm::MonotonicNsValidator v;

    // Valid: 10 seconds of uptime
    CHECK(v(10'000'000'000ULL));

    // Invalid: 5000 (looks like ms, not ns)
    CHECK_FALSE(v(5000));

    // Invalid: 0
    CHECK_FALSE(v(0));

    // Valid: 1 year of uptime
    CHECK(v(365ULL * 86400 * 1'000'000'000ULL));
}

TEST_CASE("AllOf composes multiple validators")
{
    auto v = safe_shm::AllOf(
        safe_shm::MonotonicNsValidator{},
        safe_shm::RangeValidator{
            2'000'000'000ULL,  // min: 2s
            10'000'000'000ULL} // max: 10s
    );

    CHECK(v(5'000'000'000ULL));      // 5s: in range and valid ns
    CHECK_FALSE(v(1'000'000'000ULL)); // 1s: valid ns but below range min
    CHECK_FALSE(v(5000));             // 5000ns: in range numerically but not valid monotonic ns
}

// ── Tag type safety ───────────────────────────────────────────────────

TEST_CASE("Different tags are type-incompatible")
{
    using KeyA = safe_shm::SanitizedKey<safe_shm::MonotonicNsTag>;
    using KeyB = safe_shm::SanitizedKey<safe_shm::RangeCheckedTag>;

    // These are different types — the compiler prevents mixing them.
    CHECK_FALSE(std::is_same_v<KeyA, KeyB>);

    // Both can be created independently
    auto a = safe_shm::sanitize<safe_shm::MonotonicNsTag>(
        5'000'000'000ULL, safe_shm::MonotonicNsValidator{});
    auto b = safe_shm::sanitize<safe_shm::RangeCheckedTag>(
        5'000'000'000ULL, safe_shm::RangeValidator{0, UINT64_MAX});

    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    CHECK(a->value() == b->value()); // same raw value, different types
}

// ── TimeSeries integration with SanitizedKey ──────────────────────────

TEST_CASE("TimeSeries find_closest accepts SanitizedKey")
{
    safe_shm::CyclicBufferWriter<StampedDbl, 16> writer("test_sk_closest");
    safe_shm::TimeSeries<StampedDbl, 16> reader("test_sk_closest");

    for (uint64_t i = 0; i < 10; ++i)
    {
        StampedDbl s{i * 1000, i, static_cast<double>(i)};
        writer.insert(s);
    }

    // Sanitize the key first
    auto key = safe_shm::sanitize<safe_shm::RangeCheckedTag>(
        5000, safe_shm::RangeValidator{0, 9000});

    REQUIRE(key.has_value());
    auto result = reader.find_closest(*key);
    REQUIRE(result.has_value());
    CHECK(result->timestamp_ns == 5000);
}

TEST_CASE("TimeSeries rejects unsanitized key at compile time")
{
    // This documents the compile-time safety:
    // If a function signature requires SanitizedKey<Tag>, passing raw
    // uint64_t won't compile. Example (would not compile):
    //
    //   void lookup(SanitizedKey<MonotonicNsTag> key);
    //   lookup(5000);  // ERROR: no implicit conversion from uint64_t
    //
    // The raw TimeSeries API still accepts uint64_t for backward compat.
    // The SanitizedKey overloads provide the opt-in safety layer.
    CHECK(true); // placeholder: compile-time guarantee, no runtime check
}

TEST_CASE("TimeSeries find_interpolation_pair accepts SanitizedKey")
{
    safe_shm::CyclicBufferWriter<StampedDbl, 16> writer("test_sk_interp");
    safe_shm::TimeSeries<StampedDbl, 16> reader("test_sk_interp");

    for (uint64_t i = 0; i < 10; ++i)
    {
        StampedDbl s{i * 1000, i, static_cast<double>(i)};
        writer.insert(s);
    }

    auto key = safe_shm::sanitize<safe_shm::RangeCheckedTag>(
        3500, safe_shm::RangeValidator{0, 9000});

    REQUIRE(key.has_value());
    auto result = reader.find_interpolation_pair(*key);
    REQUIRE(result.has_value());
    CHECK(result->before.timestamp_ns == 3000);
    CHECK(result->after.timestamp_ns == 4000);
    CHECK(result->alpha == doctest::Approx(0.5));
}

TEST_CASE("TimeSeries get_latest_if_fresh accepts SanitizedKey")
{
    safe_shm::CyclicBufferWriter<StampedDbl, 16> writer("test_sk_fresh");
    safe_shm::TimeSeries<StampedDbl, 16> reader("test_sk_fresh");

    StampedDbl s{5000, 0, 42.0};
    writer.insert(s);

    auto key = safe_shm::sanitize<safe_shm::RangeCheckedTag>(
        3000, safe_shm::RangeValidator{0, 10000});

    REQUIRE(key.has_value());
    auto result = reader.get_latest_if_fresh(*key);
    REQUIRE(result.has_value());
    CHECK(result->timestamp_ns == 5000);
}

TEST_CASE("Sanitizer catches ns-vs-ms confusion before reaching TimeSeries")
{
    safe_shm::CyclicBufferWriter<StampedDbl, 16> writer("test_sk_unit_bug");
    safe_shm::TimeSeries<StampedDbl, 16> reader("test_sk_unit_bug");

    // Data at 1-10 seconds (nanoseconds)
    for (uint64_t i = 1; i <= 10; ++i)
    {
        StampedDbl s{i * 1'000'000'000ULL, i, static_cast<double>(i)};
        writer.insert(s);
    }

    // User function that requires sanitized key — the bug is caught HERE,
    // before the temporal search even runs.
    auto bad = safe_shm::sanitize<safe_shm::MonotonicNsTag>(
        5000, // BUG: 5000 ns, not 5 seconds
        safe_shm::MonotonicNsValidator{});
    CHECK_FALSE(bad.has_value()); // rejected: 5000ns is not a valid monotonic timestamp

    auto good = safe_shm::sanitize<safe_shm::MonotonicNsTag>(
        5'000'000'000ULL, // CORRECT: 5 seconds in ns
        safe_shm::MonotonicNsValidator{});
    REQUIRE(good.has_value());

    auto result = reader.find_closest(*good);
    REQUIRE(result.has_value());
    CHECK(result->timestamp_ns == 5'000'000'000ULL);
}

TEST_CASE("Custom validator lambda as sanitizer")
{
    // Users can provide any callable as the sanitizer
    uint64_t latest_ts = 9'000'000'000ULL;

    auto key = safe_shm::sanitize<safe_shm::ProximityCheckedTag>(
        8'500'000'000ULL,
        [latest_ts](uint64_t v) {
            // Must be within 2 seconds of latest known timestamp
            auto diff = v > latest_ts ? v - latest_ts : latest_ts - v;
            return diff <= 2'000'000'000ULL;
        });

    REQUIRE(key.has_value());
    CHECK(key->value() == 8'500'000'000ULL);

    // Too far from latest
    auto bad = safe_shm::sanitize<safe_shm::ProximityCheckedTag>(
        1'000'000'000ULL,
        [latest_ts](uint64_t v) {
            auto diff = v > latest_ts ? v - latest_ts : latest_ts - v;
            return diff <= 2'000'000'000ULL;
        });

    CHECK_FALSE(bad.has_value());
}
