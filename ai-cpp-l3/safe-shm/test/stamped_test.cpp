#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "safe-shm/stamped.hpp"
#include "safe-shm/shm_stats.hpp"
#include "safe-shm/seqlock.hpp"

struct Telemetry
{
    double lat;
    double lon;
    float alt;
    uint32_t seq;
};

TEST_CASE("Stamped<int> satisfies FlatType")
{
    CHECK(safe_shm::FlatType<safe_shm::Stamped<int>>);
    CHECK(safe_shm::FlatType<safe_shm::Stamped<double>>);
    CHECK(safe_shm::FlatType<safe_shm::Stamped<Telemetry>>);
}

TEST_CASE("Stamped layout")
{
    safe_shm::Stamped<int> s{100, 1, 42};
    CHECK(s.timestamp_ns == 100);
    CHECK(s.sequence == 1);
    CHECK(s.data == 42);
}

TEST_CASE("stamp() helper produces valid timestamps")
{
    auto before = safe_shm::monotonic_now_ns();
    auto s = safe_shm::stamp(42, 1);
    auto after = safe_shm::monotonic_now_ns();

    CHECK(s.timestamp_ns >= before);
    CHECK(s.timestamp_ns <= after);
    CHECK(s.sequence == 1);
    CHECK(s.data == 42);
}

TEST_CASE("stamp() with struct")
{
    Telemetry t{32.0, -117.0, 150.5f, 7};
    auto s = safe_shm::stamp(t, 99);

    CHECK(s.timestamp_ns > 0);
    CHECK(s.sequence == 99);
    CHECK(s.data.lat == doctest::Approx(32.0));
    CHECK(s.data.lon == doctest::Approx(-117.0));
    CHECK(s.data.alt == doctest::Approx(150.5f));
    CHECK(s.data.seq == 7);
}

TEST_CASE("monotonic_now_ns is monotonically increasing")
{
    auto t1 = safe_shm::monotonic_now_ns();
    auto t2 = safe_shm::monotonic_now_ns();
    CHECK(t2 >= t1);
}

TEST_CASE("ShmStats satisfies FlatType")
{
    // ShmStats has default member initializers but is still trivially copyable
    // because it has no user-defined constructors
    safe_shm::ShmStats stats{};
    CHECK(stats.total_writes == 0);
    CHECK(stats.total_reads == 0);
    CHECK(stats.torn_read_retries == 0);
    CHECK(stats.lock_contentions == 0);
}

TEST_CASE("detail::Empty is small")
{
    struct Wrapper
    {
        [[no_unique_address]] safe_shm::detail::Empty e;
        int x;
    };
    // Empty should be optimized away with [[no_unique_address]]
    CHECK(sizeof(Wrapper) == sizeof(int));
}

TEST_CASE("stats_member_t selects correct type")
{
    CHECK(std::is_same_v<safe_shm::detail::stats_member_t<true>, safe_shm::ShmStats>);
    CHECK(std::is_same_v<safe_shm::detail::stats_member_t<false>, safe_shm::detail::Empty>);
}

TEST_CASE("Stamped<T> works with Seqlock")
{
    using StampedTelem = safe_shm::Stamped<Telemetry>;
    safe_shm::Seqlock<StampedTelem> sl("test_stamped_seqlock");

    auto s = safe_shm::stamp(Telemetry{32.0, -117.0, 150.5f, 1}, 42);
    sl.store(s);

    auto result = sl.load();
    CHECK(result.sequence == 42);
    CHECK(result.data.lat == doctest::Approx(32.0));
    CHECK(result.data.alt == doctest::Approx(150.5f));
    CHECK(result.timestamp_ns > 0);
}
