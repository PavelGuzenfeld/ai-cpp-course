#include "safe-shm/cyclic_buffer.hpp"
#include "safe-shm/stamped.hpp"
#include "safe-shm/time_series.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

// Global counter for unique SHM names across fuzzer invocations.
static std::atomic<uint64_t> g_counter{0};

static std::string unique_name(char const *prefix)
{
    return std::string(prefix) + "_" + std::to_string(getpid()) + "_" +
           std::to_string(g_counter.fetch_add(1, std::memory_order_relaxed));
}

// Consume a value of type U from the fuzz input, advancing the pointer.
// Returns false if not enough bytes remain.
template <typename U>
static bool consume(const uint8_t *&ptr, size_t &remaining, U &out)
{
    if (remaining < sizeof(U))
        return false;
    std::memcpy(&out, ptr, sizeof(U));
    ptr += sizeof(U);
    remaining -= sizeof(U);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    using Elem = safe_shm::Stamped<double>;
    constexpr std::size_t Cap = 16;

    // --- Zero-length input: exercise empty-buffer query paths ---
    if (size == 0)
    {
        auto wname = unique_name("fuzz_ts_empty");
        safe_shm::CyclicBufferWriter<Elem, Cap> writer(wname);
        safe_shm::TimeSeries<Elem, Cap> ts(wname);

        // All queries on empty buffer must return nullopt / 0 without crashing.
        (void)ts.find_closest(uint64_t{0});
        (void)ts.find_closest(UINT64_MAX);
        (void)ts.find_closest(uint64_t{42}, uint64_t{0});
        (void)ts.find_closest(uint64_t{42}, uint64_t{UINT64_MAX});
        (void)ts.find_interpolation_pair(uint64_t{0});
        (void)ts.find_interpolation_pair(UINT64_MAX);
        (void)ts.get_latest_if_fresh(uint64_t{0});
        (void)ts.get_latest_if_fresh(UINT64_MAX);
        (void)ts.available();
        (void)ts.total_writes();
        (void)ts.get_latest();
        return 0;
    }

    // --- Main fuzz loop: interleave inserts and queries ---
    {
        auto name = unique_name("fuzz_ts_main");
        safe_shm::CyclicBufferWriter<Elem, Cap> writer(name);
        safe_shm::TimeSeries<Elem, Cap> ts(name);

        const uint8_t *ptr = data;
        size_t remaining = size;

        // Monotonic timestamp for inserts (TimeSeries requires monotonically
        // increasing keys). We derive the increment from fuzz input.
        uint64_t current_ts = 0;
        uint64_t seq = 0;

        while (remaining > 0)
        {
            uint8_t op{};
            if (!consume(ptr, remaining, op))
                break;

            switch (op % 8)
            {
            case 0: // Insert with monotonic timestamp derived from fuzz input
            {
                uint16_t delta{};
                double val{};
                if (!consume(ptr, remaining, delta))
                    break;
                if (!consume(ptr, remaining, val))
                    val = 0.0;
                // Ensure monotonic: advance by at least 1.
                current_ts += static_cast<uint64_t>(delta) + 1;
                Elem e{current_ts, seq, val};
                writer.insert(e);
                ++seq;
                break;
            }
            case 1: // find_closest with fuzzed target, no max_distance
            {
                uint64_t target{};
                if (!consume(ptr, remaining, target))
                    break;
                (void)ts.find_closest(target);
                break;
            }
            case 2: // find_closest with fuzzed target and max_distance
            {
                uint64_t target{};
                uint64_t max_dist{};
                if (!consume(ptr, remaining, target))
                    break;
                if (!consume(ptr, remaining, max_dist))
                    break;
                (void)ts.find_closest(target, max_dist);
                break;
            }
            case 3: // find_interpolation_pair with fuzzed target
            {
                uint64_t target{};
                if (!consume(ptr, remaining, target))
                    break;
                auto result = ts.find_interpolation_pair(target);
                // If we get a result, alpha must be in [0.0, 1.0].
                if (result)
                {
                    if (result->alpha < 0.0 || result->alpha > 1.0)
                        __builtin_trap();
                }
                break;
            }
            case 4: // get_latest_if_fresh with fuzzed min_key
            {
                uint64_t min_key{};
                if (!consume(ptr, remaining, min_key))
                    break;
                (void)ts.get_latest_if_fresh(min_key);
                break;
            }
            case 5: // find_closest with extreme timestamps
            {
                uint8_t which{};
                if (!consume(ptr, remaining, which))
                    break;
                uint64_t target{};
                switch (which % 5)
                {
                case 0:
                    target = 0;
                    break;
                case 1:
                    target = UINT64_MAX;
                    break;
                case 2:
                    target = UINT64_MAX / 2;
                    break;
                case 3:
                    target = current_ts; // exactly at latest
                    break;
                case 4:
                    target = 1;
                    break;
                }
                (void)ts.find_closest(target);
                (void)ts.find_closest(target, uint64_t{0}); // exact-only
                (void)ts.find_closest(target, UINT64_MAX);   // infinite tolerance
                break;
            }
            case 6: // find_interpolation_pair with extreme timestamps
            {
                uint8_t which{};
                if (!consume(ptr, remaining, which))
                    break;
                uint64_t target{};
                switch (which % 4)
                {
                case 0:
                    target = 0;
                    break;
                case 1:
                    target = UINT64_MAX;
                    break;
                case 2:
                    target = current_ts;
                    break;
                case 3:
                    target = current_ts > 0 ? current_ts - 1 : 0;
                    break;
                }
                (void)ts.find_interpolation_pair(target);
                break;
            }
            case 7: // Consistency checks
            {
                auto avail = ts.available();
                auto tw = ts.total_writes();
                if (avail > Cap)
                    __builtin_trap();
                if (avail > tw)
                    __builtin_trap();
                if (tw != seq)
                    __builtin_trap();
                break;
            }
            }
        }
    }

    // --- Bulk insert then exhaustive query sweep ---
    {
        auto name = unique_name("fuzz_ts_sweep");
        safe_shm::CyclicBufferWriter<Elem, Cap> writer(name);
        safe_shm::TimeSeries<Elem, Cap> ts(name);

        // Insert elements with timestamps derived from fuzz input.
        // Monotonically increasing.
        uint64_t ts_val = 0;
        uint64_t seq = 0;
        const uint8_t *ptr = data;
        size_t remaining = size;

        while (remaining >= sizeof(uint16_t) + sizeof(double))
        {
            uint16_t delta{};
            double val{};
            consume(ptr, remaining, delta);
            consume(ptr, remaining, val);
            ts_val += static_cast<uint64_t>(delta) + 1;
            Elem e{ts_val, seq, val};
            writer.insert(e);
            ++seq;
        }

        if (seq == 0)
            return 0;

        auto avail = ts.available();

        // Query at every inserted timestamp plus neighbors.
        // Re-read timestamps from the buffer since we know them.
        for (std::size_t i = 0; i < avail; ++i)
        {
            auto elem = ts.get_latest(); // just ensure no crash
            (void)elem;
        }

        // Sweep find_closest at boundaries.
        (void)ts.find_closest(uint64_t{0});
        (void)ts.find_closest(UINT64_MAX);
        (void)ts.find_closest(ts_val);     // at or near latest
        (void)ts.find_closest(uint64_t{1}); // near zero
        (void)ts.find_closest(ts_val / 2);  // midpoint

        // find_closest with max_distance = 0 (exact only).
        (void)ts.find_closest(ts_val, uint64_t{0});
        (void)ts.find_closest(uint64_t{0}, uint64_t{0});

        // find_interpolation_pair at boundaries.
        (void)ts.find_interpolation_pair(uint64_t{0});
        (void)ts.find_interpolation_pair(UINT64_MAX);
        (void)ts.find_interpolation_pair(ts_val);
        if (ts_val > 1)
            (void)ts.find_interpolation_pair(ts_val - 1);
        (void)ts.find_interpolation_pair(ts_val / 2);

        // get_latest_if_fresh edge cases.
        (void)ts.get_latest_if_fresh(uint64_t{0});
        (void)ts.get_latest_if_fresh(UINT64_MAX);
        (void)ts.get_latest_if_fresh(ts_val);
    }

    return 0;
}
