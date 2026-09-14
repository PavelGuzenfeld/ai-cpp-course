#pragma once
#include "safe-shm/cyclic_buffer.hpp"
#include "safe-shm/sanitized_key.hpp"
#include "safe-shm/stamped.hpp"
#include <algorithm>
#include <cmath>
#include <optional>

namespace safe_shm
{
    /// Interpolation pair: two bracketing elements and the interpolation factor.
    template <FlatType T>
    struct InterpPair
    {
        T before;
        T after;
        double alpha; // 0.0 = exactly 'before', 1.0 = exactly 'after'
    };

    /// Default key extractor for Stamped<T>: extracts timestamp_ns.
    struct StampedKeyExtractor
    {
        template <FlatType T>
        uint64_t operator()(Stamped<T> const &s) const noexcept
        {
            return s.timestamp_ns;
        }
    };

    /// Key-based temporal access layer on top of CyclicBufferReader.
    /// Provides closest-element lookup, interpolation pair finding,
    /// and staleness detection.
    ///
    /// Elements must be inserted with monotonically increasing keys.
    /// KeyExtractor must be callable with (T const&) -> comparable type.
    template <FlatType T, std::size_t N, typename KeyExtractor = StampedKeyExtractor>
    class TimeSeries
    {
    public:
        using key_type = std::invoke_result_t<KeyExtractor, T const &>;

        explicit TimeSeries(std::string const &shm_name, KeyExtractor extractor = {})
            : reader_(shm_name), extractor_(extractor)
        {
        }

        /// Find the element whose key is closest to the target.
        /// If max_distance is provided, returns nullopt when the best match
        /// is farther than max_distance (catches unit/clock-type mismatches).
        /// Returns nullopt if the buffer is empty.
        std::optional<T> find_closest(
            key_type target,
            std::optional<key_type> max_distance = std::nullopt) const
        {
            auto avail = reader_.available();
            if (avail == 0)
                return std::nullopt;

            // Binary search on reversed view (reverse_index 0 = newest = highest key)
            std::size_t lo = 0;
            std::size_t hi = avail - 1;

            while (lo < hi)
            {
                auto mid = lo + (hi - lo) / 2;
                auto elem = reader_.try_get(mid);
                if (!elem)
                {
                    // Slot was overwritten, fall back to linear
                    return find_closest_linear(target, avail, max_distance);
                }
                auto mid_key = extractor_(*elem);
                if (mid_key > target)
                    lo = mid + 1; // go to older elements (higher reverse_index)
                else if (mid_key < target)
                {
                    if (mid == 0)
                        break;
                    hi = mid - 1; // go to newer elements (lower reverse_index)
                }
                else
                    return elem; // exact match
            }

            // Check lo and neighbors
            auto best = reader_.try_get(lo);
            if (!best)
                return find_closest_linear(target, avail);

            auto best_key = extractor_(*best);
            auto best_diff = key_diff(best_key, target);

            // Check one neighbor on each side
            if (lo > 0)
            {
                if (auto prev = reader_.try_get(lo - 1))
                {
                    auto d = key_diff(extractor_(*prev), target);
                    if (d < best_diff)
                    {
                        best = prev;
                        best_diff = d;
                    }
                }
            }
            if (lo + 1 < avail)
            {
                if (auto next = reader_.try_get(lo + 1))
                {
                    auto d = key_diff(extractor_(*next), target);
                    if (d < best_diff)
                    {
                        best = next;
                    }
                }
            }

            // Apply max_distance filter if requested
            if (best && max_distance)
            {
                auto d = key_diff(extractor_(*best), target);
                if (d > *max_distance)
                    return std::nullopt;
            }

            return best;
        }

        /// Find the two adjacent elements bracketing the target key.
        /// Returns nullopt if fewer than 2 elements or target is out of range.
        std::optional<InterpPair<T>> find_interpolation_pair(key_type target) const
        {
            auto avail = reader_.available();
            if (avail < 2)
                return std::nullopt;

            // Newest element (reverse_index 0) has the highest key
            auto newest = reader_.try_get(0);
            auto oldest = reader_.try_get(avail - 1);
            if (!newest || !oldest)
                return std::nullopt;

            auto newest_key = extractor_(*newest);
            auto oldest_key = extractor_(*oldest);

            // Out of range check
            if (target > newest_key || target < oldest_key)
                return std::nullopt;

            // Exact match on boundary
            if (target == newest_key)
                return InterpPair<T>{*newest, *newest, 0.0};
            if (target == oldest_key)
                return InterpPair<T>{*oldest, *oldest, 0.0};

            // Binary search for the bracketing pair
            // We need to find reverse_index i such that key(i) <= target < key(i-1)
            // (remember: keys decrease with increasing reverse_index)
            std::size_t lo = 0;
            std::size_t hi = avail - 1;

            while (lo + 1 < hi)
            {
                auto mid = lo + (hi - lo) / 2;
                auto elem = reader_.try_get(mid);
                if (!elem)
                    return find_interpolation_pair_linear(target, avail);

                auto mid_key = extractor_(*elem);
                if (mid_key > target)
                    lo = mid;
                else
                    hi = mid;
            }

            auto after_elem = reader_.try_get(lo);   // newer, higher key
            auto before_elem = reader_.try_get(hi);   // older, lower key
            if (!after_elem || !before_elem)
                return std::nullopt;

            auto after_key = extractor_(*after_elem);
            auto before_key = extractor_(*before_elem);

            if (after_key == before_key)
                return InterpPair<T>{*before_elem, *after_elem, 0.0};

            double alpha = static_cast<double>(target - before_key) /
                           static_cast<double>(after_key - before_key);

            return InterpPair<T>{*before_elem, *after_elem, alpha};
        }

        /// Return the latest element only if its key >= min_key.
        /// Used for staleness detection.
        std::optional<T> get_latest_if_fresh(key_type min_key) const
        {
            if (reader_.available() == 0)
                return std::nullopt;

            auto latest = reader_.try_get(0);
            if (!latest)
                return std::nullopt;

            if (extractor_(*latest) >= min_key)
                return latest;
            return std::nullopt;
        }

        // ── SanitizedKey overloads (opt-in type-safe API) ─────────────────
        // Accept any SanitizedKey<Tag>. The compiler prevents passing raw
        // uint64_t where SanitizedKey is expected, catching unit/clock bugs
        // at compile time instead of runtime.

        template <typename Tag>
        std::optional<T> find_closest(
            SanitizedKey<Tag> target,
            std::optional<key_type> max_distance = std::nullopt) const
        {
            return find_closest(target.value(), max_distance);
        }

        template <typename Tag>
        std::optional<InterpPair<T>> find_interpolation_pair(
            SanitizedKey<Tag> target) const
        {
            return find_interpolation_pair(target.value());
        }

        template <typename Tag>
        std::optional<T> get_latest_if_fresh(SanitizedKey<Tag> min_key) const
        {
            return get_latest_if_fresh(min_key.value());
        }

        // Delegate to underlying reader
        T get_latest() const noexcept { return reader_.get_latest(); }
        std::size_t available() const noexcept { return reader_.available(); }
        uint64_t total_writes() const noexcept { return reader_.total_writes(); }
        static constexpr std::size_t capacity() noexcept { return N; }

    private:
        static auto key_diff(key_type a, key_type b)
        {
            return a > b ? a - b : b - a;
        }

        /// Fallback linear search when binary search encounters overwritten slots.
        std::optional<T> find_closest_linear(
            key_type target, std::size_t avail,
            std::optional<key_type> max_distance = std::nullopt) const
        {
            std::optional<T> best;
            auto best_diff = std::numeric_limits<key_type>::max();

            for (std::size_t i = 0; i < avail; ++i)
            {
                auto elem = reader_.try_get(i);
                if (!elem)
                    continue;
                auto d = key_diff(extractor_(*elem), target);
                if (d < best_diff)
                {
                    best = elem;
                    best_diff = d;
                }
            }

            // Apply max_distance filter
            if (best && max_distance && best_diff > *max_distance)
                return std::nullopt;

            return best;
        }

        std::optional<InterpPair<T>> find_interpolation_pair_linear(
            key_type target, std::size_t avail) const
        {
            // Find the two elements bracketing target with linear scan
            std::optional<T> before_elem, after_elem;
            key_type before_key{}, after_key{};
            auto before_diff = std::numeric_limits<key_type>::max();
            auto after_diff = std::numeric_limits<key_type>::max();

            for (std::size_t i = 0; i < avail; ++i)
            {
                auto elem = reader_.try_get(i);
                if (!elem)
                    continue;
                auto k = extractor_(*elem);
                if (k <= target)
                {
                    auto d = target - k;
                    if (d < before_diff)
                    {
                        before_elem = elem;
                        before_key = k;
                        before_diff = d;
                    }
                }
                if (k >= target)
                {
                    auto d = k - target;
                    if (d < after_diff)
                    {
                        after_elem = elem;
                        after_key = k;
                        after_diff = d;
                    }
                }
            }

            if (!before_elem || !after_elem)
                return std::nullopt;

            double alpha = (after_key == before_key)
                               ? 0.0
                               : static_cast<double>(target - before_key) /
                                     static_cast<double>(after_key - before_key);

            return InterpPair<T>{*before_elem, *after_elem, alpha};
        }

        CyclicBufferReader<T, N> reader_;
        [[no_unique_address]] KeyExtractor extractor_;
    };

    /// Convenience alias: writer side is just a CyclicBufferWriter.
    template <FlatType T, std::size_t N>
    using TimeSeriesWriter = CyclicBufferWriter<T, N>;

} // namespace safe_shm
