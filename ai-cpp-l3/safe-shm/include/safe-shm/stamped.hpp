#pragma once
#include "safe-shm/flat_type.hpp"
#include <cstdint>
#include <time.h>

namespace safe_shm
{
    /// Metadata envelope: wraps any FlatType with a monotonic timestamp and sequence number.
    /// Stamped<T> is itself a FlatType when T is a FlatType.
    template <FlatType T>
    struct Stamped
    {
        uint64_t timestamp_ns; // CLOCK_MONOTONIC nanoseconds
        uint64_t sequence;     // writer-assigned monotonic counter
        T data;
    };

    static_assert(FlatType<Stamped<int>>);
    static_assert(FlatType<Stamped<double>>);

    /// Get current monotonic time in nanoseconds.
    inline uint64_t monotonic_now_ns() noexcept
    {
        struct timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
               static_cast<uint64_t>(ts.tv_nsec);
    }

    /// Create a Stamped<T> with the current monotonic time.
    template <FlatType T>
    Stamped<T> stamp(T const &data, uint64_t seq) noexcept
    {
        return {monotonic_now_ns(), seq, data};
    }

} // namespace safe_shm
