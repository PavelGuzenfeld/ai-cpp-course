#pragma once
#include "safe-shm/flat_type.hpp"
#include <cstdint>
#include <type_traits>

namespace safe_shm
{
    /// Diagnostic counters for shared memory primitives.
    /// All fields are plain uint64_t — updated via local (non-shared) atomic operations.
    struct ShmStats
    {
        uint64_t total_writes{0};
        uint64_t total_reads{0};
        uint64_t torn_read_retries{0};
        uint64_t lock_contentions{0};
    };

    namespace detail
    {
        /// Zero-size placeholder when stats are disabled.
        /// Used with [[no_unique_address]] for zero overhead.
        struct Empty
        {
        };

        template <bool EnableStats>
        using stats_member_t = std::conditional_t<EnableStats, ShmStats, Empty>;
    } // namespace detail

} // namespace safe_shm
