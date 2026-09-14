#pragma once
#include "safe-shm/flat_type.hpp"
#include "shm/shm.hpp"
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstring>
#include <linux/futex.h>
#include <optional>
#include <sys/syscall.h>
#include <unistd.h>

// Reuse the TSan suppression macro from seqlock.hpp
#if defined(__SANITIZE_THREAD__)
#define SAFE_SHM_CB_TSAN_ACTIVE 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define SAFE_SHM_CB_TSAN_ACTIVE 1
#endif
#endif

#ifdef SAFE_SHM_CB_TSAN_ACTIVE
#define SAFE_SHM_CB_NO_TSAN __attribute__((no_sanitize("thread")))
#else
#define SAFE_SHM_CB_NO_TSAN
#endif

namespace safe_shm
{
    namespace detail
    {
        /// Per-slot layout: seqlock counter + aligned data.
        /// Each slot has its own sequence counter so readers and writers
        /// on different slots never contend.
        template <FlatType T>
        struct CyclicSlot
        {
            static constexpr std::size_t seq_size = sizeof(uint32_t);
            static constexpr std::size_t data_offset =
                (seq_size + alignof(T) - 1) & ~(alignof(T) - 1);
            static constexpr std::size_t slot_size = data_offset + sizeof(T);

            // These are byte-offset helpers, not members.
            // The actual layout in shared memory is:
            //   [uint32_t seq] [padding] [T data]
        };

        /// Full shared-memory region layout for a cyclic buffer.
        /// [uint64_t total_writes] [padding] [slot_0] [slot_1] ... [slot_{N-1}]
        template <FlatType T, std::size_t N>
        struct CyclicRegion
        {
            static constexpr std::size_t header_size = sizeof(uint64_t);
            // Align first slot to max(alignof(uint32_t), alignof(T))
            static constexpr std::size_t slots_offset =
                (header_size + alignof(T) - 1) & ~(alignof(T) - 1);
            static constexpr std::size_t slot_size = CyclicSlot<T>::slot_size;
            static constexpr std::size_t total_size = slots_offset + slot_size * N;
        };
    } // namespace detail

    /// Lock-free cyclic buffer writer for cross-process shared memory.
    /// Single writer; per-slot seqlock protects against torn reads.
    /// Capacity must be a power of two for fast modulo.
    template <FlatType T, std::size_t N>
    class CyclicBufferWriter
    {
        static_assert(N > 0 && (N & (N - 1)) == 0,
                      "CyclicBuffer capacity must be a power of two");

    public:
        explicit CyclicBufferWriter(std::string const &shm_name)
            : shm_(shm::path(shm_name), detail::CyclicRegion<T, N>::total_size)
        {
        }

        SAFE_SHM_CB_NO_TSAN void insert(T const &data) noexcept
        {
            auto writes = total_writes_ref().load(std::memory_order_acquire);
            auto idx = writes & (N - 1);

            auto *s = slot_seq(idx);
            auto cur = std::atomic_ref<uint32_t>(*s).load(std::memory_order_relaxed);
            // Odd = write in progress
            std::atomic_ref<uint32_t>(*s).store(cur + 1, std::memory_order_release);

            std::memcpy(slot_data(idx), &data, sizeof(T));

            // Even = write complete
            std::atomic_ref<uint32_t>(*s).store(cur + 2, std::memory_order_release);

            // Increment total writes and wake any blocking readers
            total_writes_ref().store(writes + 1, std::memory_order_release);
            auto *tw = total_writes_ptr();
            static_cast<void>(syscall(SYS_futex, tw, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0));
        }

        uint64_t total_writes() const noexcept
        {
            return total_writes_cref().load(std::memory_order_acquire);
        }

    private:
        std::atomic_ref<uint64_t> total_writes_ref() noexcept
        {
            return std::atomic_ref<uint64_t>(*static_cast<uint64_t *>(shm_.get()));
        }

        std::atomic_ref<uint64_t const> total_writes_cref() const noexcept
        {
            return std::atomic_ref<uint64_t const>(
                *static_cast<uint64_t const *>(shm_.get()));
        }

        uint32_t *total_writes_ptr() noexcept
        {
            // Point to the lower 32 bits for futex (futex operates on uint32_t)
            return static_cast<uint32_t *>(shm_.get());
        }

        uint32_t *slot_seq(std::size_t idx) noexcept
        {
            auto *base = static_cast<char *>(shm_.get()) +
                         detail::CyclicRegion<T, N>::slots_offset +
                         idx * detail::CyclicSlot<T>::slot_size;
            return reinterpret_cast<uint32_t *>(base);
        }

        void *slot_data(std::size_t idx) noexcept
        {
            auto *base = static_cast<char *>(shm_.get()) +
                         detail::CyclicRegion<T, N>::slots_offset +
                         idx * detail::CyclicSlot<T>::slot_size +
                         detail::CyclicSlot<T>::data_offset;
            return base;
        }

        shm::Shm shm_;
    };

    /// Lock-free cyclic buffer reader for cross-process shared memory.
    /// Supports multiple concurrent readers.
    template <FlatType T, std::size_t N>
    class CyclicBufferReader
    {
        static_assert(N > 0 && (N & (N - 1)) == 0,
                      "CyclicBuffer capacity must be a power of two");

    public:
        explicit CyclicBufferReader(std::string const &shm_name)
            : shm_(shm::path(shm_name), detail::CyclicRegion<T, N>::total_size)
        {
        }

        /// Read the most recently written element. Spins on torn reads.
        SAFE_SHM_CB_NO_TSAN T get_latest() const noexcept
        {
            auto writes = total_writes_ref().load(std::memory_order_acquire);
            if (writes == 0)
                return T{};
            return read_slot((writes - 1) & (N - 1));
        }

        /// Read element at reverse_index (0 = latest, 1 = previous, ...).
        /// Spins on torn reads. UB if reverse_index >= available().
        SAFE_SHM_CB_NO_TSAN T get(std::size_t reverse_index) const noexcept
        {
            auto writes = total_writes_ref().load(std::memory_order_acquire);
            auto idx = (writes - 1 - reverse_index) & (N - 1);
            return read_slot(idx);
        }

        /// Try to read element at reverse_index. Returns nullopt if the slot
        /// was overwritten during the read (after max_retries attempts).
        SAFE_SHM_CB_NO_TSAN std::optional<T> try_get(
            std::size_t reverse_index,
            unsigned max_retries = 4) const noexcept
        {
            auto writes = total_writes_ref().load(std::memory_order_acquire);
            if (reverse_index >= std::min(writes, uint64_t{N}))
                return std::nullopt;
            auto idx = (writes - 1 - reverse_index) & (N - 1);
            return try_read_slot(idx, max_retries);
        }

        /// Number of readable elements: min(total_writes, N).
        std::size_t available() const noexcept
        {
            auto writes = total_writes_ref().load(std::memory_order_acquire);
            return static_cast<std::size_t>(std::min(writes, uint64_t{N}));
        }

        uint64_t total_writes() const noexcept
        {
            return total_writes_ref().load(std::memory_order_acquire);
        }

        /// Block until total_writes changes from last_seen.
        /// Returns the new total_writes value, or nullopt on timeout.
        std::optional<uint64_t> wait_for_write(
            uint64_t last_seen,
            std::optional<uint64_t> timeout_ns = std::nullopt) const noexcept
        {
            auto *tw = total_writes_futex_ptr();
            struct timespec ts{};
            struct timespec *ts_ptr = nullptr;
            if (timeout_ns)
            {
                ts.tv_sec = static_cast<time_t>(*timeout_ns / 1'000'000'000ULL);
                ts.tv_nsec = static_cast<long>(*timeout_ns % 1'000'000'000ULL);
                ts_ptr = &ts;
            }
            while (true)
            {
                auto cur = total_writes_ref().load(std::memory_order_acquire);
                if (cur != last_seen)
                    return cur;
                auto expected_lo = static_cast<uint32_t>(last_seen);
                auto rc = syscall(SYS_futex, tw, FUTEX_WAIT, expected_lo,
                                  ts_ptr, nullptr, 0);
                if (rc == -1 && errno == ETIMEDOUT)
                    return std::nullopt;
            }
        }

        static constexpr std::size_t capacity() noexcept { return N; }

    private:
        SAFE_SHM_CB_NO_TSAN T read_slot(std::size_t idx) const noexcept
        {
            T result;
            auto const *s = slot_seq(idx);
            while (true)
            {
                auto before = std::atomic_ref<uint32_t const>(*s)
                                  .load(std::memory_order_acquire);
                if (before & 1u)
                    continue; // write in progress
                std::memcpy(&result, slot_data(idx), sizeof(T));
                auto after = std::atomic_ref<uint32_t const>(*s)
                                 .load(std::memory_order_acquire);
                if (before == after)
                    return result;
            }
        }

        SAFE_SHM_CB_NO_TSAN std::optional<T> try_read_slot(
            std::size_t idx, unsigned max_retries) const noexcept
        {
            T result;
            auto const *s = slot_seq(idx);
            for (unsigned attempt = 0; attempt <= max_retries; ++attempt)
            {
                auto before = std::atomic_ref<uint32_t const>(*s)
                                  .load(std::memory_order_acquire);
                if (before & 1u)
                    continue;
                std::memcpy(&result, slot_data(idx), sizeof(T));
                auto after = std::atomic_ref<uint32_t const>(*s)
                                 .load(std::memory_order_acquire);
                if (before == after)
                    return result;
            }
            return std::nullopt;
        }

        std::atomic_ref<uint64_t const> total_writes_ref() const noexcept
        {
            return std::atomic_ref<uint64_t const>(
                *static_cast<uint64_t const *>(shm_.get()));
        }

        uint32_t const *total_writes_futex_ptr() const noexcept
        {
            return static_cast<uint32_t const *>(shm_.get());
        }

        uint32_t const *slot_seq(std::size_t idx) const noexcept
        {
            auto const *base = static_cast<char const *>(shm_.get()) +
                               detail::CyclicRegion<T, N>::slots_offset +
                               idx * detail::CyclicSlot<T>::slot_size;
            return reinterpret_cast<uint32_t const *>(base);
        }

        void const *slot_data(std::size_t idx) const noexcept
        {
            auto const *base = static_cast<char const *>(shm_.get()) +
                               detail::CyclicRegion<T, N>::slots_offset +
                               idx * detail::CyclicSlot<T>::slot_size +
                               detail::CyclicSlot<T>::data_offset;
            return base;
        }

        shm::Shm shm_;
    };

    /// Combined reader/writer cyclic buffer for same-process use.
    template <FlatType T, std::size_t N>
    class CyclicBuffer
    {
        static_assert(N > 0 && (N & (N - 1)) == 0,
                      "CyclicBuffer capacity must be a power of two");

    public:
        explicit CyclicBuffer(std::string const &shm_name)
            : shm_(shm::path(shm_name), detail::CyclicRegion<T, N>::total_size)
        {
        }

        SAFE_SHM_CB_NO_TSAN void insert(T const &data) noexcept
        {
            auto writes = total_writes_ref().load(std::memory_order_acquire);
            auto idx = writes & (N - 1);

            auto *s = slot_seq(idx);
            auto cur = std::atomic_ref<uint32_t>(*s).load(std::memory_order_relaxed);
            std::atomic_ref<uint32_t>(*s).store(cur + 1, std::memory_order_release);
            std::memcpy(slot_data(idx), &data, sizeof(T));
            std::atomic_ref<uint32_t>(*s).store(cur + 2, std::memory_order_release);

            total_writes_ref().store(writes + 1, std::memory_order_release);
            auto *tw = total_writes_ptr();
            static_cast<void>(syscall(SYS_futex, tw, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0));
        }

        SAFE_SHM_CB_NO_TSAN T get_latest() const noexcept
        {
            auto writes = total_writes_cref().load(std::memory_order_acquire);
            if (writes == 0)
                return T{};
            return read_slot((writes - 1) & (N - 1));
        }

        SAFE_SHM_CB_NO_TSAN T get(std::size_t reverse_index) const noexcept
        {
            auto writes = total_writes_cref().load(std::memory_order_acquire);
            auto idx = (writes - 1 - reverse_index) & (N - 1);
            return read_slot(idx);
        }

        SAFE_SHM_CB_NO_TSAN std::optional<T> try_get(
            std::size_t reverse_index,
            unsigned max_retries = 4) const noexcept
        {
            auto writes = total_writes_cref().load(std::memory_order_acquire);
            if (reverse_index >= std::min(writes, uint64_t{N}))
                return std::nullopt;
            auto idx = (writes - 1 - reverse_index) & (N - 1);
            return try_read_slot(idx, max_retries);
        }

        std::size_t available() const noexcept
        {
            auto writes = total_writes_cref().load(std::memory_order_acquire);
            return static_cast<std::size_t>(std::min(writes, uint64_t{N}));
        }

        uint64_t total_writes() const noexcept
        {
            return total_writes_cref().load(std::memory_order_acquire);
        }

        static constexpr std::size_t capacity() noexcept { return N; }

    private:
        SAFE_SHM_CB_NO_TSAN T read_slot(std::size_t idx) const noexcept
        {
            T result;
            auto const *s = slot_seq_c(idx);
            while (true)
            {
                auto before = std::atomic_ref<uint32_t const>(*s)
                                  .load(std::memory_order_acquire);
                if (before & 1u)
                    continue;
                std::memcpy(&result, slot_data_c(idx), sizeof(T));
                auto after = std::atomic_ref<uint32_t const>(*s)
                                 .load(std::memory_order_acquire);
                if (before == after)
                    return result;
            }
        }

        SAFE_SHM_CB_NO_TSAN std::optional<T> try_read_slot(
            std::size_t idx, unsigned max_retries) const noexcept
        {
            T result;
            auto const *s = slot_seq_c(idx);
            for (unsigned attempt = 0; attempt <= max_retries; ++attempt)
            {
                auto before = std::atomic_ref<uint32_t const>(*s)
                                  .load(std::memory_order_acquire);
                if (before & 1u)
                    continue;
                std::memcpy(&result, slot_data_c(idx), sizeof(T));
                auto after = std::atomic_ref<uint32_t const>(*s)
                                 .load(std::memory_order_acquire);
                if (before == after)
                    return result;
            }
            return std::nullopt;
        }

        // Mutable accessors (writer)
        std::atomic_ref<uint64_t> total_writes_ref() noexcept
        {
            return std::atomic_ref<uint64_t>(*static_cast<uint64_t *>(shm_.get()));
        }

        uint32_t *total_writes_ptr() noexcept
        {
            return static_cast<uint32_t *>(shm_.get());
        }

        uint32_t *slot_seq(std::size_t idx) noexcept
        {
            auto *base = static_cast<char *>(shm_.get()) +
                         detail::CyclicRegion<T, N>::slots_offset +
                         idx * detail::CyclicSlot<T>::slot_size;
            return reinterpret_cast<uint32_t *>(base);
        }

        void *slot_data(std::size_t idx) noexcept
        {
            return static_cast<char *>(shm_.get()) +
                   detail::CyclicRegion<T, N>::slots_offset +
                   idx * detail::CyclicSlot<T>::slot_size +
                   detail::CyclicSlot<T>::data_offset;
        }

        // Const accessors (reader)
        std::atomic_ref<uint64_t const> total_writes_cref() const noexcept
        {
            return std::atomic_ref<uint64_t const>(
                *static_cast<uint64_t const *>(shm_.get()));
        }

        uint32_t const *slot_seq_c(std::size_t idx) const noexcept
        {
            auto const *base = static_cast<char const *>(shm_.get()) +
                               detail::CyclicRegion<T, N>::slots_offset +
                               idx * detail::CyclicSlot<T>::slot_size;
            return reinterpret_cast<uint32_t const *>(base);
        }

        void const *slot_data_c(std::size_t idx) const noexcept
        {
            return static_cast<char const *>(shm_.get()) +
                   detail::CyclicRegion<T, N>::slots_offset +
                   idx * detail::CyclicSlot<T>::slot_size +
                   detail::CyclicSlot<T>::data_offset;
        }

        shm::Shm shm_;
    };

} // namespace safe_shm
