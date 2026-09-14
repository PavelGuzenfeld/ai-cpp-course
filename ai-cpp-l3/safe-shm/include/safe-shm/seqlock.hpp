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
#include <time.h>
#include <unistd.h>

// Seqlocks intentionally race on the data payload — the sequence counter
// detects and retries torn reads. Suppress TSan for the data-copy functions.
#if defined(__SANITIZE_THREAD__)
#define SAFE_SHM_TSAN_ACTIVE 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define SAFE_SHM_TSAN_ACTIVE 1
#endif
#endif

#ifdef SAFE_SHM_TSAN_ACTIVE
#define SAFE_SHM_NO_TSAN __attribute__((no_sanitize("thread")))
#else
#define SAFE_SHM_NO_TSAN
#endif

namespace safe_shm
{
    namespace detail
    {
        /// Shared memory layout: [sequence (uint32_t)] [padding] [T data]
        template <FlatType T>
        struct SeqlockRegion
        {
            static constexpr std::size_t seq_size = sizeof(uint32_t);
            static constexpr std::size_t data_offset =
                (seq_size + alignof(T) - 1) & ~(alignof(T) - 1);
            static constexpr std::size_t total_size = data_offset + sizeof(T);
        };
    } // namespace detail

    /// Lock-free writer for shared memory using a sequence counter.
    /// Writer increments sequence to odd before writing, even after.
    /// Multiple writers require external synchronization.
    template <FlatType T>
    class SeqlockWriter
    {
    public:
        explicit SeqlockWriter(std::string const &shm_name)
            : shm_(shm::path(shm_name), detail::SeqlockRegion<T>::total_size)
        {
        }

        SAFE_SHM_NO_TSAN void store(T const &data) noexcept
        {
            auto *s = seq();
            // Odd sequence = write in progress
            auto cur = std::atomic_ref<uint32_t>(*s).load(std::memory_order_relaxed);
            std::atomic_ref<uint32_t>(*s).store(cur + 1, std::memory_order_release);
            // Copy data
            std::memcpy(payload(), &data, sizeof(T));
            // Even sequence = write complete
            std::atomic_ref<uint32_t>(*s).store(cur + 2, std::memory_order_release);
            // Wake all readers blocked on futex wait
            static_cast<void>(syscall(SYS_futex, s, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0));
        }

    private:
        uint32_t *seq() noexcept
        {
            return static_cast<uint32_t *>(shm_.get());
        }

        void *payload() noexcept
        {
            return static_cast<char *>(shm_.get()) + detail::SeqlockRegion<T>::data_offset;
        }

        shm::Shm shm_;
    };

    /// Lock-free reader for shared memory using a sequence counter.
    /// Returns a local copy of the data, guaranteed consistent.
    template <FlatType T>
    class SeqlockReader
    {
    public:
        explicit SeqlockReader(std::string const &shm_name)
            : shm_(shm::path(shm_name), detail::SeqlockRegion<T>::total_size)
        {
        }

        /// Read a consistent snapshot. Retries on torn read.
        /// Non-blocking in practice (torn reads are rare).
        SAFE_SHM_NO_TSAN T load() const noexcept
        {
            T result;
            auto const *s = seq();
            while (true)
            {
                auto before = std::atomic_ref<uint32_t const>(*s)
                                  .load(std::memory_order_acquire);
                if (before & 1u)
                {
                    // Writer in progress — spin briefly then retry
                    continue;
                }
                std::memcpy(&result, payload(), sizeof(T));
                auto after = std::atomic_ref<uint32_t const>(*s)
                                 .load(std::memory_order_acquire);
                if (before == after)
                    return result;
                // Sequence changed during read — retry
            }
        }

        /// Block until new data is available (sequence != last_seen),
        /// then return a consistent snapshot.
        /// Optional timeout in nanoseconds; returns std::nullopt on timeout.
        SAFE_SHM_NO_TSAN std::optional<T> load_blocking(
            uint32_t &last_seq,
            std::optional<uint64_t> timeout_ns = std::nullopt) const noexcept
        {
            auto const *s = seq();
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
                auto cur = std::atomic_ref<uint32_t const>(*s)
                               .load(std::memory_order_acquire);
                if (cur != last_seq && (cur & 1u) == 0)
                {
                    T result;
                    std::memcpy(&result, payload(), sizeof(T));
                    auto after = std::atomic_ref<uint32_t const>(*s)
                                     .load(std::memory_order_acquire);
                    if (cur == after)
                    {
                        last_seq = cur;
                        return result;
                    }
                    continue;
                }
                // No new data — sleep on futex until writer wakes us (or timeout)
                auto rc = syscall(
                    SYS_futex,
                    const_cast<uint32_t *>(s),
                    FUTEX_WAIT, cur,
                    ts_ptr, nullptr, 0);
                if (rc == -1 && errno == ETIMEDOUT)
                    return std::nullopt;
            }
        }

        /// Returns current sequence number (even = stable, odd = write in progress).
        uint32_t sequence() const noexcept
        {
            return std::atomic_ref<uint32_t const>(*seq())
                .load(std::memory_order_acquire);
        }

    private:
        uint32_t const *seq() const noexcept
        {
            return static_cast<uint32_t const *>(shm_.get());
        }

        void const *payload() const noexcept
        {
            return static_cast<char const *>(shm_.get()) + detail::SeqlockRegion<T>::data_offset;
        }

        shm::Shm shm_;
    };

    /// Combined reader/writer seqlock for same-process use.
    template <FlatType T>
    class Seqlock
    {
    public:
        explicit Seqlock(std::string const &shm_name)
            : shm_(shm::path(shm_name), detail::SeqlockRegion<T>::total_size)
        {
        }

        SAFE_SHM_NO_TSAN void store(T const &data) noexcept
        {
            auto *s = seq();
            auto cur = std::atomic_ref<uint32_t>(*s).load(std::memory_order_relaxed);
            std::atomic_ref<uint32_t>(*s).store(cur + 1, std::memory_order_release);
            std::memcpy(payload(), &data, sizeof(T));
            std::atomic_ref<uint32_t>(*s).store(cur + 2, std::memory_order_release);
            static_cast<void>(syscall(SYS_futex, s, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0));
        }

        SAFE_SHM_NO_TSAN T load() const noexcept
        {
            T result;
            auto const *s = seq();
            while (true)
            {
                auto before = std::atomic_ref<uint32_t const>(*s)
                                  .load(std::memory_order_acquire);
                if (before & 1u)
                    continue;
                std::memcpy(&result, payload(), sizeof(T));
                auto after = std::atomic_ref<uint32_t const>(*s)
                                 .load(std::memory_order_acquire);
                if (before == after)
                    return result;
            }
        }

        /// Block until new data is available (sequence != last_seen).
        /// Optional timeout in nanoseconds; returns std::nullopt on timeout.
        SAFE_SHM_NO_TSAN std::optional<T> load_blocking(
            uint32_t &last_seq,
            std::optional<uint64_t> timeout_ns = std::nullopt) const noexcept
        {
            auto const *s = seq();
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
                auto cur = std::atomic_ref<uint32_t const>(*s)
                               .load(std::memory_order_acquire);
                if (cur != last_seq && (cur & 1u) == 0)
                {
                    T result;
                    std::memcpy(&result, payload(), sizeof(T));
                    auto after = std::atomic_ref<uint32_t const>(*s)
                                     .load(std::memory_order_acquire);
                    if (cur == after)
                    {
                        last_seq = cur;
                        return result;
                    }
                    continue;
                }
                auto rc = syscall(
                    SYS_futex,
                    const_cast<uint32_t *>(s),
                    FUTEX_WAIT, cur,
                    ts_ptr, nullptr, 0);
                if (rc == -1 && errno == ETIMEDOUT)
                    return std::nullopt;
            }
        }

        uint32_t sequence() const noexcept
        {
            return std::atomic_ref<uint32_t const>(*seq())
                .load(std::memory_order_acquire);
        }

    private:
        uint32_t *seq() noexcept
        {
            return static_cast<uint32_t *>(shm_.get());
        }

        uint32_t const *seq() const noexcept
        {
            return static_cast<uint32_t const *>(shm_.get());
        }

        void *payload() noexcept
        {
            return static_cast<char *>(shm_.get()) + detail::SeqlockRegion<T>::data_offset;
        }

        void const *payload() const noexcept
        {
            return static_cast<char const *>(shm_.get()) + detail::SeqlockRegion<T>::data_offset;
        }

        shm::Shm shm_;
    };
} // namespace safe_shm
