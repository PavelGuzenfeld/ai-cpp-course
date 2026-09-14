#pragma once
#include "shm/shm.hpp"
#include <atomic>
#include <cstdint>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace safe_shm
{
    namespace detail
    {
        inline int futex_wait(uint32_t *addr, uint32_t expected) noexcept
        {
            // FUTEX_WAIT (not PRIVATE) — works across processes sharing the same page
            return static_cast<int>(
                syscall(SYS_futex, addr, FUTEX_WAIT, expected, nullptr, nullptr, 0));
        }

        inline int futex_wake(uint32_t *addr, int count) noexcept
        {
            return static_cast<int>(
                syscall(SYS_futex, addr, FUTEX_WAKE, count, nullptr, nullptr, 0));
        }
    } // namespace detail

    /// Cross-process mutex using futex on shared memory.
    /// Replaces POSIX named semaphores with ~10x lower latency.
    /// Works on x86_64 and ARM64 (any Linux with futex support).
    class ShmLock
    {
    public:
        explicit ShmLock(std::string const &name)
            : shm_(shm::path(name), sizeof(uint32_t))
        {
        }

        void lock() noexcept
        {
            auto *s = state();
            // Fast path: uncontended lock (single atomic exchange, no syscall)
            if (std::atomic_ref<uint32_t>(*s).exchange(1, std::memory_order_acquire) == 0)
                return;
            // Slow path: contended — use futex to sleep
            while (std::atomic_ref<uint32_t>(*s).exchange(1, std::memory_order_acquire) != 0)
                detail::futex_wait(s, 1);
        }

        void unlock() noexcept
        {
            auto *s = state();
            std::atomic_ref<uint32_t>(*s).store(0, std::memory_order_release);
            detail::futex_wake(s, 1);
        }

        void destroy() noexcept
        {
            // Ensure unlocked before removing the backing file
            auto *s = state();
            if (s)
                std::atomic_ref<uint32_t>(*s).store(0, std::memory_order_release);
            shm_.destroy();
        }

    private:
        uint32_t *state() noexcept
        {
            return static_cast<uint32_t *>(shm_.get());
        }

        shm::Shm shm_;
    };
} // namespace safe_shm
