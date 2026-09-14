#include "exception-rt/exception.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <unistd.h> // write

namespace
{

constexpr std::size_t slot_size = EXCEPTION_RT_ABI_HEADER_SIZE + EXCEPTION_RT_MAX_THROWN_SIZE;

struct alignas(alignof(std::max_align_t)) Slot
{
    std::uint8_t data[slot_size];
};

std::array<Slot, EXCEPTION_RT_NUM_SLOTS> pool{};
std::array<std::atomic<bool>, EXCEPTION_RT_NUM_SLOTS> slot_used{};

// Per-thread hint to reduce contention on the first slots
thread_local std::size_t hint = 0;

// write-based abort that avoids pulling in fprintf / __gxx_personality_v0
[[noreturn]] void fatal(const char *msg) noexcept
{
    [[maybe_unused]] auto _ = ::write(STDERR_FILENO, msg, __builtin_strlen(msg));
    __builtin_trap();
}

} // namespace

extern "C"
{

    void *__cxa_allocate_exception(std::size_t thrown_size) noexcept
    {
        if (thrown_size > EXCEPTION_RT_MAX_THROWN_SIZE)
        {
            fatal("exception-rt: thrown object size exceeds EXCEPTION_RT_MAX_THROWN_SIZE\n");
        }

        for (std::size_t n = 0; n < EXCEPTION_RT_NUM_SLOTS; ++n)
        {
            auto i = (hint + n) % EXCEPTION_RT_NUM_SLOTS;
            bool expected = false;
            if (slot_used[i].compare_exchange_strong(expected, true,
                                                     std::memory_order_acquire, std::memory_order_relaxed))
            {
                hint = (i + 1) % EXCEPTION_RT_NUM_SLOTS;
                auto *base = pool[i].data;
                std::memset(base, 0, EXCEPTION_RT_ABI_HEADER_SIZE);
                return base + EXCEPTION_RT_ABI_HEADER_SIZE;
            }
        }

        fatal("exception-rt: all exception slots exhausted\n");
    }

    void __cxa_free_exception(void *thrown_object) noexcept
    {
        if (!thrown_object) return;

        auto *ptr = static_cast<std::uint8_t *>(thrown_object);
        for (std::size_t i = 0; i < EXCEPTION_RT_NUM_SLOTS; ++i)
        {
            auto *expected = pool[i].data + EXCEPTION_RT_ABI_HEADER_SIZE;
            if (ptr == expected)
            {
                slot_used[i].store(false, std::memory_order_release);
                return;
            }
        }

        fatal("exception-rt: freeing unknown exception pointer\n");
    }

} // extern "C"

namespace exception_rt
{

std::size_t slots_in_use() noexcept
{
    std::size_t count = 0;
    for (std::size_t i = 0; i < EXCEPTION_RT_NUM_SLOTS; ++i)
    {
        if (slot_used[i].load(std::memory_order_relaxed))
        {
            ++count;
        }
    }
    return count;
}

bool owns(const void *ptr) noexcept
{
    if (!ptr) return false;
    auto *p = static_cast<const std::uint8_t *>(ptr);
    auto *pool_start = pool[0].data;
    auto *pool_end = pool[EXCEPTION_RT_NUM_SLOTS - 1].data + slot_size;
    if (p < pool_start || p >= pool_end) return false;
    auto offset = static_cast<std::size_t>(p - pool_start);
    return (offset % slot_size) == EXCEPTION_RT_ABI_HEADER_SIZE;
}

} // namespace exception_rt
