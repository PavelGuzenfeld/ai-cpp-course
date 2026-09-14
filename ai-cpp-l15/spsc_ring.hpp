#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <optional>
#include <thread>

// Single-producer/single-consumer ring buffer. The publish sequence is:
// write the slot, then publish the new head with a release store. A
// consumer that observes the new head with an acquire load is guaranteed to
// see the slot write that preceded it -- the same ordering a cross-process
// consumer over shared memory depends on (see L16).
template <typename T, std::size_t N>
class SpscRing
{
    static_assert((N & (N - 1)) == 0, "N must be a power of two");

public:
    bool push(T const &value) noexcept
    {
        std::size_t const head = head_.load(std::memory_order_relaxed);
        std::size_t const next = (head + 1) & (N - 1);
        if (next == tail_.load(std::memory_order_acquire))
        {
            return false; // full
        }
        buffer_[head] = value;
        head_.store(next, std::memory_order_release);
        return true;
    }

    std::optional<T> try_pop() noexcept
    {
        std::size_t const tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire))
        {
            return std::nullopt; // empty
        }
        T value = buffer_[tail];
        tail_.store((tail + 1) & (N - 1), std::memory_order_release);
        return value;
    }

private:
    std::array<T, N> buffer_{};
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> tail_{0};
};

// The bug class from PRODUCTION_PLAN.md 3.4: the index is published
// *before* the slot is written, not merely with a weaker memory order. A
// consumer that observes the new head may read `buffer_[head]` while the
// producer is still writing it -- reading whatever was in that slot N
// iterations ago. The deliberate delay between the two statements is not a
// realistic performance characteristic; it exists to widen the race window
// so the bug reproduces deterministically in a test run instead of relying
// on timing luck (the same reasoning as L19's explicit barrier).
template <typename T, std::size_t N>
class BrokenSpscRing
{
    static_assert((N & (N - 1)) == 0, "N must be a power of two");

public:
    bool push(T const &value) noexcept
    {
        std::size_t const head = head_.load(std::memory_order_relaxed);
        std::size_t const next = (head + 1) & (N - 1);
        if (next == tail_.load(std::memory_order_acquire))
        {
            return false;
        }
        head_.store(next, std::memory_order_release); // BUG: published before the write
        std::this_thread::sleep_for(std::chrono::microseconds(20));
        buffer_[head] = value; // a consumer may already be reading this slot
        return true;
    }

    std::optional<T> try_pop() noexcept
    {
        std::size_t const tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire))
        {
            return std::nullopt;
        }
        T value = buffer_[tail];
        tail_.store((tail + 1) & (N - 1), std::memory_order_release);
        return value;
    }

private:
    std::array<T, N> buffer_{};
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> tail_{0};
};
