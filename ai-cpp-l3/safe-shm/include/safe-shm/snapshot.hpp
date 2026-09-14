#pragma once
#include "safe-shm/flat_type.hpp"
#include <atomic>
#include <cassert>

namespace safe_shm
{
    template <FlatType T>
    class Snapshot
    {
    public:
        explicit Snapshot(std::atomic<T *> *ptr) noexcept : ptr_(ptr) {}

        T const &operator*() const noexcept
        {
            auto *p = ptr_->load(std::memory_order_acquire);
            assert(p && "snapshot data is null");
            return *p;
        }

        T const *operator->() const noexcept
        {
            auto *p = ptr_->load(std::memory_order_acquire);
            assert(p && "snapshot data is null");
            return p;
        }

        T *get() const noexcept
        {
            return ptr_->load(std::memory_order_acquire);
        }

    private:
        std::atomic<T *> *ptr_;
    };
} // namespace safe_shm
