#pragma once
#include <cassert> // assert
#include <cstdint>
template <typename T>
class DoubleBufferSwapper
{
public:
    constexpr DoubleBufferSwapper(T **active_buffer, T *pre_allocated_buffer)
        : active_buffer_(active_buffer),
          pre_allocated_buffer_(pre_allocated_buffer)
    {
        assert(active_buffer_ != nullptr && "active_buffer_ is nullptr");
        assert(pre_allocated_buffer_ != nullptr && "pre_allocated_buffer_ is nullptr");
    }

    constexpr inline void set_active(T *buffer) noexcept
    {
        assert(buffer != nullptr && "buffer is nullptr");
        assert(buffer != *active_buffer_ && "buffer == *active_buffer_");
        *active_buffer_ = buffer;
    }

    constexpr inline void stage(T *buffer) noexcept
    {
        assert(buffer != nullptr && "buffer is nullptr");
        stage_ = buffer;
    }

    constexpr inline void swap() noexcept
    {
        assert(pre_allocated_buffer_ && "pre_allocated_buffer_ is null!");
        assert(stage_ && "stage_ is null!");
        assert(pre_allocated_buffer_ != stage_ && "pre_allocated_buffer_ == stage_");
        assert(reinterpret_cast<std::uintptr_t>(pre_allocated_buffer_) % alignof(T) == 0 && "unaligned pre_allocated_buffer_");
        assert(reinterpret_cast<std::uintptr_t>(stage_) % alignof(T) == 0 && "unaligned stage_");

        *pre_allocated_buffer_ = *stage_;
        *active_buffer_ = pre_allocated_buffer_;
        stage_ = nullptr;
    }

private:
    T **active_buffer_;
    T *pre_allocated_buffer_;
    T *stage_ = nullptr;
};
