#pragma once
template <typename T>
class DoubleBufferSwapper
{
public:
    constexpr DoubleBufferSwapper(T **active_buffer, T *pre_allocated_buffer)
        : active_buffer_(active_buffer),
          pre_allocated_buffer_(pre_allocated_buffer) {}

    constexpr inline void set_active(T *buffer) noexcept
    {
        *active_buffer_ = buffer;
    }

    constexpr inline void stage(T *buffer) noexcept
    {
        stage_ = buffer;
    }

    constexpr inline void swap() noexcept
    {
        *pre_allocated_buffer_ = *stage_;
        *active_buffer_ = pre_allocated_buffer_;
        stage_ = nullptr;
    }

private:
    T **active_buffer_;
    T *pre_allocated_buffer_;
    T *stage_ = nullptr;
};
