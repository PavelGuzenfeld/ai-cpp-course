#include "double-buffer-swapper/swapper.hpp"
#include <fmt/core.h>
#include <memory> // std::unique_ptr

// define LargeStruct globally to make it accessible
struct LargeStruct
{
    static constexpr std::size_t SIZE_24_MB = 24 * 1024 * 1024;
    std::size_t id;
    char data[SIZE_24_MB];
};

// specialize the fmt formatter for LargeStruct
template <>
struct fmt::formatter<LargeStruct>
{
    constexpr auto parse(fmt::format_parse_context &ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const LargeStruct &, FormatContext &ctx)
    {
        return fmt::format_to(ctx.out(), "[LargeStruct: {} bytes]", sizeof(LargeStruct));
    }
};

void test_preallocated_double_buffer_int()
{
    int *active_buffer = nullptr;
    int pre_allocated_buffer = 1;
    DoubleBufferSwapper<int> buffer(&active_buffer, &pre_allocated_buffer);
    int b = 3;
    fmt::print("Initial active_buffer addr: {}.\nPre-allocated buffer addr: {} and value: {}\n",
               static_cast<void *>(active_buffer),
               static_cast<void *>(&pre_allocated_buffer), pre_allocated_buffer);

    buffer.set_active(&b);
    assert(active_buffer == &b);
    fmt::print("Set active_buffer addr: {} value: {}\n", static_cast<void *>(active_buffer), *active_buffer);

    buffer.stage(&b);
    buffer.swap();
    assert(active_buffer == &pre_allocated_buffer);
    fmt::print("Swapped active buffers: {} value: {}\n", static_cast<void *>(active_buffer), *active_buffer);
}

void test_preallocated_large_struct()
{
    LargeStruct *active_buffer = nullptr;
    std::unique_ptr<LargeStruct> pre_allocated_buffer = std::make_unique<LargeStruct>();
    pre_allocated_buffer->id = 1;
    DoubleBufferSwapper<LargeStruct> buffer(&active_buffer, pre_allocated_buffer.get());
    std::unique_ptr<LargeStruct> b = std::make_unique<LargeStruct>();
    b->id = 2;

    fmt::print("Initial active_buffer addr: {}.\nPre-allocated buffer addr: {} and id: {}\n",
               static_cast<void *>(active_buffer),
               static_cast<void *>(pre_allocated_buffer.get()), pre_allocated_buffer->id);

    buffer.set_active(b.get());
    assert(active_buffer == b.get());
    assert(active_buffer->id == 2);
    fmt::print("Set active_buffer addr: {} id: {}\n", static_cast<void *>(active_buffer), active_buffer->id);

    buffer.stage(b.get());
    buffer.swap();
    assert(active_buffer == pre_allocated_buffer.get());
    assert(active_buffer->id == 2);
    fmt::print("Swapped active buffers: {} id: {}\n", static_cast<void *>(active_buffer), active_buffer->id);
}

int main()
{
    test_preallocated_double_buffer_int();
    test_preallocated_large_struct();
    fmt::print("Test passed.\n");
    return 0;
}
