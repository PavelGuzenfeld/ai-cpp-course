#include "double_buffer_swapper.hpp"
#include <cassert>
#include <fmt/core.h>
void test_preallocated_double_buffer()
{
    int *active_buffer = nullptr;
    int pre_allocated_buffer = 1;
    DoubleBufferSwapper<int> buffer(&active_buffer, &pre_allocated_buffer);
    int b = 3;
    fmt::print(
        "Initial active_buffer addr: {}.\nPre allocated buffer adddr {} and "
        "value {}\n",
        static_cast<void *>(active_buffer),
        static_cast<void *>(&pre_allocated_buffer), pre_allocated_buffer);

    buffer.set_active(&b);
    assert(active_buffer == &b);
    fmt::print("Set active_buffer addr: {} value: {}\n",
               static_cast<void *>(active_buffer), *active_buffer);
    buffer.stage(&b);
    buffer.swap();
    assert(active_buffer == &pre_allocated_buffer);
    fmt::print("Swapped active bufferes: {} value: {}\n",
               static_cast<void *>(active_buffer), *active_buffer);
}

int main()
{
    test_preallocated_double_buffer();
    fmt::print("Test passed.\n");
    return 0;
}