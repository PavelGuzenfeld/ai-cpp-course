#pragma once

#include <cstddef>

// Configuration — override via compiler defines (-D)
#ifndef EXCEPTION_RT_ABI_HEADER_SIZE
#define EXCEPTION_RT_ABI_HEADER_SIZE 128 // sizeof(__cxa_refcounted_exception) on x86_64 GCC
#endif

#ifndef EXCEPTION_RT_MAX_THROWN_SIZE
#define EXCEPTION_RT_MAX_THROWN_SIZE 256
#endif

#ifndef EXCEPTION_RT_NUM_SLOTS
#define EXCEPTION_RT_NUM_SLOTS 8
#endif

// Weak-symbol overrides (linked by the C++ runtime)
extern "C"
{
    void *__cxa_allocate_exception(std::size_t thrown_size) noexcept;
    void __cxa_free_exception(void *thrown_object) noexcept;
}

// Introspection API for monitoring and testing
namespace exception_rt
{
    [[nodiscard]] std::size_t slots_in_use() noexcept;
    [[nodiscard]] constexpr std::size_t num_slots() noexcept { return EXCEPTION_RT_NUM_SLOTS; }
    [[nodiscard]] constexpr std::size_t max_thrown_size() noexcept { return EXCEPTION_RT_MAX_THROWN_SIZE; }
    [[nodiscard]] constexpr std::size_t abi_header_size() noexcept { return EXCEPTION_RT_ABI_HEADER_SIZE; }
    [[nodiscard]] bool owns(const void *ptr) noexcept;
} // namespace exception_rt
