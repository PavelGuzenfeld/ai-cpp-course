#pragma once
#include <arm_neon.h> // NEON intrinsics
#include <cstddef>    // size_t
#include <cstdint>    // uint8_t
#include <cstring>    // std::memcpy
#include <thread>     // std::jthread
#include <vector>     // std::vector

// NEON memcpy implementation
void neon_memcpy(void *dest, const void *src, size_t size)
{
    auto *dst = static_cast<uint8_t *>(dest);
    auto *source = static_cast<const uint8_t *>(src);

    // Copy in 128-bit chunks (16 bytes)
    while (size >= 16)
    {
        vst1q_u8(dst, vld1q_u8(source)); // Load 16 bytes from src, store to dest
        dst += 16;
        source += 16;
        size -= 16;
    }

    // Copy remaining bytes
    while (size > 0)
    {
        *dst++ = *source++;
        size--;
    }
}

constexpr void parallel_memcpy(void *dst_, const void *src_, size_t size)
{

    static auto THREADS_COUNT = std::thread::hardware_concurrency();
    static auto thread_vector = std::vector<std::jthread>(THREADS_COUNT);
    static auto chunk_size = size / THREADS_COUNT;
    auto *dst = static_cast<uint8_t *>(dst_);
    auto *src = static_cast<const uint8_t *>(src_);

    for (size_t t = 0; t < THREADS_COUNT; ++t)
    {
        thread_vector[t] = std::jthread([=]()
                                        {
            size_t start = t * chunk_size;
            size_t end = (t == THREADS_COUNT - 1) ? size : start + chunk_size;
            neon_memcpy(dst + start, src + start, end - start); });
    }
}

constexpr size_t chunk_size = 384 * 1024; // L1d cache size
void copy_image_chunked(const void *source_, void *destination_, size_t size)
{
    auto *source = static_cast<const uint8_t *>(source_);
    auto *destination = static_cast<uint8_t *>(destination_);
    for (size_t offset = 0; offset < size; offset += chunk_size)
    {
        __builtin_prefetch(&source[offset + chunk_size]);
        size_t bytes_to_copy = std::min(chunk_size, size - offset);
        std::memcpy(destination + offset, source + offset, bytes_to_copy);
    }
}
