

#include "neon_copy.hpp" // NEON memcpy implementation
#include <chrono> // std::chrono
#include <iostream> // std::cout
#include <cstring> // std::memcpy

int main()
{
    constexpr size_t file_size = 24 * 1024 * 1024; // 24 MB
    uint8_t *src = new uint8_t[file_size];
    uint8_t *dst = new uint8_t[file_size];

    // Fill source with random data
    for (size_t i = 0; i < file_size; ++i)
    {
        src[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // Time NEON memcpy
    auto start = std::chrono::high_resolution_clock::now();
    neon_memcpy(dst, src, file_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto neon_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "NEON memcpy time: " << neon_duration << " microseconds\n";

    // Time std::memcpy
    start = std::chrono::high_resolution_clock::now();
    std::memcpy(dst, src, file_size);
    end = std::chrono::high_resolution_clock::now();
    auto memcpy_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "std::memcpy time: " << memcpy_duration << " microseconds\n";

    delete[] src;
    delete[] dst;

    return 0;
}
