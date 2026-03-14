# Lesson 1: SIMD Vector Operations — Your First C++ Speedup

## Goal

Take a simple operation — summing two vectors element-wise — and see how much
faster C++ with SIMD is compared to pure Python. By the end you will have built
a C++ shared library callable from Python and measured the performance gap.

## Prerequisites

- Python experience (numpy, lists, timeit)
- Docker installed (see the repo README)
- No C++ experience required — this lesson introduces the basics

## Background: Why Python Is Slow for Loops

Python's `for` loop processes one element at a time. Each iteration involves:
1. Fetch the next item from the iterator (Python object)
2. Unbox the int from a Python object (28 bytes → 8 bytes)
3. Perform the addition
4. Box the result back into a Python object
5. Append to the result list (may trigger reallocation)

For 1 million elements, that is 5 million Python interpreter operations.

**SIMD** (Single Instruction, Multiple Data) processes 4, 8, or 16 elements in a
single CPU instruction. A 256-bit AVX register holds eight 32-bit integers. One
`vpaddd` instruction adds all eight pairs simultaneously.

## The C++ Code: `portable_simd_sum_vectors.cpp`

```cpp
#include <algorithm>
#include <execution>
#include <vector>
#include <pybind11/pybind11.h>

auto portable_simd_sum_vectors(const std::vector<int>& a,
                                const std::vector<int>& b) -> std::vector<int>
{
    if (a.size() != b.size())
        throw std::runtime_error("Vectors must be the same size");

    std::vector<int> result(a.size());

    std::transform(std::execution::unseq,
                   a.begin(), a.end(),
                   b.begin(),
                   result.begin(),
                   [](int x, int y) { return x + y; });

    return result;
}

PYBIND11_MODULE(portable_simd_sum_vectors, m) {
    m.def("portable_simd_sum_vectors",
          &portable_simd_sum_vectors,
          "SIMD sum of two integer vectors");
}
```

### Key Concepts

| Concept | What It Does |
|---------|-------------|
| `std::vector<int>` | Contiguous array on the heap — like a Python list but all same type |
| `std::transform` | Applies a function to each element pair — like `map(f, a, b)` |
| `std::execution::unseq` | Tells the compiler it can use SIMD instructions |
| `PYBIND11_MODULE` | Creates a Python-importable `.so` from C++ ([pybind11](https://github.com/pybind/pybind11)) |
| `[](int x, int y) { return x + y; }` | Lambda — an inline anonymous function |

### `std::execution::unseq` — Portable SIMD

This execution policy is the key to SIMD without writing assembly:
- The compiler auto-vectorizes the loop using whatever SIMD the CPU supports
- On x86: SSE, AVX2, or AVX-512 depending on `-march=native`
- On ARM: NEON
- Requires linking with [TBB](https://github.com/oneapi-src/oneTBB) (`-ltbb`)

## The Python Benchmark: `sum.py`

```python
import timeit
import numpy as np
import portable_simd_sum_vectors

size = 10**6
a = list(range(size))
b = list(range(size))

python_time = timeit.timeit(lambda: [x + y for x, y in zip(a, b)], number=10)
numpy_time  = timeit.timeit(lambda: np.add(np.array(a), np.array(b)), number=10)
cpp_time    = timeit.timeit(lambda: portable_simd_sum_vectors.portable_simd_sum_vectors(a, b), number=10)

print(f"Python:   {python_time:.4f}s")
print(f"NumPy:    {numpy_time:.4f}s")
print(f"C++ SIMD: {cpp_time:.4f}s")
```

### Expected Results (approximate)

| Method | Time (10 iterations, 1M elements) | Speedup vs Python |
|--------|-----------------------------------|-------------------|
| Pure Python | ~3.5s | 1x |
| NumPy | ~0.15s | ~23x |
| C++ SIMD | ~0.04s | ~87x |

NumPy is fast because it calls C internally, but it still copies data from
Python lists into contiguous arrays. The C++ SIMD version receives contiguous
data from the start.

## Build and Run

Inside the Docker container:

```bash
cd /workspace

# Build with colcon
colcon build --packages-select portable_simd_sum_vectors
source install/setup.bash

# Run the benchmark
python3 ai-cpp-l1/sum.py
```

Or build with CMake directly:

```bash
cd ai-cpp-l1
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=g++-13
make -j$(nproc)

# The .so is in the build directory
cd ..
PYTHONPATH=build python3 sum.py
```

## The CMakeLists.txt Explained

```cmake
set(CMAKE_CXX_STANDARD 23)           # Use C++23
find_package(pybind11 REQUIRED)       # Find pybind11
find_package(TBB REQUIRED)           # TBB provides execution policies

add_library(${PROJECT_NAME} MODULE ${PROJECT_NAME}.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE
    -O3              # Maximum optimization
    -march=native    # Use all CPU instructions available
    -funroll-loops   # Unroll small loops
    -ffast-math      # Allow approximate math for speed
)

target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module TBB::tbb)
```

Key flags:
- `-O3`: Maximum compiler optimization (inlining, vectorization, dead code elimination)
- `-march=native`: Generate code for *this* CPU's instruction set (AVX2, etc.)
- `-ffast-math`: Allow the compiler to reorder/approximate floating-point operations

## Exercises

1. **Vary the size**: Try 10^3, 10^5, 10^7, 10^8. At what size does C++ SIMD
   pull ahead of NumPy? Why?

2. **Change the type**: Modify the C++ code to use `float` instead of `int`.
   Does the speedup change? (Hint: float SIMD uses different registers.)

3. **Remove SIMD**: Change `std::execution::unseq` to `std::execution::seq`.
   How much slower is the sequential version?

4. **Add a pybind11 binding**: Add a `dot_product` function that computes the
   dot product of two vectors using `std::transform_reduce`. Benchmark it.

## What You Learned

- C++ stores data contiguously in memory (cache-friendly)
- `std::execution::unseq` enables automatic SIMD vectorization
- pybind11 makes C++ callable from Python with minimal glue code
- Compiler flags (`-O3`, `-march=native`) unlock hardware-specific optimizations
- For numerical loops, C++ can be 50-100x faster than pure Python

## Lesson Files

| File | Description |
|------|-------------|
| [portable_simd_sum_vectors.cpp](portable_simd_sum_vectors.cpp) | SIMD vector addition with pybind11 bindings |
| [sum.py](sum.py) | Python benchmark comparing all approaches |
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration |
| [setup.md](setup.md) | Environment setup instructions |
