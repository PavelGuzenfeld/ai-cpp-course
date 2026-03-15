# Section 1: SIMD Vector Operations -- Your First C++ Speedup

## Video 1.1: Why Python Loops Are Slow (~10 min)

### Slides
- Slide 1: The fundamental problem -- Python processes one element at a time. Each iteration involves: fetch from iterator, unbox the int (28 bytes to 8 bytes), perform addition, box the result, append to list. For 1M elements, that is 5 million Python interpreter operations.
- Slide 2: What SIMD means -- Single Instruction, Multiple Data. A 256-bit AVX register holds eight 32-bit integers. One `vpaddd` instruction adds all eight pairs simultaneously.
- Slide 3: SIMD instruction sets by platform -- x86: SSE (128-bit), AVX2 (256-bit), AVX-512 (512-bit). ARM: NEON (128-bit). The compiler selects automatically with `-march=native`.
- Slide 4: Jetson note -- Jetson uses ARM NEON, not x86 AVX. NEON provides 128-bit SIMD (4 floats at a time vs 8 on AVX2). The `std::execution::unseq` policy works on both -- the compiler handles the difference.
- Slide 5: The benchmark comparison table -- Pure Python ~3.5s, NumPy ~0.15s (23x), C++ SIMD ~0.04s (87x) for 10 iterations of 1M element vector addition.

### Key Takeaway
- Python's per-element overhead makes loops 50-100x slower than C++ with SIMD for numerical operations.

## Video 1.2: Your First C++ Code (~12 min)

### Slides
- Slide 1: C++ vector addition code walkthrough -- `std::vector<int>`, `std::transform`, `std::execution::unseq`, lambda syntax `[](int x, int y) { return x + y; }`.
- Slide 2: Key C++ concepts for Python developers -- `std::vector<int>` is like a Python list but all same type and contiguous in memory. `auto` keyword for type inference (like Python's dynamic typing but resolved at compile time).
- Slide 3: `std::execution::unseq` explained -- This execution policy tells the compiler it can use SIMD instructions. Portable across x86 (AVX) and ARM (NEON). Requires linking with TBB (`-ltbb`).
- Slide 4: pybind11 module macro -- `PYBIND11_MODULE(name, m)` creates a Python-importable `.so` from C++. The binding code is minimal -- just `m.def("function_name", &function, "docstring")`.
- Slide 5: Lambda functions in C++ -- `[capture](params) { body }`. The `[]` is the capture list (what variables from the outer scope are available). Empty `[]` means no captures.

### Live Demo
- Walk through `portable_simd_sum_vectors.cpp` line by line, explaining each C++ construct and its Python equivalent.

### Key Takeaway
- `std::transform` with `std::execution::unseq` gives you portable SIMD without writing assembly.

## Video 1.3: Building and Benchmarking (~10 min)

### Slides
- Slide 1: CMakeLists.txt explained -- `set(CMAKE_CXX_STANDARD 23)`, `find_package(pybind11)`, `find_package(TBB)`, compiler flags `-O3 -march=native -funroll-loops -ffast-math`.
- Slide 2: What the compiler flags do -- `-O3` enables maximum optimization (inlining, vectorization, dead code elimination). `-march=native` generates code for this CPU's instruction set. `-ffast-math` allows approximate floating-point for speed.
- Slide 3: Building with colcon -- `colcon build --packages-select portable_simd_sum_vectors`, `source install/setup.bash`, `python3 ai-cpp-l1/sum.py`.
- Slide 4: Jetson build note -- On Jetson, `-march=native` targets the ARM Cortex-A78AE (Orin) or Carmel (Xavier). The compiler emits NEON instructions instead of AVX. No code changes needed -- same source, different binary.

### Live Demo
- Build the project inside Docker, run `sum.py`, show the benchmark results comparing Python, NumPy, and C++ SIMD. Vary the array size (10^3 through 10^8) and discuss when C++ SIMD pulls ahead of NumPy.

### Key Takeaway
- Compiler flags like `-O3` and `-march=native` unlock hardware-specific optimizations that are invisible in your source code.

## Video 1.4: Exercises and Exploration (~8 min)

### Slides
- Slide 1: Exercise overview -- Vary size, change types (int to float), remove SIMD (unseq to seq), add a dot product function using `std::transform_reduce`.
- Slide 2: Understanding the crossover point -- At small sizes (10^3), Python overhead is negligible and C++ function call overhead dominates. At 10^5+, C++ SIMD wins decisively. NumPy closes the gap because it calls C internally but still copies data from Python lists.
- Slide 3: Why NumPy is not as fast as raw C++ -- NumPy converts Python lists to contiguous arrays (allocation + copy), then operates on them. C++ receives contiguous data from the start (pybind11 converts `list` to `std::vector` in one pass).
- Slide 4: What you learned summary -- C++ stores data contiguously, `std::execution::unseq` enables auto-vectorization, pybind11 makes C++ callable from Python, compiler flags unlock hardware optimizations.

### Key Takeaway
- For numerical loops, C++ with SIMD provides 50-100x speedup over pure Python -- the first tool in your optimization toolbox.
