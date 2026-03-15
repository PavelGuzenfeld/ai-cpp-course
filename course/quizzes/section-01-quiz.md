# Section 1 Quiz: SIMD Vector Operations

## Q1: Why is a pure Python for-loop slow for numerical computation?

- a) Python uses a slow addition algorithm
- b) Each iteration involves boxing/unboxing Python objects and interpreter overhead
- c) Python cannot use more than one CPU core
- d) Python lists are stored on disk, not in memory

**Answer: b)** Each element in a Python list is a full Python object (28+ bytes). Every iteration requires unboxing the integer, performing the addition, boxing the result, and appending to the output list -- all through the interpreter.

## Q2: What does `std::execution::unseq` tell the compiler?

- a) To run the loop on the GPU
- b) To skip bounds checking for faster execution
- c) That it may use SIMD instructions to process multiple elements per CPU instruction
- d) That the loop iterations must run in the original sequential order

**Answer: c)** The `unseq` execution policy permits the compiler to auto-vectorize the loop, using whatever SIMD instruction set the CPU supports (SSE, AVX2, NEON, etc.).

## Q3: A 256-bit AVX register can hold how many 32-bit integers simultaneously?

- a) 4
- b) 8
- c) 16
- d) 32

**Answer: b)** 256 bits / 32 bits per integer = 8 integers. A single `vpaddd` instruction adds all eight pairs at once.

## Q4: What is the purpose of the `-march=native` compiler flag?

- a) It enables debugging symbols for the native platform
- b) It restricts the code to only use basic x86 instructions
- c) It generates code using all instruction set extensions available on the build machine's CPU
- d) It forces the compiler to produce portable binaries

**Answer: c)** `-march=native` lets the compiler emit instructions specific to the current CPU, such as AVX2 or AVX-512, which would not be available with a generic target.

## Q5: Why does the C++ SIMD version outperform NumPy even though NumPy also calls C internally?

- a) NumPy uses an older version of C
- b) NumPy must first copy data from Python lists into contiguous arrays before operating on them
- c) NumPy does not support SIMD instructions
- d) NumPy runs in a separate process which adds IPC overhead

**Answer: b)** NumPy receives Python lists, which must be converted into contiguous C arrays before computation. The C++ SIMD version receives contiguous `std::vector` data from the start, avoiding the copy overhead.

## Q6: What role does pybind11 play in this lesson?

- a) It compiles Python code into C++
- b) It provides a way to create Python-importable shared libraries from C++ code
- c) It converts Python lists into SIMD-optimized data structures at runtime
- d) It replaces the Python interpreter with a faster one

**Answer: b)** pybind11 generates the glue code that lets Python import a compiled `.so` file and call C++ functions as if they were regular Python functions.

## Q7: What would happen if you changed `std::execution::unseq` to `std::execution::seq`?

- a) The code would no longer compile
- b) The code would produce incorrect results
- c) The code would still produce correct results but lose the SIMD speedup
- d) The code would run faster because sequential access is more cache-friendly

**Answer: c)** The `seq` policy forces strictly sequential execution. The output is identical, but the compiler is no longer permitted to vectorize, so performance drops to scalar speed.

## Q8: Which of the following is NOT a benefit of `-O3` optimization?

- a) Function inlining
- b) Automatic vectorization of loops
- c) Guaranteed absence of runtime bugs
- d) Dead code elimination

**Answer: c)** `-O3` enables aggressive compiler optimizations like inlining, vectorization, and dead code elimination, but it provides no guarantees about program correctness -- it only transforms code to run faster.
