# Section 11 Quiz: Memory Safety Without Sacrifice

## Q1: How does `std::span` compare to passing a raw pointer and size as separate parameters?

- a) `std::span` is significantly slower due to bounds checking
- b) `std::span` bundles the pointer and size together, enabling bounds checking in debug mode while producing identical assembly to raw pointer access in release mode -- zero overhead safety
- c) `std::span` copies the data into a new buffer
- d) `std::span` only works with stack-allocated arrays

**Answer: b)** `std::span` carries a pointer and a length as a single object. In debug builds, element access is bounds-checked. In release builds with `-O3`, the compiler optimizes it to bare pointer arithmetic -- the same machine code you would write by hand, but with the safety net during development.

## Q2: Why is `std::optional<BBox>` safer than returning a `BBox*` that might be null?

- a) `std::optional` is faster than pointers
- b) The type system encodes the possibility of absence, forcing the caller to check `has_value()` before accessing the value; a raw pointer does not require a null check and can be dereferenced accidentally
- c) `std::optional` uses less memory than a pointer
- d) Pointers cannot hold `BBox` values

**Answer: b)** With `std::optional`, the return type itself communicates "this might not have a value." The caller must explicitly handle the empty case via `has_value()`, `value_or()`, or pattern matching. A raw pointer allows `result->area()` without any check, leading to segfaults when the pointer is null.

## Q3: What does RAII guarantee about resource cleanup?

- a) Resources are cleaned up when the programmer explicitly calls `cleanup()`
- b) Resources are automatically released when the owning object goes out of scope, even if an exception is thrown, making resource leaks impossible
- c) Resources are cleaned up by the garbage collector
- d) Resources are cleaned up at program exit

**Answer: b)** RAII ties resource acquisition to object construction and release to destruction. Since C++ guarantees destructors run when objects leave scope (including during stack unwinding from exceptions), resources like memory, file handles, and locks cannot leak.

## Q4: What is the key semantic difference between `std::unique_ptr` and `std::shared_ptr`?

- a) `unique_ptr` is for small objects; `shared_ptr` is for large objects
- b) `unique_ptr` enforces single ownership with no copying allowed; `shared_ptr` permits multiple owners through reference counting, with the resource freed when the last owner is destroyed
- c) `unique_ptr` is faster but less safe
- d) `shared_ptr` uses the GPU; `unique_ptr` uses the CPU

**Answer: b)** `unique_ptr` models exclusive ownership -- it cannot be copied, only moved. `shared_ptr` uses an atomic reference count to track how many owners exist, and the resource is freed when the count reaches zero. The ownership semantics are encoded in the type, making intent clear.

## Q5: What category of bugs does AddressSanitizer (ASAN) detect?

- a) Only memory leaks
- b) Buffer overflows, use-after-free, double-free, memory leaks, and stack buffer overflows
- c) Only null pointer dereferences
- d) Only race conditions between threads

**Answer: b)** ASAN instruments memory operations at compile time to detect a wide range of memory safety violations: heap and stack buffer overflows, use-after-free, double-free, and memory leaks. Each detected violation includes the exact source location and allocation context.

## Q6: Why is `std::array<double, 4>` preferred over `std::vector<double>` for a bounding box with exactly 4 values?

- a) `std::array` supports more element types
- b) `std::array` is stack-allocated with zero heap overhead, while `std::vector` requires a heap allocation via `malloc`; for fixed-size data created thousands of times per frame, this eliminates significant allocation cost
- c) `std::vector` cannot hold exactly 4 elements
- d) `std::array` automatically SIMD-optimizes its operations

**Answer: b)** A bounding box always has exactly 4 values, and that count is known at compile time. `std::array` places the data directly on the stack (or inline in the containing struct), avoiding the heap allocation, pointer indirection, and deallocation overhead that `std::vector` requires.

## Q7: When building with `-DENABLE_SANITIZERS=ON`, why should you also use `-DCMAKE_BUILD_TYPE=Debug`?

- a) Sanitizers do not work in Release mode
- b) Debug mode preserves frame pointers and disables optimizations, making ASAN reports include accurate file names, line numbers, and complete stack traces
- c) Release mode automatically fixes the bugs that ASAN would find
- d) Debug mode is required for CMake to recognize the sanitizer flags

**Answer: b)** With optimizations enabled (`-O2`/`-O3`), the compiler inlines functions, reorders code, and eliminates variables, making stack traces inaccurate or incomplete. Debug mode with `-fno-omit-frame-pointer` ensures ASAN can report the exact source location of each violation.

## Q8: The lesson states that "safety and performance are not opposites" in modern C++. Which example best illustrates this?

- a) Adding runtime bounds checks to every array access
- b) `std::span` provides bounds checking in debug mode and compiles to the same assembly as raw pointer access in release mode -- the safe version is equally fast
- c) Using Python instead of C++ for safety
- d) Wrapping every function call in a try-catch block

**Answer: b)** `std::span` demonstrates that compile-time safety mechanisms can be zero-cost. The debug build catches out-of-bounds access with clear error messages, while the release build produces identical machine code to hand-written pointer arithmetic. You get full safety during development and full speed in production.
