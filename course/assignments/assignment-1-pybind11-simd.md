# Assignment 1: SIMD Vector Operations with pybind11

## Objective

Build a Python-callable C++ module that performs SIMD-accelerated vector
operations and benchmark it against pure NumPy.

## Background

In Lessons 1-2, you learned how C++ parallel execution policies and cache-aware
memory access can outperform Python. In this assignment you will put those skills
together by implementing a small linear algebra library that Python code can call
directly.

## Requirements

### Part A: Implement `fast_linalg` (50 points)

Create a pybind11 module called `fast_linalg` that exposes:

1. `dot(a, b)` — dot product of two float32 vectors
2. `normalize(a)` — return a / ||a|| (in-place, return the same buffer)
3. `weighted_sum(vectors, weights)` — compute sum(w_i * v_i) for a list of vectors

All functions must:
- Accept and return numpy arrays (use `py::array_t<float>`)
- Use `std::execution::par_unseq` where beneficial
- Handle edge cases (empty arrays, zero-norm normalization)

### Part B: Benchmark (30 points)

Write `benchmark_fast_linalg.py` that compares your C++ implementation against
NumPy for:
- Vector sizes: 100, 10,000, 1,000,000
- Report ns/call for each operation and size
- Show the crossover point where C++ beats NumPy

### Part C: Jetson Comparison (20 points, optional)

If you have access to a Jetson device:
- Run the same benchmarks on Jetson
- Compare ARM NEON auto-vectorization vs x86 SIMD
- Document any performance differences in a brief report

## Deliverables

- `src/fast_linalg.cpp` — C++ implementation
- `CMakeLists.txt` — build configuration
- `benchmark_fast_linalg.py` — benchmark script
- `test_fast_linalg.py` — at least 5 tests per function

## Grading

| Criteria | Points |
|----------|--------|
| Correctness (all tests pass) | 20 |
| Performance (>2x over NumPy for 1M elements) | 20 |
| Code quality (proper error handling, const-correct) | 10 |
| Benchmark script with clear output | 15 |
| Tests cover edge cases | 15 |
| Jetson comparison (optional) | 20 |
