# Lesson 6: Hardware-Level Latency & Throughput Measurement

Before you optimize anything, you must **measure** it. This lesson teaches you to measure
performance at the hardware level — CPU cache hierarchies, memory bandwidth, GPU async timing —
so you know exactly WHERE time is spent before changing a single line of code.

## Why Measure Before Optimizing

### Amdahl's Law

If a function takes 5% of total runtime, making it infinitely fast only saves 5%.
Amdahl's law tells us: **Speedup = 1 / ((1 - P) + P/S)** where P is the fraction
of time in the optimized section and S is the speedup factor.

If `keep_track()` in tracker_engine spends 80% of its time in inference and 10% each
in preprocess/postprocess, a 10x speedup to preprocess saves you 9% total. A 2x speedup
to inference saves you 40% total. Measure first.

### The 80/20 Rule

80% of execution time is typically spent in 20% of the code. In tracker_engine's hot loop,
the per-frame pipeline is: preprocess → inference → postprocess. You need to know which
stage dominates before optimizing any of them.

## Python Timing

### time.time() — Don't Use for Benchmarks

Wall-clock time. Resolution varies by OS (often ~1ms on Linux, ~15ms on Windows).
Subject to NTP adjustments — the clock can jump backward.

### time.perf_counter_ns() — Use This

Monotonic, high-resolution. Returns nanoseconds as an integer. No float precision loss.
Best choice for measuring code sections in Python.

```python
import time
start = time.perf_counter_ns()
do_work()
elapsed_ns = time.perf_counter_ns() - start
```

### timeit — Use for Micro-benchmarks

Runs code many times, disables GC, returns minimum time. Good for comparing two
implementations of the same function. Not suitable for profiling a pipeline.

```python
import timeit
t = timeit.timeit("sum(range(1000))", number=10000)
```

## C++ Timing

### std::chrono::steady_clock — The Default Choice

Monotonic, not affected by system clock changes. Use this for production timing.

```cpp
auto start = std::chrono::steady_clock::now();
do_work();
auto elapsed = std::chrono::steady_clock::now() - start;
auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
```

### std::chrono::high_resolution_clock

May or may not be monotonic (implementation-defined). On most Linux systems, it's
an alias for steady_clock. On some platforms it's an alias for system_clock.
Prefer steady_clock explicitly.

### RDTSC — Cycle-Level Precision

Read Time Stamp Counter. Returns CPU cycle count. Sub-nanosecond precision.
Caveats: frequency scaling, core migration, not portable. Use for micro-benchmarks
only, never for wall-clock timing.

```cpp
#include <x86intrin.h>
uint64_t start = __rdtsc();
do_work();
uint64_t cycles = __rdtsc() - start;
```

## Cache Measurement

Modern CPUs have a memory hierarchy:

| Level | Typical Size | Typical Latency |
|-------|-------------|-----------------|
| L1    | 32-64 KB    | ~1 ns (4 cycles) |
| L2    | 256-512 KB  | ~4-7 ns         |
| L3    | 4-32 MB     | ~10-20 ns       |
| RAM   | GBs         | ~60-100 ns      |

### How to Detect Cache Boundaries

1. Allocate an array of size S
2. Access every element (sequential or random)
3. Measure time per access
4. Increase S and repeat

When S exceeds L1 size, latency jumps. When it exceeds L2, another jump. This is
exactly what `cache_benchmark.cpp` does.

### Sequential vs Random Access

Sequential access benefits from hardware prefetching — the CPU predicts you'll read
the next cache line and fetches it before you ask. Random access defeats prefetching
and exposes true cache/memory latency.

### Stride Effects

Accessing every Nth byte reveals cache line structure. Stride < 64 bytes: multiple
accesses hit the same cache line. Stride = 64 bytes: one access per cache line.
Stride > 64 bytes: you skip cache lines, reducing spatial locality.

## Memory Bandwidth

Bandwidth = bytes transferred / time. Measure by reading/writing large arrays:

```cpp
size_t bytes = array_size * sizeof(int);
auto start = steady_clock::now();
for (size_t i = 0; i < array_size; ++i) sum += data[i];
auto elapsed = steady_clock::now() - start;
double gb_per_sec = bytes / elapsed_seconds / 1e9;
```

Typical values: DDR4 ~25-50 GB/s, DDR5 ~50-80 GB/s. But effective bandwidth
depends on access pattern — random access gets a fraction of peak.

## GPU Timing

### Why CPU Timers Don't Work for GPU

GPU operations are **asynchronous**. When you call a CUDA kernel or a PyTorch GPU
operation, the CPU returns immediately while the GPU is still working. A CPU timer
around a GPU call measures only the launch overhead, not the actual computation.

### torch.cuda.Event — The Correct Way

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
output = model(input_tensor)
end.record()

torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

`record()` inserts a timestamp in the GPU command stream. `elapsed_time()` returns
the time between two events in milliseconds. This measures actual GPU execution time.

### Warm-up Matters

The first GPU operation triggers CUDA context initialization, kernel compilation (for
JIT-compiled kernels), and memory allocation. Always run a few warm-up iterations
before measuring.

## Profiling Tools Overview

### [py-spy](https://github.com/benfred/py-spy) (Python)

Sampling profiler. Attaches to a running Python process without modifying code.
Shows which Python functions consume the most time. Low overhead.

```bash
py-spy top --pid <PID>
py-spy record -o profile.svg -- python my_script.py
```

### [perf stat](https://perf.wiki.kernel.org/) (Linux)

Hardware performance counters. Measures cache misses, branch mispredictions, IPC
(instructions per cycle). Tells you WHY code is slow at the hardware level.

```bash
perf stat ./my_program
# Shows: cycles, instructions, cache-misses, branch-misses
```

### [nsys](https://developer.nvidia.com/nsight-systems) (NVIDIA Nsight Systems)

GPU profiler. Shows CPU/GPU timeline, kernel launches, memory transfers, synchronization.
Essential for understanding GPU pipeline stalls.

```bash
nsys profile python my_gpu_script.py
```

## Throughput vs Latency

### Latency: ms/frame

Time from input to output for a single frame. Matters for real-time systems like
tracker_engine where every frame must be processed before the next arrives.

### Throughput: frames/sec

Total frames processed per second. Can be higher than 1/latency if you pipeline
or batch operations.

### Little's Law

**L = λ × W** — the average number of items in a system (L) equals the arrival
rate (λ) times the average time each item spends in the system (W).

For a 3-stage pipeline (preprocess → inference → postprocess):
- If each stage takes 10ms, latency = 30ms
- But throughput = 1/10ms = 100 fps (pipelined)
- Pipeline has L = 3 frames in-flight at any time

### tracker_engine: Where Does Time Go?

In `keep_track()`, the hot loop looks like:
1. **Preprocess**: crop, resize, normalize (memory-bound, cache-sensitive)
2. **Inference**: neural network forward pass (compute-bound on GPU, memory-bound on CPU)
3. **Postprocess**: decode predictions, update state (usually cheap)

Measuring each stage separately tells you where to focus optimization effort.
The tools in this lesson let you do exactly that — with nanosecond precision from C++,
exposed to Python through [nanobind](https://github.com/wjakob/nanobind).

## Build and Run

```bash
# Inside Docker container
cd /workspace
colcon build --packages-select hw_latency_measurement
source install/setup.bash

# Run the cache explorer
python3 ai-cpp-l6/cache_explorer.py

# Run the latency measurement demo
python3 ai-cpp-l6/measure_latency.py

# Run the full benchmark suite
python3 ai-cpp-l6/benchmark_measurement.py

# GPU timing (requires GPU, graceful fallback otherwise)
python3 ai-cpp-l6/gpu_timer.py
```

## Exercises

1. **Find your cache sizes**: Run `cache_explorer.py` and identify the L1, L2,
   and L3 boundaries on your machine. Compare with `lscpu | grep cache`.

2. **Stride experiment**: Modify the cache benchmark to test strides of 1, 16,
   64, 128, 256 bytes. At what stride does performance drop sharply? Why?

3. **Profile tracker_engine's hot loop**: Using the `LatencyTracker` class,
   instrument a simulated preprocess → inference → postprocess pipeline.
   Which stage dominates?

4. **GPU vs CPU timing discrepancy**: Time a PyTorch matmul using both
   `time.perf_counter_ns()` and `torch.cuda.Event`. Compare the results.
   Why are they different?

5. **Percentile analysis**: Collect 10,000 samples of a timing measurement.
   Is the distribution normal? What explains the p99 spikes?

## What You Learned

- Always measure before optimizing (Amdahl's law)
- `time.perf_counter_ns()` is the right Python timer for benchmarks
- `std::chrono::steady_clock` is the right C++ timer
- CPU timers cannot measure GPU execution (use `torch.cuda.Event`)
- Cache boundaries are detectable by measuring access latency vs array size
- Throughput and latency are different metrics — pipelines can have high
  throughput despite high latency (Little's law)

## Files in This Lesson

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Build configuration for latency_timer and cache_benchmark |
| `latency_timer.cpp` | RAII ScopedTimer with named sections, exposed via nanobind |
| `cache_benchmark.cpp` | Sequential/random/stride access benchmarks to reveal cache hierarchy |
| `measure_latency.py` | Python LatencyTracker with percentile computation and ASCII charts |
| `cache_explorer.py` | Runs cache benchmarks and visualizes the results |
| `gpu_timer.py` | torch.cuda.Event wrapper with CPU fallback |
| `benchmark_measurement.py` | Full benchmark suite comparing all measurement methods |
| `test_measurement.py` | Unit tests for timer accuracy and cache benchmark ordering |
| `test_integration_measurement.py` | Integration tests for pipeline timing and boundary detection |
