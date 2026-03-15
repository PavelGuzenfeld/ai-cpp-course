# Section 6: Hardware-Level Latency and Throughput Measurement

## Video 6.1: Why Measure Before Optimizing (~8 min)

### Slides
- Slide 1: Amdahl's Law -- If a function takes 5% of total runtime, making it infinitely fast only saves 5%. Speedup = 1 / ((1 - P) + P/S). Always measure first to find the dominant cost.
- Slide 2: The 80/20 rule -- 80% of execution time is typically in 20% of the code. In tracker_engine's hot loop: preprocess, inference, postprocess. You need to know which stage dominates before optimizing any of them.
- Slide 3: tracker_engine example -- If inference is 80% of keep_track() time, a 10x speedup to preprocess saves 9% total, but a 2x speedup to inference saves 40% total.
- Slide 4: The optimization loop -- Profile, identify hotspot, choose technique, implement fix, measure improvement, repeat. Never skip the measurement step.

### Key Takeaway
- Amdahl's Law means you must measure before optimizing -- the biggest speedup comes from optimizing the dominant cost, not the easiest target.

## Video 6.2: Python and C++ Timers (~10 min)

### Slides
- Slide 1: time.time() -- Wall-clock time. Variable resolution (~1ms Linux, ~15ms Windows). Subject to NTP adjustments (clock can jump backward). Do not use for benchmarks.
- Slide 2: time.perf_counter_ns() -- Monotonic, high-resolution. Returns nanoseconds as an integer. No float precision loss. The correct choice for measuring code sections in Python.
- Slide 3: timeit -- Runs code many times, disables GC, returns minimum time. Good for micro-benchmarks comparing two implementations. Not suitable for profiling a pipeline.
- Slide 4: std::chrono::steady_clock -- The C++ equivalent of perf_counter_ns. Monotonic, not affected by system clock changes. Use for production timing.
- Slide 5: RDTSC -- Read Time Stamp Counter. CPU cycle count with sub-nanosecond precision. Caveats: frequency scaling, core migration, not portable. Use for micro-benchmarks only.

### Key Takeaway
- Use time.perf_counter_ns() in Python and std::chrono::steady_clock in C++ -- these are monotonic, high-resolution, and give reliable measurements.

## Video 6.3: Cache Measurement and Memory Bandwidth (~12 min)

### Slides
- Slide 1: Detecting cache boundaries -- Allocate arrays of increasing size, measure access time per element. When size exceeds L1, latency jumps. Exceeds L2, another jump. Exceeds L3, another.
- Slide 2: Sequential vs random access -- Sequential benefits from hardware prefetching (CPU predicts next cache line). Random access defeats prefetching and exposes true cache/memory latency.
- Slide 3: Stride effects -- Stride < 64 bytes: multiple accesses per cache line. Stride = 64 bytes: one access per cache line. Stride > 64 bytes: skip cache lines, reduce spatial locality.
- Slide 4: Memory bandwidth measurement -- Bandwidth = bytes transferred / time. Typical values: DDR4 ~25-50 GB/s, DDR5 ~50-80 GB/s. Effective bandwidth depends on access pattern.
- Slide 5: Jetson memory note -- Jetson uses LPDDR5 with unified memory. CPU and GPU share the same physical RAM and bandwidth. A CPU-intensive workload can starve the GPU of memory bandwidth and vice versa. Measuring bandwidth helps you understand contention.

### Live Demo
- Run `cache_explorer.py` and identify L1, L2, L3 boundaries. Compare with `lscpu | grep cache`. Show the latency jump at each cache level boundary.

### Key Takeaway
- Cache boundaries are directly measurable -- knowing your hardware's cache sizes tells you when your working set is too large for fast access.

## Video 6.4: GPU Timing and Profiling Tools (~10 min)

### Slides
- Slide 1: Why CPU timers fail for GPU -- GPU operations are asynchronous. The CPU returns immediately while the GPU is still working. A CPU timer measures only launch overhead, not actual computation.
- Slide 2: torch.cuda.Event -- `start.record()`, run operation, `end.record()`, `torch.cuda.synchronize()`, `start.elapsed_time(end)`. Measures actual GPU execution time in milliseconds.
- Slide 3: Warm-up matters -- First GPU operation triggers CUDA context initialization, kernel compilation, memory allocation. Always run warm-up iterations before measuring.
- Slide 4: Profiling tools overview -- py-spy (Python sampling profiler, low overhead), perf stat (hardware performance counters -- cache misses, branch mispredictions, IPC), nsys (NVIDIA Nsight Systems -- CPU/GPU timeline, kernel launches, memory transfers).
- Slide 5: Throughput vs latency -- Latency = ms/frame (matters for real-time). Throughput = frames/sec (can be higher than 1/latency with pipelining). Little's Law: L = lambda * W.
- Slide 6: Jetson profiling -- Use `tegrastats` for real-time power/thermal monitoring on Jetson. nsys works on Jetson for GPU profiling. Thermal throttling can distort benchmarks -- monitor junction temperature during measurement.

### Key Takeaway
- Use torch.cuda.Event for GPU timing, not CPU timers -- and always warm up the GPU before measuring to get accurate results.
