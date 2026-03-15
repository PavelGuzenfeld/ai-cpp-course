# Section 6 Quiz: Hardware-Level Latency and Throughput Measurement

## Q1: According to Amdahl's Law, if a function consumes 5% of total runtime, what is the maximum possible speedup from optimizing it?

- a) 5%
- b) 10%
- c) About 5.3% (1/0.95 = ~1.053x)
- d) 50%

**Answer: c)** Amdahl's Law states Speedup = 1 / ((1 - P) + P/S). With P = 0.05 and S approaching infinity, the maximum speedup is 1/0.95 = ~1.053x, or about a 5.3% improvement. This is why you should focus optimization effort on the dominant cost.

## Q2: Why should you use `time.perf_counter_ns()` instead of `time.time()` for benchmarking?

- a) `time.time()` returns the wrong timezone
- b) `time.perf_counter_ns()` is monotonic (cannot jump backward due to NTP adjustments) and returns integer nanoseconds without float precision loss
- c) `time.time()` only works on Windows
- d) `time.perf_counter_ns()` automatically subtracts system overhead

**Answer: b)** `time.time()` uses the system wall clock, which can jump backward during NTP adjustments and has variable resolution (up to ~15 ms on Windows). `perf_counter_ns()` is monotonic, high-resolution, and returns nanoseconds as integers, avoiding floating-point precision issues.

## Q3: Why do CPU timers give incorrect results when measuring GPU operations?

- a) CPU and GPU clocks run at different frequencies
- b) GPU operations are asynchronous -- the CPU returns immediately after launching a kernel while the GPU is still computing
- c) GPU operations are too fast for CPU timers to measure
- d) CPU timers cannot access GPU hardware counters

**Answer: b)** When you call a CUDA kernel or PyTorch GPU operation, the CPU enqueues the work and returns immediately. A CPU timer around the call only measures the launch overhead (microseconds), not the actual GPU computation time (potentially milliseconds).

## Q4: What is the correct way to measure GPU execution time in PyTorch?

- a) Wrap the operation with `time.perf_counter_ns()`
- b) Use `torch.cuda.Event` with `record()` before and after the operation, then `synchronize()` and call `elapsed_time()`
- c) Read the GPU temperature before and after
- d) Count the number of CUDA cores used

**Answer: b)** `torch.cuda.Event` inserts timestamps into the GPU command stream. After calling `torch.cuda.synchronize()` to ensure the GPU has finished, `start.elapsed_time(end)` returns the actual GPU execution time in milliseconds.

## Q5: Why should you run warmup iterations before collecting benchmark measurements?

- a) To give the CPU time to cool down
- b) To trigger CUDA context initialization, JIT compilation, cache filling, and lazy imports so they do not inflate the measured times
- c) To ensure the operating system allocates enough RAM
- d) Warmup is unnecessary if you use `time.perf_counter_ns()`

**Answer: b)** The first execution of any code path involves one-time costs: CUDA context setup, kernel compilation, cold instruction/data caches, and lazy module imports. Warmup iterations pay these costs before measurement begins, so the benchmark reflects steady-state performance.

## Q6: In a 3-stage pipeline where each stage takes 10 ms, what is the latency and what is the throughput?

- a) Latency = 10 ms, throughput = 100 FPS
- b) Latency = 30 ms, throughput = 100 FPS
- c) Latency = 30 ms, throughput = 33 FPS
- d) Latency = 10 ms, throughput = 33 FPS

**Answer: b)** Latency is the time a single frame spends in the pipeline from start to finish: 10 + 10 + 10 = 30 ms. Throughput is how often a completed frame exits: once every 10 ms (the pipeline rate), which is 100 FPS. Little's Law explains this: pipelining increases throughput without reducing latency.

## Q7: When reporting benchmark results, why should you use the median instead of the mean?

- a) The median is always lower than the mean
- b) The median is resistant to outliers from OS scheduling, garbage collection pauses, and other system noise that can dramatically skew the mean
- c) The mean requires more computation
- d) Medians are required by benchmarking standards

**Answer: b)** A single 100 ms GC pause among 100 measurements of ~100 microseconds would make the mean 10x higher than the typical value. The median represents the actual typical performance by ignoring outlier spikes.

## Q8: What does `perf stat -e cache-misses` measure?

- a) The number of Python function calls
- b) The number of times the CPU requested data that was not found in any cache level, requiring a fetch from RAM
- c) The amount of GPU memory used
- d) The number of disk I/O operations

**Answer: b)** `perf stat` reads hardware performance counters built into the CPU. The `cache-misses` counter tracks how many memory accesses had to go all the way to RAM because the data was not found in L1, L2, or L3 cache. This directly explains why certain access patterns are slow.
