# Section 10 Quiz: Profiling-Driven Optimization

## Q1: What is the correct order of the optimization loop?

- a) Implement fix, choose technique, profile, measure, identify hotspot
- b) Profile, identify the hotspot, choose a technique, implement the fix, measure the improvement
- c) Choose technique, implement fix, measure
- d) Identify hotspot, implement fix, profile

**Answer: b)** The disciplined workflow is: profile to understand current performance, identify which component is the bottleneck, select the appropriate technique, implement the fix, and then measure to confirm the improvement. Skipping the profiling step leads to optimizing the wrong thing.

## Q2: A Kalman filter's `predict()` method calls `np.eye(12)` every frame. Why is this slow even though the matrix is small?

- a) 12x12 matrices are too large for the CPU cache
- b) `np.eye()` calls `malloc` to allocate the array, fills it with zeros, then writes the diagonal -- the allocation overhead exceeds the actual computation time for a small matrix
- c) Identity matrices require complex mathematical computation
- d) NumPy does not support integer matrix sizes

**Answer: b)** For a 12x12 float64 array (1,152 bytes), `malloc` must acquire a lock, search the free list, and potentially request memory from the OS. This allocation overhead dominates the trivial work of filling 12 diagonal entries, making pre-allocation in `__init__` approximately 5.5x faster.

## Q3: After optimizing postprocessing from 100 microseconds to 5 microseconds (a 20x speedup), the overall pipeline only improved by 1.4x. Why?

- a) The measurement was incorrect
- b) Amdahl's Law: postprocessing was only ~7.7% of total runtime, so even making it infinitely fast can only improve the total by that fraction; inference at 61.5% of total time dominates
- c) The optimization introduced a new bottleneck
- d) 20x speedups are not possible in practice

**Answer: b)** Amdahl's Law limits overall speedup based on the fraction of time in the optimized component. With inference consuming 61.5% of total time, even infinitely fast everything-else yields at most 1/0.615 = 1.63x total speedup. The 20x local gain translates to a small global gain.

## Q4: Why should benchmarks report the median time rather than the mean?

- a) The median is always faster than the mean
- b) The median is robust to outlier spikes from GC pauses, OS scheduling, or interrupt handling that would disproportionately inflate the mean
- c) The mean cannot be computed for timing measurements
- d) The median requires fewer samples

**Answer: b)** A single 100 ms garbage collection pause among 99 measurements of ~100 microseconds would double the mean but leave the median unchanged. The median represents the typical execution time that users will experience.

## Q5: What is the critical difference between `torch.inference_mode()` and `torch.no_grad()` for inference workloads?

- a) They are identical in behavior
- b) `inference_mode` disables both gradient computation and tensor version counting, avoiding atomic operations that `no_grad` still performs
- c) `no_grad` is faster because it does less work
- d) `inference_mode` only works on CPU tensors

**Answer: b)** `torch.no_grad()` disables gradient computation but still tracks tensor versions for autograd's internal bookkeeping. Each version update requires an atomic increment, which can force GPU synchronization. `torch.inference_mode()` disables both, providing a 5-15% speedup on inference.

## Q6: You suspect `cv2.resize` is the bottleneck in your preprocessing pipeline, so you spend a week writing a custom SIMD resize kernel. After measuring, preprocessing was only 5% of total time. What mistake did you make?

- a) You used the wrong SIMD instructions
- b) You optimized based on intuition instead of profiling first -- the dominant cost was elsewhere
- c) OpenCV's resize is already optimal
- d) SIMD does not help with image resizing

**Answer: b)** Optimizing without profiling is the most common optimization mistake. A week of effort on a 5% component yields at most ~5% overall improvement. Profiling first would have revealed the actual bottleneck and directed effort to where it matters most.

## Q7: Why is clearing a pre-allocated buffer with `self._buf[:] = 0` faster than allocating a new one with `np.zeros(...)`?

- a) Setting values to zero is a no-op
- b) Clearing overwrites existing memory (a single `memset`), while allocation requires `malloc` (lock, free-list search, possible OS call) followed by `memset` -- the allocation overhead is eliminated
- c) `np.zeros` uses a slower algorithm
- d) Pre-allocated buffers are stored in GPU memory

**Answer: b)** `np.zeros()` must allocate a new memory block (involving `malloc` overhead) and then zero it. Clearing an existing buffer skips the allocation entirely and performs only the `memset`, which is the minimal required work.

## Q8: After four rounds of optimization, each successive round saves fewer microseconds (205, 95, 70). What does this diminishing returns pattern indicate?

- a) The optimizations are becoming less correct
- b) The remaining stages consume less of total runtime, so even large relative speedups produce smaller absolute savings -- you are approaching the Amdahl's Law ceiling set by the unoptimized dominant stage
- c) The CPU is overheating and throttling
- d) Later optimizations interfere with earlier ones

**Answer: b)** As you optimize non-dominant components, the dominant component (inference at 800 microseconds) becomes an increasingly larger fraction of total time. Amdahl's Law sets a hard ceiling: no amount of optimization to everything else can exceed 1/0.615 = 1.63x total speedup. The next high-impact step is optimizing inference itself.
