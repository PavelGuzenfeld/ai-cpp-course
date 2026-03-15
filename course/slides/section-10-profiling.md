# Section 10: Profiling-Driven Optimization -- The Full Workflow

## Video 10.1: The Optimization Loop (~10 min)

### Slides
- Slide 1: Real optimization is not about knowing techniques -- It is about knowing WHAT to optimize and WHEN to stop. Every previous lesson taught a technique in isolation. This lesson ties them together into a repeatable workflow.
- Slide 2: The optimization loop diagram -- Profile -> Identify hotspot -> Choose technique -> Implement fix -> Measure improvement -> Repeat. The critical discipline: never skip the measurement step.
- Slide 3: Measurement methodology -- Use `perf_counter_ns`, not `time.time()`. Warmup before measuring (cold caches, lazy imports). Report median, not mean (GC pauses distort averages). Measure the same workload before and after.
- Slide 4: tracker_engine's five performance problems -- Repeated `np.eye(12)` in predict(), `.clone().cpu().numpy().tolist()` chain, per-frame cv2.resize + copyMakeBorder, redundant `.to(device)`, `torch.no_grad` instead of `inference_mode`. All correct code, all leaving performance on the table.
- Slide 5: Jetson note -- On Jetson, thermal throttling adds noise to benchmarks. Monitor junction temperature with `tegrastats` during profiling. Run benchmarks with active cooling or let the device cool between runs for consistent results.

### Key Takeaway
- The optimization workflow is always the same: profile, identify, fix, measure, repeat -- and never skip the measurement step.

## Video 10.2: Round 1 and 2 -- Profile Baseline and Fix Kalman (~12 min)

### Slides
- Slide 1: Baseline profiling results -- preprocess ~0.15 ms, inference ~0.80 ms (simulated), kalman ~0.25 ms (surprisingly expensive), postprocess ~0.10 ms. Total ~1.3 ms/frame.
- Slide 2: Why Kalman is slow -- `np.eye(self.state_dim)` creates a NEW 12x12 identity matrix every call. `np.diag(...)` creates another. Two allocations per frame. The actual matrix math is fast -- the allocations are slow.
- Slide 3: What np.eye actually does -- Calls malloc (acquire lock, search free list, possibly request OS memory), fills buffer with zeros, writes 1.0 on diagonal. For a 12x12 array, allocation overhead can exceed fill time.
- Slide 4: The fix -- Pre-allocate Q matrix in `__init__`, reuse every frame. Before: ~250 us/call. After: ~45 us/call. Speedup: 5.5x.
- Slide 5: Round 2 -- Fix the copy chain. `result_array.copy().flatten().tolist()` creates 3 copies to get 2 floats. Fix: `float(result_array[0, 0])`. Before: ~100 us. After: ~5 us. Speedup: 20x.

### Live Demo
- Run `optimization_rounds.py` showing the before/after timing for each round. Show the code diff for each fix.

### Key Takeaway
- The highest-impact optimizations are often the simplest -- pre-allocating a matrix and removing unnecessary copies gave 5.5x and 20x speedups respectively.

## Video 10.3: Rounds 3-4 and Amdahl's Law (~12 min)

### Slides
- Slide 1: Round 3 -- Pre-allocate buffers in preprocessor. Before: ~150 us (np.zeros every frame). After: ~80 us (clear pre-allocated buffer). Speedup: 1.9x. Smaller gain because computation dominates over allocation.
- Slide 2: Round 4 -- torch.inference_mode vs torch.no_grad. inference_mode disables both gradients AND version counting. Version counting requires atomic increment on every in-place operation (device sync on GPU). Typical improvement: 5-15%.
- Slide 3: Cumulative results table -- preprocess 150->80, inference 800->800 (untouched), kalman 250->45, postprocess 100->5. Total 1300->930 us. Overall speedup: 1.4x.
- Slide 4: Amdahl's Law calculation -- We achieved 20x on postprocessing but only 1.4x overall. Inference is 61.5% of total time. Maximum possible speedup even with infinite speedup on everything else: 1/(0.615) = 1.63x. The dominant cost limits the total gain.
- Slide 5: The diminishing returns curve -- First fix saved 205 us, second saved 95 us, third saved 70 us. Each successive optimization yields less. Know when to stop optimizing the easy parts and start on the dominant cost (inference: quantization, TensorRT, pruning).

### Key Takeaway
- Amdahl's Law sets an upper bound on optimization gains -- once you hit diminishing returns, the next step is optimizing the dominant cost, which is usually a different class of problem entirely.

## Video 10.4: Common Pitfalls and When to Stop (~8 min)

### Slides
- Slide 1: Pitfall 1 -- Optimizing the wrong thing. You think cv2.resize is slow because it is "known to be slow." You spend a week on a custom SIMD kernel. Then you profile and discover preprocessing was 5% of total time. Rule: always profile first.
- Slide 2: Pitfall 2 -- Benchmark methodology errors. Measuring import time, cold cache, and function time together. Using time.time() with ~1ms resolution. Single measurement with no warmup or statistical analysis.
- Slide 3: Pitfall 3 -- Changing behavior while optimizing. Your optimized version must produce identical results. If you cannot write a test that passes for both versions, you changed behavior, not just performance.
- Slide 4: Pitfall 4 -- Micro-optimizing Python when you should use C++. If a Python function is the bottleneck and you have eliminated all waste, the next step is C++ (Lessons 1-9). Pitfall 5 -- Ignoring memory. Excessive allocation causes GC pressure, cache pollution, and fragmentation.
- Slide 5: The stopping criteria -- When further optimization of non-dominant costs yields <5% overall improvement, shift focus to the dominant cost or declare "good enough." Real-time budget met? Ship it.

### Key Takeaway
- Profile first, measure after, and know when to stop -- diminishing returns are real, and optimizing the wrong thing wastes time without improving the system.
