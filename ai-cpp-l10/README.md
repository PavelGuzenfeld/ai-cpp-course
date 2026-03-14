# Lesson 10: Profiling-Driven Optimization — The Full Workflow

## Goal

Learn the complete optimization workflow: profile, identify the bottleneck, choose
a technique, implement the fix, measure the improvement, and repeat. By the end of
this lesson, you will have optimized a realistic tracker pipeline through multiple
rounds, cut its per-frame time significantly, and internalized Amdahl's Law so you
know when to stop.

## Why This Matters

Every previous lesson taught a single technique in isolation. Real optimization
doesn't work that way. In practice, you face a pipeline with dozens of stages,
unclear bottlenecks, and limited time. The skill isn't knowing *how* to optimize —
it's knowing *what* to optimize and *when to stop*.

The [tracker_engine](https://github.com/thebandofficial/tracker_engine) codebase
is a perfect case study. It contains at least five distinct performance problems,
each in a different category:

| Problem | Category | Location |
|---------|----------|----------|
| `np.eye(12)` rebuilt every `predict()` call | Redundant allocation | `KalmanFilter.predict()` |
| `.clone().cpu().numpy().tolist()` chain | Unnecessary copies | `PostProcessor` |
| Per-frame `cv2.resize` + `cv2.copyMakeBorder` | Redundant computation | `sample_target()` |
| Redundant `.to(device)` on already-placed tensors | Wasted device transfer | `os_tracker_forward()` |
| `torch.no_grad` instead of `torch.inference_mode` | Suboptimal context | Throughout |

None of these are algorithmic bugs. The code produces correct results. But
together they add milliseconds per frame — milliseconds that matter at 30 fps.

## Build and Run

```bash
# Inside Docker container or local environment
cd /workspace/ai-cpp-l10

# Run the baseline pipeline
python3 tracker_pipeline.py

# Run the optimized pipeline
python3 tracker_pipeline_optimized.py

# Run the step-by-step optimization rounds
python3 optimization_rounds.py

# Run tests
pytest test_optimization.py test_integration_optimization.py -v
```

No compiled modules are required. Everything runs with Python 3.8+ and numpy.

## The Optimization Loop

The workflow is always the same:

```
┌─────────┐     ┌──────────┐     ┌────────┐     ┌───────────┐     ┌─────────┐
│ Profile │────>│ Identify │────>│ Choose │────>│ Implement │────>│ Measure │
│         │     │ hotspot  │     │ technique│    │ fix       │     │         │
└─────────┘     └──────────┘     └────────┘     └───────────┘     └─────────┘
     ^                                                                  │
     └──────────────────────────────────────────────────────────────────┘
                              Repeat until satisfied
```

The critical discipline: **never skip the measurement step**. Every optimization
must be validated with numbers, not assumptions.

### Measurement methodology

Good benchmarks require care:

```python
import time

def measure_ns(fn, *args, warmup=5, trials=100):
    """Measure function execution time in nanoseconds."""
    # Warmup — let caches settle, JIT compile, etc.
    for _ in range(warmup):
        fn(*args)

    # Measure
    times = []
    for _ in range(trials):
        t0 = time.perf_counter_ns()
        result = fn(*args)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    times.sort()
    # Use median, not mean — outliers from OS scheduling distort means
    median = times[len(times) // 2]
    return median, result
```

Key rules:
1. **Use `perf_counter_ns`**, not `time.time()`. The latter has microsecond
   resolution on some platforms.
2. **Warmup** before measuring. The first call is always slower (cold caches,
   lazy imports, JIT warmup).
3. **Report median**, not mean. A single 100ms GC pause shouldn't make your
   "average" 10x worse.
4. **Measure the same workload** before and after. Changing inputs invalidates
   comparison.

## Round 1: Profile the Baseline

Start by timing every stage of the pipeline:

```python
# In tracker_pipeline.py, the Pipeline.process_frame() method
# times each stage individually:
#
#   preprocess:   ~0.15 ms/frame
#   inference:    ~0.80 ms/frame  (simulated)
#   kalman:       ~0.25 ms/frame  <-- surprisingly expensive
#   postprocess:  ~0.10 ms/frame
```

The Kalman filter is the first surprise. A 12-state Kalman predict step should
be a handful of matrix multiplies — sub-microsecond work. But it's taking
0.25 ms because of allocation overhead.

### What's actually happening in predict()

```python
def predict(self):
    # This creates a NEW 12x12 identity matrix every call
    Q = np.eye(self.state_dim)                    # allocation #1
    Q[self.state_dim//2:, self.state_dim//2:] *= self.process_noise

    # This creates ANOTHER matrix
    noise = np.diag(np.random.randn(self.state_dim) * 0.01)  # allocation #2

    # The actual math is fast — the allocations are slow
    self.state = self.F @ self.state
    self.P = self.F @ self.P @ self.F.T + Q
```

Two `np.eye` / `np.diag` calls per frame. Each allocates a 12x12 float64 array
(1,152 bytes), fills it, and discards it next frame. At 30 fps, that's 69,120
bytes/second of pure waste — not much memory, but the allocation+initialization
overhead dominates the actual matrix math.

## Round 2: Fix the Kalman Filter

The fix is straightforward: pre-allocate in `__init__`, reuse every frame.

**Before:**
```python
def predict(self):
    Q = np.eye(self.state_dim)  # NEW allocation every call
    Q[self.state_dim//2:, self.state_dim//2:] *= self.process_noise
    self.state = self.F @ self.state
    self.P = self.F @ self.P @ self.F.T + Q
```

**After:**
```python
def __init__(self, state_dim=12, process_noise=0.01):
    # ... existing init ...
    # Pre-allocate Q matrix ONCE
    self._Q = np.eye(state_dim)
    self._Q[state_dim//2:, state_dim//2:] *= process_noise

def predict(self):
    # Reuse pre-allocated Q — zero allocations
    self.state = self.F @ self.state
    self.P = self.F @ self.P @ self.F.T + self._Q
```

**Measured improvement:**
```
kalman predict (before):  ~250 μs/call
kalman predict (after):   ~45 μs/call
speedup:                  ~5.5x
```

The matrix math itself was always fast. We just stopped paying the allocation
tax on every frame.

### Why pre-allocation works

NumPy's `np.eye()` does three things internally:
1. Calls `malloc` for the array buffer
2. Fills the buffer with zeros
3. Writes 1.0 along the diagonal

Steps 1 and 2 are surprisingly expensive for small arrays because `malloc` must
acquire a lock, search the free list, and possibly request memory from the OS.
For a 12x12 array, the overhead of allocation can exceed the time to fill it.

Pre-allocation moves this cost to `__init__` (called once) and eliminates it
from `predict()` (called every frame).

## Round 3: Eliminate the Copy Chain

The postprocessor extracts results from computation arrays. The original code
does this:

**Before:**
```python
def extract_position(self, result_array):
    # 4 operations to get 2 floats
    values = result_array.copy()    # copy #1: defensive copy
    temp = values.flatten()         # copy #2: flatten
    coords = temp.tolist()          # copy #3: convert to Python list
    return coords[0], coords[1]    # ... just to get 2 numbers
```

This mirrors the tracker_engine pattern where `.clone().cpu().numpy().tolist()`
creates four copies to extract two float values. Each copy allocates memory,
copies data, and creates a new Python object.

**After:**
```python
def extract_position(self, result_array):
    # Direct access — zero copies
    return float(result_array[0, 0]), float(result_array[0, 1])
```

**Measured improvement:**
```
postprocess (before):  ~100 μs/call
postprocess (after):   ~5 μs/call
speedup:               ~20x
```

### When copies are actually necessary

Not all copies are waste. You need a copy when:
- The source array will be modified and you need the original values
- The source is a slice/view and you need it to outlive the parent
- You're crossing a thread boundary and the source isn't thread-safe

But for reading two floats from an array? Never.

## Round 4: Pre-allocate Buffers

The preprocessing stage allocates output buffers every frame:

**Before:**
```python
def preprocess(self, frame):
    resized = np.zeros((self.target_h, self.target_w, 3), dtype=np.float32)
    # ... fill resized ...
    padded = np.zeros((self.pad_h, self.pad_w, 3), dtype=np.float32)
    # ... fill padded ...
    return padded
```

**After:**
```python
def __init__(self, ...):
    self._resize_buf = np.zeros((self.target_h, self.target_w, 3), dtype=np.float32)
    self._pad_buf = np.zeros((self.pad_h, self.pad_w, 3), dtype=np.float32)

def preprocess(self, frame):
    self._resize_buf[:] = 0  # clear is cheaper than allocate
    # ... fill self._resize_buf ...
    self._pad_buf[:] = 0
    # ... fill self._pad_buf ...
    return self._pad_buf
```

**Measured improvement:**
```
preprocess (before):  ~150 μs/call
preprocess (after):   ~80 μs/call
speedup:              ~1.9x
```

The speedup is smaller here because the actual computation (interpolation,
padding) dominates. But every microsecond counts in a tight loop.

## Round 5: torch.inference_mode vs torch.no_grad

This lesson uses pure numpy for portability, but the concept is critical for
PyTorch pipelines. In real tracker code:

**Before:**
```python
with torch.no_grad():
    output = model(input_tensor)
```

**After:**
```python
with torch.inference_mode():
    output = model(input_tensor)
```

Why `inference_mode` is faster:
- `no_grad` only disables gradient computation but still tracks tensor versions
  (for autograd's internal bookkeeping)
- `inference_mode` disables *both* gradients and version counting
- Version counting requires an atomic increment on every in-place operation —
  on GPU, this means a device synchronization point

Typical improvement: 5-15% on inference workloads, depending on model
architecture and batch size.

## Amdahl's Law in Practice

After four rounds of optimization, here's our cumulative improvement:

```
Stage           Before    After     Speedup
─────────────── ───────── ───────── ───────
preprocess      150 μs    80 μs     1.9x
inference       800 μs    800 μs    1.0x  (untouched)
kalman          250 μs    45 μs     5.5x
postprocess     100 μs    5 μs      20x
─────────────── ───────── ───────── ───────
TOTAL           1,300 μs  930 μs    1.4x
```

We achieved a 20x speedup on postprocessing — but the overall pipeline is only
1.4x faster. Why?

**Amdahl's Law:** The maximum speedup of a system is limited by the fraction
that *cannot* be improved.

```
                    1
Speedup = ─────────────────────
          (1 - p) + p / s

Where:
  p = fraction of time spent in the optimized part
  s = speedup of that part
```

Inference takes 800 μs out of 1,300 μs — that's 61.5% of total time. Even if
we made everything else infinitely fast, the maximum possible speedup would be:

```
1 / 0.615 = 1.63x
```

This is the most important lesson: **know when to stop optimizing the easy parts
and start working on the dominant cost**. In a real tracker, the next step would
be optimizing the neural network inference itself (quantization, TensorRT,
pruning) — which is a different class of optimization entirely.

### The diminishing returns curve

```
Optimization effort ────────────────────────────>

Speedup
  ^
  │                                    ╭───── theoretical max (Amdahl)
  │                               ╭────╯
  │                          ╭────╯
  │                    ╭─────╯
  │              ╭─────╯
  │         ╭────╯
  │    ╭────╯
  │────╯
  └────────────────────────────────────────────>
```

Each successive optimization yields less overall improvement. The first fix
(Kalman) saved 205 μs. The second (copy chain) saved 95 μs. The third
(buffers) saved 70 μs. The curve is flattening.

## Common Pitfalls

### 1. Optimizing the wrong thing

The most common mistake. You *think* preprocessing is slow because cv2.resize
is "known to be slow." You spend a week writing a custom SIMD resize kernel.
Then you profile and discover preprocessing was 5% of total time.

**Rule: Always profile first. Never optimize based on intuition.**

### 2. Benchmark methodology errors

```python
# BAD: measuring import time, cold cache, and function time together
t0 = time.time()
import numpy as np
result = np.dot(a, b)
t1 = time.time()

# BAD: using time.time() — resolution is ~1ms on Windows
t0 = time.time()
fast_function()
t1 = time.time()

# BAD: single measurement — no warmup, no statistical analysis
t0 = time.perf_counter_ns()
function()
print(f"took {time.perf_counter_ns() - t0} ns")

# GOOD: proper benchmark
times = []
for _ in range(5):  # warmup
    function()
for _ in range(100):
    t0 = time.perf_counter_ns()
    function()
    times.append(time.perf_counter_ns() - t0)
print(f"median: {sorted(times)[50]} ns")
```

### 3. Changing behavior while optimizing

Your optimized version must produce *identical* results to the original. If you
can't write a test that passes for both versions, you changed the behavior,
not just the performance.

### 4. Micro-optimizing Python when you should use C++

If a Python function is the bottleneck and you've eliminated all waste, the
next step is C++ — not more Python tricks. [Lessons 1](../ai-cpp-l1/)–[9](../ai-cpp-l9/) taught you how.

### 5. Ignoring memory

CPU time isn't the only cost. Excessive allocation causes:
- GC pressure (Python's garbage collector runs more frequently)
- Cache pollution (new allocations may evict useful data from L1/L2)
- Memory fragmentation (long-running processes slow down over time)

Use `tracemalloc` to measure allocation:

```python
import tracemalloc
tracemalloc.start()

# ... run your code ...

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024:.1f} KB, Peak: {peak / 1024:.1f} KB")
tracemalloc.stop()
```

## Exercises

1. **Add a fifth optimization round.** The `Pipeline.process_frame()` method
   creates a result dictionary every frame. Pre-allocate it in `__init__` and
   update values in-place. Measure the improvement.

2. **Profile with tracemalloc.** Run `tracker_pipeline.py` with tracemalloc
   enabled and identify which line allocates the most memory. Fix it and
   verify the peak memory drops.

3. **Apply Amdahl's Law.** Given a pipeline where inference takes 70% of total
   time, preprocessing takes 20%, and postprocessing takes 10%: if you speed
   up preprocessing by 4x, what is the overall speedup? What if you speed up
   inference by 2x instead?

4. **Write a benchmark harness.** Create a function that takes two callables
   (baseline and optimized), runs both with proper warmup and trials, and
   prints a comparison table with median, p95, and min times.

5. **Find another bottleneck.** Read the tracker_engine source code on GitHub
   and identify a performance problem not covered in this lesson. Write a
   minimal reproduction and a fix.

## What You Learned

- The optimization loop: profile, identify, choose technique, implement, measure
- Pre-allocation eliminates per-frame allocation overhead
- Copy chains are a common source of waste — extract values directly
- `torch.inference_mode` is strictly better than `torch.no_grad` for inference
- Amdahl's Law sets an upper bound on optimization gains
- Always profile first, always measure after
- Know when to stop: diminishing returns are real

## Files in This Lesson

| File | Purpose |
|------|---------|
| `ai-cpp-l10.md` | This lesson guide |
| `tracker_pipeline.py` | Baseline tracker pipeline with bottlenecks |
| `tracker_pipeline_optimized.py` | Same pipeline after all optimizations |
| `optimization_rounds.py` | Step-by-step optimization with measurements |
| `test_optimization.py` | Unit tests verifying correctness |
| `test_integration_optimization.py` | Integration tests with memory checks |
