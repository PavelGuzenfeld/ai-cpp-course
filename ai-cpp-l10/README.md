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

This lesson broke its own rule 3 and paid for it. `TestTimingImprovement`
compared one timed run of the baseline against one of the optimized path, and
CI hid the result behind `--reruns 2`. Replaying the sweep sequence 100 times
reproduced it once: the pipeline comparison at ratio 1.224 against its 1.2
bound, `test_postprocessor_improvement` at 1.529 against 1.3.

The measured distributions say those are two different problems with one
shape. Over 100 replays, the pipeline ratio has a median of 1.008 — the
optimized pipeline is not actually faster end to end — so a 1.2 bound leaves
19% of headroom, and a single sample reached 1.432. The postprocessor has a
median of 0.433 and 2x headroom, but it times 10 ms of work, and one
scheduling outlier was enough.

Taking the median of five runs per side fixed both without touching either
bound: worst case over 100 replays dropped from 1.432 to 1.097 for the pipeline
and from 0.633 to 0.457 for the postprocessor. Widening the number instead
would have hidden the fact that the end-to-end optimization did not pay at all.

That 1.008 is worth more than the flake it was hiding, and it became its own
issue. The Amdahl section below is what it turned into: the optimized stages
were 2% of the frame, the pipeline comparison could not have moved, and the
test now asserts a real bound — `optimized < baseline * 0.25` — because there
is finally something to assert.

`test_kalman_improvement` was measured too and left alone — 1.74x headroom,
flat across trial counts. Measure before you change it applies to test code as
much as to the code under it.

## Round 1: Profile the Baseline

Start by timing every stage of the pipeline. `Pipeline.process_frame` in
`tracker_pipeline.py` does it inline; `optimization_rounds.py` prints the table.
200 frames at 120×160, median per stage, x86-64 laptop:

```
preprocess:   1.955 ms/frame   93%
inference:    0.104 ms/frame    5%   (simulated)
kalman:       0.047 ms/frame    2%
postprocess:  0.003 ms/frame    0%
```

**The profile has already answered the question, and the answer is
`preprocess`.** Nothing below matters more than that column of percentages.
Write it down before reading on, because the next three sections are going to
be interesting and none of them is the one that pays.

The Kalman filter is the first *surprise* — not the first target. A 12-state
predict step should be a handful of matrix multiplies, sub-microsecond work,
and it costs 47 μs because it allocates a fresh `np.eye(12)` and an
`np.diag()` every call. That is a genuine bug and worth understanding. It is
also 2% of the frame.

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
kalman predict (before):  9.9 μs/call    # predict() alone, 5000 calls, median
kalman predict (after):   6.8 μs/call
speedup:                  1.5x
```

The matrix math itself was always fast. We just stopped paying the allocation
tax on every frame — and it was a 3 μs tax on a 47 μs stage inside a 2 ms
frame. The allocation was real, the reasoning was right, and the payoff is
1.5x of 2% of the workload.

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
postprocess (before):  2.5 μs/call
postprocess (after):   1.6 μs/call
speedup:               1.6x
```

`test_postprocessor_improvement` measures 2.3x on the component in isolation,
where it runs 10 ms of back-to-back calls with no pipeline around it. In the
pipeline the stage is 0.1% of a frame either way. Both numbers are real; only
one of them is worth anything, and the Amdahl section below says which.

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
preprocess (before):  2104 μs/call
preprocess (after):   2104 μs/call
speedup:              1.00x
```

None. Two `np.zeros` of 12 KB and 19 KB are a few microseconds against a stage
that costs two milliseconds. The `# ... fill resized ...` the snippet skips over
is a Python loop running `target_h * target_w` = 4096 times per frame, and that
is the entire cost. Round 4 optimized the line it could see.

### Round 4, second pass: the loop was the cost

(`optimization_rounds.py` calls this one Round 4 too — the only round number
the two agree on, by accident.)

Nearest-neighbour resize picks one source pixel per destination pixel, and the
source indices depend only on the input shape — not on the frame. So compute
them once and let NumPy do the gather:

```python
rows = (np.arange(target_h) * (h / target_h)).astype(np.intp)
cols = (np.arange(target_w) * (w / target_w)).astype(np.intp)
self._resize_buf[:] = frame[np.minimum(rows, h - 1)[:, None], np.minimum(cols, w - 1)]
```

`.astype(np.intp)` truncates toward zero, which is what `int(row * scale)` did,
so the output is bit-identical — `test_preprocess_matches` asserts
`assert_array_equal`, not `assert_allclose`, because that is the actual claim.

```
preprocess (loop):    2104 μs/call      # the stage alone, 200 calls, median
preprocess (gather):    63 μs/call
speedup:             33.2x
```

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

The three fixes above — pre-allocate the Kalman matrices, drop the
postprocessor's copy chain, pre-allocate the preprocess buffers — are this
README's Rounds 2, 3 and 4, and `optimization_rounds.py`'s Rounds 1, 2 and 3.
The script counts only the rounds it can run; the README numbers every section.
One run of the script, on an x86-64 laptop, 200 frames at 120×160, median per
stage:

```
Stage           Baseline   Rounds 1-3   Speedup   Share of baseline
─────────────── ────────── ──────────── ───────   ─────────────────
preprocess         1955 μs      1945 μs   1.01x    93%
inference           104 μs        96 μs   1.08x     5%
kalman               47 μs        33 μs   1.44x     2%
postprocess         2.8 μs       1.5 μs   1.87x     0%
─────────────── ────────── ──────────── ───────
TOTAL              2109 μs      2076 μs   1.02x
```

Read that TOTAL with suspicion: two more runs of the same script gave 0.94x and
0.89x. Nothing touched `inference` either, and it still shows 1.08x — that is
the noise floor of a single run, and the whole baseline-vs-rounds-1-3 comparison
is inside it. Replayed 100 times the ratio has a **median of 1.008**. The
honest reading is not "1.02x" or "0.89x", it is *no change, measured badly
enough that either number is available if you want it*.

Every component that was optimized did get faster. The pipeline did not.

**Amdahl's Law:** the maximum speedup of a system is limited by the fraction
that *cannot* be improved.

```
                    1
Speedup = ─────────────────────
          (1 - p) + p / s

Where:
  p = fraction of time spent in the optimized part
  s = speedup of that part
```

Kalman and postprocess are 50 μs out of 2109 — p = 0.024. Make them infinitely
fast, s → ∞, and the ceiling is `1 / (1 - 0.024)` = **1.02x**. No measurement
was needed to know the rounds could not pay; the arithmetic was available
before the first line was written. The work was correct and the target was
wrong.

The 93% was in plain sight the whole time. `Preprocessor.preprocess` runs a
Python loop `target_h * target_w` times per frame — 4096 interpreter iterations
for a 64×64 output — and Round 4 optimized the two `np.zeros` next to it.
Replace the loop with a gather and the same table reads:

```
Stage           Baseline    Round 4   Speedup
─────────────── ────────── ────────── ───────
preprocess         1955 μs      59 μs   33.2x
inference           104 μs      95 μs    1.09x  (untouched)
kalman               47 μs      30 μs    1.56x
postprocess         2.8 μs     1.4 μs    2.00x
─────────────── ────────── ────────── ───────
TOTAL              2109 μs     185 μs   11.4x
```

This TOTAL survives replaying: 11.61x, 11.38x, 9.39x across the three runs that
gave 1.02x, 1.08x and 0.89x above, and the pipeline test's own harness measures
a median of 11.0x over 30 replays with a worst case of 8.6x. A real effect is
one you have to work to make disappear.

Now inference is 51% of the frame and it is the thing to attack next —
quantization, TensorRT, pruning — which is a different class of optimization
entirely. That is the point: **Amdahl tells you what to work on next, and it
changes every time you land something.** Re-profile after every round, because
the answer it gave you last round is now stale.

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

That is the curve you get once you are working on the dominant cost. It is not
the curve this pipeline drew. Measured, the first three fixes moved the total by
less than the run-to-run noise, and the fourth took 2109 μs to 185. There was no
flattening curve to ride — there was one stage worth 93% and three worth 2%, and
the order the rounds happened to run in had nothing to do with which was which.

Diminishing returns are real, but they arrive *after* you have taken the
dominant cost. Before that, a flat curve means you are optimizing the wrong
thing, not that you are running out of room. `optimization_rounds.py` prints
the whole table; run it and read the TOTAL column, not the per-stage one.

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

## Round 6: Should This Even Be a Learned Model?

Every round so far assumed the algorithm was right and only the code was
slow. That's not the only decision an AI developer makes. `gst-nvmm-cpp`
benchmarked a learned Kalman filter against a tuned classical IMM for
maneuvering-target tracking and found the tuned classical filter won by
~490x on position RMSE, held its lead across every distribution shift
tested, and — the finding that matters more than the ranking — the learned
filter's uncertainty output was **structurally impossible to produce**, not
just omitted, because its measurement dimension was smaller than its state
dimension. Full numbers: [`REPORT.md`](https://github.com/PavelGuzenfeld/gst-nvmm-cpp/blob/feat/kalmannet-vs-imm-benchmark/tools/kalmannet_bench/REPORT.md).

`learned_vs_classical.py` reproduces the *shape* of that finding, not the
whole benchmark: a real 2-mode IMM (constant-velocity / constant-
acceleration, with proper Markov mode-mixing — see `filter_comparison.py`)
against a `FrozenGainFilter` standing in for a learned model: a Kalman gain
fit once, offline, on a low-maneuver regime, then frozen, never adapting
online.

```bash
python3 learned_vs_classical.py
```

**RMSE isn't the interesting number here — calibration is.** A filter can
have good RMSE and still be badly miscalibrated, and RMSE alone will never
tell you. `run_calibration_demo()` runs the *same* filter twice, correct `R`
and `R` understated 10x:

| | RMSE | ANEES (band: 0.00–3.84, chi²(1) 90%) | coverage@68 | coverage@95 |
|---|---|---|---|---|
| correct R | 0.254 | 0.76 | 0.76 | 0.98 |
| R understated 10x | 0.333 | 7.54 | 0.27 | 0.51 |

RMSE moved 31%. ANEES blew through the band by 2x. Coverage@68 — nominally
0.68 — dropped to 0.27. A filter that reports its own uncertainty
confidently and wrongly is worse than one that reports none: `FrozenGainFilter`
at least never claims a covariance it can't back up.

**The log-sum-exp bug, as a debugging exercise, not new theory** (L20
already covers the general technique). The IMM's mode-probability update
needs `exp(log_likelihood)` for each mode, then normalizes. A single
heavy-tailed residual drives both raw likelihoods to exactly `0.0` in
float64, and `0.0 / 0.0` raises rather than silently propagating:

```python
imm.step(heavy_tailed_residual, log_space=False)   # FloatingPointError
imm.step(heavy_tailed_residual, log_space=True)     # stays finite
```

Log-space with max-subtraction — same trick as L20's `log_sum_exp` — fixes
it, and this is exactly how the real evaluation was re-run after the bug
surfaced: not a hypothetical, a bug that happened during the benchmark this
lesson cites.

**Latency budget, checked before training anything.** `measure_latency()`
measures the IMM's own per-step time against a stated 2 ms budget (this
course's stand-in cycle time):

| Platform | median | p99 | within 2 ms budget |
|---|---|---|---|
| x86 (Docker, no GPU) | 0.08 ms | 0.10 ms | yes |
| Orin NX / JP6 (R36.4.3) | 0.34 ms | 0.38 ms | yes |

The budget argument holds on the target hardware too, not just x86 — the
Jetson is slower per step but the margin to the 2 ms budget is still wide.
The real benchmark's cited comparison point is sharper: the reference
learned architecture at its paper-default size measured 2.858 ms
median / 4.157 ms p99 per step — over a 2 ms budget before accuracy even
entered the discussion. Check the budget first; an architecture that can't
fit doesn't need an accuracy comparison to be disqualified.

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
- Amdahl's Law sets an upper bound on optimization gains — compute it *before*
  optimizing, and re-profile after every round, because it moves
- A component speedup that does not show up end to end means you optimized
  something that was not the cost, not that the benchmark is broken
- Always profile first, always measure after
- Know when to stop: diminishing returns are real, but a flat curve before you
  have taken the dominant cost means you are working on the wrong stage
- A tuned classical baseline beats an untuned learned one — baseline properly before reaching for a model
- RMSE cannot tell a well-calibrated filter from an overconfident one; ANEES and coverage can
- A missing uncertainty output can be structural, not an oversight — check the measurement/state dimension ratio before assuming it's fixable
- Check the latency budget before training anything; an architecture that can't fit doesn't need an accuracy comparison

## Lesson Files

| File | Description |
|------|-------------|
| [tracker_pipeline.py](tracker_pipeline.py) | Baseline tracker pipeline with bottlenecks |
| [tracker_pipeline_optimized.py](tracker_pipeline_optimized.py) | Pipeline after all optimizations applied |
| [optimization_rounds.py](optimization_rounds.py) | Step-by-step optimization with measurements |
| [filter_comparison.py](filter_comparison.py) | 2-mode IMM, frozen-gain filter, mistuned-R filter, ANEES/coverage |
| [learned_vs_classical.py](learned_vs_classical.py) | Round 6: latency budget, calibration, log-sum-exp underflow |
| [test_optimization.py](test_optimization.py) | Unit tests verifying correctness |
| [test_integration_optimization.py](test_integration_optimization.py) | Integration tests with memory checks |
