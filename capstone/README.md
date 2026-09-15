# Capstone Project: fast_tracker_utils

## Overview

Throughout this course you have profiled, optimized, and wrapped C++ code for
Python consumption.  In this capstone you will pull those skills together and
build **`fast_tracker_utils`** — a pip-installable Python package that replaces
the four slowest parts of a pure-Python object tracker with high-performance
C++ implementations.

## What You Will Build

`fast_tracker_utils` exposes four components, each targeting a real bottleneck
found in `tracker_engine`:

| # | Component | Bottleneck it fixes | Key technique |
|---|-----------|---------------------|---------------|
| 1 | **FastKalmanFilter** | Repeated `np.eye` allocation and noise-matrix construction every predict step | Pre-allocated Eigen-style matrices; in-place updates |
| 2 | **FastPreprocessor** | Three separate NumPy passes (cast, normalize, transpose) | Fused C++ kernel — single pass over the pixel data ([Lesson 7](../ai-cpp-l7/)) |
| 3 | **FastHistoryBuffer** | `.copy()` on every `latest()` call | Circular buffer with zero-copy `numpy` views ([Lesson 4](../ai-cpp-l4/)) |
| 4 | **FastStateMachine** | `if/elif` string comparisons on every frame | `std::variant`-based state machine with compile-time dispatch ([Lesson 8](../ai-cpp-l8/)) |

## Requirements

### What is provided

* `baseline/` — a pure-Python tracker that works but is slow.  Use it as the
  functional reference and the performance "before" measurement.
* `solution/` — a reference solution you can peek at **after** attempting each
  component yourself.
* `pyproject.toml` and `CMakeLists.txt` — packaging scaffolding.

### What you must implement

1. C++ source for each of the four components listed above.
2. [nanobind](https://github.com/wjakob/nanobind) bindings that expose them to
   Python — the same pattern established in
   [Lesson 5](../ai-cpp-l5/)/[Lesson 9](../ai-cpp-l9/), four separate
   extension modules (one per component) rather than one combined module.
3. A thin Python wrapper (`solution/tracker_fast.py`'s `FastKalmanFilter`,
   `FastPreprocessor`, `FastHistoryBuffer`, `FastStateMachine` classes) that
   consumes the native modules and falls back to pure NumPy when they are not
   built, so the package is importable as:

   ```python
   from fast_tracker_utils import (
       kalman_native,
       preprocess_native,
       history_native,
       state_machine_native,
   )
   ```

4. A benchmark script that proves your implementation beats the baseline.

## Acceptance Criteria

- [ ] All four components are callable from Python.
- [ ] `pip install .` succeeds in a clean virtual environment.
- [ ] Performance benchmarks show **>2x improvement** over the baseline Python
      code for each component.
- [ ] All tests pass (`pytest baseline/test_baseline.py` and
      `pytest solution/test_fast.py`).

## Step-by-Step Guide

### Step 1 — Profile the Baseline

```bash
cd baseline
python3 tracker_baseline.py        # quick smoke test
python3 benchmark_baseline.py      # record the "before" numbers
```

Look at the per-component timing breakdown.  Identify which of the four
bottlenecks is worst on your machine.

### Step 2 — Implement Each Component

Work through the components in whatever order you prefer.  A recommended
progression:

1. **FastHistoryBuffer** — simplest C++ / pybind11 surface area.
2. **FastPreprocessor** — one fused kernel, easy to validate visually.
3. **FastKalmanFilter** — linear algebra with Eigen or raw buffers.
4. **FastStateMachine** — `std::variant` and `std::visit`.

For each component:

1. Write the C++ implementation under `src/`.
2. Add [nanobind](https://github.com/wjakob/nanobind) bindings.
3. Update `CMakeLists.txt` to build the new module.
4. Write a small Python test that compares output to the baseline.

### Step 3 — Benchmark Each Component

After implementing a component, run its benchmark in isolation to confirm the
speedup meets the >2x threshold.

### Step 4 — Package Everything

```bash
pip install .
python3 -c "from fast_tracker_utils import kalman_native; print('OK')"
```

### Step 5 — Run the Final Benchmark

```bash
python3 solution/benchmark_fast.py
```

This will print a side-by-side comparison of baseline vs fast for all four
components and the full pipeline.

`bench()` runs an untimed warm-up before the timed region and times calls in
batches of 50: CPU frequency scaling needs time to ramp up, and for an
operation costing hundreds of nanoseconds, timing one call at a time measures
`time.perf_counter()`'s own overhead as much as the code under test. Skip
either and whichever component happens to run first measures its own cold
start, not steady-state cost.

### Reference solution results (this host, in Docker)

```
KalmanFilter    baseline=  18.2 us  fast=  2.2 us  speedup= 8.3x  [PASS]
Preprocessor    baseline=2052.3 us  fast=494.1 us  speedup= 4.2x  [PASS]
HistoryBuffer   baseline=   2.5 us  fast=  0.8 us  speedup= 3.2x  [PASS]
StateMachine    baseline=   0.2 us  fast=  0.2 us  speedup= 1.2x  [FAIL]
Pipeline (full) baseline=1881.3 us  fast=479.6 us  speedup= 3.9x  [PASS]
```

Pipeline time is ~98% preprocessing at baseline, so pipeline speedup tracks
preprocessor speedup almost directly (Amdahl) — that is why fixing the
preprocessor's loop order was the highest-leverage change for the >3x
pipeline bar, more than anything done to the other three components.

**StateMachine's `[FAIL]` is expected, not a bug to chase.** The harness's own
no-op floor (`bench(lambda: None)`) measures ~0.058us; both baseline and fast
measure ~0.2us, comfortably above that floor, so this is a real measurement,
not noise. A per-call crossing of the Python/C++ boundary — argument parsing,
type checking, nanobind's exception-translation setup — costs on the order of
0.15us regardless of what the call does, and the dispatch logic this
component wraps (`std::visit` over four empty structs) is cheaper than that.
No amount of internal optimization closes that gap without changing the
API's call granularity (e.g. batching several events per call), which is out
of scope for a drop-in replacement of `StateMachine.on_event(event: str)`.
This is the pedagogically useful half of the exercise: not every hot loop
benefits from a C++ rewrite, and a component this cheap per call is a case
where the FFI crossing itself is the bottleneck.

## Grading Rubric (Self-Assessment)

| Category | Points | Criteria |
|----------|--------|----------|
| Correctness | 30 | All tests pass; output matches baseline within tolerance |
| Performance | 30 | Each component achieves >2x speedup; pipeline achieves >3x |
| Code quality | 20 | Clean C++23, proper memory management, no leaks |
| Packaging | 10 | `pip install .` works; imports succeed |
| Documentation | 10 | Brief docstrings on public API; benchmark results recorded |
| **Total** | **100** | |

### Score interpretation

* **90–100** — Excellent.  Production-ready code.
* **70–89** — Good.  Minor issues but fundamentals are solid.
* **50–69** — Needs work.  Review the relevant lessons and retry.
* **< 50** — Revisit [lessons 4](../ai-cpp-l4/)–[8](../ai-cpp-l8/) before reattempting.

---

Good luck!  When you are done, run the full test suite and benchmark one final
time to confirm everything works end-to-end.

## Lesson Files

| File | Description |
|------|-------------|
| [CMakeLists.txt](CMakeLists.txt) | CMake build configuration for capstone |
| [pyproject.toml](pyproject.toml) | Package configuration for fast_tracker_utils |
