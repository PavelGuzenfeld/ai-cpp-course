"""
Benchmark the fast tracker and compare against the baseline.

Runs both implementations side-by-side and prints speedup ratios.

Usage:
    python3 benchmark_fast.py
"""

from __future__ import annotations

import statistics
import sys
import time
import tracemalloc

import numpy as np

# Add baseline directory to path so we can import it for comparison
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "baseline"))

from tracker_baseline import (  # noqa: E402
    HistoryBuffer as BaselineHistory,
    KalmanFilter as BaselineKalman,
    Pipeline as BaselinePipeline,
    Preprocessor as BaselinePreprocessor,
    StateMachine as BaselineStateMachine,
)

from tracker_fast import (  # noqa: E402
    FastHistoryBuffer,
    FastKalmanFilter,
    FastPreprocessor,
    FastStateMachine,
    Pipeline as FastPipeline,
)

NUM_FRAMES = 1000
IMAGE_SHAPE = (640, 640, 3)


def bench(func, iterations: int = NUM_FRAMES) -> list[float]:
    """Return per-call timings in seconds."""
    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        timings.append(time.perf_counter() - t0)
    return timings


def stats(timings: list[float]) -> dict[str, float]:
    """Compute mean, p50, p95, p99 in microseconds."""
    us = [t * 1e6 for t in timings]
    s = sorted(us)
    return {
        "mean": statistics.mean(us),
        "p50": statistics.median(us),
        "p95": s[int(len(s) * 0.95)],
        "p99": s[int(len(s) * 0.99)],
    }


def print_comparison(name: str, baseline_t: list[float], fast_t: list[float]) -> None:
    b = stats(baseline_t)
    f = stats(fast_t)
    speedup = b["mean"] / f["mean"] if f["mean"] > 0 else float("inf")
    status = "PASS" if speedup >= 2.0 else "FAIL"
    print(f"  {name:<25s}  baseline={b['mean']:9.1f} us  fast={f['mean']:9.1f} us  "
          f"speedup={speedup:5.1f}x  [{status}]")


def main() -> None:
    rng = np.random.default_rng(42)
    images = [rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8) for _ in range(NUM_FRAMES)]
    measurements = [np.array([100.0 + i, 200.0 + i], dtype=np.float64) for i in range(NUM_FRAMES)]
    bboxes = [np.array([100.0 + i, 200.0 + i, 150.0 + i, 250.0 + i], dtype=np.float64)
              for i in range(NUM_FRAMES)]

    print("=" * 85)
    print("Fast vs Baseline Tracker Benchmark")
    print("=" * 85)
    print(f"  Frames: {NUM_FRAMES}   Image size: {IMAGE_SHAPE}")
    print()

    # --- KalmanFilter ------------------------------------------------------
    bkf = BaselineKalman()
    fkf = FastKalmanFilter()
    idx = [0]

    def b_kalman():
        bkf.predict()
        bkf.update(measurements[idx[0] % NUM_FRAMES])
        idx[0] += 1

    idx[0] = 0
    bt_kalman = bench(b_kalman)

    idx[0] = 0

    def f_kalman():
        fkf.predict()
        fkf.update(measurements[idx[0] % NUM_FRAMES])
        idx[0] += 1

    idx[0] = 0
    ft_kalman = bench(f_kalman)

    # --- Preprocessor ------------------------------------------------------
    bprep = BaselinePreprocessor()
    fprep = FastPreprocessor()
    idx[0] = 0

    def b_prep():
        bprep.preprocess(images[idx[0] % NUM_FRAMES])
        idx[0] += 1

    idx[0] = 0
    bt_prep = bench(b_prep)
    idx[0] = 0

    def f_prep():
        fprep.preprocess(images[idx[0] % NUM_FRAMES])
        idx[0] += 1

    idx[0] = 0
    ft_prep = bench(f_prep)

    # --- HistoryBuffer -----------------------------------------------------
    bhist = BaselineHistory(capacity=100)
    fhist = FastHistoryBuffer(capacity=100)
    idx[0] = 0

    def b_hist():
        bhist.push(bboxes[idx[0] % NUM_FRAMES])
        bhist.latest(10)
        idx[0] += 1

    idx[0] = 0
    bt_hist = bench(b_hist)
    idx[0] = 0

    def f_hist():
        fhist.push(bboxes[idx[0] % NUM_FRAMES])
        fhist.latest(10)
        idx[0] += 1

    idx[0] = 0
    ft_hist = bench(f_hist)

    # --- StateMachine ------------------------------------------------------
    bsm = BaselineStateMachine()
    fsm = FastStateMachine()
    events = ["detect", "detect", "miss", "detect", "miss"]
    idx[0] = 0

    def b_sm():
        bsm.on_event(events[idx[0] % len(events)])
        idx[0] += 1

    idx[0] = 0
    bt_sm = bench(b_sm)
    idx[0] = 0

    def f_sm():
        fsm.on_event(events[idx[0] % len(events)])
        idx[0] += 1

    idx[0] = 0
    ft_sm = bench(f_sm)

    # --- Print results -----------------------------------------------------
    print("Per-component comparison:")
    print("-" * 85)
    print_comparison("KalmanFilter", bt_kalman, ft_kalman)
    print_comparison("Preprocessor", bt_prep, ft_prep)
    print_comparison("HistoryBuffer", bt_hist, ft_hist)
    print_comparison("StateMachine", bt_sm, ft_sm)

    # --- Full pipeline -----------------------------------------------------
    print()
    print("Full pipeline comparison:")
    print("-" * 85)

    bpipeline = BaselinePipeline()
    fpipeline = FastPipeline()

    tracemalloc.start()
    bt_pipe = []
    for i in range(NUM_FRAMES):
        det = bboxes[i] if i % 2 == 0 else None
        t0 = time.perf_counter()
        bpipeline.process_frame(images[i], det)
        bt_pipe.append(time.perf_counter() - t0)
    _, b_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    ft_pipe = []
    for i in range(NUM_FRAMES):
        det = bboxes[i] if i % 2 == 0 else None
        t0 = time.perf_counter()
        fpipeline.process_frame(images[i], det)
        ft_pipe.append(time.perf_counter() - t0)
    _, f_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print_comparison("Pipeline (full)", bt_pipe, ft_pipe)
    print()
    print(f"  Memory — baseline peak: {b_peak / 1024 / 1024:.1f} MB  "
          f"fast peak: {f_peak / 1024 / 1024:.1f} MB")
    print()
    print("=" * 85)


if __name__ == "__main__":
    main()
