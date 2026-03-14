"""
Benchmark the pure-Python baseline tracker.

Runs 1000 frames through the pipeline and reports per-frame latency
statistics and peak memory usage.  These numbers are the "before" that
your fast C++ implementation must beat.

Usage:
    python3 benchmark_baseline.py
"""

from __future__ import annotations

import statistics
import time
import tracemalloc

import numpy as np

from tracker_baseline import (
    HistoryBuffer,
    KalmanFilter,
    Pipeline,
    Preprocessor,
    StateMachine,
)

NUM_FRAMES = 1000
IMAGE_SHAPE = (640, 640, 3)


def benchmark_component(name: str, func, iterations: int = NUM_FRAMES) -> list[float]:
    """Run *func* for *iterations* and return per-call timings in seconds."""
    timings: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        timings.append(time.perf_counter() - t0)
    return timings


def print_stats(name: str, timings: list[float]) -> None:
    """Pretty-print timing statistics for one component."""
    timings_us = [t * 1e6 for t in timings]
    mean = statistics.mean(timings_us)
    p50 = statistics.median(timings_us)
    sorted_t = sorted(timings_us)
    p95 = sorted_t[int(len(sorted_t) * 0.95)]
    p99 = sorted_t[int(len(sorted_t) * 0.99)]
    print(f"  {name:<25s}  mean={mean:9.1f} us  "
          f"p50={p50:9.1f} us  p95={p95:9.1f} us  p99={p99:9.1f} us")


def main() -> None:
    rng = np.random.default_rng(42)

    # Pre-generate test data ------------------------------------------------
    images = [rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8)
              for _ in range(NUM_FRAMES)]
    measurements = [np.array([100.0 + i, 200.0 + i], dtype=np.float64)
                    for i in range(NUM_FRAMES)]
    bboxes = [np.array([100.0 + i, 200.0 + i, 150.0 + i, 250.0 + i],
                        dtype=np.float64) for i in range(NUM_FRAMES)]

    print("=" * 78)
    print("Baseline Tracker Benchmark")
    print("=" * 78)
    print(f"  Frames: {NUM_FRAMES}   Image size: {IMAGE_SHAPE}")
    print()

    # --- Individual component benchmarks -----------------------------------
    print("Per-component timings:")
    print("-" * 78)

    # KalmanFilter
    kf = KalmanFilter()
    idx = [0]

    def bench_kalman():
        kf.predict()
        kf.update(measurements[idx[0] % NUM_FRAMES])
        idx[0] += 1

    timings_kalman = benchmark_component("KalmanFilter", bench_kalman)
    print_stats("KalmanFilter", timings_kalman)

    # Preprocessor
    prep = Preprocessor()
    idx[0] = 0

    def bench_preprocess():
        prep.preprocess(images[idx[0] % NUM_FRAMES])
        idx[0] += 1

    timings_preprocess = benchmark_component("Preprocessor", bench_preprocess)
    print_stats("Preprocessor", timings_preprocess)

    # HistoryBuffer
    hist = HistoryBuffer(capacity=100)
    idx[0] = 0

    def bench_history():
        hist.push(bboxes[idx[0] % NUM_FRAMES])
        hist.latest(10)
        idx[0] += 1

    timings_history = benchmark_component("HistoryBuffer", bench_history)
    print_stats("HistoryBuffer", timings_history)

    # StateMachine
    sm = StateMachine()
    events = ["detect", "detect", "miss", "detect", "miss"]
    idx[0] = 0

    def bench_state():
        sm.on_event(events[idx[0] % len(events)])
        idx[0] += 1

    timings_state = benchmark_component("StateMachine", bench_state)
    print_stats("StateMachine", timings_state)

    # --- Full pipeline benchmark -------------------------------------------
    print()
    print("Full pipeline timing:")
    print("-" * 78)

    tracemalloc.start()
    pipeline = Pipeline()
    pipeline_timings: list[float] = []

    for i in range(NUM_FRAMES):
        detection = bboxes[i] if i % 2 == 0 else None
        t0 = time.perf_counter()
        pipeline.process_frame(images[i], detection)
        pipeline_timings.append(time.perf_counter() - t0)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print_stats("Pipeline (full)", pipeline_timings)
    print()
    print(f"  Peak memory: {peak_mem / 1024 / 1024:.1f} MB")
    print()
    print("=" * 78)
    print("Save these numbers — you need to beat them by at least 2x.")
    print("=" * 78)


if __name__ == "__main__":
    main()
