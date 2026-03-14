"""
Lesson 5: Benchmark all before/after optimization pairs.

Measures:
  - Memory with tracemalloc
  - CPU time with perf_counter_ns
  - Prints a clear table with speedup ratios
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from typing import Any, Callable

import numpy as np

from bbox_slots import BboxSlow, BboxSlots, BboxDataclass
from numpy_views import HistoryCopy, HistoryView
from preallocated_buffers import VelocityTrackerSlow, VelocityTrackerFast
from thread_pool_io import save_images_threads, save_images_pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def measure_memory(fn: Callable[[], Any]) -> tuple[Any, float, float]:
    """Run fn() and return (result, current_kb, peak_kb)."""
    tracemalloc.start()
    result = fn()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, current / 1024, peak / 1024


def measure_time_ns(fn: Callable[[], None], iterations: int) -> float:
    """Run fn() `iterations` times, return average ns per call."""
    # Warmup
    for _ in range(min(100, iterations)):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    return (time.perf_counter_ns() - t0) / iterations


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_row(label: str, time_ns: float, mem_kb: float = 0) -> None:
    if mem_kb > 0:
        print(f"  {label:35s}  {time_ns:10.0f} ns/call  {mem_kb:8.1f} KB")
    else:
        print(f"  {label:35s}  {time_ns:10.0f} ns/call")


def print_comparison(slow_ns: float, fast_ns: float) -> None:
    if fast_ns > 0:
        ratio = slow_ns / fast_ns
        print(f"  {'Speedup':35s}  {ratio:10.2f}x")
    print()


# ---------------------------------------------------------------------------
# 1. __slots__ benchmark
# ---------------------------------------------------------------------------

def bench_slots() -> tuple[float, float, float, float, float, float]:
    print_header("__slots__: BboxSlow vs BboxSlots vs BboxDataclass")

    n = 100_000

    # Memory
    _, mem_slow, _ = measure_memory(
        lambda: [BboxSlow(float(i), float(i), 10.0, 10.0) for i in range(n)]
    )
    _, mem_slots, _ = measure_memory(
        lambda: [BboxSlots(float(i), float(i), 10.0, 10.0) for i in range(n)]
    )
    _, mem_dc, _ = measure_memory(
        lambda: [BboxDataclass(float(i), float(i), 10.0, 10.0) for i in range(n)]
    )

    print(f"  Memory for {n:,} objects:")
    print(f"    BboxSlow (dict):    {mem_slow:8.1f} KB")
    print(f"    BboxSlots:          {mem_slots:8.1f} KB")
    print(f"    BboxDataclass:      {mem_dc:8.1f} KB")
    if mem_slots > 0:
        print(f"    Slots savings:      {mem_slow / mem_slots:.2f}x less memory")
    print()

    # Speed: construction + property access + iou
    iters = 100_000

    def slow_work() -> None:
        b = BboxSlow(1.0, 2.0, 10.0, 20.0)
        _ = b.cx
        _ = b.cy
        _ = b.area

    def slots_work() -> None:
        b = BboxSlots(1.0, 2.0, 10.0, 20.0)
        _ = b.cx
        _ = b.cy
        _ = b.area

    def dc_work() -> None:
        b = BboxDataclass(1.0, 2.0, 10.0, 20.0)
        _ = b.cx
        _ = b.cy
        _ = b.area

    t_slow = measure_time_ns(slow_work, iters)
    t_slots = measure_time_ns(slots_work, iters)
    t_dc = measure_time_ns(dc_work, iters)

    print(f"  Construction + property access ({iters:,} iterations):")
    print_row("BboxSlow (dict)", t_slow)
    print_row("BboxSlots", t_slots)
    print_row("BboxDataclass", t_dc)
    print_comparison(t_slow, t_slots)

    return t_slow, t_slots, t_dc, mem_slow, mem_slots, mem_dc


# ---------------------------------------------------------------------------
# 2. numpy views benchmark
# ---------------------------------------------------------------------------

def bench_views() -> tuple[float, float]:
    print_header("Numpy views vs copies: HistoryCopy vs HistoryView")

    cap = 100
    cols = 4
    iters = 10_000

    hc = HistoryCopy(capacity=cap, cols=cols)
    hv = HistoryView(capacity=cap, cols=cols)

    # Fill both buffers
    for i in range(cap):
        row = np.array([i, i + 1, i + 2, i + 3], dtype=np.float64)
        hc.push(row)
        hv.push(row)

    def copy_work() -> None:
        _ = hc.latest(20)

    def view_work() -> None:
        _ = hv.latest(20)

    t_copy = measure_time_ns(copy_work, iters)
    t_view = measure_time_ns(view_work, iters)

    print_row("HistoryCopy.latest(20)", t_copy)
    print_row("HistoryView.latest(20)", t_view)
    print_comparison(t_copy, t_view)

    return t_copy, t_view


# ---------------------------------------------------------------------------
# 3. Pre-allocated buffers benchmark
# ---------------------------------------------------------------------------

def bench_buffers() -> tuple[float, float]:
    print_header("Pre-allocated buffers: VelocityTrackerSlow vs Fast")

    rng = np.random.default_rng(42)
    positions = np.cumsum(rng.normal(3.0, 0.5, size=(50, 2)), axis=0)

    slow = VelocityTrackerSlow(threshold=2.0, ema_alpha=0.3)
    fast = VelocityTrackerFast(max_history=100, threshold=2.0, ema_alpha=0.3)

    # Verify identical results
    v_slow = slow.compute_velocity(positions)
    v_fast = fast.compute_velocity(positions)
    print(f"  Results match: {np.isclose(v_slow, v_fast)} "
          f"(slow={v_slow:.6f}, fast={v_fast:.6f})")

    iters = 10_000

    t_slow = measure_time_ns(lambda: slow.compute_velocity(positions), iters)
    t_fast = measure_time_ns(lambda: fast.compute_velocity(positions), iters)

    print_row("VelocityTrackerSlow", t_slow)
    print_row("VelocityTrackerFast", t_fast)
    print_comparison(t_slow, t_fast)

    return t_slow, t_fast


# ---------------------------------------------------------------------------
# 4. Thread pool benchmark
# ---------------------------------------------------------------------------

def bench_threads() -> tuple[float, float]:
    print_header("Thread pool: Thread-per-task vs ThreadPoolExecutor")

    n_images = 50
    items = [(f"/tmp/frame_{i:04d}.png", b"\x00" * 64) for i in range(n_images)]

    iters = 20  # thread tests are slow

    def threads_work() -> None:
        save_images_threads(items)

    def pool_work() -> None:
        save_images_pool(items, max_workers=4)

    # Use shorter simulated delays for benchmarking
    # The actual timing will be dominated by thread creation overhead
    t_threads = measure_time_ns(threads_work, iters)
    t_pool = measure_time_ns(pool_work, iters)

    print_row("Thread-per-task", t_threads)
    print_row("ThreadPool(4)", t_pool)
    print(f"  {'Peak threads (per-task)':35s}  {n_images}")
    print(f"  {'Peak threads (pool)':35s}  4")
    print()

    return t_threads, t_pool


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict[str, tuple[float, float]]) -> None:
    print_header("SUMMARY")
    print(f"  {'Optimization':35s}  {'Slow (ns)':>12s}  {'Fast (ns)':>12s}  {'Speedup':>8s}")
    print(f"  {'-' * 35}  {'-' * 12}  {'-' * 12}  {'-' * 8}")
    for name, (slow, fast) in results.items():
        ratio = slow / fast if fast > 0 else float('inf')
        print(f"  {name:35s}  {slow:12.0f}  {fast:12.0f}  {ratio:7.2f}x")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Lesson 5: Python Optimization Benchmarks")
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}")

    t_slow_s, t_slots_s, t_dc_s, *_ = bench_slots()
    t_copy_v, t_view_v = bench_views()
    t_slow_b, t_fast_b = bench_buffers()
    t_threads, t_pool = bench_threads()

    print_summary({
        "__slots__ (construct+props)": (t_slow_s, t_slots_s),
        "dataclass(slots=True)": (t_slow_s, t_dc_s),
        "numpy view vs copy": (t_copy_v, t_view_v),
        "pre-allocated buffers": (t_slow_b, t_fast_b),
        "thread pool vs thread-per-task": (t_threads, t_pool),
    })


if __name__ == "__main__":
    main()
