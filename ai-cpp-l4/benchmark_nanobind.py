"""
Benchmark: Python vs C++ (nanobind) implementations.

Compares:
  1. BoundingBox property access and IOU computation
  2. Buffer pool acquire/release cycles
  3. History latest() — copy vs view

Usage:
    python3 benchmark_nanobind.py
"""

import sys
import time
import numpy as np

# ---------------------------------------------------------------------------
# Import modules
# ---------------------------------------------------------------------------

sys.path.insert(0, ".")

from bbox_slow import BBox as BBoxPython

try:
    from bbox_native import BBox as BBoxCpp
except ImportError:
    BBoxCpp = None
    print("WARNING: bbox_native not built — skipping C++ BBox benchmarks")

try:
    from buffer_pool_native import BufferPool
except ImportError:
    BufferPool = None
    print("WARNING: buffer_pool_native not built — skipping C++ BufferPool benchmarks")

try:
    from history_view_native import HistoryView
except ImportError:
    HistoryView = None
    print("WARNING: history_view_native not built — skipping C++ HistoryView benchmarks")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_ns(ns: int) -> str:
    """Format nanoseconds into a human-readable string."""
    if ns < 1_000:
        return f"{ns} ns"
    elif ns < 1_000_000:
        return f"{ns / 1_000:.1f} us"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.1f} ms"
    else:
        return f"{ns / 1_000_000_000:.2f} s"


def print_header(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_row(label: str, py_ns: int, cpp_ns: int):
    speedup = py_ns / cpp_ns if cpp_ns > 0 else float("inf")
    print(f"  {label:<35} {fmt_ns(py_ns):>12}  {fmt_ns(cpp_ns):>12}  {speedup:>8.1f}x")


# ---------------------------------------------------------------------------
# 1. BoundingBox Benchmarks
# ---------------------------------------------------------------------------

def bench_bbox_property_access(iterations: int = 200_000):
    print_header("BoundingBox — Property Access")
    print(f"  {'Operation':<35} {'Python':>12}  {'C++ (nb)':>12}  {'Speedup':>8}")
    print(f"  {'-' * 35} {'-' * 12}  {'-' * 12}  {'-' * 8}")

    # Python
    bp = BBoxPython(10.0, 20.0, 100.0, 50.0)
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        _ = bp.cx
        _ = bp.cy
        _ = bp.area
        _ = bp.aspect_ratio
    py_prop_ns = time.perf_counter_ns() - t0

    # C++
    if BBoxCpp is None:
        return
    bc = BBoxCpp(10.0, 20.0, 100.0, 50.0)
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        _ = bc.cx
        _ = bc.cy
        _ = bc.area
        _ = bc.aspect_ratio
    cpp_prop_ns = time.perf_counter_ns() - t0

    print_row(f"4 props x {iterations} iters", py_prop_ns, cpp_prop_ns)


def bench_bbox_iou(iterations: int = 200_000):
    print_header("BoundingBox — IOU Computation")
    print(f"  {'Operation':<35} {'Python':>12}  {'C++ (nb)':>12}  {'Speedup':>8}")
    print(f"  {'-' * 35} {'-' * 12}  {'-' * 12}  {'-' * 8}")

    # Python
    a_py = BBoxPython(10, 20, 100, 50)
    b_py = BBoxPython(50, 30, 120, 60)
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        a_py.iou(b_py)
    py_ns = time.perf_counter_ns() - t0

    # C++
    if BBoxCpp is None:
        return
    a_cpp = BBoxCpp(10, 20, 100, 50)
    b_cpp = BBoxCpp(50, 30, 120, 60)
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        a_cpp.iou(b_cpp)
    cpp_ns = time.perf_counter_ns() - t0

    print_row(f"iou() x {iterations}", py_ns, cpp_ns)


def bench_bbox_bulk(num_boxes: int = 500, iterations: int = 1_000):
    print_header("BoundingBox — Bulk Operations (simulated frame)")
    print(f"  {'Operation':<35} {'Python':>12}  {'C++ (nb)':>12}  {'Speedup':>8}")
    print(f"  {'-' * 35} {'-' * 12}  {'-' * 12}  {'-' * 8}")

    # Python
    boxes_py = [BBoxPython(i * 2.0, i * 1.5, 50.0 + i, 30.0 + i) for i in range(num_boxes)]
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        for b in boxes_py:
            _ = b.area
            _ = b.cx
    py_ns = time.perf_counter_ns() - t0

    # C++
    if BBoxCpp is None:
        return
    boxes_cpp = [BBoxCpp(i * 2.0, i * 1.5, 50.0 + i, 30.0 + i) for i in range(num_boxes)]
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        for b in boxes_cpp:
            _ = b.area
            _ = b.cx
    cpp_ns = time.perf_counter_ns() - t0

    print_row(f"{num_boxes} boxes x {iterations} frames", py_ns, cpp_ns)


# ---------------------------------------------------------------------------
# 2. Buffer Pool Benchmarks
# ---------------------------------------------------------------------------

def bench_buffer_pool(iterations: int = 50_000):
    print_header("Buffer Pool — Acquire/Release Cycles")
    print(f"  {'Operation':<35} {'Python':>12}  {'C++ (nb)':>12}  {'Speedup':>8}")
    print(f"  {'-' * 35} {'-' * 12}  {'-' * 12}  {'-' * 8}")

    buf_size = 1024

    # Python baseline: plain numpy alloc/dealloc
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        buf = np.zeros(buf_size, dtype=np.float64)
        del buf
    py_ns = time.perf_counter_ns() - t0

    # C++ pool
    if BufferPool is None:
        return
    pool = BufferPool(capacity=4, buffer_size=buf_size)
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        buf = pool.acquire()
        pool.release(buf)
    cpp_ns = time.perf_counter_ns() - t0

    print_row(f"acquire/release x {iterations}", py_ns, cpp_ns)


# ---------------------------------------------------------------------------
# 3. History View Benchmarks
# ---------------------------------------------------------------------------

class HistoryPython:
    """Pure Python circular buffer that copies on latest() — mirrors tracker_engine."""

    def __init__(self, max_entries: int, row_size: int):
        self.max_entries = max_entries
        self.row_size = row_size
        self.buffer = np.zeros((max_entries, row_size), dtype=np.float64)
        self.head = 0
        self.count = 0

    def push(self, row: np.ndarray):
        self.buffer[self.head] = row
        self.head = (self.head + 1) % self.max_entries
        if self.count < self.max_entries:
            self.count += 1

    def latest(self, n: int = 1) -> np.ndarray:
        n = min(n, self.count)
        indices = [(self.head - 1 - i) % self.max_entries for i in range(n)]
        return self.buffer[indices].copy()  # copy every time!


def bench_history_latest(iterations: int = 50_000):
    print_header("History — latest() Copy vs View")
    print(f"  {'Operation':<35} {'Python':>12}  {'C++ (nb)':>12}  {'Speedup':>8}")
    print(f"  {'-' * 35} {'-' * 12}  {'-' * 12}  {'-' * 8}")

    max_entries = 100
    row_size = 8

    # Fill Python history
    hp = HistoryPython(max_entries, row_size)
    for i in range(max_entries):
        hp.push(np.full(row_size, float(i)))

    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        hp.latest(10)
    py_ns = time.perf_counter_ns() - t0

    # Fill C++ history
    if HistoryView is None:
        return
    hc = HistoryView(max_entries, row_size)
    for i in range(max_entries):
        hc.push(np.full(row_size, float(i)))

    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        hc.latest(10)
    cpp_ns = time.perf_counter_ns() - t0

    print_row(f"latest(10) x {iterations}", py_ns, cpp_ns)


def bench_history_push(iterations: int = 50_000):
    print_header("History — push() Performance")
    print(f"  {'Operation':<35} {'Python':>12}  {'C++ (nb)':>12}  {'Speedup':>8}")
    print(f"  {'-' * 35} {'-' * 12}  {'-' * 12}  {'-' * 8}")

    max_entries = 100
    row_size = 8
    row = np.ones(row_size, dtype=np.float64)

    # Python
    hp = HistoryPython(max_entries, row_size)
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        hp.push(row)
    py_ns = time.perf_counter_ns() - t0

    # C++
    if HistoryView is None:
        return
    hc = HistoryView(max_entries, row_size)
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        hc.push(row)
    cpp_ns = time.perf_counter_ns() - t0

    print_row(f"push() x {iterations}", py_ns, cpp_ns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nNanobind Benchmark Suite — Lesson 4")
    print("Comparing pure Python vs C++ (nanobind) implementations\n")

    bench_bbox_property_access()
    bench_bbox_iou()
    bench_bbox_bulk()
    bench_buffer_pool()
    bench_history_latest()
    bench_history_push()

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")
