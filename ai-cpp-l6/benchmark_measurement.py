"""
Benchmark measurement — comprehensive demo that runs all L6 benchmarks:
  1. Cache benchmark with visualization
  2. Latency timer on a simulated pipeline
  3. Python vs C++ timing precision comparison
  4. GPU vs CPU timing (with fallback)
"""

import time
import statistics
import sys

from measure_latency import LatencyTracker
from gpu_timer import GpuTimer, _HAS_CUDA

try:
    import cache_benchmark
    import latency_timer
    _HAS_CPP_MODULES = True
except ImportError:
    _HAS_CPP_MODULES = False
    print("WARNING: C++ modules not built. Build with CMake first.")
    print("  mkdir build && cd build && cmake .. && make\n")


def run_cache_benchmark_demo():
    """Run cache benchmark and display summary results."""
    if not _HAS_CPP_MODULES:
        print("Skipping cache benchmark (C++ module not available)")
        return

    print("=" * 72)
    print("  CACHE HIERARCHY BENCHMARK")
    print("=" * 72)
    print()

    result = cache_benchmark.run_cache_benchmark(iterations=3)

    sizes = list(result.sizes_bytes)
    seq_ns = list(result.sequential_ns_per_access)
    rand_ns = list(result.random_ns_per_access)

    print(f"{'Size':>10}  {'Sequential':>12}  {'Random':>12}  {'Ratio':>8}")
    print(f"{'':>10}  {'(ns/access)':>12}  {'(ns/access)':>12}  {'(R/S)':>8}")
    print("-" * 50)

    for i, size in enumerate(sizes):
        if size >= 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.0f} MB"
        else:
            size_str = f"{size / 1024:.0f} KB"

        ratio = rand_ns[i] / seq_ns[i] if seq_ns[i] > 0 else 0
        print(f"{size_str:>10}  {seq_ns[i]:>12.2f}  {rand_ns[i]:>12.2f}  {ratio:>8.1f}x")

    print()


def run_latency_timer_demo():
    """Run latency timer on a simulated tracking pipeline."""
    print("=" * 72)
    print("  LATENCY TIMER — SIMULATED PIPELINE")
    print("=" * 72)
    print()

    tracker = LatencyTracker()

    n_frames = 30
    print(f"Running {n_frames} frames through preprocess -> inference -> postprocess\n")

    for _ in range(n_frames):
        with tracker.section("preprocess"):
            time.sleep(0.002)  # ~2ms
        with tracker.section("inference"):
            time.sleep(0.012)  # ~12ms
        with tracker.section("postprocess"):
            time.sleep(0.001)  # ~1ms

    tracker.print_report()


def compare_timing_precision():
    """Compare Python time.perf_counter_ns vs C++ steady_clock precision."""
    print("=" * 72)
    print("  TIMING PRECISION COMPARISON")
    print("=" * 72)
    print()

    # Python timing overhead
    py_overheads = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        end = time.perf_counter_ns()
        py_overheads.append(end - start)

    py_mean = statistics.mean(py_overheads)
    py_median = statistics.median(py_overheads)
    py_min = min(py_overheads)

    print(f"  Python time.perf_counter_ns() overhead:")
    print(f"    min:    {py_min:>6} ns")
    print(f"    median: {py_median:>6.0f} ns")
    print(f"    mean:   {py_mean:>6.1f} ns")

    # C++ timing overhead
    if _HAS_CPP_MODULES:
        cpp_overheads = []
        for _ in range(1000):
            overhead = latency_timer.measure_steady_clock_ns()
            cpp_overheads.append(overhead)

        cpp_mean = statistics.mean(cpp_overheads)
        cpp_median = statistics.median(cpp_overheads)
        cpp_min = min(cpp_overheads)

        print(f"\n  C++ std::chrono::steady_clock overhead:")
        print(f"    min:    {cpp_min:>6} ns")
        print(f"    median: {cpp_median:>6.0f} ns")
        print(f"    mean:   {cpp_mean:>6.1f} ns")

        if py_median > 0 and cpp_median > 0:
            ratio = py_median / cpp_median if cpp_median > 0 else float('inf')
            print(f"\n  Python/C++ ratio: {ratio:.1f}x")
    else:
        print("\n  C++ module not available — skipping C++ comparison")

    print()


def show_gpu_vs_cpu_timing():
    """Show GPU vs CPU timing differences."""
    print("=" * 72)
    print("  GPU vs CPU TIMING")
    print("=" * 72)
    print()

    try:
        import torch
    except ImportError:
        print("  PyTorch not installed. Skipping GPU/CPU comparison.")
        print()
        return

    size = 2048

    # CPU measurement
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    _ = torch.mm(a, b)  # warm up

    cpu_times = []
    for _ in range(5):
        with GpuTimer("cpu_matmul", device="cpu") as t:
            _ = torch.mm(a, b)
        cpu_times.append(t.elapsed_ms)

    print(f"  Matrix multiply ({size}x{size}) on CPU:")
    print(f"    mean: {statistics.mean(cpu_times):.2f} ms")
    print(f"    min:  {min(cpu_times):.2f} ms")

    if _HAS_CUDA:
        a_gpu = torch.randn(size, size, device="cuda")
        b_gpu = torch.randn(size, size, device="cuda")

        # Warm up
        for _ in range(5):
            _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()

        gpu_times = []
        for _ in range(5):
            with GpuTimer("gpu_matmul", device="cuda") as t:
                _ = torch.mm(a_gpu, b_gpu)
            gpu_times.append(t.elapsed_ms)

        print(f"\n  Matrix multiply ({size}x{size}) on GPU:")
        print(f"    mean: {statistics.mean(gpu_times):.2f} ms")
        print(f"    min:  {min(gpu_times):.2f} ms")

        speedup = statistics.mean(cpu_times) / statistics.mean(gpu_times)
        print(f"\n  GPU speedup: {speedup:.1f}x")
    else:
        print("\n  No CUDA GPU available. Run on a CUDA system to see comparison.")

    print()


def main():
    print()
    print("###############################################")
    print("#  Lesson 6: Hardware-Level Measurement Demo  #")
    print("###############################################")
    print()

    run_cache_benchmark_demo()
    run_latency_timer_demo()
    compare_timing_precision()
    show_gpu_vs_cpu_timing()

    print("Done. Review the results above to understand where time is spent.")
    print("Key takeaway: MEASURE before you optimize.\n")


if __name__ == "__main__":
    main()
