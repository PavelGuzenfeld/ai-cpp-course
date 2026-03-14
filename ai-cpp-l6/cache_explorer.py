"""
Cache hierarchy explorer — uses C++ cache_benchmark module to
measure and visualize memory access latency across array sizes.
"""

import math
import sys

try:
    import cache_benchmark
except ImportError:
    print("cache_benchmark module not found. Build with CMake first.")
    print("  mkdir build && cd build && cmake .. && make")
    sys.exit(1)


def format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.0f}MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f}KB"
    else:
        return f"{size_bytes}B"


def detect_boundaries(sizes: list[int], latencies: list[float],
                      threshold: float = 1.5) -> list[dict]:
    """Detect cache level boundaries by finding jumps in latency.

    A boundary is detected when the latency ratio between consecutive
    sizes exceeds the threshold.
    """
    boundaries = []
    for i in range(1, len(latencies)):
        if latencies[i - 1] > 0:
            ratio = latencies[i] / latencies[i - 1]
            if ratio > threshold:
                boundaries.append({
                    "index": i,
                    "size_before": sizes[i - 1],
                    "size_after": sizes[i],
                    "latency_before": latencies[i - 1],
                    "latency_after": latencies[i],
                    "ratio": ratio,
                })
    return boundaries


def text_plot(sizes: list[int], seq_ns: list[float], rand_ns: list[float],
              boundaries: list[dict]):
    """Print a text-based plot of access latency vs array size."""
    max_latency = max(max(rand_ns), max(seq_ns))
    chart_width = 50
    label_width = 8

    print()
    print("=" * 72)
    print("  CACHE LATENCY vs ARRAY SIZE")
    print("=" * 72)
    print()
    print(f"{'Size':>{label_width}}  {'Sequential':>10}  {'Random':>10}  Chart (random)")
    print(f"{'':>{label_width}}  {'(ns/acc)':>10}  {'(ns/acc)':>10}")
    print("-" * 72)

    boundary_sizes = {b["size_after"] for b in boundaries}

    for i, size in enumerate(sizes):
        size_str = format_size(size)
        seq_val = seq_ns[i]
        rand_val = rand_ns[i]

        # Scale bar to chart width using log scale for better visualization
        if max_latency > 0 and rand_val > 0:
            log_ratio = math.log2(rand_val) / math.log2(max_latency) if max_latency > 1 else 0
            bar_len = int(log_ratio * chart_width)
        else:
            bar_len = 0
        bar_len = max(1, min(bar_len, chart_width))

        bar = "█" * bar_len

        # Mark boundaries
        marker = " ◄" if size in boundary_sizes else ""

        print(f"{size_str:>{label_width}}  {seq_val:>10.2f}  {rand_val:>10.2f}  {bar}{marker}")

    print()


def annotate_boundaries(boundaries: list[dict]):
    """Print detected cache level boundaries."""
    if not boundaries:
        print("  No clear cache boundaries detected.")
        return

    cache_names = ["L1 → L2", "L2 → L3", "L3 → RAM"]

    print("  DETECTED CACHE BOUNDARIES:")
    print("  " + "-" * 50)

    for i, b in enumerate(boundaries):
        name = cache_names[i] if i < len(cache_names) else f"Level {i + 1} → Level {i + 2}"
        print(f"  {name}: between {format_size(b['size_before'])} and {format_size(b['size_after'])}")
        print(f"    Latency jump: {b['latency_before']:.1f} ns → {b['latency_after']:.1f} ns "
              f"({b['ratio']:.1f}x)")

    print()


def text_plot_stride(strides: list[int], ns_per_access: list[float]):
    """Print text-based plot of stride effect on access latency."""
    max_val = max(ns_per_access) if ns_per_access else 1
    chart_width = 40

    print()
    print("=" * 72)
    print("  STRIDE EFFECT ON CACHE UTILIZATION")
    print("=" * 72)
    print()
    print(f"{'Stride':>10}  {'ns/access':>10}  Chart")
    print("-" * 60)

    for i, stride in enumerate(strides):
        val = ns_per_access[i]
        bar_len = int((val / max_val) * chart_width) if max_val > 0 else 0
        bar_len = max(1, min(bar_len, chart_width))
        bar = "█" * bar_len

        stride_str = format_size(stride)
        print(f"{stride_str:>10}  {val:>10.2f}  {bar}")

    print()
    print("  Note: At stride >= 64 bytes (1 cache line), each access hits a")
    print("  different cache line. Larger strides waste more cache capacity.")
    print()


def main():
    print("Running cache hierarchy benchmark...")
    print("This may take a minute for large array sizes.\n")

    # Run the main cache benchmark
    result = cache_benchmark.run_cache_benchmark(iterations=5)

    sizes = list(result.sizes_bytes)
    seq_ns = list(result.sequential_ns_per_access)
    rand_ns = list(result.random_ns_per_access)

    # Detect boundaries using random access latencies
    boundaries = detect_boundaries(sizes, rand_ns, threshold=1.5)

    text_plot(sizes, seq_ns, rand_ns, boundaries)
    annotate_boundaries(boundaries)

    # Run stride benchmark
    print("Running stride benchmark...")
    stride_result = cache_benchmark.run_stride_benchmark(
        array_size_bytes=64 * 1024 * 1024, iterations=3
    )

    strides = list(stride_result.strides)
    stride_ns = list(stride_result.ns_per_access)

    text_plot_stride(strides, stride_ns)


if __name__ == "__main__":
    main()
