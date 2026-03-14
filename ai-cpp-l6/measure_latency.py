"""
Latency measurement utilities — Python wrapper for C++ ScopedTimer
with histogram/percentile computation and ASCII visualization.
"""

import time
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field


class LatencyTracker:
    """Tracks latency measurements for named sections.

    Can be used as a context manager:
        tracker = LatencyTracker()
        with tracker.section("preprocess"):
            do_preprocess()
    """

    def __init__(self):
        self._timings: dict[str, list[float]] = {}  # section -> list of ns

    def section(self, name: str):
        """Return a context manager that times a named section."""
        return _SectionTimer(self, name)

    def record(self, name: str, elapsed_ns: float):
        """Manually record a timing measurement."""
        if name not in self._timings:
            self._timings[name] = []
        self._timings[name].append(elapsed_ns)

    @property
    def sections(self) -> list[str]:
        """Return list of all section names."""
        return list(self._timings.keys())

    def count(self, name: str) -> int:
        """Return number of samples for a section."""
        return len(self._timings.get(name, []))

    def total_count(self) -> int:
        """Return total number of samples across all sections."""
        return sum(len(v) for v in self._timings.values())

    def timings_ns(self, name: str) -> list[float]:
        """Return raw timing data in nanoseconds."""
        return list(self._timings.get(name, []))

    def timings_ms(self, name: str) -> list[float]:
        """Return timing data in milliseconds."""
        return [t / 1e6 for t in self._timings.get(name, [])]

    def percentile(self, name: str, p: float) -> float:
        """Compute the p-th percentile (0-100) in nanoseconds."""
        data = self._timings.get(name, [])
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = (p / 100.0) * (len(sorted_data) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(sorted_data) - 1)
        frac = idx - lower
        return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac

    def p50(self, name: str) -> float:
        """Median latency in nanoseconds."""
        return self.percentile(name, 50)

    def p95(self, name: str) -> float:
        """95th percentile latency in nanoseconds."""
        return self.percentile(name, 95)

    def p99(self, name: str) -> float:
        """99th percentile latency in nanoseconds."""
        return self.percentile(name, 99)

    def mean(self, name: str) -> float:
        """Mean latency in nanoseconds."""
        data = self._timings.get(name, [])
        return statistics.mean(data) if data else 0.0

    def stddev(self, name: str) -> float:
        """Standard deviation in nanoseconds."""
        data = self._timings.get(name, [])
        return statistics.stdev(data) if len(data) > 1 else 0.0

    def reset(self):
        """Clear all recorded timings."""
        self._timings.clear()

    def summary(self, name: str) -> dict:
        """Return a summary dict for a section."""
        return {
            "name": name,
            "count": self.count(name),
            "mean_ms": self.mean(name) / 1e6,
            "p50_ms": self.p50(name) / 1e6,
            "p95_ms": self.p95(name) / 1e6,
            "p99_ms": self.p99(name) / 1e6,
            "stddev_ms": self.stddev(name) / 1e6,
        }

    def print_report(self):
        """Print a formatted report of all sections with ASCII bar chart."""
        if not self._timings:
            print("No timing data recorded.")
            return

        print()
        print("=" * 72)
        print("  LATENCY REPORT")
        print("=" * 72)

        # Find the maximum p95 for scaling the bar chart
        max_val = max(self.p95(name) for name in self._timings)
        if max_val == 0:
            max_val = 1

        bar_width = 30

        for name in self._timings:
            s = self.summary(name)
            print(f"\n  [{name}]  ({s['count']} samples)")
            print(f"    mean: {s['mean_ms']:8.3f} ms")
            print(f"     p50: {s['p50_ms']:8.3f} ms")
            print(f"     p95: {s['p95_ms']:8.3f} ms")
            print(f"     p99: {s['p99_ms']:8.3f} ms")
            print(f"  stddev: {s['stddev_ms']:8.3f} ms")

            # ASCII bar chart showing p50 and p95
            p50_len = int((self.p50(name) / max_val) * bar_width)
            p95_len = int((self.p95(name) / max_val) * bar_width)
            bar = "█" * p50_len + "▓" * (p95_len - p50_len) + "░" * (bar_width - p95_len)
            print(f"          [{bar}]")

        print()
        print("=" * 72)
        print("  █ = p50   ▓ = p50-p95   ░ = remaining")
        print("=" * 72)
        print()


class _SectionTimer:
    """Context manager that times a section and records it in a LatencyTracker."""

    def __init__(self, tracker: LatencyTracker, name: str):
        self._tracker = tracker
        self._name = name
        self._start = 0

    def __enter__(self):
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter_ns() - self._start
        self._tracker.record(self._name, elapsed)
        return False


def demo_tracking_pipeline():
    """Demo: time a simulated tracking pipeline."""
    tracker = LatencyTracker()

    print("Simulating tracker_engine pipeline (20 frames)...")
    print("Stages: preprocess -> inference -> postprocess\n")

    for frame in range(20):
        with tracker.section("preprocess"):
            # Simulate crop + resize + normalize
            time.sleep(0.002)  # ~2ms

        with tracker.section("inference"):
            # Simulate neural network forward pass
            time.sleep(0.015)  # ~15ms

        with tracker.section("postprocess"):
            # Simulate decode + state update
            time.sleep(0.001)  # ~1ms

    tracker.print_report()

    # Show per-frame total
    total_per_frame = []
    for i in range(20):
        t = (
            tracker.timings_ns("preprocess")[i]
            + tracker.timings_ns("inference")[i]
            + tracker.timings_ns("postprocess")[i]
        )
        total_per_frame.append(t / 1e6)  # ms

    avg_total = statistics.mean(total_per_frame)
    print(f"  Average total per frame: {avg_total:.2f} ms")
    print(f"  Estimated throughput:    {1000 / avg_total:.1f} fps")


if __name__ == "__main__":
    demo_tracking_pipeline()
