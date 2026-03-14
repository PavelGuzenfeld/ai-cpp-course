"""Integration tests for Lesson 6 measurement tools."""

import time
import pytest

from measure_latency import LatencyTracker
from cache_explorer import detect_boundaries

try:
    import cache_benchmark
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


class TestPipelineTimingBreakdown:
    """Integration test: multi-stage pipeline timing breakdown."""

    def test_pipeline_stages_sum_to_total(self):
        """Individual stage timings should sum to approximately the total."""
        tracker = LatencyTracker()
        total_tracker = LatencyTracker()

        n_frames = 10
        for _ in range(n_frames):
            frame_start = time.perf_counter_ns()

            with tracker.section("preprocess"):
                time.sleep(0.003)  # 3ms

            with tracker.section("inference"):
                time.sleep(0.010)  # 10ms

            with tracker.section("postprocess"):
                time.sleep(0.002)  # 2ms

            frame_end = time.perf_counter_ns()
            total_tracker.record("total", frame_end - frame_start)

        # Verify counts
        assert tracker.count("preprocess") == n_frames
        assert tracker.count("inference") == n_frames
        assert tracker.count("postprocess") == n_frames

        # Sum of stage means should be close to total mean
        stage_sum_ns = (
            tracker.mean("preprocess")
            + tracker.mean("inference")
            + tracker.mean("postprocess")
        )
        total_mean_ns = total_tracker.mean("total")

        # Allow 20% tolerance for overhead between stages
        ratio = stage_sum_ns / total_mean_ns if total_mean_ns > 0 else 0
        assert 0.8 <= ratio <= 1.2, (
            f"Stage sum ({stage_sum_ns / 1e6:.2f} ms) should be close to "
            f"total ({total_mean_ns / 1e6:.2f} ms), ratio={ratio:.3f}"
        )

    def test_inference_dominates(self):
        """Inference stage should dominate the pipeline timing."""
        tracker = LatencyTracker()

        for _ in range(10):
            with tracker.section("preprocess"):
                time.sleep(0.002)
            with tracker.section("inference"):
                time.sleep(0.015)
            with tracker.section("postprocess"):
                time.sleep(0.001)

        # Inference should be > 70% of total stage time
        total = tracker.mean("preprocess") + tracker.mean("inference") + tracker.mean("postprocess")
        inference_fraction = tracker.mean("inference") / total if total > 0 else 0

        assert inference_fraction > 0.7, (
            f"Inference should dominate pipeline, but fraction = {inference_fraction:.2f}"
        )


class TestPercentileAccuracy:
    """Test percentile computation with known distributions."""

    def test_uniform_percentiles(self):
        """Percentiles of uniformly spaced data should be predictable."""
        tracker = LatencyTracker()
        # Create 100 evenly spaced values from 1 to 100
        for i in range(1, 101):
            tracker.record("uniform", float(i))

        # p50 should be ~50.5 (interpolated median of 1..100)
        p50 = tracker.p50("uniform")
        assert 49 <= p50 <= 52, f"p50 = {p50}, expected ~50.5"

        # p95 should be ~95
        p95 = tracker.p95("uniform")
        assert 93 <= p95 <= 97, f"p95 = {p95}, expected ~95"

        # p99 should be ~99
        p99 = tracker.p99("uniform")
        assert 97 <= p99 <= 101, f"p99 = {p99}, expected ~99"

    def test_percentile_ordering(self):
        """p50 <= p95 <= p99 should always hold."""
        tracker = LatencyTracker()
        for _ in range(50):
            with tracker.section("ordering"):
                time.sleep(0.001)

        p50 = tracker.p50("ordering")
        p95 = tracker.p95("ordering")
        p99 = tracker.p99("ordering")

        assert p50 <= p95, f"p50 ({p50}) should be <= p95 ({p95})"
        assert p95 <= p99, f"p95 ({p95}) should be <= p99 ({p99})"

    def test_single_value_percentile(self):
        """With a single value, all percentiles should equal that value."""
        tracker = LatencyTracker()
        tracker.record("single", 42.0)

        assert tracker.p50("single") == 42.0
        assert tracker.p95("single") == 42.0
        assert tracker.p99("single") == 42.0


class TestCacheBoundaryDetection:
    """Test that cache benchmark detects at least one boundary."""

    @pytest.mark.skipif(not _HAS_CPP, reason="C++ modules not built")
    def test_detects_at_least_one_boundary(self):
        """Cache benchmark should detect at least one latency jump."""
        result = cache_benchmark.run_cache_benchmark(iterations=3)

        sizes = list(result.sizes_bytes)
        rand_ns = list(result.random_ns_per_access)

        # Use random access latencies to detect boundaries
        boundaries = detect_boundaries(sizes, rand_ns, threshold=1.3)

        assert len(boundaries) >= 1, (
            "Should detect at least one cache boundary (L1->L2, L2->L3, or L3->RAM). "
            f"Random latencies: {[f'{ns:.1f}' for ns in rand_ns]}"
        )

    @pytest.mark.skipif(not _HAS_CPP, reason="C++ modules not built")
    def test_larger_arrays_slower_random_access(self):
        """Random access to 256MB should be slower than 4KB."""
        small = cache_benchmark.benchmark_random(4 * 1024, iterations=5)
        large = cache_benchmark.benchmark_random(64 * 1024 * 1024, iterations=3)

        assert large > small, (
            f"64MB random access ({large:.2f} ns) should be slower than "
            f"4KB random access ({small:.2f} ns)"
        )

    @pytest.mark.skipif(not _HAS_CPP, reason="C++ modules not built")
    def test_sequential_relatively_stable(self):
        """Sequential access latency should not vary as dramatically as random."""
        result = cache_benchmark.run_cache_benchmark(iterations=2)

        seq_ns = list(result.sequential_ns_per_access)

        # Sequential access with prefetching should not show huge jumps
        # The ratio between max and min should be much smaller than for random
        if min(seq_ns) > 0:
            seq_ratio = max(seq_ns) / min(seq_ns)
            rand_ns = list(result.random_ns_per_access)
            rand_ratio = max(rand_ns) / min(rand_ns) if min(rand_ns) > 0 else 1

            assert seq_ratio < rand_ratio, (
                f"Sequential variation ({seq_ratio:.1f}x) should be less than "
                f"random variation ({rand_ratio:.1f}x)"
            )
