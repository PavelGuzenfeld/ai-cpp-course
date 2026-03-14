"""Unit tests for Lesson 6 measurement tools."""

import time
import pytest

from measure_latency import LatencyTracker
from gpu_timer import GpuTimer, _HAS_CUDA

# Try importing C++ modules — tests that need them will be skipped if unavailable
try:
    import latency_timer
    import cache_benchmark
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


class TestScopedTimer:
    """Test the C++ ScopedTimer via nanobind."""

    @pytest.fixture(autouse=True)
    def reset_timer(self):
        if _HAS_CPP:
            latency_timer.Timer.instance().reset()
        yield

    @pytest.mark.skipif(not _HAS_CPP, reason="C++ modules not built")
    def test_timer_records_measurement(self):
        """ScopedTimer should record a timing measurement."""
        with latency_timer.ScopedTimer("test_section"):
            time.sleep(0.01)

        timer = latency_timer.Timer.instance()
        assert timer.count() == 1
        assert timer.count_for("test_section") == 1

    @pytest.mark.skipif(not _HAS_CPP, reason="C++ modules not built")
    def test_timer_accuracy(self):
        """Timing should be within 20% of a known sleep duration."""
        sleep_ms = 50
        with latency_timer.ScopedTimer("accuracy_test"):
            time.sleep(sleep_ms / 1000.0)

        timer = latency_timer.Timer.instance()
        timings = timer.timings_for("accuracy_test")
        assert len(timings) == 1

        measured_ms = timings[0] / 1e6
        assert measured_ms >= sleep_ms * 0.8, f"Too fast: {measured_ms:.1f}ms < {sleep_ms * 0.8}ms"
        assert measured_ms <= sleep_ms * 1.2, f"Too slow: {measured_ms:.1f}ms > {sleep_ms * 1.2}ms"

    @pytest.mark.skipif(not _HAS_CPP, reason="C++ modules not built")
    def test_multiple_sections(self):
        """Multiple named sections should be tracked independently."""
        for _ in range(3):
            with latency_timer.ScopedTimer("section_a"):
                time.sleep(0.001)
        for _ in range(5):
            with latency_timer.ScopedTimer("section_b"):
                time.sleep(0.001)

        timer = latency_timer.Timer.instance()
        assert timer.count_for("section_a") == 3
        assert timer.count_for("section_b") == 5
        assert timer.count() == 8

    @pytest.mark.skipif(not _HAS_CPP, reason="C++ modules not built")
    def test_reset(self):
        """Reset should clear all recorded timings."""
        with latency_timer.ScopedTimer("before_reset"):
            time.sleep(0.001)

        timer = latency_timer.Timer.instance()
        assert timer.count() == 1
        timer.reset()
        assert timer.count() == 0


class TestCacheBenchmark:
    """Test the C++ cache benchmark module."""

    @pytest.mark.skipif(not _HAS_CPP, reason="C++ modules not built")
    def test_sequential_faster_than_random(self):
        """Sequential access should be faster than random access."""
        size = 16 * 1024 * 1024  # 16MB — larger than L1/L2, likely L3 or RAM
        seq_ns = cache_benchmark.benchmark_sequential(size, iterations=3)
        rand_ns = cache_benchmark.benchmark_random(size, iterations=3)

        assert seq_ns > 0, "Sequential timing should be positive"
        assert rand_ns > 0, "Random timing should be positive"
        assert seq_ns < rand_ns, (
            f"Sequential ({seq_ns:.2f} ns) should be faster than "
            f"random ({rand_ns:.2f} ns)"
        )

    @pytest.mark.skipif(not _HAS_CPP, reason="C++ modules not built")
    def test_benchmark_returns_results(self):
        """run_cache_benchmark should return non-empty results."""
        result = cache_benchmark.run_cache_benchmark(iterations=1)
        assert len(result.sizes_bytes) > 0
        assert len(result.sequential_ns_per_access) == len(result.sizes_bytes)
        assert len(result.random_ns_per_access) == len(result.sizes_bytes)

    @pytest.mark.skipif(not _HAS_CPP, reason="C++ modules not built")
    def test_stride_benchmark_returns_results(self):
        """run_stride_benchmark should return non-empty results."""
        result = cache_benchmark.run_stride_benchmark(
            array_size_bytes=4 * 1024 * 1024, iterations=1
        )
        assert len(result.strides) > 0
        assert len(result.ns_per_access) == len(result.strides)
        assert all(ns > 0 for ns in result.ns_per_access)


class TestLatencyTracker:
    """Test the Python LatencyTracker."""

    def test_records_correct_count(self):
        """LatencyTracker should record the correct number of samples."""
        tracker = LatencyTracker()
        n = 10
        for _ in range(n):
            with tracker.section("test"):
                pass  # just measure overhead
        assert tracker.count("test") == n
        assert tracker.total_count() == n

    def test_multiple_sections(self):
        """Multiple sections should be tracked independently."""
        tracker = LatencyTracker()
        for _ in range(3):
            with tracker.section("a"):
                pass
        for _ in range(7):
            with tracker.section("b"):
                pass
        assert tracker.count("a") == 3
        assert tracker.count("b") == 7
        assert tracker.total_count() == 10
        assert set(tracker.sections) == {"a", "b"}

    def test_timing_is_positive(self):
        """Recorded timings should be positive."""
        tracker = LatencyTracker()
        with tracker.section("positive_test"):
            time.sleep(0.001)
        timings = tracker.timings_ns("positive_test")
        assert len(timings) == 1
        assert timings[0] > 0

    def test_percentiles(self):
        """Percentile computation should return sensible values."""
        tracker = LatencyTracker()
        # Manually record known values
        for val in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            tracker.record("known", val)

        p50 = tracker.p50("known")
        p95 = tracker.p95("known")
        p99 = tracker.p99("known")

        assert p50 == pytest.approx(550, rel=0.01)
        assert p95 > p50
        assert p99 >= p95

    def test_reset(self):
        """Reset should clear all data."""
        tracker = LatencyTracker()
        with tracker.section("before"):
            pass
        assert tracker.total_count() == 1
        tracker.reset()
        assert tracker.total_count() == 0
        assert tracker.sections == []


class TestGpuTimer:
    """Test the GPU timer context manager."""

    def test_context_manager_returns_positive(self):
        """GpuTimer should return a positive elapsed time."""
        with GpuTimer("test_op") as t:
            time.sleep(0.005)

        assert t.result is not None
        assert t.elapsed_ms > 0

    def test_result_has_correct_name(self):
        """TimingResult should have the correct operation name."""
        with GpuTimer("my_operation") as t:
            time.sleep(0.001)

        assert t.result.name == "my_operation"

    def test_cpu_fallback(self):
        """When device='cpu', should use CPU timing."""
        with GpuTimer("cpu_op", device="cpu") as t:
            time.sleep(0.01)

        assert t.result is not None
        assert t.result.device == "cpu"
        assert t.elapsed_ms > 5  # should be at least ~10ms

    def test_device_reported_correctly(self):
        """Device field should reflect actual device used."""
        with GpuTimer("device_test", device="cpu") as t:
            pass
        assert t.result.device == "cpu"

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
    def test_cuda_timing(self):
        """On CUDA systems, should use GPU timing."""
        import torch
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")

        # Warm up
        _ = torch.mm(a, b)
        torch.cuda.synchronize()

        with GpuTimer("cuda_matmul", device="cuda") as t:
            _ = torch.mm(a, b)

        assert t.result is not None
        assert t.result.device == "cuda"
        assert t.elapsed_ms > 0
