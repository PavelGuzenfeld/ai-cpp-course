"""
Unit tests for optimization correctness.

Verifies that each optimization stage produces identical results to baseline.

    pytest test_optimization.py -v
"""

import time
import numpy as np
import pytest

from tracker_pipeline import (
    KalmanFilter,
    PostProcessor,
    Preprocessor,
    Pipeline,
    generate_test_frame,
    simulate_inference,
)
from tracker_pipeline_optimized import (
    KalmanFilterOptimized,
    PostProcessorOptimized,
    PreprocessorOptimized,
    PipelineOptimized,
)


# ---------------------------------------------------------------------------
# Kalman filter correctness
# ---------------------------------------------------------------------------

class TestKalmanCorrectness:
    """Verify optimized Kalman filter produces identical outputs."""

    def test_predict_matches(self):
        """Single predict step produces same state."""
        np.random.seed(99)
        baseline = KalmanFilter(state_dim=12, measure_dim=4)
        np.random.seed(99)
        optimized = KalmanFilterOptimized(state_dim=12, measure_dim=4)

        np.random.seed(123)
        result_b = baseline.predict()
        np.random.seed(123)
        result_o = optimized.predict()

        np.testing.assert_allclose(result_b, result_o, atol=1e-12)

    def test_update_matches(self):
        """Single update step produces same state."""
        np.random.seed(99)
        baseline = KalmanFilter(state_dim=12, measure_dim=4)
        np.random.seed(99)
        optimized = KalmanFilterOptimized(state_dim=12, measure_dim=4)

        measurement = [100.0, 200.0, 50.0, 60.0]
        result_b = baseline.update(measurement)
        result_o = optimized.update(measurement)

        np.testing.assert_allclose(result_b, result_o, atol=1e-12)

    def test_predict_update_sequence(self):
        """Multiple predict+update cycles produce same state."""
        np.random.seed(42)
        baseline = KalmanFilter(state_dim=12, measure_dim=4)
        np.random.seed(42)
        optimized = KalmanFilterOptimized(state_dim=12, measure_dim=4)

        for i in range(50):
            np.random.seed(1000 + i)
            baseline.predict()
            np.random.seed(1000 + i)
            optimized.predict()

            m = [100.0 + i, 200.0 - i, 50.0, 60.0]
            result_b = baseline.update(m)
            result_o = optimized.update(m)

            np.testing.assert_allclose(
                result_b, result_o, atol=1e-10,
                err_msg=f"Mismatch at step {i}"
            )

    def test_covariance_matches(self):
        """P matrix stays in sync between implementations."""
        np.random.seed(42)
        baseline = KalmanFilter(state_dim=12, measure_dim=4)
        np.random.seed(42)
        optimized = KalmanFilterOptimized(state_dim=12, measure_dim=4)

        for i in range(20):
            np.random.seed(2000 + i)
            baseline.predict()
            np.random.seed(2000 + i)
            optimized.predict()

            m = [50.0 * np.sin(i), 50.0 * np.cos(i), 30.0, 40.0]
            baseline.update(m)
            optimized.update(m)

        np.testing.assert_allclose(
            baseline.P, optimized.P, atol=1e-8,
            err_msg="Covariance matrices diverged"
        )


# ---------------------------------------------------------------------------
# PostProcessor correctness
# ---------------------------------------------------------------------------

class TestPostProcessorCorrectness:
    """Verify optimized postprocessor produces identical outputs."""

    def test_format_result_matches(self):
        """format_result returns same values."""
        baseline = PostProcessor()
        optimized = PostProcessorOptimized()

        state = np.array([[1.5, 2.7, 3.1, 4.9]], dtype=np.float64)

        result_b = baseline.format_result(state)
        result_o = optimized.format_result(state)

        for key in ["x", "y", "w", "h"]:
            assert abs(result_b[key] - result_o[key]) < 1e-12, (
                f"Key {key}: baseline={result_b[key]}, optimized={result_o[key]}"
            )

    def test_extract_position_matches(self):
        """extract_position returns same values."""
        baseline = PostProcessor()
        optimized = PostProcessorOptimized()

        arr = np.array([[123.456, 789.012, 50.0, 60.0]], dtype=np.float64)

        bx, by = baseline.extract_position(arr)
        ox, oy = optimized.extract_position(arr)

        assert abs(bx - ox) < 1e-12
        assert abs(by - oy) < 1e-12

    def test_extract_size_matches(self):
        """extract_size returns same values."""
        baseline = PostProcessor()
        optimized = PostProcessorOptimized()

        arr = np.array([[10.0, 20.0, 300.5, 400.7]], dtype=np.float64)

        bw, bh = baseline.extract_size(arr)
        ow, oh = optimized.extract_size(arr)

        assert abs(bw - ow) < 1e-12
        assert abs(bh - oh) < 1e-12

    def test_history_accumulates(self):
        """Results history works the same in both."""
        baseline = PostProcessor()
        optimized = PostProcessorOptimized()

        for i in range(10):
            state = np.array([[float(i), float(i * 2), 10.0, 20.0]])
            baseline.format_result(state)
            optimized.format_result(state)

        assert len(baseline.results_history) == len(optimized.results_history)
        for b, o in zip(baseline.results_history, optimized.results_history):
            for key in ["x", "y", "w", "h"]:
                assert abs(b[key] - o[key]) < 1e-12


# ---------------------------------------------------------------------------
# Preprocessor correctness
# ---------------------------------------------------------------------------

class TestPreprocessorCorrectness:
    """Verify optimized preprocessor produces identical outputs."""

    def test_preprocess_matches(self):
        """Single frame preprocessing produces same output."""
        baseline = Preprocessor(target_h=64, target_w=64, pad_h=80, pad_w=80)
        optimized = PreprocessorOptimized(target_h=64, target_w=64, pad_h=80, pad_w=80)

        frame = generate_test_frame(height=120, width=160, seed=42)

        result_b = baseline.preprocess(frame)
        result_o = optimized.preprocess(frame)

        np.testing.assert_allclose(result_b, result_o, atol=1e-6)

    def test_preprocess_multiple_frames(self):
        """Multiple frames all match (tests buffer reuse correctness)."""
        baseline = Preprocessor(target_h=64, target_w=64, pad_h=80, pad_w=80)
        optimized = PreprocessorOptimized(target_h=64, target_w=64, pad_h=80, pad_w=80)

        for seed in range(20):
            frame = generate_test_frame(height=120, width=160, seed=seed)

            result_b = baseline.preprocess(frame)
            # Must copy because optimized returns a view of internal buffer
            result_o = optimized.preprocess(frame).copy()

            np.testing.assert_allclose(
                result_b, result_o, atol=1e-6,
                err_msg=f"Mismatch at frame seed={seed}"
            )

    def test_output_shape(self):
        """Output shape is correct for both."""
        baseline = Preprocessor(target_h=32, target_w=32, pad_h=48, pad_w=48)
        optimized = PreprocessorOptimized(target_h=32, target_w=32, pad_h=48, pad_w=48)

        frame = generate_test_frame(height=100, width=100, seed=0)

        assert baseline.preprocess(frame).shape == (48, 48, 3)
        assert optimized.preprocess(frame).shape == (48, 48, 3)


# ---------------------------------------------------------------------------
# Full pipeline correctness
# ---------------------------------------------------------------------------

class TestPipelineCorrectness:
    """Verify full pipeline output matches between baseline and optimized."""

    def test_single_frame_matches(self):
        """Single frame through full pipeline."""
        np.random.seed(42)
        baseline = Pipeline()
        frame = generate_test_frame(height=120, width=160, seed=0)
        result_b = baseline.process_frame(frame)

        np.random.seed(42)
        optimized = PipelineOptimized()
        frame = generate_test_frame(height=120, width=160, seed=0)
        result_o = optimized.process_frame(frame)

        for key in ["x", "y", "w", "h"]:
            assert abs(result_b[key] - result_o[key]) < 1e-6, (
                f"Key {key}: baseline={result_b[key]}, optimized={result_o[key]}"
            )

    def test_multi_frame_matches(self):
        """50 frames through full pipeline, check final result."""
        np.random.seed(42)
        baseline = Pipeline()
        for i in range(50):
            frame = generate_test_frame(height=120, width=160, seed=i)
            result_b = baseline.process_frame(frame)

        np.random.seed(42)
        optimized = PipelineOptimized()
        for i in range(50):
            frame = generate_test_frame(height=120, width=160, seed=i)
            result_o = optimized.process_frame(frame)

        for key in ["x", "y", "w", "h"]:
            assert abs(result_b[key] - result_o[key]) < 1e-4, (
                f"Key {key}: baseline={result_b[key]}, optimized={result_o[key]}"
            )


# ---------------------------------------------------------------------------
# Timing sanity checks
# ---------------------------------------------------------------------------

class TestTimingImprovement:
    """Verify optimized version is not slower than baseline.

    Uses generous tolerance — we only check that optimization didn't
    make things worse, not that it achieved a specific speedup.
    """

    def _run_timed(self, pipeline_class, num_frames=100):
        """Run pipeline and return total time in ns."""
        np.random.seed(42)
        pipeline = pipeline_class()

        # Warmup
        for i in range(10):
            frame = generate_test_frame(height=120, width=160, seed=i)
            pipeline.process_frame(frame)

        # Measure
        t0 = time.perf_counter_ns()
        for i in range(num_frames):
            frame = generate_test_frame(height=120, width=160, seed=i + 10)
            pipeline.process_frame(frame)
        t1 = time.perf_counter_ns()

        return t1 - t0

    def test_optimized_not_slower(self):
        """Optimized pipeline should not be significantly slower."""
        baseline_time = self._run_timed(Pipeline)
        optimized_time = self._run_timed(PipelineOptimized)

        # Allow 20% tolerance — optimized should not be >1.2x slower
        assert optimized_time < baseline_time * 1.2, (
            f"Optimized ({optimized_time / 1e6:.1f} ms) is slower than "
            f"baseline ({baseline_time / 1e6:.1f} ms)"
        )

    def test_kalman_improvement(self):
        """Optimized Kalman filter should be faster."""
        n = 5000

        np.random.seed(42)
        baseline = KalmanFilter(state_dim=12, measure_dim=4)
        for _ in range(100):
            baseline.predict()

        t0 = time.perf_counter_ns()
        for _ in range(n):
            baseline.predict()
        baseline_time = time.perf_counter_ns() - t0

        np.random.seed(42)
        optimized = KalmanFilterOptimized(state_dim=12, measure_dim=4)
        for _ in range(100):
            optimized.predict()

        t0 = time.perf_counter_ns()
        for _ in range(n):
            optimized.predict()
        optimized_time = time.perf_counter_ns() - t0

        # Optimized should not be slower (allow 30% tolerance for noisy envs)
        assert optimized_time < baseline_time * 1.3, (
            f"Kalman optimized ({optimized_time / 1e6:.1f} ms) is slower than "
            f"baseline ({baseline_time / 1e6:.1f} ms)"
        )

    def test_postprocessor_improvement(self):
        """Optimized postprocessor should be faster."""
        n = 10000
        state = np.array([[100.0, 200.0, 50.0, 60.0]], dtype=np.float64)

        baseline = PostProcessor()
        t0 = time.perf_counter_ns()
        for _ in range(n):
            baseline.format_result(state)
        baseline_time = time.perf_counter_ns() - t0

        optimized = PostProcessorOptimized()
        t0 = time.perf_counter_ns()
        for _ in range(n):
            optimized.format_result(state)
        optimized_time = time.perf_counter_ns() - t0

        # Optimized should not be slower
        assert optimized_time < baseline_time * 1.3, (
            f"PostProcessor optimized ({optimized_time / 1e6:.1f} ms) is slower than "
            f"baseline ({baseline_time / 1e6:.1f} ms)"
        )
