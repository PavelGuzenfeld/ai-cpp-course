"""
Unit tests for optimization correctness.

Verifies that each optimization stage produces identical results to baseline.

    pytest test_optimization.py -v
"""

import time
import numpy as np
import pytest
from scipy import stats

from tracker_pipeline import (
    KalmanFilter,
    PostProcessor,
    Preprocessor,
    Pipeline,
    generate_test_frame,
    simulate_inference,
)
from filter_comparison import FrozenGainFilter, _mix_initial_conditions
from learned_vs_classical import (
    measure_latency,
    run_calibration_demo,
    run_imm_vs_frozen,
    run_underflow_demo,
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

        np.testing.assert_array_equal(result_b, result_o)

    def test_preprocess_multiple_frames(self):
        """Multiple frames all match (tests buffer reuse correctness)."""
        baseline = Preprocessor(target_h=64, target_w=64, pad_h=80, pad_w=80)
        optimized = PreprocessorOptimized(target_h=64, target_w=64, pad_h=80, pad_w=80)

        for seed in range(20):
            frame = generate_test_frame(height=120, width=160, seed=seed)

            result_b = baseline.preprocess(frame)
            # Must copy because optimized returns a view of internal buffer
            result_o = optimized.preprocess(frame).copy()

            np.testing.assert_array_equal(
                result_b, result_o,
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
    """Bounds derived from replayed measurements, not picked to pass.

    Each bound's replay distribution is in the README.
    """

    def _run_timed(self, pipeline_class, num_frames=100, trials=5):
        """Median of `trials` timed runs, in ns.

        Fresh pipeline per trial: they accumulate result history.
        """
        # Generated outside the timed region. It is identical work in both
        # arms, so it only ever compresses the ratio towards 1.
        warmup_frames = [generate_test_frame(height=120, width=160, seed=i)
                         for i in range(10)]
        frames = [generate_test_frame(height=120, width=160, seed=i + 10)
                  for i in range(num_frames)]

        times = []
        for _ in range(trials):
            np.random.seed(42)
            pipeline = pipeline_class()

            for frame in warmup_frames:
                pipeline.process_frame(frame)

            t0 = time.perf_counter_ns()
            for frame in frames:
                pipeline.process_frame(frame)
            times.append(time.perf_counter_ns() - t0)

        return float(np.median(times))

    def test_optimized_pipeline_is_several_times_faster_end_to_end(self):
        baseline_time = self._run_timed(Pipeline)
        optimized_time = self._run_timed(PipelineOptimized)

        # 30 replays: median ratio 0.091, worst 0.116. 0.25 leaves 2.2x over
        # the worst sample -- the same margin #86 settled on for the old bound.
        assert optimized_time < baseline_time * 0.25, (
            f"Optimized ({optimized_time / 1e6:.1f} ms) is not 4x faster than "
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

        def timed(cls, trials=5):
            times = []
            for _ in range(trials):
                instance = cls()
                for _ in range(1000):
                    instance.format_result(state)
                t0 = time.perf_counter_ns()
                for _ in range(n):
                    instance.format_result(state)
                times.append(time.perf_counter_ns() - t0)
            return float(np.median(times))

        baseline_time = timed(PostProcessor)
        optimized_time = timed(PostProcessorOptimized)

        # 1.3 unchanged. This times 10 ms of work, so one scheduling outlier
        # reached 1.529 once in 100 sweeps (#86); a median of 5 discards it.
        assert optimized_time < baseline_time * 1.3, (
            f"PostProcessor optimized ({optimized_time / 1e6:.1f} ms) is slower than "
            f"baseline ({baseline_time / 1e6:.1f} ms)"
        )


class TestImmMixing:
    """The mixing step is what makes an IMM an IMM rather than two filters
    averaged after the fact -- test it directly, not just through the
    filter's overall statistical behavior."""

    def test_mix_weights_are_a_normalized_blend_for_each_target_mode(self):
        x_modes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 5.0, 0.0])]
        p_modes = [np.eye(3), np.eye(3) * 2]
        mode_probs = np.array([0.5, 0.5])
        transition = np.array([[0.9, 0.1], [0.1, 0.9]])

        c, mixed_x, mixed_p = _mix_initial_conditions(x_modes, p_modes, mode_probs, transition)

        # Symmetric prior + symmetric transition: predicted mode
        # probabilities must stay exactly 0.5 / 0.5.
        assert c[0] == pytest.approx(0.5)
        assert c[1] == pytest.approx(0.5)

        # Mode 0 (persistence 0.9) should stay closer to x_modes[0] than an
        # unweighted 50/50 average would -- a tight, hand-computed bound,
        # not just "some blend happened".
        expected_mode0 = 0.9 * x_modes[0] + 0.1 * x_modes[1]
        assert mixed_x[0] == pytest.approx(expected_mode0)

    def test_mix_weights_reduce_to_the_prior_when_transition_is_identity(self):
        """A transition matrix with no mode switching must leave each
        mode's state untouched -- the simplest case with a known answer."""
        x_modes = [np.array([3.0, 1.0, 0.0]), np.array([-2.0, 0.0, 0.0])]
        p_modes = [np.eye(3), np.eye(3)]
        mode_probs = np.array([0.3, 0.7])
        transition = np.eye(2)

        _, mixed_x, _ = _mix_initial_conditions(x_modes, p_modes, mode_probs, transition)

        assert mixed_x[0] == pytest.approx(x_modes[0])
        assert mixed_x[1] == pytest.approx(x_modes[1])


class TestLearnedVsClassical:
    """Round 6: should this be a learned model at all?"""

    def test_frozen_gain_filter_never_reports_a_covariance(self):
        """Mirrors gst-nvmm-cpp's finding #1: the learned filter's inability
        to produce uncertainty is structural, not an oversight -- a
        FrozenGainFilter simply has no covariance attribute to read."""
        frozen = FrozenGainFilter(dt=0.1)
        assert not hasattr(frozen, "P")

    def test_imm_anees_falls_inside_its_chi_square_band(self):
        """A correctly calibrated 1-dof filter's ANEES should land inside
        the chi-square(1) 90% band -- this is the positive control for the
        miscalibration test below."""
        comparison = run_imm_vs_frozen()
        lo, hi = stats.chi2.ppf(0.05, df=1), stats.chi2.ppf(0.95, df=1)
        assert lo <= comparison["imm_anees"] <= hi

    def test_raw_exp_mode_weighting_underflows_on_a_heavy_tailed_residual(self):
        result = run_underflow_demo()
        assert result["raw_space_underflowed"]

    def test_log_space_mode_weighting_survives_the_same_residual(self):
        result = run_underflow_demo()
        assert result["log_space_survived"]

    def test_understated_measurement_noise_breaks_calibration_without_moving_rmse_much(self):
        """The core of the lesson: RMSE alone cannot distinguish a
        well-calibrated filter from an overconfident one. Understating R by
        10x should barely move RMSE but should blow the ANEES band and drag
        coverage well below its nominal fraction."""
        calib = run_calibration_demo()
        lo, hi = stats.chi2.ppf(0.05, df=1), stats.chi2.ppf(0.95, df=1)

        assert lo <= calib["correct"]["anees"] <= hi
        assert calib["understated_10x"]["anees"] > hi

        assert calib["understated_10x"]["coverage_68"] < 0.5
        assert calib["understated_10x"]["coverage_95"] < 0.8

        rmse_ratio = calib["understated_10x"]["rmse"] / calib["correct"]["rmse"]
        assert rmse_ratio < 2.0, (
            f"RMSE moved {rmse_ratio:.2f}x -- calibration test no longer isolates "
            f"the miscalibration from an accuracy change"
        )

    def test_imm_step_latency_stays_within_the_stated_budget(self):
        latency = measure_latency(budget_ms=2.0)
        assert latency["within_budget"], (
            f"median step time {latency['median_ms']:.4f} ms exceeds "
            f"the {latency['budget_ms']} ms budget"
        )
