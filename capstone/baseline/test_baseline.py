"""
Tests for the pure-Python baseline tracker components.

Run with:
    pytest test_baseline.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from tracker_baseline import (
    HistoryBuffer,
    KalmanFilter,
    Pipeline,
    Preprocessor,
    StateMachine,
)


# ---------------------------------------------------------------------------
# KalmanFilter
# ---------------------------------------------------------------------------

class TestKalmanFilter:
    def test_initial_state_is_zero(self):
        kf = KalmanFilter()
        assert np.allclose(kf.state, np.zeros(4))

    def test_predict_returns_array(self):
        kf = KalmanFilter()
        predicted = kf.predict()
        assert isinstance(predicted, np.ndarray)
        assert predicted.shape == (4,)

    def test_predict_advances_state(self):
        kf = KalmanFilter()
        kf.state = np.array([10.0, 20.0, 1.0, 2.0])
        predicted = kf.predict()
        # x should increase by vx*dt, y by vy*dt
        assert predicted[0] == pytest.approx(11.0, abs=0.1)
        assert predicted[1] == pytest.approx(22.0, abs=0.1)

    def test_update_moves_toward_measurement(self):
        kf = KalmanFilter()
        kf.state = np.array([0.0, 0.0, 0.0, 0.0])
        kf.predict()
        updated = kf.update(np.array([10.0, 20.0]))
        # State should move toward the measurement
        assert updated[0] > 0.0
        assert updated[1] > 0.0

    def test_predict_update_cycle(self):
        kf = KalmanFilter()
        for i in range(50):
            kf.predict()
            kf.update(np.array([float(i), float(i) * 2]))
        # After 50 iterations tracking a linear signal, state should be close
        assert kf.state[0] == pytest.approx(49.0, abs=5.0)
        assert kf.state[1] == pytest.approx(98.0, abs=10.0)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_output_shape(self):
        prep = Preprocessor()
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = prep.preprocess(img)
        assert result.shape == (3, 480, 640)

    def test_output_dtype(self):
        prep = Preprocessor()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = prep.preprocess(img)
        assert result.dtype == np.float32

    def test_output_range(self):
        prep = Preprocessor()
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = prep.preprocess(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_known_values(self):
        prep = Preprocessor()
        img = np.full((2, 2, 3), 255, dtype=np.uint8)
        result = prep.preprocess(img)
        assert np.allclose(result, 1.0)

    def test_contiguous(self):
        prep = Preprocessor()
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = prep.preprocess(img)
        assert result.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# HistoryBuffer
# ---------------------------------------------------------------------------

class TestHistoryBuffer:
    def test_push_and_len(self):
        buf = HistoryBuffer(capacity=10)
        for i in range(5):
            buf.push(np.array([i, i, i + 10, i + 10], dtype=np.float64))
        assert len(buf) == 5

    def test_latest_returns_most_recent(self):
        buf = HistoryBuffer(capacity=10)
        for i in range(5):
            buf.push(np.array([i, i, i + 10, i + 10], dtype=np.float64))
        latest = buf.latest(1)
        assert len(latest) == 1
        assert np.allclose(latest[0], [4, 4, 14, 14])

    def test_latest_n(self):
        buf = HistoryBuffer(capacity=10)
        for i in range(5):
            buf.push(np.array([i, i, i + 10, i + 10], dtype=np.float64))
        latest = buf.latest(3)
        assert len(latest) == 3
        # Most recent first
        assert np.allclose(latest[0], [4, 4, 14, 14])
        assert np.allclose(latest[1], [3, 3, 13, 13])
        assert np.allclose(latest[2], [2, 2, 12, 12])

    def test_wrap_around(self):
        buf = HistoryBuffer(capacity=5)
        for i in range(10):
            buf.push(np.array([i, i, i + 10, i + 10], dtype=np.float64))
        assert len(buf) == 5
        latest = buf.latest(1)
        assert np.allclose(latest[0], [9, 9, 19, 19])

    def test_latest_clamps_to_size(self):
        buf = HistoryBuffer(capacity=100)
        for i in range(3):
            buf.push(np.array([i, i, i, i], dtype=np.float64))
        latest = buf.latest(50)
        assert len(latest) == 3


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------

class TestStateMachine:
    def test_initial_state(self):
        sm = StateMachine()
        assert sm.state == "idle"

    def test_idle_to_tracking(self):
        sm = StateMachine()
        result = sm.on_event("detect")
        assert result == "tracking"

    def test_tracking_stays_on_detect(self):
        sm = StateMachine()
        sm.on_event("detect")  # idle -> tracking
        result = sm.on_event("detect")
        assert result == "tracking"

    def test_tracking_to_lost(self):
        sm = StateMachine()
        sm.on_event("detect")  # idle -> tracking
        result = sm.on_event("miss")
        assert result == "lost"

    def test_lost_to_tracking(self):
        sm = StateMachine()
        sm.on_event("detect")  # idle -> tracking
        sm.on_event("miss")    # tracking -> lost
        result = sm.on_event("detect")
        assert result == "tracking"

    def test_lost_to_expired(self):
        sm = StateMachine()
        sm.on_event("detect")  # idle -> tracking
        sm.on_event("miss")    # tracking -> lost
        # Need > 10 frames in lost state with miss events
        for _ in range(11):
            result = sm.on_event("miss")
        assert result == "expired"

    def test_expired_to_idle(self):
        sm = StateMachine()
        sm.on_event("detect")
        sm.on_event("miss")
        for _ in range(12):
            sm.on_event("miss")
        assert sm.state == "expired"
        result = sm.on_event("reset")
        assert result == "idle"

    def test_idle_ignores_miss(self):
        sm = StateMachine()
        result = sm.on_event("miss")
        assert result == "idle"


# ---------------------------------------------------------------------------
# Pipeline (integration)
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_process_frame_returns_dict(self):
        pipeline = Pipeline()
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(img)
        assert isinstance(result, dict)
        assert "preprocessed_shape" in result
        assert "predicted" in result
        assert "state" in result

    def test_process_frame_with_detection(self):
        pipeline = Pipeline()
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        det = np.array([100.0, 200.0, 150.0, 250.0], dtype=np.float64)
        result = pipeline.process_frame(img, det)
        assert result["state"] == "tracking"

    def test_multiple_frames(self):
        pipeline = Pipeline()
        rng = np.random.default_rng(0)
        for i in range(20):
            img = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
            det = np.array([i, i, i + 50, i + 50], dtype=np.float64) if i % 2 == 0 else None
            result = pipeline.process_frame(img, det)
        assert result["history_len"] == 20
