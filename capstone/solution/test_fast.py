"""
Tests for the fast tracker — validates that fast implementations produce
the same results as the baseline.

Run with:
    pytest test_fast.py -v
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "baseline"))

from tracker_baseline import (  # noqa: E402
    HistoryBuffer as BaselineHistory,
    KalmanFilter as BaselineKalman,
    Pipeline as BaselinePipeline,
    Preprocessor as BaselinePreprocessor,
    StateMachine as BaselineStateMachine,
)

from tracker_fast import (  # noqa: E402
    FastHistoryBuffer,
    FastKalmanFilter,
    FastPreprocessor,
    FastStateMachine,
    Pipeline as FastPipeline,
)


# ---------------------------------------------------------------------------
# FastKalmanFilter
# ---------------------------------------------------------------------------

class TestFastKalmanFilter:
    def test_predict_matches_baseline(self):
        bkf = BaselineKalman(dt=1.0, process_noise=0.01, measurement_noise=0.1)
        fkf = FastKalmanFilter(dt=1.0, process_noise=0.01, measurement_noise=0.1)
        bkf.state[:] = [10, 20, 1, 2]
        fkf.state[:] = [10, 20, 1, 2]

        bp = bkf.predict()
        fp = fkf.predict()
        assert np.allclose(bp, fp, atol=1e-10)

    def test_update_matches_baseline(self):
        bkf = BaselineKalman()
        fkf = FastKalmanFilter()
        bkf.state[:] = fkf.state[:] = [5, 10, 0.5, 1.0]
        bkf.covariance[:] = fkf.covariance[:] = np.eye(4) * 2.0

        bkf.predict()
        fkf.predict()
        meas = np.array([8.0, 15.0])
        bu = bkf.update(meas)
        fu = fkf.update(meas)
        assert np.allclose(bu, fu, atol=1e-10)

    def test_multi_step_convergence(self):
        bkf = BaselineKalman()
        fkf = FastKalmanFilter()
        rng = np.random.default_rng(123)
        for _ in range(100):
            bkf.predict()
            fkf.predict()
            m = rng.uniform(0, 100, size=2)
            bkf.update(m)
            fkf.update(m)
        assert np.allclose(bkf.state, fkf.state, atol=1e-8)


# ---------------------------------------------------------------------------
# FastPreprocessor
# ---------------------------------------------------------------------------

class TestFastPreprocessor:
    def test_output_matches_baseline(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        bp = BaselinePreprocessor()
        fp = FastPreprocessor()
        b_out = bp.preprocess(img)
        f_out = fp.preprocess(img)
        assert b_out.shape == f_out.shape
        assert np.allclose(b_out, f_out, atol=1e-5)

    def test_output_shape_and_dtype(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        fp = FastPreprocessor()
        out = fp.preprocess(img)
        assert out.shape == (3, 100, 200)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# FastHistoryBuffer
# ---------------------------------------------------------------------------

class TestFastHistoryBuffer:
    def test_push_and_latest(self):
        fh = FastHistoryBuffer(capacity=10)
        for i in range(5):
            fh.push(np.array([i, i, i + 10, i + 10], dtype=np.float64))
        latest = fh.latest(1)
        assert np.allclose(latest[0] if isinstance(latest, list) else latest[0],
                           [4, 4, 14, 14])

    def test_matches_baseline(self):
        bh = BaselineHistory(capacity=20)
        fh = FastHistoryBuffer(capacity=20)
        rng = np.random.default_rng(7)
        for _ in range(30):
            bbox = rng.uniform(0, 500, size=4)
            bh.push(bbox)
            fh.push(bbox)
        b_latest = bh.latest(5)
        f_latest = fh.latest(5)
        for b, f in zip(b_latest,
                        f_latest if isinstance(f_latest, list) else [f_latest[i] for i in range(len(f_latest))]):
            assert np.allclose(b, f, atol=1e-12)

    def test_len(self):
        fh = FastHistoryBuffer(capacity=10)
        for i in range(15):
            fh.push(np.array([i, i, i, i], dtype=np.float64))
        assert len(fh) == 10


# ---------------------------------------------------------------------------
# FastStateMachine
# ---------------------------------------------------------------------------

class TestFastStateMachine:
    def test_transitions_match_baseline(self):
        bsm = BaselineStateMachine()
        fsm = FastStateMachine()
        events = (["detect"] * 5 + ["miss"] * 15 + ["reset"]
                  + ["detect"] * 3 + ["miss"] * 2)
        for ev in events:
            b_state = bsm.on_event(ev)
            f_state = fsm.on_event(ev)
            assert b_state == f_state, f"Mismatch on event '{ev}': {b_state} vs {f_state}"


# ---------------------------------------------------------------------------
# Full pipeline cross-validation
# ---------------------------------------------------------------------------

class TestPipelineCrossValidation:
    def test_100_frames_match(self):
        """Both pipelines must produce the same state and predictions for
        100 frames with identical input."""
        rng = np.random.default_rng(42)
        bp = BaselinePipeline()
        fp = FastPipeline()

        for i in range(100):
            img = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
            det = (np.array([50.0 + i, 80.0 + i, 100.0 + i, 130.0 + i],
                            dtype=np.float64) if i % 3 != 2 else None)
            b_res = bp.process_frame(img, det)
            f_res = fp.process_frame(img, det)

            assert b_res["preprocessed_shape"] == f_res["preprocessed_shape"], \
                f"Frame {i}: shape mismatch"
            assert np.allclose(b_res["predicted"], f_res["predicted"], atol=1e-6), \
                f"Frame {i}: predicted mismatch"
            assert b_res["state"] == f_res["state"], \
                f"Frame {i}: state mismatch: {b_res['state']} vs {f_res['state']}"
            assert b_res["history_len"] == f_res["history_len"], \
                f"Frame {i}: history length mismatch"
