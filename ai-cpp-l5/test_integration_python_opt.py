"""
Lesson 5: Integration tests -- simulate a full tracking loop.

Verifies that "fast" (optimized) versions produce identical results to "slow"
(tracker_engine-style) versions, while using less memory.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from bbox_slots import BboxSlow, BboxSlots, BboxDataclass
from numpy_views import HistoryCopy, HistoryView
from preallocated_buffers import VelocityTrackerSlow, VelocityTrackerFast
from thread_pool_io import save_images_threads, save_images_pool


class TestTrackingLoopIntegration:
    """Simulate a tracking loop that creates bboxes, updates history,
    computes velocity, and saves debug images -- then verify that fast
    versions produce identical results to slow versions."""

    N_FRAMES = 200
    HISTORY_CAP = 50

    @pytest.fixture
    def trajectory(self) -> np.ndarray:
        """Generate a synthetic target trajectory: starts moving, then stops."""
        rng = np.random.default_rng(123)
        # Moving phase: 150 frames at ~4 px/frame
        moving = np.cumsum(rng.normal(4.0, 0.5, size=(150, 2)), axis=0)
        # Stationary phase: 50 frames with tiny jitter
        last_pos = moving[-1]
        stationary = last_pos + rng.normal(0.0, 0.05, size=(50, 2))
        return np.vstack([moving, stationary])

    def test_bbox_results_match(self, trajectory):
        """All bbox variants produce identical property values."""
        for i in range(min(50, len(trajectory))):
            x, y = trajectory[i]
            w, h = 40.0 + i * 0.1, 30.0 + i * 0.05

            slow = BboxSlow(x, y, w, h)
            fast = BboxSlots(x, y, w, h)
            dc = BboxDataclass(x, y, w, h)

            assert slow.cx == pytest.approx(fast.cx)
            assert slow.cy == pytest.approx(fast.cy)
            assert slow.area == pytest.approx(fast.area)
            assert slow.cx == pytest.approx(dc.cx)
            assert slow.cy == pytest.approx(dc.cy)
            assert slow.area == pytest.approx(dc.area)

    def test_bbox_iou_consistency(self, trajectory):
        """IOU between corresponding pairs matches across all variants."""
        for i in range(0, min(50, len(trajectory)) - 1):
            x1, y1 = trajectory[i]
            x2, y2 = trajectory[i + 1]
            w, h = 40.0, 30.0

            slow_a = BboxSlow(x1, y1, w, h)
            slow_b = BboxSlow(x2, y2, w, h)

            fast_a = BboxSlots(x1, y1, w, h)
            fast_b = BboxSlots(x2, y2, w, h)

            dc_a = BboxDataclass(x1, y1, w, h)
            dc_b = BboxDataclass(x2, y2, w, h)

            iou_slow = slow_a.iou(slow_b)
            iou_fast = fast_a.iou(fast_b)
            iou_dc = dc_a.iou(dc_b)

            assert iou_slow == pytest.approx(iou_fast, abs=1e-12)
            assert iou_slow == pytest.approx(iou_dc, abs=1e-12)

    def test_history_results_match(self, trajectory):
        """HistoryCopy and HistoryView produce identical latest() data."""
        hc = HistoryCopy(capacity=self.HISTORY_CAP, cols=2)
        hv = HistoryView(capacity=self.HISTORY_CAP, cols=2)

        for i in range(len(trajectory)):
            hc.push(trajectory[i])
            hv.push(trajectory[i])

            if (i + 1) % 10 == 0:
                n = min(20, hc.count)
                result_copy = hc.latest(n)
                result_view = hv.latest(n)
                np.testing.assert_array_equal(result_copy, result_view)

    def test_history_wrap_around_consistency(self, trajectory):
        """After wrap-around, both variants return the same data."""
        hc = HistoryCopy(capacity=self.HISTORY_CAP, cols=2)
        hv = HistoryView(capacity=self.HISTORY_CAP, cols=2)

        # Push more than capacity to force wrap-around
        for i in range(len(trajectory)):
            hc.push(trajectory[i])
            hv.push(trajectory[i])

        # Request exactly capacity entries
        result_copy = hc.latest(self.HISTORY_CAP)
        result_view = hv.latest(self.HISTORY_CAP)
        np.testing.assert_array_equal(result_copy, result_view)

    def test_velocity_tracker_full_loop(self, trajectory):
        """Run both velocity trackers through the entire trajectory."""
        slow = VelocityTrackerSlow(threshold=2.0, ema_alpha=0.3)
        fast = VelocityTrackerFast(
            max_history=self.HISTORY_CAP + 1,
            threshold=2.0,
            ema_alpha=0.3,
        )

        hc = HistoryCopy(capacity=self.HISTORY_CAP, cols=2)

        for i in range(len(trajectory)):
            hc.push(trajectory[i])

            if hc.count >= 5:
                positions = hc.latest(min(hc.count, self.HISTORY_CAP))
                v_slow = slow.compute_velocity(positions)
                v_fast = fast.compute_velocity(positions)
                assert v_slow == pytest.approx(v_fast, rel=1e-10), (
                    f"Frame {i}: slow={v_slow}, fast={v_fast}"
                )

                moving_slow = slow.is_target_moving(positions)
                moving_fast = fast.is_target_moving(positions)
                assert moving_slow == moving_fast, f"Frame {i}: mismatch"

    def test_velocity_detects_phase_change(self, trajectory):
        """Velocity tracker correctly identifies moving vs stationary phases."""
        tracker = VelocityTrackerSlow(threshold=1.0, ema_alpha=0.3)

        # Moving phase (frames 100-149): should be moving
        moving_positions = trajectory[100:150]
        assert tracker.is_target_moving(moving_positions) is True

        # Stationary phase (frames 175-199): should be stationary
        stationary_positions = trajectory[175:200]
        assert tracker.is_target_moving(stationary_positions) is False

    def test_thread_pool_saves_all_images(self):
        """Both thread strategies save all images in a simulated loop."""
        n_frames = 30
        items = [(f"/tmp/track_frame_{i:04d}.png", b"\x00" * 128)
                 for i in range(n_frames)]

        results_threads = save_images_threads(items)
        results_pool = save_images_pool(items, max_workers=4)

        paths_threads = set(results_threads)
        paths_pool = set(results_pool)
        expected = {path for path, _ in items}

        assert paths_threads == expected
        assert paths_pool == expected

    def test_full_pipeline(self, trajectory):
        """End-to-end: bbox -> history -> velocity -> image save.

        Verifies the entire optimized pipeline matches the slow pipeline.
        """
        # Slow pipeline
        hc = HistoryCopy(capacity=self.HISTORY_CAP, cols=4)
        vt_slow = VelocityTrackerSlow(threshold=2.0, ema_alpha=0.3)

        # Fast pipeline
        hv = HistoryView(capacity=self.HISTORY_CAP, cols=4)
        vt_fast = VelocityTrackerFast(
            max_history=self.HISTORY_CAP + 1,
            threshold=2.0,
            ema_alpha=0.3,
        )

        slow_moving_flags = []
        fast_moving_flags = []

        for i in range(len(trajectory)):
            x, y = trajectory[i]
            w, h = 40.0, 30.0

            bbox_slow = BboxSlow(x, y, w, h)
            bbox_fast = BboxSlots(x, y, w, h)

            row_slow = np.array([bbox_slow.cx, bbox_slow.cy, w, h])
            row_fast = np.array([bbox_fast.cx, bbox_fast.cy, w, h])

            hc.push(row_slow)
            hv.push(row_fast)

            if hc.count >= 5:
                pos_slow = hc.latest(min(hc.count, self.HISTORY_CAP))[:, :2]
                pos_fast = hv.latest(min(hv.count, self.HISTORY_CAP))[:, :2]

                np.testing.assert_array_equal(pos_slow, pos_fast)

                slow_moving_flags.append(vt_slow.is_target_moving(pos_slow))
                fast_moving_flags.append(vt_fast.is_target_moving(pos_fast))

        assert slow_moving_flags == fast_moving_flags


class TestMemoryComparison:
    """Verify that optimized versions use less memory."""

    def test_slots_use_less_memory(self):
        n = 10_000

        tracemalloc.start()
        slow_boxes = [BboxSlow(float(i), float(i), 10.0, 10.0) for i in range(n)]
        mem_slow, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        fast_boxes = [BboxSlots(float(i), float(i), 10.0, 10.0) for i in range(n)]
        mem_fast, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Slots version should use meaningfully less memory
        assert mem_fast < mem_slow, (
            f"Expected slots ({mem_fast}) < dict ({mem_slow})"
        )
        # Keep references alive until after measurement
        assert len(slow_boxes) == n
        assert len(fast_boxes) == n

    def test_dataclass_slots_use_less_memory(self):
        n = 10_000

        tracemalloc.start()
        slow_boxes = [BboxSlow(float(i), float(i), 10.0, 10.0) for i in range(n)]
        mem_slow, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        dc_boxes = [BboxDataclass(float(i), float(i), 10.0, 10.0) for i in range(n)]
        mem_dc, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert mem_dc < mem_slow, (
            f"Expected dataclass slots ({mem_dc}) < dict ({mem_slow})"
        )
        assert len(slow_boxes) == n
        assert len(dc_boxes) == n

    def test_view_allocates_less_than_copy(self):
        """Views should cause less peak allocation than copies over many calls."""
        cap = 200
        cols = 8  # larger rows to make allocation differences more visible

        hc = HistoryCopy(capacity=cap, cols=cols)
        hv = HistoryView(capacity=cap, cols=cols)
        for i in range(cap):
            row = np.arange(cols, dtype=np.float64) + i
            hc.push(row)
            hv.push(row)

        n_calls = 5000

        # Measure PEAK allocation for copy version
        tracemalloc.start()
        results_copy = []
        for _ in range(n_calls):
            results_copy.append(hc.latest(50))
        _, peak_copy = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del results_copy

        # Measure PEAK allocation for view version
        tracemalloc.start()
        results_view = []
        for _ in range(n_calls):
            results_view.append(hv.latest(50))
        _, peak_view = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del results_view

        # View version should have lower peak allocation
        assert peak_view < peak_copy, (
            f"Expected view peak ({peak_view}) < copy peak ({peak_copy})"
        )
