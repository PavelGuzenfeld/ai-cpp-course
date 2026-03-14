"""
Integration tests for Lesson 4 nanobind modules.

Tests real-world usage patterns:
  - BoundingBox in a simulated tracking loop
  - BufferPool feeding a processing pipeline
  - HistoryView with concurrent-style push/read patterns
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

from bbox_native import BBox
from buffer_pool_native import BufferPool
from history_view_native import HistoryView


# ===========================================================================
# BBox Integration Tests — Simulated Tracking Loop
# ===========================================================================

class TestBBoxTrackingLoop:
    def test_track_and_update(self):
        """Simulate a tracking loop: create detections, update positions,
        compute IOU with previous frame."""
        num_frames = 50
        num_objects = 20

        # Initial detections
        prev_boxes = [
            BBox(x=float(i * 30), y=float(i * 20), w=50.0, h=40.0)
            for i in range(num_objects)
        ]

        for frame in range(num_frames):
            # Simulate new detections with slight movement
            curr_boxes = []
            for i, pb in enumerate(prev_boxes):
                dx = float((frame + i) % 5) - 2.0
                dy = float((frame + i) % 3) - 1.0
                cb = BBox(pb.x + dx, pb.y + dy, pb.w, pb.h)
                curr_boxes.append(cb)

            # Compute IOU between consecutive frames
            for prev, curr in zip(prev_boxes, curr_boxes):
                iou = prev.iou(curr)
                # With small movements, IOU should be high
                assert iou > 0.5, f"IOU too low: {iou}"

            # Access properties (simulating rendering/logging)
            for b in curr_boxes:
                _ = b.cx
                _ = b.cy
                _ = b.area
                assert b.area > 0

            prev_boxes = curr_boxes

    def test_nms_simulation(self):
        """Simulate non-maximum suppression: sort by area, suppress overlapping."""
        boxes = [
            BBox(10, 10, 100, 80),
            BBox(15, 12, 95, 78),   # overlaps with first
            BBox(200, 200, 50, 50), # far away
            BBox(205, 195, 55, 55), # overlaps with third
        ]

        iou_threshold = 0.3
        keep = []

        # Simple greedy NMS
        sorted_boxes = sorted(boxes, key=lambda b: b.area, reverse=True)
        suppressed = [False] * len(sorted_boxes)

        for i, box_i in enumerate(sorted_boxes):
            if suppressed[i]:
                continue
            keep.append(box_i)
            for j in range(i + 1, len(sorted_boxes)):
                if not suppressed[j] and box_i.iou(sorted_boxes[j]) > iou_threshold:
                    suppressed[j] = True

        # Should keep at least 2 (the two clusters)
        assert len(keep) >= 2
        # Should suppress at least 1 overlapping box
        assert len(keep) < len(boxes)

    def test_bbox_array_roundtrip_batch(self):
        """Convert many bboxes to arrays and back."""
        originals = [BBox(i * 10.0, i * 5.0, 30.0 + i, 20.0 + i) for i in range(100)]

        # Batch to_array
        arrays = [np.asarray(b.to_array()) for b in originals]

        # Stack into a single numpy array
        stacked = np.stack(arrays)
        assert stacked.shape == (100, 4)

        # Reconstruct
        restored = [BBox.from_array(stacked[i]) for i in range(100)]

        for orig, rest in zip(originals, restored):
            assert rest.x == pytest.approx(orig.x)
            assert rest.y == pytest.approx(orig.y)
            assert rest.w == pytest.approx(orig.w)
            assert rest.h == pytest.approx(orig.h)


# ===========================================================================
# BufferPool Integration Tests — Processing Pipeline
# ===========================================================================

class TestBufferPoolPipeline:
    def test_pipeline_acquire_process_release(self):
        """Simulate a pipeline: acquire buffer, fill with data, process, release."""
        pool = BufferPool(capacity=4, buffer_size=256)

        for iteration in range(20):
            # Acquire
            idx = pool.acquire_index()
            buf = pool.get_buffer(idx)
            arr = np.asarray(buf)

            # Fill with "sensor data"
            arr[:] = np.sin(np.linspace(0, 2 * np.pi, 256)) * iteration

            # "Process" — compute statistics
            mean_val = np.mean(arr)
            max_val = np.max(arr)

            # Release
            pool.release(idx)

            assert pool.available == 4
            assert pool.active == 0

    def test_multiple_buffers_in_flight(self):
        """Simulate having multiple buffers active simultaneously."""
        pool = BufferPool(capacity=4, buffer_size=64)
        active_indices = []

        # Acquire all 4 buffers
        for i in range(4):
            idx = pool.acquire_index()
            active_indices.append(idx)

        assert pool.available == 0
        assert pool.active == 4

        # Pool should be exhausted
        with pytest.raises(RuntimeError):
            pool.acquire_index()

        # Release in reverse order
        for idx in reversed(active_indices):
            pool.release(idx)

        assert pool.available == 4
        assert pool.active == 0

        assert pool.available == 4
        assert pool.active == 0

    def test_rapid_acquire_release(self):
        """Stress test: rapid acquire/release cycles."""
        pool = BufferPool(capacity=2, buffer_size=1024)

        for _ in range(10_000):
            idx = pool.acquire_index()
            pool.release(idx)

        assert pool.available == 2
        assert pool.active == 0

    def test_buffer_zeroed_on_reacquire(self):
        """Verify buffers are zeroed when re-acquired."""
        pool = BufferPool(capacity=1, buffer_size=16)

        # Acquire and write data
        idx = pool.acquire_index()
        buf = pool.get_buffer(idx)
        arr = np.asarray(buf)
        arr[:] = 42.0
        pool.release(idx)

        # Re-acquire: should be zeroed
        idx2 = pool.acquire_index()
        buf2 = pool.get_buffer(idx2)
        arr2 = np.asarray(buf2)
        np.testing.assert_array_equal(arr2, np.zeros(16))


# ===========================================================================
# HistoryView Integration Tests — Tracking History
# ===========================================================================

class TestHistoryViewTracking:
    def test_tracking_history_accumulation(self):
        """Simulate accumulating object positions over time."""
        h = HistoryView(max_entries=100, row_size=4)  # [x, y, w, h]

        # Simulate 200 frames (wraps around the 100-entry buffer)
        for frame in range(200):
            x = 100.0 + frame * 0.5
            y = 200.0 + frame * 0.3
            w = 50.0
            ht = 40.0
            h.push(np.array([x, y, w, ht]))

        assert h.count == 100

        # Get the last 10 positions
        recent = np.asarray(h.latest(10))
        assert recent.shape == (10, 4)

        # Verify ordering: most recent should have highest x
        # The last push was frame=199: x = 100 + 199*0.5 = 199.5
        np.testing.assert_almost_equal(recent[-1, 0], 199.5)
        np.testing.assert_almost_equal(recent[-2, 0], 199.0)

    def test_velocity_computation_from_history(self):
        """Use history to compute velocity (diff of consecutive positions)."""
        h = HistoryView(max_entries=50, row_size=2)  # [x, y]

        # Object moving at constant velocity
        vx, vy = 3.0, -2.0
        for t in range(30):
            h.push(np.array([100.0 + vx * t, 200.0 + vy * t]))

        # Get last 5 positions and compute velocities
        positions = np.asarray(h.latest(5))
        velocities = np.diff(positions, axis=0)

        # All velocities should be [vx, vy]
        for vel in velocities:
            np.testing.assert_almost_equal(vel[0], vx)
            np.testing.assert_almost_equal(vel[1], vy)

    def test_wraparound_data_integrity(self):
        """Verify data integrity when the circular buffer wraps around multiple times."""
        h = HistoryView(max_entries=5, row_size=1)

        # Push 13 values: wraps around 2.6 times
        for i in range(13):
            h.push(np.array([float(i)]))

        assert h.count == 5

        # Last 5 should be [8, 9, 10, 11, 12]
        result = np.asarray(h.latest(5)).flatten()
        np.testing.assert_array_almost_equal(result, [8.0, 9.0, 10.0, 11.0, 12.0])

    def test_interleaved_push_and_read(self):
        """Simulate concurrent-style pattern: push then read repeatedly."""
        h = HistoryView(max_entries=20, row_size=3)

        for i in range(100):
            h.push(np.array([float(i), float(i * 2), float(i * 3)]))

            # Read after every push
            if i > 0:
                recent = np.asarray(h.latest(min(i + 1, 20)))
                # Most recent entry should match what we just pushed
                np.testing.assert_almost_equal(recent[-1, 0], float(i))
                np.testing.assert_almost_equal(recent[-1, 1], float(i * 2))
                np.testing.assert_almost_equal(recent[-1, 2], float(i * 3))

    def test_full_history_retrieval(self):
        """Fill buffer exactly and retrieve all entries."""
        h = HistoryView(max_entries=10, row_size=2)

        for i in range(10):
            h.push(np.array([float(i), float(i + 100)]))

        result = np.asarray(h.latest(10))
        assert result.shape == (10, 2)

        for i in range(10):
            np.testing.assert_almost_equal(result[i, 0], float(i))
            np.testing.assert_almost_equal(result[i, 1], float(i + 100))
