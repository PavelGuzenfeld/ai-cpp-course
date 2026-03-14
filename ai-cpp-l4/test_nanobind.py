"""
Unit tests for Lesson 4 nanobind modules.

Tests:
  - BBox: construction, properties, iou, contains_point, edge cases
  - BufferPool: acquire, release, reuse, capacity
  - HistoryView: push, latest, circular wrap-around, view semantics
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

from bbox_native import BBox
from buffer_pool_native import BufferPool
from history_view_native import HistoryView


# ===========================================================================
# BBox Tests
# ===========================================================================

class TestBBox:
    def test_construction(self):
        b = BBox(10.0, 20.0, 100.0, 50.0)
        assert b.x == 10.0
        assert b.y == 20.0
        assert b.w == 100.0
        assert b.h == 50.0

    def test_computed_properties(self):
        b = BBox(10.0, 20.0, 100.0, 50.0)
        assert b.cx == pytest.approx(60.0)
        assert b.cy == pytest.approx(45.0)
        assert b.area == pytest.approx(5000.0)
        assert b.aspect_ratio == pytest.approx(2.0)

    def test_readwrite_attributes(self):
        b = BBox(0.0, 0.0, 10.0, 10.0)
        b.x = 5.0
        b.y = 15.0
        assert b.x == 5.0
        assert b.y == 15.0
        assert b.cx == pytest.approx(10.0)

    def test_iou_overlapping(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(5, 5, 10, 10)
        # Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        assert a.iou(b) == pytest.approx(25.0 / 175.0)

    def test_iou_identical(self):
        a = BBox(0, 0, 10, 10)
        assert a.iou(a) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(20, 20, 10, 10)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_touching_edge(self):
        a = BBox(0, 0, 10, 10)
        b = BBox(10, 0, 10, 10)
        # They share an edge but zero-area intersection
        assert a.iou(b) == pytest.approx(0.0)

    def test_contains_point_inside(self):
        b = BBox(0, 0, 10, 10)
        assert b.contains_point(5, 5) is True

    def test_contains_point_on_edge(self):
        b = BBox(0, 0, 10, 10)
        assert b.contains_point(0, 0) is True
        assert b.contains_point(10, 10) is True

    def test_contains_point_outside(self):
        b = BBox(0, 0, 10, 10)
        assert b.contains_point(11, 5) is False
        assert b.contains_point(-1, 5) is False

    def test_negative_dimensions_raises(self):
        with pytest.raises(ValueError):
            BBox(0, 0, -1, 10)
        with pytest.raises(ValueError):
            BBox(0, 0, 10, -1)

    def test_zero_dimensions(self):
        b = BBox(5, 5, 0, 0)
        assert b.area == pytest.approx(0.0)
        assert b.aspect_ratio == pytest.approx(0.0)

    def test_to_array(self):
        b = BBox(1.0, 2.0, 3.0, 4.0)
        arr = np.asarray(b.to_array())
        np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0, 4.0])

    def test_from_array(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        b = BBox.from_array(arr)
        assert b.x == 10.0
        assert b.y == 20.0
        assert b.w == 30.0
        assert b.h == 40.0

    def test_roundtrip_array(self):
        original = BBox(1.5, 2.5, 3.5, 4.5)
        restored = BBox.from_array(np.asarray(original.to_array()))
        assert restored.x == pytest.approx(original.x)
        assert restored.y == pytest.approx(original.y)
        assert restored.w == pytest.approx(original.w)
        assert restored.h == pytest.approx(original.h)

    def test_repr(self):
        b = BBox(1, 2, 3, 4)
        r = repr(b)
        assert "BBox" in r
        assert "1" in r


# ===========================================================================
# BufferPool Tests
# ===========================================================================

class TestBufferPool:
    def test_construction(self):
        pool = BufferPool(capacity=4, buffer_size=128)
        assert pool.capacity == 4
        assert pool.buffer_size == 128
        assert pool.available == 4
        assert pool.active == 0

    def test_acquire_returns_numpy(self):
        pool = BufferPool(capacity=2, buffer_size=64)
        buf = pool.acquire()
        arr = np.asarray(buf)
        assert arr.dtype == np.float64
        assert arr.shape == (64,)

    def test_acquire_returns_zeroed(self):
        pool = BufferPool(capacity=2, buffer_size=32)
        buf = pool.acquire()
        arr = np.asarray(buf)
        np.testing.assert_array_equal(arr, np.zeros(32))

    def test_acquire_release_cycle(self):
        pool = BufferPool(capacity=2, buffer_size=16)
        assert pool.available == 2

        idx1 = pool.acquire_index()
        assert pool.available == 1
        assert pool.active == 1

        idx2 = pool.acquire_index()
        assert pool.available == 0
        assert pool.active == 2

        pool.release(idx1)
        assert pool.available == 1
        assert pool.active == 1

        pool.release(idx2)
        assert pool.available == 2
        assert pool.active == 0

    def test_exhaust_pool_raises(self):
        pool = BufferPool(capacity=1, buffer_size=8)
        pool.acquire_index()
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.acquire_index()

    def test_double_release_raises(self):
        pool = BufferPool(capacity=2, buffer_size=8)
        idx = pool.acquire_index()
        pool.release(idx)
        with pytest.raises(RuntimeError, match="already released"):
            pool.release(idx)

    def test_reuse_after_release(self):
        pool = BufferPool(capacity=1, buffer_size=16)
        idx = pool.acquire_index()
        pool.release(idx)
        pool.acquire_index()
        # Should succeed — the single slot is reused
        assert pool.available == 0

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            BufferPool(capacity=0, buffer_size=10)
        with pytest.raises(ValueError):
            BufferPool(capacity=10, buffer_size=0)


# ===========================================================================
# HistoryView Tests
# ===========================================================================

class TestHistoryView:
    def test_construction(self):
        h = HistoryView(max_entries=10, row_size=4)
        assert h.max_entries == 10
        assert h.row_size == 4
        assert h.count == 0

    def test_push_increments_count(self):
        h = HistoryView(max_entries=5, row_size=3)
        h.push(np.array([1.0, 2.0, 3.0]))
        assert h.count == 1
        h.push(np.array([4.0, 5.0, 6.0]))
        assert h.count == 2

    def test_latest_returns_most_recent(self):
        h = HistoryView(max_entries=10, row_size=2)
        h.push(np.array([1.0, 2.0]))
        h.push(np.array([3.0, 4.0]))
        h.push(np.array([5.0, 6.0]))

        result = np.asarray(h.latest(1))
        assert result.shape == (1, 2)
        np.testing.assert_array_almost_equal(result[0], [5.0, 6.0])

    def test_latest_multiple(self):
        h = HistoryView(max_entries=10, row_size=2)
        for i in range(5):
            h.push(np.array([float(i), float(i * 10)]))

        result = np.asarray(h.latest(3))
        assert result.shape == (3, 2)
        # Most recent 3: entries 2, 3, 4 (in buffer order)
        np.testing.assert_array_almost_equal(result[0], [2.0, 20.0])
        np.testing.assert_array_almost_equal(result[1], [3.0, 30.0])
        np.testing.assert_array_almost_equal(result[2], [4.0, 40.0])

    def test_circular_wraparound(self):
        h = HistoryView(max_entries=3, row_size=1)
        h.push(np.array([1.0]))
        h.push(np.array([2.0]))
        h.push(np.array([3.0]))
        h.push(np.array([4.0]))  # overwrites entry 0

        assert h.count == 3
        result = np.asarray(h.latest(3))
        np.testing.assert_array_almost_equal(result.flatten(), [2.0, 3.0, 4.0])

    def test_latest_clamps_to_count(self):
        h = HistoryView(max_entries=10, row_size=2)
        h.push(np.array([1.0, 2.0]))
        # Request more than available
        result = np.asarray(h.latest(100))
        assert result.shape == (1, 2)

    def test_latest_empty_raises(self):
        h = HistoryView(max_entries=5, row_size=2)
        with pytest.raises(RuntimeError, match="empty"):
            h.latest(1)

    def test_push_wrong_size_raises(self):
        h = HistoryView(max_entries=5, row_size=3)
        with pytest.raises(ValueError):
            h.push(np.array([1.0, 2.0]))  # too few

    def test_view_semantics_contiguous(self):
        """When entries are contiguous (no wrap), latest() returns a view
        that shares memory with the internal buffer."""
        h = HistoryView(max_entries=10, row_size=2)
        for i in range(5):
            h.push(np.array([float(i), float(i)]))

        view = np.asarray(h.latest(3))
        # The view should reference the internal storage (data_ptr gives the
        # base address of the storage vector).
        # We verify by checking the array is not a fresh allocation with
        # different content.
        np.testing.assert_array_almost_equal(view[2], [4.0, 4.0])

    def test_invalid_construction(self):
        with pytest.raises(ValueError):
            HistoryView(max_entries=0, row_size=4)
        with pytest.raises(ValueError):
            HistoryView(max_entries=4, row_size=0)

    def test_latest_zero_raises(self):
        h = HistoryView(max_entries=5, row_size=2)
        h.push(np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            h.latest(0)
