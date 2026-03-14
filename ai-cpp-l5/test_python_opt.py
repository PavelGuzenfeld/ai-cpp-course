"""
Lesson 5: Unit tests for all optimization variants.

Tests:
  - bbox variants: construction, properties, iou correctness
  - history variants: push, latest, wrap-around
  - velocity tracker variants: same output from both
  - thread pool: all images saved correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from bbox_slots import BboxSlow, BboxSlots, BboxDataclass
from numpy_views import HistoryCopy, HistoryView
from preallocated_buffers import VelocityTrackerSlow, VelocityTrackerFast
from thread_pool_io import save_images_threads, save_images_pool


# ===================================================================
# Bbox tests
# ===================================================================

BBOX_CLASSES = [BboxSlow, BboxSlots, BboxDataclass]


@pytest.mark.parametrize("cls", BBOX_CLASSES, ids=lambda c: c.__name__)
class TestBboxConstruction:
    def test_init(self, cls):
        b = cls(10.0, 20.0, 100.0, 50.0)
        assert b.x == 10.0
        assert b.y == 20.0
        assert b.w == 100.0
        assert b.h == 50.0

    def test_center(self, cls):
        b = cls(10.0, 20.0, 100.0, 50.0)
        assert b.cx == pytest.approx(60.0)
        assert b.cy == pytest.approx(45.0)

    def test_area(self, cls):
        b = cls(0.0, 0.0, 10.0, 20.0)
        assert b.area == pytest.approx(200.0)

    def test_area_zero(self, cls):
        b = cls(5.0, 5.0, 0.0, 10.0)
        assert b.area == pytest.approx(0.0)


@pytest.mark.parametrize("cls", BBOX_CLASSES, ids=lambda c: c.__name__)
class TestBboxIou:
    def test_identical(self, cls):
        b = cls(0.0, 0.0, 10.0, 10.0)
        assert b.iou(b) == pytest.approx(1.0)

    def test_no_overlap(self, cls):
        a = cls(0.0, 0.0, 10.0, 10.0)
        b = cls(20.0, 20.0, 10.0, 10.0)
        assert a.iou(b) == pytest.approx(0.0)

    def test_partial_overlap(self, cls):
        a = cls(0.0, 0.0, 10.0, 10.0)
        b = cls(5.0, 5.0, 10.0, 10.0)
        # Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        assert a.iou(b) == pytest.approx(25.0 / 175.0)

    def test_contained(self, cls):
        a = cls(0.0, 0.0, 20.0, 20.0)
        b = cls(5.0, 5.0, 5.0, 5.0)
        # Intersection = area(b) = 25, Union = 400 + 25 - 25 = 400
        assert a.iou(b) == pytest.approx(25.0 / 400.0)

    def test_symmetric(self, cls):
        a = cls(0.0, 0.0, 10.0, 10.0)
        b = cls(3.0, 4.0, 12.0, 8.0)
        assert a.iou(b) == pytest.approx(b.iou(a))

    def test_touching_edge(self, cls):
        a = cls(0.0, 0.0, 10.0, 10.0)
        b = cls(10.0, 0.0, 10.0, 10.0)
        assert a.iou(b) == pytest.approx(0.0)


class TestBboxSlots:
    def test_no_dict_on_slots(self):
        b = BboxSlots(1.0, 2.0, 3.0, 4.0)
        assert not hasattr(b, "__dict__")

    def test_no_dict_on_dataclass(self):
        b = BboxDataclass(1.0, 2.0, 3.0, 4.0)
        assert not hasattr(b, "__dict__")

    def test_has_dict_on_slow(self):
        b = BboxSlow(1.0, 2.0, 3.0, 4.0)
        assert hasattr(b, "__dict__")


# ===================================================================
# History buffer tests
# ===================================================================

HISTORY_CLASSES = [HistoryCopy, HistoryView]


@pytest.mark.parametrize("cls", HISTORY_CLASSES, ids=lambda c: c.__name__)
class TestHistoryBasic:
    def test_empty(self, cls):
        h = cls(capacity=10, cols=4)
        result = h.latest(5)
        assert result.shape == (0, 4)

    def test_push_and_latest(self, cls):
        h = cls(capacity=10, cols=2)
        h.push(np.array([1.0, 2.0]))
        h.push(np.array([3.0, 4.0]))
        h.push(np.array([5.0, 6.0]))

        result = h.latest(2)
        expected = np.array([[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_latest_all(self, cls):
        h = cls(capacity=10, cols=2)
        for i in range(5):
            h.push(np.array([float(i), float(i * 10)]))
        result = h.latest()
        assert result.shape == (5, 2)

    def test_latest_more_than_available(self, cls):
        h = cls(capacity=10, cols=2)
        h.push(np.array([1.0, 2.0]))
        result = h.latest(5)
        assert result.shape == (1, 2)

    def test_count(self, cls):
        h = cls(capacity=5, cols=2)
        assert h.count == 0
        for i in range(3):
            h.push(np.array([float(i), 0.0]))
        assert h.count == 3

    def test_count_capped(self, cls):
        h = cls(capacity=5, cols=2)
        for i in range(10):
            h.push(np.array([float(i), 0.0]))
        assert h.count == 5


@pytest.mark.parametrize("cls", HISTORY_CLASSES, ids=lambda c: c.__name__)
class TestHistoryWrapAround:
    def test_wrap_around_values(self, cls):
        h = cls(capacity=4, cols=1)
        for i in range(6):  # wraps at 4
            h.push(np.array([float(i)]))

        # Should return last 3: [3, 4, 5]
        result = h.latest(3)
        expected = np.array([[3.0], [4.0], [5.0]])
        np.testing.assert_array_equal(result, expected)

    def test_wrap_around_full_capacity(self, cls):
        h = cls(capacity=4, cols=1)
        for i in range(6):
            h.push(np.array([float(i)]))

        result = h.latest(4)
        expected = np.array([[2.0], [3.0], [4.0], [5.0]])
        np.testing.assert_array_equal(result, expected)


class TestHistoryMemorySharing:
    def test_copy_does_not_share(self):
        h = HistoryCopy(capacity=10, cols=2)
        for i in range(5):
            h.push(np.array([float(i), 0.0]))
        result = h.latest(3)
        assert not np.shares_memory(h._buf, result)

    def test_view_shares_when_contiguous(self):
        h = HistoryView(capacity=10, cols=2)
        for i in range(5):
            h.push(np.array([float(i), 0.0]))
        result = h.latest(3)
        assert np.shares_memory(h._buf, result)


# ===================================================================
# Velocity tracker tests
# ===================================================================

class TestVelocityTracker:
    @pytest.fixture
    def positions(self):
        rng = np.random.default_rng(42)
        return np.cumsum(rng.normal(3.0, 0.5, size=(30, 2)), axis=0)

    @pytest.fixture
    def stationary_positions(self):
        # Tiny jitter around a fixed point
        rng = np.random.default_rng(42)
        return 100.0 + rng.normal(0.0, 0.01, size=(30, 2))

    def test_same_output(self, positions):
        slow = VelocityTrackerSlow(threshold=2.0, ema_alpha=0.3)
        fast = VelocityTrackerFast(max_history=100, threshold=2.0, ema_alpha=0.3)

        v_slow = slow.compute_velocity(positions)
        v_fast = fast.compute_velocity(positions)

        assert v_slow == pytest.approx(v_fast, rel=1e-10)

    def test_moving_target(self, positions):
        slow = VelocityTrackerSlow(threshold=2.0, ema_alpha=0.3)
        fast = VelocityTrackerFast(max_history=100, threshold=2.0, ema_alpha=0.3)

        assert slow.is_target_moving(positions) is True
        assert fast.is_target_moving(positions) is True

    def test_stationary_target(self, stationary_positions):
        slow = VelocityTrackerSlow(threshold=2.0, ema_alpha=0.3)
        fast = VelocityTrackerFast(max_history=100, threshold=2.0, ema_alpha=0.3)

        assert slow.is_target_moving(stationary_positions) is False
        assert fast.is_target_moving(stationary_positions) is False

    def test_too_few_positions(self):
        slow = VelocityTrackerSlow()
        fast = VelocityTrackerFast()

        single = np.array([[1.0, 2.0]])
        assert slow.compute_velocity(single) == 0.0
        assert fast.compute_velocity(single) == 0.0

    def test_empty_positions(self):
        slow = VelocityTrackerSlow()
        fast = VelocityTrackerFast()

        empty = np.empty((0, 2))
        assert slow.compute_velocity(empty) == 0.0
        assert fast.compute_velocity(empty) == 0.0

    def test_two_positions(self):
        slow = VelocityTrackerSlow(ema_alpha=0.3)
        fast = VelocityTrackerFast(ema_alpha=0.3)

        pts = np.array([[0.0, 0.0], [3.0, 4.0]])  # distance = 5.0
        assert slow.compute_velocity(pts) == pytest.approx(5.0)
        assert fast.compute_velocity(pts) == pytest.approx(5.0)


# ===================================================================
# Thread pool tests
# ===================================================================

class TestThreadPool:
    def _make_items(self, n: int) -> list[tuple[str, bytes]]:
        return [(f"/tmp/test_{i}.png", b"\x00" * 64) for i in range(n)]

    def test_threads_all_saved(self):
        items = self._make_items(20)
        results = save_images_threads(items)
        assert len(results) == 20

    def test_pool_all_saved(self):
        items = self._make_items(20)
        results = save_images_pool(items, max_workers=4)
        assert len(results) == 20

    def test_threads_correct_paths(self):
        items = self._make_items(10)
        results = save_images_threads(items)
        expected_paths = {path for path, _ in items}
        assert set(results) == expected_paths

    def test_pool_correct_paths(self):
        items = self._make_items(10)
        results = save_images_pool(items, max_workers=4)
        expected_paths = {path for path, _ in items}
        assert set(results) == expected_paths

    def test_pool_single_worker(self):
        items = self._make_items(5)
        results = save_images_pool(items, max_workers=1)
        assert len(results) == 5

    def test_empty_items(self):
        results_t = save_images_threads([])
        results_p = save_images_pool([], max_workers=4)
        assert results_t == []
        assert results_p == []
