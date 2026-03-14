"""
Defensive coding patterns from tracker_engine anti-patterns.

tracker_engine contains several Python anti-patterns that cause subtle bugs
or make the code harder to maintain. This module demonstrates each pattern
with a "before" (buggy/fragile) and "after" (correct) version, plus unit
tests that prove the difference.

Patterns covered:
  1. Optional vs hasattr — tracker_engine uses hasattr(self, 'velocity_tracker')
  2. Mutable default argument — score_bboxes() modifies input list in-place
  3. Enum vs string state — string-based state comparisons are typo-prone
  4. clamp_dims bug — kalman_filter.py overwrites w,h with wrong index

Usage:
    python3 defensive_patterns.py
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# Pattern 1: Optional vs hasattr
# ============================================================================

class TrackerBefore:
    """Anti-pattern: uses hasattr to check for optional component.

    tracker_engine does this:
        if hasattr(self, 'velocity_tracker'):
            self.velocity_tracker.update(pos)

    Problem: hasattr silently swallows AttributeError from *any* source,
    including bugs inside __getattr__. It also prevents static type checkers
    from catching missing attributes.
    """

    def __init__(self, use_velocity: bool = False):
        if use_velocity:
            self.velocity_tracker = _VelocityTracker()

    def update(self, position: tuple[float, float]) -> str:
        if hasattr(self, "velocity_tracker"):
            self.velocity_tracker.update(position)
            return "velocity updated"
        return "no velocity tracker"


class TrackerAfter:
    """Fix: declare velocity_tracker as Optional with a default of None.

    Benefits:
      - Type checker knows the attribute always exists
      - IDE autocompletion works
      - No silent exception swallowing
      - Intent is explicit in the class signature
    """

    def __init__(self, use_velocity: bool = False):
        self.velocity_tracker: Optional[_VelocityTracker] = None
        if use_velocity:
            self.velocity_tracker = _VelocityTracker()

    def update(self, position: tuple[float, float]) -> str:
        if self.velocity_tracker is not None:
            self.velocity_tracker.update(position)
            return "velocity updated"
        return "no velocity tracker"


class _VelocityTracker:
    def __init__(self):
        self.positions: list[tuple[float, float]] = []

    def update(self, pos: tuple[float, float]) -> None:
        self.positions.append(pos)


# ============================================================================
# Pattern 2: Mutable default argument
# ============================================================================

def add_item_before(item: str, items: list[str] = []) -> list[str]:
    """Anti-pattern: mutable default argument.

    tracker_engine's score_bboxes() has a similar pattern where a list
    parameter defaults to []. The default list is shared across all calls,
    so items accumulate unexpectedly.
    """
    items.append(item)
    return items


def add_item_after(item: str, items: Optional[list[str]] = None) -> list[str]:
    """Fix: use None as default and create a new list inside.

    This ensures each call without an explicit list gets its own fresh list.
    """
    if items is None:
        items = []
    items.append(item)
    return items


# ============================================================================
# Pattern 3: Enum vs string state
# ============================================================================

class TrackerState(Enum):
    """Proper enum for tracker state machine.

    Using an enum instead of string literals gives:
      - IDE autocompletion
      - Compile-time typo detection
      - Exhaustiveness checking with match/case
      - No silent bugs from "trackng" vs "tracking"
    """
    INITIALIZING = auto()
    TRACKING = auto()
    LOST = auto()
    DELETED = auto()


def process_state_before(state: str) -> str:
    """Anti-pattern: string-based state comparison.

    tracker_engine uses: if state == "tracking": ...
    A typo like "trackng" silently falls through.
    """
    if state == "tracking":
        return "predicting"
    elif state == "lost":
        return "searching"
    elif state == "deleted":
        return "cleanup"
    else:
        return "unknown"


def process_state_after(state: TrackerState) -> str:
    """Fix: enum-based state comparison.

    TrackerState.TRACKNG would be a NameError caught immediately.
    """
    if state == TrackerState.TRACKING:
        return "predicting"
    elif state == TrackerState.LOST:
        return "searching"
    elif state == TrackerState.DELETED:
        return "cleanup"
    elif state == TrackerState.INITIALIZING:
        return "initializing"
    else:
        return "unknown"


# ============================================================================
# Pattern 4: clamp_dims bug from kalman_filter.py
# ============================================================================

CLAMP_DIM_MIN = 1e-4

def clamp_dims_before(mean: "np.ndarray") -> "np.ndarray":
    """Bug from tracker_engine's kalman_filter.py (line 258-262).

    The original code:
        mean[..., 2:4] = np.clip(mean[..., 5], a_min=CLAMP_DIM_MIN, a_max=None)

    This overwrites indices 2 and 3 (w, h) with the scalar at index 5 (s).
    The state vector is [x, y, w, h, vx, vy, vw, vh] so index 5 is vy.
    The intent was to clamp w and h to a minimum value, but instead it
    replaces both with vy.
    """
    result = mean.copy()
    # BUG: assigns index 5 (vy) to indices 2:4 (w, h)
    result[..., 2:4] = np.clip(result[..., 5], a_min=CLAMP_DIM_MIN, a_max=None)
    return result


def clamp_dims_after(mean: "np.ndarray") -> "np.ndarray":
    """Fix: clamp indices 2:4 using their own values, not index 5.

    The corrected code clamps w and h in-place to the minimum value,
    preserving the actual width and height rather than overwriting them.
    """
    result = mean.copy()
    # FIXED: clamp w,h (indices 2:4) using their own values
    result[..., 2:4] = np.clip(result[..., 2:4], a_min=CLAMP_DIM_MIN, a_max=None)
    return result


# ============================================================================
# Unit Tests
# ============================================================================

class TestOptionalVsHasattr(unittest.TestCase):
    """Tests for Pattern 1: Optional vs hasattr."""

    def test_before_with_velocity(self):
        t = TrackerBefore(use_velocity=True)
        self.assertEqual(t.update((1.0, 2.0)), "velocity updated")

    def test_before_without_velocity(self):
        t = TrackerBefore(use_velocity=False)
        self.assertEqual(t.update((1.0, 2.0)), "no velocity tracker")

    def test_after_with_velocity(self):
        t = TrackerAfter(use_velocity=True)
        self.assertEqual(t.update((1.0, 2.0)), "velocity updated")

    def test_after_without_velocity(self):
        t = TrackerAfter(use_velocity=False)
        self.assertEqual(t.update((1.0, 2.0)), "no velocity tracker")

    def test_after_attribute_always_exists(self):
        """The 'after' version always has the attribute, even when None."""
        t = TrackerAfter(use_velocity=False)
        self.assertTrue(hasattr(t, "velocity_tracker"))
        self.assertIsNone(t.velocity_tracker)

    def test_before_attribute_missing(self):
        """The 'before' version lacks the attribute entirely."""
        t = TrackerBefore(use_velocity=False)
        self.assertFalse(hasattr(t, "velocity_tracker"))


class TestMutableDefault(unittest.TestCase):
    """Tests for Pattern 2: mutable default argument."""

    def test_before_accumulates(self):
        """Demonstrate the bug: items persist across calls."""
        # Reset the default list for a clean test
        add_item_before.__defaults__ = ([],)

        result1 = add_item_before("a")
        result2 = add_item_before("b")
        # Bug: result2 contains both "a" and "b" because the default
        # list is shared across calls
        self.assertEqual(result2, ["a", "b"])
        # And result1 is the SAME object, so it changed too
        self.assertIs(result1, result2)

    def test_after_independent(self):
        """Fix: each call gets its own list."""
        result1 = add_item_after("a")
        result2 = add_item_after("b")
        # Each call gets a fresh list
        self.assertEqual(result1, ["a"])
        self.assertEqual(result2, ["b"])
        self.assertIsNot(result1, result2)

    def test_after_explicit_list(self):
        """When an explicit list is provided, it should be used."""
        my_list = ["existing"]
        result = add_item_after("new", items=my_list)
        self.assertEqual(result, ["existing", "new"])
        self.assertIs(result, my_list)


class TestEnumVsString(unittest.TestCase):
    """Tests for Pattern 3: enum vs string state."""

    def test_before_typo_silent(self):
        """String typo silently produces wrong result."""
        # "trackng" is a typo — no error, just wrong behavior
        result = process_state_before("trackng")
        self.assertEqual(result, "unknown")  # silently wrong

    def test_before_correct(self):
        result = process_state_before("tracking")
        self.assertEqual(result, "predicting")

    def test_after_correct(self):
        result = process_state_after(TrackerState.TRACKING)
        self.assertEqual(result, "predicting")

    def test_after_typo_caught(self):
        """Enum typo raises AttributeError immediately."""
        with self.assertRaises(AttributeError):
            # TrackerState.TRACKNG does not exist — immediate error
            _ = TrackerState.TRACKNG  # type: ignore[attr-defined]

    def test_after_all_states(self):
        self.assertEqual(process_state_after(TrackerState.TRACKING), "predicting")
        self.assertEqual(process_state_after(TrackerState.LOST), "searching")
        self.assertEqual(process_state_after(TrackerState.DELETED), "cleanup")
        self.assertEqual(process_state_after(TrackerState.INITIALIZING), "initializing")


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class TestClampDims(unittest.TestCase):
    """Tests for Pattern 4: clamp_dims bug."""

    def test_before_overwrites_wh(self):
        """Bug: w,h get overwritten with vy (index 5)."""
        # State vector: [x, y, w, h, vx, vy, vw, vh]
        mean = np.array([10.0, 20.0, 5.0, 8.0, 1.0, 2.0, 0.5, 0.3])
        result = clamp_dims_before(mean)

        # Bug: w and h are now both equal to vy (2.0), not their original values
        self.assertAlmostEqual(result[2], 2.0)  # was 5.0, now vy
        self.assertAlmostEqual(result[3], 2.0)  # was 8.0, now vy

    def test_after_preserves_wh(self):
        """Fix: w,h are clamped using their own values."""
        mean = np.array([10.0, 20.0, 5.0, 8.0, 1.0, 2.0, 0.5, 0.3])
        result = clamp_dims_after(mean)

        # w and h should remain unchanged (both > CLAMP_DIM_MIN)
        self.assertAlmostEqual(result[2], 5.0)
        self.assertAlmostEqual(result[3], 8.0)

    def test_after_clamps_small_dims(self):
        """Fix correctly clamps very small w,h to minimum."""
        mean = np.array([10.0, 20.0, 1e-6, 1e-6, 1.0, 2.0, 0.5, 0.3])
        result = clamp_dims_after(mean)

        self.assertAlmostEqual(result[2], CLAMP_DIM_MIN)
        self.assertAlmostEqual(result[3], CLAMP_DIM_MIN)

    def test_before_bug_with_negative_vy(self):
        """Bug is especially dangerous when vy is negative."""
        mean = np.array([10.0, 20.0, 5.0, 8.0, 1.0, -3.0, 0.5, 0.3])
        result = clamp_dims_before(mean)

        # vy is -3.0, but clamp makes it CLAMP_DIM_MIN — still wrong values
        self.assertAlmostEqual(result[2], CLAMP_DIM_MIN)  # was 5.0
        self.assertAlmostEqual(result[3], CLAMP_DIM_MIN)  # was 8.0

    def test_batched_state(self):
        """Both functions handle batched state vectors."""
        means = np.array([
            [10.0, 20.0, 5.0, 8.0, 1.0, 2.0, 0.5, 0.3],
            [15.0, 25.0, 1e-6, 3.0, 0.0, 0.0, 0.0, 0.0],
        ])
        result = clamp_dims_after(means)
        self.assertAlmostEqual(result[0, 2], 5.0)
        self.assertAlmostEqual(result[0, 3], 8.0)
        self.assertAlmostEqual(result[1, 2], CLAMP_DIM_MIN)
        self.assertAlmostEqual(result[1, 3], 3.0)


# ============================================================================
# Demo runner
# ============================================================================

def demo():
    """Print demonstrations of each anti-pattern and its fix."""

    print("=" * 70)
    print("  Defensive Coding Patterns from tracker_engine")
    print("=" * 70)

    # --- Pattern 1 ---
    print("\n--- Pattern 1: Optional vs hasattr ---\n")
    print("BEFORE (hasattr):")
    t_before = TrackerBefore(use_velocity=False)
    print(f"  hasattr(t, 'velocity_tracker') = {hasattr(t_before, 'velocity_tracker')}")
    print(f"  Attribute is completely absent — invisible to type checkers\n")

    print("AFTER (Optional):")
    t_after = TrackerAfter(use_velocity=False)
    print(f"  t.velocity_tracker = {t_after.velocity_tracker}")
    print(f"  Attribute always exists, type is Optional[VelocityTracker]")
    print(f"  Type checker can verify all access paths")

    # --- Pattern 2 ---
    print("\n--- Pattern 2: Mutable Default Argument ---\n")
    add_item_before.__defaults__ = ([],)  # reset for demo

    print("BEFORE (mutable default list):")
    r1 = add_item_before("detection_1")
    print(f"  Call 1: add_item('detection_1') -> {r1}")
    r2 = add_item_before("detection_2")
    print(f"  Call 2: add_item('detection_2') -> {r2}")
    print(f"  Bug: items accumulated across calls!\n")

    print("AFTER (None default):")
    r1 = add_item_after("detection_1")
    print(f"  Call 1: add_item('detection_1') -> {r1}")
    r2 = add_item_after("detection_2")
    print(f"  Call 2: add_item('detection_2') -> {r2}")
    print(f"  Fix: each call gets a fresh list")

    # --- Pattern 3 ---
    print("\n--- Pattern 3: Enum vs String State ---\n")
    print("BEFORE (string):")
    print(f"  process_state('tracking')  -> {process_state_before('tracking')}")
    print(f"  process_state('trackng')   -> {process_state_before('trackng')}")
    print(f"  Typo silently returns 'unknown'\n")

    print("AFTER (enum):")
    print(f"  process_state(TrackerState.TRACKING) -> {process_state_after(TrackerState.TRACKING)}")
    print(f"  TrackerState.TRACKNG -> AttributeError (caught at write time)")

    # --- Pattern 4 ---
    print("\n--- Pattern 4: clamp_dims Bug ---\n")
    if HAS_NUMPY:
        mean = np.array([10.0, 20.0, 5.0, 8.0, 1.0, 2.0, 0.5, 0.3])
        print(f"  State vector: [x, y, w, h, vx, vy, vw, vh]")
        print(f"  Input:  w={mean[2]}, h={mean[3]}, vy={mean[5]}")

        buggy = clamp_dims_before(mean)
        print(f"\n  BEFORE (buggy):")
        print(f"    mean[2:4] = clip(mean[5]) -> w={buggy[2]}, h={buggy[3]}")
        print(f"    Both w and h became vy ({mean[5]})!")

        fixed = clamp_dims_after(mean)
        print(f"\n  AFTER (fixed):")
        print(f"    mean[2:4] = clip(mean[2:4]) -> w={fixed[2]}, h={fixed[3]}")
        print(f"    w and h preserved correctly")
    else:
        print("  [SKIP] numpy not available — cannot demonstrate clamp_dims bug")

    # --- Run tests ---
    print(f"\n{'=' * 70}")
    print("  Running unit tests...")
    print(f"{'=' * 70}\n")

    # Reset mutable default before tests
    add_item_before.__defaults__ = ([],)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestOptionalVsHasattr))
    suite.addTests(loader.loadTestsFromTestCase(TestMutableDefault))
    suite.addTests(loader.loadTestsFromTestCase(TestEnumVsString))
    suite.addTests(loader.loadTestsFromTestCase(TestClampDims))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'=' * 70}")
    if result.wasSuccessful():
        print("  All tests passed.")
    else:
        print(f"  {len(result.failures)} failures, {len(result.errors)} errors.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    demo()
