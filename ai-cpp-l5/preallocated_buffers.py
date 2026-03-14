"""
Lesson 5: Pre-allocated buffers

Two implementations of a velocity tracker:
  - VelocityTrackerSlow: allocates 7+ temp arrays every call (tracker_engine)
  - VelocityTrackerFast: pre-allocates all buffers in __init__, reuses them

Both compute the same result: an EMA-weighted velocity from a position history,
then decide whether the target is moving (velocity > threshold).
"""

from __future__ import annotations

import numpy as np


class VelocityTrackerSlow:
    """tracker_engine style: creates temporary arrays on every call.

    Each call to is_target_moving() allocates ~7 numpy arrays.
    At 30 fps this is 210+ allocations/second just for velocity estimation.
    """

    def __init__(self, threshold: float = 2.0, ema_alpha: float = 0.3) -> None:
        self.threshold = threshold
        self.ema_alpha = ema_alpha

    def compute_velocity(self, positions: np.ndarray) -> float:
        """Compute EMA-weighted average velocity from position history.

        Args:
            positions: (N, 2) array of (x, y) positions, oldest first.

        Returns:
            Scalar velocity value.
        """
        if len(positions) < 2:
            return 0.0

        # alloc 1: frame-to-frame differences
        diffs = np.diff(positions, axis=0)

        # alloc 2: per-step distances
        distances = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)

        # alloc 3: EMA weights (exponentially increasing toward recent)
        n = len(distances)
        indices = np.arange(n, dtype=np.float64)

        # alloc 4: weight values
        weights = (1.0 - self.ema_alpha) ** (n - 1 - indices)

        # alloc 5: weighted distances
        weighted = distances * weights

        # alloc 6: cumulative sum for running EMA
        cumsum = np.cumsum(weighted)

        # alloc 7: cumulative weight sum
        weight_sum = np.cumsum(weights)

        velocity = cumsum[-1] / weight_sum[-1]
        return float(velocity)

    def is_target_moving(self, positions: np.ndarray) -> bool:
        return self.compute_velocity(positions) > self.threshold


class VelocityTrackerFast:
    """Pre-allocated version: zero allocations per call after __init__.

    All temporary buffers are created once and reused via numpy's out= parameter
    and in-place operations.
    """

    def __init__(
        self,
        max_history: int = 100,
        threshold: float = 2.0,
        ema_alpha: float = 0.3,
    ) -> None:
        self.threshold = threshold
        self.ema_alpha = ema_alpha
        self.max_history = max_history

        # Pre-allocate all working buffers
        self._diffs = np.empty((max_history - 1, 2), dtype=np.float64)
        self._dx_sq = np.empty(max_history - 1, dtype=np.float64)
        self._dy_sq = np.empty(max_history - 1, dtype=np.float64)
        self._distances = np.empty(max_history - 1, dtype=np.float64)
        self._indices = np.empty(max_history - 1, dtype=np.float64)
        self._weights = np.empty(max_history - 1, dtype=np.float64)
        self._weighted = np.empty(max_history - 1, dtype=np.float64)
        self._cumsum = np.empty(max_history - 1, dtype=np.float64)
        self._weight_cumsum = np.empty(max_history - 1, dtype=np.float64)

    def compute_velocity(self, positions: np.ndarray) -> float:
        """Same computation as VelocityTrackerSlow, zero allocations."""
        if len(positions) < 2:
            return 0.0

        n = len(positions) - 1  # number of differences

        # Compute diffs in-place
        diffs = self._diffs[:n]
        np.subtract(positions[1 : n + 1], positions[:n], out=diffs)

        # Compute distances = sqrt(dx^2 + dy^2) in-place
        dx_sq = self._dx_sq[:n]
        dy_sq = self._dy_sq[:n]
        distances = self._distances[:n]

        np.multiply(diffs[:, 0], diffs[:, 0], out=dx_sq)
        np.multiply(diffs[:, 1], diffs[:, 1], out=dy_sq)
        np.add(dx_sq, dy_sq, out=distances)
        np.sqrt(distances, out=distances)

        # Compute EMA weights in-place
        indices = self._indices[:n]
        weights = self._weights[:n]

        # Fill indices 0..n-1
        for i in range(n):
            indices[i] = i
        # weights = (1 - alpha) ^ (n - 1 - i)
        np.subtract(n - 1, indices, out=weights)
        base = 1.0 - self.ema_alpha
        np.power(base, weights, out=weights)

        # weighted = distances * weights
        weighted = self._weighted[:n]
        np.multiply(distances, weights, out=weighted)

        # Cumulative sums for final velocity
        cumsum = self._cumsum[:n]
        weight_cumsum = self._weight_cumsum[:n]
        np.cumsum(weighted, out=cumsum)
        np.cumsum(weights, out=weight_cumsum)

        velocity = cumsum[-1] / weight_cumsum[-1]
        return float(velocity)

    def is_target_moving(self, positions: np.ndarray) -> bool:
        return self.compute_velocity(positions) > self.threshold


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import time
    import tracemalloc

    rng = np.random.default_rng(42)
    positions = np.cumsum(rng.normal(3.0, 0.5, size=(50, 2)), axis=0)

    slow = VelocityTrackerSlow(threshold=2.0, ema_alpha=0.3)
    fast = VelocityTrackerFast(max_history=100, threshold=2.0, ema_alpha=0.3)

    # Verify identical results
    v_slow = slow.compute_velocity(positions)
    v_fast = fast.compute_velocity(positions)
    print(f"Slow velocity: {v_slow:.6f}")
    print(f"Fast velocity: {v_fast:.6f}")
    print(f"Match: {np.isclose(v_slow, v_fast)}")

    # Memory allocation comparison — this is the real win
    iterations = 5_000

    tracemalloc.start()
    for _ in range(iterations):
        slow.compute_velocity(positions)
    _, peak_slow = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    for _ in range(iterations):
        fast.compute_velocity(positions)
    _, peak_fast = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nPeak memory (slow): {peak_slow / 1024:.1f} KB")
    print(f"Peak memory (fast): {peak_fast / 1024:.1f} KB")
    print(f"Memory saved: {(1 - peak_fast / peak_slow) * 100:.0f}%")

    # Timing
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        slow.compute_velocity(positions)
    t_slow = (time.perf_counter_ns() - t0) / iterations

    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        fast.compute_velocity(positions)
    t_fast = (time.perf_counter_ns() - t0) / iterations

    print(f"\nSlow: {t_slow:.0f} ns/call")
    print(f"Fast: {t_fast:.0f} ns/call")

    # Note: numpy's out= parameter has overhead that may make it slower for
    # small arrays. The real value of pre-allocation is:
    # 1. Reduced GC pressure (fewer temporary objects)
    # 2. Predictable memory usage (no allocation spikes)
    # 3. Better cache behavior over sustained runs
    # For maximum speedup, use C++ with pre-allocated buffers (see L4).


if __name__ == "__main__":
    _demo()
