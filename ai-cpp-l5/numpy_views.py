"""
Lesson 5: Numpy views vs. copies

Two implementations of a circular history buffer:
  - HistoryCopy: returns .copy() on latest()  (tracker_engine style)
  - HistoryView: returns a view (slice) on latest()

Demonstrates np.shares_memory() and when views are unsafe.
"""

from __future__ import annotations

import numpy as np


class HistoryCopy:
    """Circular buffer that always returns copies (tracker_engine pattern).

    Every call to latest() allocates a new numpy array, even when the caller
    only needs to read the data for one frame.
    """

    def __init__(self, capacity: int, cols: int = 4) -> None:
        self._buf = np.zeros((capacity, cols), dtype=np.float64)
        self._capacity = capacity
        self._count = 0
        self._idx = 0

    @property
    def count(self) -> int:
        return min(self._count, self._capacity)

    def push(self, row: np.ndarray) -> None:
        """Append a row, wrapping around when full."""
        self._buf[self._idx % self._capacity] = row
        self._idx += 1
        self._count += 1

    def latest(self, n: int | None = None) -> np.ndarray:
        """Return the last n entries.  Always copies."""
        available = self.count
        if n is None or n > available:
            n = available
        if n == 0:
            return np.empty((0, self._buf.shape[1]), dtype=self._buf.dtype)

        end = self._idx % self._capacity
        if end >= n:
            return self._buf[end - n : end].copy()  # <-- copy every time
        else:
            # Wrap-around: must concatenate two slices (copy is inherent)
            part1 = self._buf[self._capacity - (n - end) :]
            part2 = self._buf[:end]
            return np.concatenate([part1, part2])


class HistoryView:
    """Circular buffer that returns views when possible.

    When the requested window does not wrap around, latest() returns a
    zero-copy view into the internal buffer.  This avoids allocation entirely.
    """

    def __init__(self, capacity: int, cols: int = 4) -> None:
        self._buf = np.zeros((capacity, cols), dtype=np.float64)
        self._capacity = capacity
        self._count = 0
        self._idx = 0

    @property
    def count(self) -> int:
        return min(self._count, self._capacity)

    def push(self, row: np.ndarray) -> None:
        self._buf[self._idx % self._capacity] = row
        self._idx += 1
        self._count += 1

    def latest(self, n: int | None = None) -> np.ndarray:
        """Return the last n entries.  Returns a *view* when contiguous."""
        available = self.count
        if n is None or n > available:
            n = available
        if n == 0:
            return np.empty((0, self._buf.shape[1]), dtype=self._buf.dtype)

        end = self._idx % self._capacity
        if end >= n:
            return self._buf[end - n : end]  # <-- view, no copy
        else:
            # Wrap-around: must concatenate (unavoidable copy)
            part1 = self._buf[self._capacity - (n - end) :]
            part2 = self._buf[:end]
            return np.concatenate([part1, part2])

    def latest_copy(self, n: int | None = None) -> np.ndarray:
        """Explicit copy -- use when the caller stores data across frames."""
        return self.latest(n).copy()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    print("=== np.shares_memory() demo ===\n")

    cap = 20
    hv = HistoryView(capacity=cap, cols=4)
    hc = HistoryCopy(capacity=cap, cols=4)

    # Push 10 rows (no wrap-around)
    for i in range(10):
        row = np.array([i, i + 1, i + 2, i + 3], dtype=np.float64)
        hv.push(row)
        hc.push(row)

    view_result = hv.latest(5)
    copy_result = hc.latest(5)

    print(f"HistoryView.latest(5) shares memory with buffer: "
          f"{np.shares_memory(hv._buf, view_result)}")  # True
    print(f"HistoryCopy.latest(5) shares memory with buffer: "
          f"{np.shares_memory(hc._buf, copy_result)}")  # False

    # Demonstrate the danger of views: mutation propagates
    print(f"\nBefore mutation: view_result[0] = {view_result[0]}")
    hv.push(np.array([99, 99, 99, 99], dtype=np.float64))
    # The buffer slot that view_result[0] pointed to may now hold new data
    # if we had wrapped around.  In this case (no wrap), it's still safe.

    print(f"After pushing to buffer: view_result[0] = {view_result[0]}")
    print("  (still safe because no wrap-around overwrite happened)")

    # Force wrap-around to show when views become invalid
    print("\n=== Wrap-around demo ===\n")
    hv2 = HistoryView(capacity=5, cols=2)
    for i in range(4):
        hv2.push(np.array([i, i * 10], dtype=np.float64))

    snapshot = hv2.latest(3)  # view into slots [1,2,3]
    print(f"Snapshot (view) before overwrite: {snapshot.tolist()}")

    # Overwrite slot 1 via wrap-around
    hv2.push(np.array([100, 1000], dtype=np.float64))
    hv2.push(np.array([200, 2000], dtype=np.float64))
    print(f"Snapshot (view) after overwrite:  {snapshot.tolist()}")
    print("  The view now shows corrupted data from the overwrite!")
    print("  Use .copy() when you need to hold data across frames.")


if __name__ == "__main__":
    _demo()
