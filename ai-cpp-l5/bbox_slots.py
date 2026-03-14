"""
Lesson 5: __slots__ -- Eliminate per-instance __dict__

Three implementations of the same BoundingBox interface:
  - BboxSlow:      regular class with __dict__ (tracker_engine style)
  - BboxSlots:     same class with __slots__
  - BboxDataclass: @dataclass(slots=True)  (Python 3.10+)

All three expose: x, y, w, h, cx, cy, area, iou()
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Regular class -- every instance carries a __dict__ hash table
# ---------------------------------------------------------------------------

class BboxSlow:
    """Bounding box without __slots__. Each instance has a hidden __dict__."""

    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    # -- properties ---------------------------------------------------------

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def w(self) -> float:
        return self._w

    @property
    def h(self) -> float:
        return self._h

    @property
    def cx(self) -> float:
        return self._x + self._w / 2

    @property
    def cy(self) -> float:
        return self._y + self._h / 2

    @property
    def area(self) -> float:
        return self._w * self._h

    # -- methods ------------------------------------------------------------

    def iou(self, other: BboxSlow) -> float:
        """Intersection-over-Union with another bbox."""
        x1 = max(self._x, other._x)
        y1 = max(self._y, other._y)
        x2 = min(self._x + self._w, other._x + other._w)
        y2 = min(self._y + self._h, other._y + other._h)

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area


# ---------------------------------------------------------------------------
# 2. Same class with __slots__ -- no __dict__, fixed memory layout
# ---------------------------------------------------------------------------

class BboxSlots:
    """Bounding box with __slots__. ~56% less memory per instance."""

    __slots__ = ('_x', '_y', '_w', '_h')

    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def w(self) -> float:
        return self._w

    @property
    def h(self) -> float:
        return self._h

    @property
    def cx(self) -> float:
        return self._x + self._w / 2

    @property
    def cy(self) -> float:
        return self._y + self._h / 2

    @property
    def area(self) -> float:
        return self._w * self._h

    def iou(self, other: BboxSlots) -> float:
        x1 = max(self._x, other._x)
        y1 = max(self._y, other._y)
        x2 = min(self._x + self._w, other._x + other._w)
        y2 = min(self._y + self._h, other._y + other._h)

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area


# ---------------------------------------------------------------------------
# 3. Python 3.10+ dataclass with slots=True
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BboxDataclass:
    """Bounding box as a slotted dataclass. Least boilerplate."""

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h

    def iou(self, other: BboxDataclass) -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import tracemalloc

    classes = [
        ("BboxSlow (dict)", BboxSlow),
        ("BboxSlots", BboxSlots),
        ("BboxDataclass", BboxDataclass),
    ]

    n = 100_000
    for label, cls in classes:
        tracemalloc.start()
        boxes = [cls(float(i), float(i), 10.0, 10.0) for i in range(n)]
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        single = sys.getsizeof(boxes[0])
        has_dict = hasattr(boxes[0], "__dict__")
        dict_size = sys.getsizeof(boxes[0].__dict__) if has_dict else 0

        print(f"{label:25s}  "
              f"sizeof={single:4d}  "
              f"dict={dict_size:4d}  "
              f"total={current / 1024:.0f} KB  "
              f"peak={peak / 1024:.0f} KB")

    # Quick IOU check
    a = BboxSlots(0, 0, 10, 10)
    b = BboxSlots(5, 5, 10, 10)
    print(f"\nIOU test: BboxSlots(0,0,10,10) & (5,5,10,10) = {a.iou(b):.4f}")


if __name__ == "__main__":
    _demo()
