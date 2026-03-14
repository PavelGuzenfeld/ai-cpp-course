"""
Pure Python BoundingBox — the 'before' version.

This mirrors the pattern found in tracker_engine where every computed
attribute is a @property, each incurring Python descriptor protocol
overhead on every access. In a tracking loop with hundreds of boxes
per frame, this adds up.
"""

import numpy as np


class BBox:
    """No __slots__ — every instance carries a __dict__ (56+ bytes overhead).
    Every attribute access goes through @property descriptor protocol."""

    def __init__(self, x: float, y: float, w: float, h: float):
        if w < 0 or h < 0:
            raise ValueError("width and height must be non-negative")
        self._x = float(x)
        self._y = float(y)
        self._w = float(w)
        self._h = float(h)

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, value: float):
        self._x = float(value)

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, value: float):
        self._y = float(value)

    @property
    def w(self) -> float:
        return self._w

    @w.setter
    def w(self, value: float):
        if value < 0:
            raise ValueError("width must be non-negative")
        self._w = float(value)

    @property
    def h(self) -> float:
        return self._h

    @h.setter
    def h(self, value: float):
        if value < 0:
            raise ValueError("height must be non-negative")
        self._h = float(value)

    @property
    def cx(self) -> float:
        return self._x + self._w / 2.0

    @property
    def cy(self) -> float:
        return self._y + self._h / 2.0

    @property
    def area(self) -> float:
        return self._w * self._h

    @property
    def aspect_ratio(self) -> float:
        return self._w / self._h if self._h > 0 else 0.0

    def iou(self, other: "BBox") -> float:
        x1 = max(self._x, other._x)
        y1 = max(self._y, other._y)
        x2 = min(self._x + self._w, other._x + other._w)
        y2 = min(self._y + self._h, other._y + other._h)

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def contains_point(self, px: float, py: float) -> bool:
        return self._x <= px <= self._x + self._w and self._y <= py <= self._y + self._h

    def to_array(self) -> np.ndarray:
        return np.array([self._x, self._y, self._w, self._h], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "BBox":
        if len(arr) != 4:
            raise ValueError("array must have exactly 4 elements")
        return cls(arr[0], arr[1], arr[2], arr[3])

    def __repr__(self) -> str:
        return f"BBox(x={self._x}, y={self._y}, w={self._w}, h={self._h})"
