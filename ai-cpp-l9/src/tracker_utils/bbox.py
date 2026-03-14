"""High-level Python API wrapping the C++ BBox extension."""

from __future__ import annotations

from tracker_utils._native import BBox as _NativeBBox


class BBox:
    """Bounding box with C++ acceleration under the hood.

    This class wraps the nanobind C++ BBox, adding Pythonic convenience
    methods (alternate constructors, equality, repr) while keeping all
    heavy computation in C++.

    Coordinates follow the (x, y, w, h) convention where (x, y) is the
    top-left corner, w is width, and h is height.
    """

    __slots__ = ("_native",)

    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        self._native = _NativeBBox(x, y, w, h)

    # ── Alternate constructors ──────────────────────────────────────

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BBox:
        """Create a BBox from top-left corner (x, y) and size (w, h).

        This is the same as the default constructor, provided for
        symmetry with the other factory methods.
        """
        return cls(x, y, w, h)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> BBox:
        """Create a BBox from two corners: top-left (x1, y1) and
        bottom-right (x2, y2).

        Raises ValueError if x2 < x1 or y2 < y1.
        """
        if x2 < x1 or y2 < y1:
            raise ValueError(
                f"Bottom-right corner ({x2}, {y2}) must be >= "
                f"top-left corner ({x1}, {y1})"
            )
        return cls(x1, y1, x2 - x1, y2 - y1)

    @classmethod
    def from_center(cls, cx: float, cy: float, w: float, h: float) -> BBox:
        """Create a BBox from center point (cx, cy) and size (w, h)."""
        return cls(cx - w / 2.0, cy - h / 2.0, w, h)

    # ── Properties (delegated to C++) ───────────────────────────────

    @property
    def x(self) -> float:
        return self._native.x

    @x.setter
    def x(self, value: float) -> None:
        self._native.x = value

    @property
    def y(self) -> float:
        return self._native.y

    @y.setter
    def y(self, value: float) -> None:
        self._native.y = value

    @property
    def w(self) -> float:
        return self._native.w

    @w.setter
    def w(self, value: float) -> None:
        self._native.w = value

    @property
    def h(self) -> float:
        return self._native.h

    @h.setter
    def h(self, value: float) -> None:
        self._native.h = value

    @property
    def cx(self) -> float:
        """Center X coordinate."""
        return self._native.cx

    @property
    def cy(self) -> float:
        """Center Y coordinate."""
        return self._native.cy

    @property
    def area(self) -> float:
        """Area of the bounding box (w * h)."""
        return self._native.area

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (w / h). Returns 0 if h == 0."""
        return self._native.aspect_ratio

    # ── Methods ─────────────────────────────────────────────────────

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return the box as (x1, y1, x2, y2) — top-left and bottom-right corners."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def iou(self, other: BBox) -> float:
        """Compute Intersection over Union with another BBox."""
        return self._native.iou(other._native)

    def contains_point(self, px: float, py: float) -> bool:
        """Check whether a point (px, py) lies inside this box."""
        return self._native.contains_point(px, py)

    def to_array(self):
        """Return the box as a numpy array [x, y, w, h]."""
        return self._native.to_array()

    # ── Dunder methods ──────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"BBox(x={self.x:.1f}, y={self.y:.1f}, w={self.w:.1f}, h={self.h:.1f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BBox):
            return NotImplemented
        return (
            self.x == other.x
            and self.y == other.y
            and self.w == other.w
            and self.h == other.h
        )
