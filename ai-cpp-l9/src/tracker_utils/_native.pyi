"""Type stubs for the C++ native extension (tracker_utils._native).

These stubs provide IDE autocomplete and type checking for the nanobind
BBox class. The actual implementation is in _native.cpp.
"""

import numpy as np
from numpy.typing import NDArray

class BBox:
    """Axis-aligned bounding box backed by C++.

    Coordinates use the (x, y, w, h) convention where (x, y) is the
    top-left corner.
    """

    x: float
    """Top-left X coordinate."""

    y: float
    """Top-left Y coordinate."""

    w: float
    """Width of the bounding box."""

    h: float
    """Height of the bounding box."""

    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        """Create a bounding box.

        Args:
            x: Top-left X coordinate.
            y: Top-left Y coordinate.
            w: Width (must be >= 0).
            h: Height (must be >= 0).

        Raises:
            ValueError: If w or h is negative.
        """
        ...

    @property
    def cx(self) -> float:
        """Center X coordinate (x + w/2)."""
        ...

    @property
    def cy(self) -> float:
        """Center Y coordinate (y + h/2)."""
        ...

    @property
    def area(self) -> float:
        """Area of the bounding box (w * h)."""
        ...

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (w / h). Returns 0 if h == 0."""
        ...

    def iou(self, other: BBox) -> float:
        """Compute Intersection over Union with another bounding box.

        Args:
            other: The other bounding box.

        Returns:
            IoU value in [0, 1].
        """
        ...

    def contains_point(self, px: float, py: float) -> bool:
        """Check whether a point lies inside this bounding box.

        Args:
            px: X coordinate of the point.
            py: Y coordinate of the point.

        Returns:
            True if the point is inside the box (inclusive of edges).
        """
        ...

    def to_array(self) -> NDArray[np.float64]:
        """Return the box as a numpy array [x, y, w, h].

        Returns:
            A numpy array of shape (4,) with dtype float64.
        """
        ...

    @staticmethod
    def from_array(arr: NDArray[np.float64]) -> BBox:
        """Create a BBox from a numpy array of 4 doubles.

        Args:
            arr: Array of shape (4,) containing [x, y, w, h].

        Returns:
            A new BBox instance.
        """
        ...

    def __repr__(self) -> str: ...
