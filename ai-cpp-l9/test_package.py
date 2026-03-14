"""Unit tests for the tracker-utils package.

Run with: pytest test_package.py -v
"""

import math
import pathlib

import pytest


class TestImport:
    """Verify the package can be imported and has expected attributes."""

    def test_import_package(self):
        import tracker_utils
        assert hasattr(tracker_utils, "BBox")

    def test_version_is_accessible(self):
        from tracker_utils import __version__
        assert isinstance(__version__, str)
        # Version should look like a semver string (at least X.Y.Z)
        parts = __version__.split(".")
        assert len(parts) >= 3, f"Expected semver, got {__version__}"

    def test_native_module_is_accessible(self):
        from tracker_utils import _native
        assert hasattr(_native, "BBox")


class TestBBoxWrapper:
    """Test the high-level Python BBox wrapper."""

    def test_basic_construction(self):
        from tracker_utils import BBox
        b = BBox(10, 20, 100, 50)
        assert b.x == 10
        assert b.y == 20
        assert b.w == 100
        assert b.h == 50

    def test_from_xywh(self):
        from tracker_utils.bbox import BBox
        b = BBox.from_xywh(10, 20, 100, 50)
        assert b.x == 10
        assert b.w == 100

    def test_from_xyxy(self):
        from tracker_utils.bbox import BBox
        b = BBox.from_xyxy(10, 20, 110, 70)
        assert b.x == 10
        assert b.y == 20
        assert b.w == 100
        assert b.h == 50

    def test_from_xyxy_invalid(self):
        from tracker_utils.bbox import BBox
        with pytest.raises(ValueError, match="must be >="):
            BBox.from_xyxy(100, 20, 10, 70)

    def test_from_center(self):
        from tracker_utils.bbox import BBox
        b = BBox.from_center(60, 45, 100, 50)
        assert b.x == 10
        assert b.y == 20
        assert b.w == 100
        assert b.h == 50

    def test_to_xyxy(self):
        from tracker_utils.bbox import BBox
        b = BBox(10, 20, 100, 50)
        x1, y1, x2, y2 = b.to_xyxy()
        assert x1 == 10
        assert y1 == 20
        assert x2 == 110
        assert y2 == 70

    def test_area(self):
        from tracker_utils import BBox
        b = BBox(0, 0, 100, 50)
        assert b.area == 5000

    def test_center(self):
        from tracker_utils import BBox
        b = BBox(10, 20, 100, 50)
        assert b.cx == 60
        assert b.cy == 45

    def test_aspect_ratio(self):
        from tracker_utils import BBox
        b = BBox(0, 0, 100, 50)
        assert b.aspect_ratio == 2.0

    def test_iou_identical(self):
        from tracker_utils import BBox
        b = BBox(0, 0, 100, 100)
        assert math.isclose(b.iou(b), 1.0)

    def test_iou_no_overlap(self):
        from tracker_utils import BBox
        a = BBox(0, 0, 50, 50)
        b = BBox(100, 100, 50, 50)
        assert b.iou(a) == 0.0

    def test_iou_partial_overlap(self):
        from tracker_utils import BBox
        a = BBox(0, 0, 100, 100)
        b = BBox(50, 50, 100, 100)
        iou = a.iou(b)
        assert 0 < iou < 1

    def test_contains_point(self):
        from tracker_utils import BBox
        b = BBox(10, 20, 100, 50)
        assert b.contains_point(50, 40)
        assert not b.contains_point(0, 0)

    def test_repr(self):
        from tracker_utils import BBox
        b = BBox(10, 20, 100, 50)
        r = repr(b)
        assert "BBox" in r
        assert "10" in r

    def test_equality(self):
        from tracker_utils import BBox
        a = BBox(10, 20, 100, 50)
        b = BBox(10, 20, 100, 50)
        assert a == b

    def test_inequality(self):
        from tracker_utils import BBox
        a = BBox(10, 20, 100, 50)
        b = BBox(10, 20, 100, 51)
        assert a != b

    def test_negative_dimensions_raise(self):
        from tracker_utils import BBox
        with pytest.raises((ValueError, Exception)):
            BBox(0, 0, -10, 50)

    def test_setters(self):
        from tracker_utils import BBox
        b = BBox(10, 20, 100, 50)
        b.x = 99
        assert b.x == 99


class TestNativeModule:
    """Test the C++ native module directly."""

    def test_native_bbox_creation(self):
        from tracker_utils._native import BBox
        b = BBox(x=1, y=2, w=3, h=4)
        assert b.x == 1
        assert b.area == 12

    def test_native_to_array(self):
        import numpy as np
        from tracker_utils._native import BBox
        b = BBox(x=1, y=2, w=3, h=4)
        arr = b.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4,)
        assert list(arr) == [1, 2, 3, 4]

    def test_native_from_array(self):
        import numpy as np
        from tracker_utils._native import BBox
        arr = np.array([10, 20, 30, 40], dtype=np.float64)
        b = BBox.from_array(arr)
        assert b.x == 10
        assert b.h == 40


class TestTypeStub:
    """Verify the type stub file exists in the installed package."""

    def test_pyi_file_exists(self):
        import tracker_utils
        pkg_dir = pathlib.Path(tracker_utils.__file__).parent
        stub = pkg_dir / "_native.pyi"
        assert stub.exists(), f"Type stub not found at {stub}"

    def test_py_typed_marker_exists(self):
        import tracker_utils
        pkg_dir = pathlib.Path(tracker_utils.__file__).parent
        marker = pkg_dir / "py.typed"
        assert marker.exists(), f"py.typed marker not found at {marker}"
