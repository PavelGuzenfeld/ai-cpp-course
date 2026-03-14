"""Integration tests for the tracker-utils package.

These tests verify the package works correctly when installed via pip,
not just when running from the source tree.

Run with: pytest test_integration_package.py -v
"""

import subprocess
import sys

import pytest


class TestInstalledPackage:
    """Verify the package behaves correctly after pip install."""

    def test_import_in_subprocess(self):
        """Import the package in a fresh Python process to verify
        it works independently of the test runner's import state."""
        result = subprocess.run(
            [sys.executable, "-c", "from tracker_utils import BBox; print(BBox(1,2,3,4))"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"
        assert "BBox" in result.stdout

    def test_version_in_subprocess(self):
        """Verify version is accessible from a clean process."""
        result = subprocess.run(
            [sys.executable, "-c",
             "from tracker_utils import __version__; print(__version__)"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Version check failed: {result.stderr}"
        version = result.stdout.strip()
        assert len(version.split(".")) >= 3, f"Bad version format: {version}"

    def test_native_extension_is_compiled(self):
        """Verify the native extension is a compiled .so, not a .py fallback."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import tracker_utils._native as m; print(m.__file__)"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Native import failed: {result.stderr}"
        path = result.stdout.strip()
        assert ".so" in path or ".pyd" in path, (
            f"Expected compiled extension (.so/.pyd), got: {path}"
        )

    def test_round_trip_xywh_xyxy(self):
        """End-to-end: create from xyxy, convert back, verify values."""
        from tracker_utils.bbox import BBox

        original = (10.0, 20.0, 110.0, 70.0)
        b = BBox.from_xyxy(*original)
        result = b.to_xyxy()
        for a, b_val in zip(original, result):
            assert abs(a - b_val) < 1e-10, f"{a} != {b_val}"

    def test_iou_commutative(self):
        """IoU(a, b) == IoU(b, a) — verified through the full wrapper stack."""
        from tracker_utils import BBox

        a = BBox(0, 0, 100, 100)
        b = BBox(30, 30, 100, 100)
        assert abs(a.iou(b) - b.iou(a)) < 1e-10

    def test_numpy_round_trip(self):
        """BBox -> numpy array -> BBox preserves values."""
        import numpy as np
        from tracker_utils._native import BBox

        original = BBox(x=1.5, y=2.5, w=3.5, h=4.5)
        arr = original.to_array()
        restored = BBox.from_array(arr)

        assert restored.x == original.x
        assert restored.y == original.y
        assert restored.w == original.w
        assert restored.h == original.h


class TestPackageMetadata:
    """Verify package metadata is correct."""

    def test_metadata_name(self):
        from importlib.metadata import metadata
        meta = metadata("tracker-utils")
        assert meta["Name"] == "tracker-utils"

    def test_metadata_requires_python(self):
        from importlib.metadata import metadata
        meta = metadata("tracker-utils")
        assert ">=3.10" in meta["Requires-Python"]

    def test_metadata_has_description(self):
        from importlib.metadata import metadata
        meta = metadata("tracker-utils")
        assert meta["Summary"] is not None
        assert len(meta["Summary"]) > 0
