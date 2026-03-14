"""
Unit tests for Lesson 8: Compile-Time Concepts for Performance.

Tests:
  - FlatType concept: which types satisfy it and which don't
  - Compile-time LUTs: grayscale and gamma correctness
  - State machine: all transitions produce correct states
  - Image template: correct compile-time sizes and data access
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "build"))

try:
    import compile_time_lut
    import concepts_demo
    import state_machine
    HAS_MODULES = True
except ImportError:
    HAS_MODULES = False

pytestmark = pytest.mark.skipif(not HAS_MODULES, reason="L8 compiled modules not built")


# =========================================================================
# FlatType concept tests
# =========================================================================
class TestFlatTypeConcept:
    """Verify compile-time concept checks for FlatType."""

    def test_int_is_flat(self):
        assert concepts_demo.int_is_flat is True

    def test_float_is_flat(self):
        assert concepts_demo.float_is_flat is True

    def test_double_is_flat(self):
        assert concepts_demo.double_is_flat is True

    def test_bbox_is_flat(self):
        assert concepts_demo.bbox_is_flat is True

    def test_point2d_is_flat(self):
        assert concepts_demo.point2d_is_flat is True

    def test_tracking_result_is_flat(self):
        assert concepts_demo.tracking_result_is_flat is True

    def test_string_is_not_flat(self):
        assert concepts_demo.string_is_flat is False

    def test_vector_is_not_flat(self):
        assert concepts_demo.vector_int_is_flat is False


class TestImageLikeConcept:
    """Verify compile-time concept checks for ImageLike."""

    def test_image_satisfies_imagelike(self):
        assert concepts_demo.image_is_imagelike is True


# =========================================================================
# Serialize / Deserialize (FlatType constrained)
# =========================================================================
class TestSerialization:
    """Test serialization of FlatType-constrained structs."""

    def test_bbox_roundtrip(self):
        bbox = concepts_demo.BBox()
        bbox.x = 10.5
        bbox.y = 20.5
        bbox.w = 100.0
        bbox.h = 200.0

        buf = concepts_demo.serialize_bbox(bbox)
        assert isinstance(buf, list) or isinstance(buf, bytes) or len(buf) > 0

        restored = concepts_demo.deserialize_bbox(buf)
        assert abs(restored.x - 10.5) < 1e-5
        assert abs(restored.y - 20.5) < 1e-5
        assert abs(restored.w - 100.0) < 1e-5
        assert abs(restored.h - 200.0) < 1e-5

    def test_deserialize_too_small_buffer(self):
        with pytest.raises(RuntimeError):
            concepts_demo.deserialize_bbox([0, 1, 2])


# =========================================================================
# Image template tests
# =========================================================================
class TestImageTemplate:
    """Test compile-time Image<W, H, C> template."""

    def test_rgb_dimensions(self):
        img = concepts_demo.ImageRGB()
        assert img.width() == 64
        assert img.height() == 48
        assert img.channels() == 3

    def test_rgb_compile_size(self):
        assert concepts_demo.image_rgb_compile_size() == 64 * 48 * 3

    def test_gray_compile_size(self):
        assert concepts_demo.image_gray_compile_size() == 64 * 48 * 1

    def test_hd_rgb_compile_size(self):
        assert concepts_demo.hd_rgb_compile_size() == 1920 * 1080 * 3

    def test_rgb_size_method(self):
        img = concepts_demo.ImageRGB()
        assert img.size() == 64 * 48 * 3

    def test_gray_dimensions(self):
        img = concepts_demo.ImageGray()
        assert img.width() == 64
        assert img.height() == 48
        assert img.channels() == 1

    def test_pixel_access(self):
        img = concepts_demo.ImageRGB()
        img.set_pixel(10, 20, 0, 42)
        img.set_pixel(10, 20, 1, 128)
        img.set_pixel(10, 20, 2, 255)
        assert img.get_pixel(10, 20, 0) == 42
        assert img.get_pixel(10, 20, 1) == 128
        assert img.get_pixel(10, 20, 2) == 255

    def test_pixel_out_of_range(self):
        img = concepts_demo.ImageRGB()
        with pytest.raises(IndexError):
            img.set_pixel(64, 0, 0, 0)
        with pytest.raises(IndexError):
            img.get_pixel(0, 48, 0)

    def test_default_pixels_are_zero(self):
        img = concepts_demo.ImageRGB()
        assert img.get_pixel(0, 0, 0) == 0
        assert img.get_pixel(63, 47, 2) == 0


# =========================================================================
# Numeric clamp tests
# =========================================================================
class TestNumericConcept:
    """Test Numeric-constrained clamp function."""

    def test_clamp_int_within_range(self):
        assert concepts_demo.clamp_int(5, 0, 10) == 5

    def test_clamp_int_below(self):
        assert concepts_demo.clamp_int(-5, 0, 10) == 0

    def test_clamp_int_above(self):
        assert concepts_demo.clamp_int(15, 0, 10) == 10

    def test_clamp_float(self):
        assert abs(concepts_demo.clamp_float(0.5, 0.0, 1.0) - 0.5) < 1e-6
        assert abs(concepts_demo.clamp_float(-1.0, 0.0, 1.0) - 0.0) < 1e-6
        assert abs(concepts_demo.clamp_float(2.0, 0.0, 1.0) - 1.0) < 1e-6


# =========================================================================
# Grayscale LUT tests
# =========================================================================
class TestGrayscaleLUT:
    """Test compile-time grayscale LUT correctness."""

    def test_black_pixel(self):
        """BGR (0, 0, 0) should produce grayscale 0."""
        bgr = np.zeros((1, 1, 3), dtype=np.uint8)
        gray = compile_time_lut.apply_grayscale_lut(bgr)
        assert gray[0, 0] == 0

    def test_white_pixel(self):
        """BGR (255, 255, 255) should produce grayscale close to 255."""
        bgr = np.full((1, 1, 3), 255, dtype=np.uint8)
        gray = compile_time_lut.apply_grayscale_lut(bgr)
        # LUT uses integer truncation, so sum of floors may be slightly less than 255
        assert gray[0, 0] >= 250

    def test_pure_blue(self):
        """BGR (255, 0, 0) -> weight 0.114 -> ~29."""
        bgr = np.zeros((1, 1, 3), dtype=np.uint8)
        bgr[0, 0, 0] = 255
        gray = compile_time_lut.apply_grayscale_lut(bgr)
        assert abs(int(gray[0, 0]) - 29) <= 1

    def test_pure_green(self):
        """BGR (0, 255, 0) -> weight 0.587 -> ~149."""
        bgr = np.zeros((1, 1, 3), dtype=np.uint8)
        bgr[0, 0, 1] = 255
        gray = compile_time_lut.apply_grayscale_lut(bgr)
        assert abs(int(gray[0, 0]) - 149) <= 1

    def test_pure_red(self):
        """BGR (0, 0, 255) -> weight 0.299 -> ~76."""
        bgr = np.zeros((1, 1, 3), dtype=np.uint8)
        bgr[0, 0, 2] = 255
        gray = compile_time_lut.apply_grayscale_lut(bgr)
        assert abs(int(gray[0, 0]) - 76) <= 1

    def test_output_shape(self):
        """Output should be (H, W) for input (H, W, 3)."""
        bgr = np.zeros((100, 200, 3), dtype=np.uint8)
        gray = compile_time_lut.apply_grayscale_lut(bgr)
        assert gray.shape == (100, 200)

    def test_runtime_vs_lut_match(self):
        """Runtime and LUT methods should produce similar results."""
        bgr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        gray_lut = compile_time_lut.apply_grayscale_lut(bgr)
        gray_runtime = compile_time_lut.apply_grayscale_runtime(bgr)
        # Allow +-1 difference due to integer truncation vs float rounding
        diff = np.abs(gray_lut.astype(int) - gray_runtime.astype(int))
        assert np.max(diff) <= 2

    def test_grayscale_against_opencv_formula(self):
        """Compare LUT output against manual BT.601 computation."""
        bgr = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        gray_lut = compile_time_lut.apply_grayscale_lut(bgr)

        # Manual computation using the same BT.601 weights
        b = bgr[:, :, 0].astype(float)
        g = bgr[:, :, 1].astype(float)
        r = bgr[:, :, 2].astype(float)
        expected = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)

        diff = np.abs(gray_lut.astype(int) - expected.astype(int))
        assert np.max(diff) <= 2


# =========================================================================
# Gamma LUT tests
# =========================================================================
class TestGammaLUT:
    """Test compile-time gamma correction LUT."""

    def test_identity_gamma(self):
        """Gamma 1.0 should be identity (output == input)."""
        gray = np.arange(256, dtype=np.uint8).reshape(1, 256)
        result = compile_time_lut.apply_gamma_lut(gray, 1.0)
        np.testing.assert_array_equal(result, gray)

    def test_gamma_black_stays_black(self):
        """Gamma correction of 0 should always be 0."""
        gray = np.zeros((1, 1), dtype=np.uint8)
        result = compile_time_lut.apply_gamma_lut(gray, 2.2)
        assert result[0, 0] == 0

    def test_gamma_white_stays_white(self):
        """Gamma correction of 255 should always be 255."""
        gray = np.full((1, 1), 255, dtype=np.uint8)
        result = compile_time_lut.apply_gamma_lut(gray, 2.2)
        assert result[0, 0] == 255

    def test_gamma_2_2_darkens(self):
        """Gamma > 1 should darken midtones (128 -> something less)."""
        gray = np.full((1, 1), 128, dtype=np.uint8)
        result = compile_time_lut.apply_gamma_lut(gray, 2.2)
        assert result[0, 0] < 128

    def test_gamma_0_45_brightens(self):
        """Gamma < 1 should brighten midtones (128 -> something more)."""
        gray = np.full((1, 1), 128, dtype=np.uint8)
        result = compile_time_lut.apply_gamma_lut(gray, 0.45)
        assert result[0, 0] > 128

    def test_gamma_output_shape(self):
        gray = np.zeros((100, 200), dtype=np.uint8)
        result = compile_time_lut.apply_gamma_lut(gray, 2.2)
        assert result.shape == (100, 200)

    def test_gamma_runtime_vs_lut_match(self):
        """Runtime and LUT gamma should produce similar results."""
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result_lut = compile_time_lut.apply_gamma_lut(gray, 2.2)
        result_runtime = compile_time_lut.apply_gamma_runtime(gray, 2.2)
        diff = np.abs(result_lut.astype(int) - result_runtime.astype(int))
        # constexpr pow uses series approximation — may differ from std::pow
        # for extreme values; up to 10 difference is acceptable for 8-bit output
        assert np.max(diff) <= 10

    def test_lut_value_lookup(self):
        """Test individual LUT value access."""
        # Channel 0 = B (weight 0.114), intensity 100 -> floor(100 * 0.114) = 11
        val = compile_time_lut.get_grayscale_lut_value(0, 100)
        assert val == 11

        # Channel 1 = G (weight 0.587), intensity 100 -> floor(100 * 0.587) = 58
        val = compile_time_lut.get_grayscale_lut_value(1, 100)
        assert val == 58


# =========================================================================
# State machine tests
# =========================================================================
class TestStringStateMachine:
    """Test the C++ string-based state machine."""

    def test_initial_state(self):
        sm = state_machine.StringStateMachine()
        assert sm.state() == "idle"

    def test_idle_to_tracking(self):
        sm = state_machine.StringStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        assert sm.state() == "tracking"

    def test_idle_stays_idle(self):
        sm = state_machine.StringStateMachine()
        sm.update(False)
        assert sm.state() == "idle"

    def test_tracking_to_lost(self):
        sm = state_machine.StringStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        sm.update(False)
        assert sm.state() == "lost"
        assert sm.lost_frames() == 1

    def test_lost_to_tracking(self):
        sm = state_machine.StringStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        sm.update(False)
        sm.update(True, 15.0, 25.0, 35.0, 45.0)
        assert sm.state() == "tracking"

    def test_lost_to_search(self):
        sm = state_machine.StringStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        sm.update(False)  # lost, frames=1
        for _ in range(30):
            sm.update(False)  # lost, frames 2..31
        assert sm.state() == "search"

    def test_search_to_tracking(self):
        sm = state_machine.StringStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        sm.update(False)
        for _ in range(30):
            sm.update(False)
        assert sm.state() == "search"
        sm.update(True, 50.0, 60.0, 70.0, 80.0)
        assert sm.state() == "tracking"

    def test_search_stays_search(self):
        sm = state_machine.StringStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        sm.update(False)
        for _ in range(30):
            sm.update(False)
        assert sm.state() == "search"
        sm.update(False)
        assert sm.state() == "search"

    def test_target_updated(self):
        sm = state_machine.StringStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        t = sm.target()
        assert t == (10.0, 20.0, 30.0, 40.0)


class TestVariantStateMachine:
    """Test the C++ variant-based state machine."""

    def test_initial_state(self):
        sm = state_machine.VariantStateMachine()
        assert sm.state() == "idle"

    def test_idle_to_tracking(self):
        sm = state_machine.VariantStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        assert sm.state() == "tracking"

    def test_idle_stays_idle(self):
        sm = state_machine.VariantStateMachine()
        sm.update(False)
        assert sm.state() == "idle"

    def test_tracking_to_lost(self):
        sm = state_machine.VariantStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        sm.update(False)
        assert sm.state() == "lost"
        assert sm.lost_frames() == 1

    def test_lost_to_tracking(self):
        sm = state_machine.VariantStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        sm.update(False)
        sm.update(True, 15.0, 25.0, 35.0, 45.0)
        assert sm.state() == "tracking"

    def test_lost_to_search(self):
        sm = state_machine.VariantStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        sm.update(False)  # lost, frames=1
        for _ in range(30):
            sm.update(False)  # lost, frames 2..31
        assert sm.state() == "search"

    def test_search_to_tracking(self):
        sm = state_machine.VariantStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        sm.update(False)
        for _ in range(30):
            sm.update(False)
        assert sm.state() == "search"
        sm.update(True, 50.0, 60.0, 70.0, 80.0)
        assert sm.state() == "tracking"

    def test_search_stays_search(self):
        sm = state_machine.VariantStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        sm.update(False)
        for _ in range(30):
            sm.update(False)
        assert sm.state() == "search"
        sm.update(False)
        assert sm.state() == "search"

    def test_target_updated(self):
        sm = state_machine.VariantStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        t = sm.target()
        assert t == (10.0, 20.0, 30.0, 40.0)

    def test_lost_frames_reset_on_reacquire(self):
        sm = state_machine.VariantStateMachine()
        sm.update(True, 10.0, 20.0, 30.0, 40.0)
        sm.update(False)
        assert sm.lost_frames() == 1
        sm.update(False)
        assert sm.lost_frames() == 2
        sm.update(True, 50.0, 60.0, 70.0, 80.0)
        assert sm.lost_frames() == 0
