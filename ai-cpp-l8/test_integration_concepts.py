"""
Integration tests for Lesson 8: Compile-Time Concepts for Performance.

Tests:
  - Tracking state machine processing 1000 frames with realistic transitions
  - C++ variant SM and Python string SM produce identical state sequences
  - LUT-based image processing on synthetic test images
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "build"))

try:
    import compile_time_lut
    import state_machine
    from state_machine_slow import StringStateMachine as PyStringSM
    HAS_MODULES = True
except ImportError:
    HAS_MODULES = False

pytestmark = pytest.mark.skipif(not HAS_MODULES, reason="L8 compiled modules not built")


# =========================================================================
# Tracking simulation helpers
# =========================================================================
def generate_detection_sequence(num_frames: int, seed: int = 42) -> list:
    """
    Generate a realistic detection sequence for a tracking scenario.

    Returns a list of (has_detection, x, y, w, h) tuples simulating:
    - Object appears at frame 10
    - Moves smoothly with occasional detection failures
    - Disappears for a stretch (triggering lost -> search)
    - Reappears later
    """
    rng = np.random.RandomState(seed)
    sequence = []

    x, y, w, h = 100.0, 200.0, 50.0, 60.0

    for frame in range(num_frames):
        if frame < 10:
            # No detection initially
            sequence.append((False, 0.0, 0.0, 0.0, 0.0))
        elif 10 <= frame < 200:
            # Object moving smoothly, occasional drops
            x += rng.uniform(-2, 3)
            y += rng.uniform(-1, 2)
            if rng.random() < 0.05:
                # 5% detection failure
                sequence.append((False, 0.0, 0.0, 0.0, 0.0))
            else:
                sequence.append((True, x, y, w, h))
        elif 200 <= frame < 260:
            # Object disappears for 60 frames (triggers lost -> search)
            sequence.append((False, 0.0, 0.0, 0.0, 0.0))
        elif 260 <= frame < 500:
            # Object reappears
            x += rng.uniform(-1, 2)
            y += rng.uniform(-1, 1)
            if rng.random() < 0.03:
                sequence.append((False, 0.0, 0.0, 0.0, 0.0))
            else:
                sequence.append((True, x, y, w, h))
        elif 500 <= frame < 600:
            # Another disappearance
            sequence.append((False, 0.0, 0.0, 0.0, 0.0))
        else:
            # Object back again
            x += rng.uniform(-2, 2)
            y += rng.uniform(-2, 2)
            if rng.random() < 0.02:
                sequence.append((False, 0.0, 0.0, 0.0, 0.0))
            else:
                sequence.append((True, x, y, w, h))

    return sequence


class TestTrackingSimulation:
    """Simulate a full tracking scenario and verify state machine behavior."""

    def test_variant_sm_1000_frames(self):
        """Process 1000 frames through the variant state machine."""
        detections = generate_detection_sequence(1000)
        sm = state_machine.VariantStateMachine()

        states = []
        for det in detections:
            sm.update(*det)
            states.append(sm.state())

        # Verify we visit all states during the simulation
        unique_states = set(states)
        assert "idle" in unique_states
        assert "tracking" in unique_states
        assert "lost" in unique_states
        assert "search" in unique_states

        # First 10 frames should be idle (no detections)
        for i in range(10):
            assert states[i] == "idle", f"Frame {i} should be idle"

        # Frame 10 has a detection -> should transition to tracking
        assert states[10] == "tracking"

    def test_string_sm_1000_frames(self):
        """Process 1000 frames through the C++ string state machine."""
        detections = generate_detection_sequence(1000)
        sm = state_machine.StringStateMachine()

        states = []
        for det in detections:
            sm.update(*det)
            states.append(sm.state())

        unique_states = set(states)
        assert "idle" in unique_states
        assert "tracking" in unique_states
        assert "lost" in unique_states
        assert "search" in unique_states

    def test_variant_and_string_identical_states(self):
        """C++ variant SM and C++ string SM must produce identical state sequences."""
        detections = generate_detection_sequence(1000)

        sm_str = state_machine.StringStateMachine()
        sm_var = state_machine.VariantStateMachine()

        for frame_idx, det in enumerate(detections):
            sm_str.update(*det)
            sm_var.update(*det)
            assert sm_str.state() == sm_var.state(), (
                f"Frame {frame_idx}: string SM says '{sm_str.state()}' "
                f"but variant SM says '{sm_var.state()}'"
            )

    def test_python_and_cpp_string_identical_states(self):
        """Python string SM and C++ string SM must produce identical state sequences."""
        detections = generate_detection_sequence(1000)

        py_sm = PyStringSM()
        cpp_sm = state_machine.StringStateMachine()

        for frame_idx, det in enumerate(detections):
            py_sm.update(*det)
            cpp_sm.update(*det)
            assert py_sm.state == cpp_sm.state(), (
                f"Frame {frame_idx}: Python SM says '{py_sm.state}' "
                f"but C++ SM says '{cpp_sm.state()}'"
            )

    def test_all_three_sms_identical(self):
        """All three state machine implementations produce identical sequences."""
        detections = generate_detection_sequence(1000)

        py_sm = PyStringSM()
        cpp_str = state_machine.StringStateMachine()
        cpp_var = state_machine.VariantStateMachine()

        for frame_idx, det in enumerate(detections):
            py_sm.update(*det)
            cpp_str.update(*det)
            cpp_var.update(*det)

            py_state = py_sm.state
            str_state = cpp_str.state()
            var_state = cpp_var.state()

            assert py_state == str_state == var_state, (
                f"Frame {frame_idx}: Python='{py_state}', "
                f"C++ string='{str_state}', C++ variant='{var_state}'"
            )

    def test_search_state_reached_during_long_absence(self):
        """Verify search state is reached after >30 frames of no detection."""
        sm = state_machine.VariantStateMachine()
        sm.update(True, 100.0, 200.0, 50.0, 50.0)  # idle -> tracking
        sm.update(False)  # tracking -> lost

        for i in range(30):
            sm.update(False)

        assert sm.state() == "search"


# =========================================================================
# LUT integration tests on synthetic images
# =========================================================================
class TestLUTImageProcessing:
    """Test LUT-based image processing on synthetic test images."""

    def test_grayscale_gradient_image(self):
        """Process a gradient image and verify monotonic output."""
        # Create a horizontal gradient: each column is a uniform color
        bgr = np.zeros((100, 256, 3), dtype=np.uint8)
        for i in range(256):
            bgr[:, i, :] = i  # All channels equal -> gray = i * (0.114+0.587+0.299) = i

        gray = compile_time_lut.apply_grayscale_lut(bgr)

        # Output should be monotonically non-decreasing along columns
        col_means = gray[0, :]
        for i in range(1, 256):
            assert col_means[i] >= col_means[i - 1], (
                f"Grayscale not monotonic at col {i}: "
                f"{col_means[i]} < {col_means[i-1]}"
            )

    def test_grayscale_uniform_image(self):
        """Uniform color image should produce uniform grayscale output."""
        bgr = np.full((50, 50, 3), 128, dtype=np.uint8)
        gray = compile_time_lut.apply_grayscale_lut(bgr)

        # All pixels should be the same value
        assert np.all(gray == gray[0, 0])

    def test_gamma_roundtrip(self):
        """
        Apply gamma 2.2 then gamma 1/2.2 (approx 0.45) should approximate identity.
        Due to quantization, exact roundtrip is not possible, but values should be close.
        """
        gray = np.arange(0, 256, dtype=np.uint8).reshape(1, 256)

        darkened = compile_time_lut.apply_gamma_lut(gray, 2.2)
        restored = compile_time_lut.apply_gamma_lut(darkened, 0.45)

        # Allow generous tolerance due to 8-bit quantization at each step
        diff = np.abs(restored.astype(int) - gray.astype(int))
        mean_error = np.mean(diff)
        assert mean_error < 15, f"Mean roundtrip error too high: {mean_error}"

    def test_large_image_processing(self):
        """Process a realistic-sized image (1280x720) without errors."""
        bgr = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)

        gray = compile_time_lut.apply_grayscale_lut(bgr)
        assert gray.shape == (720, 1280)
        assert gray.dtype == np.uint8

        gamma_out = compile_time_lut.apply_gamma_lut(gray, 2.2)
        assert gamma_out.shape == (720, 1280)
        assert gamma_out.dtype == np.uint8

    def test_checkerboard_image(self):
        """Process a checkerboard pattern and verify structure is preserved."""
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create 10x10 checkerboard with white and black squares
        for row in range(10):
            for col in range(10):
                if (row + col) % 2 == 0:
                    bgr[row * 10:(row + 1) * 10, col * 10:(col + 1) * 10, :] = 255

        gray = compile_time_lut.apply_grayscale_lut(bgr)

        # White squares should all have the same value
        white_val = gray[0, 0]
        black_val = gray[0, 10]

        assert white_val > 200  # Should be close to 255
        assert black_val == 0

    def test_runtime_and_lut_on_large_image(self):
        """Runtime and LUT should produce similar results on a large image."""
        bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        gray_lut = compile_time_lut.apply_grayscale_lut(bgr)
        gray_runtime = compile_time_lut.apply_grayscale_runtime(bgr)

        diff = np.abs(gray_lut.astype(int) - gray_runtime.astype(int))
        assert np.max(diff) <= 2
        assert np.mean(diff) < 1.0
