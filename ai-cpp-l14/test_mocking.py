"""
Unit tests for Lesson 14: mocking a vendor C API.

Tests:
  - MockDevice produces deterministic, monotonically-timestamped frames
  - The real-hardware path is skipped (not stubbed) on a box with no device
"""

import os
import sys

import pytest

sys.path.insert(0, ".")

from device_mock_native import MockDevice  # noqa: E402

# A real build would check for the actual device node here (e.g.
# /dev/video0). No such marker exists on x86 CI, so REAL_DEVICE_AVAILABLE
# is always False in this environment -- the point of this lesson is that
# the test below is *skipped*, not silently passed with a stub.
REAL_DEVICE_AVAILABLE = os.path.exists("/dev/ai_cpp_l14_real_device")


class TestMockDeviceFrames:
    def test_frame_has_expected_dimensions(self):
        device = MockDevice("mock://camera0")
        frame = device.read_frame()
        assert frame.width == 64
        assert frame.height == 48

    def test_timestamps_increase_monotonically(self):
        device = MockDevice("mock://camera0")
        t0 = device.read_frame().timestamp_ns
        t1 = device.read_frame().timestamp_ns
        t2 = device.read_frame().timestamp_ns
        assert t0 < t1 < t2

    def test_frame_count_increments_per_read(self):
        device = MockDevice("mock://camera0")
        assert device.frame_count == 0
        device.read_frame()
        assert device.frame_count == 1
        device.read_frame()
        assert device.frame_count == 2

    def test_frames_are_deterministic_given_the_same_call_sequence(self):
        a = MockDevice("mock://camera0")
        b = MockDevice("mock://camera0")
        for _ in range(5):
            fa, fb = a.read_frame(), b.read_frame()
            assert fa.timestamp_ns == fb.timestamp_ns
            assert fa.data_checksum == fb.data_checksum

    def test_first_frame_data_checksum_matches_the_documented_fill_pattern(self):
        # device_mock_native.cpp fills byte i of frame N with (N + i) & 0xFF.
        # Frame 0, i in [0, 64): every term is already < 256, so the sum is
        # just 0 + 1 + ... + 63.
        device = MockDevice("mock://camera0")
        assert device.read_frame().data_checksum == sum(range(64))

    def test_second_frame_data_checksum_matches_the_documented_fill_pattern(self):
        device = MockDevice("mock://camera0")
        device.read_frame()
        assert device.read_frame().data_checksum == sum(1 + i for i in range(64))


@pytest.mark.skipif(
    not REAL_DEVICE_AVAILABLE,
    reason="no real device node present -- this is a hardware-only test, "
    "skipped rather than stubbed (see docs/extending.md's rule in the README)",
)
class TestRealDevice:
    def test_real_device_matches_mock_frame_shape(self):
        # Left intentionally unimplemented: this test exists to be filled
        # in and run on Orin NX / JP6 where a real device node is present.
        # Skipped everywhere else -- never faked green.
        raise NotImplementedError("run on hardware with a real device node")
