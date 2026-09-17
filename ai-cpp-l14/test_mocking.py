"""
Unit tests for Lesson 14: mocking a vendor C API.

Tests:
  - Contract tests, run against BOTH the mock and the real V4L2 device
  - Mock-only tests, which assert the mock's own synthetic fill pattern
  - The real-hardware path is skipped (not stubbed) on a box with no device

The split between the first two is the lesson. A test that only passes
against the mock is a test of the mock, not of the code under it.
"""

import glob
import sys

import pytest

sys.path.insert(0, ".")

from device_mock_native import MockDevice  # noqa: E402

# The real path is V4L2 (see real_device_v4l2.cpp), so the marker is a node
# that can actually exist. No camera on this box means these tests skip --
# the lesson's rule is skip, never stub.
REAL_DEVICE_PATH = next(iter(sorted(glob.glob("/dev/video*"))), None)
REAL_DEVICE_AVAILABLE = REAL_DEVICE_PATH is not None


def open_mock():
    return MockDevice("mock://camera0")


def open_real():
    from device_real_native import RealDevice

    return RealDevice(REAL_DEVICE_PATH)


# Contract tests run against every implementation of the API. Adding an
# implementation means adding it here, not writing a parallel test file.
IMPLEMENTATIONS = [pytest.param(open_mock, id="mock")]
if REAL_DEVICE_AVAILABLE:
    IMPLEMENTATIONS.append(pytest.param(open_real, id="real-v4l2"))


@pytest.mark.parametrize("open_device", IMPLEMENTATIONS)
class TestTheContractBothImplementationsOwe:
    """Everything asserted here is true of the API, not of one implementation.

    A real camera will not be 64x48, so nothing in this class may name a
    resolution or a pixel value.
    """

    def test_a_frame_reports_a_nonzero_resolution(self, open_device):
        frame = open_device().read_frame()
        assert frame.width > 0
        assert frame.height > 0

    def test_resolution_is_stable_across_frames(self, open_device):
        device = open_device()
        first, second = device.read_frame(), device.read_frame()
        assert (first.width, first.height) == (second.width, second.height)

    def test_timestamps_do_not_go_backwards(self, open_device):
        device = open_device()
        stamps = [device.read_frame().timestamp_ns for _ in range(3)]
        assert stamps == sorted(stamps)

    def test_frame_count_tracks_reads(self, open_device):
        device = open_device()
        assert device.frame_count == 0
        device.read_frame()
        device.read_frame()
        assert device.frame_count == 2


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
    reason="no /dev/video* node present -- hardware-only, skipped rather "
    "than stubbed (see the README's rule)",
)
class TestRealDevice:
    def test_real_device_matches_mock_frame_shape(self):
        """The acceptance question: does the mock's *shape* hold on hardware?

        Shape, not values. Both must report a resolution, a monotonic
        timestamp and a per-frame-varying payload; only the mock may claim
        64x48 and an arithmetic fill.
        """
        real = open_real()
        mock = open_mock()
        real_frame, mock_frame = real.read_frame(), mock.read_frame()

        for frame in (real_frame, mock_frame):
            assert frame.width > 0 and frame.height > 0
            assert isinstance(frame.timestamp_ns, int)
            assert isinstance(frame.data_checksum, int)

    def test_the_real_device_payload_changes_between_frames(self):
        # The mock guarantees this by construction. If a real capture returns
        # a constant checksum the buffer is not being refilled, which is the
        # bug a mock can never catch for you.
        device = open_real()
        checksums = {device.read_frame().data_checksum for _ in range(5)}
        assert len(checksums) > 1


class TestTheRealModuleWithoutRealHardware:
    """Needs no camera, so it runs everywhere -- including CI.

    Without this the real implementation's error path would be untested on
    every machine that lacks a device, which is most of them.
    """

    def test_opening_a_nonexistent_node_raises_rather_than_returning_junk(self):
        from device_real_native import RealDevice

        with pytest.raises(RuntimeError):
            RealDevice("/dev/video-does-not-exist")

    def test_opening_a_node_that_is_not_a_capture_device_raises(self):
        # /dev/null opens fine and then fails VIDIOC_QUERYCAP. Catches the
        # implementation returning a handle for anything openable.
        from device_real_native import RealDevice

        with pytest.raises(RuntimeError):
            RealDevice("/dev/null")
