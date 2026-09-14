"""
Integration test for Lesson 14: a small pipeline consuming a stream of
frames from the mocked device, as a real capture loop would.
"""

import sys

sys.path.insert(0, ".")

from device_mock_native import MockDevice  # noqa: E402


class TestCaptureLoop:
    def test_100_frame_capture_run_has_consistent_cadence(self):
        device = MockDevice("mock://camera0")
        frames = [device.read_frame() for _ in range(100)]

        assert device.frame_count == 100
        assert all(f.width == 64 and f.height == 48 for f in frames)

        # ~30fps cadence: consecutive timestamps step by the same amount.
        deltas = {b.timestamp_ns - a.timestamp_ns for a, b in zip(frames, frames[1:])}
        assert deltas == {33_333_333}

    def test_two_independent_devices_do_not_share_state(self):
        camera0 = MockDevice("mock://camera0")
        camera1 = MockDevice("mock://camera1")

        for _ in range(10):
            camera0.read_frame()

        assert camera0.frame_count == 10
        assert camera1.frame_count == 0
