"""
Integration test for Lesson 20: a 1000-frame stream with one poisoned
sample, exercising the diagnostic skill from the README exercise -- given a
filter producing garbage at frame N, find the frame that poisoned it.
"""

import math
import sys

sys.path.insert(0, ".")

from robustness_native import CoastingFilter, NaiveFilter  # noqa: E402


def _make_stream(length: int, poison_at: int) -> list[float]:
    stream = [float(i % 7) for i in range(length)]
    stream[poison_at] = float("nan")
    return stream


def _find_poisoning_frame(stream: list[float]) -> int:
    """The diagnostic: replay the naive filter and report the first frame
    at which its state turns non-finite."""
    f = NaiveFilter(alpha=0.5)
    for i, x in enumerate(stream):
        f.update(x)
        if not math.isfinite(f.value):
            return i
    raise ValueError("filter never wedged")


class TestLongRunningStreamWithOnePoisonedFrame:
    def test_naive_filter_wedges_at_frame_900_and_never_recovers(self):
        stream = _make_stream(length=1000, poison_at=900)

        f = NaiveFilter(alpha=0.5)
        for x in stream:
            f.update(x)

        assert math.isnan(f.value)

    def test_coasting_filter_survives_the_same_stream(self):
        stream = _make_stream(length=1000, poison_at=900)

        f = CoastingFilter(alpha=0.5)
        for x in stream:
            f.update(x)

        assert math.isfinite(f.value)

    def test_diagnostic_finds_the_exact_poisoning_frame(self):
        stream = _make_stream(length=1000, poison_at=900)
        assert _find_poisoning_frame(stream) == 900

    def test_diagnostic_is_exact_across_multiple_positions(self):
        for poison_at in (0, 1, 500, 999):
            stream = _make_stream(length=1000, poison_at=poison_at)
            assert _find_poisoning_frame(stream) == poison_at
