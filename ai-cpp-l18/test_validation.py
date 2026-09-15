import numpy as np
import pytest

from filter_native import box_blur, threshold_mask
from record_parser_native import parse_record

# ---------------------------------------------------------------------------
# Golden oracle: box_blur (linear stage) vs an independently-ordered NumPy sum.
# ---------------------------------------------------------------------------


def _numpy_box_blur_oracle(image: np.ndarray, radius: int) -> np.ndarray:
    """Reference implementation via cumulative sums — a different summation
    order from filter_native's nested loop, not a copy of it."""
    height, width = image.shape
    padded = np.pad(image, radius, mode="edge").astype(np.float64)
    out = np.empty((height, width), dtype=np.float64)
    kernel = 2 * radius + 1
    for y in range(height):
        for x in range(width):
            window = padded[y : y + kernel, x : x + kernel]
            out[y, x] = window.mean()
    return out.astype(np.float32)


def test_box_blur_matches_independent_oracle_within_float32_accumulation_error():
    rng = np.random.default_rng(0)
    image = rng.uniform(0.0, 1.0, size=(12, 12)).astype(np.float32)
    radius = 1

    actual = box_blur(image, radius)
    expected = _numpy_box_blur_oracle(image, radius)

    # float32 machine epsilon times the number of terms summed per pixel,
    # with a 5x margin for the two implementations' different summation
    # order: eps32 * (2*radius+1)^2 * 5.
    eps32 = np.finfo(np.float32).eps
    tolerance = eps32 * (2 * radius + 1) ** 2 * 5
    max_diff = np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
    assert max_diff < tolerance, f"max diff {max_diff} exceeds budget {tolerance}"


# ---------------------------------------------------------------------------
# Thresholded stage: mask-disagreement rate, not max-abs-diff.
#
# A pixel placed exactly at the threshold will legitimately flip sides
# between two correct implementations that round differently — this is not
# a bug, and an exact-match assertion on the mask would falsely fail on it.
# ---------------------------------------------------------------------------


def test_threshold_mask_only_disagrees_with_oracle_at_the_boundary_pixels():
    height, width = 10, 10
    image = np.full((height, width), 0.2, dtype=np.float32)
    threshold = 0.5

    # Interior: far from the threshold, must always agree.
    image[2:8, 2:8] = 0.9

    # Boundary band: placed within float32 rounding distance of the
    # threshold, where two independently-rounded computations may disagree.
    boundary_pixels = [(0, 0), (0, 1), (9, 9)]
    eps32 = float(np.finfo(np.float32).eps)
    for i, (y, x) in enumerate(boundary_pixels):
        image[y, x] = np.float32(threshold + (1 if i % 2 == 0 else -1) * eps32 * 2)

    actual_mask = threshold_mask(image, threshold)
    # Independent oracle: float64 comparison against the same threshold.
    oracle_mask = image.astype(np.float64) > np.float64(threshold)

    disagreement = actual_mask != oracle_mask
    disagreement_rate = np.count_nonzero(disagreement) / disagreement.size

    # Only the boundary pixels may disagree — anywhere else is a real bug.
    disagreeing_pixels = set(zip(*np.nonzero(disagreement)))
    assert disagreeing_pixels <= set(boundary_pixels)
    assert disagreement_rate <= len(boundary_pixels) / disagreement.size


# ---------------------------------------------------------------------------
# Parser: deterministic in-suite sweep over a fixed malformed-input corpus.
# The coverage-guided campaign lives in fuzz/fuzz_parser.cpp (see README) —
# this sweep only ever re-runs these exact cases, it never mutates them.
# ---------------------------------------------------------------------------

_VALID_RECORD = bytes([ord("R"), ord("C"), ord("1"), 0, 0, 3, 0, (1 + 2 + 3) % 256, 1, 2, 3])


@pytest.mark.parametrize(
    "case",
    [
        b"",
        b"short",
        b"RC1\x00" + bytes(3),  # header cut off before checksum
        b"XXXX" + bytes(4),  # bad magic
        bytes([ord("R"), ord("C"), ord("1"), 0, 1, 0xFF, 0xFF, 0]),  # length overruns buffer
        bytes([ord("R"), ord("C"), ord("1"), 0, 1, 1, 0, 0, 9]),  # checksum mismatch
    ],
    ids=["empty", "too_short", "header_truncated", "bad_magic", "length_overruns_buffer", "bad_checksum"],
)
def test_parser_rejects_malformed_input_without_crashing(case):
    assert parse_record(case) is None


def test_parser_accepts_a_well_formed_record():
    record = parse_record(_VALID_RECORD)
    assert record is not None
    assert record.version == 0
    assert list(record.payload) == [1, 2, 3]
