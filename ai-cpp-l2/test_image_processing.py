"""
Unit tests for stride_view_native: the correct (stride-aware) copy against
the buggy (width-substituted-for-stride) copy, on small hand-computed
buffers where the padding and the shear are exact, not approximate.
"""
import numpy as np
import pytest

import stride_view_native as sv


def _padded_buffer(rows: list[bytes], row_bytes: int, stride: int, fill: int = 0xAA) -> np.ndarray:
    """rows[i] is the real pixel data for row i; pads each row to `stride`
    bytes with `fill`, so a wrong read that ignores the padding picks up a
    detectable, non-zero marker instead of coincidentally-plausible zeros."""
    assert stride >= row_bytes
    buf = np.full(stride * len(rows), fill, dtype=np.uint8)
    for i, row in enumerate(rows):
        assert len(row) == row_bytes
        buf[i * stride:i * stride + row_bytes] = np.frombuffer(row, dtype=np.uint8)
    return buf


class TestCopyUsingStride:
    def test_single_row_no_padding_is_identity(self):
        src = _padded_buffer([b"\x01\x02\x03\x04"], row_bytes=4, stride=4)
        out = sv.copy_using_stride(src, width=4, height=1, channels=1, stride=4)
        assert bytes(out) == b"\x01\x02\x03\x04"

    def test_padded_rows_reconstruct_exactly(self):
        rows = [b"\x01\x02\x03\x04", b"\x05\x06\x07\x08", b"\x09\x0a\x0b\x0c"]
        src = _padded_buffer(rows, row_bytes=4, stride=6)
        out = sv.copy_using_stride(src, width=4, height=3, channels=1, stride=6)
        assert bytes(out) == b"".join(rows)


class TestCopyUsingWidthAsStride:
    def test_first_row_is_unaffected_by_the_bug(self):
        """Row 0 starts at offset 0 under both the real stride and the
        (wrong) width-as-stride assumption -- the bug is invisible until
        row 1."""
        rows = [b"\x01\x02\x03\x04", b"\x05\x06\x07\x08"]
        src = _padded_buffer(rows, row_bytes=4, stride=6)
        out = sv.copy_using_width_as_stride(src, width=4, height=2, channels=1, stride=6)
        assert bytes(out[:4]) == rows[0]

    def test_second_row_shears_in_the_padding_bytes(self):
        """With stride=6 and row_bytes=4, the wrong read for row 1 starts
        at offset 4 (width * channels) instead of the real row start at
        offset 6 -- it picks up row 0's 2 padding-fill bytes followed by
        row 1's first 2 real bytes, not row 1's real data."""
        rows = [b"\x01\x02\x03\x04", b"\x05\x06\x07\x08"]
        src = _padded_buffer(rows, row_bytes=4, stride=6, fill=0xAA)
        out = sv.copy_using_width_as_stride(src, width=4, height=2, channels=1, stride=6)
        assert bytes(out[4:8]) == b"\xaa\xaa\x05\x06"
        assert bytes(out[4:8]) != rows[1]

    def test_matches_correct_copy_when_stride_equals_row_bytes(self):
        """No padding means there is nothing to substitute width for --
        the two copies must agree exactly."""
        rows = [b"\x01\x02\x03\x04", b"\x05\x06\x07\x08"]
        src = _padded_buffer(rows, row_bytes=4, stride=4)
        correct = sv.copy_using_stride(src, width=4, height=2, channels=1, stride=4)
        broken = sv.copy_using_width_as_stride(src, width=4, height=2, channels=1, stride=4)
        assert bytes(correct) == bytes(broken)


class TestInputValidation:
    def test_buffer_smaller_than_stride_times_height_raises(self):
        src = np.zeros(4, dtype=np.uint8)
        with pytest.raises(ValueError):
            sv.copy_using_stride(src, width=4, height=2, channels=1, stride=4)
