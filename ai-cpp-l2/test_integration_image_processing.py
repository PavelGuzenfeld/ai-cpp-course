"""
Integration test for the L2 stride lesson: loads the real course asset
(bmp-2048x1365.bmp), builds a deliberately padded copy of it (as a real
hardware surface would be), and confirms the stride-aware copy reconstructs
it exactly while the width-substituted-for-stride copy silently diverges.
"""
import os

import cv2
import numpy as np

import stride_view_native as sv

HERE = os.path.dirname(os.path.abspath(__file__))
BMP_PATH = os.path.join(HERE, "bmp-2048x1365.bmp")
ROW_PADDING_BYTES = 64  # matches stride_demo.cpp's kRowPaddingBytes


def _load_padded():
    img = cv2.imread(BMP_PATH, cv2.IMREAD_COLOR)
    assert img is not None, f"could not load {BMP_PATH}"
    height, width, channels = img.shape
    row_bytes = width * channels
    stride = row_bytes + ROW_PADDING_BYTES

    padded = np.full(stride * height, 0xAA, dtype=np.uint8)
    flat = img.reshape(height, row_bytes)
    for y in range(height):
        padded[y * stride:y * stride + row_bytes] = flat[y]
    return img, padded, width, height, channels, stride, row_bytes


class TestStrideAwareCopyMatchesSource:
    def test_reconstructs_the_image_exactly(self):
        img, padded, width, height, channels, stride, _row_bytes = _load_padded()
        out = sv.copy_using_stride(padded, width=width, height=height, channels=channels, stride=stride)
        assert bytes(out) == img.tobytes()


class TestWidthSubstitutedForStrideShears:
    def test_first_row_still_matches(self):
        img, padded, width, height, channels, stride, row_bytes = _load_padded()
        out = sv.copy_using_width_as_stride(padded, width=width, height=height, channels=channels, stride=stride)
        assert bytes(out[:row_bytes]) == img.tobytes()[:row_bytes]

    def test_every_row_from_the_second_onward_is_wrong(self):
        """Once the shear starts it never resynchronizes -- each row keeps
        reading from an offset that drifts further from where the real row
        boundary is, so this is not a one-row glitch."""
        img, padded, width, height, channels, stride, row_bytes = _load_padded()
        out = sv.copy_using_width_as_stride(padded, width=width, height=height, channels=channels, stride=stride)
        out_bytes = bytes(out)
        img_bytes = img.tobytes()
        for y in range(1, height):
            row_slice = slice(y * row_bytes, (y + 1) * row_bytes)
            assert out_bytes[row_slice] != img_bytes[row_slice], f"row {y} unexpectedly matched"
