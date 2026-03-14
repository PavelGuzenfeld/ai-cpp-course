import os
import tempfile

import numpy as np
import pytest


@pytest.fixture
def synthetic_image():
    """Create a 640x480 RGB image with a known gradient pattern."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Red channel: horizontal gradient
    img[:, :, 0] = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
    # Green channel: vertical gradient
    img[:, :, 1] = np.tile(
        np.linspace(0, 255, 480, dtype=np.uint8).reshape(-1, 1), (1, 640)
    )
    # Blue channel: diagonal gradient
    img[:, :, 2] = (img[:, :, 0].astype(int) + img[:, :, 1].astype(int)) // 2
    return img


@pytest.fixture
def sample_bboxes():
    """Return a list of bounding-box dicts for testing."""
    return [
        {"x": 10, "y": 20, "width": 100, "height": 50, "label": "car", "confidence": 0.95},
        {"x": 200, "y": 150, "width": 80, "height": 120, "label": "person", "confidence": 0.87},
        {"x": 50, "y": 300, "width": 60, "height": 40, "label": "dog", "confidence": 0.72},
    ]


@pytest.fixture
def gpu_available():
    """Check whether a CUDA GPU is available via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture
def tmp_image_dir():
    """Provide a temporary directory for image save/load tests."""
    with tempfile.TemporaryDirectory(prefix="ai_cpp_test_") as tmpdir:
        yield tmpdir
