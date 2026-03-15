"""
Unit tests for Jetson lesson components.

Tests:
  - Unified memory allocation works on GPU
  - Fused kernel correctness (normalize + transpose)
  - Device detection function
  - Skips appropriately if no GPU or not on Jetson

Usage:
    pytest test_jetson.py -v
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import helpers with graceful fallback
# ---------------------------------------------------------------------------

try:
    import torch

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_TORCH = False
    HAS_CUDA = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device available")


def _is_jetson():
    """Check if running on Jetson hardware."""
    model_path = Path("/proc/device-tree/model")
    if not model_path.exists():
        return False
    try:
        model_str = model_path.read_text().strip().rstrip("\x00").lower()
        return "jetson" in model_str or "nvidia" in model_str
    except OSError:
        return False


requires_jetson = pytest.mark.skipif(not _is_jetson(), reason="Not running on Jetson")


# ---------------------------------------------------------------------------
# Device detection tests
# ---------------------------------------------------------------------------

class TestDeviceDetection:
    """Tests for detect_jetson() and related helpers."""

    def test_detect_jetson_returns_dict_or_none(self):
        """detect_jetson() should return a dict on Jetson, None otherwise."""
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_jetson import detect_jetson

        result = detect_jetson()
        if _is_jetson():
            assert isinstance(result, dict)
            assert "model" in result
            assert "cuda_cores" in result
            assert "memory_total_mb" in result
        else:
            assert result is None

    def test_detect_jetson_model_string(self):
        """On Jetson, model string should contain 'jetson' or 'nvidia'."""
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_jetson import detect_jetson

        result = detect_jetson()
        if result is not None:
            model_lower = result["model"].lower()
            assert "jetson" in model_lower or "nvidia" in model_lower

    def test_detect_jetson_memory(self):
        """On Jetson, memory should be a positive integer."""
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_jetson import detect_jetson

        result = detect_jetson()
        if result is not None:
            assert isinstance(result["memory_total_mb"], int)
            assert result["memory_total_mb"] > 0

    @requires_jetson
    def test_power_mode_readable(self):
        """On Jetson, power mode should be readable via nvpmodel."""
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_jetson import _get_power_mode

        mode = _get_power_mode()
        # May be None if nvpmodel requires sudo, but should not raise
        assert mode is None or isinstance(mode, str)


# ---------------------------------------------------------------------------
# Unified memory tests
# ---------------------------------------------------------------------------

class TestUnifiedMemory:
    """Tests for unified memory allocation behavior."""

    @requires_cuda
    def test_cpu_to_gpu_transfer(self):
        """Tensors created on CPU should transfer to GPU correctly."""
        n = 1024
        cpu_tensor = torch.randn(n)
        gpu_tensor = cpu_tensor.to("cuda:0")

        assert gpu_tensor.device.type == "cuda"
        assert gpu_tensor.shape == (n,)
        np.testing.assert_allclose(
            cpu_tensor.numpy(), gpu_tensor.cpu().numpy(), rtol=1e-6
        )

    @requires_cuda
    def test_unified_allocation_roundtrip(self):
        """Data should survive CPU -> GPU -> CPU roundtrip without corruption."""
        data = np.random.randn(1000).astype(np.float32)
        cpu_tensor = torch.from_numpy(data.copy())
        gpu_tensor = cpu_tensor.to("cuda:0")
        result = gpu_tensor.cpu().numpy()

        np.testing.assert_allclose(data, result, rtol=1e-6)

    @requires_cuda
    def test_gpu_compute_produces_correct_result(self):
        """GPU vector addition should match CPU computation."""
        n = 4096
        a = torch.randn(n)
        b = torch.randn(n)
        expected = a + b

        a_gpu = a.to("cuda:0")
        b_gpu = b.to("cuda:0")
        result_gpu = (a_gpu + b_gpu).cpu()

        np.testing.assert_allclose(
            expected.numpy(), result_gpu.numpy(), rtol=1e-5
        )

    @requires_cuda
    def test_large_unified_allocation(self):
        """Allocating a moderately large tensor should work on GPU."""
        # 16 MB of float32
        n = 4 * 1024 * 1024
        t = torch.randn(n, device="cuda:0")
        assert t.shape == (n,)
        assert t.device.type == "cuda"

        # Verify it is usable
        result = t.sum().item()
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# Fused kernel correctness tests
# ---------------------------------------------------------------------------

class TestFusedKernel:
    """Tests for preprocessing kernel correctness."""

    def _numpy_reference_preprocess(self, image, mean, std):
        """Reference NumPy implementation of image preprocessing."""
        img = image.astype(np.float32) / np.float32(255.0)
        img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        return np.ascontiguousarray(img)

    def test_numpy_preprocess_shape(self):
        """Preprocessing should produce CHW output from HWC input."""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        result = self._numpy_reference_preprocess(image, mean, std)
        assert result.shape == (3, 480, 640)
        assert result.dtype == np.float32

    def test_numpy_preprocess_normalization(self):
        """Preprocessing should correctly normalize pixel values."""
        # Create a known image
        image = np.full((2, 2, 3), 128, dtype=np.uint8)
        mean = [0.5, 0.5, 0.5]
        std = [1.0, 1.0, 1.0]

        result = self._numpy_reference_preprocess(image, mean, std)

        # 128/255 = 0.50196, then (0.50196 - 0.5) / 1.0 = 0.00196
        expected_val = np.float32((np.float32(128.0) / np.float32(255.0) - np.float32(0.5)) / np.float32(1.0))
        np.testing.assert_allclose(result, expected_val, rtol=1e-4)

    @requires_cuda
    def test_torch_gpu_preprocess_matches_numpy(self):
        """PyTorch GPU preprocessing should match NumPy reference."""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # NumPy reference
        expected = self._numpy_reference_preprocess(image, mean, std)

        # PyTorch GPU version
        device = torch.device("cuda:0")
        image_gpu = torch.from_numpy(image).to(device)
        mean_gpu = torch.tensor(mean, device=device).view(1, 1, 3)
        std_gpu = torch.tensor(std, device=device).view(1, 1, 3)

        result_gpu = image_gpu.float() / 255.0
        result_gpu = (result_gpu - mean_gpu) / std_gpu
        result_gpu = result_gpu.permute(2, 0, 1).contiguous()
        result = result_gpu.cpu().numpy()

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_preprocess_various_sizes(self):
        """Preprocessing should work for different image sizes."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        for h, w in [(224, 224), (480, 640), (1080, 1920), (64, 64)]:
            image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            result = self._numpy_reference_preprocess(image, mean, std)
            assert result.shape == (3, h, w), f"Failed for size {h}x{w}"

    def test_cpp_fused_kernel_if_available(self):
        """If the C++ fused kernel module is built, test its correctness."""
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            sys.path.insert(0, str(Path(__file__).parent / "build"))
            import gpu_preprocess as gpu_mod
        except ImportError:
            pytest.skip("gpu_preprocess module not built")

        if not gpu_mod.cuda_available():
            pytest.skip("CUDA not available in gpu_preprocess")

        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        expected = self._numpy_reference_preprocess(image, mean, std)
        result = gpu_mod.fused_preprocess(image, mean, std)

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

class TestImports:
    """Verify that lesson modules are importable."""

    def test_import_benchmark_jetson(self):
        """benchmark_jetson module should be importable."""
        sys.path.insert(0, str(Path(__file__).parent))
        import benchmark_jetson
        assert hasattr(benchmark_jetson, "detect_jetson")
        assert hasattr(benchmark_jetson, "main")

    def test_import_power_mode_demo(self):
        """power_mode_demo module should be importable."""
        sys.path.insert(0, str(Path(__file__).parent))
        import power_mode_demo
        assert hasattr(power_mode_demo, "is_jetson")
        assert hasattr(power_mode_demo, "main")
