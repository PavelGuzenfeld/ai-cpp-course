"""
Integration tests for the Jetson lesson.

Tests:
  - Full preprocessing pipeline on GPU
  - Unified vs explicit memory produces identical results
  - Power mode detection
  - Skips if not on Jetson or no GPU available

Usage:
    pytest test_integration_jetson.py -v
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports with graceful fallback
# ---------------------------------------------------------------------------

try:
    import torch

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_TORCH = False
    HAS_CUDA = False


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
requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device available")


# ---------------------------------------------------------------------------
# Full preprocessing pipeline tests
# ---------------------------------------------------------------------------

@requires_cuda
class TestPreprocessingPipeline:
    """Integration tests for the full GPU preprocessing pipeline."""

    def _numpy_preprocess(self, image, mean, std):
        """Reference NumPy preprocessing: scale, normalize, HWC->CHW."""
        img = image.astype(np.float32) / 255.0
        img = (img - np.array(mean)) / np.array(std)
        img = img.transpose(2, 0, 1)
        return np.ascontiguousarray(img)

    def test_full_pipeline_single_image(self):
        """Full pipeline: load image, preprocess on GPU, verify output."""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        device = torch.device("cuda:0")

        # Transfer to GPU
        image_gpu = torch.from_numpy(image).to(device)

        # Preprocess on GPU
        mean_gpu = torch.tensor(mean, device=device).view(1, 1, 3)
        std_gpu = torch.tensor(std, device=device).view(1, 1, 3)
        result_gpu = image_gpu.float() / 255.0
        result_gpu = (result_gpu - mean_gpu) / std_gpu
        result_gpu = result_gpu.permute(2, 0, 1).contiguous()

        # Transfer back and verify
        result = result_gpu.cpu().numpy()
        expected = self._numpy_preprocess(image, mean, std)

        assert result.shape == (3, 480, 640)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_full_pipeline_batch(self):
        """Pipeline should handle a batch of images correctly."""
        batch_size = 4
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        images = [
            np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]

        device = torch.device("cuda:0")
        mean_gpu = torch.tensor(mean, device=device).view(1, 1, 1, 3)
        std_gpu = torch.tensor(std, device=device).view(1, 1, 1, 3)

        # Stack into batch, transfer, preprocess
        batch_np = np.stack(images)
        batch_gpu = torch.from_numpy(batch_np).to(device)
        result_gpu = batch_gpu.float() / 255.0
        result_gpu = (result_gpu - mean_gpu) / std_gpu
        result_gpu = result_gpu.permute(0, 3, 1, 2).contiguous()

        result = result_gpu.cpu().numpy()

        assert result.shape == (batch_size, 3, h, w)

        # Verify each image in the batch
        for i in range(batch_size):
            expected = self._numpy_preprocess(images[i], mean, std)
            np.testing.assert_allclose(
                result[i], expected, rtol=1e-4, atol=1e-5,
                err_msg=f"Batch image {i} mismatch"
            )

    def test_pipeline_deterministic(self):
        """Same input should produce identical output across runs."""
        image = np.full((64, 64, 3), 100, dtype=np.uint8)
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]

        device = torch.device("cuda:0")
        mean_gpu = torch.tensor(mean, device=device).view(1, 1, 3)
        std_gpu = torch.tensor(std, device=device).view(1, 1, 3)

        results = []
        for _ in range(5):
            image_gpu = torch.from_numpy(image).to(device)
            r = image_gpu.float() / 255.0
            r = (r - mean_gpu) / std_gpu
            r = r.permute(2, 0, 1).contiguous()
            results.append(r.cpu().numpy())

        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0], results[i],
                err_msg=f"Run {i} produced different output"
            )

    def test_pipeline_fp16(self):
        """FP16 pipeline should produce reasonable results (reduced precision)."""
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        device = torch.device("cuda:0")
        mean_gpu = torch.tensor(mean, device=device, dtype=torch.float16).view(1, 1, 3)
        std_gpu = torch.tensor(std, device=device, dtype=torch.float16).view(1, 1, 3)

        image_gpu = torch.from_numpy(image).to(device).half() / 255.0
        result_gpu = (image_gpu - mean_gpu) / std_gpu
        result_gpu = result_gpu.permute(2, 0, 1).contiguous()

        result = result_gpu.float().cpu().numpy()
        expected = self._numpy_preprocess(image, mean, std)

        assert result.shape == (3, 224, 224)
        # FP16 has lower precision, so use a wider tolerance
        np.testing.assert_allclose(result, expected, rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# Unified vs Explicit memory: identical results
# ---------------------------------------------------------------------------

@requires_cuda
class TestUnifiedVsExplicitResults:
    """Verify that unified and explicit memory paths produce identical results."""

    def test_vector_add_identical(self):
        """Vector addition should produce identical results regardless of
        allocation strategy."""
        n = 1024 * 1024
        device = torch.device("cuda:0")

        # Explicit: allocate directly on GPU
        a_explicit = torch.randn(n, device=device)
        b_explicit = torch.randn(n, device=device)
        result_explicit = (a_explicit + b_explicit).cpu().numpy()

        # Unified-style: allocate on CPU, transfer to GPU
        a_cpu = a_explicit.cpu()
        b_cpu = b_explicit.cpu()
        a_unified = a_cpu.to(device)
        b_unified = b_cpu.to(device)
        result_unified = (a_unified + b_unified).cpu().numpy()

        np.testing.assert_array_equal(result_explicit, result_unified)

    def test_matmul_identical(self):
        """Matrix multiplication should produce identical results for both
        allocation strategies."""
        n = 256
        device = torch.device("cuda:0")

        # Create reference data
        a_data = np.random.randn(n, n).astype(np.float32)
        b_data = np.random.randn(n, n).astype(np.float32)

        # Explicit path
        a_gpu = torch.from_numpy(a_data).to(device)
        b_gpu = torch.from_numpy(b_data).to(device)
        result_explicit = (a_gpu @ b_gpu).cpu().numpy()

        # Unified-style path (CPU tensor -> GPU)
        a_cpu = torch.from_numpy(a_data.copy())
        b_cpu = torch.from_numpy(b_data.copy())
        result_unified = (a_cpu.to(device) @ b_cpu.to(device)).cpu().numpy()

        np.testing.assert_allclose(
            result_explicit, result_unified, rtol=1e-5, atol=1e-5
        )

    def test_preprocess_identical(self):
        """Preprocessing should produce identical results regardless of how
        memory was allocated."""
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        device = torch.device("cuda:0")

        mean_t = torch.tensor(mean, device=device).view(1, 1, 3)
        std_t = torch.tensor(std, device=device).view(1, 1, 3)

        def preprocess_on_gpu(img_tensor):
            r = img_tensor.float() / 255.0
            r = (r - mean_t) / std_t
            return r.permute(2, 0, 1).contiguous()

        # Path 1: allocate on GPU directly
        img_gpu = torch.from_numpy(image).to(device)
        result1 = preprocess_on_gpu(img_gpu).cpu().numpy()

        # Path 2: allocate on CPU, then transfer (unified-style)
        img_cpu = torch.from_numpy(image.copy())
        img_transferred = img_cpu.to(device)
        result2 = preprocess_on_gpu(img_transferred).cpu().numpy()

        np.testing.assert_array_equal(result1, result2)


# ---------------------------------------------------------------------------
# Power mode detection (Jetson-only)
# ---------------------------------------------------------------------------

@requires_jetson
class TestPowerModeDetection:
    """Tests for power mode detection on Jetson hardware."""

    def test_nvpmodel_available(self):
        """nvpmodel command should be available on Jetson."""
        try:
            result = subprocess.run(
                ["which", "nvpmodel"],
                capture_output=True, text=True, timeout=5,
            )
            assert result.returncode == 0, "nvpmodel not found in PATH"
        except (subprocess.SubprocessError, FileNotFoundError):
            pytest.fail("Could not check for nvpmodel")

    def test_power_mode_query(self):
        """Querying power mode should return non-empty output."""
        try:
            result = subprocess.run(
                ["nvpmodel", "-q"],
                capture_output=True, text=True, timeout=5,
            )
            # nvpmodel -q may require sudo; check if we got output
            if result.returncode == 0:
                assert len(result.stdout.strip()) > 0
            else:
                pytest.skip("nvpmodel -q requires elevated privileges")
        except (subprocess.SubprocessError, FileNotFoundError):
            pytest.skip("nvpmodel not available")

    def test_detect_jetson_returns_power_mode(self):
        """detect_jetson() should include power_mode field on Jetson."""
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_jetson import detect_jetson

        info = detect_jetson()
        assert info is not None, "detect_jetson() returned None on Jetson"
        assert "power_mode" in info

    def test_tegrastats_available(self):
        """tegrastats should be available on Jetson."""
        try:
            result = subprocess.run(
                ["which", "tegrastats"],
                capture_output=True, text=True, timeout=5,
            )
            assert result.returncode == 0, "tegrastats not found in PATH"
        except (subprocess.SubprocessError, FileNotFoundError):
            pytest.fail("Could not check for tegrastats")

    def test_clock_frequencies_readable(self):
        """At least one clock frequency should be readable on Jetson."""
        sys.path.insert(0, str(Path(__file__).parent))
        from power_mode_demo import get_clock_frequencies

        freqs = get_clock_frequencies()
        assert len(freqs) > 0, "No clock frequencies could be read"

    def test_device_tree_model(self):
        """/proc/device-tree/model should contain a Jetson identifier."""
        model_path = Path("/proc/device-tree/model")
        assert model_path.exists()

        model_str = model_path.read_text().strip().rstrip("\x00").lower()
        assert "jetson" in model_str or "nvidia" in model_str


# ---------------------------------------------------------------------------
# Cross-module integration
# ---------------------------------------------------------------------------

class TestCrossModuleIntegration:
    """Test that lesson modules work together correctly."""

    def test_benchmark_jetson_detect_matches_power_demo(self):
        """Both modules should agree on whether this is a Jetson device."""
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_jetson import detect_jetson
        from power_mode_demo import is_jetson

        jetson_info = detect_jetson()
        jetson_flag = is_jetson()

        if jetson_info is not None:
            assert jetson_flag is True
        else:
            assert jetson_flag is False

    @requires_cuda
    def test_benchmark_functions_run_without_error(self):
        """Benchmark functions should execute without raising exceptions."""
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_jetson import (
            benchmark_unified_vs_explicit,
            benchmark_fused_preprocess,
        )

        # These should run without error on any CUDA system
        benchmark_unified_vs_explicit()
        benchmark_fused_preprocess()
