"""
Integration tests for Lesson 7 GPU pipeline.

Tests:
  - Full pipeline: preprocess -> mock inference -> postprocess
  - GPU pipeline produces same results as CPU pipeline
  - Pinned memory pool under sustained load

Run:
    pytest test_integration_gpu.py -v

Tests requiring GPU are skipped automatically when no CUDA device is available.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add build directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "build"))

# ─── GPU availability check ──────────────────────────────────────────────────

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

requires_gpu = pytest.mark.skipif(not HAS_CUDA, reason="No CUDA GPU available")
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ─── Helpers ──────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def numpy_preprocess(image, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reference numpy preprocessing."""
    img = image.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    img = img.transpose(2, 0, 1)
    return np.ascontiguousarray(img)


# ─── Full Pipeline Tests ─────────────────────────────────────────────────────

class TestFullPipeline:
    """Test the complete preprocess -> inference -> postprocess pipeline."""

    @requires_torch
    def test_cpu_pipeline_end_to_end(self):
        """Full pipeline on CPU using numpy preprocess + PyTorch inference."""
        import torch
        import torch.nn as nn

        # Simple model
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 4),
        ).eval()

        # Preprocess
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        preprocessed = numpy_preprocess(image)

        # Inference
        with torch.inference_mode():
            tensor = torch.from_numpy(preprocessed).unsqueeze(0).float()
            output = model(tensor)

        # Postprocess
        result = output.numpy()
        assert result.shape == (1, 4)
        assert np.all(np.isfinite(result))

    @requires_gpu
    def test_gpu_pipeline_end_to_end(self):
        """Full pipeline on GPU: preprocess -> inference -> postprocess."""
        import torch
        import torch.nn as nn

        device = torch.device("cuda")

        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 4),
        ).to(device).eval()

        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        preprocessed = numpy_preprocess(image)

        with torch.inference_mode():
            tensor = torch.from_numpy(preprocessed).unsqueeze(0).float().to(device)
            output = model(tensor)
            result = output.cpu().numpy()

        assert result.shape == (1, 4)
        assert np.all(np.isfinite(result))

    @requires_gpu
    def test_gpu_matches_cpu_pipeline(self):
        """GPU pipeline should produce the same results as CPU pipeline."""
        import torch
        import torch.nn as nn

        # Use deterministic model
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 4),
        ).eval()

        image = np.random.RandomState(42).randint(0, 256, (32, 32, 3), dtype=np.uint8)
        preprocessed = numpy_preprocess(image)
        tensor = torch.from_numpy(preprocessed).unsqueeze(0).float()

        # CPU inference
        with torch.inference_mode():
            cpu_result = model(tensor).numpy()

        # GPU inference with same model
        device = torch.device("cuda")
        model_gpu = model.to(device)

        with torch.inference_mode():
            gpu_result = model_gpu(tensor.to(device)).cpu().numpy()

        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-4, atol=1e-4)


# ─── Preprocess Consistency Tests ─────────────────────────────────────────────

class TestPreprocessConsistency:
    """Verify C++ preprocess modules match numpy reference."""

    def test_gpu_module_matches_cpu_module(self):
        """gpu_preprocess and gpu_preprocess_cpu_ref should give same results."""
        try:
            import gpu_preprocess
            import gpu_preprocess_cpu_ref
        except ImportError:
            pytest.skip("Preprocess modules not built")

        rng = np.random.RandomState(123)
        image = rng.randint(0, 256, (48, 64, 3), dtype=np.uint8)

        gpu_result = np.asarray(gpu_preprocess.fused_preprocess(
            image, IMAGENET_MEAN, IMAGENET_STD
        ))
        cpu_result = np.asarray(gpu_preprocess_cpu_ref.fused_preprocess(
            image, IMAGENET_MEAN, IMAGENET_STD
        ))

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5, atol=1e-5)

    def test_cpp_matches_numpy_multiple_images(self):
        """C++ preprocess should match numpy for many different images."""
        try:
            import gpu_preprocess_cpu_ref as cpu_ref
        except ImportError:
            pytest.skip("gpu_preprocess_cpu_ref not built")

        rng = np.random.RandomState(777)

        for _ in range(20):
            h = rng.randint(8, 128)
            w = rng.randint(8, 128)
            image = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)

            expected = numpy_preprocess(image)
            result = np.asarray(cpu_ref.fused_preprocess(
                image, IMAGENET_MEAN, IMAGENET_STD
            ))

            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5,
                                       err_msg=f"Mismatch for image shape ({h}, {w}, 3)")


# ─── Pinned Memory Sustained Load Tests ──────────────────────────────────────

class TestPinnedMemorySustainedLoad:
    """Test pinned memory pool under sustained load."""

    @pytest.fixture(autouse=True)
    def _load_pool(self):
        try:
            from pinned_allocator import PinnedBufferPool
            self.Pool = PinnedBufferPool
        except ImportError:
            pytest.skip("pinned_allocator not built")

    def test_1000_acquire_release_cycles(self):
        """Pool should handle 1000 acquire/release cycles without issues."""
        buffer_size = 640 * 480 * 3  # Realistic frame size
        pool = self.Pool(n_buffers=4, buffer_size=buffer_size)

        for i in range(1000):
            arr, idx = pool.acquire_with_index()
            # Write some data to exercise the memory
            view = np.asarray(arr)
            view[0] = i % 256
            view[-1] = (i + 1) % 256
            pool.release(idx)

        assert pool.available_count() == 4

    def test_sustained_multi_buffer_rotation(self):
        """Rotate through multiple buffers under sustained load."""
        pool = self.Pool(n_buffers=3, buffer_size=1024)

        # Acquire all 3
        held = []
        for _ in range(3):
            arr, idx = pool.acquire_with_index()
            held.append((arr, idx))

        assert pool.available_count() == 0

        # Rotate: release oldest, acquire new, 1000 times
        for i in range(1000):
            # Release the oldest
            _, old_idx = held.pop(0)
            pool.release(old_idx)

            # Acquire a new one
            arr, idx = pool.acquire_with_index()
            view = np.asarray(arr)
            view[:] = i % 256
            held.append((arr, idx))

        # Release all remaining
        for _, idx in held:
            pool.release(idx)

        assert pool.available_count() == 3

    def test_concurrent_pattern(self):
        """Simulate a double-buffering pattern (producer/consumer)."""
        pool = self.Pool(n_buffers=2, buffer_size=4096)

        for frame in range(500):
            # "Producer" acquires buffer, fills it
            buf, idx = pool.acquire_with_index()
            view = np.asarray(buf)
            view[:10] = frame % 256  # Simulate writing frame data

            # "Consumer" processes and releases
            assert view[0] == frame % 256
            pool.release(idx)

        assert pool.available_count() == 2

    @requires_gpu
    def test_pinned_memory_gpu_transfer(self):
        """Test that pinned pool buffers can actually be transferred to GPU."""
        import torch

        buffer_size = 1024
        pool = self.Pool(n_buffers=2, buffer_size=buffer_size)

        for _ in range(100):
            arr, idx = pool.acquire_with_index()
            view = np.asarray(arr)
            view[:] = 42

            # Transfer to GPU via torch
            tensor = torch.from_numpy(view).float()
            gpu_tensor = tensor.cuda()
            assert gpu_tensor.sum().item() == 42.0 * buffer_size

            pool.release(idx)


# ─── GPU Stream Overlap Tests ─────────────────────────────────────────────────

class TestGpuStreamOverlap:
    """Test CUDA stream usage patterns."""

    @requires_gpu
    def test_multi_stream_execution(self):
        """Multiple CUDA streams should execute without errors."""
        import torch

        device = torch.device("cuda")
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)

        with torch.cuda.stream(stream1):
            c = torch.matmul(a, b)

        with torch.cuda.stream(stream2):
            d = torch.matmul(b, a)

        torch.cuda.synchronize()

        assert c.shape == (1000, 1000)
        assert d.shape == (1000, 1000)

    @requires_gpu
    def test_pinned_async_transfer(self):
        """Async transfer from pinned memory should produce correct results."""
        import torch

        data = torch.randn(1000, pin_memory=True)
        data.fill_(3.14)

        stream = torch.cuda.Stream()
        gpu_data = torch.empty(1000, device="cuda")

        with torch.cuda.stream(stream):
            gpu_data.copy_(data, non_blocking=True)

        stream.synchronize()

        result = gpu_data.cpu()
        np.testing.assert_allclose(result.numpy(), 3.14, rtol=1e-5)


# ─── Batch Inference Tests ───────────────────────────────────────────────────

class TestBatchInference:
    """Test batched vs sequential inference correctness."""

    @requires_torch
    def test_batched_matches_sequential(self):
        """Batched inference should give same results as sequential."""
        import torch
        import torch.nn as nn

        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 1),
        ).to(device).eval()

        n_candidates = 5
        candidates = [torch.randn(1, 3, 32, 32, device=device) for _ in range(n_candidates)]

        # Sequential
        with torch.inference_mode():
            sequential_results = [model(c) for c in candidates]
            sequential = torch.cat(sequential_results, dim=0)

        # Batched
        with torch.inference_mode():
            batch = torch.cat(candidates, dim=0)
            batched = model(batch)

        np.testing.assert_allclose(
            sequential.cpu().numpy(),
            batched.cpu().numpy(),
            rtol=1e-5, atol=1e-5,
        )
