"""
Unit tests for Lesson 7 GPU programming modules.

Tests:
  - gpu_preprocess: fused kernel output matches CPU reference
  - pinned_allocator: acquire, release, reuse cycle
  - Graceful skip when GPU/modules not available

Run:
    pytest test_gpu.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add build directory to path for compiled modules
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "build"))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_image():
    """A small test image (deterministic)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (64, 80, 3), dtype=np.uint8)


@pytest.fixture
def imagenet_params():
    """ImageNet normalization parameters."""
    return {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


def cpu_reference_preprocess(image, mean, std):
    """Pure numpy reference implementation for correctness checks."""
    img = image.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    img = img.transpose(2, 0, 1)
    return np.ascontiguousarray(img)


# ─── GPU Preprocess Tests ─────────────────────────────────────────────────────

class TestGpuPreprocess:
    """Tests for the fused preprocess kernel."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        try:
            import gpu_preprocess
            self.mod = gpu_preprocess
        except ImportError:
            pytest.skip("gpu_preprocess module not built")

    def test_output_shape(self, sample_image, imagenet_params):
        """Output should be CHW float32."""
        result = self.mod.fused_preprocess(
            sample_image, imagenet_params["mean"], imagenet_params["std"]
        )
        result = np.asarray(result)
        h, w, c = sample_image.shape
        assert result.shape == (c, h, w), f"Expected {(c, h, w)}, got {result.shape}"
        assert result.dtype == np.float32

    def test_matches_numpy_reference(self, sample_image, imagenet_params):
        """GPU/C++ output should match numpy reference within float tolerance."""
        expected = cpu_reference_preprocess(
            sample_image, imagenet_params["mean"], imagenet_params["std"]
        )
        result = np.asarray(self.mod.fused_preprocess(
            sample_image, imagenet_params["mean"], imagenet_params["std"]
        ))
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_single_channel(self):
        """Should work with single-channel (grayscale) images."""
        gray = np.random.randint(0, 256, (32, 32, 1), dtype=np.uint8)
        result = np.asarray(self.mod.fused_preprocess(gray, [0.5], [0.5]))
        assert result.shape == (1, 32, 32)

    def test_various_sizes(self, imagenet_params):
        """Should handle different image sizes."""
        for h, w in [(1, 1), (16, 16), (480, 640), (100, 200)]:
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            result = np.asarray(self.mod.fused_preprocess(
                img, imagenet_params["mean"], imagenet_params["std"]
            ))
            assert result.shape == (3, h, w)

    def test_all_zeros(self, imagenet_params):
        """All-black image should produce -(mean/std) for each channel."""
        black = np.zeros((8, 8, 3), dtype=np.uint8)
        result = np.asarray(self.mod.fused_preprocess(
            black, imagenet_params["mean"], imagenet_params["std"]
        ))
        for c in range(3):
            expected_val = (0.0 - imagenet_params["mean"][c]) / imagenet_params["std"][c]
            np.testing.assert_allclose(result[c], expected_val, rtol=1e-5)

    def test_all_white(self, imagenet_params):
        """All-white (255) image should produce (1-mean)/std for each channel."""
        white = np.full((8, 8, 3), 255, dtype=np.uint8)
        result = np.asarray(self.mod.fused_preprocess(
            white, imagenet_params["mean"], imagenet_params["std"]
        ))
        for c in range(3):
            expected_val = (1.0 - imagenet_params["mean"][c]) / imagenet_params["std"][c]
            np.testing.assert_allclose(result[c], expected_val, rtol=1e-5)

    def test_invalid_mean_length(self, sample_image):
        """Should raise on mismatched mean/std length."""
        with pytest.raises((ValueError, RuntimeError)):
            self.mod.fused_preprocess(sample_image, [0.5, 0.5], [0.5, 0.5, 0.5])

    def test_cuda_available_returns_bool(self):
        """cuda_available() should return a bool."""
        result = self.mod.cuda_available()
        assert isinstance(result, bool)


# ─── CPU Reference Tests ─────────────────────────────────────────────────────

class TestCpuReference:
    """Tests for the CPU reference preprocess module."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        try:
            import gpu_preprocess_cpu_ref
            self.mod = gpu_preprocess_cpu_ref
        except ImportError:
            pytest.skip("gpu_preprocess_cpu_ref module not built")

    def test_matches_numpy(self, sample_image, imagenet_params):
        """C++ CPU reference should match numpy exactly."""
        expected = cpu_reference_preprocess(
            sample_image, imagenet_params["mean"], imagenet_params["std"]
        )
        result = np.asarray(self.mod.fused_preprocess(
            sample_image, imagenet_params["mean"], imagenet_params["std"]
        ))
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_numpy_style_preprocess(self, sample_image, imagenet_params):
        """The three-step numpy-style C++ version should also match."""
        if not hasattr(self.mod, 'numpy_style_preprocess'):
            pytest.skip("numpy_style_preprocess not available")

        expected = cpu_reference_preprocess(
            sample_image, imagenet_params["mean"], imagenet_params["std"]
        )
        result = np.asarray(self.mod.numpy_style_preprocess(
            sample_image, imagenet_params["mean"], imagenet_params["std"]
        ))
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_cuda_not_available(self):
        """CPU reference should report no CUDA."""
        assert self.mod.cuda_available() is False


# ─── Pinned Allocator Tests ──────────────────────────────────────────────────

class TestPinnedAllocator:
    """Tests for the pinned memory pool."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        try:
            from pinned_allocator import PinnedBufferPool
            self.Pool = PinnedBufferPool
        except ImportError:
            pytest.skip("pinned_allocator module not built")

    def test_create_pool(self):
        """Pool creation should succeed."""
        pool = self.Pool(n_buffers=4, buffer_size=1024)
        assert pool.total_count() == 4
        assert pool.available_count() == 4
        assert pool.buffer_size() == 1024

    def test_acquire_returns_numpy(self):
        """Acquired buffer should be a writable numpy-compatible array."""
        pool = self.Pool(n_buffers=2, buffer_size=256)
        buf = pool.acquire()
        arr = np.asarray(buf)
        assert arr.shape == (256,)
        assert arr.dtype == np.uint8
        # Should be writable
        arr[0] = 42
        assert arr[0] == 42

    def test_acquire_with_index(self):
        """acquire_with_index should return (array, index) tuple."""
        pool = self.Pool(n_buffers=2, buffer_size=128)
        arr, idx = pool.acquire_with_index()
        assert isinstance(idx, int)
        assert idx >= 0
        assert pool.available_count() == 1

    def test_release_makes_buffer_available(self):
        """Releasing a buffer should make it available again."""
        pool = self.Pool(n_buffers=2, buffer_size=128)
        _, idx = pool.acquire_with_index()
        assert pool.available_count() == 1
        pool.release(idx)
        assert pool.available_count() == 2

    def test_acquire_all_then_fail(self):
        """Acquiring more buffers than available should raise."""
        pool = self.Pool(n_buffers=2, buffer_size=64)
        pool.acquire()
        pool.acquire()
        with pytest.raises(RuntimeError, match="no buffers available"):
            pool.acquire()

    def test_reuse_cycle(self):
        """Buffers should be reusable after release."""
        pool = self.Pool(n_buffers=1, buffer_size=64)

        for i in range(10):
            arr, idx = pool.acquire_with_index()
            np.asarray(arr)[0] = i
            pool.release(idx)

        # Should still work after 10 cycles
        arr, idx = pool.acquire_with_index()
        assert pool.available_count() == 0
        pool.release(idx)
        assert pool.available_count() == 1

    def test_data_persists_in_buffer(self):
        """Data written to a buffer should persist until overwritten."""
        pool = self.Pool(n_buffers=1, buffer_size=16)

        arr, idx = pool.acquire_with_index()
        np.asarray(arr)[:] = 99
        pool.release(idx)

        arr2, idx2 = pool.acquire_with_index()
        # Same buffer should have our data
        assert np.asarray(arr2)[0] == 99
        pool.release(idx2)

    def test_invalid_index_raises(self):
        """Releasing an invalid index should raise."""
        pool = self.Pool(n_buffers=2, buffer_size=64)
        with pytest.raises((IndexError, RuntimeError)):
            pool.release(999)

    def test_zero_buffers_raises(self):
        """Creating a pool with 0 buffers should raise."""
        with pytest.raises((ValueError, RuntimeError)):
            self.Pool(n_buffers=0, buffer_size=64)

    def test_zero_size_raises(self):
        """Creating a pool with 0 buffer_size should raise."""
        with pytest.raises((ValueError, RuntimeError)):
            self.Pool(n_buffers=2, buffer_size=0)

    def test_repr(self):
        """Pool repr should contain useful info."""
        pool = self.Pool(n_buffers=3, buffer_size=512)
        r = repr(pool)
        assert "3" in r
        assert "512" in r

    def test_is_pinned_returns_bool(self):
        """is_pinned() should return a bool."""
        pool = self.Pool(n_buffers=1, buffer_size=64)
        assert isinstance(pool.is_pinned(), bool)
