"""
Benchmark GPU vs CPU preprocessing, pinned memory, batching, and torch.compile.

Run from the build directory:
    python ../benchmark_gpu.py

Falls back gracefully to CPU-only benchmarks when no GPU is available.
"""

import time
import sys
from pathlib import Path

import numpy as np

# ─── Helpers ──────────────────────────────────────────────────────────────────

def time_fn(fn, n_iters=100, warmup=10):
    """Time a function over n_iters, returning mean and std in milliseconds."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    arr = np.array(times)
    return arr.mean(), arr.std()


def print_table(title, rows):
    """Print a formatted benchmark results table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Method':<35} {'Mean (ms)':>10} {'Std (ms)':>10} {'Speedup':>8}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 8}")

    baseline = rows[0][1] if rows else 1.0
    for name, mean, std in rows:
        speedup = baseline / mean if mean > 0 else float('inf')
        print(f"  {name:<35} {mean:>10.3f} {std:>10.3f} {speedup:>7.1f}x")
    print()


# ─── Benchmark 1: CPU vs GPU Preprocess ───────────────────────────────────────

def benchmark_preprocess():
    """Compare CPU numpy-style preprocess vs C++ fused preprocess (CPU or GPU)."""
    title = "Benchmark 1: Preprocessing — CPU numpy vs Fused Kernel"

    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Simulate a 640x480 RGB image (typical tracker input)
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    rows = []

    # Numpy baseline (what tracker_engine does)
    def numpy_preprocess():
        img = image.astype(np.float32) / 255.0
        img = (img - np.array(mean)) / np.array(std)
        img = img.transpose(2, 0, 1)
        return np.ascontiguousarray(img)

    m, s = time_fn(numpy_preprocess)
    rows.append(("NumPy (tracker_engine style)", m, s))

    # Try C++ CPU reference
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            import gpu_preprocess_cpu_ref as cpu_ref
        except ImportError:
            # Try build directory
            sys.path.insert(0, str(Path(__file__).parent / "build"))
            import gpu_preprocess_cpu_ref as cpu_ref

        def cpp_cpu_preprocess():
            return cpu_ref.fused_preprocess(image, mean, std)

        m, s = time_fn(cpp_cpu_preprocess)
        rows.append(("C++ CPU fused (std::transform)", m, s))

        # Numpy-style through C++
        if hasattr(cpu_ref, 'numpy_style_preprocess'):
            def cpp_numpy_style():
                return cpu_ref.numpy_style_preprocess(image, mean, std)
            m, s = time_fn(cpp_numpy_style)
            rows.append(("C++ CPU 3-step (numpy-like)", m, s))

    except ImportError:
        rows.append(("C++ CPU fused", float('nan'), float('nan')))
        print("  [SKIP] gpu_preprocess_cpu_ref not built")

    # Try GPU version
    try:
        try:
            import gpu_preprocess as gpu_mod
        except ImportError:
            sys.path.insert(0, str(Path(__file__).parent / "build"))
            import gpu_preprocess as gpu_mod

        if gpu_mod.cuda_available():
            def gpu_preprocess():
                return gpu_mod.fused_preprocess(image, mean, std)

            m, s = time_fn(gpu_preprocess)
            rows.append(("CUDA fused kernel", m, s))
        else:
            rows.append(("CUDA fused kernel", float('nan'), float('nan')))
            print("  [SKIP] No CUDA GPU detected")

    except ImportError:
        print("  [SKIP] gpu_preprocess module not built")

    print_table(title, rows)


# ─── Benchmark 2: Regular vs Pinned Memory ───────────────────────────────────

def benchmark_pinned_memory():
    """Compare regular allocation vs pinned memory pool for transfer simulation."""
    title = "Benchmark 2: Memory Allocation — Per-frame vs Pinned Pool"

    buffer_size = 640 * 480 * 3  # Typical frame size
    rows = []

    # Baseline: per-frame numpy allocation (what tracker_engine does)
    def per_frame_alloc():
        buf = np.empty(buffer_size, dtype=np.uint8)
        buf[:] = 42  # Simulate filling
        return buf

    m, s = time_fn(per_frame_alloc, n_iters=500)
    rows.append(("Per-frame np.empty + fill", m, s))

    # Pinned pool
    try:
        try:
            from pinned_allocator import PinnedBufferPool
        except ImportError:
            sys.path.insert(0, str(Path(__file__).parent / "build"))
            from pinned_allocator import PinnedBufferPool

        pool = PinnedBufferPool(n_buffers=4, buffer_size=buffer_size)

        def pooled_alloc():
            arr, idx = pool.acquire_with_index()
            np.asarray(arr)[:] = 42  # Simulate filling
            pool.release(idx)

        m, s = time_fn(pooled_alloc, n_iters=500)
        rows.append(("Pinned pool acquire/release", m, s))
        print(f"  Pool info: {pool}")

    except ImportError:
        print("  [SKIP] pinned_allocator not built")

    # PyTorch pinned memory comparison (if available)
    try:
        import torch

        def torch_regular():
            t = torch.empty(buffer_size, dtype=torch.uint8)
            t.fill_(42)
            return t

        m, s = time_fn(torch_regular, n_iters=500)
        rows.append(("torch.empty (pageable)", m, s))

        def torch_pinned():
            t = torch.empty(buffer_size, dtype=torch.uint8, pin_memory=True)
            t.fill_(42)
            return t

        m, s = time_fn(torch_pinned, n_iters=500)
        rows.append(("torch.empty (pin_memory=True)", m, s))

        if torch.cuda.is_available():
            # Measure actual transfer speed
            pinned_t = torch.empty(buffer_size, dtype=torch.uint8, pin_memory=True)
            regular_t = torch.empty(buffer_size, dtype=torch.uint8)
            pinned_t.fill_(42)
            regular_t.fill_(42)

            def transfer_regular():
                return regular_t.cuda(non_blocking=False)

            def transfer_pinned():
                return pinned_t.cuda(non_blocking=False)

            m, s = time_fn(transfer_regular, n_iters=200)
            rows.append(("Transfer pageable → GPU", m, s))

            m, s = time_fn(transfer_pinned, n_iters=200)
            rows.append(("Transfer pinned → GPU", m, s))

    except ImportError:
        print("  [SKIP] PyTorch not available for memory comparison")

    print_table(title, rows)


# ─── Benchmark 3: Sequential vs Batched Inference ────────────────────────────

def benchmark_batching():
    """Compare one-by-one inference vs batched inference using PyTorch."""
    title = "Benchmark 3: Sequential vs Batched Inference"

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")
        print("  [SKIP] PyTorch not available\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    # Simple conv network as proxy for a validation model
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 1),
    ).to(device).eval()

    n_candidates = 10
    candidate_size = (3, 64, 64)

    candidates = [torch.randn(1, *candidate_size, device=device) for _ in range(n_candidates)]
    batch = torch.cat(candidates, dim=0)

    # Sequential (tracker_engine track_restoration style)
    def sequential_inference():
        results = []
        with torch.inference_mode():
            for c in candidates:
                results.append(model(c))
        return results

    m, s = time_fn(sequential_inference, n_iters=50, warmup=5)
    rows.append((f"Sequential ({n_candidates} candidates)", m, s))

    # Batched
    def batched_inference():
        with torch.inference_mode():
            return model(batch)

    m, s = time_fn(batched_inference, n_iters=50, warmup=5)
    rows.append((f"Batched ({n_candidates} candidates)", m, s))

    print_table(title, rows)


# ─── Benchmark 4: torch.compile Effect ───────────────────────────────────────

def benchmark_torch_compile():
    """Measure torch.compile speedup on a post-processing function."""
    title = "Benchmark 4: torch.compile on Post-processing"

    try:
        import torch
    except ImportError:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")
        print("  [SKIP] PyTorch not available\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    # Simulate post-processing: NMS-like score filtering + box adjustment
    def postprocess(boxes, scores, threshold=0.5):
        mask = scores > threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        # Scale boxes
        filtered_boxes = filtered_boxes * 2.0 - 0.5
        # Clamp
        filtered_boxes = torch.clamp(filtered_boxes, 0.0, 1.0)
        return filtered_boxes, filtered_scores

    boxes = torch.rand(1000, 4, device=device)
    scores = torch.rand(1000, device=device)

    m, s = time_fn(lambda: postprocess(boxes, scores), n_iters=200, warmup=20)
    rows.append(("Eager postprocess", m, s))

    try:
        compiled_postprocess = torch.compile(postprocess)
        # Warmup for compilation
        for _ in range(5):
            compiled_postprocess(boxes, scores)

        m, s = time_fn(lambda: compiled_postprocess(boxes, scores), n_iters=200, warmup=20)
        rows.append(("torch.compile postprocess", m, s))
    except Exception as e:
        print(f"  [SKIP] torch.compile failed: {e}")

    print_table(title, rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Lesson 7 — GPU Programming Benchmarks")
    print(f"Image size: 640x480x3 (uint8)")

    # System info
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: not installed")

    benchmark_preprocess()
    benchmark_pinned_memory()
    benchmark_batching()
    benchmark_torch_compile()

    print("=" * 70)
    print("  All benchmarks complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
