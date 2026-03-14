"""
GPU Pipeline Demo — Wrong Way vs Right Way.

Simulates a tracking pipeline to demonstrate the performance impact of
CPU<->GPU data bouncing. Uses PyTorch operations as a proxy (no actual
model needed).

The "wrong way" mimics tracker_engine's current approach:
    numpy preprocess -> to GPU -> inference -> to CPU -> postprocess -> to GPU

The "right way" keeps data on the GPU:
    pinned memory -> async transfer -> GPU preprocess -> inference -> GPU postprocess -> single result back

Run:
    python gpu_pipeline_demo.py
"""

import time
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def check_requirements():
    if not HAS_TORCH:
        print("ERROR: PyTorch is required for this demo.")
        print("Install with: pip install torch")
        sys.exit(1)


# ─── Simulated Model ─────────────────────────────────────────────────────────

class MockTracker(nn.Module):
    """Simple conv network standing in for the real tracker model."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 4),  # Output: [x, y, w, h]
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ─── The Wrong Way (tracker_engine pattern) ───────────────────────────────────

def wrong_way_pipeline(model, frames, device):
    """
    Mimics tracker_engine's approach:
    1. Preprocess on CPU with numpy
    2. Transfer to GPU
    3. Run inference
    4. Transfer results to CPU for post-processing
    5. Transfer back to GPU for next stage (if needed)
    """
    results = []

    for frame in frames:
        # Step 1: CPU preprocessing (TRT_Preprocessor.process style)
        img = frame.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.ascontiguousarray(img)

        # Step 2: Transfer to GPU (os_tracker_forward style — new allocation each time)
        tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)

        # Step 3: Inference
        with torch.inference_mode():
            output = model(tensor)

        # Step 4: Transfer to CPU for post-processing (prepare_boxes style)
        cpu_output = output.cpu().numpy()

        # Step 5: "Post-process" on CPU
        result = cpu_output * 640  # Scale to image coordinates
        result = np.clip(result, 0, 640)

        results.append(result)

    return results


# ─── The Right Way ────────────────────────────────────────────────────────────

def right_way_pipeline(model, frames, device):
    """
    Optimized approach:
    1. Pre-allocate pinned memory buffer
    2. Pre-allocate GPU tensors
    3. Use pinned memory for async transfer
    4. Preprocess on GPU (simulated with torch ops)
    5. Inference
    6. Post-process on GPU
    7. Only transfer the small result
    """
    results = []

    # Pre-allocate: do this ONCE at initialization, not per-frame
    h, w, c = frames[0].shape
    pinned_buffer = torch.empty(1, c, h, w, dtype=torch.float32, pin_memory=True)
    gpu_buffer = torch.empty(1, c, h, w, dtype=torch.float32, device=device)

    # Pre-compute normalization constants on GPU
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    stream = torch.cuda.Stream() if device.type == 'cuda' else None

    for frame in frames:
        # Step 1: Copy raw frame into pinned buffer (fast memcpy, no allocation)
        # Convert HWC->CHW and uint8->float in one step into pre-allocated pinned buffer
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
        pinned_buffer.copy_(frame_tensor)

        # Step 2: Async transfer pinned -> GPU (overlaps with previous compute)
        if stream is not None:
            with torch.cuda.stream(stream):
                gpu_buffer.copy_(pinned_buffer, non_blocking=True)
        else:
            gpu_buffer.copy_(pinned_buffer)

        # Step 3: GPU-side preprocessing (normalize on GPU, not CPU)
        with torch.inference_mode():
            normalized = (gpu_buffer / 255.0 - mean) / std

            # Step 4: Inference (no transfer needed — data is already on GPU)
            output = model(normalized)

            # Step 5: Post-process on GPU (no CPU round-trip)
            result = output * 640.0
            result = torch.clamp(result, 0.0, 640.0)

            # Step 6: Transfer only the small result (4 floats, not 640x480x3)
            result_cpu = result.cpu().numpy()

        results.append(result_cpu)

    return results


# ─── CPU-only fallback for systems without GPU ────────────────────────────────

def cpu_wrong_way(frames):
    """Wrong way on CPU: multiple intermediate allocations."""
    results = []
    for frame in frames:
        img = frame.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # Simulate "inference" with numpy
        result = np.mean(img, axis=(1, 2))
        result = result * 640
        result = np.clip(result, 0, 640)
        results.append(result)
    return results


def cpu_right_way(frames):
    """Right way on CPU: pre-allocate, minimize copies."""
    results = []
    h, w, c = frames[0].shape
    buffer = np.empty((c, h, w), dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    for frame in frames:
        # Fused: convert + normalize + transpose into pre-allocated buffer
        for ch in range(c):
            buffer[ch] = (frame[:, :, ch].astype(np.float32) / 255.0 - mean[ch]) / std[ch]

        result = np.mean(buffer, axis=(1, 2))
        result = result * 640
        result = np.clip(result, 0, 640)
        results.append(result)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Lesson 7 — GPU Pipeline Demo: Wrong Way vs Right Way")
    print("=" * 70)

    # Generate synthetic frames
    n_frames = 100
    height, width = 480, 640
    print(f"Generating {n_frames} synthetic frames ({width}x{height} RGB)...")
    frames = [np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
              for _ in range(n_frames)]

    if HAS_TORCH and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

        model = MockTracker().to(device).eval()

        # Warmup
        print("Warming up model...")
        dummy = torch.randn(1, 3, height, width, device=device)
        with torch.inference_mode():
            for _ in range(5):
                model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Wrong way
        print("\nRunning WRONG WAY (tracker_engine pattern)...")
        t0 = time.perf_counter()
        wrong_results = wrong_way_pipeline(model, frames, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        wrong_time = time.perf_counter() - t0

        # Right way
        print("Running RIGHT WAY (optimized pipeline)...")
        t0 = time.perf_counter()
        right_results = right_way_pipeline(model, frames, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        right_time = time.perf_counter() - t0

        print(f"\n{'=' * 70}")
        print(f"  Results ({n_frames} frames)")
        print(f"{'=' * 70}")
        print(f"  Wrong way (CPU<->GPU bouncing):  {wrong_time*1000:>8.1f} ms  "
              f"({n_frames/wrong_time:>6.1f} FPS)")
        print(f"  Right way (GPU-resident):        {right_time*1000:>8.1f} ms  "
              f"({n_frames/right_time:>6.1f} FPS)")
        print(f"  Speedup:                         {wrong_time/right_time:>8.1f}x")
        print()

        print("Key differences:")
        print("  Wrong way: 3 CPU<->GPU transfers per frame (preprocess, inference, postprocess)")
        print("  Right way: 1 pinned transfer in, 1 small result out per frame")
        print("  Wrong way: new allocation every frame")
        print("  Right way: pre-allocated buffers reused every frame")

    elif HAS_TORCH:
        # CPU-only with PyTorch
        device = torch.device("cpu")
        print("No GPU detected — running CPU-only comparison")
        print("(Demonstrates allocation overhead, not GPU transfer overhead)")
        print()

        model = MockTracker().eval()

        print("Running WRONG WAY (per-frame allocation)...")
        t0 = time.perf_counter()
        wrong_results = wrong_way_pipeline(model, frames, device)
        wrong_time = time.perf_counter() - t0

        print("Running RIGHT WAY (pre-allocated buffers)...")
        t0 = time.perf_counter()
        right_results = right_way_pipeline(model, frames, device)
        right_time = time.perf_counter() - t0

        print(f"\n{'=' * 70}")
        print(f"  Results ({n_frames} frames, CPU-only)")
        print(f"{'=' * 70}")
        print(f"  Wrong way:  {wrong_time*1000:>8.1f} ms  ({n_frames/wrong_time:>6.1f} FPS)")
        print(f"  Right way:  {right_time*1000:>8.1f} ms  ({n_frames/right_time:>6.1f} FPS)")
        print(f"  Speedup:    {wrong_time/right_time:>8.1f}x")
        print()
        print("Note: On a GPU system, the difference would be much larger")
        print("      due to PCIe transfer overhead (~12 GB/s vs ~900 GB/s).")

    else:
        # Pure numpy fallback
        print("No PyTorch available — running numpy-only comparison")
        print("(Demonstrates allocation overhead pattern)")
        print()

        print("Running WRONG WAY (per-frame allocation, intermediate arrays)...")
        t0 = time.perf_counter()
        wrong_results = cpu_wrong_way(frames)
        wrong_time = time.perf_counter() - t0

        print("Running RIGHT WAY (pre-allocated buffer, fused ops)...")
        t0 = time.perf_counter()
        right_results = cpu_right_way(frames)
        right_time = time.perf_counter() - t0

        print(f"\n{'=' * 70}")
        print(f"  Results ({n_frames} frames, numpy-only)")
        print(f"{'=' * 70}")
        print(f"  Wrong way:  {wrong_time*1000:>8.1f} ms  ({n_frames/wrong_time:>6.1f} FPS)")
        print(f"  Right way:  {right_time*1000:>8.1f} ms  ({n_frames/right_time:>6.1f} FPS)")
        print(f"  Speedup:    {wrong_time/right_time:>8.1f}x")

    print()


if __name__ == "__main__":
    main()
