#!/usr/bin/env python3
"""
Concrete tracker_engine GPU Anti-Pattern Fixes — Practical addition to Lesson 7.

Shows specific before/after code for five common anti-patterns found in
tracker_engine, with timing comparisons where possible.

Run:
    python tracker_engine_fixes.py
"""

import sys
import time
import textwrap

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required.  pip install numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def bench(func, *args, warmup: int = 5, repeats: int = 50, **kwargs) -> float:
    """Return median time in microseconds."""
    for _ in range(warmup):
        func(*args, **kwargs)
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        func(*args, **kwargs)
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


def print_fix(number: int, title: str, before: str, after: str,
              us_before: float | None = None, us_after: float | None = None):
    """Pretty-print a before/after fix."""
    print(f"\n{'=' * 70}")
    print(f"  Fix {number}: {title}")
    print(f"{'=' * 70}")
    print()
    print("  BEFORE (anti-pattern):")
    for line in before.strip().splitlines():
        print(f"    {line}")
    print()
    print("  AFTER (fixed):")
    for line in after.strip().splitlines():
        print(f"    {line}")
    print()
    if us_before is not None and us_after is not None:
        speedup = us_before / us_after if us_after > 0 else float("inf")
        print(f"  Timing: {us_before:.1f} us -> {us_after:.1f} us ({speedup:.1f}x faster)")
    elif us_before is None and us_after is None:
        print("  (torch not available — skipping timing)")
    print()


# ---------------------------------------------------------------------------
# Fix 1: prepare_boxes — drop unnecessary .cpu() calls
# ---------------------------------------------------------------------------

def fix1_prepare_boxes():
    before_code = """\
def prepare_boxes(boxes_tensor, device):
    # ANTI-PATTERN: unnecessary .cpu() then back to device
    boxes_np = boxes_tensor.cpu().numpy()
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    w = boxes_np[:, 2] - x1
    h = boxes_np[:, 3] - y1
    result = np.stack([x1, y1, w, h], axis=1)
    return torch.from_numpy(result).float().to(device)"""

    after_code = """\
def prepare_boxes(boxes_tensor, device):
    # FIXED: stay on GPU, no copies
    x1 = boxes_tensor[:, 0]
    y1 = boxes_tensor[:, 1]
    w = boxes_tensor[:, 2] - x1
    h = boxes_tensor[:, 3] - y1
    return torch.stack([x1, y1, w, h], dim=1).float()"""

    us_before = None
    us_after = None

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        boxes = torch.rand(100, 4, device=device) * 1000

        def slow(b):
            boxes_np = b.cpu().numpy()
            x1, y1 = boxes_np[:, 0], boxes_np[:, 1]
            w = boxes_np[:, 2] - x1
            h = boxes_np[:, 3] - y1
            result = np.stack([x1, y1, w, h], axis=1)
            return torch.from_numpy(result).float().to(device)

        def fast(b):
            x1, y1 = b[:, 0], b[:, 1]
            w = b[:, 2] - x1
            h = b[:, 3] - y1
            return torch.stack([x1, y1, w, h], dim=1).float()

        us_before = bench(slow, boxes)
        us_after = bench(fast, boxes)

    print_fix(1, "prepare_boxes: drop unnecessary .cpu() calls",
              before_code, after_code, us_before, us_after)


# ---------------------------------------------------------------------------
# Fix 2: os_tracker_forward — remove redundant .to(device)
# ---------------------------------------------------------------------------

def fix2_redundant_to():
    before_code = """\
def os_tracker_forward(model, template, search, device):
    # ANTI-PATTERN: tensor is already on device, .to() still synchronizes
    template = template.to(device)  # already on device!
    search = search.to(device)      # already on device!
    return model(template, search)"""

    after_code = """\
def os_tracker_forward(model, template, search, device):
    # FIXED: skip .to() if already on the right device
    if template.device != device:
        template = template.to(device)
    if search.device != device:
        search = search.to(device)
    return model(template, search)

# Or even simpler — .to() on same device is a no-op in recent PyTorch,
# but the explicit check avoids the method call overhead entirely."""

    us_before = None
    us_after = None

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.rand(1, 3, 128, 128, device=device)

        def slow(tensor):
            return tensor.to(device)

        def fast(tensor):
            if tensor.device == device:
                return tensor
            return tensor.to(device)

        us_before = bench(slow, t)
        us_after = bench(fast, t)

    print_fix(2, "os_tracker_forward: remove redundant .to(device)",
              before_code, after_code, us_before, us_after)


# ---------------------------------------------------------------------------
# Fix 3: TRT_Preprocessor — use torch ops instead of numpy
# ---------------------------------------------------------------------------

def fix3_trt_preprocessor():
    before_code = """\
def preprocess(image_tensor, mean, std):
    # ANTI-PATTERN: GPU -> CPU -> numpy -> CPU -> GPU
    img_np = image_tensor.cpu().numpy()
    img_np = (img_np - mean) / std
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(img_np).float().cuda()"""

    after_code = """\
def preprocess(image_tensor, mean, std):
    # FIXED: stay on GPU throughout
    img = (image_tensor - mean) / std
    img = img.permute(2, 0, 1)  # HWC -> CHW
    return img.float()"""

    us_before = None
    us_after = None

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = torch.rand(480, 640, 3, device=device)
        mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std_np = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        mean_t = torch.tensor([0.485, 0.456, 0.406], device=device)
        std_t = torch.tensor([0.229, 0.224, 0.225], device=device)

        def slow(image):
            img_np = image.cpu().numpy()
            img_np = (img_np - mean_np) / std_np
            img_np = img_np.transpose(2, 0, 1)
            return torch.from_numpy(img_np).float().to(device)

        def fast(image):
            out = (image - mean_t) / std_t
            out = out.permute(2, 0, 1)
            return out.float()

        us_before = bench(slow, img)
        us_after = bench(fast, img)

    print_fix(3, "TRT_Preprocessor: use torch ops instead of numpy",
              before_code, after_code, us_before, us_after)


# ---------------------------------------------------------------------------
# Fix 4: phase_cross_correlation — use .item() for scalars
# ---------------------------------------------------------------------------

def fix4_item_scalar():
    before_code = """\
def get_shift(correlation_tensor):
    # ANTI-PATTERN: 4 operations to extract a single scalar
    peak = correlation_tensor.max()
    value = peak.clone().cpu().numpy().tolist()
    return value"""

    after_code = """\
def get_shift(correlation_tensor):
    # FIXED: .item() — single call, returns Python scalar directly
    return correlation_tensor.max().item()"""

    us_before = None
    us_after = None

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        corr = torch.rand(256, 256, device=device)

        def slow(t):
            peak = t.max()
            return peak.clone().cpu().numpy().tolist()

        def fast(t):
            return t.max().item()

        us_before = bench(slow, corr)
        us_after = bench(fast, corr)

    print_fix(4, "phase_cross_correlation: use .item() for scalar extraction",
              before_code, after_code, us_before, us_after)


# ---------------------------------------------------------------------------
# Fix 5: torch.no_grad -> torch.inference_mode
# ---------------------------------------------------------------------------

def fix5_inference_mode():
    before_code = """\
with torch.no_grad():
    output = model(input_tensor)
# no_grad disables gradient computation but still tracks
# tensor metadata for autograd (version counters, etc.)"""

    after_code = """\
with torch.inference_mode():
    output = model(input_tensor)
# inference_mode is stricter: disables autograd entirely,
# skips version counter bumps, enables more optimisations.
# Tensors created inside cannot be used for backward() —
# which is exactly what we want during inference."""

    us_before = None
    us_after = None

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.rand(64, 512, device=device)
        w = torch.rand(512, 512, device=device)

        def with_no_grad():
            with torch.no_grad():
                a = torch.matmul(x, w)
                b = torch.relu(a)
                c = torch.matmul(b, w)
                return c

        def with_inference_mode():
            with torch.inference_mode():
                a = torch.matmul(x, w)
                b = torch.relu(a)
                c = torch.matmul(b, w)
                return c

        us_before = bench(with_no_grad)
        us_after = bench(with_inference_mode)

    print_fix(5, "torch.no_grad -> torch.inference_mode",
              before_code, after_code, us_before, us_after)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("tracker_engine GPU Anti-Pattern Fixes")
    print()
    if HAS_TORCH:
        print(f"  PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        else:
            print("  GPU:     not available (running on CPU)")
    else:
        print("  PyTorch: not installed (showing code patterns only)")

    fix1_prepare_boxes()
    fix2_redundant_to()
    fix3_trt_preprocessor()
    fix4_item_scalar()
    fix5_inference_mode()

    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print()
    print("  Common theme: avoid unnecessary GPU<->CPU round-trips.")
    print("  Each .cpu(), .numpy(), .to(device) call can trigger a")
    print("  CUDA synchronization point, stalling the GPU pipeline.")
    print()
    print("  Rules of thumb:")
    print("  1. Stay on GPU: do math with torch ops, not numpy")
    print("  2. Use .item() to extract scalars")
    print("  3. Guard .to(device) with a device check")
    print("  4. Use inference_mode() over no_grad() for inference")
    print("  5. Profile with torch.profiler to find hidden syncs")
    print()


if __name__ == "__main__":
    main()
