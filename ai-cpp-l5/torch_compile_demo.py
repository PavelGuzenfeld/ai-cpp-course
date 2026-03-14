"""
torch.compile() effect on tracker_engine post-processing.

tracker_engine has torch.compile() commented out in its inference pipeline.
This demo shows the performance difference by implementing a simplified
version of tracker_engine's prepare_boxes() post-processing:
  - Score thresholding
  - xywh-to-xyxy conversion
  - Simplified NMS filtering

The compile step fuses these operations into optimized kernels, avoiding
intermediate tensor allocations and Python overhead.

Usage:
    python3 torch_compile_demo.py
"""

from __future__ import annotations

import time
import sys


def _check_torch():
    """Return torch module or None if unavailable."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def prepare_boxes(predictions, score_threshold: float = 0.5):
    """Simulate tracker_engine's prepare_boxes post-processing.

    Args:
        predictions: tensor of shape (N, 5) — [x, y, w, h, score]
        score_threshold: minimum score to keep a detection

    Returns:
        (filtered_boxes_xyxy, filtered_scores)
    """
    import torch

    scores = predictions[:, 4]

    # Step 1: score thresholding
    mask = scores > score_threshold
    kept = predictions[mask]

    if kept.shape[0] == 0:
        return torch.empty(0, 4, device=predictions.device), torch.empty(0, device=predictions.device)

    x, y, w, h = kept[:, 0], kept[:, 1], kept[:, 2], kept[:, 3]
    kept_scores = kept[:, 4]

    # Step 2: xywh -> xyxy conversion
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0

    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    # Step 3: clamp to image bounds [0, 1]
    boxes_xyxy = torch.clamp(boxes_xyxy, 0.0, 1.0)

    # Step 4: simplified NMS — suppress boxes with high overlap
    # For each box, compute area; keep only boxes whose area is above a
    # minimum (filters degenerate boxes that collapse after clamping)
    widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    areas = widths * heights
    valid = areas > 1e-4
    boxes_xyxy = boxes_xyxy[valid]
    kept_scores = kept_scores[valid]

    # Sort by score descending (prep for NMS — we keep top-K)
    order = torch.argsort(kept_scores, descending=True)
    max_keep = min(100, order.shape[0])
    boxes_xyxy = boxes_xyxy[order[:max_keep]]
    kept_scores = kept_scores[order[:max_keep]]

    return boxes_xyxy, kept_scores


def time_function(fn, n_iters: int = 200, warmup: int = 20):
    """Time a function, returning (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    import statistics
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std


def main():
    torch = _check_torch()
    if torch is None:
        print("torch is not installed — skipping torch.compile demo.")
        print("Install with: pip install torch")
        return

    print("torch.compile() Post-Processing Demo")
    print(f"PyTorch {torch.__version__}")
    print(f"Python {sys.version.split()[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate synthetic predictions: (N, 5) — x, y, w, h, score
    # Simulates a typical detection output with ~50% above threshold
    n_predictions = 2000
    torch.manual_seed(42)
    predictions = torch.rand(n_predictions, 5, device=device)
    # Scale x,y to [0,1], w,h to [0.01, 0.3]
    predictions[:, 2:4] = predictions[:, 2:4] * 0.29 + 0.01

    print(f"\nInput: {n_predictions} predictions, threshold=0.5")
    print()

    # --- Eager mode ---
    eager_boxes, eager_scores = prepare_boxes(predictions, 0.5)
    eager_mean, eager_std = time_function(
        lambda: prepare_boxes(predictions, 0.5),
        n_iters=300,
        warmup=30,
    )
    print(f"  Eager mode:")
    print(f"    Output: {eager_boxes.shape[0]} boxes kept")
    print(f"    Time:   {eager_mean:.4f} +/- {eager_std:.4f} ms")

    # --- Compiled mode ---
    try:
        compiled_prepare = torch.compile(prepare_boxes)

        # Compilation warmup (first calls trigger compilation)
        print("\n  Compiling... ", end="", flush=True)
        t0 = time.perf_counter()
        for _ in range(5):
            compiled_prepare(predictions, 0.5)
        compile_time = (time.perf_counter() - t0) * 1000
        print(f"done ({compile_time:.0f} ms)")

        compiled_boxes, compiled_scores = compiled_prepare(predictions, 0.5)
        compiled_mean, compiled_std = time_function(
            lambda: compiled_prepare(predictions, 0.5),
            n_iters=300,
            warmup=30,
        )

        print(f"\n  Compiled mode:")
        print(f"    Output: {compiled_boxes.shape[0]} boxes kept")
        print(f"    Time:   {compiled_mean:.4f} +/- {compiled_std:.4f} ms")

        # --- Comparison ---
        print(f"\n  {'=' * 50}")
        if compiled_mean > 0:
            speedup = eager_mean / compiled_mean
            print(f"  Speedup: {speedup:.2f}x")
        print(f"  Compilation overhead: {compile_time:.0f} ms (one-time cost)")

        # Verify correctness
        match = torch.allclose(eager_scores.sort().values,
                               compiled_scores.sort().values, atol=1e-5)
        print(f"  Results match: {match}")

        frames_to_amortize = int(compile_time / max(eager_mean - compiled_mean, 0.001))
        print(f"  Break-even after ~{frames_to_amortize} frames")

    except Exception as e:
        print(f"\n  torch.compile not available: {e}")
        print("  (Requires PyTorch >= 2.0)")

    print()


if __name__ == "__main__":
    main()
