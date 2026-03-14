"""
Sequential vs batched inference demo — tracker_engine bottleneck.

tracker_engine's track_restoration validates candidate detections one at a
time, issuing a separate forward pass for each candidate. This demo shows
the performance difference between:
  - Sequential: loop over N candidates, one forward pass each
  - Batched: stack all candidates into a single tensor, one forward pass

The batched approach amortizes GPU kernel launch overhead and enables
hardware parallelism across the batch dimension.

Usage:
    python3 batch_inference_demo.py
"""

from __future__ import annotations

import sys
import time


def _check_torch():
    """Return torch module or None if unavailable."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def time_function(fn, n_iters: int = 50, warmup: int = 10):
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


def build_validation_model(torch_mod):
    """Build a simple CNN that mimics a re-identification / validation model.

    Architecture: conv -> relu -> pool -> flatten -> linear
    Input: (B, 3, 64, 64) crop of a candidate detection
    Output: (B, 1) validation score
    """
    nn = torch_mod.nn

    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )
    return model


def main():
    torch = _check_torch()
    if torch is None:
        print("torch is not installed — skipping batch inference demo.")
        print("Install with: pip install torch")
        return

    print("Sequential vs Batched Inference Demo")
    print(f"PyTorch {torch.__version__}")
    print(f"Python {sys.version.split()[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_validation_model(torch).to(device).eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    candidate_counts = [5, 10, 20, 50]
    crop_size = (3, 64, 64)

    print(f"\nCrop size: {crop_size}")
    print()
    print(f"  {'N candidates':>14}  {'Sequential (ms)':>16}  {'Batched (ms)':>14}  {'Speedup':>8}")
    print(f"  {'-' * 14}  {'-' * 16}  {'-' * 14}  {'-' * 8}")

    for n_candidates in candidate_counts:
        torch.manual_seed(42)

        # Generate candidate crops
        candidates = [
            torch.randn(1, *crop_size, device=device)
            for _ in range(n_candidates)
        ]
        batch = torch.cat(candidates, dim=0)

        # --- Sequential (tracker_engine track_restoration style) ---
        def sequential():
            results = []
            with torch.inference_mode():
                for c in candidates:
                    results.append(model(c))
            return results

        seq_mean, seq_std = time_function(sequential, n_iters=80, warmup=15)

        # --- Batched ---
        def batched():
            with torch.inference_mode():
                return model(batch)

        bat_mean, bat_std = time_function(batched, n_iters=80, warmup=15)

        speedup = seq_mean / bat_mean if bat_mean > 0 else float("inf")

        print(
            f"  {n_candidates:>14d}  "
            f"{seq_mean:>10.3f} +/- {seq_std:>4.2f}  "
            f"{bat_mean:>8.3f} +/- {bat_std:>4.2f}  "
            f"{speedup:>7.1f}x"
        )

        # Verify results match
        with torch.inference_mode():
            seq_results = torch.cat(sequential(), dim=0)
            bat_results = batched()
            if not torch.allclose(seq_results, bat_results, atol=1e-5):
                print(f"    WARNING: results differ (max delta: "
                      f"{(seq_results - bat_results).abs().max():.6f})")

    print()
    print("=" * 70)
    print("  Key takeaway: batching amortizes per-inference overhead.")
    print("  tracker_engine's track_restoration should batch all candidate")
    print("  validations into a single forward pass.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
