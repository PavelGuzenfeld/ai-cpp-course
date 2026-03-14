"""
Profiling a simulated tracker_engine pipeline with stage-level breakdown.

Uses the LatencyTracker from measure_latency.py to time each stage of a
tracking pipeline: preprocess, inference, and postprocess. Prints a
breakdown with percentages and applies Amdahl's law to show which stage
yields the most benefit from optimization.

This demonstrates the profiling workflow from Lesson 6: measure first,
then optimize the bottleneck that matters.

Usage:
    python3 profile_pipeline.py
"""

from __future__ import annotations

import sys
import time
import math

# Import LatencyTracker from the same lesson directory
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from measure_latency import LatencyTracker

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# Simulated pipeline stages
# ============================================================================

def preprocess(image: "np.ndarray | None" = None, size: tuple[int, int] = (640, 480)):
    """Simulate tracker_engine's preprocess: resize + normalize.

    In tracker_engine this involves:
      - Letterbox resize to model input size
      - uint8 -> float32 conversion
      - ImageNet normalization (mean/std)
      - HWC -> CHW transpose
    """
    if HAS_NUMPY:
        if image is None:
            image = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        # Simulate resize (create new array at target size)
        resized = np.empty((416, 416, 3), dtype=np.float32)
        # Simulate normalize
        resized[:] = image[:416, :416, :].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        resized = (resized - mean) / std
        # HWC -> CHW
        result = resized.transpose(2, 0, 1)
        return np.ascontiguousarray(result)
    else:
        # Fallback: just sleep to simulate work
        time.sleep(0.002)
        return None


def inference(input_tensor=None):
    """Simulate tracker_engine's inference: neural network forward pass.

    In the real pipeline this is a YOLO or similar detector. We simulate
    it with a large matrix multiplication to create realistic CPU load.
    """
    if HAS_NUMPY:
        # Simulate computation with matrix operations
        a = np.random.randn(256, 256).astype(np.float32)
        b = np.random.randn(256, 256).astype(np.float32)
        for _ in range(3):
            a = a @ b
            a = np.maximum(a, 0)  # ReLU
        return a
    else:
        time.sleep(0.015)
        return None


def postprocess(raw_output=None, n_detections: int = 50):
    """Simulate tracker_engine's postprocess: NMS + state update.

    In tracker_engine this involves:
      - Decode raw network output to bounding boxes
      - Apply score thresholding
      - Non-maximum suppression
      - Update Kalman filter states
      - Match detections to existing tracks
    """
    if HAS_NUMPY:
        # Simulate NMS-like filtering
        boxes = np.random.rand(n_detections, 4).astype(np.float32)
        scores = np.random.rand(n_detections).astype(np.float32)

        # Score threshold
        mask = scores > 0.5
        boxes = boxes[mask]
        scores = scores[mask]

        # Simplified NMS: sort by score, keep top-K
        order = np.argsort(scores)[::-1]
        boxes = boxes[order[:20]]

        # Simulate Kalman update per track
        n_tracks = min(len(boxes), 15)
        states = np.random.randn(n_tracks, 8).astype(np.float32)
        H = np.eye(4, 8, dtype=np.float32)
        for i in range(n_tracks):
            # Kalman predict + update (simplified)
            predicted = H @ states[i]
            innovation = boxes[i] - predicted
            states[i, :4] += 0.5 * innovation

        return states
    else:
        time.sleep(0.001)
        return None


# ============================================================================
# Pipeline profiler
# ============================================================================

def profile_pipeline(n_frames: int = 50):
    """Run the full pipeline for n_frames and collect per-stage timings."""
    tracker = LatencyTracker()

    print(f"Profiling pipeline for {n_frames} frames...\n")

    if HAS_NUMPY:
        image = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    else:
        image = None

    for frame in range(n_frames):
        with tracker.section("preprocess"):
            tensor = preprocess(image)

        with tracker.section("inference"):
            raw = inference(tensor)

        with tracker.section("postprocess"):
            result = postprocess(raw)

    return tracker


def print_breakdown(tracker: LatencyTracker):
    """Print a stage-by-stage breakdown with percentages."""
    sections = tracker.sections
    means_ms = {s: tracker.mean(s) / 1e6 for s in sections}
    total_ms = sum(means_ms.values())

    print("=" * 70)
    print("  PIPELINE STAGE BREAKDOWN")
    print("=" * 70)
    print()
    print(f"  {'Stage':<20} {'Mean (ms)':>10} {'% of Total':>12} {'Bar':>20}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 12} {'-' * 20}")

    bar_width = 20
    for section in sections:
        mean = means_ms[section]
        pct = (mean / total_ms * 100) if total_ms > 0 else 0
        bar_len = int(pct / 100 * bar_width)
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        print(f"  {section:<20} {mean:>10.3f} {pct:>11.1f}% [{bar}]")

    print(f"  {'-' * 20} {'-' * 10}")
    print(f"  {'TOTAL':<20} {total_ms:>10.3f}")
    print(f"  {'Throughput':<20} {1000 / total_ms:>10.1f} fps")
    print()


def amdahl_analysis(tracker: LatencyTracker):
    """Apply Amdahl's law to show optimization priority.

    Amdahl's law: if a stage takes fraction f of total time and we speed
    it up by factor S, the overall speedup is:
        1 / ((1 - f) + f/S)

    This shows diminishing returns: optimizing a stage that's only 5% of
    runtime gives at most 1.05x overall, even with infinite speedup.
    """
    sections = tracker.sections
    means_ms = {s: tracker.mean(s) / 1e6 for s in sections}
    total_ms = sum(means_ms.values())

    print("=" * 70)
    print("  AMDAHL'S LAW — OPTIMIZATION PRIORITY")
    print("=" * 70)
    print()
    print("  If we make each stage 2x faster, what's the overall speedup?")
    print()
    print(f"  {'Stage':<20} {'Fraction':>10} {'2x speedup':>12} {'10x speedup':>13}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 12} {'-' * 13}")

    # Sort by fraction descending (optimize biggest bottleneck first)
    sorted_sections = sorted(sections, key=lambda s: means_ms[s], reverse=True)

    for section in sorted_sections:
        f = means_ms[section] / total_ms if total_ms > 0 else 0

        # Amdahl: overall speedup = 1 / ((1 - f) + f/S)
        speedup_2x = 1.0 / ((1 - f) + f / 2.0)
        speedup_10x = 1.0 / ((1 - f) + f / 10.0)

        print(
            f"  {section:<20} {f:>10.1%} "
            f"{speedup_2x:>11.2f}x "
            f"{speedup_10x:>12.2f}x"
        )

    # Overall recommendation
    bottleneck = sorted_sections[0]
    bottleneck_frac = means_ms[bottleneck] / total_ms if total_ms > 0 else 0

    print()
    print(f"  Recommendation: optimize '{bottleneck}' first")
    print(f"  ({bottleneck_frac:.0%} of total — gives the most overall speedup)")

    # Show theoretical maximum
    max_speedup = 1.0 / (1 - bottleneck_frac) if bottleneck_frac < 1 else float("inf")
    print(f"  Theoretical max speedup (infinite {bottleneck} speedup): {max_speedup:.2f}x")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    print("Pipeline Profiling Demo — tracker_engine Bottleneck Analysis")
    print(f"Python {sys.version.split()[0]}")
    if HAS_NUMPY:
        print(f"NumPy {np.__version__}")
    else:
        print("NumPy not available — using sleep-based simulation")
    print()

    tracker = profile_pipeline(n_frames=50)

    # Full latency report (from measure_latency.py)
    tracker.print_report()

    # Stage breakdown with percentages
    print_breakdown(tracker)

    # Amdahl's law analysis
    amdahl_analysis(tracker)


if __name__ == "__main__":
    main()
