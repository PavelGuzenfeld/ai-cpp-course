#!/usr/bin/env python3
"""
OpenCV Resize Benchmark — Practical addition to Lesson 2.

Compares cv2.resize interpolation methods on synthetic images at common
video/camera resolutions.  Optionally benchmarks the C++ cpp_image_processor
pybind11 module built in this lesson.

Run:
    python opencv_benchmark.py
"""

import time
import sys

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required.  pip install numpy")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required.  pip install opencv-python")
    sys.exit(1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_synthetic_image(height: int, width: int) -> np.ndarray:
    """Generate a 3-channel uint8 image with gradients + noise."""
    rng = np.random.default_rng(42)
    # horizontal gradient
    grad_x = np.linspace(0, 255, width, dtype=np.float32)
    # vertical gradient
    grad_y = np.linspace(0, 255, height, dtype=np.float32)
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = grad_x[np.newaxis, :]                       # blue channel
    img[:, :, 1] = grad_y[:, np.newaxis]                        # green channel
    img[:, :, 2] = rng.integers(0, 256, (height, width), dtype=np.uint8)  # red noise
    return img


def bench(func, *args, warmup: int = 2, repeats: int = 10, **kwargs) -> float:
    """Return median wall-clock time in milliseconds."""
    for _ in range(warmup):
        func(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# benchmark definitions
# ---------------------------------------------------------------------------

RESOLUTIONS = [
    ("1080p", 1920, 1080),
    ("4K",    3840, 2160),
    ("8K",    7680, 4320),
]

TARGET_SIZE = (640, 480)  # common inference input

INTERPOLATIONS = [
    ("INTER_NEAREST", cv2.INTER_NEAREST),
    ("INTER_LINEAR",  cv2.INTER_LINEAR),
    ("INTER_AREA",    cv2.INTER_AREA),
]


def run_opencv_benchmarks() -> list[dict]:
    rows = []
    for res_name, w, h in RESOLUTIONS:
        img = make_synthetic_image(h, w)
        for interp_name, interp_flag in INTERPOLATIONS:
            ms = bench(cv2.resize, img, TARGET_SIZE, interpolation=interp_flag)
            rows.append({
                "source": res_name,
                "method": interp_name,
                "backend": "cv2.resize",
                "ms": ms,
            })
    return rows


def run_cpp_benchmarks() -> list[dict]:
    """Try to import the C++ module built in this lesson."""
    try:
        import cpp_image_processor  # type: ignore[import-not-found]
    except ImportError:
        print("\n[INFO] cpp_image_processor module not found — skipping C++ benchmark.")
        print("       Build it with: cd ai-cpp-l2 && mkdir -p build && cd build && cmake .. && make\n")
        return []

    rows = []
    for res_name, w, h in RESOLUTIONS:
        img = make_synthetic_image(h, w)
        ms = bench(cpp_image_processor.resize, img, TARGET_SIZE[0], TARGET_SIZE[1])
        rows.append({
            "source": res_name,
            "method": "C++ pybind11",
            "backend": "cpp_image_processor",
            "ms": ms,
        })
    return rows


# ---------------------------------------------------------------------------
# cache-effect demonstration
# ---------------------------------------------------------------------------

def cache_effects_demo() -> list[dict]:
    """
    Resize images of increasing size to show how performance degrades
    non-linearly once the source image exceeds L2/L3 cache.
    """
    sizes = [
        (320,  240),
        (640,  480),
        (1280, 720),
        (1920, 1080),
        (3840, 2160),
        (7680, 4320),
    ]
    rows = []
    for w, h in sizes:
        img = make_synthetic_image(h, w)
        megapixels = w * h / 1e6
        data_mb = img.nbytes / (1024 * 1024)
        ms = bench(cv2.resize, img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        rows.append({
            "resolution": f"{w}x{h}",
            "megapixels": megapixels,
            "data_MB": data_mb,
            "ms": ms,
            "us_per_mpix": (ms * 1000.0) / megapixels if megapixels > 0 else 0,
        })
    return rows


# ---------------------------------------------------------------------------
# pretty printing
# ---------------------------------------------------------------------------

def print_table(title: str, headers: list[str], rows: list[list[str]],
                col_widths: list[int] | None = None):
    if col_widths is None:
        col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    def fmt_row(cells):
        return "|" + "|".join(c.center(w) for c, w in zip(cells, col_widths)) + "|"

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(sep)
    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print(sep)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("OpenCV Resize Benchmark")
    print(f"  NumPy  {np.__version__}")
    print(f"  OpenCV {cv2.__version__}")
    print(f"  Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    # --- interpolation comparison ---
    results = run_opencv_benchmarks()
    cpp_results = run_cpp_benchmarks()
    all_results = results + cpp_results

    headers = ["Source", "Method", "Backend", "Time (ms)"]
    table_rows = [
        [r["source"], r["method"], r["backend"], f"{r['ms']:.2f}"]
        for r in all_results
    ]
    print_table("Interpolation Method Comparison", headers, table_rows)

    # --- cache effects ---
    cache_rows = cache_effects_demo()
    headers2 = ["Resolution", "MPix", "Data (MB)", "Time (ms)", "us/MPix"]
    table_rows2 = [
        [r["resolution"],
         f"{r['megapixels']:.2f}",
         f"{r['data_MB']:.1f}",
         f"{r['ms']:.2f}",
         f"{r['us_per_mpix']:.1f}"]
        for r in cache_rows
    ]
    print_table("Cache Effects — INTER_LINEAR resize to 640x480", headers2, table_rows2)

    print("\nKey takeaways:")
    print("  - INTER_NEAREST is fastest but lowest quality (no interpolation)")
    print("  - INTER_AREA is best for downscaling (anti-aliased) but slowest")
    print("  - us/MPix increases with image size due to cache pressure")
    print("  - The C++ module (if available) can outperform Python cv2 wrapper")
    print()


if __name__ == "__main__":
    main()
