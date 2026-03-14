#!/usr/bin/env python3
"""
Kalman Filter Pre-Allocation — Practical addition to Lesson 5.

Shows the actual optimisation from tracker_engine: pre-allocating matrices
in __init__ instead of rebuilding them every predict() call.

Run:
    python kalman_preallocated.py
"""

import sys
import time

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required.  pip install numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Kalman state dimensions (tracker_engine uses 12-dim state)
# ---------------------------------------------------------------------------
# State: [x, y, z, w, h, d, vx, vy, vz, vw, vh, vd]
#   positions (6) + velocities (6) = 12
NDIM = 6
DT = 1.0  # time step


# ---------------------------------------------------------------------------
# SLOW: rebuilds matrices every predict()
# ---------------------------------------------------------------------------

class KalmanFilterSlow:
    """Original tracker_engine pattern — allocates every call."""

    def __init__(self, ndim: int = NDIM, dt: float = DT):
        self.ndim = ndim
        self.dt = dt
        # State is 2*ndim (position + velocity)
        self.sdim = 2 * ndim

    def predict(self, mean: np.ndarray, covariance: np.ndarray,
                std_pos: np.ndarray, std_vel: np.ndarray):
        """Predict step — rebuilds motion_mat and noise_cov from scratch."""
        # ANTI-PATTERN: creates a new identity matrix every call
        motion_mat = np.eye(self.sdim, dtype=np.float64)
        for i in range(self.ndim):
            motion_mat[i, self.ndim + i] = self.dt

        # ANTI-PATTERN: creates diagonal matrix from scratch every call
        std = np.r_[std_pos, std_vel]
        motion_cov = np.diag(np.square(std))

        # Kalman predict equations
        mean = motion_mat @ mean
        covariance = motion_mat @ covariance @ motion_mat.T + motion_cov

        return mean, covariance


# ---------------------------------------------------------------------------
# FAST: pre-allocates matrices, updates in-place
# ---------------------------------------------------------------------------

class KalmanFilterFast:
    """Optimised version — pre-allocates and updates in-place."""

    def __init__(self, ndim: int = NDIM, dt: float = DT):
        self.ndim = ndim
        self.dt = dt
        self.sdim = 2 * ndim

        # Pre-allocate motion matrix (constant structure)
        self._motion_mat = np.eye(self.sdim, dtype=np.float64)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Pre-allocate noise covariance (only diagonal changes)
        self._motion_cov = np.zeros((self.sdim, self.sdim), dtype=np.float64)

    def predict(self, mean: np.ndarray, covariance: np.ndarray,
                std_pos: np.ndarray, std_vel: np.ndarray):
        """Predict step — reuses pre-allocated matrices."""
        # Update diagonal of noise covariance in-place
        # No new allocation — just writes into existing memory
        diag = self._motion_cov.ravel()[:: self.sdim + 1]  # view of diagonal
        diag[:self.ndim] = np.square(std_pos)
        diag[self.ndim:] = np.square(std_vel)

        # Kalman predict equations (same math, no allocation overhead)
        mean = self._motion_mat @ mean
        covariance = (self._motion_mat @ covariance @ self._motion_mat.T
                      + self._motion_cov)

        return mean, covariance


# ---------------------------------------------------------------------------
# Bonus: clamp_dims bug fix
# ---------------------------------------------------------------------------

def clamp_dims_buggy(bbox: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
    """
    Original buggy version: clamp only clips min, not max.
    If a detection box extends past image boundaries, width/height can
    exceed the image dimensions.
    """
    h, w = img_shape
    bbox = bbox.copy()
    bbox[0] = max(0, bbox[0])      # x1
    bbox[1] = max(0, bbox[1])      # y1
    # BUG: does not clamp x2, y2 to image bounds
    # bbox[2] and bbox[3] could be > w, h
    return bbox


def clamp_dims_fixed(bbox: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
    """Fixed version: clamp all four coordinates."""
    h, w = img_shape
    bbox = bbox.copy()
    bbox[0] = np.clip(bbox[0], 0, w)   # x1
    bbox[1] = np.clip(bbox[1], 0, h)   # y1
    bbox[2] = np.clip(bbox[2], 0, w)   # x2
    bbox[3] = np.clip(bbox[3], 0, h)   # y2
    return bbox


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench(func, *args, warmup: int = 100, repeats: int = 5000, **kwargs) -> float:
    """Return median time in microseconds."""
    for _ in range(warmup):
        func(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


def verify_equivalence(kf_slow: KalmanFilterSlow, kf_fast: KalmanFilterFast,
                       n_trials: int = 100):
    """Verify both implementations produce identical results."""
    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        sdim = kf_slow.sdim
        mean = rng.standard_normal(sdim)
        cov = rng.standard_normal((sdim, sdim))
        cov = cov @ cov.T + np.eye(sdim) * 0.1  # make positive definite

        std_pos = np.abs(rng.standard_normal(kf_slow.ndim)) + 0.01
        std_vel = np.abs(rng.standard_normal(kf_slow.ndim)) + 0.01

        m_slow, c_slow = kf_slow.predict(mean.copy(), cov.copy(), std_pos, std_vel)
        m_fast, c_fast = kf_fast.predict(mean.copy(), cov.copy(), std_pos, std_vel)

        assert np.allclose(m_slow, m_fast, atol=1e-12), "mean mismatch"
        assert np.allclose(c_slow, c_fast, atol=1e-12), "covariance mismatch"

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Kalman Filter Pre-Allocation Benchmark")
    print(f"  State dimension: {2 * NDIM} ({NDIM} position + {NDIM} velocity)")
    print(f"  NumPy: {np.__version__}")
    print()

    kf_slow = KalmanFilterSlow()
    kf_fast = KalmanFilterFast()

    # Verify correctness
    print("Verifying equivalence (100 random trials) ... ", end="", flush=True)
    assert verify_equivalence(kf_slow, kf_fast)
    print("PASSED")
    print()

    # Prepare test data
    rng = np.random.default_rng(0)
    sdim = kf_slow.sdim
    mean = rng.standard_normal(sdim)
    cov = rng.standard_normal((sdim, sdim))
    cov = cov @ cov.T + np.eye(sdim) * 0.1

    std_pos = np.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])
    std_vel = np.array([0.01, 0.01, 0.01, 0.005, 0.005, 0.005])

    # Benchmark
    us_slow = bench(kf_slow.predict, mean.copy(), cov.copy(), std_pos, std_vel)
    us_fast = bench(kf_fast.predict, mean.copy(), cov.copy(), std_pos, std_vel)

    print(f"{'Implementation':<30} {'Time (us)':>10} {'Speedup':>10}")
    print("-" * 52)
    print(f"{'KalmanFilterSlow (rebuild)':<30} {us_slow:>10.1f} {'1.0x':>10}")
    print(f"{'KalmanFilterFast (pre-alloc)':<30} {us_fast:>10.1f} {us_slow/us_fast:>9.1f}x")
    print()

    # Breakdown: what costs time in the slow version?
    print("Allocation breakdown (slow version):")

    def just_eye():
        return np.eye(sdim, dtype=np.float64)

    def just_diag():
        std = np.r_[std_pos, std_vel]
        return np.diag(np.square(std))

    def just_inplace_diag():
        mat = kf_fast._motion_cov
        diag = mat.ravel()[:: sdim + 1]
        diag[:NDIM] = np.square(std_pos)
        diag[NDIM:] = np.square(std_vel)

    us_eye = bench(just_eye)
    us_diag = bench(just_diag)
    us_inplace = bench(just_inplace_diag)

    print(f"  np.eye({sdim})                : {us_eye:.1f} us")
    print(f"  np.diag(np.square(...))     : {us_diag:.1f} us")
    print(f"  in-place diagonal update    : {us_inplace:.1f} us  "
          f"({us_diag/us_inplace:.1f}x faster)")
    print()

    # At 100 Hz with 50 tracked objects, predict() is called 5000 times/sec
    calls_per_sec = 5000
    overhead_ms = (us_slow - us_fast) * calls_per_sec / 1e3
    print(f"Impact at 100 Hz with 50 tracks ({calls_per_sec} calls/sec):")
    print(f"  Saved per second: {overhead_ms:.1f} ms")
    print()

    # --- Bonus: clamp_dims fix ---
    print("=" * 52)
    print("  Bonus: clamp_dims bug fix")
    print("=" * 52)
    print()

    bbox = np.array([-10.0, -5.0, 700.0, 500.0])  # extends past 640x480 image
    img_shape = (480, 640)

    buggy = clamp_dims_buggy(bbox, img_shape)
    fixed = clamp_dims_fixed(bbox, img_shape)

    print(f"  Input bbox:  {bbox}")
    print(f"  Image shape: {img_shape[1]}x{img_shape[0]}")
    print(f"  Buggy result: {buggy}  <- x2=700 exceeds width=640")
    print(f"  Fixed result: {fixed}  <- all coords clamped")
    print()

    print("Key takeaways:")
    print("  - Pre-allocate matrices that have constant structure")
    print("  - Update only the changing elements (diagonal) in-place")
    print("  - np.eye() + np.diag() allocation adds up at high call rates")
    print("  - ravel()[::n+1] gives a writable view of the diagonal")
    print("  - Always clamp all bounding box coordinates to image bounds")
    print()


if __name__ == "__main__":
    main()
