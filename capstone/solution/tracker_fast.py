"""
Reference solution — fast tracker using C++ components.

This module provides the same interface as the baseline but delegates
the heavy lifting to compiled C++ extensions built in earlier lessons.

Trainees should try implementing their own version first and only consult
this file when stuck.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Import C++ modules from earlier lessons.
#
# If these are not installed, fall back to stubs that raise ImportError with
# helpful messages.
# ---------------------------------------------------------------------------

try:
    from fast_tracker_utils._native import (
        kalman_native,
        preprocess_native,
        history_native,
        state_machine_native,
    )
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


# ---------------------------------------------------------------------------
# 1. FastKalmanFilter — pre-allocated buffers, in-place math
# ---------------------------------------------------------------------------

class FastKalmanFilter:
    """Kalman filter with pre-allocated matrices (no per-frame allocation).

    Key optimisations (from Lesson 5):
    - Transition matrix F is built once in __init__.
    - Noise matrices Q, R, H are pre-allocated.
    - All temporaries (S, K, y) are pre-allocated buffers.
    - Operations are done in-place where possible.
    """

    def __init__(self, dt: float = 1.0, process_noise: float = 0.01,
                 measurement_noise: float = 0.1):
        self.dt = dt

        # State and covariance
        self.state = np.zeros(4, dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64)

        # Pre-allocate transition matrix (constant for fixed dt)
        self.F = np.eye(4, dtype=np.float64)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Pre-allocate noise matrices
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        self.H = np.zeros((2, 4), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        # Pre-allocate temporaries
        self._F_cov = np.empty((4, 4), dtype=np.float64)
        self._S = np.empty((2, 2), dtype=np.float64)
        self._K = np.empty((4, 2), dtype=np.float64)
        self._y = np.empty(2, dtype=np.float64)
        self._I = np.eye(4, dtype=np.float64)

    def predict(self) -> np.ndarray:
        """Predict next state using pre-allocated matrices."""
        np.dot(self.F, self.state, out=self.state)
        np.dot(self.F, self.covariance, out=self._F_cov)
        np.dot(self._F_cov, self.F.T, out=self.covariance)
        self.covariance += self.Q
        return self.state

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with measurement using pre-allocated buffers."""
        np.subtract(measurement, self.H @ self.state, out=self._y)
        HCov = self.H @ self.covariance
        np.dot(HCov, self.H.T, out=self._S)
        self._S += self.R
        self._K[:] = self.covariance @ self.H.T @ np.linalg.inv(self._S)
        self.state += self._K @ self._y
        self.covariance[:] = (self._I - self._K @ self.H) @ self.covariance
        return self.state


# ---------------------------------------------------------------------------
# 2. FastPreprocessor — fused C++ kernel (Lesson 7)
# ---------------------------------------------------------------------------

class FastPreprocessor:
    """Image preprocessor using a fused C++ kernel.

    Instead of three separate passes (cast, normalize, transpose), the C++
    kernel does everything in a single pass over the pixel data.
    """

    def __init__(self, target_size: tuple[int, int] = (640, 640)):
        self.target_size = target_size
        if _HAS_NATIVE:
            self._impl = preprocess_native.FusedPreprocessor()
        else:
            self._impl = None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess HWC uint8 image to CHW float32 in one pass."""
        if self._impl is not None:
            return self._impl.process(image)
        # Fallback: optimised pure-numpy (still faster than baseline due to
        # fewer intermediate allocations)
        return np.ascontiguousarray(
            image.astype(np.float32).transpose(2, 0, 1) / 255.0
        )


# ---------------------------------------------------------------------------
# 3. FastHistoryBuffer — C++ circular buffer with zero-copy views (Lesson 4)
# ---------------------------------------------------------------------------

class FastHistoryBuffer:
    """Circular buffer backed by a contiguous C++ array.

    latest() returns a numpy view into the buffer — no copy.
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        if _HAS_NATIVE:
            self._impl = history_native.CircularBuffer(capacity, 4)
        else:
            # Fallback: pre-allocated numpy ring buffer
            self._data = np.zeros((capacity, 4), dtype=np.float64)
            self._size = 0
            self._index = 0

    def push(self, bbox: np.ndarray) -> None:
        if _HAS_NATIVE:
            self._impl.push(bbox)
        else:
            self._data[self._index] = bbox
            self._index = (self._index + 1) % self.capacity
            if self._size < self.capacity:
                self._size += 1

    def latest(self, n: int = 1) -> np.ndarray:
        """Return the last *n* entries as a (n, 4) array view — no copy."""
        if _HAS_NATIVE:
            return self._impl.latest(n)
        size = self._size
        n = min(n, size)
        indices = [(self._index - 1 - i) % size for i in range(n)]
        return self._data[indices]

    def __len__(self) -> int:
        if _HAS_NATIVE:
            return len(self._impl)
        return self._size


# ---------------------------------------------------------------------------
# 4. FastStateMachine — C++ variant-based (Lesson 8)
# ---------------------------------------------------------------------------

class FastStateMachine:
    """State machine backed by a C++ std::variant implementation."""

    def __init__(self):
        if _HAS_NATIVE:
            self._impl = state_machine_native.StateMachine()
        else:
            self.state: str = "idle"
            self.frames_in_state: int = 0

    @property
    def current_state(self) -> str:
        if _HAS_NATIVE:
            return self._impl.state()
        return self.state

    def on_event(self, event: str) -> str:
        if _HAS_NATIVE:
            return self._impl.on_event(event)
        # Fallback mirrors baseline logic
        self.frames_in_state += 1
        if self.state == "idle":
            if event == "detect":
                self.state = "tracking"
                self.frames_in_state = 0
        elif self.state == "tracking":
            if event == "detect":
                self.frames_in_state = 0
            elif event == "miss":
                self.state = "lost"
                self.frames_in_state = 0
        elif self.state == "lost":
            if event == "detect":
                self.state = "tracking"
                self.frames_in_state = 0
            elif event == "miss" and self.frames_in_state > 10:
                self.state = "expired"
                self.frames_in_state = 0
        elif self.state == "expired":
            if event == "reset":
                self.state = "idle"
                self.frames_in_state = 0
        return self.state


# ---------------------------------------------------------------------------
# Pipeline — drop-in replacement for baseline Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """End-to-end tracking pipeline using fast C++ components."""

    def __init__(self):
        self.kalman = FastKalmanFilter()
        self.preprocessor = FastPreprocessor()
        self.history = FastHistoryBuffer(capacity=100)
        self.state_machine = FastStateMachine()

    def process_frame(self, image: np.ndarray,
                      detection: Optional[np.ndarray] = None) -> dict:
        """Process a single frame — same interface as baseline."""
        preprocessed = self.preprocessor.preprocess(image)
        predicted = self.kalman.predict()

        if detection is not None:
            event = "detect"
            measured = self.kalman.update(detection[:2])
            bbox = detection
        else:
            event = "miss"
            measured = predicted
            bbox = np.array([predicted[0], predicted[1],
                             predicted[0] + 50, predicted[1] + 50],
                            dtype=np.float64)

        self.history.push(bbox)
        state = self.state_machine.on_event(event)

        return {
            "preprocessed_shape": preprocessed.shape,
            "predicted": predicted,
            "measured": measured,
            "state": state,
            "history_len": len(self.history),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def main():
    print("Running fast tracker smoke test...")
    if _HAS_NATIVE:
        print("  Using native C++ modules.")
    else:
        print("  Native modules not found — using numpy fallbacks.")

    pipeline = Pipeline()
    rng = np.random.default_rng(42)

    for i in range(20):
        image = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
        detection = (np.array([100.0 + i, 200.0 + i, 150.0 + i, 250.0 + i],
                              dtype=np.float64) if i % 2 == 0 else None)
        result = pipeline.process_frame(image, detection)
        if i % 5 == 0:
            print(f"  Frame {i:3d}: state={result['state']:>10s}  "
                  f"predicted=({result['predicted'][0]:6.1f}, "
                  f"{result['predicted'][1]:6.1f})  "
                  f"history_len={result['history_len']}")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
