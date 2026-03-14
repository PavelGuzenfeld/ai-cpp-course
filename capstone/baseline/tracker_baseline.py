"""
Pure-Python mini tracker — functional but deliberately slow.

This module contains four components that mirror real bottlenecks found in
tracker_engine.  Each one works correctly but leaves significant performance
on the table.  Your job in the capstone is to replace each component with a
fast C++ equivalent while keeping the same interface.

Run directly for a quick smoke test:
    python3 tracker_baseline.py
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. KalmanFilter — rebuilds identity matrices every call
# ---------------------------------------------------------------------------

class KalmanFilter:
    """Simplified 4-state (x, y, vx, vy) Kalman filter.

    Bottleneck: np.eye() and noise-matrix allocation happen inside predict()
    on every single frame instead of being pre-allocated.
    """

    def __init__(self, dt: float = 1.0, process_noise: float = 0.01,
                 measurement_noise: float = 0.1):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        # state: [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64)

    def predict(self) -> np.ndarray:
        """Predict next state.  Allocates fresh matrices every call."""
        # Transition matrix — rebuilt every time (wasteful)
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = self.dt
        F[1, 3] = self.dt

        # Process noise — rebuilt every time (wasteful)
        Q = np.eye(4, dtype=np.float64) * self.process_noise

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
        return self.state.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with a 2-element measurement [x, y]."""
        # Measurement matrix — rebuilt every time
        H = np.zeros((2, 4), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        # Measurement noise — rebuilt every time
        R = np.eye(2, dtype=np.float64) * self.measurement_noise

        y = measurement - H @ self.state
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.covariance = (np.eye(4, dtype=np.float64) - K @ H) @ self.covariance
        return self.state.copy()


# ---------------------------------------------------------------------------
# 2. Preprocessor — three separate numpy passes
# ---------------------------------------------------------------------------

class Preprocessor:
    """Image preprocessor that normalises and transposes.

    Bottleneck: three distinct passes over the data (cast, normalize,
    transpose) instead of a single fused kernel.
    """

    def __init__(self, target_size: tuple[int, int] = (640, 640)):
        self.target_size = target_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess HWC uint8 image to CHW float32 normalised [0,1].

        Three separate numpy operations — each one traverses the full array.
        """
        # Pass 1: cast to float32
        img_float = image.astype(np.float32)

        # Pass 2: normalize to [0, 1]
        img_norm = img_float / 255.0

        # Pass 3: transpose HWC -> CHW
        img_chw = np.transpose(img_norm, (2, 0, 1))

        # Ensure contiguous layout
        return np.ascontiguousarray(img_chw)


# ---------------------------------------------------------------------------
# 3. HistoryBuffer — copies on every access
# ---------------------------------------------------------------------------

class HistoryBuffer:
    """Fixed-size history ring buffer for bounding boxes.

    Bottleneck: latest() returns a full copy every time, even when the caller
    only needs a read-only view.
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._buffer: List[np.ndarray] = []
        self._index: int = 0
        self._full: bool = False

    def push(self, bbox: np.ndarray) -> None:
        """Append a bounding box [x1, y1, x2, y2]."""
        if not self._full:
            self._buffer.append(bbox.copy())
            if len(self._buffer) == self.capacity:
                self._full = True
        else:
            self._buffer[self._index] = bbox.copy()
        self._index = (self._index + 1) % self.capacity

    def latest(self, n: int = 1) -> List[np.ndarray]:
        """Return the last *n* entries.  Copies every element (wasteful)."""
        size = len(self._buffer)
        if n > size:
            n = size
        result = []
        for i in range(n):
            idx = (self._index - 1 - i) % size
            result.append(self._buffer[idx].copy())  # full copy each time
        return result

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# 4. StateMachine — string-based with if/elif chains
# ---------------------------------------------------------------------------

class StateMachine:
    """Tracker state machine with string states.

    Bottleneck: every transition does string comparisons through an if/elif
    chain.  A variant-based C++ state machine can dispatch at compile time.
    """

    STATES = ("idle", "tracking", "lost", "expired")

    def __init__(self):
        self.state: str = "idle"
        self.frames_in_state: int = 0

    def on_event(self, event: str) -> str:
        """Process an event and return the new state.

        Uses string comparisons — slow compared to variant-based dispatch.
        """
        self.frames_in_state += 1

        if self.state == "idle":
            if event == "detect":
                self.state = "tracking"
                self.frames_in_state = 0
        elif self.state == "tracking":
            if event == "detect":
                self.frames_in_state = 0  # reset counter, stay tracking
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
# Pipeline — ties everything together
# ---------------------------------------------------------------------------

class Pipeline:
    """End-to-end tracking pipeline using the four baseline components."""

    def __init__(self):
        self.kalman = KalmanFilter()
        self.preprocessor = Preprocessor()
        self.history = HistoryBuffer(capacity=100)
        self.state_machine = StateMachine()

    def process_frame(self, image: np.ndarray,
                      detection: Optional[np.ndarray] = None) -> dict:
        """Process a single frame through the full pipeline.

        Returns a dict with preprocessed image, predicted state, and
        tracker metadata.
        """
        # Preprocess
        preprocessed = self.preprocessor.preprocess(image)

        # Predict
        predicted = self.kalman.predict()

        # Update with detection if available
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

        # History
        self.history.push(bbox)

        # State machine
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
    print("Running baseline tracker smoke test...")
    pipeline = Pipeline()
    rng = np.random.default_rng(42)

    for i in range(20):
        # Synthetic 640x640 RGB image
        image = rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)

        # Simulate detection every other frame
        if i % 2 == 0:
            detection = np.array([100.0 + i, 200.0 + i,
                                  150.0 + i, 250.0 + i], dtype=np.float64)
        else:
            detection = None

        result = pipeline.process_frame(image, detection)
        if i % 5 == 0:
            print(f"  Frame {i:3d}: state={result['state']:>10s}  "
                  f"predicted=({result['predicted'][0]:6.1f}, "
                  f"{result['predicted'][1]:6.1f})  "
                  f"history_len={result['history_len']}")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
