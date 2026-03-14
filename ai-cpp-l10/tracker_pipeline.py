"""
Baseline tracker pipeline with realistic bottlenecks.

This module simulates a visual object tracker pipeline with the same
categories of performance problems found in production tracker code:

- Redundant matrix allocation in Kalman filter predict()
- Unnecessary copy chains in postprocessing
- Per-frame buffer allocation in preprocessing

Run standalone to see per-stage timing:

    python3 tracker_pipeline.py
"""

import time
import numpy as np


class KalmanFilter:
    """12-state Kalman filter for bounding box tracking.

    State vector: [x, y, w, h, x', y', w', h', x'', y'', w'', h'']
    (position, velocity, acceleration for each of x, y, width, height)

    BUG (intentional): rebuilds identity matrices on every predict() call.
    """

    def __init__(self, state_dim=12, measure_dim=4, process_noise=0.01):
        self.state_dim = state_dim
        self.measure_dim = measure_dim
        self.process_noise = process_noise

        # State vector
        self.state = np.zeros((state_dim, 1), dtype=np.float64)

        # State transition matrix (constant velocity model)
        self.F = np.eye(state_dim, dtype=np.float64)
        dt = 1.0
        for i in range(state_dim // 3):
            self.F[i, i + state_dim // 3] = dt
            if i + 2 * (state_dim // 3) < state_dim:
                self.F[i, i + 2 * (state_dim // 3)] = 0.5 * dt * dt
                self.F[i + state_dim // 3, i + 2 * (state_dim // 3)] = dt

        # Measurement matrix
        self.H = np.zeros((measure_dim, state_dim), dtype=np.float64)
        for i in range(measure_dim):
            self.H[i, i] = 1.0

        # Covariance matrices
        self.P = np.eye(state_dim, dtype=np.float64) * 10.0
        self.R = np.eye(measure_dim, dtype=np.float64) * 0.1

    def predict(self):
        """Predict next state.

        BOTTLENECK: Creates np.eye(12) and np.diag() every single call.
        These allocations dominate the actual matrix math.
        """
        # Wasteful: creates a NEW 12x12 identity matrix every call
        Q = np.eye(self.state_dim, dtype=np.float64)
        Q[self.state_dim // 2:, self.state_dim // 2:] *= self.process_noise

        # Wasteful: another allocation for noise
        noise_vec = np.random.randn(self.state_dim) * 0.001
        noise = np.diag(noise_vec)

        # The actual math (fast)
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + Q + noise

        return self.state[:self.measure_dim].flatten()

    def update(self, measurement):
        """Update state with measurement."""
        z = np.array(measurement, dtype=np.float64).reshape(-1, 1)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I = np.eye(self.state_dim, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        return self.state[:self.measure_dim].flatten()


class PostProcessor:
    """Extract tracking results from pipeline arrays.

    BOTTLENECK: Uses a copy chain to extract simple scalar values,
    mirroring the .clone().cpu().numpy().tolist() pattern.
    """

    def __init__(self):
        self.results_history = []

    def extract_position(self, result_array):
        """Extract (x, y) position from result array.

        BOTTLENECK: 3 copies to get 2 floats.
        """
        # Copy #1: defensive copy (unnecessary — we only read)
        values = result_array.copy()
        # Copy #2: flatten (unnecessary — could index directly)
        temp = values.flatten()
        # Copy #3: convert to Python list (unnecessary — could use float())
        coords = temp.tolist()
        return coords[0], coords[1]

    def extract_size(self, result_array):
        """Extract (w, h) size from result array.

        Same copy chain problem.
        """
        values = result_array.copy()
        temp = values.flatten()
        all_values = temp.tolist()
        return all_values[2], all_values[3]

    def format_result(self, state_vector):
        """Format full tracking result.

        BOTTLENECK: Converts entire state vector through copy chain.
        """
        copied = state_vector.copy()
        flat = copied.flatten()
        as_list = flat.tolist()
        result = {
            "x": as_list[0],
            "y": as_list[1],
            "w": as_list[2],
            "h": as_list[3],
        }
        self.results_history.append(result)
        return result


class Preprocessor:
    """Prepare frames for inference.

    BOTTLENECK: Allocates output buffers every frame.
    """

    def __init__(self, target_h=256, target_w=256, pad_h=288, pad_w=288):
        self.target_h = target_h
        self.target_w = target_w
        self.pad_h = pad_h
        self.pad_w = pad_w

    def preprocess(self, frame):
        """Resize and pad a frame.

        BOTTLENECK: np.zeros allocates new buffers every call.
        """
        h, w = frame.shape[:2]

        # Allocation #1: resize buffer (new every frame)
        resized = np.zeros(
            (self.target_h, self.target_w, 3), dtype=np.float32
        )

        # Simple bilinear-ish resize (not actual bilinear, but representative)
        scale_h = h / self.target_h
        scale_w = w / self.target_w
        for row in range(self.target_h):
            src_row = min(int(row * scale_h), h - 1)
            for col in range(self.target_w):
                src_col = min(int(col * scale_w), w - 1)
                resized[row, col] = frame[src_row, src_col]

        # Normalize to [0, 1]
        resized /= 255.0

        # Allocation #2: padded buffer (new every frame)
        padded = np.zeros(
            (self.pad_h, self.pad_w, 3), dtype=np.float32
        )

        # Center the resized image in the padded buffer
        off_h = (self.pad_h - self.target_h) // 2
        off_w = (self.pad_w - self.target_w) // 2
        padded[off_h:off_h + self.target_h, off_w:off_w + self.target_w] = resized

        return padded


def simulate_inference(preprocessed_frame):
    """Simulate neural network inference.

    In a real tracker, this would be a PyTorch/TensorRT forward pass.
    We simulate it with a simple computation that takes realistic time.
    """
    # Simulate feature extraction with actual computation
    features = np.mean(preprocessed_frame, axis=(0, 1))
    response = np.outer(features, features)
    # Simulate a detection: return bbox [x, y, w, h]
    cx = float(np.sum(response[0])) % 640
    cy = float(np.sum(response[1])) % 480
    w = 50.0 + float(np.sum(response[2])) % 100
    h = 50.0 + float(np.sum(response[2])) % 80
    return np.array([[cx, cy, w, h]], dtype=np.float64)


class Pipeline:
    """Complete tracking pipeline: preprocess -> inference -> kalman -> postprocess."""

    def __init__(self):
        self.preprocessor = Preprocessor(
            target_h=64, target_w=64, pad_h=80, pad_w=80
        )
        self.kalman = KalmanFilter(state_dim=12, measure_dim=4)
        self.postprocessor = PostProcessor()
        self.timings = {
            "preprocess": [],
            "inference": [],
            "kalman": [],
            "postprocess": [],
        }

    def process_frame(self, frame):
        """Process a single frame through the full pipeline."""
        # Stage 1: Preprocess
        t0 = time.perf_counter_ns()
        preprocessed = self.preprocessor.preprocess(frame)
        t1 = time.perf_counter_ns()

        # Stage 2: Inference
        detection = simulate_inference(preprocessed)
        t2 = time.perf_counter_ns()

        # Stage 3: Kalman filter
        self.kalman.predict()
        state = self.kalman.update(detection[0])
        t3 = time.perf_counter_ns()

        # Stage 4: Postprocess
        result = self.postprocessor.format_result(
            state.reshape(1, -1)
        )
        t4 = time.perf_counter_ns()

        self.timings["preprocess"].append(t1 - t0)
        self.timings["inference"].append(t2 - t1)
        self.timings["kalman"].append(t3 - t2)
        self.timings["postprocess"].append(t4 - t3)

        return result

    def get_timing_summary(self):
        """Return median timings for each stage in microseconds."""
        summary = {}
        for stage, times in self.timings.items():
            if times:
                sorted_times = sorted(times)
                median = sorted_times[len(sorted_times) // 2]
                summary[stage] = median / 1000.0  # ns -> μs
        return summary


def generate_test_frame(height=480, width=640, seed=None):
    """Generate a synthetic test frame."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    frame = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return frame


def main():
    """Run the baseline pipeline and print timing results."""
    print("=" * 60)
    print("Baseline Tracker Pipeline")
    print("=" * 60)

    np.random.seed(42)
    pipeline = Pipeline()

    num_frames = 200
    warmup_frames = 20

    print(f"\nRunning {num_frames} frames ({warmup_frames} warmup)...")

    for i in range(num_frames):
        frame = generate_test_frame(height=120, width=160, seed=i)
        result = pipeline.process_frame(frame)

    # Discard warmup timings
    for stage in pipeline.timings:
        pipeline.timings[stage] = pipeline.timings[stage][warmup_frames:]

    summary = pipeline.get_timing_summary()

    print(f"\nPer-stage median times (over {num_frames - warmup_frames} frames):")
    print("-" * 40)
    total = 0.0
    for stage, us in summary.items():
        print(f"  {stage:<15s} {us:>10.1f} μs")
        total += us
    print("-" * 40)
    print(f"  {'TOTAL':<15s} {total:>10.1f} μs")

    print(f"\nLast result: {result}")
    print(f"Results history length: {len(pipeline.postprocessor.results_history)}")


if __name__ == "__main__":
    main()
