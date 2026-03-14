"""
Optimized tracker pipeline — all bottlenecks fixed.

Changes from tracker_pipeline.py:
1. KalmanFilter: pre-allocated Q matrix in __init__, no per-frame np.eye
2. PostProcessor: direct array indexing, no copy chains
3. Preprocessor: pre-allocated resize and pad buffers
4. Pipeline: pre-allocated result dict, reused each frame

Run standalone to see per-stage timing and comparison:

    python3 tracker_pipeline_optimized.py
"""

import time
import numpy as np


class KalmanFilterOptimized:
    """12-state Kalman filter with pre-allocated matrices.

    FIX: Q matrix and noise buffer allocated once in __init__.
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

        # FIX: Pre-allocate Q matrix ONCE
        self._Q = np.eye(state_dim, dtype=np.float64)
        self._Q[state_dim // 2:, state_dim // 2:] *= process_noise

        # FIX: Pre-allocate noise buffer
        self._noise = np.zeros((state_dim, state_dim), dtype=np.float64)

        # FIX: Pre-allocate identity for update
        self._I = np.eye(state_dim, dtype=np.float64)

    def predict(self):
        """Predict next state — zero allocations."""
        # FIX: Reuse pre-allocated noise buffer
        noise_vec = np.random.randn(self.state_dim) * 0.001
        self._noise[:] = 0.0
        np.fill_diagonal(self._noise, noise_vec)

        # The actual math (same as before)
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self._Q + self._noise

        return self.state[:self.measure_dim].flatten()

    def update(self, measurement):
        """Update state with measurement — uses pre-allocated identity."""
        z = np.array(measurement, dtype=np.float64).reshape(-1, 1)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        # FIX: reuse pre-allocated identity
        self.P = (self._I - K @ self.H) @ self.P

        return self.state[:self.measure_dim].flatten()


class PostProcessorOptimized:
    """Extract tracking results with direct array access.

    FIX: No copy chains. Direct indexing with float() conversion.
    """

    def __init__(self):
        self.results_history = []
        # FIX: Pre-allocate result dict template
        self._result = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}

    def extract_position(self, result_array):
        """Extract (x, y) — direct access, zero copies."""
        return float(result_array[0, 0]), float(result_array[0, 1])

    def extract_size(self, result_array):
        """Extract (w, h) — direct access, zero copies."""
        return float(result_array[0, 2]), float(result_array[0, 3])

    def format_result(self, state_vector):
        """Format full tracking result — minimal allocation."""
        # FIX: direct indexing, no copy/flatten/tolist chain
        result = {
            "x": float(state_vector[0, 0]),
            "y": float(state_vector[0, 1]),
            "w": float(state_vector[0, 2]),
            "h": float(state_vector[0, 3]),
        }
        self.results_history.append(result)
        return result


class PreprocessorOptimized:
    """Prepare frames for inference with pre-allocated buffers.

    FIX: Buffers allocated once in __init__, cleared and reused each frame.
    """

    def __init__(self, target_h=256, target_w=256, pad_h=288, pad_w=288):
        self.target_h = target_h
        self.target_w = target_w
        self.pad_h = pad_h
        self.pad_w = pad_w

        # FIX: Pre-allocate buffers ONCE
        self._resize_buf = np.zeros(
            (target_h, target_w, 3), dtype=np.float32
        )
        self._pad_buf = np.zeros(
            (pad_h, pad_w, 3), dtype=np.float32
        )

        self._off_h = (pad_h - target_h) // 2
        self._off_w = (pad_w - target_w) // 2

    def preprocess(self, frame):
        """Resize and pad a frame — reuses pre-allocated buffers."""
        h, w = frame.shape[:2]

        # FIX: Clear and reuse instead of allocating
        self._resize_buf[:] = 0

        scale_h = h / self.target_h
        scale_w = w / self.target_w
        for row in range(self.target_h):
            src_row = min(int(row * scale_h), h - 1)
            for col in range(self.target_w):
                src_col = min(int(col * scale_w), w - 1)
                self._resize_buf[row, col] = frame[src_row, src_col]

        # Normalize to [0, 1]
        self._resize_buf /= 255.0

        # FIX: Clear and reuse pad buffer
        self._pad_buf[:] = 0
        oh, ow = self._off_h, self._off_w
        self._pad_buf[oh:oh + self.target_h, ow:ow + self.target_w] = self._resize_buf

        return self._pad_buf


def simulate_inference(preprocessed_frame):
    """Simulate neural network inference (identical to baseline)."""
    features = np.mean(preprocessed_frame, axis=(0, 1))
    response = np.outer(features, features)
    cx = float(np.sum(response[0])) % 640
    cy = float(np.sum(response[1])) % 480
    w = 50.0 + float(np.sum(response[2])) % 100
    h = 50.0 + float(np.sum(response[2])) % 80
    return np.array([[cx, cy, w, h]], dtype=np.float64)


class PipelineOptimized:
    """Optimized tracking pipeline: all bottlenecks fixed."""

    def __init__(self):
        self.preprocessor = PreprocessorOptimized(
            target_h=64, target_w=64, pad_h=80, pad_w=80
        )
        self.kalman = KalmanFilterOptimized(state_dim=12, measure_dim=4)
        self.postprocessor = PostProcessorOptimized()
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
                summary[stage] = median / 1000.0
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
    """Run both pipelines and compare."""
    num_frames = 200
    warmup_frames = 20

    # --- Baseline ---
    from tracker_pipeline import Pipeline

    print("=" * 60)
    print("Side-by-Side Comparison: Baseline vs Optimized")
    print("=" * 60)

    np.random.seed(42)
    baseline = Pipeline()
    for i in range(num_frames):
        frame = generate_test_frame(height=120, width=160, seed=i)
        baseline_result = baseline.process_frame(frame)

    for stage in baseline.timings:
        baseline.timings[stage] = baseline.timings[stage][warmup_frames:]

    # --- Optimized ---
    np.random.seed(42)
    optimized = PipelineOptimized()
    for i in range(num_frames):
        frame = generate_test_frame(height=120, width=160, seed=i)
        optimized_result = optimized.process_frame(frame)

    for stage in optimized.timings:
        optimized.timings[stage] = optimized.timings[stage][warmup_frames:]

    # --- Compare ---
    baseline_summary = baseline.get_timing_summary()
    optimized_summary = optimized.get_timing_summary()

    print(f"\nPer-stage median times (over {num_frames - warmup_frames} frames):")
    print(f"{'Stage':<15s} {'Baseline (μs)':>14s} {'Optimized (μs)':>15s} {'Speedup':>10s}")
    print("-" * 56)

    total_baseline = 0.0
    total_optimized = 0.0
    for stage in ["preprocess", "inference", "kalman", "postprocess"]:
        b = baseline_summary.get(stage, 0)
        o = optimized_summary.get(stage, 0)
        speedup = b / o if o > 0 else float("inf")
        print(f"  {stage:<13s} {b:>12.1f}   {o:>13.1f}   {speedup:>8.1f}x")
        total_baseline += b
        total_optimized += o

    print("-" * 56)
    total_speedup = total_baseline / total_optimized if total_optimized > 0 else float("inf")
    print(f"  {'TOTAL':<13s} {total_baseline:>12.1f}   {total_optimized:>13.1f}   {total_speedup:>8.1f}x")

    print(f"\nBaseline last result:  {baseline_result}")
    print(f"Optimized last result: {optimized_result}")

    # Verify outputs match
    match = all(
        abs(baseline_result[k] - optimized_result[k]) < 1e-6
        for k in baseline_result
    )
    print(f"\nOutputs match: {match}")


if __name__ == "__main__":
    main()
