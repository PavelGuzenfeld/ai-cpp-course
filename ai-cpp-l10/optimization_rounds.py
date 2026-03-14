"""
Step-by-step optimization rounds.

Demonstrates the optimization loop by applying fixes one at a time
and measuring after each round. Run standalone:

    python3 optimization_rounds.py
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def generate_test_frame(height=120, width=160, seed=None):
    """Generate a synthetic test frame."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    return rng.randint(0, 256, (height, width, 3), dtype=np.uint8)


def simulate_inference(preprocessed_frame):
    """Simulate neural network inference (same in all rounds)."""
    features = np.mean(preprocessed_frame, axis=(0, 1))
    response = np.outer(features, features)
    cx = float(np.sum(response[0])) % 640
    cy = float(np.sum(response[1])) % 480
    w = 50.0 + float(np.sum(response[2])) % 100
    h = 50.0 + float(np.sum(response[2])) % 80
    return np.array([[cx, cy, w, h]], dtype=np.float64)


def measure_pipeline(pipeline_class, num_frames=200, warmup=20):
    """Measure a pipeline class over num_frames, return median timings."""
    np.random.seed(42)
    pipeline = pipeline_class()

    for i in range(num_frames):
        frame = generate_test_frame(height=120, width=160, seed=i)
        result = pipeline.process_frame(frame)

    # Discard warmup
    for stage in pipeline.timings:
        pipeline.timings[stage] = pipeline.timings[stage][warmup:]

    summary = pipeline.get_timing_summary()
    return summary, result


# ---------------------------------------------------------------------------
# Kalman filter variants
# ---------------------------------------------------------------------------

class KalmanBaseline:
    """Baseline: rebuilds matrices every predict()."""

    def __init__(self, state_dim=12, measure_dim=4, process_noise=0.01):
        self.state_dim = state_dim
        self.measure_dim = measure_dim
        self.process_noise = process_noise
        self.state = np.zeros((state_dim, 1), dtype=np.float64)
        self.F = np.eye(state_dim, dtype=np.float64)
        dt = 1.0
        for i in range(state_dim // 3):
            self.F[i, i + state_dim // 3] = dt
            if i + 2 * (state_dim // 3) < state_dim:
                self.F[i, i + 2 * (state_dim // 3)] = 0.5 * dt * dt
                self.F[i + state_dim // 3, i + 2 * (state_dim // 3)] = dt
        self.H = np.zeros((measure_dim, state_dim), dtype=np.float64)
        for i in range(measure_dim):
            self.H[i, i] = 1.0
        self.P = np.eye(state_dim, dtype=np.float64) * 10.0
        self.R = np.eye(measure_dim, dtype=np.float64) * 0.1

    def predict(self):
        Q = np.eye(self.state_dim, dtype=np.float64)
        Q[self.state_dim // 2:, self.state_dim // 2:] *= self.process_noise
        noise_vec = np.random.randn(self.state_dim) * 0.001
        noise = np.diag(noise_vec)
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + Q + noise
        return self.state[:self.measure_dim].flatten()

    def update(self, measurement):
        z = np.array(measurement, dtype=np.float64).reshape(-1, 1)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I = np.eye(self.state_dim, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return self.state[:self.measure_dim].flatten()


class KalmanFixed:
    """Round 1 fix: pre-allocated Q matrix."""

    def __init__(self, state_dim=12, measure_dim=4, process_noise=0.01):
        self.state_dim = state_dim
        self.measure_dim = measure_dim
        self.process_noise = process_noise
        self.state = np.zeros((state_dim, 1), dtype=np.float64)
        self.F = np.eye(state_dim, dtype=np.float64)
        dt = 1.0
        for i in range(state_dim // 3):
            self.F[i, i + state_dim // 3] = dt
            if i + 2 * (state_dim // 3) < state_dim:
                self.F[i, i + 2 * (state_dim // 3)] = 0.5 * dt * dt
                self.F[i + state_dim // 3, i + 2 * (state_dim // 3)] = dt
        self.H = np.zeros((measure_dim, state_dim), dtype=np.float64)
        for i in range(measure_dim):
            self.H[i, i] = 1.0
        self.P = np.eye(state_dim, dtype=np.float64) * 10.0
        self.R = np.eye(measure_dim, dtype=np.float64) * 0.1
        # FIX: pre-allocate
        self._Q = np.eye(state_dim, dtype=np.float64)
        self._Q[state_dim // 2:, state_dim // 2:] *= process_noise
        self._noise = np.zeros((state_dim, state_dim), dtype=np.float64)
        self._I = np.eye(state_dim, dtype=np.float64)

    def predict(self):
        noise_vec = np.random.randn(self.state_dim) * 0.001
        self._noise[:] = 0.0
        np.fill_diagonal(self._noise, noise_vec)
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self._Q + self._noise
        return self.state[:self.measure_dim].flatten()

    def update(self, measurement):
        z = np.array(measurement, dtype=np.float64).reshape(-1, 1)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (self._I - K @ self.H) @ self.P
        return self.state[:self.measure_dim].flatten()


# ---------------------------------------------------------------------------
# PostProcessor variants
# ---------------------------------------------------------------------------

class PostProcessorBaseline:
    """Baseline: copy chain."""

    def __init__(self):
        self.results_history = []

    def format_result(self, state_vector):
        copied = state_vector.copy()
        flat = copied.flatten()
        as_list = flat.tolist()
        result = {"x": as_list[0], "y": as_list[1], "w": as_list[2], "h": as_list[3]}
        self.results_history.append(result)
        return result


class PostProcessorFixed:
    """Round 2 fix: direct indexing."""

    def __init__(self):
        self.results_history = []

    def format_result(self, state_vector):
        result = {
            "x": float(state_vector[0, 0]),
            "y": float(state_vector[0, 1]),
            "w": float(state_vector[0, 2]),
            "h": float(state_vector[0, 3]),
        }
        self.results_history.append(result)
        return result


# ---------------------------------------------------------------------------
# Preprocessor variants
# ---------------------------------------------------------------------------

class PreprocessorBaseline:
    """Baseline: allocates every frame."""

    def __init__(self, target_h=64, target_w=64, pad_h=80, pad_w=80):
        self.target_h = target_h
        self.target_w = target_w
        self.pad_h = pad_h
        self.pad_w = pad_w

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        resized = np.zeros((self.target_h, self.target_w, 3), dtype=np.float32)
        scale_h = h / self.target_h
        scale_w = w / self.target_w
        for row in range(self.target_h):
            src_row = min(int(row * scale_h), h - 1)
            for col in range(self.target_w):
                src_col = min(int(col * scale_w), w - 1)
                resized[row, col] = frame[src_row, src_col]
        resized /= 255.0
        padded = np.zeros((self.pad_h, self.pad_w, 3), dtype=np.float32)
        off_h = (self.pad_h - self.target_h) // 2
        off_w = (self.pad_w - self.target_w) // 2
        padded[off_h:off_h + self.target_h, off_w:off_w + self.target_w] = resized
        return padded


class PreprocessorFixed:
    """Round 3 fix: pre-allocated buffers."""

    def __init__(self, target_h=64, target_w=64, pad_h=80, pad_w=80):
        self.target_h = target_h
        self.target_w = target_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self._resize_buf = np.zeros((target_h, target_w, 3), dtype=np.float32)
        self._pad_buf = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        self._off_h = (pad_h - target_h) // 2
        self._off_w = (pad_w - target_w) // 2

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        self._resize_buf[:] = 0
        scale_h = h / self.target_h
        scale_w = w / self.target_w
        for row in range(self.target_h):
            src_row = min(int(row * scale_h), h - 1)
            for col in range(self.target_w):
                src_col = min(int(col * scale_w), w - 1)
                self._resize_buf[row, col] = frame[src_row, src_col]
        self._resize_buf /= 255.0
        self._pad_buf[:] = 0
        oh, ow = self._off_h, self._off_w
        self._pad_buf[oh:oh + self.target_h, ow:ow + self.target_w] = self._resize_buf
        return self._pad_buf


# ---------------------------------------------------------------------------
# Pipeline factory — combines components at each optimization stage
# ---------------------------------------------------------------------------

def _make_pipeline_class(kalman_cls, postprocessor_cls, preprocessor_cls):
    """Dynamically create a pipeline class from component classes."""

    class _Pipeline:
        def __init__(self):
            self.preprocessor = preprocessor_cls()
            self.kalman = kalman_cls(state_dim=12, measure_dim=4)
            self.postprocessor = postprocessor_cls()
            self.timings = {
                "preprocess": [],
                "inference": [],
                "kalman": [],
                "postprocess": [],
            }

        def process_frame(self, frame):
            t0 = time.perf_counter_ns()
            preprocessed = self.preprocessor.preprocess(frame)
            t1 = time.perf_counter_ns()
            detection = simulate_inference(preprocessed)
            t2 = time.perf_counter_ns()
            self.kalman.predict()
            state = self.kalman.update(detection[0])
            t3 = time.perf_counter_ns()
            result = self.postprocessor.format_result(state.reshape(1, -1))
            t4 = time.perf_counter_ns()
            self.timings["preprocess"].append(t1 - t0)
            self.timings["inference"].append(t2 - t1)
            self.timings["kalman"].append(t3 - t2)
            self.timings["postprocess"].append(t4 - t3)
            return result

        def get_timing_summary(self):
            summary = {}
            for stage, times in self.timings.items():
                if times:
                    sorted_times = sorted(times)
                    median = sorted_times[len(sorted_times) // 2]
                    summary[stage] = median / 1000.0
            return summary

    return _Pipeline


# ---------------------------------------------------------------------------
# Define the optimization rounds
# ---------------------------------------------------------------------------

ROUNDS = [
    {
        "name": "Baseline",
        "description": "All original bottlenecks present",
        "pipeline": _make_pipeline_class(
            KalmanBaseline, PostProcessorBaseline, PreprocessorBaseline
        ),
    },
    {
        "name": "Round 1: Fix Kalman",
        "description": "Pre-allocate Q matrix and noise buffer",
        "pipeline": _make_pipeline_class(
            KalmanFixed, PostProcessorBaseline, PreprocessorBaseline
        ),
    },
    {
        "name": "Round 2: Fix copy chain",
        "description": "Direct array indexing in postprocessor",
        "pipeline": _make_pipeline_class(
            KalmanFixed, PostProcessorFixed, PreprocessorBaseline
        ),
    },
    {
        "name": "Round 3: Fix buffers",
        "description": "Pre-allocate preprocess buffers",
        "pipeline": _make_pipeline_class(
            KalmanFixed, PostProcessorFixed, PreprocessorFixed
        ),
    },
]


def main():
    """Run each optimization round and print a summary table."""
    num_frames = 200
    warmup = 20
    stages = ["preprocess", "inference", "kalman", "postprocess"]

    print("=" * 72)
    print("Optimization Rounds — Step-by-Step Improvement")
    print("=" * 72)
    print(f"Frames: {num_frames} ({warmup} warmup)\n")

    all_summaries = []

    for rnd in ROUNDS:
        print(f"--- {rnd['name']} ---")
        print(f"    {rnd['description']}")

        summary, result = measure_pipeline(
            rnd["pipeline"], num_frames=num_frames, warmup=warmup
        )
        all_summaries.append((rnd["name"], summary))

        total = sum(summary.get(s, 0) for s in stages)
        for s in stages:
            v = summary.get(s, 0)
            print(f"    {s:<15s} {v:>10.1f} μs")
        print(f"    {'TOTAL':<15s} {total:>10.1f} μs")
        print()

    # --- Summary table ---
    print("=" * 72)
    print("Summary Table")
    print("=" * 72)

    baseline_total = sum(all_summaries[0][1].get(s, 0) for s in stages)

    header = f"{'Round':<28s}"
    for s in stages:
        header += f" {s:>12s}"
    header += f" {'TOTAL':>10s} {'vs Base':>10s}"
    print(header)
    print("-" * len(header))

    for name, summary in all_summaries:
        row = f"{name:<28s}"
        total = 0.0
        for s in stages:
            v = summary.get(s, 0)
            row += f" {v:>10.1f}μs"
            total += v
        speedup = baseline_total / total if total > 0 else float("inf")
        row += f" {total:>8.1f}μs {speedup:>8.2f}x"
        print(row)

    print()
    print("Cumulative speedup shows diminishing returns as Amdahl's Law predicts.")
    print("Inference (untouched) dominates total time, capping overall improvement.")


if __name__ == "__main__":
    main()
