"""
Two measured benchmark-harness bugs from gst-nvmm-cpp, reproduced as
runnable demos: the wrong number first, then the fix.

1. Timing from the wrong start point (commit d1d8ce7): the clock started
   at pipeline startup, folding in preroll plus one-time engine warmup.
   The skew is length-dependent -- a fixed one-time cost is a large
   fraction of a short clip's total and a small fraction of a long one's,
   so short clips look disproportionately worse for no algorithmic reason.

2. An EMA of inter-arrival gaps over-reporting throughput (docs/harness.md):
   averaging gaps between events weights every event equally regardless of
   how much wall-clock time it represents. A burst of near-simultaneous
   arrivals pulls the EMA down and the implied rate up, long after the
   burst has already drained. Counting events in a wall-clock window is
   not fooled by this because it weights time, not events.
"""
from __future__ import annotations

import time

# --- Bug 1: timing from the wrong start point -------------------------------


def run_pipeline_wrong_start(n_frames: int, startup_s: float, per_frame_s: float) -> float:
    """BUG: the clock starts before the one-time warmup/preroll, so that
    fixed cost is folded into the reported per-frame time."""
    t0 = time.perf_counter()
    time.sleep(startup_s)
    for _ in range(n_frames):
        time.sleep(per_frame_s)
    elapsed = time.perf_counter() - t0
    return elapsed / n_frames


def run_pipeline_correct_start(n_frames: int, startup_s: float, per_frame_s: float) -> float:
    """FIX: the clock starts at the first unit of real work, after warmup
    has already happened."""
    time.sleep(startup_s)
    t0 = time.perf_counter()
    for _ in range(n_frames):
        time.sleep(per_frame_s)
    elapsed = time.perf_counter() - t0
    return elapsed / n_frames


# --- Bug 2: EMA of inter-arrival gaps over-reports on a bursty drain --------


def make_bursty_arrivals(burst_size: int, intra_burst_gap_s: float,
                          inter_burst_gap_s: float, n_bursts: int) -> list[float]:
    """Synthetic arrival timestamps: `n_bursts` bursts of `burst_size`
    events packed `intra_burst_gap_s` apart, separated by
    `inter_burst_gap_s` between the last event of one burst and the first
    of the next."""
    timestamps: list[float] = []
    t = 0.0
    for _ in range(n_bursts):
        for _ in range(burst_size):
            timestamps.append(t)
            t += intra_burst_gap_s
        t += inter_burst_gap_s - intra_burst_gap_s
    return timestamps


def ema_rate_estimate(timestamps: list[float], alpha: float) -> float:
    """BUG: rate estimated as 1 / EMA(inter-arrival gap), sampled at the
    last timestamp."""
    gaps = [t1 - t0 for t0, t1 in zip(timestamps, timestamps[1:])]
    ema_gap = gaps[0]
    for g in gaps[1:]:
        ema_gap = alpha * g + (1 - alpha) * ema_gap
    return 1.0 / ema_gap


def count_window_rate_estimate(timestamps: list[float], window_s: float) -> float:
    """FIX: count events in the last `window_s` seconds of wall-clock time
    and divide by that window."""
    t_end = timestamps[-1]
    count = sum(1 for t in timestamps if t > t_end - window_s)
    return count / window_s


# --- Warmup discard: choosing N by inspection -------------------------------


def choose_warmup_count(timings: list[float], stable_run: int = 5, tolerance: float = 0.2) -> int:
    """Returns the number of leading samples to discard: the first index
    from which `stable_run` consecutive samples all fall within
    `tolerance` (relative) of their own median. Chosen by inspecting the
    data itself, not by a fixed convention like "always discard 10"."""
    n = len(timings)
    for start in range(n - stable_run + 1):
        window = timings[start:start + stable_run]
        med = sorted(window)[len(window) // 2]
        if med > 0 and all(abs(x - med) / med <= tolerance for x in window):
            return start
    return 0
