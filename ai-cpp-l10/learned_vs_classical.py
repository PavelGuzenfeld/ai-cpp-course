"""
Round 6: should this even be a learned model?

The optimization loop in this lesson always assumed the algorithm was
right and only the implementation was slow. This round asks the question
that comes before that: given a per-step latency budget, is a tuned
classical filter or a frozen "learned" one the right choice, and does
either's uncertainty output deserve to be trusted?

Run standalone:

    python3 learned_vs_classical.py
"""

import time

import numpy as np
from scipy import stats

from filter_comparison import FrozenGainFilter, MistunedKalman, TwoModeIMM, anees, coverage


def make_maneuvering_trajectory(n=200, dt=0.1, maneuver_start=100, accel_amplitude=15.0, seed=0):
    rng = np.random.default_rng(seed)
    true_pos = np.zeros(n)
    pos, vel = 0.0, 1.0
    for i in range(n):
        accel = accel_amplitude * np.sin(2 * np.pi * 0.8 * (i - maneuver_start) * dt) \
            if i >= maneuver_start else 0.0
        vel += accel * dt
        pos += vel * dt
        true_pos[i] = pos
    measurement_noise_std = 0.3
    z = true_pos + rng.normal(0, measurement_noise_std, size=n)
    return true_pos, z


def run_imm_vs_frozen(n=200, dt=0.1, maneuver_start=100):
    true_pos, z = make_maneuvering_trajectory(n=n, dt=dt, maneuver_start=maneuver_start)

    imm = TwoModeIMM(dt=dt)
    frozen = FrozenGainFilter(dt=dt, fit_regime_accel_std=0.05)

    imm_errors, imm_covs = [], []
    frozen_errors = []
    for i in range(n):
        x_imm, p_imm, _ = imm.step(z[i])
        imm_errors.append(true_pos[i] - x_imm[0])
        imm_covs.append(p_imm[:1, :1])

        x_frozen = frozen.step(z[i])
        frozen_errors.append(true_pos[i] - x_frozen[0])

    imm_errors = np.array(imm_errors)
    frozen_errors = np.array(frozen_errors)

    return {
        "imm_rmse": float(np.sqrt(np.mean(imm_errors[maneuver_start:] ** 2))),
        "frozen_rmse": float(np.sqrt(np.mean(frozen_errors[maneuver_start:] ** 2))),
        "imm_anees": anees([np.array([e]) for e in imm_errors], imm_covs),
        # The frozen filter never produces a covariance -- not omitted here,
        # structurally absent, same as gst-nvmm-cpp's vanilla learned filter
        # whose measurement dimension is smaller than its state dimension.
        "frozen_anees": None,
    }


def run_calibration_demo(n=300, dt=0.1, r_true=0.25):
    """Same RMSE-relevant dynamics, two calibrations: R set correctly, and
    R understated 10x (the filter believes its measurements are 10x more
    precise than they are). RMSE barely moves; ANEES and coverage do not."""
    rng = np.random.default_rng(1)
    true_pos = np.cumsum(rng.normal(1.0, 0.05, size=n)) * dt
    z = true_pos + rng.normal(0, np.sqrt(r_true), size=n)

    results = {}
    for label, r_scale in [("correct", 1.0), ("understated_10x", 0.1)]:
        kf = MistunedKalman(dt=dt, r_true=r_true, r_scale=r_scale)
        errors, covs, stds = [], [], []
        for i in range(n):
            x, p = kf.step(z[i])
            errors.append(true_pos[i] - x[0])
            covs.append(p[:1, :1])
            stds.append(np.sqrt(p[0, 0]))
        errors = np.array(errors)
        results[label] = {
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "anees": anees([np.array([e]) for e in errors], covs),
            "coverage_68": coverage(errors, np.array(stds), 1.0),
            "coverage_95": coverage(errors, np.array(stds), 2.0),
        }
    return results


def run_underflow_demo():
    """A heavy-tailed residual on one step, raw exp() vs log-space mode
    weighting. Same scenario, one line different."""
    imm_raw = TwoModeIMM(dt=0.1)
    imm_log = TwoModeIMM(dt=0.1)
    heavy_tailed_z = 500.0  # a residual far outside either mode's density

    log_space_ok = True
    try:
        imm_log.step(heavy_tailed_z, log_space=True)
    except FloatingPointError:
        log_space_ok = False

    raw_space_raised = False
    try:
        imm_raw.step(heavy_tailed_z, log_space=False)
    except FloatingPointError:
        raw_space_raised = True

    return {"log_space_survived": log_space_ok, "raw_space_underflowed": raw_space_raised}


def measure_latency(budget_ms, n=500):
    imm = TwoModeIMM(dt=0.1)
    _, z = make_maneuvering_trajectory(n=n)
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        imm.step(z[i])
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times[20:])  # drop warmup
    return {
        "median_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "budget_ms": budget_ms,
        "within_budget": bool(np.median(times) < budget_ms),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Round 6: learned vs. tuned classical, calibration, latency")
    print("=" * 60)

    comparison = run_imm_vs_frozen()
    print(f"\nIMM RMSE (post-maneuver): {comparison['imm_rmse']:.3f}")
    print(f"Frozen-gain RMSE (post-maneuver): {comparison['frozen_rmse']:.3f}")
    print(f"IMM ANEES (1 dof): {comparison['imm_anees']:.2f}")
    print("Frozen-gain ANEES: N/A -- this filter has no covariance to evaluate")

    print("\n--- Calibration: RMSE does not reveal miscalibration ---")
    calib = run_calibration_demo()
    chi2_lo, chi2_hi = stats.chi2.ppf(0.05, df=1), stats.chi2.ppf(0.95, df=1)
    print(f"chi-square(1 dof) 90% band: [{chi2_lo:.2f}, {chi2_hi:.2f}]")
    for label, r in calib.items():
        print(f"  {label}: RMSE={r['rmse']:.3f} ANEES={r['anees']:.2f} "
              f"coverage@68={r['coverage_68']:.2f} coverage@95={r['coverage_95']:.2f}")

    print("\n--- Log-sum-exp underflow ---")
    underflow = run_underflow_demo()
    print(f"log-space survives a heavy-tailed residual: {underflow['log_space_survived']}")
    print(f"raw exp() underflows on the same residual: {underflow['raw_space_underflowed']}")

    print("\n--- Latency budget ---")
    latency = measure_latency(budget_ms=2.0)
    print(f"median: {latency['median_ms']:.4f} ms, p99: {latency['p99_ms']:.4f} ms, "
          f"budget: {latency['budget_ms']} ms, within budget: {latency['within_budget']}")
