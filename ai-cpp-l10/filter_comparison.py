"""
A reduced 2-mode IMM (constant-velocity / constant-acceleration, tuned
process noise) against a frozen-gain "learned" filter that never adapts to
a regime it wasn't fit on -- the shape of the real comparison in
gst-nvmm-cpp's kalmannet-vs-imm benchmark, not the whole benchmark.
"""

import numpy as np


def _kf_predict_update(x, p, f, q, h, r, z):
    x_pred = f @ x
    p_pred = f @ p @ f.T + q
    y = z - h @ x_pred
    s = h @ p_pred @ h.T + r
    k = p_pred @ h.T @ np.linalg.inv(s)
    x_new = x_pred + k @ y
    p_new = (np.eye(len(x)) - k @ h) @ p_pred
    likelihood = float(
        np.exp(-0.5 * (y @ np.linalg.solve(s, y))) / np.sqrt(np.linalg.det(2 * np.pi * s))
    )
    return x_new, p_new, likelihood, float(s[0, 0])


def _mix_initial_conditions(x_modes, p_modes, mode_probs, transition):
    """IMM mixing step: for each mode j, blend every mode i's prior state
    into a single initial condition, weighted by mode_probs[i] * P(i->j),
    normalized so each mode's blend weights sum to 1 -- without that
    normalization this silently stops being an IMM and just becomes an
    arbitrary linear combination."""
    c = transition.T @ mode_probs
    mix_weight = (transition * mode_probs[:, None]) / c[None, :]

    mixed_x, mixed_p = [], []
    n_modes = len(x_modes)
    for j in range(n_modes):
        x0 = sum(mix_weight[i, j] * x_modes[i] for i in range(n_modes))
        p0 = sum(
            mix_weight[i, j] * (p_modes[i] + np.outer(x_modes[i] - x0, x_modes[i] - x0))
            for i in range(n_modes)
        )
        mixed_x.append(x0)
        mixed_p.append(p0)
    return c, mixed_x, mixed_p


class TwoModeIMM:
    """A real (not merely averaged) 2-mode IMM: per-mode state/covariance
    carried across cycles, mixed at each cycle's start via a Markov mode
    transition matrix -- the step that makes this an IMM rather than two
    independent filters blended after the fact."""

    def __init__(self, dt=0.1, mode_persistence=0.95):
        self.dt = dt
        self.f_cv = np.array([[1, dt, 0], [0, 1, 0], [0, 0, 0]])
        self.f_ca = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]])
        self.q_cv = np.diag([0.01, 0.1, 0.0])
        self.q_ca = np.diag([0.01, 0.1, 4.0])
        self.h = np.array([[1.0, 0.0, 0.0]])
        self.r = np.array([[0.25]])
        p = mode_persistence
        self.transition = np.array([[p, 1 - p], [1 - p, p]])

        self.x_modes = [np.zeros(3), np.zeros(3)]
        self.p_modes = [np.eye(3) * 10.0, np.eye(3) * 10.0]
        self.mode_probs = np.array([0.5, 0.5])

    def step(self, z, log_space=True):
        # 1-2. Mix each mode's initial condition from the prior cycle's
        # per-mode state/covariance, weighted by the Markov transition.
        c, mixed_x, mixed_p = _mix_initial_conditions(
            self.x_modes, self.p_modes, self.mode_probs, self.transition)

        # 3. Mode-matched filtering.
        z_vec = np.array([z])
        x_cv, p_cv, lik_cv, s_cv = _kf_predict_update(
            mixed_x[0], mixed_p[0], self.f_cv, self.q_cv, self.h, self.r, z_vec)
        x_ca, p_ca, lik_ca, s_ca = _kf_predict_update(
            mixed_x[1], mixed_p[1], self.f_ca, self.q_ca, self.h, self.r, z_vec)
        self.x_modes = [x_cv, x_ca]
        self.p_modes = [p_cv, p_ca]

        # 4. Mode probability update.
        if log_space:
            log_lik = np.array([np.log(max(lik_cv, 1e-300)), np.log(max(lik_ca, 1e-300))])
            log_w = log_lik + np.log(c)
            peak = np.max(log_w)
            w = np.exp(log_w - peak)
        else:
            # Raw likelihoods: a heavy-tailed outlier drives both densities
            # to exactly 0.0 in float64, and the normalization below divides
            # zero by zero.
            w = np.array([lik_cv, lik_ca]) * c

        total = w.sum()
        if total == 0.0:
            raise FloatingPointError("mode likelihoods underflowed to zero -- cannot normalize")
        self.mode_probs = w / total

        # 5. Combined output estimate.
        x = self.mode_probs[0] * x_cv + self.mode_probs[1] * x_ca
        p = self.mode_probs[0] * (p_cv + np.outer(x_cv - x, x_cv - x)) + \
            self.mode_probs[1] * (p_ca + np.outer(x_ca - x, x_ca - x))
        innovation_var = self.mode_probs[0] * s_cv + self.mode_probs[1] * s_ca
        return x.copy(), p.copy(), innovation_var


class FrozenGainFilter:
    """A "learned" filter stand-in: a Kalman gain fit once, offline, on a
    low-maneuver regime, then frozen -- it never adapts online, and unlike
    the IMM it never carries a covariance to report. That absence is itself
    the point: gst-nvmm-cpp's vanilla learned filter could not produce one
    either, because its measurement dimension is smaller than its state
    dimension -- not omitted, structurally impossible."""

    def __init__(self, dt=0.1, fit_regime_accel_std=0.05):
        f = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]])
        q = np.diag([0.01, 0.1, fit_regime_accel_std**2])
        h = np.array([[1.0, 0.0, 0.0]])
        r = np.array([[0.25]])
        p = np.eye(3) * 10.0
        for _ in range(200):
            p_pred = f @ p @ f.T + q
            s = h @ p_pred @ h.T + r
            k = p_pred @ h.T @ np.linalg.inv(s)
            p = (np.eye(3) - k @ h) @ p_pred
        self._f = f
        self._k_frozen = k
        self._h = h
        self.x = np.zeros(3)

    def step(self, z):
        x_pred = self._f @ self.x
        y = np.array([z]) - self._h @ x_pred
        self.x = x_pred + self._k_frozen @ y
        return self.x.copy()


class MistunedKalman:
    """A single constant-velocity Kalman filter whose R can be understated
    relative to the true measurement noise -- same RMSE-relevant dynamics
    as a correctly tuned filter, wrong calibration. Demonstrates that RMSE
    alone cannot tell a well-calibrated filter from an overconfident one."""

    def __init__(self, dt=0.1, r_true=0.25, r_scale=1.0):
        self.f = np.array([[1, dt], [0, 1]])
        self.q = np.diag([0.01, 0.1])
        self.h = np.array([[1.0, 0.0]])
        self.r = np.array([[r_true * r_scale]])
        self.x = np.zeros(2)
        self.p = np.eye(2) * 10.0

    def step(self, z):
        self.x, self.p, _likelihood, s = _kf_predict_update(
            self.x, self.p, self.f, self.q, self.h, self.r, np.array([z]))
        return self.x.copy(), self.p.copy()


def anees(errors, covariances):
    """Average normalized estimation error squared: mean of e^T P^-1 e."""
    values = [float(e @ np.linalg.solve(p, e)) for e, p in zip(errors, covariances)]
    return float(np.mean(values))


def coverage(errors, stds, n_sigma):
    """Fraction of |error| within n_sigma standard deviations."""
    return float(np.mean(np.abs(errors) <= n_sigma * stds))
