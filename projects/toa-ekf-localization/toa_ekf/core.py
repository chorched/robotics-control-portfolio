r"""Acoustic TOA localization of a mobile robot: model, calibration and three estimators.

A robot emits a sound pulse every ``dt`` seconds at a known emission time. Eight
microphones at known positions :math:`m_i` record the time of arrival

.. math::  y_i = \frac{\lVert p - m_i \rVert}{c} + b_i + e_i, \qquad e_i \sim \mathcal N(0, \sigma_i^2),

with an unknown per-microphone timing bias :math:`b_i` (cable and ADC latency) and noise
variance :math:`\sigma_i^2`.

1. **Calibration.** With the robot parked at a known position :math:`p_0`, the bias and
   variance of each channel follow from the sample mean and variance of
   :math:`y_i - \lVert p_0 - m_i\rVert / c`.
2. **Weighted nonlinear least squares (WNLS).** Per pulse, Gauss-Newton on the bias-corrected
   ranges with weights :math:`1/\sigma_i^2`; covariance :math:`(H^\top W H)^{-1}`.
3. **Pseudo-measurement Kalman filter.** A constant-velocity Kalman filter that treats each
   WNLS fix as a linear position measurement with the WNLS covariance.
4. **Direct-TOA EKF.** The same motion model, but the eight TOAs are the measurement, with the
   analytical Jacobian :math:`\partial y_i / \partial p = (p - m_i)^\top / (c\,\lVert p - m_i\rVert)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_SOUND = 343.0  # m/s


@dataclass
class Setup:
    mics: np.ndarray  # (n, 2) microphone positions, m
    bias: np.ndarray  # (n,) true timing bias, s
    sigma: np.ndarray  # (n,) true timing noise std, s
    dt: float = 0.1  # pulse period, s

    @property
    def n(self) -> int:
        return len(self.mics)


def default_setup(seed: int = 0) -> Setup:
    """Eight microphones around a 6 m x 4 m room, biases up to 0.5 ms, noise 40-120 us."""
    rng = np.random.default_rng(seed)
    mics = np.array([[0, 0], [3, -0.2], [6, 0], [6.2, 2], [6, 4], [3, 4.2], [0, 4], [-0.2, 2]], float)
    bias = rng.uniform(-5e-4, 5e-4, len(mics))
    sigma = rng.uniform(4e-5, 1.2e-4, len(mics))
    return Setup(mics, bias, sigma)


# ------------------------------------------------------------------------ model


def toa(p: np.ndarray, mics: np.ndarray) -> np.ndarray:
    """Noise- and bias-free time of arrival from position p (2,) to every microphone."""
    return np.linalg.norm(p[None, :] - mics, axis=1) / C_SOUND


def toa_jacobian(p: np.ndarray, mics: np.ndarray) -> np.ndarray:
    """d toa / d p, shape (n, 2)."""
    diff = p[None, :] - mics
    return diff / (C_SOUND * np.linalg.norm(diff, axis=1, keepdims=True))


def measure(p: np.ndarray, s: Setup, rng: np.random.Generator) -> np.ndarray:
    return toa(p, s.mics) + s.bias + rng.normal(0.0, s.sigma)


# ------------------------------------------------------------------ calibration


def calibrate(y_static: np.ndarray, p0: np.ndarray, mics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """y_static: (N, n) TOAs recorded at the known position p0 -> (bias_hat, var_hat)."""
    r = y_static - toa(p0, mics)[None, :]
    return r.mean(axis=0), r.var(axis=0, ddof=1)


# ------------------------------------------------------------------- estimators


def wnls(y: np.ndarray, bias: np.ndarray, var: np.ndarray, mics: np.ndarray, p_init: np.ndarray,
         iters: int = 20, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Newton fix of one pulse. Returns (position, covariance)."""
    W = np.diag(1.0 / var)
    p = np.asarray(p_init, float).copy()
    for _ in range(iters):
        H = toa_jacobian(p, mics)
        r = y - bias - toa(p, mics)
        step = np.linalg.solve(H.T @ W @ H, H.T @ W @ r)
        p = p + step
        if np.linalg.norm(step) < tol:
            break
    H = toa_jacobian(p, mics)
    return p, np.linalg.inv(H.T @ W @ H)


def cv_model(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity model, state [x, y, vx, vy], white-acceleration noise intensity q."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    Q = np.zeros((4, 4))
    for i in range(2):
        Q[np.ix_([i, i + 2], [i, i + 2])] = q * G
    return F, Q


@dataclass
class Track:
    x: np.ndarray  # (T, 4) estimates
    P: np.ndarray  # (T, 4, 4) covariances
    nis: np.ndarray  # (T,) normalised innovation squared
    dof: int  # measurement dimension of the NIS


def pseudo_kf(fixes: np.ndarray, fix_covs: np.ndarray, dt: float, q: float, x0, P0) -> Track:
    F, Q = cv_model(dt, q)
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    x, P = np.asarray(x0, float), np.asarray(P0, float)
    xs, Ps, nis = [], [], []
    for z, R in zip(fixes, fix_covs):
        x, P = F @ x, F @ P @ F.T + Q
        S = H @ P @ H.T + R
        nu = z - H @ x
        K = P @ H.T @ np.linalg.inv(S)
        x, P = x + K @ nu, (np.eye(4) - K @ H) @ P
        xs.append(x), Ps.append(P), nis.append(float(nu @ np.linalg.solve(S, nu)))
    return Track(np.array(xs), np.array(Ps), np.array(nis), 2)


def ekf(ys: np.ndarray, bias: np.ndarray, var: np.ndarray, mics: np.ndarray, dt: float, q: float, x0, P0) -> Track:
    F, Q = cv_model(dt, q)
    R = np.diag(var)
    x, P = np.asarray(x0, float), np.asarray(P0, float)
    xs, Ps, nis = [], [], []
    for y in ys:
        x, P = F @ x, F @ P @ F.T + Q
        Hp = toa_jacobian(x[:2], mics)
        H = np.hstack([Hp, np.zeros_like(Hp)])
        S = H @ P @ H.T + R
        nu = y - bias - toa(x[:2], mics)
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ nu
        IKH = np.eye(4) - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T  # Joseph form
        xs.append(x), Ps.append(P), nis.append(float(nu @ np.linalg.solve(S, nu)))
    return Track(np.array(xs), np.array(Ps), np.array(nis), len(mics))


# -------------------------------------------------------------------- scenario


def trajectory(T: int, dt: float) -> np.ndarray:
    """A smooth lap through the room: (T, 4) true states [x, y, vx, vy]."""
    t = np.arange(T) * dt
    w = 2 * np.pi / (T * dt)
    x = 3.0 + 2.0 * np.sin(w * t)
    y = 2.0 + 1.2 * np.sin(2 * w * t)
    return np.column_stack([x, y, 2.0 * w * np.cos(w * t), 2.4 * w * np.cos(2 * w * t)])


def nees(track: Track, truth: np.ndarray) -> np.ndarray:
    e = truth[:, :2] - track.x[:, :2]
    return np.einsum("ti,tij,tj->t", e, np.linalg.inv(track.P[:, :2, :2]), e)


def ellipse(mean: np.ndarray, cov: np.ndarray, k: float = 2.4477, n: int = 60) -> np.ndarray:
    """95% confidence ellipse of a 2-D Gaussian (k^2 = chi2_2(0.95) = 5.991)."""
    vals, vecs = np.linalg.eigh(cov)
    a = np.linspace(0, 2 * np.pi, n)
    return mean[None, :] + (vecs @ (k * np.sqrt(vals)[:, None] * np.vstack([np.cos(a), np.sin(a)]))).T
