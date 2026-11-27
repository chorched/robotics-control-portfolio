import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toa_ekf.core import calibrate, default_setup, measure, toa, toa_jacobian, wnls  # noqa: E402
from toa_ekf.experiment import run_once  # noqa: E402


def test_jacobian_matches_finite_differences():
    s = default_setup()
    p = np.array([2.3, 1.7])
    J = toa_jacobian(p, s.mics)
    h = 1e-6
    num = np.column_stack([(toa(p + h * e, s.mics) - toa(p - h * e, s.mics)) / (2 * h) for e in np.eye(2)])
    assert np.allclose(J, num, rtol=1e-6, atol=1e-12)


def test_calibration_recovers_bias_and_noise():
    s = default_setup()
    rng = np.random.default_rng(1)
    p0 = np.array([3.0, 2.0])
    Y = np.array([measure(p0, s, rng) for _ in range(5000)])
    b, v = calibrate(Y, p0, s.mics)
    assert np.all(np.abs(b - s.bias) < 4 * s.sigma / np.sqrt(5000))
    assert np.allclose(np.sqrt(v), s.sigma, rtol=0.05)


def test_wnls_is_exact_without_noise():
    s = default_setup()
    p = np.array([4.2, 1.1])
    est, cov = wnls(toa(p, s.mics) + s.bias, s.bias, s.sigma**2, s.mics, np.array([3.0, 2.0]))
    assert np.allclose(est, p, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_filters_are_consistent_and_beat_per_pulse_fixes():
    r = run_once(0)
    assert r["rmse"]["ekf"] < r["rmse"]["wnls"] < r["rmse"]["wnls_no_bias_correction"]
    assert 6.0 < r["nis_mean"]["ekf"] < 10.0  # dof 8
    assert 1.0 < r["nis_mean"]["pseudo_kf"] < 3.0  # dof 2
