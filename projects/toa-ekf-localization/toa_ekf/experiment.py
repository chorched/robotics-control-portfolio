"""Run the full study on simulated data and write results/metrics.json + results/run_example.npz.

    python -m toa_ekf.experiment
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import (calibrate, default_setup, ekf, measure, nees, pseudo_kf, toa, trajectory, wnls)

CHI2_95 = {2: 5.991, 8: 15.507}
T = 300  # 30 s at 10 Hz
Q = 0.05  # white-acceleration intensity used by the filters
P0_GOOD = np.diag([0.5**2, 0.5**2, 0.3**2, 0.3**2])


def simulate(seed: int, setup=None):
    s = setup or default_setup(0)
    rng = np.random.default_rng(seed)
    # truth: smooth lap plus small random accelerations
    X = trajectory(T, s.dt)
    X[:, :2] += np.cumsum(np.cumsum(rng.normal(0, 0.002, (T, 2)), axis=0), axis=0) * 0.05
    Y = np.array([measure(x[:2], s, rng) for x in X])
    p0 = np.array([3.0, 2.0])
    Y_static = np.array([measure(p0, s, rng) for _ in range(200)])
    return s, X, Y, Y_static, p0


def run_once(seed: int, x0_err=(0.3, -0.3), P0=P0_GOOD) -> dict:
    s, X, Y, Y_static, p0 = simulate(seed)
    b_hat, v_hat = calibrate(Y_static, p0, s.mics)

    fixes, covs, raw = [], [], []
    guess = p0.copy()
    for y in Y:
        p, Pc = wnls(y, b_hat, v_hat, s.mics, guess)
        fixes.append(p), covs.append(Pc)
        raw.append(wnls(y, np.zeros(s.n), v_hat, s.mics, guess)[0])  # ignores the timing bias
        guess = p
    fixes, covs, raw = np.array(fixes), np.array(covs), np.array(raw)

    x0 = np.r_[X[0, :2] + np.array(x0_err), 0.0, 0.0]
    kf = pseudo_kf(fixes, covs, s.dt, Q, x0, P0)
    ek = ekf(Y, b_hat, v_hat, s.mics, s.dt, Q, x0, P0)

    def rmse(est):
        return float(np.sqrt(np.mean(np.sum((est[:, :2] - X[:, :2]) ** 2, axis=1))))

    return {
        "setup": s, "X": X, "Y": Y, "fixes": fixes, "fix_covs": covs, "raw": raw, "kf": kf, "ekf": ek,
        "bias_hat": b_hat, "var_hat": v_hat, "Y_static": Y_static, "p0": p0,
        "rmse": {"wnls_no_bias_correction": rmse(raw), "wnls": rmse(fixes), "pseudo_kf": rmse(kf.x),
                 "ekf": rmse(ek.x)},
        "nis_mean": {"pseudo_kf": float(kf.nis[20:].mean()), "ekf": float(ek.nis[20:].mean())},
        "nis_in_95": {"pseudo_kf": float(np.mean(kf.nis[20:] < CHI2_95[2])),
                      "ekf": float(np.mean(ek.nis[20:] < CHI2_95[8]))},
        "nees_mean": {"pseudo_kf": float(nees(kf, X)[20:].mean()), "ekf": float(nees(ek, X)[20:].mean())},
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = [run_once(seed) for seed in range(50)]
    keys = ["wnls_no_bias_correction", "wnls", "pseudo_kf", "ekf"]
    mc = {k: {"mean_mm": 1000 * float(np.mean([r["rmse"][k] for r in runs])),
              "std_mm": 1000 * float(np.std([r["rmse"][k] for r in runs]))} for k in keys}
    cons = {f: {"nis_mean": float(np.mean([r["nis_mean"][f] for r in runs])),
                "nis_dof": 2 if f == "pseudo_kf" else 8,
                "nis_share_below_chi2_95": float(np.mean([r["nis_in_95"][f] for r in runs])),
                "nees_mean": float(np.mean([r["nees_mean"][f] for r in runs]))} for f in ("pseudo_kf", "ekf")}

    # initialisation sensitivity: 1.5 m initial error with an over-confident vs an honest P0
    init = {}
    for name, P0 in [("overconfident_P0", np.diag([0.01**2] * 2 + [0.01**2] * 2)), ("honest_P0", np.diag([1.5**2] * 2 + [0.5**2] * 2))]:
        rs = [run_once(seed, x0_err=(1.5, 1.0), P0=P0) for seed in range(20)]
        init[name] = {k: 1000 * float(np.mean([r["rmse"][k] for r in rs])) for k in ("pseudo_kf", "ekf")}
        init[name]["ekf_first_2s_mm"] = 1000 * float(np.mean(
            [np.sqrt(np.mean(np.sum((r["ekf"].x[:20, :2] - r["X"][:20, :2]) ** 2, axis=1))) for r in rs]))

    ex = runs[0]
    s = ex["setup"]
    calib = {"bias_true_us": (1e6 * s.bias).round(2).tolist(), "bias_hat_us": (1e6 * ex["bias_hat"]).round(2).tolist(),
             "sigma_true_us": (1e6 * s.sigma).round(2).tolist(),
             "sigma_hat_us": (1e6 * np.sqrt(ex["var_hat"])).round(2).tolist(),
             "n_static_pulses": int(len(ex["Y_static"]))}
    out = {"monte_carlo_runs": len(runs), "pulses_per_run": T, "rmse": mc, "consistency": cons,
           "initialisation": init, "calibration_example": calib}
    (root / "results" / "metrics.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(root / "results" / "run_example.npz", X=ex["X"], fixes=ex["fixes"], fix_covs=ex["fix_covs"],
                        raw=ex["raw"], kf_x=ex["kf"].x, kf_P=ex["kf"].P, kf_nis=ex["kf"].nis, ekf_x=ex["ekf"].x,
                        ekf_P=ex["ekf"].P, ekf_nis=ex["ekf"].nis, mics=s.mics,
                        static_resid=ex["Y_static"] - toa(ex["p0"], s.mics)[None, :], bias=s.bias, sigma=s.sigma)
    print(json.dumps({"rmse_mm": {k: round(v["mean_mm"], 1) for k, v in mc.items()}, "consistency": cons,
                      "initialisation": init}, indent=1))


if __name__ == "__main__":
    main()
