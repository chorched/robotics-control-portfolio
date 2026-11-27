"""Figures from results/ (run ``python -m toa_ekf.experiment`` first).

    python figures/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parents[1] / "common"))
sys.path.insert(0, str(ROOT))
import viz_style as V  # noqa: E402
from toa_ekf.core import ellipse  # noqa: E402
from toa_ekf.experiment import run_once  # noqa: E402

V.apply()
M = json.loads((ROOT / "results" / "metrics.json").read_text())
E = np.load(ROOT / "results" / "run_example.npz")


def overview():
    X, mics = E["X"], E["mics"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(X[:, 0], X[:, 1], color=V.TEXT, lw=2.2, label="true path")
    ax.plot(E["raw"][:, 0], E["raw"][:, 1], color=V.ORANGE, lw=1, alpha=0.8, label="WNLS, timing bias ignored")
    ax.plot(E["ekf_x"][:, 0], E["ekf_x"][:, 1], color=V.AQUA, lw=1.6, label="direct-TOA EKF")
    for t in range(5, len(X), 25):
        el = ellipse(E["ekf_x"][t, :2], E["ekf_P"][t, :2, :2] * 400)
        ax.fill(el[:, 0], el[:, 1], color=V.AQUA, alpha=0.18, lw=0)
    ax.fill([], [], color=V.AQUA, alpha=0.3, label="EKF 95% ellipse (×20 scale)")
    ax.scatter(mics[:, 0], mics[:, 1], marker="^", s=90, color=V.VIOLET, zorder=4, label="microphones")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-0.8, 7.0)
    ax.set_ylim(-0.6, 4.9)
    ax.legend(loc="upper center", ncol=3, fontsize=8.3, bbox_to_anchor=(0.5, 1.16))
    ax.set_title("Eight microphones, one moving robot", pad=48)
    V.caption(fig, "Simulated 6 m × 4 m room, 10 Hz pulses for 30 s. Timing biases of up to 0.5 ms mean up to 17 cm "
                   "of range error per microphone if not calibrated.")
    V.save(fig, HERE / "overview.png")


def calibration():
    c = M["calibration_example"]
    n = len(c["bias_true_us"])
    x = np.arange(n)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.5))
    ax.bar(x - 0.19, c["bias_true_us"], width=0.36, color=V.MUTED, label="true", zorder=2)
    se = np.array(c["sigma_hat_us"]) / np.sqrt(c["n_static_pulses"])
    ax.bar(x + 0.19, c["bias_hat_us"], width=0.36, color=V.BLUE, label="estimated", zorder=2,
           yerr=1.96 * se, error_kw=dict(ecolor=V.TEXT_2, capsize=2, lw=1))
    ax.axhline(0, color=V.MUTED, lw=0.8)
    ax.set_xticks(x, [f"mic {i + 1}" for i in x], fontsize=8.5)
    ax.set_ylabel("timing bias (µs)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_title("Bias from 200 static pulses")
    ax2.bar(x - 0.19, c["sigma_true_us"], width=0.36, color=V.MUTED, label="true", zorder=2)
    ax2.bar(x + 0.19, c["sigma_hat_us"], width=0.36, color=V.BLUE, label="estimated", zorder=2)
    ax2.set_xticks(x, [f"mic {i + 1}" for i in x], fontsize=8.5)
    ax2.set_ylabel("noise std (µs)")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Noise level per channel → filter weights")
    V.caption(fig, "Robot parked at a known position; bias = sample mean, variance = sample variance of the "
                   "TOA residuals. Error bars: 95% interval of the mean.")
    fig.tight_layout()
    V.save(fig, HERE / "result_calibration.png")


def accuracy():
    r = M["rmse"]
    names = [("wnls_no_bias_correction", "WNLS,\nno bias correction", V.ORANGE), ("wnls", "WNLS\n(per pulse)", V.BLUE),
             ("pseudo_kf", "pseudo-measurement\nKalman filter", V.VIOLET), ("ekf", "direct-TOA\nEKF", V.AQUA)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1, 1.25]})
    for i, (k, lab, col) in enumerate(names):
        ax.bar(i, r[k]["mean_mm"], yerr=r[k]["std_mm"], color=col, width=0.6, zorder=2,
               error_kw=dict(ecolor=V.TEXT_2, capsize=3))
        ax.text(i, r[k]["mean_mm"] * 1.15, f"{r[k]['mean_mm']:.0f}", ha="center", fontsize=9.5, weight="bold")
    ax.set_yscale("log")
    ax.set_ylim(5, 400)
    ax.set_xticks(range(4), [lab for _, lab, _ in names], fontsize=8.3)
    ax.set_ylabel("position RMSE (mm, log scale)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Accuracy over {M['monte_carlo_runs']} Monte Carlo runs")

    t = np.arange(len(E["ekf_nis"])) * 0.1
    ax2.plot(t, E["ekf_nis"], color=V.AQUA, lw=1, label="EKF NIS (8 TOAs)")
    ax2.axhline(15.507, color=V.ORANGE, ls="--", lw=1.4)
    ax2.text(29.8, 16.2, "χ²₈ 95% bound", ha="right", fontsize=8.5, color=V.TEXT)
    ax2.axhline(8, color=V.MUTED, lw=1, ls=":")
    ax2.text(29.8, 8.5, "expected value 8", ha="right", fontsize=8.5, color=V.TEXT_2)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("normalised innovation squared")
    ax2.set_ylim(0, 30)
    c = M["consistency"]["ekf"]
    ax2.set_title(f"Filter consistency: mean NIS {c['nis_mean']:.1f}, {c['nis_share_below_chi2_95']:.0%} below bound")
    V.caption(fig, "Pseudo-measurement KF: mean NIS 1.5 (dof 2). Both filters have mean NEES ≈ 1.8 (dof 2): "
                   "consistent, slightly conservative because the process noise is set above the true motion noise.")
    fig.tight_layout()
    V.save(fig, HERE / "result_accuracy.png")


def initialisation():
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    for name, P0, col in [("over-confident P0 (1 cm)", np.diag([0.01**2] * 4), V.ORANGE),
                          ("honest P0 (1.5 m)", np.diag([1.5**2] * 2 + [0.5**2] * 2), V.AQUA)]:
        errs = []
        for seed in range(20):
            r = run_once(seed, x0_err=(1.5, 1.0), P0=P0)
            errs.append(np.linalg.norm(r["ekf"].x[:, :2] - r["X"][:, :2], axis=1))
        e = np.median(errs, axis=0)
        ax.plot(np.arange(len(e)) * 0.1, e * 1000, color=col, label=f"EKF, {name}")
    ax.set_yscale("log")
    ax.set_xlim(0, 10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position error (mm, median of 20 runs)")
    ax.legend(loc="upper right", fontsize=8.8)
    ax.set_title("Start 1.8 m off: an over-confident P0 makes the EKF distrust the data at first")
    V.caption(fig, "The debugging lesson from the original project: wrong initial state and covariance made the "
                   "filter look broken; fixing P0, then checking innovations, fixed it.")
    V.save(fig, HERE / "result_initialisation.png")


if __name__ == "__main__":
    overview()
    calibration()
    accuracy()
    initialisation()
    print("figures written to", HERE)
