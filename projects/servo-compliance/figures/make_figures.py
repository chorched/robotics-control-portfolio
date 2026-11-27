"""Regenerate the figures of this project from results/.

    python figures/make_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parents[1] / "common"))
sys.path.insert(0, str(ROOT))
import viz_style as V  # noqa: E402
from servo_compliance.admittance import g_max, spectral_radius  # noqa: E402

V.apply()
REG = json.loads((ROOT / "results" / "registered.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------- block diagram


def block_diagram():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 42)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, title, sub="", color=V.BLUE, fill=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    facecolor=fill or "#eef4fc", edgecolor=color, lw=1.6))
        ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), title, ha="center", va="center", fontsize=10,
                weight="bold", color=V.TEXT)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center", fontsize=8.3, color=V.TEXT_2)

    def arrow(p, q, label="", lpos=None, color=V.TEXT_2, style="-|>", rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, lw=1.4, color=color,
                                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            x, y = lpos or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 1.6)
            ax.text(x, y, label, ha="center", va="bottom", fontsize=8.5, color=V.TEXT_2)

    # forward path (top row)
    ax.add_patch(plt.Circle((7, 30), 2.2, facecolor=V.SURFACE, edgecolor=V.TEXT_2, lw=1.4))
    ax.text(7, 30, "Σ", ha="center", va="center", fontsize=11)
    ax.text(1.0, 31.6, "$f_d$", fontsize=10)
    arrow((0.5, 30), (4.8, 30))
    box(12, 25, 19, 10, "Admittance", r"$\dot p_r = S_\delta(\hat f - f_d)/D_a$", V.VIOLET, "#efedfb")
    arrow((9.2, 30), (12, 30), "$e$", (10.6, 30.6))
    box(36, 25, 14, 10, "IK", "damped LS", V.TEXT_2, "#f2f1ee")
    arrow((31, 30), (36, 30), "$p_r$")
    box(55, 23, 22, 14, "STS3215 servo", "position loop, $k_p$ = 14.1\n(fitted to real data)", V.BLUE)
    arrow((50, 30), (55, 30), "$q_{cmd}$")
    box(83, 23, 22, 14, "Arm + contact", "$k_{arm}$ ≈ 680 N/m in series\nwith plate $k_e$", V.AQUA, "#e9f7f1")
    arrow((77, 30), (83, 30), "$q$")
    arrow((105, 30), (109.5, 30), "$f$", (107.3, 30.6))

    # feedback path (bottom row)
    box(70, 3, 26, 11, "Present_Load", "PWM duty, quantised\n±1 count noise", V.ORANGE, "#fdf0ea")
    box(38, 3, 26, 11, "Force estimator", r"$\hat f = -(JJ^\top)^{-1}J(\tau - g(q) - \tau_{fric})$", V.ORANGE, "#fdf0ea")
    box(14, 3, 18, 11, "Delay", "$z^{-d}$, d = 0, 1, 2", V.ORANGE, "#fdf0ea")
    arrow((94, 23), (88, 14), r"$\tau_{act}$", (95.5, 17.5))
    arrow((70, 8.5), (64, 8.5))
    arrow((38, 8.5), (32, 8.5))
    arrow((14, 8.5), (7, 8.5), style="-")
    arrow((7, 8.5), (7, 27.8), r"$-\hat f$", (4.2, 17))

    ax.text(0.5, 40.5, "Force control on a position-only servo arm", fontsize=13, weight="bold")
    ax.text(57.5, 18.2, "no force sensor: the force is inferred from what the servo reports", fontsize=8.5,
            color=V.TEXT_2, ha="center", style="italic")
    V.caption(fig, "One 50 Hz control period. Stability depends only on g = k_eff·dt/D_a and the delay d "
                   "(see result_stability.png).")
    V.save(fig, HERE / "block_diagram.png")


# ------------------------------------------------------------------- stability


def stability():
    s = REG["stability_Bc2t"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.6), sharex="col",
                             gridspec_kw={"height_ratios": [1.25, 1]})
    for col, (d, key, xmax) in enumerate([(1, "d1", 1.45), (0, "d0", 2.85)]):
        ax, ax2 = axes[0, col], axes[1, col]
        gs = np.linspace(0.02, xmax, 400)
        rho = [spectral_radius(g, d) for g in gs]
        ax.plot(gs, rho, color=V.BLUE)
        ax.axhline(1, color=V.MUTED, lw=1, ls=":")
        ax.axvspan(g_max(d), xmax, color=V.ORANGE, alpha=0.08, lw=0)
        ax.axvline(g_max(d), color=V.ORANGE, lw=1.6, ls="--")
        ax.text(g_max(d) + 0.02 * xmax, 1.42, f"$g_{{max}}$ = {g_max(d):.0f}", color=V.TEXT, fontsize=9.5)
        ax.set_ylim(0, 1.6)
        ax.set_ylabel("spectral radius")
        ax.set_title(f"d = {d} sample{'s' if d != 1 else ''} of sensing delay")

        g, st = np.array(s[key]["g"]), np.array(s[key]["stable"])
        ax2.bar(g, st / s["seeds"], width=0.06 * xmax / 1.45, color=[V.AQUA if x < g_max(d) else V.ORANGE for x in g],
                zorder=2)
        for x, k in zip(g, st):
            ax2.text(x, k / s["seeds"] + 0.04, f"{k}/5", ha="center", fontsize=8.5, color=V.TEXT_2)
        ax2.axvline(g_max(d), color=V.ORANGE, lw=1.6, ls="--")
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel("stable runs (MuJoCo)")
        ax2.set_xlabel("$g = k_{eff}\\,\\Delta t / D_a$")
        ax2.grid(axis="x", visible=False)
    fig.suptitle("Predicted vs observed stability boundary", x=0.06, ha="left", fontsize=13, weight="bold")
    V.caption(fig, "Top: roots of z^(d+1) − z^d + g; the loop is stable iff 0 < g < 2 sin(π / (2(2d+1))). "
                   "Bottom: calibrated SO-101 pressing a 300 N/m plate (k_eff = 246.6 N/m), 5 seeds per point, run Bc2t.")
    fig.tight_layout()
    V.save(fig, HERE / "result_stability.png")


def traces():
    rows = list(csv.DictReader(open(ROOT / "results" / "press_traces.csv")))
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    for tag, color, label in [("stab_d1_g0.5", V.AQUA, "g = 0.5 (stable)"),
                              ("stab_d1_g0.9", V.BLUE, "g = 0.9 (near the boundary)"),
                              ("stab_d1_g1.2", V.ORANGE, "g = 1.2 (past g_max = 1)")]:
        r = [x for x in rows if x["tag"] == tag]
        t = np.array([float(x["t_s"]) for x in r])
        f = np.array([float(x["f_true_N"]) for x in r])
        ax.plot(t, f, color=color, lw=1.8, label=label)
    ax.axhline(2.0, color=V.MUTED, lw=1, ls=":")
    ax.text(8.6, 2.12, "$f_d$ = 2 N", va="bottom", ha="right", fontsize=9, color=V.TEXT_2)
    ax.set_xlim(0, 8.6)
    ax.set_xlabel("time after first contact (s)")
    ax.set_ylabel("true contact force (N)")
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0, 3.4)
    ax.set_title("Above the boundary the reference limit-cycles between its travel limits (~2 Hz), d = 1")
    V.caption(fig, "Run Bc2t, E1 estimator (gravity + Coulomb friction compensated), k_e = 300 N/m.")
    V.save(fig, HERE / "result_traces.png")


# ----------------------------------------------------------------- calibration


def calibration():
    c = REG["calibration_Bc1bt"]["nrmse_arm"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
    splits = ["train", "validation", "test"]
    models = [("publicOld", "public, old calib", V.MUTED), ("publicNew", "public, new calib", V.BLUE),
              ("fitted", "fitted (this work)", V.AQUA)]
    w = 0.26
    for i, (key, lab, col) in enumerate(models):
        vals = [c[s][key] for s in splits]
        ax.bar(np.arange(3) + (i - 1) * w, vals, width=w - 0.02, color=col, label=lab, zorder=2)
        if key != "publicOld":
            ax.text(2 + (i - 1) * w, vals[2] + 0.006, f"{vals[2]:.3f}", ha="center", fontsize=8.5, weight="bold")
    ax.set_xticks(range(3), splits)
    ax.set_ylabel("arm NRMSE (lower is better)")
    ax.legend(loc="upper left", fontsize=8.3)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 0.33)
    ax.set_title("Open-loop replay error")

    data = np.genfromtxt(ROOT / "results" / "calibration_trace.csv", delimiter=",", skip_header=1)
    t = data[:, 0]
    ax2.plot(t, data[:, 2], color=V.MUTED, lw=1.2, ls="--", label="commanded")
    ax2.plot(t, data[:, 1], color=V.TEXT, lw=2.2, label="measured (real robot)")
    ax2.plot(t, data[:, 4], color=V.BLUE, lw=1.6, label="sim, public params")
    ax2.plot(t, data[:, 3], color=V.AQUA, lw=1.6, label="sim, fitted params")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("elbow_flex (rad)")
    ax2.legend(loc="best", fontsize=8.3)
    ax2.set_title("Held-out test trajectory (pick-and-place)")
    V.caption(fig, "SO-101, public NeuralActuator 0 g trajectories (16/2/2 files). Fitted kp = 14.1, kv = 0.47, "
                   "frictionloss = 0.28 vs public kp = 998. Run Bc1bt; reproduced by servo_compliance/calibration.py.\n"
                   "Right: elbow_flex, NRMSE 0.03 fitted vs 0.18 public on this trajectory; on shoulder_lift the "
                   "public model is better here (0.07 vs 0.16) - the gain is in the joint average.")
    fig.tight_layout()
    V.save(fig, HERE / "result_calibration.png")


# -------------------------------------------------------------- load detection


def load_detection():
    L, R = REG["load_detection_B6t"], REG["recalibration_B7t"]
    names = ["OMX\ngo-up→pick", "OMX\npick→go-up", "SO-101\ngo-up→pick", "SO-101\npick→go-up"]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    ax = axes[0]
    ax.bar(x - 0.19, L["auroc_b1"], width=0.36, color=V.MUTED, label="physics residual", zorder=2)
    ax.bar(x + 0.19, L["auroc_b4"], width=0.36, color=V.BLUE, label="1D-CNN (this work)", zorder=2)
    ax.scatter([1 + 0.19], [0.69], marker="D", s=40, color=V.ORANGE, zorder=3, label="NeuralActuator force head")
    ax.axhline(0.5, color=V.MUTED, lw=1, ls=":")
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("1. Ranking transfers")
    ax.set_ylabel("AUROC on the unseen task")
    ax.legend(loc="upper left", fontsize=7.8, ncol=1)

    ax = axes[1]
    ax.bar(x, L["frr_source_threshold_b4"], width=0.5, color=V.ORANGE, zorder=2)
    for xi, v in zip(x, L["frr_source_threshold_b4"]):
        ax.text(xi, v + 0.03, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("2. Source threshold misfires")
    ax.set_ylabel("false rejection on the new task")

    ax = axes[2]
    ax.bar(x, R["best_pJoint_k_le_4"], width=0.5, color=[V.AQUA if v >= 0.8 else V.MUTED for v in R["best_pJoint_k_le_4"]],
           zorder=2)
    for xi, v in zip(x, R["best_pJoint_k_le_4"]):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0.8, color=V.TEXT_2, lw=1, ls="--")
    ax.text(3.45, 0.82, "gate 0.8", fontsize=8, color=V.TEXT_2, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_title("3. Re-thresholding with ≤ 4 labels/class")
    ax.set_ylabel("P(FRR ≤ 10% and FA ≤ 10%)")
    for a in axes:
        a.set_xticks(x, names, fontsize=8)
        a.grid(axis="x", visible=False)
    V.caption(fig, "Leave-task-out on the public NAD dataset, 40–50 target trajectories per direction (runs B6t, B7t). "
                   "Pre-registered gate (≥ 3/4 directions reach 0.8) FAILED with 2/4 - kept as a negative result.")
    fig.tight_layout()
    V.save(fig, HERE / "result_load_detection.png")


if __name__ == "__main__":
    block_diagram()
    stability()
    traces()
    calibration()
    load_detection()
    print("figures written to", HERE)
