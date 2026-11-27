"""Regenerate the figures of this project from results/claims.json.

    python -m residual_rl.claims && python figures/make_figures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parents[1] / "common"))
import viz_style as V  # noqa: E402

V.apply()
S = json.loads((ROOT / "results" / "claims.json").read_text())
G, C = S["groups"], S["comparisons"]
CONDS = ["C0", "C1", "C2", "C3"]
COND_LABEL = {"C0": "C0 nominal", "C1": "C1 training\ndisturbances", "C2": "C2 heavier\n(out of distr.)",
              "C3": "C3 mass +20%\n(out of distr.)"}


def overview():
    fig = plt.figure(figsize=(11.5, 4.0))
    ax = fig.add_axes([0.0, 0.0, 0.64, 1.0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", facecolor=fill,
                                    edgecolor=color, lw=1.6))
        ax.text(x + w / 2, y + h / 2 + 1.8, title, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center", fontsize=8.3, color=V.TEXT_2)

    def arrow(p, q, label="", off=(0, 1.5)):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.4, color=V.TEXT_2))
        if label:
            ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, ha="center", fontsize=8.8,
                    color=V.TEXT_2)

    box(10, 26, 26, 11, "Impedance controller", r"$J^\top(K e + D\dot e) + g(q)$", V.BLUE, "#eef4fc")
    box(10, 6, 26, 11, "PPO residual policy", r"$a \in [-1,1]^7$, MLP 256×256", V.AQUA, "#e9f7f1")
    ax.add_patch(plt.Circle((47, 21.5), 2.4, facecolor=V.SURFACE, edgecolor=V.TEXT_2, lw=1.4))
    ax.text(47, 21.5, "+", ha="center", va="center", fontsize=13)
    box(58, 14, 26, 15, "Fetch arm (MuJoCo)", "7 joint torques\nfriction, mass error,\nforce pulses", V.ORANGE,
        "#fdf0ea")
    arrow((36, 31.5), (45.3, 23.3), r"$\tau_{imp}$", (0, 2.5))
    arrow((36, 11.5), (45.3, 19.7), r"$0.1\,\tau_{max}\, a$", (1, -4.8))
    arrow((49.4, 21.5), (58, 21.5), r"$\tau$")
    ax.plot([71, 71, 5, 5], [14, 2.5, 2.5, 31.5], color=V.MUTED, lw=1.2)
    ax.annotate("", xy=(10, 31.5), xytext=(5, 31.5), arrowprops=dict(arrowstyle="-|>", color=V.MUTED, lw=1.2))
    ax.annotate("", xy=(10, 11.5), xytext=(5, 11.5), arrowprops=dict(arrowstyle="-|>", color=V.MUTED, lw=1.2))
    ax.text(40, 0.2, "observation: state, reference, tracking error (32-D)", fontsize=8, color=V.MUTED, ha="center")
    ax.text(2, 41, "Residual RL: learn only what the model-based controller misses", fontsize=12.5, weight="bold")

    ax2 = fig.add_axes([0.69, 0.14, 0.29, 0.72])
    t = np.linspace(0, 2 * math.pi, 300)
    ax2.plot(0.1 * (np.cos(t) - 1), 0.1 * np.sin(t), color=V.BLUE, label="T1 circle")
    ax2.plot(0.1 * np.sin(t), 0.05 * np.sin(2 * t), color=V.ORANGE, label="T2 figure-eight")
    ax2.scatter([0], [0], color=V.TEXT, s=18, zorder=3)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.text(-0.19, 0.085, "T1 circle", color=V.TEXT, fontsize=9)
    ax2.text(0.03, 0.062, "T2 figure-eight", color=V.TEXT, fontsize=9)
    ax2.set_title("Reference paths, 8 s each")
    V.caption(fig, "Training uses condition C1; C2 and C3 are disturbance regimes never seen in training.")
    V.save(fig, HERE / "overview.png")


def errors():
    arms = [("joint_pd", "joint PD (tuned)", V.BLUE), ("impedance", "impedance (tuned)", V.ORANGE),
            ("residual@1M", "residual PPO", V.AQUA), ("direct@1M", "plain PPO", V.MUTED)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, traj in zip(axes, ["T1", "T2"]):
        for ci, cond in enumerate(CONDS):
            for ai, (arm, lab, col) in enumerate(arms):
                g = G[arm][cond][traj]["e_rms_ss"]
                y = ci + (ai - 1.5) * 0.17
                ax.errorbar(g["mean"] * 1000, y, xerr=[[(g["mean"] - g["ci"][0]) * 1000], [(g["ci"][1] - g["mean"]) * 1000]],
                            fmt="o", color=col, ms=7, mec=V.SURFACE, mew=1.2, capsize=2.5, lw=1.2,
                            label=lab if ci == 0 else None)
        ax.set_xscale("log")
        ax.set_xlim(3, 160)
        ax.set_xticks([3, 5, 10, 20, 50, 100], ["3", "5", "10", "20", "50", "100"])
        ax.set_yticks(range(4), [COND_LABEL[c] for c in CONDS], fontsize=8.8)
        ax.invert_yaxis()
        ax.set_xlabel("steady-state tracking error (mm, log scale)")
        ax.set_title(f"{traj}: {'circle' if traj == 'T1' else 'figure-eight'}")
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.55, 1.06), fontsize=9)
    V.caption(fig, "RMS error after t ≥ 1 s; mean over 5 training seeds × 50 evaluation episodes "
                   "(baselines: 50 episodes), 95% hierarchical-bootstrap intervals.")
    fig.tight_layout()
    V.save(fig, HERE / "result_errors.png")


def paired():
    rows = [c for c in C if c["metric"] == "e_rms_ss" and (c["a"], c["b"]) == ("residual@1M", "impedance")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1.25, 1]})
    for i, c in enumerate(rows):
        lo, hi = c["ci"]
        face = V.AQUA if c["resolved"] else V.SURFACE
        ax.plot([lo * 1000, hi * 1000], [i, i], color=V.AQUA, lw=2)
        ax.scatter([c["mean_diff"] * 1000], [i], s=60, color=face, edgecolor=V.AQUA, lw=1.8, zorder=3)
        if not c["resolved"]:
            ax.text(c["mean_diff"] * 1000 + 1.4, i, "no decidable difference:\nwinner terminates more (3.2% vs 2.0%)",
                    va="center", fontsize=7.8, color=V.TEXT_2)
    ax.axvline(0, color=V.MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [f"{c['condition']} / {c['trajectory']}" for c in rows])
    ax.invert_yaxis()
    ax.set_xlim(-40, 12)
    ax.set_xlabel("residual PPO − impedance, steady-state error (mm)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Paired difference: 7 of 8 cells claimed")

    x = np.arange(4)
    for k, (arm, lab, col) in enumerate([("impedance", "impedance", V.ORANGE), ("residual@1M", "residual PPO", V.AQUA),
                                         ("direct@1M", "plain PPO", V.MUTED)]):
        vals = [G[arm][c]["T1"]["terminated"]["mean"] for c in CONDS]
        ax2.bar(x + (k - 1) * 0.27, vals, width=0.25, color=col, label=lab, zorder=2)
    ax2.set_xticks(x, CONDS)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("episodes terminated early")
    ax2.legend(loc="upper right", fontsize=8.3)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Plain PPO is not safe to run (T1)")
    V.caption(fig, "Common random numbers: both arms see the same initial state and disturbance draw. A claim needs "
                   "a 95% interval excluding 0, a winner that does not terminate more often, and all 5 training seeds.")
    fig.tight_layout()
    V.save(fig, HERE / "result_paired.png")


if __name__ == "__main__":
    overview()
    errors()
    paired()
    print("figures written to", HERE)
