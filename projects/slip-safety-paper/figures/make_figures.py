"""Figures for the slip-safety project page.

concept.png  - schematic (illustrative curves, not data)
result_eta.png - measured drop rates from results/registered.json (run S4c)
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
import viz_style as V  # noqa: E402

V.apply()
R = json.loads((ROOT / "results" / "registered.json").read_text(encoding="utf-8"))


def concept():
    k = np.arange(0, 60)
    jump, arrive = 18, 40  # load jump inside the chunk; next chunk arrives at step 40
    demand = np.where(k < jump, 1.0, 1.0 + 0.9 * (1 - np.exp(-(k - jump) / 2.0)))
    cap_step = np.full(k.shape, 1.25)
    for i in range(1, len(k)):  # per-step CBF reacts only when the margin is already gone, rate-limited
        need = demand[i - 1] * 1.05
        cap_step[i] = min(cap_step[i - 1] + 0.06, max(cap_step[i - 1], need)) if demand[i - 1] > cap_step[i - 1] * 0.97 else cap_step[i - 1]
    cap_prev = np.full(k.shape, 1.25)
    pre = 8  # preview sees the predicted load for the remaining blind steps and squeezes early
    for i in range(1, len(k)):
        target = 2.05 if i >= jump - pre else 1.25
        cap_prev[i] = min(cap_prev[i - 1] + 0.06, target) if target > cap_prev[i - 1] else cap_prev[i - 1]

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.axvspan(0, arrive, color=V.VIOLET, alpha=0.06, lw=0)
    ax.annotate("", xy=(arrive, 2.33), xytext=(0, 2.33), arrowprops=dict(arrowstyle="<->", color=V.VIOLET, lw=1.2))
    ax.text(arrive / 2, 2.38, "blind window: the policy executes its chunk open loop for m steps",
            ha="center", fontsize=9, color=V.TEXT)
    ax.axvline(arrive, color=V.VIOLET, lw=1.2, ls=":")
    ax.text(arrive + 0.6, 2.12, "next chunk arrives", fontsize=8.5, color=V.TEXT_2)
    ax.plot(k, demand, color=V.TEXT, lw=2.2, label="tangential load the grasp must hold")
    ax.plot(k, cap_step, color=V.ORANGE, lw=1.8, label="friction capacity μ·f_n, per-step barrier (reacts)")
    ax.plot(k, cap_prev, color=V.AQUA, lw=1.8, label="friction capacity μ·f_n, chunk preview (anticipates)")
    slip = demand > cap_step
    ax.fill_between(k, cap_step, demand, where=slip, color=V.ORANGE, alpha=0.25, lw=0)
    i0 = np.argmax(slip)
    ax.text(31.5, 1.45, "← slip: the grip force cannot rise fast enough", fontsize=8.5, color=V.TEXT_2)
    ax.axvline(jump, color=V.MUTED, lw=1, ls="--")
    ax.text(jump - 0.6, 0.62, "load jump\n(e.g. heavier object)", fontsize=8.5, color=V.TEXT_2, ha="right")
    ax.set_xlim(0, 59)
    ax.set_ylim(0.5, 2.5)
    ax.set_yticks([])
    ax.set_xlabel("control step inside the current chunk")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Grip safety during the blind window of an action-chunked policy")
    V.caption(fig, "Schematic, illustrative curves. The preview layer checks the friction cone h = μ f_n − |f_t| "
                   "over the remaining blind steps, with a conformally calibrated error margin.")
    V.save(fig, HERE / "concept.png")


def result_eta():
    d = R["S4c_T1_drop_vs_eta"]
    eta = np.array(d["eta"])
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    series = [("C2", "per-step friction-cone barrier", V.ORANGE, "s"),
              ("C3", "chunk preview, fixed worst-case m", V.BLUE, "D"),
              ("C4", "chunk preview, latency-aware m", V.AQUA, "o")]
    for key, lab, col, mk in series:
        ax.plot(eta, d[key], color=col, marker=mk, label=lab, mec=V.SURFACE, mew=1.2)
    ax.plot(eta, d["C0"], color=V.MUTED, lw=1.2, ls="--", label="no safety layer")
    ax.set_xlabel("static load utilisation η after the load jump  (1 = at the friction limit)")
    ax.set_ylabel("object-drop rate")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center left", fontsize=8.5, bbox_to_anchor=(0.0, 0.55))
    ax.annotate("η = 0.95:  C2 27%  →  C4 3%", xy=(0.95, 0.06), xytext=(0.80, 0.42), fontsize=9,
                arrowprops=dict(arrowstyle="-|>", color=V.TEXT_2, lw=1))
    ax.set_title("Preview pays off when the grasp is close to its friction limit")
    V.caption(fig, "Run S4c, T1 (MuJoCo Panda pick-lift-transport, mass ×2 mid-transport), 720 episodes per point "
                   "(μ, speed, latency pooled). Pre-registered H1c: 197 → 23 drops in the window η 0.85–0.95.")
    V.save(fig, HERE / "result_eta.png")


if __name__ == "__main__":
    concept()
    result_eta()
    print("figures written to", HERE)
