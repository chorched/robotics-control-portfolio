"""Regenerate the figures of this project from data/ and results/.

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
from paired_cs import BettingCS  # noqa: E402

V.apply()
REG = json.loads((ROOT / "results" / "registered.json").read_text(encoding="utf-8"))


def budget():
    """Hero: rollouts needed to pick a near-best checkpoint out of 78."""
    p = REG["pool_A4p"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    rates = np.array(p["poolRates"])
    order = np.argsort(rates)[::-1]
    best = rates.max()
    near = rates[order] >= best - 0.05
    ax.bar(np.arange(78), rates[order], width=0.8, color=np.where(near, V.AQUA, V.SEQ_BLUE[2]), zorder=2)
    ax.axhline(best - 0.05, color=V.TEXT_2, lw=1, ls="--")
    ax.text(77, best - 0.045, "within 5 points of the best", ha="right", va="bottom", fontsize=8.5, color=V.TEXT_2)
    ax.set_xlabel("78 candidate checkpoints, sorted")
    ax.set_ylabel("success rate (3,000 initial states)")
    ax.set_xticks([])
    ax.grid(axis="x", visible=False)
    ax.set_title(f"The pool: {int(near.sum())} candidates are good enough")

    rng = np.random.default_rng(0)
    for i, (key, lab, col) in enumerate([("S2_marg_rollouts", "per-candidate\nconfidence sequences", V.BLUE),
                                         ("S2_pair_rollouts", "paired\nconfidence sequences", V.AQUA)]):
        v = np.array(p[key]) / 1000
        ax2.scatter(i + rng.uniform(-0.14, 0.14, len(v)), v, s=28, color=col, edgecolor=V.SURFACE, lw=1, zorder=3)
        ax2.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=V.TEXT, lw=2.2, zorder=4)
        ax2.text(i + 0.29, np.median(v), f"median {np.median(v):.0f}k", va="center", fontsize=8.8)
    ax2.axhline(p["S0"] / 1000, color=V.ORANGE, lw=1.6, ls="--")
    ax2.text(-0.45, p["S0"] / 1000 + 6, f"exhaustive fixed-sample: {p['S0'] / 1000:.0f}k", fontsize=8.8, color=V.TEXT)
    ax2.axhline(p["cap"] / 1000, color=V.MUTED, lw=1, ls=":")
    ax2.text(1.45, p["cap"] / 1000 + 6, "cap 200k", fontsize=8, color=V.TEXT_2, ha="right")
    ax2.set_xticks([0, 1], ["per-candidate\nconfidence sequences", "paired\nconfidence sequences"])
    ax2.set_xlim(-0.5, 1.6)
    ax2.set_ylim(0, 400)
    ax2.set_ylabel("rollouts until a certified pick (k)")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Rollouts spent, 20 replications each")
    V.caption(fig, f"AdroitHandPen, BC and flow-matching checkpoints (run A4p). Paired: 20/20 stop, median 32.7% of "
                   f"the exhaustive budget. Per-candidate: {p['S2_marg_notStopped']}/20 hit the 200k cap without stopping.")
    fig.tight_layout()
    V.save(fig, HERE / "result_budget.png")


def why_pairing():
    """Width of the confidence interval on p_best - p_rival: paired vs two marginals."""
    with np.load(ROOT / "data" / "selection_pool.npz") as z:
        Y = z["Y"].astype(float)
    rates = Y.mean(1)
    a, b = np.argsort(rates)[::-1][:2]
    n_obs = Y.shape[1]
    perm = np.random.default_rng(3).permutation(n_obs)
    marg = BettingCS(2, 0.05)
    pair = BettingCS(1, 0.05)
    ns, w_m, w_p = [], [], []
    marks = set(np.unique(np.geomspace(20, n_obs, 40).astype(int)))
    for i, s in enumerate(perm, start=1):
        marg.update(np.array([Y[a, s], Y[b, s]]))
        pair.update(np.array([(Y[a, s] - Y[b, s] + 1) / 2]))
        if i in marks:
            Lm, Um = marg.compute_interval()
            Lp, Up = pair.compute_interval()
            ns.append(i)
            w_m.append((Um[0] - Lm[0]) + (Um[1] - Lm[1]))
            w_p.append(2 * (Up[0] - Lp[0]))
    corr = np.corrcoef(Y[a], Y[b])[0, 1]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(ns, w_m, color=V.BLUE, label="two independent intervals")
    ax.plot(ns, w_p, color=V.AQUA, label="one paired interval on the difference")
    ax.set_xscale("log")
    ax.set_xlabel("initial states evaluated (both candidates on each)")
    ax.set_ylabel("width of 95% interval on $p_a - p_b$")
    ax.legend(loc="upper right")
    ax.set_title("Why pairing helps: shared initial states cancel out")
    V.caption(fig, f"Two best candidates of the pool; outcome correlation across initial states r = {corr:.2f}. "
                   "Some initial states are easy for every policy, some are hard for every policy.")
    V.save(fig, HERE / "why_pairing.png")


def checkpoint_selection():
    cells = REG["checkpoint_selection_R84"]["cells"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.2, 1]})
    groups = [("mse_mask_N12", "BC, 12 demos"), ("mse_mask_N20", "BC, 20 demos"),
              ("fm_mask_N12", "flow matching, 12"), ("fm_mask_N20", "flow matching, 20")]
    for gi, (g, lab) in enumerate(groups):
        for s in ("s17", "s29", "s43"):
            c = cells[f"{g}_{s}"]
            y0, y1 = c["offline_selected"]["rate"], c["task_selected"]["rate"]
            ax.plot([gi - 0.15, gi + 0.15], [y0, y1], color=V.MUTED, lw=1, zorder=2)
            ax.scatter([gi - 0.15], [y0], color=V.ORANGE, s=34, zorder=3, edgecolor=V.SURFACE, lw=1)
            ax.scatter([gi + 0.15], [y1], color=V.AQUA, s=34, zorder=3, edgecolor=V.SURFACE, lw=1)
    ax.scatter([], [], color=V.ORANGE, label="picked by offline action loss")
    ax.scatter([], [], color=V.AQUA, label="picked by task success")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xticks(range(4), [lab for _, lab in groups], fontsize=8.8)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("success on 300 held-out initial states")
    ax.grid(axis="x", visible=False)
    ax.set_title("Offline loss picks the wrong BC checkpoint")

    r = REG["replication_V19A3"]
    seeds = r["mse_mask_N12"]["per_seed_pp"]
    v = np.array(list(seeds.values()))
    ax2.bar(range(len(v)), v, color=V.AQUA, width=0.6, zorder=2)
    ax2.axhline(r["mse_mask_N12"]["mean_pp"], color=V.TEXT, lw=1.6)
    lo, hi = r["mse_mask_N12"]["t_df6_95"]
    ax2.axhspan(lo, hi, color=V.AQUA, alpha=0.15, lw=0)
    ax2.text(6.4, r["mse_mask_N12"]["mean_pp"] + 1.2, f"mean +{r['mse_mask_N12']['mean_pp']:.1f} pt", ha="right", fontsize=9)
    ax2.set_xticks(range(len(v)), [f"seed\n{s}" for s in seeds], fontsize=8)
    ax2.set_ylim(0, 40)
    ax2.set_ylabel("gain of task-based selection (points)")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Replicated on 7 new training seeds (BC, 12)")
    V.caption(fig, "Left: run R84, 3 training seeds per cell. Right: run V19A3; band = df 6 t-interval. "
                   "BC validation loss bottoms out at step 500 while task success peaks at 2,500–10,000 steps.")
    fig.tight_layout()
    V.save(fig, HERE / "result_checkpoint_selection.png")


def transfer():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [0.8, 1.2]})
    ra = REG["roboarena_A7t"]
    vals = [ra["S2_marg_median"], ra["S2_pair_median"]]
    ax.bar([0, 1], vals, color=[V.BLUE, V.AQUA], width=0.55, zorder=2)
    for x, val in zip([0, 1], vals):
        ax.text(x, val + 300, f"{val:,.0f}", ha="center", fontsize=10, weight="bold")
    ax.set_xticks([0, 1], ["per-candidate", "paired"])
    ax.set_ylabel("episodes to a 5-point pick (median)")
    ax.set_ylim(0, 20000)
    ax.grid(axis="x", visible=False)
    ax.set_title("Real-robot logs (RoboArena): ~70% fewer")

    pr = REG["proxies_A6t"]
    names = list(pr)
    x = np.arange(len(names))
    ax2.bar(x - 0.19, [pr[n]["tauRef"] for n in names], width=0.36, color=V.BLUE, label="all 78 candidates", zorder=2)
    ax2.bar(x + 0.19, [pr[n]["tauRefTop20"] for n in names], width=0.36, color=V.ORANGE, label="top 20 only", zorder=2)
    ax2.axhline(0, color=V.MUTED, lw=1)
    ax2.set_xticks(x, [n.replace(", ", ",\n") for n in names], fontsize=8.5)
    ax2.set_ylim(-0.35, 1.0)
    ax2.set_ylabel("Kendall τ with true success")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Cheap proxies sort out the tail, not the best")
    V.caption(fig, "Left: re-sampling of 599 public RoboArena sessions, 7 policies, δ = 0.05 (run A7t) - not new robot trials. "
                   "Right: run A6t; 'friction × 0.5' rollouts agree with nominal ones 92% of the time.")
    fig.tight_layout()
    V.save(fig, HERE / "result_transfer.png")


if __name__ == "__main__":
    budget()
    why_pairing()
    checkpoint_selection()
    transfer()
    print("figures written to", HERE)
