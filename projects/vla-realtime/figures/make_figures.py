"""Regenerate every figure of this project from results/*.json.

    python figures/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parents[1] / "common"))
import viz_style as V  # noqa: E402

V.apply()
R = json.loads((ROOT / "results" / "libero_results.json").read_text(encoding="utf-8"))
TOY = json.loads((ROOT / "results" / "toy_results.json").read_text(encoding="utf-8"))


def err_bar(ax, x, k, n, color, width=0.6, horizontal=False):
    p = k / n
    lo, hi = V.wilson95(k, n)
    if horizontal:
        ax.barh(x, p, height=width, color=color, zorder=2)
        ax.errorbar(p, x, xerr=[[p - lo], [hi - p]], fmt="none", ecolor=V.TEXT_2, elinewidth=1.2, capsize=3, zorder=3)
    else:
        ax.bar(x, p, width=width, color=color, zorder=2)
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=V.TEXT_2, elinewidth=1.2, capsize=3, zorder=3)
    return hi


# ------------------------------------------------------------------ 1. timeline


def timeline():
    E, NOM, H = 8, 17, 50
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.set_xlim(-1.2, 30)
    ax.set_ylim(-0.6, 4.6)
    ax.axis("off")

    def step_ticks(y):
        for t in range(0, 30):
            ax.plot([t, t], [y - 0.08, y + 0.08], color=V.MUTED, lw=0.8)
        ax.plot([0, 29.5], [y, y], color=V.MUTED, lw=0.8)

    # request in flight
    y0 = 4.0
    ax.text(-1.2, y0 + 0.45, "Policy query", fontsize=10, weight="bold", color=V.TEXT)
    ax.add_patch(FancyBboxPatch((0, y0 - 0.18), E, 0.36, boxstyle="round,pad=0,rounding_size=0.12",
                                color=V.VIOLET, alpha=0.9, lw=0))
    ax.text(E / 2, y0, f"in flight: E = {E} control steps", color="white", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(E, y0 - 0.45), xytext=(E, y0 - 0.2), arrowprops=dict(arrowstyle="-|>", color=V.TEXT_2))
    ax.text(0, y0 - 0.45, "obs captured at $t_0$", fontsize=8.5, color=V.TEXT_2, va="top")
    ax.text(E + 0.2, y0 - 0.45, "chunk arrives", fontsize=8.5, color=V.TEXT_2, va="top")

    def row(y, drop, label, color, verdict):
        ax.text(-1.2, y + 0.55, label, fontsize=10, weight="bold", color=V.TEXT)
        step_ticks(y - 0.42)
        # applied from step E onwards: action index drop + (t - E)
        for t in range(E, 30):
            k = drop + (t - E)
            if k >= H:
                break
            ok = k == t
            ax.add_patch(Rectangle((t + 0.06, y - 0.25), 0.88, 0.5, color=color if ok else V.ORANGE,
                                   alpha=1.0 if ok else 0.95, lw=0))
            if t < E + 6 or t % 4 == 0:
                ax.text(t + 0.5, y, f"$a_{{{k}}}$", ha="center", va="center", color="white", fontsize=7.5)
        for t in range(0, E):
            ax.add_patch(Rectangle((t + 0.06, y - 0.25), 0.88, 0.5, color=V.GRID, lw=0))
        ax.text(29.6, y, verdict, ha="right", va="center", fontsize=8.8, color=V.TEXT_2)
        ax.text(E / 2, y, "previous chunk", ha="center", va="center", fontsize=8, color=V.TEXT_2)

    row(2.45, E, "Elapsed rule   d = min(⌈L/dt⌉, E) = 8", V.AQUA, "")
    row(0.75, NOM, "Nominal rule   d = ⌈L/dt⌉ = 17   (loop 2.1× slower than wall clock)", V.ORANGE, "")
    ax.text(E + 0.1, 2.45 - 0.75, "step $t$ plays $a_t$: on schedule", fontsize=8.5, color=V.TEXT_2)
    ax.text(E + 0.1, 0.75 - 0.75, "step $t$ plays $a_{t+9}$: 9 still-future actions thrown away per query",
            fontsize=8.5, color=V.TEXT_2)
    ax.text(29.5, -0.55, "control step $t$  (50 ms each)", ha="right", fontsize=8.5, color=V.MUTED)
    fig.suptitle("Which part of an action chunk is already in the past?", x=0.06, ha="left",
                 fontsize=13, weight="bold", y=1.0)
    V.caption(fig, "Schematic of the async executor. L = wall-clock query latency, dt = control period, "
                   "E = control steps that actually elapsed. Numbers match run C4t (L ≈ 848 ms, E ≈ 8).")
    V.save(fig, HERE / "timeline.png")


# ----------------------------------------------------------------- 2. drop rule


def drop_rule():
    c = R["drop_rule_C4t"]["cells"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = [V.MUTED, V.ORANGE, V.AQUA]
    for i, cell in enumerate(c):
        hi = err_bar(ax, 2 - i, cell["k"], cell["n"], colors[i], width=0.55, horizontal=True)
        ax.text(hi + 0.02, 2 - i, f"{cell['k']}/{cell['n']}", va="center", fontsize=10, color=V.TEXT, weight="bold")
    ax.set_yticks([2, 1, 0], [x["label"] for x in c])
    ax.set_xlim(0, 1)
    ax.set_xlabel("success rate (Wilson 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Same load, same initial states: only the drop rule changes")

    vals = [c[1]["drop_minus_elapsed"], c[2]["drop_minus_elapsed"]]
    ax2.bar([0, 1], vals, color=[V.ORANGE, V.AQUA], width=0.55, zorder=2)
    for x, v in zip([0, 1], vals):
        ax2.text(x, v + 0.25, f"{v:.1f}", ha="center", fontsize=10, weight="bold")
    ax2.set_xticks([0, 1], ["nominal", "elapsed"])
    ax2.set_ylim(0, 11.5)
    ax2.set_ylabel("actions per query")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Future actions discarded")
    d = R["drop_rule_C4t"]["H4"]
    V.caption(fig, f"Run C4t, SmolVLA on LIBERO-Object tasks 0–2, 24 episodes per cell. Pre-registered H4: "
                   f"difference {d['diff']:+.3f}, Newcombe 95% CI [{d['newcombe95'][0]:.2f}, {d['newcombe95'][1]:.2f}].")
    fig.tight_layout()
    V.save(fig, HERE / "result_drop_rule.png")


# ----------------------------------------------------------------- 3. mechanism


def mechanism():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    d = R["dose_response_C5t"]
    x = np.array(d["inject_ms"])
    for key, color, name, off in [("sync_k", V.BLUE, "sync, replan every 8", -4), ("async_elapsed_k", V.AQUA, "async, elapsed drop", 4)]:
        k = np.array(d[key])
        p = k / d["n"]
        ci = np.array([V.wilson95(int(v), d["n"]) for v in k])
        ax.errorbar(x + off, p, yerr=[p - ci[:, 0], ci[:, 1] - p], color=color, marker="o", capsize=3, lw=2,
                    markeredgecolor=V.SURFACE, markeredgewidth=1.5, label=name)
        ax.text(x[0] + off - 8, p[0], f"{k[0]}/24", ha="right", va="center", fontsize=9, color=V.TEXT_2)
    ax.set_xticks(x, [f"+{v} ms" for v in x])
    ax.set_xlim(-40, 330)
    ax.set_ylim(-0.03, 1)
    ax.set_ylabel("success rate")
    ax.set_xlabel("injected latency")
    ax.legend(loc="upper right")
    ax.set_title("Latency dose–response (C5t)")

    c = R["chunk_budget_C7t"]["cells"]
    for i, cell in enumerate(c):
        hi = err_bar(ax2, i, cell["k"], cell["n"], [V.ORANGE, V.AQUA][i], width=0.5)
        ax2.text(i, hi + 0.03, f"{cell['k']}/24\n{cell['held_per_query']:.2f} held steps / query",
                 ha="center", fontsize=9, color=V.TEXT_2)
    ax2.set_xticks([0, 1], [f"$L_c$ = {c[0]['Lc']}  (< E)", f"$L_c$ = {c[1]['Lc']}  (> E)"])
    ax2.set_ylim(0, 1)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("At +300 ms, E ≈ 11.7 steps per query (C7t)")
    h = R["chunk_budget_C7t"]["H12"]
    V.caption(fig, f"Async needs each chunk to cover the steps that pass while the next one is computed (L_c > E). "
                   f"Pre-registered H12: {h['diff']:+.3f}, Newcombe 95% CI [{h['newcombe95'][0]:.2f}, {h['newcombe95'][1]:.2f}].")
    fig.tight_layout()
    V.save(fig, HERE / "result_mechanism.png")


# ------------------------------------------------------------- 4. denoising steps


def denoise():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    s = R["denoise_sweep_V19C3"]["sync_replan8"]
    p = np.array(s["k"]) / 24
    jitter = {1: -0.18, 2: 0.18}
    for ns, e, pk, k in zip(s["num_steps"], s["E_median"], p, s["k"]):
        ax.scatter(e + jitter.get(ns, 0), pk, s=70, color=V.BLUE, edgecolor=V.SURFACE, linewidth=1.5, zorder=3)
        ax.text(e + jitter.get(ns, 0), pk + 0.035, f"{ns} step{'s' if ns > 1 else ''}", ha="center", fontsize=8.5,
                color=V.TEXT_2)
    ax.plot([e + jitter.get(ns, 0) for ns, e in zip(s["num_steps"], s["E_median"])], p, color=V.BLUE, lw=1,
            alpha=0.4, zorder=2)
    ax.set_xlabel("E: control steps elapsed per query (median)")
    ax.set_ylabel("success rate (24 episodes)")
    ax.set_ylim(0.5, 1.07)
    ax.set_title("Fewer denoising steps → smaller E → higher success")
    ax.text(0.98, 0.04, f"Spearman ρ = {s['spearman_success_vs_E']:.2f}", transform=ax.transAxes, ha="right",
            fontsize=9, color=V.TEXT_2)

    h = R["heldout_V19C4"]
    for i, (k, lab) in enumerate([(h["steps3_k"], "3 steps"), (h["steps10_k"], "10 steps (default)")]):
        hi = err_bar(ax2, i, k, h["pairs"], [V.AQUA, V.BLUE][i], width=0.5)
        ax2.text(i, hi + 0.03, f"{k}/56", ha="center", fontsize=10, weight="bold")
    ax2.set_xticks([0, 1], ["3 denoising steps", "10 (default)"])
    ax2.set_ylim(0, 1.12)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Confirmation on held-out tasks 3–9")
    V.caption(fig, f"Left: V19C3 + V19C1b, sync replan-every-8, tasks 0–2. Right: V19C4, 56 paired episodes; "
                   f"Newcombe 95% CI of the difference [{h['newcombe95'][0]:.2f}, {h['newcombe95'][1]:.2f}], "
                   f"exact McNemar p = {h['mcnemar']['p_exact']:.1e}; median latency {h['latency_ms_median']['steps3']} vs "
                   f"{h['latency_ms_median']['steps10']} ms.")
    fig.tight_layout()
    V.save(fig, HERE / "result_denoise.png")


# ------------------------------------------------------------------- 5. toy grid


def toy_grid():
    g = TOY["chunk_budget"]
    S = np.array(g["success"]) / g["n"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", V.SEQ_BLUE)
    im = ax.imshow(S, cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
    ax.set_xticks(range(len(g["Lc"])), g["Lc"])
    ax.set_yticks(range(len(g["E"])), g["E"])
    ax.set_xlabel("$L_c$: fresh actions kept per chunk")
    ax.set_ylabel("E: control steps per query")
    ax.grid(False)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            ax.text(j, i, f"{S[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if S[i, j] > 0.55 else V.TEXT)
    # boundary L_c = E (only where both grids overlap)
    pts = [(g["Lc"].index(e), i) for i, e in enumerate(g["E"]) if e in g["Lc"]]
    ax.plot([x for x, _ in pts], [y for _, y in pts], color=V.ORANGE, lw=2, ls="--")
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.outline.set_visible(False)
    cb.set_label("success rate")
    ax.set_title("Toy reproduction: async starves below $L_c = E$")
    V.caption(fig, f"examples/toy_tracking.py, {g['n']} episodes per cell, elapsed drop rule. "
                   "Dashed line: L_c = E. Long chunks also lose ground: more open-loop time before the next correction.")
    V.save(fig, HERE / "toy_chunk_budget.png")


if __name__ == "__main__":
    timeline()
    drop_rule()
    mechanism()
    denoise()
    toy_grid()
    print("figures written to", HERE)
