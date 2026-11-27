"""Does a payload detector's decision threshold transfer to a new task?

Input: ``results/load_scores.json`` - one score per target-task trajectory for each
leave-task-out direction (train and calibrate on task A, score every trajectory of
task B), plus the threshold set on task A (5% false-rejection quantile).

Three questions, numpy only:

1. Ranking: presence AUROC on the target task.
2. Transfer: false rejection (FRR) and false acceptance (FA) of the source threshold.
3. Recalibration: label k payload and k payload-free target trajectories, set a new
   threshold (R1 = min labelled positive, R2 = midpoint between max labelled negative
   and min labelled positive), and measure FRR / FA on the remaining trajectories.
   ``pJoint`` = share of 2,000 random labellings with FRR <= 0.10 and FA <= 0.10.
   Pre-registered gate: >= 3 of 4 directions reach pJoint >= 0.8 with some k <= 4.

    python -m servo_compliance.threshold_transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KS = (1, 2, 4, 8)
TAU = 0.10


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie)."""
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def recalibration(pos, neg, k: int, rule: str, rng, n_draws: int = 2000):
    p_idx = np.argsort(rng.random((n_draws, len(pos))), axis=1)[:, :k]
    n_idx = np.argsort(rng.random((n_draws, len(neg))), axis=1)[:, :k]
    return p_idx, n_idx


def evaluate(pos, neg, thr, p_idx, n_idx, k):
    ar = np.arange(len(thr))[:, None]
    below = pos[None, :] < thr[:, None]
    above = neg[None, :] >= thr[:, None]
    frr = (below.sum(1) - below[ar, p_idx].sum(1)) / (len(pos) - k)
    fa = (above.sum(1) - above[ar, n_idx].sum(1)) / (len(neg) - k)
    return frr, fa


def analyse(scores: dict, methods=("b1", "b4"), seed: int = 0) -> dict:
    rows = scores["trajectories"]
    dirs = sorted({(r["platform"], r["direction"]) for r in rows})
    sets = {}
    for plat, dr in dirs:
        sub = [r for r in rows if (r["platform"], r["direction"]) == (plat, dr)]
        for m in methods:
            pos = np.array([r[m] for r in sub if r["payloadG"] > 0], float)
            neg = np.array([r[m] for r in sub if r["payloadG"] == 0], float)
            sets[(plat, dr, m)] = (pos, neg, scores["sourceThresholds"][plat][dr][m])

    out = {"directions": []}
    rng = np.random.default_rng(seed)
    draws = {}
    for plat, dr in dirs:  # same labelled subsets for every method and rule
        pos, neg, _ = sets[(plat, dr, methods[-1])]
        for k in KS:
            draws[(plat, dr, k)] = recalibration(pos, neg, k, "", rng)
    for plat, dr in dirs:
        entry = {"platform": plat, "direction": dr}
        for m in methods:
            pos, neg, thr = sets[(plat, dr, m)]
            rec = {"auroc": auroc(pos, neg), "nPos": len(pos), "nNeg": len(neg),
                   "sourceThreshold": {"frr": float((pos < thr).mean()), "fa": float((neg >= thr).mean())},
                   "recal": {}}
            for rule in ("R1", "R2"):
                for k in KS:
                    p_idx, n_idx = draws[(plat, dr, k)]
                    t = pos[p_idx].min(1) if rule == "R1" else (neg[n_idx].max(1) + pos[p_idx].min(1)) / 2
                    frr, fa = evaluate(pos, neg, t, p_idx, n_idx, k)
                    rec["recal"][f"{rule}_k{k}"] = float(((frr <= TAU) & (fa <= TAU)).mean())
            entry[m] = rec
        out["directions"].append(entry)
    passing = [e for e in out["directions"]
               if max(v for key, v in e["b4"]["recal"].items() if int(key.split("_k")[1]) <= 4) >= 0.8]
    out["gate"] = {"rule": ">= 3 of 4 directions with pJoint >= 0.8 at some k <= 4 (b4)",
                   "passing": len(passing), "result": "PASS" if len(passing) >= 3 else "FAIL"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "load_scores.json")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    res = analyse(json.loads(args.scores.read_text()))
    for e in res["directions"]:
        b1, b4 = e["b1"], e["b4"]
        best = max(v for key, v in b4["recal"].items() if int(key.split("_k")[1]) <= 4)
        print(f"{e['platform']:5s} {e['direction']:16s} AUROC b4 {b4['auroc']:.3f} b1 {b1['auroc']:.3f} | "
              f"source threshold on target: FRR {b4['sourceThreshold']['frr']:.3f} FA {b4['sourceThreshold']['fa']:.3f}"
              f" | best pJoint (k<=4) {best:.3f}")
    print("gate:", res["gate"])
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
