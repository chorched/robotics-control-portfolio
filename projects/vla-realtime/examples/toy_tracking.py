"""Toy version of the LIBERO experiments: no GPU, numpy only, runs in seconds.

A point "end effector" has to follow a moving target in the plane. A chunking
policy sees the state at step t0 and plans ``H`` velocity commands for steps
t0, t0+1, ... in open loop. The episode succeeds when the effector stays within
``tol`` of the target for ``hold_k`` consecutive steps. An unknown drift pushes
the effector, so stale or held commands hurt.

The policy query takes E control steps. The same executors used for SmolVLA
(``realtime_chunking``) run the loop, so this reproduces the mechanism - not the
numbers - of the real study:

  1. nominal vs elapsed drop when the loop runs slower than real time;
  2. success over the grid (E, L_c): asynchronous execution needs L_c > E.

    python examples/toy_tracking.py            # prints both experiments
    python examples/toy_tracking.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realtime_chunking import AsyncChunkExecutor, SimulatedRequester, SyncChunkExecutor  # noqa: E402

DT = 0.05
H = 50  # chunk length returned by the policy
T_MAX = 240
TOL = 0.03
HOLD_K = 15
V_MAX = 0.6
TAU = 0.4  # s; slow enough that corrections survive the expired prefix


def target(t: np.ndarray | float, phase: float) -> np.ndarray:
    s = np.asarray(t, dtype=float) * DT
    return np.stack([0.25 * np.sin(0.9 * s + phase), 0.18 * np.sin(1.7 * s + 2 * phase)], axis=-1)


def make_policy(phase: float):
    """Open-loop planner: chase the target's future positions from the observed state."""

    def chunk_fn(obs) -> np.ndarray:
        p, t0 = obs
        p = np.array(p, dtype=float)
        future = target(np.arange(t0, t0 + H + 1), phase)
        acts = np.empty((H, 2))
        for k in range(H):
            # feed-forward target velocity + correction spread over TAU seconds
            v = (future[k + 1] - future[k]) / DT + (future[k] - p) / TAU
            n = np.linalg.norm(v)
            if n > V_MAX:
                v *= V_MAX / n
            acts[k] = v
            p = p + v * DT
        return acts

    return chunk_fn


def hold(_last):
    return np.zeros(2)


def run_episode(executor, phase: float, drift: np.ndarray, rng: np.random.Generator) -> dict:
    executor.reset()
    p = target(0, phase) + rng.normal(0, 0.05, 2)
    streak, success = 0, False
    for t in range(T_MAX):
        a = executor.step((p.copy(), t), t)
        p = p + DT * (np.asarray(a) + drift) + rng.normal(0, 0.0015, 2)
        if np.linalg.norm(p - target(t + 1, phase)) < TOL:
            streak += 1
            if streak >= HOLD_K and t > 40:
                success = True
                break
        else:
            streak = 0
    return {"success": success, "stats": executor.stats}


def evaluate(make_executor, n_episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    wins, miss, dme, held_q = 0, [], [], []
    for i in range(n_episodes):
        phase = rng.uniform(0, 2 * np.pi)
        drift = rng.normal(0, 0.01, 2)
        ex = make_executor(make_policy(phase), seed + i)
        out = run_episode(ex, phase, drift, rng)
        wins += out["success"]
        miss.append(out["stats"].deadline_miss_rate)
        dme.append(out["stats"].mean_drop_minus_elapsed)
        held_q.append(out["stats"].held_per_query)
    return {
        "success": wins,
        "n": n_episodes,
        "deadline_miss": float(np.mean(miss)),
        "drop_minus_elapsed": float(np.mean(dme)),
        "held_per_query": float(np.mean(held_q)),
    }


def latency(e_mean: float, jitter: int = 1):
    return lambda rng: int(round(e_mean)) + int(rng.integers(-jitter, jitter + 1))


def drop_rule_experiment(n: int, seed: int) -> dict:
    """Loop runs 2.1x slower than real time (the C4t condition): E ~ 8 steps, wall latency ~ 840 ms."""
    out = {}
    for name, mode in [("async_nominal", "nominal"), ("async_elapsed", "elapsed")]:
        out[name] = evaluate(
            lambda pol, s, mode=mode: AsyncChunkExecutor(
                SimulatedRequester(pol, DT, latency(8), slowdown=2.1, seed=s),
                DT, h_trig=10, drop_mode=mode, hold_fn=hold,
            ),
            n, seed,
        )
    out["sync_replan8"] = evaluate(
        lambda pol, s: SyncChunkExecutor(
            SimulatedRequester(pol, DT, latency(8), seed=s), DT, replan_every=8, hold_fn=hold
        ),
        n, seed,
    )
    out["paused_reference"] = evaluate(
        lambda pol, s: SyncChunkExecutor(
            SimulatedRequester(pol, DT, latency(8), seed=s), DT, replan_every=8,
            charge_latency=False, hold_fn=hold,
        ),
        n, seed,
    )
    return out


def chunk_budget_grid(n: int, seed: int, es=(2, 4, 6, 8, 10, 12, 14), lcs=(4, 6, 8, 10, 12, 14, 16, 20)) -> dict:
    """Success over (E, L_c) with the elapsed drop rule and real-time loop."""
    grid = np.zeros((len(es), len(lcs)), dtype=int)
    held = np.zeros_like(grid, dtype=float)
    for i, e in enumerate(es):
        for j, lc in enumerate(lcs):
            r = evaluate(
                lambda pol, s, e=e, lc=lc: AsyncChunkExecutor(
                    SimulatedRequester(pol, DT, latency(e), seed=s),
                    DT, h_trig=max(1, e), drop_mode="elapsed", chunk_use_limit=lc, hold_fn=hold,
                ),
                n, seed,
            )
            grid[i, j] = r["success"]
            held[i, j] = r["held_per_query"]
    return {"E": list(es), "Lc": list(lcs), "success": grid.tolist(), "held_per_query": held.tolist(), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    drop = drop_rule_experiment(args.episodes, args.seed)
    print("Experiment 1 - loop 2.1x slower than real time, E ~ 8 steps")
    for k, v in drop.items():
        print(f"  {k:18s} success {v['success']:3d}/{v['n']}  deadline-miss {v['deadline_miss']:.2f}"
              f"  (dropped - elapsed)/query {v['drop_minus_elapsed']:+.1f}")
    grid = chunk_budget_grid(max(20, args.episodes // 2), args.seed)
    print("\nExperiment 2 - success over E (rows) x L_c (cols), elapsed drop rule")
    print("      " + " ".join(f"{lc:4d}" for lc in grid["Lc"]))
    for e, row in zip(grid["E"], grid["success"]):
        print(f"E={e:3d} " + " ".join(f"{x:4d}" for x in row))
    if args.json:
        args.json.write_text(json.dumps({"drop_rule": drop, "chunk_budget": grid}, indent=1))


if __name__ == "__main__":
    main()
