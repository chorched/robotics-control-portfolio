import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realtime_chunking import (  # noqa: E402
    AsyncChunkExecutor,
    SimulatedRequester,
    SyncChunkExecutor,
    expired_prefix,
    mcnemar_exact,
    newcombe_diff,
)

DT = 0.05


def indexed_policy(obs):
    """Action k of a chunk planned at step t0 carries the value t0 + k (the step it is meant for)."""
    _, t0 = obs
    return np.arange(t0, t0 + 50, dtype=float)[:, None]


def run(executor, steps=200):
    applied = []
    for t in range(steps):
        a = executor.step((None, t), t)
        applied.append((t, None if a is None else float(np.asarray(a).ravel()[0])))
    return applied


def hold(_):
    return np.array([np.nan])


def test_expired_prefix_rules():
    assert expired_prefix(0.40, DT, 8, "nominal") == 8
    assert expired_prefix(0.85, DT, 8, "nominal") == 17  # slow loop: wall time overstates
    assert expired_prefix(0.85, DT, 8, "elapsed") == 8
    assert expired_prefix(0.20, DT, 8, "elapsed") == 4  # never drops more than nominal


def test_elapsed_rule_keeps_actions_on_schedule_when_loop_is_slow():
    req = SimulatedRequester(indexed_policy, DT, lambda r: 8, slowdown=2.1)
    ex = AsyncChunkExecutor(req, DT, h_trig=8, drop_mode="elapsed", hold_fn=hold)
    applied = [(t, a) for t, a in run(ex) if not np.isnan(a)]
    # With h_trig == E the queue drains exactly when the next chunk lands: every
    # applied action is the one planned for that step.
    assert all(t == a for t, a in applied[10:])
    assert ex.stats.mean_drop_minus_elapsed == 0


def test_nominal_rule_skips_future_actions_when_loop_is_slow():
    req = SimulatedRequester(indexed_policy, DT, lambda r: 8, slowdown=2.1)
    ex = AsyncChunkExecutor(req, DT, h_trig=8, drop_mode="nominal", hold_fn=hold)
    applied = [(t, a) for t, a in run(ex) if not np.isnan(a)]
    assert ex.stats.mean_drop_minus_elapsed > 8  # ceil(0.84/0.05) - 8 = 9 future actions lost
    assert any(a - t >= 9 for t, a in applied[10:])  # commands meant for the future


def test_queue_starves_when_chunk_budget_is_below_E():
    req = SimulatedRequester(indexed_policy, DT, lambda r: 10)
    short = AsyncChunkExecutor(req, DT, h_trig=10, chunk_use_limit=6, hold_fn=hold)
    run(short)
    req2 = SimulatedRequester(indexed_policy, DT, lambda r: 10)
    long = AsyncChunkExecutor(req2, DT, h_trig=10, chunk_use_limit=14, hold_fn=hold)
    run(long)
    assert short.stats.held_per_query > 3
    assert long.stats.held_per_query < 1


def test_sync_charges_latency_as_holds():
    req = SimulatedRequester(indexed_policy, DT, lambda r: 8)
    ex = SyncChunkExecutor(req, DT, replan_every=8, hold_fn=hold)
    run(ex, 160)
    assert ex.stats.queries == 160 // 15 + 1
    assert abs(ex.stats.held / ex.stats.steps - 7 / 15) < 0.05


def test_stats_match_registered_values():
    d, lo, hi = newcombe_diff(15, 24, 0, 24)
    assert round(lo, 4) == 0.3837
    d, lo, hi = newcombe_diff(51, 56, 17, 56)
    assert (round(lo, 3), round(hi, 3)) == (0.441, 0.723)
    assert abs(mcnemar_exact(36, 2) - 5.3988e-9) < 1e-12
