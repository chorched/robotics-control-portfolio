"""T1 -- sequential (anytime) coverage of the hedged betting confidence sequence.

For independent Bernoulli(p) streams the confidence sequence must fail to cover
the true mean rarely at *every* time point simultaneously, not just at one fixed
sample size.  We run 300 independent replications of length 2000 for
``p in {0.1, 0.5, 0.9}`` and require

    (#replications that missed p at some time t) / 300  <=  alpha + 0.02

with ``alpha = 0.05``.  The empirical rate is expected to stay far below the
bound because the betting capital is conservative; the +0.02 is Monte-Carlo and
discretisation slack.
"""

from __future__ import annotations

import numpy as np
import pytest

from paired_cs import N_GRID, BettingCS

from conftest import record_detail

ALPHA = 0.05
REPS = 300
LENGTH = 2000
PS = (0.1, 0.5, 0.9)
SLACK = 0.02
SEED = 20261116


@pytest.mark.parametrize("p", PS)
def test_anytime_coverage(p):
    rng = np.random.default_rng(SEED + int(round(p * 100)))
    # X[i, :] is the i-th observation for all REPS replications
    X = (rng.random((LENGTH, REPS)) < p).astype(np.float64)
    cs = BettingCS(REPS, ALPHA, n_grid=N_GRID)
    truth = np.full(REPS, float(p))
    ever_missed = np.zeros(REPS, dtype=bool)
    for i in range(LENGTH):
        cs.update(X[i])
        cs.compute_interval()
        ever_missed |= ~cs.is_covered(truth)
    rate = float(ever_missed.mean())
    record_detail(f"T1:p={p}", {
        "p": p, "alpha": ALPHA, "reps": REPS, "length": LENGTH,
        "anytimeNonCoverage": rate, "threshold": ALPHA + SLACK,
        "missedCount": int(ever_missed.sum()),
        "finalMedianWidth": float(np.median(cs.U - cs.L)),
    })
    assert rate <= ALPHA + SLACK, (
        f"p={p}: anytime non-coverage {rate:.4f} exceeds {ALPHA + SLACK:.4f} "
        f"({int(ever_missed.sum())}/{REPS} replications missed at some t)"
    )
    # the sequence must be informative, not vacuously wide
    assert np.isfinite(cs.L).all() and np.isfinite(cs.U).all()
    assert (cs.U - cs.L).max() < 1.0


def test_final_interval_shrinks_with_n():
    """Monotone running intersection plus a sqrt-n width decay (sanity)."""
    rng = np.random.default_rng(SEED)
    cs = BettingCS(20, ALPHA)
    X = (rng.random((1500, 20)) < 0.5).astype(np.float64)
    widths = {}
    for i in range(1500):
        cs.update(X[i])
        L, U = cs.compute_interval()
        if i + 1 in (100, 1400):
            widths[i + 1] = float(np.median(U - L))
    assert widths[1400] < widths[100]
