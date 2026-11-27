"""Paired (same-initial-state) confidence-sequence certification of a selection.

Given a reference table of per-candidate outcomes on a set of *shared* initial
states, the regret of a selected candidate ``kh`` is

    regret = max_j p_j - p_kh

where ``j`` runs over every other candidate and ``p`` is the success rate.  A
one-sided statement is obtained from an anytime-valid confidence sequence on
the *paired* mean of ``x = (Y_j - Y_kh + 1) / 2``:

    CS_t on mean(x)  ->  [2L - 1, 2U - 1]  on p_j - p_kh  ->  [-upper, -lower] on p_kh - p_j

so the lower bound on the regret is ``max(0, max_j l_j)`` and the upper bound is
``max(0, max_j u_j)`` where ``l_j``/``u_j`` bound ``p_j - p_kh``.

Because both outcomes come from the *same* initial state the difference is
positively correlated and the paired interval is far tighter than combining two
independent intervals.  Each of the ``m`` comparisons carries level
``alpha / m``; ``m`` must be supplied by the caller (``n_selected * (K - 1)``
when several candidates are certified on the same data) so that the joint
statement holds at level ``alpha``.

A rival that shares *no* initial state with the selected candidate contributes
no paired data at all.  Its difference ``p_j - p_kh`` is then only known to lie
in the trivial interval ``[-1, 1]``, so it enters the regret as ``l_j = -1``,
``u_j = 1`` -- i.e. it silently caps the upper bound at 1 and makes the verdict
``undecidable``.  Dropping such rivals instead would report an upper bound that
ignores a candidate the selection may have lost badly to.  They are listed in
``missingRivals``.  Only when *every* rival has ``n == 0`` is a ``ValueError``
raised, because then there is no selected-candidate data to certify at all.

Assumption: the initial states are i.i.d. draws from the evaluation
distribution and every candidate is scored on that same draw.  Nothing about
independence *across candidates* is required -- that is exactly what the
pairing exploits.
"""

from __future__ import annotations

import math

import numpy as np

from .cs import N_GRID, BettingCS
from .select import Table, load_table

__all__ = ["diff_interval", "pair_interval", "certify", "certify_many", "verdict_of"]


def _num(x, decimals: int):
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return float(round(x, decimals))


def verdict_of(regret_lower: float, regret_upper: float, delta: float) -> str:
    """Three-valued verdict for a regret interval.

    ``correct``      the whole interval lies inside the tolerance
                     (``upper <= delta``): no more than ``delta`` is lost.
    ``wrong``        the whole interval lies above the tolerance
                     (``lower > delta``): more than ``delta`` is lost.
    ``undecidable``  the interval straddles ``delta``.
    """
    if not (math.isfinite(regret_lower) and math.isfinite(regret_upper)):
        return "undecidable"
    if regret_upper <= delta:
        return "correct"
    if regret_lower > delta:
        return "wrong"
    return "undecidable"


def diff_interval(d, alpha: float, n_grid: int = N_GRID):
    """Confidence interval for the mean of an aligned difference sequence.

    ``d`` holds ``Y_j - Y_k`` for one initial state per element, in the order
    the observations are consumed.  Returns ``(lower, upper, n)``; the bounds
    are ``nan`` when ``n == 0``.
    """
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    n = int(d.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    cs = BettingCS(1, alpha, n_grid=n_grid)
    x = (d + 1.0) / 2.0                       # in [0, 1] because d in [-1, 1]
    for i in range(n):
        cs.update(x[i:i + 1])
    L, U = cs.compute_interval()
    return float(2.0 * L[0] - 1.0), float(2.0 * U[0] - 1.0), n


def pair_interval(a, b, alpha: float, n_grid: int = N_GRID):
    """Paired confidence interval for ``p_a - p_b`` from aligned outcome vectors.

    ``a`` and ``b`` must be index-aligned on the same initial states (see
    :meth:`paired_cs.select.Table.aligned`).
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"aligned vectors differ in length: {a.shape} vs {b.shape}")
    return diff_interval(a - b, alpha, n_grid=n_grid)


def certify(
    table: Table,
    selected,
    delta: float = 0.05,
    alpha: float = 0.05,
    n_comparisons: int = 1,
    n_grid: int = N_GRID,
    decimals: int = 6,
) -> dict:
    """Certify the regret of one selected candidate against every rival.

    Parameters
    ----------
    table : reference :class:`~paired_cs.select.Table` (candidates x shared
        initial states).
    selected : candidate index or label (see :meth:`Table.index_of`).
    delta : tolerance in success-rate units (0.05 = 5 pp).
    alpha : overall level of the joint statement.
    n_comparisons : number of comparisons covered by ``alpha``; every pair is
        run at ``alpha / n_comparisons``.
    n_grid : confidence-sequence grid resolution.
    decimals : rounding applied to the reported bounds (``None`` keeps full
        precision).

    Returns
    -------
    dict with ``selected``/``selectedIndex``, ``delta``, ``alpha``,
    ``nComparisons``, ``alphaPerComparison``, ``regretLower``, ``regretUpper``,
    ``verdict``, ``decisiveRival``/``decisiveRivalIndex``, ``nDecisive``,
    ``missingRivals``/``nMissingRivals`` and the per-rival list ``pairs``.

    Rivals with ``n == 0`` (no initial state in common with the selection) are
    kept in ``pairs`` with the trivial bounds ``l = -1``, ``u = 1`` and are
    named in ``missingRivals``.  Because ``u = 1`` enters the maximum, the
    reported ``regretUpper`` is 1 and the verdict is ``undecidable`` -- no
    rival can be silently dropped from the bound.
    """
    kh = table.index_of(selected)
    K = table.K
    delta = float(delta)
    alpha = float(alpha)
    m = int(n_comparisons)
    if m < 1:
        raise ValueError("n_comparisons must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")
    alpha_cmp = alpha / m

    rows = []
    for j in range(K):
        if j == kh:
            continue
        _, D = table.aligned(j, kh)
        n = int(D.size)
        if n == 0:
            # no shared initial state: p_j - p_kh is only known to lie in [-1, 1]
            l, u = -1.0, 1.0
        else:
            l, u, n = diff_interval(D, alpha_cmp, n_grid=n_grid)
        half = (u - l) / 2.0
        rows.append({
            "rival": table.labels[j],
            "rivalIndex": int(j),
            "n": int(n),
            "l": l,
            "u": u,
            "pointDiff": float(np.mean(D)) if n else float("nan"),
            "halfWidth": half,
            "sd": float(np.std(D, ddof=1)) if n > 1 else float("nan"),
        })

    missing = [r for r in rows if r["n"] == 0]
    usable = [r for r in rows if r["n"] > 0]
    if not usable:
        raise ValueError(
            f"candidate {table.labels[kh]!r} has no initial state in common with any rival"
        )
    # every rival contributes, including the ones with no paired data ([-1, 1])
    lo = max(0.0, max(r["l"] for r in rows))
    hi = max(0.0, max(r["u"] for r in rows))
    dec = max(rows, key=lambda r: r["u"])

    rnd = (lambda x: x) if decimals is None else (lambda x: _num(x, decimals))
    out = {
        "selected": table.labels[kh],
        "selectedIndex": int(kh),
        "delta": delta,
        "alpha": alpha,
        "nComparisons": m,
        "alphaPerComparison": alpha_cmp,
        "nCandidates": int(K),
        "regretLower": rnd(lo),
        "regretUpper": rnd(hi),
        "verdict": verdict_of(lo, hi, delta),
        "decisiveRival": table.labels[int(dec["rivalIndex"])],
        "decisiveRivalIndex": int(dec["rivalIndex"]),
        "nDecisive": int(dec["n"]),
        "missingRivals": [{
            "rival": r["rival"],
            "rivalIndex": int(r["rivalIndex"]),
        } for r in missing],
        "nMissingRivals": int(len(missing)),
        "pairs": [{
            "rival": r["rival"],
            "rivalIndex": r["rivalIndex"],
            "n": r["n"],
            "l": rnd(r["l"]),
            "u": rnd(r["u"]),
            "pointDiff": rnd(r["pointDiff"]),
            "halfWidth": rnd(r["halfWidth"]),
            "sd": rnd(r["sd"]),
        } for r in rows],
    }
    return out


def certify_many(table: Table, ids, delta=0.05, alpha=0.05, n_comparisons=None,
                 n_grid: int = N_GRID, decimals: int = 6) -> dict:
    """Certify several selected candidates with a shared joint level.

    ``n_comparisons`` defaults to ``len(ids) * (K - 1)``, the convention used by
    the A5p batch (every certified candidate against every rival).
    """
    ids = list(ids)
    if n_comparisons is None:
        n_comparisons = len(ids) * (table.K - 1)
    results = [
        certify(table, i, delta=delta, alpha=alpha, n_comparisons=n_comparisons,
                n_grid=n_grid, decimals=decimals)
        for i in ids
    ]
    return {
        "delta": float(delta),
        "alpha": float(alpha),
        "nComparisons": int(n_comparisons),
        "nCandidates": int(table.K),
        "perSelected": results,
    }
