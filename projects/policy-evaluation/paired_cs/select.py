"""Sequential checkpoint selection with hedged-betting confidence sequences.

Two programs, both charging one rollout per live candidate per round on the
outcomes of a single pseudo-population draw (all candidates are evaluated on
the *same* initial state, so a round is fully described by ``Y[:, i]``):

``S2_marg``
    one marginal sequence per candidate at level ``alpha / K``; a candidate is
    selected once its lower bound clears every other candidate's upper bound
    by at most ``delta``.

``S2_pair``
    one sequence per unordered pair of *live* candidates on the transformed
    observation ``x = (Y_j - Y_k + 1) / 2`` at level ``alpha / (K (K-1) / 2)``;
    the interval for ``p_j - p_k`` is ``[2L-1, 2U-1]``.  A candidate is
    selected once it is provably within ``delta`` of every other live
    candidate.

Reference tables
----------------
A *candidate x initial state* table is an ``npz`` file with

``Y``          ``(K, n)`` array, values in ``[0, 1]`` (success indicators).
``seeds``      optional ``(n,)`` integer array labelling each column.
``evaluated``  optional ``(K, n)`` 0/1 mask; ``0`` means "no observation".
``candidate``  optional ``(K,)`` string labels.

``run_selection`` consumes either a callable ``() -> (K,)``, or a 2-D array
``(K, n_pool)`` that is then sampled with replacement (columns = pooled
initial states) using ``seed``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .cs import N_GRID, BettingCS

__all__ = [
    "METHODS",
    "Table",
    "load_table",
    "make_index_sampler",
    "run_selection",
    "run_replications",
]

METHODS = ("S2_marg", "S2_pair")


# ---------------------------------------------------------------------------
# input tables
# ---------------------------------------------------------------------------
class Table:
    """Candidate x initial-state outcome table.

    Attributes
    ----------
    Y : ``(K, n)`` float64 array of outcomes in ``[0, 1]``.
    observed : ``(K, n)`` bool array; ``False`` marks a missing observation.
    seeds : ``(n,)`` int64 labels (``arange(n)`` when the file has no seeds).
    labels : ``(K,)`` list of candidate identifiers (``"c0".."cK-1"`` default).
    """

    def __init__(self, Y, observed=None, seeds=None, labels=None):
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (candidates x initial states), got {Y.shape}")
        self.Y = Y
        K, n = Y.shape
        if observed is None:
            self.observed = np.ones((K, n), dtype=bool)
        else:
            observed = np.asarray(observed)
            if observed.shape != (K, n):
                raise ValueError(f"evaluated mask shape {observed.shape} != Y shape {Y.shape}")
            self.observed = observed.astype(bool)
        if seeds is None:
            self.seeds = np.arange(n, dtype=np.int64)
        else:
            seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
            if seeds.shape != (n,):
                raise ValueError(f"seeds shape {seeds.shape} != ({n},)")
            if np.unique(seeds).size != n:
                raise ValueError("seeds must be unique")
            self.seeds = seeds
        if labels is None:
            self.labels = [f"c{k}" for k in range(K)]
        else:
            labels = [str(v) for v in np.asarray(labels).reshape(-1)]
            if len(labels) != K:
                raise ValueError(f"labels length {len(labels)} != K={K}")
            self.labels = labels
        if np.any(self.Y[self.observed] < 0.0) or np.any(self.Y[self.observed] > 1.0):
            raise ValueError("observed outcomes must lie in [0, 1]")

    @property
    def K(self) -> int:
        return int(self.Y.shape[0])

    @property
    def n(self) -> int:
        return int(self.Y.shape[1])

    def index_of(self, ident) -> int:
        """Resolve a candidate index or label to an index."""
        if isinstance(ident, (int, np.integer)):
            idx = int(ident)
        else:
            s = str(ident)
            if s in self.labels:
                return self.labels.index(s)
            try:
                idx = int(s)
            except ValueError:
                raise ValueError(f"unknown candidate id {ident!r}") from None
        if not 0 <= idx < self.K:
            raise ValueError(f"candidate index {idx} out of range [0, {self.K})")
        return idx

    def aligned(self, a: int, b: int):
        """Paired observations for candidates ``a`` and ``b``.

        Returns ``(seeds, D)`` where ``seeds`` are the labels of the initial
        states observed for *both* candidates, in ascending order, and
        ``D = Y[a] - Y[b]`` on exactly those states.
        """
        both = self.observed[a] & self.observed[b]
        order = np.argsort(self.seeds[both], kind="stable")
        seeds = self.seeds[both][order]
        D = self.Y[a][both][order] - self.Y[b][both][order]
        return seeds, D


def load_table(path, key: str = "Y") -> Table:
    """Load a candidate x initial-state table from an ``npz`` file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"{path.name} has no array {key!r}; keys={sorted(z.files)}")
        Y = np.asarray(z[key])
        observed = np.asarray(z["evaluated"]) if "evaluated" in z.files else None
        seeds = np.asarray(z["seeds"]) if "seeds" in z.files else None
        if "candidate_label" in z.files:
            labels = [str(v) for v in np.asarray(z["candidate_label"]).reshape(-1)]
        elif "candidate" in z.files:
            labels = [str(v) for v in np.asarray(z["candidate"]).reshape(-1)]
        else:
            labels = None
    if Y.ndim == 1:          # a single candidate row: accept it as (1, n)
        Y = Y.reshape(1, -1)
    return Table(Y, observed=observed, seeds=seeds, labels=labels)


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------
def make_index_sampler(Y, seed: int) -> Callable[[], np.ndarray]:
    """Sampler drawing one pooled initial-state index per call, with replacement.

    ``Y`` has shape ``(K, n_pool)``; each call returns ``Y[:, i]`` for a fresh
    ``i`` drawn uniformly from ``range(n_pool)``.
    """
    Ym = np.asarray(Y, dtype=np.float64)
    if Ym.ndim != 2:
        raise ValueError("Y must be 2-D (K, n_pool)")
    rng = np.random.default_rng(seed)
    n_pool = Ym.shape[1]

    def draw() -> np.ndarray:
        return Ym[:, int(rng.integers(n_pool))]

    return draw


def _as_draw(Y_sampler, K: int, seed: int) -> Callable[[], np.ndarray]:
    if callable(Y_sampler):
        return Y_sampler
    arr = np.asarray(Y_sampler, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Y_sampler must be callable or a 2-D array (K, n_pool)")
    if arr.shape[0] != K:
        raise ValueError(f"Y has {arr.shape[0]} rows, expected K={K}")
    return make_index_sampler(arr, seed)


def _row(draw: Callable[[], np.ndarray], K: int) -> np.ndarray:
    y = np.asarray(draw(), dtype=np.float64).reshape(-1)
    if y.shape != (K,):
        raise ValueError(f"sampler must return shape ({K},), got {y.shape}")
    return y


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def run_selection(
    Y_sampler,
    K: int,
    delta: float,
    method: str,
    seed: int,
    cap: int,
    alpha: float = 0.05,
) -> dict:
    """Run one sequential selection replication.

    Parameters
    ----------
    Y_sampler : callable or array
        ``callable() -> array (K,)`` giving every candidate's outcome for the
        next pseudo-population draw (all candidates share the draw).  A 2-D
        array ``(K, n_pool)`` is accepted and sampled with replacement.
    K : number of candidates.
    delta : tolerance, "within ``delta`` of the best" (0.05 = 5 pp).
    method : ``"S2_marg"`` or ``"S2_pair"``.
    seed : recorded in the result; seeds the draw when ``Y_sampler`` is an array.
    cap : hard cap on total rollouts for this replication.
    alpha : overall level of the joint procedure (default 0.05).

    Returns
    -------
    dict with keys ``method, seed, delta, K, alpha, cap, stopped,
    totalRollouts, rounds, chosen, chosenL, chosenU, runtimeSec, csEmpty,
    csDegenerate, stopWhy, nLive``.

    Notes
    -----
    * A round is one observation per live candidate, so
      ``totalRollouts = sum_rounds (#live candidates at the start of the round)``.
    * ``chosen`` is ``None`` when the cap is reached (``stopped = False``).
    * Tie-breaking is deterministic: among qualifying candidates, the one with
      the best (largest) decisive lower bound wins, then the lowest index.
    """
    method = str(method)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    K = int(K)
    cap = int(cap)
    delta = float(delta)
    draw = _as_draw(Y_sampler, K, int(seed))
    t0 = time.perf_counter()

    if method == "S2_marg":
        res = _run_marg(draw, K, delta, cap, alpha)
    else:
        res = _run_pair(draw, K, delta, cap, alpha)

    res.update(
        method=method,
        seed=int(seed),
        delta=delta,
        K=K,
        alpha=float(alpha),
        cap=cap,
        runtimeSec=time.perf_counter() - t0,
    )
    return res


_WORKER = {}


def _init_worker(Y, K, delta, method, cap, alpha):
    _WORKER.update(Y=Y, K=K, delta=delta, method=method, cap=cap, alpha=alpha)


def _run_one(seed):
    return run_selection(
        _WORKER["Y"], _WORKER["K"], _WORKER["delta"], _WORKER["method"],
        seed, _WORKER["cap"], alpha=_WORKER["alpha"],
    )


def run_replications(
    Y,
    K: int,
    delta: float,
    method: str,
    seeds: Sequence[int],
    cap: int,
    alpha: float = 0.05,
    n_jobs: int = 1,
) -> list:
    """Run several independent replications (one per element of ``seeds``).

    ``Y`` is a ``(K, n_pool)`` table sampled with replacement inside each
    replication, exactly as :func:`run_selection` does.  ``n_jobs > 1`` spreads
    the replications over forked/spawned worker processes; results are returned
    in the order of ``seeds``.
    """
    seeds = [int(s) for s in seeds]
    Y = np.asarray(Y, dtype=np.float64)
    if n_jobs is None or int(n_jobs) <= 1 or len(seeds) <= 1:
        return [
            run_selection(Y, K, delta, method, s, cap, alpha=alpha) for s in seeds
        ]
    import concurrent.futures as cf

    jobs = min(int(n_jobs), len(seeds))
    out = []
    with cf.ProcessPoolExecutor(
        max_workers=jobs,
        initializer=_init_worker,
        initargs=(Y, int(K), float(delta), str(method), int(cap), float(alpha)),
    ) as ex:
        for res in ex.map(_run_one, seeds):
            out.append(res)
    return out


# ---------------------------------------------------------------------------
# S2-marg: one marginal sequence per candidate
# ---------------------------------------------------------------------------
def _run_marg(draw, K, delta, cap, alpha):
    cs = BettingCS(K, alpha / K)
    live = np.arange(K)          # live[r] -> original candidate index
    total = 0
    rounds = 0
    why = "cap"

    while True:
        n = live.size
        if n <= 1:
            why = "single"
            break
        if total + n > cap:
            why = "cap"
            break

        y = _row(draw, K)
        cs.update(y[live])
        total += n
        rounds += 1
        L, U = cs.compute_interval()

        # elimination: rank candidate k out when U_k < max_j L_j
        live_mask = ~(U < float(L.max()))
        if not live_mask.all():
            live = live[live_mask]
            cs = cs.subset(live_mask)
            L, U = cs.L, cs.U
            if live.size <= 1:
                why = "single"
                break

        # stopping: some live kh with L_kh >= max_{j != kh} U_j - delta
        sel = _qualify_marg(L, U, delta)
        if sel is not None:
            return _stop_result(True, total, rounds, int(live[sel]),
                                float(L[sel]), float(U[sel]), cs,
                                int(live.size), "rule")

    chosen = int(live[0]) if live.size == 1 else None
    if why == "single":
        sel = 0
        return _stop_result(True, total, rounds, chosen, float(cs.L[sel]),
                            float(cs.U[sel]), cs, 1, "single")
    return _stop_result(False, total, rounds, None, float("nan"),
                        float("nan"), cs, int(live.size), why)


def _qualify_marg(L, U, delta):
    """Index of the candidate satisfying the S2-marg stop rule, else ``None``.

    The rule is checked for *every* live candidate (a candidate with a wide
    upper bound can qualify even when it is not the largest lower bound), and
    among qualifying candidates the one with the largest lower bound wins.
    """
    n = L.size
    if n <= 1:
        return 0
    amax = int(np.argmax(U))
    maxU_all = float(U[amax])
    if n > 1:
        second = float(np.max(np.delete(U, amax)))
    else:
        second = maxU_all
    maxU_excl = np.where(np.arange(n) == amax, second, maxU_all)
    qual = L >= maxU_excl - delta
    if not qual.any():
        return None
    idx = np.flatnonzero(qual)
    return int(idx[np.argmax(L[idx])])


# ---------------------------------------------------------------------------
# S2-pair: one sequence per unordered pair of live candidates
# ---------------------------------------------------------------------------
def _run_pair(draw, K, delta, cap, alpha):
    if K < 2:  # no pairs to test: the single candidate is trivially selected
        return _stop_result(True, 0, 0, 0, float("nan"), float("nan"),
                            BettingCS(1, alpha), 1, "single")
    n_pairs = K * (K - 1) // 2
    cs = BettingCS(n_pairs, alpha / n_pairs)
    pj, pk = np.triu_indices(K, 1)          # pair endpoints (original indices)
    live = np.arange(K)
    total = 0
    rounds = 0
    why = "cap"

    while True:
        n = live.size
        if n <= 1:
            why = "single"
            break
        if total + n > cap:
            why = "cap"
            break

        y = _row(draw, K)
        cs.update((y[pj] - y[pk] + 1.0) / 2.0)   # x in {0, 0.5, 1}
        total += n
        rounds += 1
        L, U = cs.compute_interval()
        low = 2.0 * L - 1.0                     # bound on p_j - p_k
        upp = 2.0 * U - 1.0

        # dense skew-symmetric bound matrix on the live candidates
        D = np.full((K, K), np.inf)
        D[pj, pk] = low
        D[pk, pj] = -upp
        np.fill_diagonal(D, 0.0)

        # elimination: k out if some j proves p_j - p_k > 0
        kill = (D > 0.0).any(axis=0) & np.isin(np.arange(K), live)
        if kill.any():
            keep_c = ~kill[live]
            if keep_c.any():
                live_new = live[keep_c]
                if live_new.size <= 1:
                    live = live_new
                    why = "single"
                    break
                keep_p = keep_c[pj] & keep_c[pk]
                live = live_new
                cs = cs.subset(keep_p)
                pj, pk = pj[keep_p], pk[keep_p]
                low, upp = low[keep_p], upp[keep_p]
                D = np.full((K, K), np.inf)
                D[pj, pk] = low
                D[pk, pj] = -upp
                np.fill_diagonal(D, 0.0)

        # stopping: kh within delta of every other live candidate
        slack = D.min(axis=1)[live]             # lower bound on p_c - p_other
        ok = slack >= -delta - 1e-12
        if ok.any():
            cand = np.flatnonzero(ok)
            sel = cand[0] if cand.size == 1 else cand[int(np.argmax(slack[cand]))]
            return _stop_result(True, total, rounds, int(live[sel]),
                                float(slack[sel]), float("nan"), cs,
                                int(live.size), "rule")

    if why == "single":
        return _stop_result(True, total, rounds, int(live[0]), float("nan"),
                            float("nan"), cs, 1, "single")
    return _stop_result(False, total, rounds, None, float("nan"),
                        float("nan"), cs, int(live.size), why)


def _stop_result(stopped, total, rounds, chosen, chosenL, chosenU, cs, n_live, why):
    return dict(
        stopped=bool(stopped),
        totalRollouts=int(total),
        rounds=int(rounds),
        chosen=chosen,
        chosenL=float(chosenL),
        chosenU=float(chosenU),
        csEmpty=int(cs.n_empty),
        csDegenerate=int(cs.n_degenerate),
        stopWhy=why,
        nLive=int(n_live),
    )
