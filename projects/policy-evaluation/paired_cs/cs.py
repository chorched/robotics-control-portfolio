"""Batched hedged-betting confidence sequence for means of ``[0, 1]`` observations.

Behavioural port of ``BettingCS`` from the A2t batch (``A2t/cs_select.py``),
which was validated in A2t/A4p/A5p.  Definitions (``m`` indexes the grid, ``t``
the number of observations already seen for that sequence)::

    mu_t   = (0.5 + sum_{i<=t} x_i) / (t + 1)
    s2_t   = (0.25 + sum_{i<=t} (x_i - mu_i)^2) / (t + 1),  s2_0 = 0.25
    lam_i  = sqrt( 2 ln(2/a) / (s2_{i-1} * i * ln(i+1)) )      (predictable)
    K+_t(m)= prod_i ( 1 + min(lam_i, 0.5/m)     * (x_i - m) )
    K-_t(m)= prod_i ( 1 - min(lam_i, 0.5/(1-m)) * (x_i - m) )
    K_t(m) = 0.5 * K+_t(m) + 0.5 * K-_t(m)
    CS_t   = { m : K_t(m) < 1/a }

The grid is ``m_k = (k + 0.5) / n_grid``.  The reported interval is
``[min(CS_t), max(CS_t)]`` intersected with the previous interval (running
intersection, so reported bounds are monotone).  Everything is carried in log
capital (``logaddexp`` for the hedge mix) to avoid overflow/underflow.

Reference: Waudby-Smith & Ramdas (2024), hedged capital processes.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["N_GRID", "grid_points", "BettingCS"]

N_GRID = 400
_EPS = 1e-12


def grid_points(n_grid: int = N_GRID) -> np.ndarray:
    """Midpoint grid on (0, 1): ``(k + 0.5) / n_grid``."""
    return (np.arange(n_grid, dtype=np.float64) + 0.5) / float(n_grid)


class BettingCS:
    """Batched hedged-betting confidence sequence for means in [0, 1].

    Parameters
    ----------
    n : number of independent sequences tracked in parallel.
    alpha : scalar or length-``n`` array of miscoverage levels (per sequence).
    n_grid : number of grid points for the confidence set.

    The class is row-oriented: :meth:`update` takes one observation per live
    sequence.  Rows can be removed/kept with :meth:`subset`, which is how the
    sequential selection programs drop eliminated candidates.
    """

    def __init__(self, n: int, alpha, n_grid: int = N_GRID) -> None:
        self.n = int(n)
        self.m = grid_points(n_grid)
        self.n_grid = int(n_grid)
        # constant betting caps: min(lam, 0.5/m) and min(lam, 0.5/(1-m))
        self.cap_hi = 0.5 / self.m
        self.cap_lo = 0.5 / (1.0 - self.m)
        a = np.asarray(alpha, dtype=np.float64)
        self.alpha = np.full(self.n, float(a)) if a.ndim == 0 else a.astype(np.float64).copy()
        if self.alpha.shape != (self.n,):
            raise ValueError("alpha must be a scalar or have length n")
        if np.any(self.alpha <= 0.0) or np.any(self.alpha >= 1.0):
            raise ValueError("alpha must lie strictly inside (0, 1)")

        self.t = np.zeros(self.n, dtype=np.int64)
        self.S = np.zeros(self.n, dtype=np.float64)   # sum x_i
        self.D = np.zeros(self.n, dtype=np.float64)   # sum (x_i - mu_i)^2
        self.s2_prev = np.full(self.n, 0.25, dtype=np.float64)  # s2_t
        self.A = np.zeros((self.n, self.n_grid), dtype=np.float64)  # log K+
        self.B = np.zeros((self.n, self.n_grid), dtype=np.float64)  # log K-
        self.L = np.full(self.n, np.nan, dtype=np.float64)
        self.U = np.full(self.n, np.nan, dtype=np.float64)
        self.lam_last = np.zeros(self.n, dtype=np.float64)
        self.n_empty = 0          # steps whose grid set was empty
        self.n_degenerate = 0     # steps whose intersection was empty

    # -- core recursion ----------------------------------------------------
    def update(self, x) -> None:
        """Absorb one observation per row (``x`` has shape ``(n,)``)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape != (self.n,):
            raise ValueError(f"expected {self.n} observations, got {x.shape}")
        if np.any(x < -_EPS) or np.any(x > 1.0 + _EPS):
            raise ValueError("observations must lie in [0, 1]")
        i = self.t + 1                                    # step index, 1-based
        log_2_over_a = 2.0 * np.log(2.0 / self.alpha)     # per-row constant
        lam = np.sqrt(log_2_over_a / (self.s2_prev * i * np.log(i + 1.0)))
        self.lam_last = lam

        m = self.m
        # min(lam, 0.5/m) and min(lam, 0.5/(1-m)) keep every factor >= 0.5.
        cap_plus = np.minimum(lam[:, None], self.cap_hi[None, :])
        cap_minus = np.minimum(lam[:, None], self.cap_lo[None, :])
        d = x[:, None] - m[None, :]
        self.A += np.log1p(cap_plus * d)
        self.B += np.log1p(-cap_minus * d)

        self.S += x
        mu = (0.5 + self.S) / (i + 1.0)                   # mu_i
        self.D += (x - mu) ** 2
        self.s2_prev = (0.25 + self.D) / (i + 1.0)        # s2_i
        self.t = i

    # -- interval ----------------------------------------------------------
    def log_capital(self) -> np.ndarray:
        """``log K_t(m)``, shape ``(n, n_grid)``."""
        return np.logaddexp(self.A, self.B) - math.log(2.0)

    def raw_bounds(self):
        """Grid bounds of ``{m : K_t(m) < 1/a}`` before intersection."""
        neg = self.log_capital() < (-np.log(self.alpha))[:, None]
        has = neg.any(axis=1)
        first = neg.argmax(axis=1)
        last = self.n_grid - 1 - neg[:, ::-1].argmax(axis=1)
        L = np.where(has, self.m[first], np.nan)
        U = np.where(has, self.m[last], np.nan)
        return L, U, has

    def compute_interval(self) -> tuple:
        """Interval after intersecting with the previous one; cached in ``L``/``U``."""
        Lr, Ur, has = self.raw_bounds()
        first = np.isnan(self.L)
        L = np.where(has, Lr, np.where(first, 0.0, self.L))
        U = np.where(has, Ur, np.where(first, 1.0, self.U))
        self.n_empty += int((~has).sum())

        newL = np.maximum(L, np.where(first, -np.inf, self.L))
        newU = np.minimum(U, np.where(first, np.inf, self.U))
        bad = newL > newU
        self.n_degenerate += int(bad.sum())
        if bad.any():  # numerically empty intersection: keep this step's set
            newL = np.where(bad, L, newL)
            newU = np.where(bad, U, newU)
        self.L, self.U = newL, newU
        return self.L, self.U

    def is_covered(self, truth) -> np.ndarray:
        """Boolean per row: does the current interval contain ``truth``?

        ``False`` while the sequence has no interval yet (``NaN`` bounds).
        """
        truth = np.asarray(truth, dtype=np.float64).reshape(-1)
        with np.errstate(invalid="ignore"):
            return (self.L <= truth) & (truth <= self.U)

    # -- bookkeeping -------------------------------------------------------
    def subset(self, keep) -> "BettingCS":
        """Return a new instance holding only the rows selected by ``keep``."""
        keep = np.asarray(keep, dtype=bool)
        out = BettingCS.__new__(BettingCS)
        out.n = int(keep.sum())
        out.m = self.m
        out.n_grid = self.n_grid
        out.cap_hi = self.cap_hi
        out.cap_lo = self.cap_lo
        out.alpha = self.alpha[keep].copy()
        out.t = self.t[keep].copy()
        out.S = self.S[keep].copy()
        out.D = self.D[keep].copy()
        out.s2_prev = self.s2_prev[keep].copy()
        out.A = self.A[keep].copy()
        out.B = self.B[keep].copy()
        out.L = self.L[keep].copy()
        out.U = self.U[keep].copy()
        out.lam_last = self.lam_last[keep].copy()
        out.n_empty = self.n_empty
        out.n_degenerate = self.n_degenerate
        return out
