"""Package metadata and public API for :mod:`paired_cs`.

``paired_cs`` packages the sequential selection programs and the paired
(same-initial-state) certification procedure that were validated in the A2t /
A4p / A5p batches, so that they can be reused on other candidate tables without
copying scripts.

Public API
----------
``BettingCS``        batched hedged-betting confidence sequence for means in [0,1]
``grid_points``      midpoint grid used by the confidence sequence
``Table``            candidate x initial-state outcome table (+ ``npz`` loader)
``load_table``       read ``Y`` / ``seeds`` / ``evaluated`` / ``candidate`` from npz
``run_selection``    one replication of ``S2_marg`` or ``S2_pair``
``run_replications`` several replications, optionally across processes
``certify``          paired regret interval + three-valued verdict for one candidate
``certify_many``     the same for several candidates under one joint level
"""

from __future__ import annotations

from .certify import certify, certify_many, diff_interval, pair_interval, verdict_of
from .cs import N_GRID, BettingCS, grid_points
from .select import (
    METHODS,
    Table,
    load_table,
    make_index_sampler,
    run_replications,
    run_selection,
)

__version__ = "1.0.1"

__all__ = [
    "__version__",
    "N_GRID",
    "BettingCS",
    "grid_points",
    "METHODS",
    "Table",
    "load_table",
    "make_index_sampler",
    "run_selection",
    "run_replications",
    "certify",
    "certify_many",
    "diff_interval",
    "pair_interval",
    "verdict_of",
]
