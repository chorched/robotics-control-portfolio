"""T5 -- a rival with no shared initial state must not be dropped from the bound.

The synthetic reference table has 4 candidates.  ``cand0`` is selected and is
scored on the seeds ``0..N-1``; ``cand1``/``cand2`` are scored on the same
seeds, while ``cand3`` is *only* scored on a disjoint seed range, so the pair
``(cand3, cand0)`` has ``n == 0``.

Required behaviour (the bug fixed in 1.0.1): ``cand3`` enters the regret with
the trivial interval ``[-1, 1]`` instead of being filtered out of
``max_j u_j``.  Hence ``regretUpper == 1.0`` (not a small number that would let
the run be certified ``correct``), the verdict is ``undecidable``, and
``missingRivals`` names ``cand3``.  Once ``cand3`` is also evaluated on the
shared seeds the certification becomes decidable again, and the bounds of the
other rivals are unaffected.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from paired_cs import certify, load_table

from conftest import record_detail

K = 4
N = 2000                      # shared initial states
M = 500                       # disjoint states for the "missing" rival
SEED = 777
DELTA = 0.05
ALPHA = 0.05
LABELS = [f"cand{k}" for k in range(K)]
RATES = np.array([0.60, 0.50, 0.45, 0.20])
SHARED_SEEDS = np.arange(N, dtype=np.int64)
DISJOINT_SEEDS = np.arange(900_000, 900_000 + M, dtype=np.int64)


def _build(path, rival3_shared: bool):
    """Write the 4-candidate table; ``cand3`` is disjoint unless told otherwise."""
    rng = np.random.default_rng(SEED)
    seeds = np.union1d(SHARED_SEEDS, DISJOINT_SEEDS)
    Y = np.zeros((K, seeds.size), dtype=np.int8)
    evaluated = np.zeros((K, seeds.size), dtype=np.int8)
    shared_cols = np.searchsorted(seeds, SHARED_SEEDS)
    disj_cols = np.searchsorted(seeds, DISJOINT_SEEDS)
    for k in range(3):
        Y[k, shared_cols] = (rng.random(N) < RATES[k]).astype(np.int8)
        evaluated[k, shared_cols] = 1
    Y[3, disj_cols] = (rng.random(M) < RATES[3]).astype(np.int8)
    evaluated[3, disj_cols] = 1
    if rival3_shared:
        Y[3, shared_cols] = (rng.random(N) < RATES[3]).astype(np.int8)
        evaluated[3, shared_cols] = 1
    np.savez_compressed(path, Y=Y, seeds=seeds, evaluated=evaluated,
                        candidate=np.array(LABELS, dtype="<U16"))
    return path


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    d = tmp_path_factory.mktemp("t5")
    return {
        "missing": _build(d / "missing.npz", rival3_shared=False),
        "shared": _build(d / "shared.npz", rival3_shared=True),
    }


def test_missing_rival_makes_certification_undecidable(tables):
    tab = load_table(tables["missing"])
    res = certify(tab, 0, delta=DELTA, alpha=ALPHA, n_comparisons=K - 1)

    by_index = {p["rivalIndex"]: p for p in res["pairs"]}
    # cand3 shares nothing with cand0 -> trivial interval [-1, 1]
    assert by_index[3]["n"] == 0
    assert by_index[3]["l"] == pytest.approx(-1.0)
    assert by_index[3]["u"] == pytest.approx(1.0)

    # the un-shared rival caps the regret at 1 -> undecidable, not "correct"
    assert res["regretUpper"] == pytest.approx(1.0)
    assert res["regretLower"] == pytest.approx(0.0)
    assert res["verdict"] == "undecidable"

    assert res["nMissingRivals"] == 1
    assert res["missingRivals"] == [{"rival": "cand3", "rivalIndex": 3}]
    assert res["decisiveRival"] == "cand3"
    assert res["decisiveRivalIndex"] == 3
    assert res["nDecisive"] == 0

    record_detail("T5:missing", {
        "nCandidates": K, "sharedSeeds": int(N), "disjointSeeds": int(M),
        "verdict": res["verdict"], "regretLower": res["regretLower"],
        "regretUpper": res["regretUpper"],
        "missingRivals": res["missingRivals"],
        "nMissingRivals": res["nMissingRivals"],
        "decisiveRival": res["decisiveRival"],
        "nDecisive": res["nDecisive"],
        "pairsN": {str(k): v["n"] for k, v in by_index.items()},
    })


def test_shared_states_restore_decidable_verdict(tables):
    tab = load_table(tables["shared"])
    res = certify(tab, 0, delta=DELTA, alpha=ALPHA, n_comparisons=K - 1)

    assert res["missingRivals"] == []
    assert res["nMissingRivals"] == 0
    assert all(p["n"] == N for p in res["pairs"])
    assert res["verdict"] == "correct"
    assert res["regretUpper"] <= DELTA
    assert res["regretUpper"] < 1.0

    record_detail("T5:shared", {
        "verdict": res["verdict"], "regretLower": res["regretLower"],
        "regretUpper": res["regretUpper"],
        "missingRivals": res["missingRivals"],
        "pairsN": [p["n"] for p in res["pairs"]],
    })


def test_all_rivals_missing_still_raises(tmp_path):
    """Only the all-rivals-missing case keeps raising ValueError."""
    rng = np.random.default_rng(SEED + 1)
    seeds = np.concatenate([np.arange(0, 100), np.arange(5000, 5100), np.arange(9000, 9100)])
    Y = (rng.random((3, seeds.size)) < 0.5).astype(np.int8)
    evaluated = np.zeros((3, seeds.size), dtype=np.int8)
    evaluated[0, 0:100] = 1
    evaluated[1, 100:200] = 1
    evaluated[2, 200:300] = 1
    path = tmp_path / "all_missing.npz"
    np.savez_compressed(path, Y=Y, seeds=seeds.astype(np.int64), evaluated=evaluated,
                        candidate=np.array(["a", "b", "c"], dtype="<U8"))
    tab = load_table(path)
    with pytest.raises(ValueError, match="no initial state in common"):
        certify(tab, 0, delta=DELTA, alpha=ALPHA, n_comparisons=2)


def test_cli_reports_missing_rivals(tables):
    cmd = [sys.executable, "-m", "paired_cs", "certify",
           "--reference", str(tables["missing"]), "--selected", "0",
           "--delta", str(DELTA), "--alpha", str(ALPHA),
           "--n-comparisons", str(K - 1)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout)
    assert out["missingRivals"] == [{"rival": "cand3", "rivalIndex": 3}]
    assert out["nMissingRivals"] == 1
    assert out["verdict"] == "undecidable"
    assert out["regretUpper"] == pytest.approx(1.0)
    record_detail("T5:cli", {
        "verdict": out["verdict"], "regretUpper": out["regretUpper"],
        "missingRivals": out["missingRivals"],
        "nMissingRivals": out["nMissingRivals"],
    })
