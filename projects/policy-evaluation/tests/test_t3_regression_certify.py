"""T3 - paired certification reproduces the recorded regret bounds (run A5p)."""

from __future__ import annotations

import numpy as np

import conftest as C
from paired_cs import certify, load_table

REF = C.EXPECTED["certify"]


def test_reference_table_seed_counts():
    with np.load(C.REFERENCE) as z:
        assert z["evaluated"].sum(axis=1).tolist() == REF["perCandidateSeedCount"]


def test_certify_reproduces_regret_bounds():
    tab = load_table(C.REFERENCE)
    for ref in REF["perSelected"]:
        res = certify(tab, ref["index"], delta=REF["delta"], alpha=REF["alphaTotal"],
                      n_comparisons=REF["nComparisons"])
        assert abs(res["regretLower"] - ref["regretLower"]) <= 1e-9
        assert abs(res["regretUpper"] - ref["regretUpper"]) <= 1e-9
        assert res["verdict"] == ref["verdict"] == "correct"
        assert res["decisiveRival"] == ref["decisiveRival"]
        assert res["nDecisive"] == ref["nDecisive"]
