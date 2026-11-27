"""T4 -- command line smoke test on a small synthetic table.

Both subcommands must exit 0 and print a valid JSON object with the documented
keys.  The same tiny table is used for the table-format round trip.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from paired_cs import load_table

K = 6
N = 400
SEED = 12345
LABELS = [f"cand{k}" for k in range(K)]


def _make_table(path):
    rng = np.random.default_rng(SEED)
    p = np.array([0.20, 0.35, 0.55, 0.52, 0.51, 0.30])
    Y = (rng.random((K, N)) < p[:, None]).astype(np.int8)
    seeds = np.arange(1_000_000, 1_000_000 + N, dtype=np.int64)
    np.savez_compressed(path, Y=Y, seeds=seeds, candidate=np.array(LABELS, dtype="<U16"))
    return Y, seeds


def _run(*args):
    cmd = [sys.executable, "-m", "paired_cs", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return proc


def test_cli_select_and_certify(tmp_path):
    table = tmp_path / "synthetic.npz"
    Y, seeds = _make_table(table)

    tab = load_table(table)
    assert tab.K == K and tab.n == N
    assert tab.labels == LABELS
    assert np.array_equal(tab.seeds, seeds)
    assert np.array_equal(tab.Y, Y.astype(np.float64))

    sel = _run("select", "--table", str(table), "--key", "Y", "--delta", "0.05",
               "--alpha", "0.05", "--method", "pair", "--seed", "0", "--cap", "200000")
    assert sel.returncode == 0, sel.stderr[-2000:]
    sjson = json.loads(sel.stdout)
    for key in ("selected", "selectedLabel", "totalRollouts", "stopReason",
                "stopped", "rounds", "method", "seed", "nCandidates"):
        assert key in sjson, f"select output missing {key!r}: {sjson}"
    assert sjson["method"] == "S2_pair"
    assert isinstance(sjson["totalRollouts"], int) and sjson["totalRollouts"] > 0
    assert sjson["stopReason"] in ("rule", "single", "cap")
    if sjson["selected"] is not None:
        assert sjson["selectedLabel"] == LABELS[sjson["selected"]]
    assert sjson["nCandidates"] == K

    m = 1 * (K - 1)
    cert = _run("certify", "--reference", str(table), "--selected", "2",
                "--delta", "0.05", "--alpha", "0.05", "--n-comparisons", str(m))
    assert cert.returncode == 0, cert.stderr[-2000:]
    cjson = json.loads(cert.stdout)
    for key in ("selected", "selectedIndex", "regretLower", "regretUpper",
                "verdict", "nComparisons", "alphaPerComparison"):
        assert key in cjson, f"certify output missing {key!r}: {cjson}"
    assert cjson["selected"] == LABELS[2] and cjson["selectedIndex"] == 2
    assert cjson["verdict"] in ("correct", "wrong", "undecidable")
    assert 0.0 <= cjson["regretLower"] <= cjson["regretUpper"] <= 1.0
    assert abs(cjson["alphaPerComparison"] - 0.05 / m) < 1e-18

    # a label (not just an index) is accepted for --selected
    by_label = _run("certify", "--reference", str(table), "--selected", LABELS[2],
                    "--delta", "0.05", "--alpha", "0.05", "--n-comparisons", str(m))
    assert by_label.returncode == 0, by_label.stderr[-2000:]
    assert json.loads(by_label.stdout)["regretUpper"] == cjson["regretUpper"]

    # a bad reference exits non-zero with a JSON error on stderr
    bad = _run("certify", "--reference", str(tmp_path / "missing.npz"),
               "--selected", "0", "--n-comparisons", "1")
    assert bad.returncode != 0
    assert "error" in json.loads(bad.stderr)

    from conftest import record_detail
    record_detail("T4:cli", {
        "tableShape": [int(Y.shape[0]), int(Y.shape[1])],
        "select": {k: sjson[k] for k in ("method", "seed", "selected", "selectedLabel",
                                        "totalRollouts", "rounds", "stopReason")},
        "certify": {k: cjson[k] for k in ("selected", "selectedIndex", "regretLower",
                                          "regretUpper", "verdict", "nComparisons")},
        "labelLookupConsistent": True,
        "errorPathExitCode": int(bad.returncode),
    })
