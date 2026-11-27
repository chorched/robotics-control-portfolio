"""T2 - paired sequential selection reproduces the recorded runs bit for bit.

The fast test replays one replication (seed 7, the shortest). Set RUN_SLOW=1 to
replay all 20 replications (a few minutes on a multi-core machine) and compare the
median rollout count, every per-seed count and every selected candidate.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import statistics
import subprocess
import sys

import pytest

import conftest as C

REF = C.EXPECTED["selection"]["S2_pair"]


def run_select(seed: int) -> dict:
    cmd = [sys.executable, "-m", "paired_cs", "select", "--table", str(C.POOL), "--key", "Y", "--delta", "0.05",
           "--alpha", "0.05", "--method", "pair", "--seed", str(seed), "--cap", "200000"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=C.ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


def test_single_replication_matches():
    r = run_select(7)
    assert r["stopReason"] == "rule"
    assert r["totalRollouts"] == REF["rollouts"][7]
    assert r["selectedLabel"] == REF["selected"][7]


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1 for the 20-replication check")
def test_all_replications_match():
    with cf.ThreadPoolExecutor(max_workers=20) as ex:
        reps = sorted(ex.map(run_select, range(20)), key=lambda r: r["seed"])
    rollouts = [r["totalRollouts"] for r in reps]
    assert statistics.median(rollouts) == REF["medianRollouts"] == 119678.5
    assert rollouts == REF["rollouts"]
    assert [r["selectedLabel"] for r in reps] == REF["selected"]
