import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from residual_rl import claims  # noqa: E402


@pytest.fixture(scope="module")
def summary():
    return claims.summarise(claims.load(ROOT / "results" / "eval.csv.gz"))


def test_claim_counts_match_the_recorded_run(summary):
    comps = summary["comparisons"]
    assert len(comps) == 160 and sum(c["resolved"] for c in comps) == 102


def test_residual_vs_impedance_seven_of_eight(summary):
    rows = [c for c in summary["comparisons"]
            if c["metric"] == "e_rms_ss" and (c["a"], c["b"]) == ("residual@1M", "impedance")]
    assert sum(c["resolved"] for c in rows) == 7
    c2t1 = next(c for c in rows if (c["condition"], c["trajectory"]) == ("C2", "T1"))
    assert c2t1["ci"][1] < 0 and not c2t1["resolved"]  # interval excludes 0, but the winner terminates more


def test_decide_rule():
    assert claims.decide(-0.2, -0.1, 0.0, 0.0, True) == ("a", True)
    assert claims.decide(-0.2, -0.1, 0.1, 0.0, True) == ("a", False)
    assert claims.decide(-0.2, 0.1, 0.0, 0.0, True) == (None, False)
    assert claims.decide(-0.2, -0.1, 0.0, 0.0, False) == ("a", False)


def test_baselines_reproduce_recorded_nominal_error():
    pytest.importorskip("gymnasium_robotics")
    from residual_rl.controllers import JointPD
    from residual_rl.residual import CONFIG, build_env, evaluate_episode

    imp = evaluate_episode(None, "residual", "C0", "T1", 10000)  # zero residual = impedance alone
    assert imp["e_rms_ss"] == pytest.approx(0.034718, abs=1e-6)
    core = build_env("direct", "C0", "T1").unwrapped
    pd = JointPD(**CONFIG["joint_pd"])
    obs, _ = core.reset(seed=10000)
    errs = []
    for k in range(1, 401):
        obs, _, term, trunc, info = core.step(pd(obs, core))
        errs.append((k, info["e_norm"]))
    ss = np.array([v for kk, v in errs if kk >= 50])
    assert np.sqrt(np.mean(ss**2)) == pytest.approx(0.0065135, abs=1e-6)
