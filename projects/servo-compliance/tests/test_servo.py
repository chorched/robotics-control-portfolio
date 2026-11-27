import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from servo_compliance import admittance as A  # noqa: E402
from servo_compliance import sts3215  # noqa: E402
from servo_compliance.threshold_transfer import analyse, auroc  # noqa: E402


@pytest.mark.parametrize("d", [0, 1, 2, 3])
def test_closed_form_boundary_matches_roots(d):
    gm = A.g_max(d)
    assert A.spectral_radius(gm * 0.99, d) < 1 < A.spectral_radius(gm * 1.01, d)


def test_known_limits():
    assert A.g_max(0) == pytest.approx(2.0)
    assert A.g_max(1) == pytest.approx(1.0)
    assert A.g_max(2) == pytest.approx((5**0.5 - 1) / 2)


def test_press_stable_below_and_limit_cycles_above():
    calm = A.simulate_press(0.5, 1)["f"][-150:]
    wild = A.simulate_press(1.2, 1)["f"][-150:]
    assert abs(calm.mean() - 2.0) <= 0.1 + 1e-9 and calm.std() < 0.01
    assert wild.std() > 0.5


def test_current_is_sign_magnitude():
    raw = np.array([0x0000 | 10, 0x8000 | 10])
    assert np.allclose(sts3215.decode_current(raw), [0.065, -0.065])
    assert sts3215.unsigned_misread_current(raw)[1] > 200  # the unsigned read: > 200 A


def test_load_direction_bit():
    assert np.allclose(sts3215.decode_load([500, 0x400 | 500]), [0.5, -0.5])


def test_auroc():
    assert auroc(np.array([2.0, 3.0]), np.array([1.0])) == 1.0
    assert auroc(np.array([1.0]), np.array([1.0])) == 0.5


def test_threshold_transfer_reproduces_registered_gate():
    res = analyse(json.loads((ROOT / "results" / "load_scores.json").read_text()))
    assert res["gate"]["passing"] == 2 and res["gate"]["result"] == "FAIL"
    omx = res["directions"][0]
    assert omx["b4"]["recal"]["R2_k2"] == pytest.approx(0.938, abs=1e-3)
