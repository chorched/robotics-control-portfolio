"""Shared paths: the shipped candidate x initial-state tables and reference values."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
POOL = DATA / "selection_pool.npz"  # 78 candidates x 3000 initial states (run A4p)
REFERENCE = DATA / "reference.npz"  # 78 candidates x up to 10000 states, with missing cells (A4r + A5e)
EXPECTED = json.loads((DATA / "expected.json").read_text(encoding="utf-8"))


def record_detail(key: str, payload: dict) -> None:  # kept for the copied tests; no-op here
    return None
