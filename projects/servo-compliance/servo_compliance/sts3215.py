"""Decoding Feetech STS3215 telemetry registers.

Two registers are easy to misread:

* ``Present_Current`` (address 69) is **sign-magnitude**: bit 15 is the direction,
  bits 0-14 the magnitude in units of 6.5 mA. Reading it as an unsigned 16-bit
  integer (as LeRobot does) turns every negative current into a value above 32768.
* ``Present_Load`` (address 60) is the **output PWM duty cycle**, bit 10 = direction,
  bits 0-9 = magnitude in 0.1 % steps. It is a commanded quantity, not a torque
  measurement; any torque reading must be calibrated against a model (see
  ``calibration.py``), and on this hardware it tracks actuator torque better than the
  current register does (R^2 0.56-0.83 vs 0.01-0.71 per joint on public 0 g data).
"""

from __future__ import annotations

import numpy as np

CURRENT_LSB_A = 6.5e-3
LOAD_LSB = 0.001  # fraction of full duty per count


def decode_current(raw) -> np.ndarray:
    """Raw 16-bit register -> signed current in amperes."""
    raw = np.asarray(raw, dtype=np.int64)
    mag = raw & 0x7FFF
    sign = np.where(raw & 0x8000, -1, 1)
    return sign * mag * CURRENT_LSB_A


def decode_load(raw) -> np.ndarray:
    """Raw register -> signed PWM duty in [-1, 1]."""
    raw = np.asarray(raw, dtype=np.int64)
    mag = raw & 0x3FF
    sign = np.where(raw & 0x400, -1, 1)
    return sign * mag * LOAD_LSB


def unsigned_misread_current(raw) -> np.ndarray:
    """What an unsigned read gives - kept to show the failure in tests and docs."""
    return np.asarray(raw, dtype=np.int64) * CURRENT_LSB_A
