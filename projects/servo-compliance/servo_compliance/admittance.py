r"""Pure-damping admittance on a position-controlled arm, with sensing delay.

Control law (vertical axis, force measured positive when pressing harder)::

    p_r[k+1] = p_r[k] - (dt / D_a) * S_delta( f_hat[k-d] - f_d )

``S_delta`` is a dead-zone of half-width ``delta``. Against a contact of effective
stiffness ``k_eff`` (arm and environment springs in series) and with a stiff inner
position loop, f[k] = k_eff * (penetration), so the force error ``e = f - f_d``
obeys, outside the dead zone,

    e[k+1] = e[k] - g * e[k-d],        g = k_eff * dt / D_a

Its characteristic polynomial is  z^{d+1} - z^d + g = 0. The loop is stable iff all
roots lie inside the unit circle, which happens for

    0 < g < g_max(d) = 2 sin( pi / (2 (2d + 1)) )

i.e. g < 2 without delay, g < 1 with one sample of delay, g < 0.618 with two.

Steady state: once the loop settles the dead zone holds |f_hat - f_d| <= delta, and
with estimator bias e_ss = f - f_hat the true force satisfies
|f_ss - f_d| <= delta + |e_ss| (triangle inequality).
"""

from __future__ import annotations

import math

import numpy as np


def gain(k_eff: float, dt: float, damping: float) -> float:
    return k_eff * dt / damping


def damping_for_gain(g: float, k_eff: float, dt: float) -> float:
    return k_eff * dt / g


def char_poly(g: float, d: int) -> np.ndarray:
    """Coefficients of z^{d+1} - z^d + g (highest power first)."""
    c = np.zeros(d + 2)
    c[0], c[1], c[-1] = 1.0, -1.0, g
    if d == 0:
        c = np.array([1.0, g - 1.0])
    return c


def spectral_radius(g: float, d: int) -> float:
    return float(np.max(np.abs(np.roots(char_poly(g, d)))))


def g_max(d: int) -> float:
    """Closed-form stability limit of the delayed integrator loop."""
    return 2.0 * math.sin(math.pi / (2.0 * (2 * d + 1)))


def dead_zone(x: float, delta: float) -> float:
    return math.copysign(max(abs(x) - delta, 0.0), x)


def simulate_press(
    g: float,
    d: int,
    k_eff: float = 300.0,
    dt: float = 0.02,
    f_d: float = 2.0,
    delta: float = 0.1,
    t_end: float = 8.0,
    travel: tuple[float, float] = (-0.008, 0.012),
    noise_n: float = 0.0,
    seed: int = 0,
) -> dict:
    """Idealised 1-DoF press: reference = penetration, force = k_eff * max(p, 0).

    ``travel`` clamps the reference (retract, press) as on the real arm, which turns
    instability into a bounded limit cycle instead of a runaway.
    """
    rng = np.random.default_rng(seed)
    D_a = damping_for_gain(g, k_eff, dt)
    n = int(round(t_end / dt))
    p = f_d / k_eff * 0.5  # start in contact, below the target force
    hist: list[float] = []
    f = np.empty(n)
    for k in range(n):
        f_true = k_eff * max(p, 0.0)
        f[k] = f_true
        hist.append(f_true + (rng.normal(0, noise_n) if noise_n else 0.0))
        f_meas = hist[-1 - d] if len(hist) > d else hist[0]
        p = p - (dt / D_a) * dead_zone(f_meas - f_d, delta)  # f < f_d -> press deeper
        p = min(max(p, travel[0]), travel[1])
    return {"t": np.arange(n) * dt, "f": f, "D_a": D_a}
