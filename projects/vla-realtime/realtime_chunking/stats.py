"""Interval estimates used for every success-rate comparison in this project."""

from __future__ import annotations

import math

Z95 = 1.959963985


def wilson95(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    den = 1.0 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / den
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Newcombe (method 10) hybrid score interval for p1 - p2. Returns (diff, lo, hi)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson95(k1, n1)
    l2, u2 = wilson95(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, max(-1.0, lo), min(1.0, hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
