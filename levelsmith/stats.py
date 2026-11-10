"""Small statistics helpers (stdlib only)."""

from __future__ import annotations

import math
import random

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference p1 - p2 with Newcombe's hybrid score 95% interval."""
    if n1 == 0 or n2 == 0:
        return (0.0, -1.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (d, lo, hi)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation with average ranks for ties."""
    return pearson(_ranks(list(xs)), _ranks(list(ys)))


def spearman_ci(
    xs: list[float], ys: list[float], n_boot: int = 2000, seed: int = 0
) -> tuple[float | None, float | None, float | None]:
    """Spearman rho with a percentile bootstrap 95% interval."""
    rho = spearman(xs, ys)
    if rho is None:
        return (None, None, None)
    rng = random.Random(seed)
    n = len(xs)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        r = spearman([xs[i] for i in idx], [ys[i] for i in idx])
        if r is not None:
            vals.append(r)
    if len(vals) < n_boot // 2:
        return (rho, None, None)
    vals.sort()
    return (rho, vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


def mean(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def median(xs) -> float | None:
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2
