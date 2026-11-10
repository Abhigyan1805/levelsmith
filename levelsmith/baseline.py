"""Procedural baseline: random levels filtered by the same solver and judge.

No search guidance and no knowledge of the requested band beyond the filter:
sample a random grid, judge it, repeat until one is accepted or the attempt
cap is reached.
"""

from __future__ import annotations

import random
import time

from .game import CLAY, COAL, EXIT, FLOOR, START, WALL, MAX_CLAY, MAX_COALS
from .judge import Verdict, judge


def random_level_text(rng: random.Random, width: int, height: int) -> str:
    n = width * height
    cells = [FLOOR] * n
    wall_frac = rng.uniform(0.08, 0.30)
    for i in range(n):
        if rng.random() < wall_frac:
            cells[i] = WALL
    free = [i for i in range(n)]
    rng.shuffle(free)
    n_coals = rng.randint(0, min(MAX_COALS, 4))
    n_clay = rng.randint(0, min(MAX_CLAY, 4))
    picks = iter(free)
    cells[next(picks)] = START
    cells[next(picks)] = EXIT
    for _ in range(n_coals):
        cells[next(picks)] = COAL
    for _ in range(n_clay):
        cells[next(picks)] = CLAY
    return "\n".join(
        "".join(cells[r * width : (r + 1) * width]) for r in range(height)
    )


def generate(
    band: str,
    width: int,
    height: int,
    seed: int,
    seen: set[str],
    max_attempts: int = 2000,
) -> dict:
    """Rejection-sample one level for `band`. Returns a JSON-safe record."""
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    t0 = time.perf_counter()
    accepted: Verdict | None = None
    attempts = 0
    for attempts in range(1, max_attempts + 1):
        v = judge(random_level_text(rng, width, height), band, seen)
        counts[v.code] = counts.get(v.code, 0) + 1
        if v.accepted:
            accepted = v
            break
    elapsed = time.perf_counter() - t0
    rec = {
        "band": band,
        "width": width,
        "height": height,
        "seed": seed,
        "attempts": attempts,
        "verdict_counts": counts,
        "seconds": round(elapsed, 4),
        "accepted": accepted is not None,
    }
    if accepted is not None:
        rec.update(accepted.summary())
        rec["canonical"] = accepted.canonical
    return rec
