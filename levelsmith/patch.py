"""Solver-side repair: search for a one-edit change that makes a level valid.

Given a well-formed level that the judge rejected (unsolvable, trivial or
wrong difficulty), try every single edit of these kinds and keep those the
judge accepts for the requested band:

  move the exit to another floor cell     floor -> wall      wall -> floor
  floor -> coal      coal -> floor        clay -> floor      floor -> clay

The search is exhaustive over those edits and uses the same judge as
everything else, so a returned patch is a verified level.
"""

from __future__ import annotations

from .game import (
    CLAY, COAL, EXIT, FLOOR, WALL, MAX_CLAY, MAX_COALS, LevelError, parse_level,
)
from .judge import BANDS, OK, Verdict, judge


def _edits(rows: list[str]):
    """Yield (description, new_rows) for every candidate edit."""
    h, w = len(rows), len(rows[0])
    flat = "".join(rows)
    n_coals = flat.count(COAL)
    n_clay = flat.count(CLAY)
    exit_at = flat.index(EXIT)

    def build(changes: dict[int, str]) -> list[str]:
        cells = list(flat)
        for i, ch in changes.items():
            cells[i] = ch
        return ["".join(cells[r * w : (r + 1) * w]) for r in range(h)]

    def rc(i: int) -> str:
        return f"({i // w + 1},{i % w + 1})"

    for i, ch in enumerate(flat):
        if ch == FLOOR:
            yield f"move the exit from {rc(exit_at)} to {rc(i)}", build({exit_at: FLOOR, i: EXIT})
            yield f"turn the floor at {rc(i)} into a wall", build({i: WALL})
            if n_coals < MAX_COALS:
                yield f"add a coal at {rc(i)}", build({i: COAL})
            if n_clay < MAX_CLAY:
                yield f"turn the floor at {rc(i)} into wet clay", build({i: CLAY})
        elif ch == WALL:
            yield f"turn the wall at {rc(i)} into floor", build({i: FLOOR})
        elif ch == COAL:
            yield f"remove the coal at {rc(i)}", build({i: FLOOR})
        elif ch == CLAY:
            yield f"turn the clay at {rc(i)} into floor", build({i: FLOOR})


def find_patch(
    level_text: str, band: str, seen: set[str] | frozenset[str] = frozenset()
) -> dict | None:
    """Best single edit that makes `level_text` acceptable for `band`, or None.

    "Best" = optimal move count closest to the middle of the band; ties go to
    the first edit in scan order, so the result is deterministic.
    """
    try:
        level = parse_level(level_text)
    except LevelError:
        return None
    lo, hi = BANDS[band]
    mid = (lo + hi) / 2
    best: tuple[float, str, Verdict] | None = None
    n_tried = n_ok = 0
    for desc, rows in _edits(list(level.rows)):
        n_tried += 1
        v = judge("\n".join(rows), band, seen)
        if v.code != OK:
            continue
        n_ok += 1
        score = abs(v.solve.optimal_moves - mid)
        if best is None or score < best[0]:
            best = (score, desc, v)
    if best is None:
        return None
    _, desc, v = best
    out = v.summary()
    out.update(
        {"edit": desc, "canonical": v.canonical, "edits_tried": n_tried, "edits_valid": n_ok}
    )
    return out
