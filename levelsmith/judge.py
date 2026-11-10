"""Verdicts: turn level text into accept/reject plus structured feedback.

The same judge is used for LLM proposals and for the random baseline, so the
acceptance criteria are identical for both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .game import Level, LevelError, canonical, parse_level
from .solver import SolveResult, solve

# Difficulty bands: inclusive ranges of optimal move count.
BANDS: dict[str, tuple[int, int]] = {
    "easy": (3, 5),
    "medium": (6, 9),
    "hard": (10, 16),
}
TRIVIAL_MAX = 2  # solvable in this many moves or fewer counts as trivial

# Verdict codes, in the order they are checked.
OK = "ok"
MALFORMED = "malformed"
TOO_COMPLEX = "too_complex"
UNSOLVABLE = "unsolvable"
DUPLICATE = "duplicate"
TRIVIAL = "trivial"
WRONG_DIFFICULTY = "wrong_difficulty"


@dataclass
class Verdict:
    code: str
    detail: str = ""
    level: Level | None = None
    solve: SolveResult | None = None
    canonical: str | None = None
    data: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.code == OK

    @property
    def solvable(self) -> bool:
        return self.solve is not None and self.solve.solvable and self.solve.complete

    def summary(self) -> dict:
        """Compact JSON-safe record for logs."""
        out = {"code": self.code, "detail": self.detail}
        if self.level is not None:
            out["level"] = self.level.text
        if self.solve is not None and self.solve.complete:
            s = self.solve
            out["features"] = {
                "optimal_moves": s.optimal_moves,
                "solution": s.solution,
                "n_optimal_solutions": s.n_optimal_solutions,
                "states_to_solve": s.states_to_solve,
                "reachable_states": s.reachable_states,
                "avg_branching": round(s.avg_branching, 3),
                "dead_end_states": s.dead_end_states,
                "dead_end_ratio": round(s.dead_end_ratio, 3),
                "trap_moves": s.trap_moves,
            }
        return out


def judge(text: str, band: str, seen: set[str] | frozenset[str] = frozenset()) -> Verdict:
    """Check one proposed level against the requested band.

    `seen` holds canonical keys of levels already accepted in this run.
    """
    lo, hi = BANDS[band]
    try:
        level = parse_level(text)
    except LevelError as e:
        return Verdict(MALFORMED, e.message, data={"parse_code": e.code})

    res = solve(level)
    key = canonical(level)
    v = Verdict(OK, level=level, solve=res, canonical=key)

    if not res.complete:
        v.code = TOO_COMPLEX
        v.detail = (
            f"state space exceeds the solver cap ({res.reachable_states} states "
            "explored); use fewer clay tiles or coals"
        )
        return v
    if not res.solvable:
        v.code = UNSOLVABLE
        v.detail = "no sequence of moves solves this level"
        return v
    if key in seen:
        v.code = DUPLICATE
        v.detail = "this level (up to rotation/reflection) was already produced"
        return v
    n = res.optimal_moves
    if n <= TRIVIAL_MAX:
        v.code = TRIVIAL
        v.detail = f"solvable in {n} move(s): {res.solution}"
        return v
    if not lo <= n <= hi:
        v.code = WRONG_DIFFICULTY
        v.detail = (
            f"optimal solution is {n} moves ({res.solution}); requested "
            f"{band} = {lo}..{hi} moves"
        )
        v.data["direction"] = "too_easy" if n < lo else "too_hard"
        return v
    return v
