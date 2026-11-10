"""The level-design agent: propose -> verify with the solver -> repair.

Two modes share everything except what the model sees after a rejection:
  repair : the conversation continues with structured solver feedback.
  blind  : a fresh request with no feedback (resampling control).
Round 1 is identical in both modes and is the "one-shot" result.
"""

from __future__ import annotations

import re
import time

from .game import DIRS, Level, MAX_CLAY, MAX_COALS, make_mover
from .judge import (
    BANDS,
    DUPLICATE,
    MALFORMED,
    TOO_COMPLEX,
    TRIVIAL,
    UNSOLVABLE,
    WRONG_DIFFICULTY,
    Verdict,
    judge,
)
from .llm import BudgetExceeded, LLMClient
from .patch import find_patch

MAX_RESPONSE_CHARS = 6000  # anything longer is rejected before parsing

SYSTEM_PROMPT = f"""You design levels for Kiln Slide, a grid puzzle.

RULES
- The ember moves U, D, L or R. A move slides it in a straight line until the
  next cell is a wall or the edge of the grid. It cannot stop earlier.
- A move in a direction where the ember is already blocked is not allowed.
- The ember collects every coal (*) it passes over or stops on.
- Wet clay (~) can be crossed or rested on. As soon as the ember leaves a clay
  tile, that tile bakes into a wall for the rest of the level.
- The level is solved when all coals are collected and the ember comes to
  REST on the exit (E). Sliding over the exit without stopping does nothing.

LEVEL FORMAT
- A rectangular text grid, one row per line, no spaces.
- Characters: # wall, . floor, S start (exactly one), E exit (exactly one),
  * coal (0 to {MAX_COALS}), ~ wet clay (0 to {MAX_CLAY}).
- Cells outside the grid are walls, so a border of # is optional.
- Coordinates in feedback are (row, column), both starting at 1 from the top
  left.

EXAMPLE 1 (5 wide, 5 tall, optimal 3 moves: D R U)
.....
.S#..
..E.#
#.*..
...#.
D: from (2,2) slide to (5,2). R: stop at (5,3) because (5,4) is a wall.
U: pass the coal at (4,3), stop on E at (3,3) because (2,3) is a wall.

EXAMPLE 2 (6 wide, 5 tall, optimal 5 moves: U L R D L)
..~.*.
......
E....S
.#....
......
U: to (1,6). L: collect the coal, cross the clay at (1,3), stop at (1,1);
the clay bakes into a wall. R: stop at (1,2) against the baked clay.
D: stop at (3,2) because (4,2) is a wall. L: stop on E at (3,1).

DIFFICULTY is the length of the shortest solution (optimal move count):
easy = {BANDS['easy'][0]}-{BANDS['easy'][1]} moves, medium = {BANDS['medium'][0]}-{BANDS['medium'][1]} moves, hard = {BANDS['hard'][0]}-{BANDS['hard'][1]} moves.
Every level you propose is checked by an exact solver, which finds the true
shortest solution. A level only counts if the solver's optimal move count is
inside the requested range, so a shortcut you did not notice makes it fail.

OUTPUT
Reply with exactly two things and nothing else: the level inside one fenced
code block, then one line with the solution you intend.
```
<grid rows>
```
solution: <letters from U D L R, no spaces>
No explanation, no reasoning text, no second attempt. Do not copy the examples."""


def request_prompt(band: str, width: int, height: int) -> str:
    lo, hi = BANDS[band]
    return (
        f"Design one new {band} level: exactly {width} columns wide and "
        f"{height} rows tall, with an optimal solution of {lo} to {hi} moves."
    )


_FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)\n?```", re.S)
_SOLUTION = re.compile(r"solution\s*[:=]\s*`?([UDLR ,]+)`?", re.I)


def extract(text: str) -> tuple[str | None, str | None, str]:
    """Pull (level_text, claimed_solution, error) out of a model reply."""
    if len(text) > MAX_RESPONSE_CHARS:
        return None, None, f"reply too long ({len(text)} characters)"
    blocks = _FENCE.findall(text)
    if not blocks:
        return None, None, "no fenced code block found in the reply"
    level_text = blocks[-1]  # if the model revised itself, the last block is its answer
    m = None
    for m in _SOLUTION.finditer(text):
        pass  # keep the last match
    claimed = re.sub(r"[ ,]", "", m.group(1)).upper() if m else None
    return level_text, claimed or None, ""


def rc(level: Level, cell: int) -> str:
    r, c = divmod(cell, level.width)
    return f"({r + 1},{c + 1})"


def trace_claimed(level: Level, claimed: str | None, limit: int = 20) -> str:
    """Simulate the model's own solution and say where it goes wrong."""
    if not claimed:
        return ""
    move = make_mover(level)
    state = level.initial_state()
    lines = []
    for i, d in enumerate(claimed[:limit], 1):
        nxt = move(state, d) if d in DIRS else None
        if nxt is None:
            lines.append(
                f"  move {i} ({d}): blocked, the ember at {rc(level, state[0])} cannot move that way"
            )
            break
        state = nxt
        lines.append(
            f"  move {i} ({d}): rests at {rc(level, state[0])}, "
            f"coals {state[1].bit_count()}/{len(level.coals)}"
        )
        if level.is_goal(state):
            lines.append("  -> solved here")
            break
    else:
        if not level.is_goal(state):
            lines.append(
                f"  -> ends at {rc(level, state[0])}, not solved "
                f"(exit is at {rc(level, level.exit)})"
            )
    return "Simulating your stated solution:\n" + "\n".join(lines)


def feedback(v: Verdict, band: str, width: int, height: int, claimed: str | None) -> str:
    """Structured feedback for one rejected proposal."""
    lo, hi = BANDS[band]
    head = f"REJECTED: {v.code}."
    tail = "Reply with a corrected level in the same output format."
    if v.code == MALFORMED:
        return f"{head} {v.detail}. The level must be exactly {width} wide and {height} tall. {tail}"
    lvl, s = v.level, v.solve
    parts = [head, v.detail + "."]
    if v.code == UNSOLVABLE:
        t = trace_claimed(lvl, claimed)
        if t:
            parts.append(t)
        if not s.exit_restable:
            parts.append(
                f"The ember can never come to rest on the exit at {rc(lvl, lvl.exit)}: "
                "from every reachable position it slides past it or cannot reach it. "
                "Put a wall directly behind the exit along some approach line."
            )
        missing = [rc(lvl, c) for c, ok in zip(lvl.coals, s.coals_collectable) if not ok]
        if missing:
            parts.append("Coals that can never be collected: " + ", ".join(missing) + ".")
        elif lvl.coals and s.max_coals_collected < len(lvl.coals):
            parts.append(
                f"Each coal is reachable, but at most {s.max_coals_collected} of "
                f"{len(lvl.coals)} can be collected in one run."
            )
        elif s.exit_restable:
            parts.append(
                "The exit can be rested on and the coals can be collected, but not "
                "in an order that still allows reaching the exit afterwards."
            )
        rests = [rc(lvl, c) for c in s.rest_cells[:40]]
        parts.append(
            f"Cells where the ember can come to rest ({len(s.rest_cells)}): " + " ".join(rests)
        )
    elif v.code in (TRIVIAL, WRONG_DIFFICULTY):
        n = s.optimal_moves
        if n < lo:
            parts.append(
                f"Too short by at least {lo - n} move(s). The solver's shortest solution "
                f"is {s.solution}. Block that shortcut or add coals that force detours."
            )
        else:
            parts.append(
                f"Too long by at least {n - hi} move(s). Remove coals or walls so a "
                "shorter route exists."
            )
        if claimed and claimed != s.solution:
            parts.append(f"(Your stated solution was {claimed}, {len(claimed)} moves.)")
    elif v.code == DUPLICATE:
        parts.append("Change the layout substantially.")
    elif v.code == TOO_COMPLEX:
        pass
    parts.append(tail)
    return "\n".join(parts)


def check(text: str, band: str, width: int, height: int, seen) -> tuple[Verdict, str | None]:
    """Extract, parse, size-check and judge one model reply."""
    level_text, claimed, err = extract(text)
    if level_text is None:
        return Verdict(MALFORMED, err, data={"parse_code": "no_level"}), claimed
    v = judge(level_text, band, seen)
    if v.level is not None and (v.level.width, v.level.height) != (width, height):
        got = f"{v.level.width}x{v.level.height}"
        v = Verdict(
            MALFORMED,
            f"wrong size: got {got} (width x height), requested {width}x{height}",
            data={"parse_code": "wrong_size", "level": v.level.text},
        )
    return v, claimed


def run_request(
    client: LLMClient,
    mode: str,
    band: str,
    width: int,
    height: int,
    seen: set[str],
    max_rounds: int = 4,
    temperature: float = 1.0,
    tag: dict | None = None,
    lock=None,
    patch: bool = False,
) -> dict:
    """Run one level request to acceptance or exhaustion. Returns a JSON-safe record.

    With `patch`, a rejected proposal is first handed to the solver's one-edit
    patch search (levelsmith.patch); if that finds a valid level the request
    ends there without another model call.
    """
    assert mode in ("repair", "blind")
    first = {"role": "user", "content": request_prompt(band, width, height)}
    messages = [first]
    rounds = []
    accepted = None
    patched = None
    stop_reason = "max_rounds"
    t0 = time.perf_counter()
    for rnd in range(1, max_rounds + 1):
        try:
            resp = client.complete(
                SYSTEM_PROMPT, messages, temperature, tag={**(tag or {}), "round": rnd}
            )
        except BudgetExceeded as e:
            stop_reason = f"budget: {e}"
            break
        rec = {
            "round": rnd,
            "llm_ok": resp.ok,
            "latency_s": resp.latency_s,
            "prompt_tokens": resp.prompt_tokens,
            "output_tokens": resp.output_tokens,
            "reasoning_tokens": resp.reasoning_tokens,
            "cost_usd": resp.cost_usd,
            "reply": resp.text[:MAX_RESPONSE_CHARS],
        }
        if not resp.ok:
            rec["verdict"] = {"code": "llm_error", "detail": resp.error}
            rounds.append(rec)
            stop_reason = "llm_error"
            break
        if lock:
            with lock:
                v, claimed = check(resp.text, band, width, height, seen)
                if v.accepted:
                    seen.add(v.canonical)
        else:
            v, claimed = check(resp.text, band, width, height, seen)
            if v.accepted:
                seen.add(v.canonical)
        rec["claimed_solution"] = claimed
        rec["verdict"] = v.summary()
        if v.code == MALFORMED:
            rec["verdict"]["parse_code"] = v.data.get("parse_code")
        rounds.append(rec)
        if v.accepted:
            accepted = v
            stop_reason = "accepted"
            break
        if patch and v.level is not None and v.code != MALFORMED:
            fixed = find_patch(v.level.text, band, seen)
            if fixed is not None:
                rec["patch"] = fixed["edit"]
                patched = fixed
                if lock:
                    with lock:
                        seen.add(fixed["canonical"])
                else:
                    seen.add(fixed["canonical"])
                stop_reason = "accepted"
                break
        if mode == "repair":
            fb = feedback(v, band, width, height, claimed)
            rec["feedback"] = fb
            messages = messages + [
                {"role": "assistant", "content": resp.text[:MAX_RESPONSE_CHARS]},
                {"role": "user", "content": fb},
            ]
        # blind mode: same single-message request again, nothing carried over
    done = accepted is not None or patched is not None
    out = {
        "mode": mode,
        "band": band,
        "width": width,
        "height": height,
        "accepted": done,
        "accepted_round": len(rounds) if done else None,
        "patched": patched is not None,
        "stop_reason": stop_reason,
        "n_calls": len(rounds),
        "seconds": round(time.perf_counter() - t0, 3),
        "cost_usd": sum(r["cost_usd"] for r in rounds),
        "rounds": rounds,
    }
    if accepted is not None:
        out["level"] = accepted.level.text
        out["canonical"] = accepted.canonical
        out["features"] = accepted.summary()["features"]
    elif patched is not None:
        out["level"] = patched["level"]
        out["canonical"] = patched["canonical"]
        out["features"] = patched["features"]
    return out
