"""Kiln Slide: rules, level format and strict parser.

Rules (the whole game):
  1. Each move slides the ember in one of four directions until a wall stops
     it. It collects every coal it passes over, and every wet clay tile it
     leaves bakes into a wall.
  2. The level is solved when every coal is collected and the ember comes to
     rest on the exit.

Level format: a rectangular text grid, one row per line.
  #  wall            .  floor
  S  ember start     E  exit
  *  coal            ~  wet clay
Cells outside the grid count as walls. A move that does not change the
ember's position is not a legal move.
"""

from __future__ import annotations

from dataclasses import dataclass

WALL, FLOOR, START, EXIT, COAL, CLAY = "#", ".", "S", "E", "*", "~"
ALPHABET = frozenset(WALL + FLOOR + START + EXIT + COAL + CLAY)

MIN_SIZE = 4
MAX_SIZE = 10
MAX_COALS = 5
MAX_CLAY = 6

# Direction letter -> (row delta, column delta). Order is fixed so that solver
# output is deterministic.
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


class LevelError(ValueError):
    """Raised when level text breaks the format. `code` is machine readable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Level:
    rows: tuple[str, ...]
    width: int
    height: int
    start: int  # cell index = row * width + col
    exit: int
    walls: frozenset[int]
    coals: tuple[int, ...]  # cell indices, in reading order
    clay: tuple[int, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.rows)

    @property
    def all_coals_mask(self) -> int:
        return (1 << len(self.coals)) - 1

    @property
    def all_clay_mask(self) -> int:
        return (1 << len(self.clay)) - 1

    def initial_state(self) -> "State":
        # (position, collected-coal bitmask, still-wet-clay bitmask)
        return (self.start, 0, self.all_clay_mask)

    def is_goal(self, state: "State") -> bool:
        return state[0] == self.exit and state[1] == self.all_coals_mask


State = tuple[int, int, int]


def parse_level(text: str) -> Level:
    """Strict parser. Rejects anything that is not exactly a legal level."""
    if not isinstance(text, str):
        raise LevelError("not_text", "level must be a string")
    lines = text.strip("\n").split("\n")
    lines = [ln.rstrip("\r") for ln in lines]
    if any(ln != ln.strip() for ln in lines):
        raise LevelError("whitespace", "rows must not have leading or trailing spaces")
    if not lines or lines == [""]:
        raise LevelError("empty", "level is empty")
    height = len(lines)
    width = len(lines[0])
    if not (MIN_SIZE <= height <= MAX_SIZE and MIN_SIZE <= width <= MAX_SIZE):
        raise LevelError(
            "size",
            f"grid is {width}x{height}; width and height must each be "
            f"{MIN_SIZE}..{MAX_SIZE}",
        )
    for i, ln in enumerate(lines):
        if len(ln) != width:
            raise LevelError(
                "ragged",
                f"row {i + 1} has {len(ln)} characters but row 1 has {width}; "
                "all rows must be the same length",
            )
        bad = sorted(set(ln) - ALPHABET)
        if bad:
            raise LevelError(
                "alphabet",
                f"row {i + 1} contains illegal character(s) {bad!r}; "
                "only # . S E * ~ are allowed",
            )
    flat = "".join(lines)
    for ch, name in ((START, "start S"), (EXIT, "exit E")):
        n = flat.count(ch)
        if n != 1:
            raise LevelError(
                "count", f"level must contain exactly one {name}, found {n}"
            )
    coals = tuple(i for i, ch in enumerate(flat) if ch == COAL)
    clay = tuple(i for i, ch in enumerate(flat) if ch == CLAY)
    if len(coals) > MAX_COALS:
        raise LevelError(
            "count", f"too many coals: {len(coals)} (maximum {MAX_COALS})"
        )
    if len(clay) > MAX_CLAY:
        raise LevelError(
            "count", f"too many clay tiles: {len(clay)} (maximum {MAX_CLAY})"
        )
    return Level(
        rows=tuple(lines),
        width=width,
        height=height,
        start=flat.index(START),
        exit=flat.index(EXIT),
        walls=frozenset(i for i, ch in enumerate(flat) if ch == WALL),
        coals=coals,
        clay=clay,
    )


def make_mover(level: Level):
    """Return a fast `move(state, dir_letter) -> State | None` for this level."""
    w, h = level.width, level.height
    walls = level.walls
    coal_bit = {cell: 1 << i for i, cell in enumerate(level.coals)}
    clay_bit = {cell: 1 << i for i, cell in enumerate(level.clay)}

    def move(state: State, d: str) -> State | None:
        pos, got, wet = state
        dr, dc = DIRS[d]
        r, c = divmod(pos, w)
        baked = 0
        moved = False
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            nxt = nr * w + nc
            if nxt in walls:
                break
            bit = clay_bit.get(nxt)
            if bit is not None and not wet & bit:
                break  # baked clay is a wall
            # Leave the current cell; wet clay bakes behind the ember.
            cur_bit = clay_bit.get(r * w + c)
            if cur_bit is not None:
                baked |= cur_bit
            r, c = nr, nc
            moved = True
            cb = coal_bit.get(nxt)
            if cb is not None:
                got |= cb
        if not moved:
            return None
        return (r * w + c, got, wet & ~baked)

    return move


def apply_moves(level: Level, moves: str) -> State:
    """Play a move string from the start. Raises ValueError on an illegal move."""
    move = make_mover(level)
    state = level.initial_state()
    for i, d in enumerate(moves):
        if d not in DIRS:
            raise ValueError(f"unknown move {d!r} at index {i}")
        nxt = move(state, d)
        if nxt is None:
            raise ValueError(f"move {d!r} at index {i} is blocked")
        state = nxt
    return state


def is_solution(level: Level, moves: str) -> bool:
    try:
        return level.is_goal(apply_moves(level, moves))
    except ValueError:
        return False


def _transform(rows: tuple[str, ...], k: int) -> tuple[str, ...]:
    """k in 0..7: the eight rotations/reflections of a grid."""
    g = [list(r) for r in rows]
    if k >= 4:
        g = [row[::-1] for row in g]
    for _ in range(k % 4):
        g = [list(r) for r in zip(*g[::-1])]
    return tuple("".join(r) for r in g)


def canonical(level: Level) -> str:
    """Symmetry-invariant key: two levels that differ only by rotation or
    reflection get the same key."""
    return min("\n".join(_transform(level.rows, k)) for k in range(8))
