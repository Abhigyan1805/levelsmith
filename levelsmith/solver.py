"""Exact solver for Kiln Slide.

Breadth-first search over the full reachable state graph. Because every move
costs 1, BFS depth is the optimal move count. Exploring the whole graph (not
stopping at the first goal) also gives exact dead-end counts by a backward
pass from the goal states.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict

from .game import DIRS, Level, make_mover

DEFAULT_MAX_STATES = 250_000


@dataclass
class SolveResult:
    solvable: bool
    complete: bool  # False if the state cap was hit before the graph was exhausted
    optimal_moves: int | None = None
    solution: str | None = None  # one optimal solution, letters from UDLR
    n_optimal_solutions: int | None = None
    states_to_solve: int | None = None  # states expanded by BFS before first goal
    reachable_states: int = 0
    avg_branching: float = 0.0
    dead_end_states: int = 0  # reachable states from which no goal is reachable
    dead_end_ratio: float = 0.0
    trap_moves: int | None = None  # moves off the optimal path that enter a dead end
    # Diagnostics, mainly for feedback on unsolvable levels.
    rest_cells: list[int] = field(default_factory=list)
    coals_collectable: list[bool] = field(default_factory=list)
    max_coals_collected: int = 0
    exit_restable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def solve(level: Level, max_states: int = DEFAULT_MAX_STATES) -> SolveResult:
    move = make_mover(level)
    start = level.initial_state()
    goal_pos, goal_mask = level.exit, level.all_coals_mask

    depth = {start: 0}
    parent: dict = {start: None}
    n_paths = {start: 1}
    succ: dict = {}
    order = []
    queue = deque([start])
    first_goal = None
    states_to_solve = None
    complete = True

    while queue:
        s = queue.popleft()
        order.append(s)
        if s[0] == goal_pos and s[1] == goal_mask:
            # The game ends on a goal state; it has no successors.
            succ[s] = ()
            if first_goal is None:
                first_goal = s
                states_to_solve = len(order) - 1
            continue
        d = depth[s] + 1
        out = []
        for letter in DIRS:
            t = move(s, letter)
            if t is None:
                continue
            out.append(t)
            if t not in depth:
                depth[t] = d
                parent[t] = (s, letter)
                n_paths[t] = n_paths[s]
                queue.append(t)
            elif depth[t] == d:
                n_paths[t] += n_paths[s]
        succ[s] = tuple(out)
        if len(depth) > max_states:
            complete = False
            break

    res = SolveResult(solvable=first_goal is not None, complete=complete)
    res.reachable_states = len(depth)
    res.rest_cells = sorted({s[0] for s in depth})
    union = 0
    best = 0
    for s in depth:
        union |= s[1]
        cnt = s[1].bit_count()
        if cnt > best:
            best = cnt
    res.coals_collectable = [bool(union >> i & 1) for i in range(len(level.coals))]
    res.max_coals_collected = best
    res.exit_restable = any(s[0] == goal_pos for s in depth)

    if not complete:
        # Too large to analyse exactly. Callers treat this as a rejection.
        if first_goal is not None:
            res.optimal_moves = depth[first_goal]
        return res

    non_goal = [s for s in order if not level.is_goal(s)]
    if non_goal:
        res.avg_branching = sum(len(succ[s]) for s in non_goal) / len(non_goal)

    # Backward reachability from goal states.
    pred: dict = {s: [] for s in order}
    for s in order:
        for t in succ[s]:
            pred[t].append(s)
    alive = {s for s in order if level.is_goal(s)}
    stack = list(alive)
    while stack:
        t = stack.pop()
        for s in pred[t]:
            if s not in alive:
                alive.add(s)
                stack.append(s)
    res.dead_end_states = len(order) - len(alive)
    res.dead_end_ratio = res.dead_end_states / len(order)

    if first_goal is None:
        return res

    res.optimal_moves = depth[first_goal]
    res.states_to_solve = states_to_solve
    best_depth = depth[first_goal]
    res.n_optimal_solutions = sum(
        n_paths[s] for s in order if level.is_goal(s) and depth[s] == best_depth
    )
    letters = []
    path_states = []
    s = first_goal
    while parent[s] is not None:
        s, letter = parent[s]
        letters.append(letter)
        path_states.append(s)
    res.solution = "".join(reversed(letters))
    res.trap_moves = sum(
        1 for s in path_states for t in succ[s] if t not in alive
    )
    return res
