import itertools

from levelsmith.game import DIRS, is_solution, make_mover, parse_level
from levelsmith.solver import solve


def lv(*rows):
    return parse_level("\n".join(rows))


def brute_force_optimal(level, max_len):
    """Independent check: try every move string by increasing length."""
    for n in range(1, max_len + 1):
        for combo in itertools.product(DIRS, repeat=n):
            if is_solution(level, "".join(combo)):
                return n
    return None


def test_one_move_level():
    res = solve(lv("S...E", ".....", ".....", "....."))
    assert res.solvable and res.optimal_moves == 1 and res.solution == "R"


def test_solution_is_valid_and_optimal_against_brute_force():
    levels = [
        lv("S...E", ".....", ".....", "*...."),
        lv("S.#..", "...*.", ".#...", "...E#"),
        lv("#..E.", "S.~..", "....#", ".*..."),
        lv("..#...", "S....#", ".#.*..", "....#.", "E.~..."),
    ]
    for level in levels:
        res = solve(level)
        bf = brute_force_optimal(level, 7)
        if bf is None:
            assert not res.solvable or res.optimal_moves > 7
        else:
            assert res.solvable
            assert res.optimal_moves == bf
            assert len(res.solution) == bf
            assert is_solution(level, res.solution)


def test_unsolvable_exit_never_a_rest_cell():
    # Open room: the ember only ever rests in the four corners.
    level = lv("S....", ".....", "..E..", ".....")
    res = solve(level)
    assert not res.solvable and res.complete
    assert not res.exit_restable
    assert res.optimal_moves is None and res.solution is None
    assert res.dead_end_states == res.reachable_states == 4


def test_unsolvable_coal_out_of_reach():
    level = lv("S...E", ".....", ".....", "..#..", ".#*#.", "..#..")
    res = solve(level)
    assert not res.solvable
    assert res.exit_restable
    assert res.coals_collectable == [False]
    assert res.max_coals_collected == 0


def test_clay_is_required_as_a_stopper():
    # The ember crosses the clay, then slides back and stops against the wall
    # the clay has baked into, which leaves it resting on the exit.
    level = lv("S~E.#", ".....", ".....", ".....")
    res = solve(level)
    assert res.solvable and res.solution == "RL"
    assert is_solution(level, res.solution)
    # Same level with plain floor instead of clay: the exit is never a rest cell.
    no_clay = solve(lv("S.E.#", ".....", ".....", "....."))
    assert no_clay.complete and not no_clay.solvable and not no_clay.exit_restable


def test_dead_ends_counted():
    # Stepping right crosses the clay and seals the corridor behind the ember,
    # leaving it boxed in at the right end with the coal still uncollected.
    level = lv("#####", "*S~.#", "E####", "#####")
    res = solve(level)
    assert res.solvable and res.solution == "LD"
    assert res.dead_end_states >= 1
    assert res.trap_moves >= 1
    assert 0 < res.dead_end_ratio < 1


def test_no_dead_ends_in_reversible_level():
    res = solve(lv("S...E", ".....", ".....", "*...."))
    assert res.solvable
    assert res.dead_end_states == 0 and res.trap_moves == 0


def test_optimal_solution_count():
    # Two equally short routes: RD and DR.
    res = solve(lv("S...", "....", "....", "...E"))
    assert res.optimal_moves == 2
    assert res.n_optimal_solutions == 2


def test_state_cap_reports_incomplete():
    level = lv("S.....", ".*.~..", "..#...", ".~..*.", "......", ".....E")
    res = solve(level, max_states=3)
    assert not res.complete


def test_branching_and_reachable_counts():
    level = lv("S...E", ".....", ".....", ".....")
    res = solve(level)
    # States: start corner (2 moves: R -> goal, D -> bottom-left), ...
    assert res.reachable_states >= 2
    assert 1.0 <= res.avg_branching <= 4.0


def test_solver_agrees_with_mover_on_solution_path():
    level = lv("..#...", "S....#", ".#.*..", "....#.", "E.~...")
    res = solve(level)
    if res.solvable:
        move = make_mover(level)
        s = level.initial_state()
        for d in res.solution:
            s = move(s, d)
            assert s is not None
        assert level.is_goal(s)
