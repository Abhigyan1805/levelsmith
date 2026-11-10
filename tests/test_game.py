import pytest

from levelsmith.game import (
    LevelError,
    apply_moves,
    canonical,
    is_solution,
    make_mover,
    parse_level,
)


def lv(*rows):
    return parse_level("\n".join(rows))


def cell(level, r, c):
    return r * level.width + c


# ---------- parser ----------

def test_parse_basic():
    level = lv("S..E", "....", ".*~.", "....")
    assert (level.width, level.height) == (4, 4)
    assert level.start == 0 and level.exit == 3
    assert level.coals == (9,) and level.clay == (10,)


@pytest.mark.parametrize(
    "text,code",
    [
        ("", "empty"),
        ("S.E\n...\n...", "size"),  # 3 wide
        ("S..E\n....\n....\n....\n" + "....\n" * 7, "size"),  # 11 tall
        ("S..E\n...\n....\n....", "ragged"),
        ("S..E\n....\n..x.\n....", "alphabet"),
        ("S..E\n....\n.. .\n....", "alphabet"),
        ("S...\n....\n....\n....", "count"),  # no exit
        ("S..E\n...S\n....\n....", "count"),  # two starts
        ("S..E\n....\n....\n....\n ", "whitespace"),
        ("S*****E\n*......\n.......\n.......", "count"),  # 6 coals
        ("S~~~~~E\n~~.....\n.......\n.......", "count"),  # 7 clay
    ],
)
def test_parse_rejects(text, code):
    with pytest.raises(LevelError) as e:
        parse_level(text)
    assert e.value.code == code


def test_parse_rejects_non_string():
    with pytest.raises(LevelError):
        parse_level(None)


# ---------- movement ----------

def test_slide_until_wall_and_grid_edge():
    level = lv("S....", "..#..", ".....", "....E")
    move = make_mover(level)
    s = level.initial_state()
    assert move(s, "R")[0] == cell(level, 0, 4)  # stops at the grid edge
    assert move(s, "D")[0] == cell(level, 3, 0)
    s2 = move(move(s, "D"), "R")  # along the bottom row onto E at the edge
    assert s2[0] == level.exit


def test_wall_stops_slide():
    level = lv("S.#.E", ".....", ".....", ".....")
    move = make_mover(level)
    assert move(level.initial_state(), "R")[0] == cell(level, 0, 1)


def test_blocked_move_is_illegal():
    level = lv("S#..E", ".....", ".....", ".....")
    move = make_mover(level)
    assert move(level.initial_state(), "R") is None
    assert move(level.initial_state(), "U") is None
    assert move(level.initial_state(), "L") is None
    with pytest.raises(ValueError):
        apply_moves(level, "R")


def test_coals_collected_in_passing():
    level = lv("S*.*E", ".....", ".....", ".....")
    state = apply_moves(level, "R")
    assert state[1] == 0b11
    assert level.is_goal(state)


def test_exit_needs_all_coals():
    level = lv("S...E", ".....", ".....", "*....")
    assert not level.is_goal(apply_moves(level, "R"))
    assert is_solution(level, "DUR")


def test_exit_does_not_stop_the_ember():
    level = lv("S.E..", ".....", ".....", ".....")
    state = apply_moves(level, "R")
    assert state[0] == cell(level, 0, 4)
    assert not level.is_goal(state)


def test_clay_bakes_after_being_crossed():
    # First R crosses the clay and stops at the edge. Going back L, the baked
    # clay is now a wall, so the ember stops right of it.
    level = lv("S.~..", ".....", ".....", "....E")
    move = make_mover(level)
    s1 = move(level.initial_state(), "R")
    assert s1[0] == cell(level, 0, 4)
    assert s1[2] == 0  # clay no longer wet
    s2 = move(s1, "L")
    assert s2[0] == cell(level, 0, 3)


def test_clay_stays_wet_while_ember_rests_on_it():
    # Wall right after the clay: the ember comes to rest on the clay tile.
    level = lv("S.~#.", ".....", ".....", "....E")
    move = make_mover(level)
    s1 = move(level.initial_state(), "R")
    assert s1[0] == cell(level, 0, 2)
    assert s1[2] == 1  # still wet under the ember
    s2 = move(s1, "D")
    assert s2[2] == 0  # baked once the ember left
    s3 = move(move(s2, "L"), "U")  # bottom-left corner, then back up to S
    assert s3[0] == cell(level, 0, 0)
    s4 = move(s3, "R")  # baked clay now stops the slide one cell early
    assert s4[0] == cell(level, 0, 1)


def test_baked_clay_blocks_adjacent_move():
    level = lv("S~...", ".....", ".....", "....E")
    move = make_mover(level)
    s1 = move(level.initial_state(), "R")  # crosses clay at (0,1)
    s2 = move(s1, "L")  # stops at (0,2), next to the baked clay
    assert s2[0] == cell(level, 0, 2)
    assert move(s2, "L") is None


def test_is_solution_rejects_bad_letters():
    level = lv("S...E", ".....", ".....", ".....")
    assert is_solution(level, "R")
    assert not is_solution(level, "X")
    assert not is_solution(level, "")


# ---------- canonical form ----------

def test_canonical_is_symmetry_invariant():
    a = lv("S..#", "..*.", "~...", "...E")
    rot = lv(".~.S", "....", "..*.", "E..#")  # a rotated 90 degrees clockwise
    mirror = lv("#..S", ".*..", "...~", "E...")
    assert canonical(a) == canonical(rot) == canonical(mirror)
    other = lv("S..#", "..*.", "....", "~..E")
    assert canonical(a) != canonical(other)
