import re

import pytest

from levelsmith import agent, baseline, judge as J
from levelsmith.game import is_solution, parse_level
from levelsmith.llm import Budget, BudgetExceeded, LLMClient, LLMResponse, cost_of
from levelsmith.solver import solve
from levelsmith.stats import newcombe_diff, spearman, wilson


class FakeClient(LLMClient):
    """Returns scripted replies; no network."""

    provider = "fake"

    def __init__(self, replies, budget=None, **kw):
        super().__init__("fake-model", budget or Budget(100, 1.0), **kw)
        self.replies = list(replies)
        self.seen_messages = []

    def complete(self, system, messages, temperature=1.0, tag=None):
        self.budget.begin()
        self.seen_messages.append([dict(m) for m in messages])
        text = self.replies.pop(0)
        self.budget.end(0.001)
        return LLMResponse(ok=bool(text), text=text, cost_usd=0.001, latency_s=0.5)


def reply(level, solution="R"):
    return f"```\n{level}\n```\nsolution: {solution}"


EASY_6x5 = "..~.*.\n......\nE....S\n.#....\n......"  # optimal 5: ULRDL
UNSOLVABLE_5x4 = "S....\n.....\n..E..\n....."
TRIVIAL_5x4 = "S...E\n.....\n.....\n....."


# ---------- prompt examples are what the prompt says they are ----------

def test_prompt_examples_are_correct():
    blocks = re.findall(
        r"EXAMPLE \d+ \(.*?optimal (\d+) moves: ([UDLR ]+)\)\n((?:[#.SE*~]+\n)+)",
        agent.SYSTEM_PROMPT,
    )
    assert len(blocks) == 2
    for n, sol, grid in blocks:
        level = parse_level(grid.strip())
        res = solve(level)
        assert res.optimal_moves == int(n)
        assert is_solution(level, sol.replace(" ", ""))


def test_prompt_states_the_band_ranges():
    for band, (lo, hi) in J.BANDS.items():
        assert f"{band} = {lo}-{hi} moves" in agent.SYSTEM_PROMPT


# ---------- extraction ----------

def test_extract_level_and_solution():
    lvl, sol, err = agent.extract(reply(EASY_6x5, "U L R D L"))
    assert lvl == EASY_6x5 and sol == "ULRDL" and err == ""


def test_extract_takes_last_block():
    text = reply("S..E\n....\n....\n....") + "\nactually:\n" + reply(EASY_6x5, "ULRDL")
    lvl, sol, _ = agent.extract(text)
    assert lvl == EASY_6x5 and sol == "ULRDL"


def test_extract_rejects_missing_fence_and_oversized_reply():
    assert agent.extract("S..E\n....")[0] is None
    assert agent.extract("x" * (agent.MAX_RESPONSE_CHARS + 1))[0] is None


# ---------- judge verdicts ----------

@pytest.mark.parametrize(
    "text,band,code",
    [
        (EASY_6x5, "easy", J.OK),
        (EASY_6x5, "hard", J.WRONG_DIFFICULTY),
        (UNSOLVABLE_5x4, "easy", J.UNSOLVABLE),
        (TRIVIAL_5x4, "easy", J.TRIVIAL),
        ("S..E\n...", "easy", J.MALFORMED),
    ],
)
def test_judge_codes(text, band, code):
    assert J.judge(text, band).code == code


def test_judge_duplicate_detects_symmetric_copy():
    first = J.judge(EASY_6x5, "easy")
    mirrored = "\n".join(row[::-1] for row in EASY_6x5.split("\n"))
    again = J.judge(mirrored, "easy", {first.canonical})
    assert again.code == J.DUPLICATE


def test_check_enforces_requested_size():
    v, _ = agent.check(reply(EASY_6x5), "easy", 7, 5, set())
    assert v.code == J.MALFORMED and v.data["parse_code"] == "wrong_size"
    v, _ = agent.check(reply(EASY_6x5), "easy", 6, 5, set())
    assert v.accepted


# ---------- feedback ----------

def test_feedback_for_unsolvable_names_the_problem():
    v, claimed = agent.check(reply(UNSOLVABLE_5x4, "RDL"), "easy", 5, 4, set())
    fb = agent.feedback(v, "easy", 5, 4, claimed)
    assert "REJECTED: unsolvable" in fb
    assert "never come to rest on the exit at (3,3)" in fb
    assert "move 1 (R): rests at (1,5)" in fb
    assert "(1,1)" in fb and "(4,5)" in fb  # rest cells listed


def test_feedback_for_wrong_difficulty_gives_solver_solution():
    v, claimed = agent.check(reply(EASY_6x5, "ULRDL"), "hard", 6, 5, set())
    fb = agent.feedback(v, "hard", 6, 5, claimed)
    assert "wrong_difficulty" in fb and "ULRDL" in fb and "Too short" in fb


def test_feedback_for_uncollectable_coal():
    text = "S...E\n.....\n.....\n..#..\n.#*#.\n..#.."
    v, claimed = agent.check(reply(text), "easy", 5, 6, set())
    fb = agent.feedback(v, "easy", 5, 6, claimed)
    assert "Coals that can never be collected: (5,3)" in fb


# ---------- the loop ----------

def test_repair_loop_feeds_back_and_accepts():
    client = FakeClient([reply(UNSOLVABLE_5x4), reply(EASY_6x5, "ULRDL")])
    seen = set()
    rec = agent.run_request(client, "repair", "easy", 6, 5, seen, max_rounds=4)
    assert rec["accepted"] and rec["accepted_round"] == 2 and rec["n_calls"] == 2
    assert rec["features"]["optimal_moves"] == 5
    # Second call saw the first attempt and the feedback.
    assert len(client.seen_messages[1]) == 3
    assert "REJECTED" in client.seen_messages[1][2]["content"]
    assert len(seen) == 1


def test_blind_mode_carries_nothing_over():
    client = FakeClient([reply(UNSOLVABLE_5x4), reply(EASY_6x5, "ULRDL")])
    rec = agent.run_request(client, "blind", "easy", 6, 5, set(), max_rounds=4)
    assert rec["accepted"] and rec["accepted_round"] == 2
    assert all(len(m) == 1 for m in client.seen_messages)
    assert "feedback" not in rec["rounds"][0]


def test_loop_stops_at_max_rounds():
    client = FakeClient([reply(TRIVIAL_5x4)] * 3)
    rec = agent.run_request(client, "repair", "easy", 5, 4, set(), max_rounds=3)
    assert not rec["accepted"] and rec["stop_reason"] == "max_rounds"
    assert [r["verdict"]["code"] for r in rec["rounds"]] == ["trivial"] * 3


def test_loop_rejects_duplicate_of_earlier_level():
    seen = set()
    c1 = FakeClient([reply(EASY_6x5)])
    agent.run_request(c1, "repair", "easy", 6, 5, seen, max_rounds=1)
    c2 = FakeClient([reply(EASY_6x5)])
    rec = agent.run_request(c2, "repair", "easy", 6, 5, seen, max_rounds=1)
    assert rec["rounds"][0]["verdict"]["code"] == "duplicate"


def test_loop_stops_when_call_budget_runs_out():
    client = FakeClient([reply(TRIVIAL_5x4)] * 5, budget=Budget(2, 1.0))
    rec = agent.run_request(client, "repair", "easy", 5, 4, set(), max_rounds=5)
    assert rec["n_calls"] == 2 and rec["stop_reason"].startswith("budget")


# ---------- budget and cost ----------

def test_budget_caps():
    b = Budget(max_calls=2, max_usd=1.0)
    b.begin(); b.end(0.1)
    b.begin(); b.end(0.1)
    with pytest.raises(BudgetExceeded):
        b.begin()
    b = Budget(max_calls=10, max_usd=0.05, reserve_usd=0.02)
    b.begin(); b.end(0.04)
    with pytest.raises(BudgetExceeded):
        b.begin()  # 0.04 spent + 0.02 reserve would pass the cap


def test_cost_of_unknown_model_is_zero():
    assert cost_of("no-such-model", 1000, 0, 1000) == 0.0
    assert cost_of("gemini-3.6-flash", 1_000_000, 0, 0) == pytest.approx(0.75)


# ---------- baseline ----------

def test_baseline_is_deterministic_and_valid():
    a = baseline.generate("medium", 7, 7, seed=123, seen=set())
    b = baseline.generate("medium", 7, 7, seed=123, seen=set())
    assert a["accepted"] and a["level"] == b["level"]
    level = parse_level(a["level"])
    assert (level.width, level.height) == (7, 7)
    lo, hi = J.BANDS["medium"]
    assert lo <= solve(level).optimal_moves <= hi


# ---------- stats ----------

def test_wilson_and_newcombe():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == pytest.approx(1.0)
    d, lo, hi = newcombe_diff(8, 10, 4, 10)
    assert d == pytest.approx(0.4) and lo < d < hi


def test_spearman_handles_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([0, 0, 1, 1, 2, 2], [3, 4, 6, 7, 11, 12]) > 0.9
    assert spearman([1, 1, 1], [1, 2, 3]) is None


# ---------- solver-side one-edit patch ----------

def test_patch_fixes_an_unsolvable_level_with_one_edit():
    from levelsmith.patch import find_patch

    broken = EASY_6x5.replace("~", ".")  # example 2 without its clay tile
    assert J.judge(broken, "easy").code == J.UNSOLVABLE
    p = find_patch(broken, "easy")
    assert p is not None and p["code"] == "ok"
    patched = parse_level(p["level"])
    # Exactly one edit: at most two cells differ (moving the exit touches two).
    diff = sum(a != b for a, b in zip(broken.replace("\n", ""), p["level"].replace("\n", "")))
    assert 1 <= diff <= 2
    lo, hi = J.BANDS["easy"]
    assert lo <= solve(patched).optimal_moves <= hi
    assert is_solution(patched, p["features"]["solution"])


def test_patch_returns_none_when_no_single_edit_works():
    from levelsmith.patch import find_patch

    # A 4x4 open room cannot hold a 10-move optimum after one edit.
    assert find_patch("S...\n....\n....\n...E", "hard") is None
    assert find_patch("not a level", "easy") is None


def test_patch_is_deterministic():
    from levelsmith.patch import find_patch

    broken = EASY_6x5.replace("~", ".")
    a = find_patch(broken, "easy")
    b = find_patch(broken, "easy")
    assert a["level"] == b["level"] and a["edit"] == b["edit"]


def test_loop_with_patch_accepts_without_a_second_call():
    broken = EASY_6x5.replace("~", ".")
    client = FakeClient([reply(broken), reply(EASY_6x5)])
    rec = agent.run_request(client, "repair", "easy", 6, 5, set(), max_rounds=4, patch=True)
    assert rec["accepted"] and rec["patched"] and rec["n_calls"] == 1
    assert rec["rounds"][0]["verdict"]["code"] == "unsolvable"
    assert is_solution(parse_level(rec["level"]), rec["features"]["solution"])


def test_loop_without_patch_is_unchanged():
    broken = EASY_6x5.replace("~", ".")
    client = FakeClient([reply(broken), reply(EASY_6x5, "ULRDL")])
    rec = agent.run_request(client, "repair", "easy", 6, 5, set(), max_rounds=4)
    assert rec["accepted"] and not rec["patched"] and rec["n_calls"] == 2
