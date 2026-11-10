"""Turn raw run files into results/summary.json and results/summary.md.

  python -m levelsmith.report
"""

from __future__ import annotations

import json
import os

from .agent import SYSTEM_PROMPT
from .game import _transform, canonical, parse_level
from .judge import BANDS
from .llm import cost_of
from .baseline import random_level_text
from .evaluate import make_requests
from .judge import judge
from .patch import find_patch
from .solver import solve
from .stats import mean, median, newcombe_diff, spearman_ci, wilson

RAW = os.path.join("results", "raw")
BAND_INDEX = {b: i for i, b in enumerate(BANDS)}


def load(name: str) -> dict | None:
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def fmt_rate(r: dict) -> str:
    if not r["n"]:
        return "n/a"
    return (
        f"{100 * r['rate']:.0f}% ({r['k']}/{r['n']}; "
        f"{100 * r['ci95'][0]:.0f}-{100 * r['ci95'][1]:.0f}%)"
    )


def accepted_within(records: list[dict], rounds: int) -> int:
    return sum(1 for r in records if r["accepted"] and r["accepted_round"] <= rounds)


def hamming_novelty(levels: list[str]) -> dict:
    """For each level, distance to its nearest neighbour among the others:
    fraction of cells that differ, minimised over the 8 symmetries, comparing
    only levels with the same dimensions."""
    parsed = [parse_level(t) for t in levels]
    dists = []
    for i, a in enumerate(parsed):
        best = None
        for j, b in enumerate(parsed):
            if i == j:
                continue
            for k in range(8):
                rows = _transform(b.rows, k)
                if len(rows) != a.height or len(rows[0]) != a.width:
                    continue
                d = sum(
                    x != y for ra, rb in zip(a.rows, rows) for x, y in zip(ra, rb)
                ) / (a.width * a.height)
                if best is None or d < best:
                    best = d
        if best is not None:
            dists.append(best)
    return {
        "n_levels": len(levels),
        "n_distinct": len({canonical(p) for p in parsed}),
        "nearest_neighbour_distance_mean": mean(dists),
        "nearest_neighbour_distance_min": min(dists) if dists else None,
    }


def level_features(records: list[dict]) -> dict:
    """Design features of accepted levels, recomputed with the solver."""
    acc = [r for r in records if r.get("accepted")]
    if not acc:
        return {"n": 0}
    clay_levels = 0
    clay_essential = 0
    walls, coals, clay = [], [], []
    for r in acc:
        lvl = parse_level(r["level"])
        cells = lvl.width * lvl.height
        walls.append(len(lvl.walls) / cells)
        coals.append(len(lvl.coals))
        clay.append(len(lvl.clay))
        if lvl.clay:
            clay_levels += 1
            plain = solve(parse_level(r["level"].replace("~", ".")))
            if not plain.solvable or plain.optimal_moves != r["features"]["optimal_moves"]:
                clay_essential += 1
    f = [r["features"] for r in acc]
    return {
        "n": len(acc),
        "optimal_moves_mean": mean(x["optimal_moves"] for x in f),
        "reachable_states_median": median(x["reachable_states"] for x in f),
        "avg_branching_mean": mean(x["avg_branching"] for x in f),
        "dead_end_ratio_mean": mean(x["dead_end_ratio"] for x in f),
        "levels_with_dead_ends": rate(sum(1 for x in f if x["dead_end_states"] > 0), len(f)),
        "trap_moves_mean": mean(x["trap_moves"] for x in f),
        "unique_optimal_solution": rate(
            sum(1 for x in f if x["n_optimal_solutions"] == 1), len(f)
        ),
        "wall_density_mean": mean(walls),
        "coals_mean": mean(coals),
        "clay_mean": mean(clay),
        "uses_clay": rate(clay_levels, len(acc)),
        # Clay matters if replacing it with floor changes solvability or the optimum.
        "clay_changes_solution": rate(clay_essential, len(acc)),
    }


def difficulty_control(proposals: list[tuple[str, dict]]) -> dict:
    """proposals: (requested band, verdict summary) for a set of LLM proposals."""
    solv = [(b, v["features"]["optimal_moves"]) for b, v in proposals
            if v.get("features", {}).get("optimal_moves") is not None]
    out = {"n_proposals": len(proposals), "n_solvable": len(solv), "by_band": {}}
    for band, (lo, hi) in BANDS.items():
        ms = [m for b, m in solv if b == band]
        n_band = sum(1 for b, _ in proposals if b == band)
        hits = sum(1 for m in ms if lo <= m <= hi)
        out["by_band"][band] = {
            "requested_range": [lo, hi],
            "n_proposals": n_band,
            "n_solvable": len(ms),
            "optimal_moves_mean": mean(ms),
            "optimal_moves_median": median(ms),
            "optimal_moves_min": min(ms) if ms else None,
            "optimal_moves_max": max(ms) if ms else None,
            "in_band_given_solvable": rate(hits, len(ms)),
            "below_band": sum(1 for m in ms if m < lo),
            "above_band": sum(1 for m in ms if m > hi),
        }
    hits = sum(1 for b, m in solv if BANDS[b][0] <= m <= BANDS[b][1])
    out["in_band_given_solvable"] = rate(hits, len(solv))
    rho, lo, hi = spearman_ci([BAND_INDEX[b] for b, _ in solv], [m for _, m in solv])
    out["spearman_band_vs_optimal"] = {"rho": rho, "ci95": [lo, hi], "n": len(solv)}
    return out


PATCHABLE_CODES = ("unsolvable", "trivial", "wrong_difficulty")


def patch_replay(records: list[dict], max_rounds: int) -> dict:
    """Replay a run with the solver's one-edit patch applied after every
    rejection. The replay is exact: a patched run is identical to the logged
    run up to the first round whose proposal is accepted or patchable, and
    stops there."""
    seen: set[str] = set()
    first_ok: list[int | None] = []  # round at which each request succeeds
    rejected_r1 = patchable_r1 = 0
    patched_levels = []
    for r in sorted(records, key=lambda r: r["id"]):
        hit = None
        for rd in r["rounds"]:
            v = rd["verdict"]
            if v["code"] == "ok":
                hit = rd["round"]
                seen.add(r["canonical"])
                break
            p = find_patch(v["level"], r["band"], seen) if v["code"] in PATCHABLE_CODES else None
            if rd["round"] == 1:
                rejected_r1 += 1
                patchable_r1 += p is not None
            if p is not None:
                hit = rd["round"]
                seen.add(p["canonical"])
                patched_levels.append({"id": r["id"], "band": r["band"], "round": rd["round"],
                                       "original": v["level"], **p})
                break
        first_ok.append(hit)
    n = len(first_ok)
    return {
        "by_round": {
            str(k): rate(sum(1 for h in first_ok if h is not None and h <= k), n)
            for k in range(1, max_rounds + 1)
        },
        "round1_rejected_one_edit_from_valid": rate(patchable_r1, rejected_r1),
        "patched_levels": patched_levels,
    }


def baseline_patch(requests: list[dict], draws: int = 5) -> dict:
    """The same one-shot and one-shot-plus-patch measurement for random grids:
    the first `draws` samples of each request's random stream."""
    import random as _random
    import time as _time

    n = ok = patched = rejected_wellformed = 0
    t_patch = 0.0
    for r in requests:
        rng = _random.Random(r["seed"])
        for _ in range(draws):
            text = random_level_text(rng, r["width"], r["height"])
            v = judge(text, r["band"])
            n += 1
            if v.accepted:
                ok += 1
                continue
            rejected_wellformed += 1
            t0 = _time.perf_counter()
            p = find_patch(text, r["band"])
            t_patch += _time.perf_counter() - t0
            patched += p is not None
    return {
        "one_shot": rate(ok, n),
        "one_shot_plus_patch": rate(ok + patched, n),
        "rejected_one_edit_from_valid": rate(patched, rejected_wellformed),
        "patch_search_seconds_mean": t_patch / max(rejected_wellformed, 1),
    }


def summarize_llm(label: str) -> dict | None:
    runs = {m: load(f"{label}_{m}.json") for m in ("repair", "blind")}
    runs = {m: d for m, d in runs.items() if d and d["records"]}
    if not runs:
        return None
    out: dict = {"label": label, "modes": {}}
    any_meta = next(iter(runs.values()))["meta"]
    out["model"] = any_meta["model"]
    out["provider"] = any_meta["provider"]
    out["settings"] = {
        k: any_meta.get(k)
        for k in ("max_rounds", "temperature", "max_output_tokens", "thinking_budget", "thinking")
    }
    max_rounds = any_meta["max_rounds"]

    first_round: list[tuple[str, dict]] = []
    for mode, d in runs.items():
        recs = list(d["records"].values())
        n = len(recs)
        calls = [rd for r in recs for rd in r["rounds"]]
        if d["meta"]["provider"] == "gemini":
            # Free-tier calls cost nothing; report the list-price equivalent,
            # recomputed from the logged token counts.
            for rd in calls:
                rd["cost_usd"] = cost_of(
                    d["meta"]["model"], rd["prompt_tokens"], 0, rd["output_tokens"]
                )
        n_acc = sum(1 for r in recs if r["accepted"])
        verdicts: dict[str, int] = {}
        for rd in calls:
            code = rd["verdict"]["code"]
            verdicts[code] = verdicts.get(code, 0) + 1
        later = [rd for r in recs for rd in r["rounds"] if rd["round"] >= 2]
        m = {
            "n_requests": n,
            "n_requests_planned": d["meta"]["n_requests"],
            "accepted_within_max_rounds": rate(n_acc, n),
            "accepted_by_round": {
                str(k): rate(accepted_within(recs, k), n) for k in range(1, max_rounds + 1)
            },
            "by_band": {},
            "verdicts_all_proposals": dict(sorted(verdicts.items())),
            "per_call_accept_rate_round1": rate(
                sum(1 for r in recs if r["accepted_round"] == 1), n
            ),
            "per_call_accept_rate_rounds2plus": rate(
                sum(1 for rd in later if rd["verdict"]["code"] == "ok"), len(later)
            ),
            "calls": len(calls),
            "calls_per_request_mean": len(calls) / n,
            "cost_usd_total": sum(rd["cost_usd"] for rd in calls),
            "latency_s_total": sum(rd["latency_s"] for rd in calls),
            "latency_s_median_per_call": median(rd["latency_s"] for rd in calls),
            "output_tokens_mean_per_call": mean(rd["output_tokens"] for rd in calls),
            "prompt_tokens_mean_per_call": mean(rd["prompt_tokens"] for rd in calls),
            "incomplete_requests": len(d.get("incomplete", [])),
        }
        m["cost_usd_per_accepted"] = m["cost_usd_total"] / n_acc if n_acc else None
        m["latency_s_per_accepted"] = m["latency_s_total"] / n_acc if n_acc else None
        m["calls_per_accepted"] = len(calls) / n_acc if n_acc else None
        for band in BANDS:
            br = [r for r in recs if r["band"] == band]
            m["by_band"][band] = {
                "one_shot": rate(accepted_within(br, 1), len(br)),
                "within_max_rounds": rate(sum(1 for r in br if r["accepted"]), len(br)),
            }
        acc_levels = [r["level"] for r in recs if r["accepted"]]
        m["novelty"] = hamming_novelty(acc_levels) if acc_levels else None
        m["duplicates_rejected"] = verdicts.get("duplicate", 0)
        m["features_accepted"] = level_features(recs)
        m["difficulty_all_proposals"] = difficulty_control(
            [(r["band"], rd["verdict"]) for r in recs for rd in r["rounds"]]
        )
        m["meta"] = {
            k: d["meta"].get(k)
            for k in ("started_utc", "finished_utc", "deepseek_balance_before",
                      "deepseek_balance_after", "total_calls_logged", "total_usd_logged")
        }
        m["with_patch"] = patch_replay(recs, max_rounds)
        out["modes"][mode] = m
        first_round += [(r["band"], r["rounds"][0]["verdict"]) for r in recs if r["rounds"]]

    # Round 1 is the same condition in both modes, so pool it for the one-shot rate.
    k = sum(1 for _, v in first_round if v["code"] == "ok")
    one = {"accepted": rate(k, len(first_round)), "by_band": {}, "verdicts": {}}
    for band in BANDS:
        fb = [v for b, v in first_round if b == band]
        one["by_band"][band] = rate(sum(1 for v in fb if v["code"] == "ok"), len(fb))
    for _, v in first_round:
        one["verdicts"][v["code"]] = one["verdicts"].get(v["code"], 0) + 1
    one["verdicts"] = dict(sorted(one["verdicts"].items()))
    one["solvable_and_well_formed"] = rate(
        sum(1 for _, v in first_round
            if v.get("features", {}).get("optimal_moves") is not None),
        len(first_round),
    )
    one["difficulty"] = difficulty_control(first_round)
    k1 = sum(m["with_patch"]["by_round"]["1"]["k"] for m in out["modes"].values())
    one["accepted_with_patch"] = rate(k1, len(first_round))
    rej = [m["with_patch"]["round1_rejected_one_edit_from_valid"] for m in out["modes"].values()]
    one["rejected_one_edit_from_valid"] = rate(sum(x["k"] for x in rej), sum(x["n"] for x in rej))
    out["one_shot_pooled"] = one

    if "repair" in out["modes"] and "blind" in out["modes"]:
        a = out["modes"]["repair"]["accepted_within_max_rounds"]
        b = out["modes"]["blind"]["accepted_within_max_rounds"]
        d, lo, hi = newcombe_diff(a["k"], a["n"], b["k"], b["n"])
        out["repair_minus_blind"] = {"diff": d, "ci95": [lo, hi]}
        a = out["modes"]["repair"]["per_call_accept_rate_rounds2plus"]
        b = out["modes"]["blind"]["per_call_accept_rate_rounds2plus"]
        d, lo, hi = newcombe_diff(a["k"], a["n"], b["k"], b["n"])
        out["repair_minus_blind_per_call_rounds2plus"] = {"diff": d, "ci95": [lo, hi]}

    # Did any accepted level copy a prompt example?
    import re

    ex = {
        canonical(parse_level(g.strip()))
        for g in re.findall(r"moves: [UDLR ]+\)\n((?:[#.SE*~]+\n)+)", SYSTEM_PROMPT)
    }
    out["accepted_copies_of_prompt_examples"] = sum(
        1 for d in runs.values() for r in d["records"].values()
        if r.get("accepted") and r["canonical"] in ex
    )
    return out


def summarize_baseline() -> dict | None:
    d = load("baseline.json")
    if not d:
        return None
    recs = list(d["records"].values())
    out = {"n_requests": len(recs), "by_band": {}, "meta": d["meta"]}
    out["requests_filled"] = rate(sum(1 for r in recs if r["accepted"]), len(recs))
    tot_attempts = sum(r["attempts"] for r in recs)
    tot_acc = sum(1 for r in recs if r["accepted"])
    out["per_sample_accept_rate"] = rate(tot_acc, tot_attempts)
    out["seconds_per_accepted"] = sum(r["seconds"] for r in recs) / tot_acc
    out["attempts_per_accepted"] = tot_attempts / tot_acc
    for band in BANDS:
        br = [r for r in recs if r["band"] == band]
        att = sum(r["attempts"] for r in br)
        acc = sum(1 for r in br if r["accepted"])
        counts: dict[str, int] = {}
        for r in br:
            for code, c in r["verdict_counts"].items():
                counts[code] = counts.get(code, 0) + c
        out["by_band"][band] = {
            "requests_filled": rate(acc, len(br)),
            "per_sample_accept_rate": rate(acc, att),
            "attempts_per_accepted": att / acc if acc else None,
            "seconds_per_accepted": sum(r["seconds"] for r in br) / acc if acc else None,
            "optimal_moves_mean": mean(
                r["features"]["optimal_moves"] for r in br if r["accepted"]
            ),
            "verdict_counts": dict(sorted(counts.items())),
        }
    sample = d["unfiltered_sample"]
    solv = sorted(s["optimal_moves"] for s in sample if s["solvable"])
    q = lambda p: solv[min(len(solv) - 1, int(p * len(solv)))]
    out["unfiltered"] = {
        "n": len(sample),
        "solvable": rate(len(solv), len(sample)),
        "optimal_moves_quartiles": [q(0.25), q(0.5), q(0.75)],
        "optimal_moves_min_max": [solv[0], solv[-1]],
        "share_in_band": {
            b: rate(sum(1 for m in solv if lo <= m <= hi), len(sample))
            for b, (lo, hi) in BANDS.items()
        },
    }
    out["novelty"] = hamming_novelty([r["level"] for r in recs if r["accepted"]])
    out["features_accepted"] = level_features(recs)
    out["patch"] = baseline_patch(make_requests())
    return out


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def num(x, nd=2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def to_markdown(s: dict) -> str:
    L: list[str] = []
    llms = s["llm"]
    base = s["baseline"]

    L += ["### Valid-level rate", "",
          "Accepted = parses, right size, solvable, not a duplicate, and the solver's "
          "optimal move count is inside the requested band. Intervals are 95% Wilson.", "",
          "| Model | One-shot (round 1) | Repair loop, up to 5 rounds | Blind resampling, up to 5 rounds | Repair minus blind |",
          "|---|---|---|---|---|"]
    for x in llms:
        rp, bl = x["modes"].get("repair"), x["modes"].get("blind")
        diff = x.get("repair_minus_blind")
        L.append(
            f"| {x['model']} | {fmt_rate(x['one_shot_pooled']['accepted'])} | "
            f"{fmt_rate(rp['accepted_within_max_rounds']) if rp else 'not run'} | "
            f"{fmt_rate(bl['accepted_within_max_rounds']) if bl else 'not run'} | "
            + (f"{100 * diff['diff']:+.0f} pts ({100 * diff['ci95'][0]:+.0f} to {100 * diff['ci95'][1]:+.0f})"
               if diff else "n/a") + " |"
        )
    L += ["", "By band (one-shot pooled over both runs / repair loop / blind resampling):", "",
          "| Model | Band | One-shot | Repair loop | Blind resampling |", "|---|---|---|---|---|"]
    for x in llms:
        for band in BANDS:
            rp, bl = x["modes"].get("repair"), x["modes"].get("blind")
            L.append(
                f"| {x['model']} | {band} | {fmt_rate(x['one_shot_pooled']['by_band'][band])} | "
                f"{fmt_rate(rp['by_band'][band]['within_max_rounds']) if rp else 'not run'} | "
                f"{fmt_rate(bl['by_band'][band]['within_max_rounds']) if bl else 'not run'} |"
            )
    L += ["", "Cumulative acceptance by round:", "",
          "| Model | Mode | R1 | R2 | R3 | R4 | R5 |", "|---|---|---|---|---|---|---|"]
    for x in llms:
        for mode, m in x["modes"].items():
            cells = " | ".join(pct(m["accepted_by_round"][str(k)]["rate"]) for k in range(1, 6))
            L.append(f"| {x['model']} | {mode} | {cells} |")
    L += ["", "Per-call acceptance in rounds 2-5 (after a first rejection):", "",
          "| Model | With solver feedback | Blind resample | Difference |", "|---|---|---|---|"]
    for x in llms:
        rp, bl = x["modes"].get("repair"), x["modes"].get("blind")
        diff = x.get("repair_minus_blind_per_call_rounds2plus")
        L.append(
            f"| {x['model']} | {fmt_rate(rp['per_call_accept_rate_rounds2plus']) if rp else 'not run'} | "
            f"{fmt_rate(bl['per_call_accept_rate_rounds2plus']) if bl else 'not run'} | "
            + (f"{100 * diff['diff']:+.0f} pts ({100 * diff['ci95'][0]:+.0f} to {100 * diff['ci95'][1]:+.0f})"
               if diff else "n/a") + " |"
        )

    L += ["", "### Solver one-edit patch", "",
          "After a rejection the solver tries every single-cell edit (move the exit, "
          "add or remove one wall, coal or clay tile) and keeps one that the judge "
          "accepts. No extra model calls. The loop columns replay the logged runs "
          "with the patch applied after each rejection.", "",
          "| Source | One-shot | One-shot + patch | Rejected proposals one edit from valid | Repair loop + patch (5 rounds) | Blind + patch (5 rounds) |",
          "|---|---|---|---|---|---|"]
    for x in llms:
        o = x["one_shot_pooled"]
        rp, bl = x["modes"].get("repair"), x["modes"].get("blind")
        L.append(
            f"| {x['model']} | {fmt_rate(o['accepted'])} | {fmt_rate(o['accepted_with_patch'])} | "
            f"{fmt_rate(o['rejected_one_edit_from_valid'])} | "
            f"{fmt_rate(rp['with_patch']['by_round']['5']) if rp else 'not run'} | "
            f"{fmt_rate(bl['with_patch']['by_round']['5']) if bl else 'not run'} |"
        )
    if base:
        bp = base["patch"]
        L.append(
            f"| Random grids | {fmt_rate(bp['one_shot'])} | {fmt_rate(bp['one_shot_plus_patch'])} | "
            f"{fmt_rate(bp['rejected_one_edit_from_valid'])} | n/a | n/a |"
        )
        L += ["", f"Mean patch search time: {1000 * bp['patch_search_seconds_mean']:.0f} ms per rejected level."]
    for lp in s.get("live_patch_runs", []):
        L += ["",
              f"Live run with the patch inside the loop ({lp['model']}, repair mode, new samples): "
              f"{fmt_rate(lp['accepted'])} accepted, of which {lp['accepted_unpatched']['k']} without "
              f"a patch and {lp['accepted_patched']['k']} patched; {fmt_rate(lp['accepted_at_round1'])} "
              f"after a single model call; {num(lp['calls_per_accepted'])} calls and "
              f"${num(lp['cost_usd_per_accepted'], 5)} per accepted level."]

    L += ["", "### Why proposals were rejected", "",
          "| Model | Mode | Proposals | ok | unsolvable | wrong difficulty | trivial | malformed | duplicate | other |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in llms:
        for mode, m in x["modes"].items():
            v = dict(m["verdicts_all_proposals"])
            known = ["ok", "unsolvable", "wrong_difficulty", "trivial", "malformed", "duplicate"]
            cells = " | ".join(str(v.pop(k, 0)) for k in known)
            L.append(f"| {x['model']} | {mode} | {m['calls']} | {cells} | {sum(v.values())} |")

    L += ["", "### Difficulty control (one-shot proposals)", "",
          "Requested band against the solver's optimal move count, for round-1 "
          "proposals that were solvable.", "",
          "| Model | Band (requested moves) | Solvable / proposals | Median optimal | Range | In band, of solvable | Below / above |",
          "|---|---|---|---|---|---|---|"]
    for x in llms:
        dc = x["one_shot_pooled"]["difficulty"]
        for band, b in dc["by_band"].items():
            lo, hi = b["requested_range"]
            L.append(
                f"| {x['model']} | {band} ({lo}-{hi}) | {b['n_solvable']}/{b['n_proposals']} | "
                f"{num(b['optimal_moves_median'], 1)} | "
                f"{b['optimal_moves_min']}-{b['optimal_moves_max']} | "
                f"{fmt_rate(b['in_band_given_solvable'])} | {b['below_band']} / {b['above_band']} |"
            )
    L += [""]
    for x in llms:
        sp = x["one_shot_pooled"]["difficulty"]["spearman_band_vs_optimal"]
        ib = x["one_shot_pooled"]["difficulty"]["in_band_given_solvable"]
        ci = sp["ci95"]
        L.append(
            f"- {x['model']}: Spearman rho between requested band and measured optimal moves = "
            f"{num(sp['rho'])} (bootstrap 95% CI {num(ci[0])} to {num(ci[1])}, n={sp['n']} solvable "
            f"one-shot proposals); in band given solvable: {fmt_rate(ib)}."
        )

    L += ["", "### Cost and latency per accepted level", "",
          "| Generator | Accepted | Calls or samples per accepted | Seconds per accepted | USD per accepted |",
          "|---|---|---|---|---|"]
    for x in llms:
        for mode, m in x["modes"].items():
            a = m["accepted_within_max_rounds"]
            L.append(
                f"| {x['model']}, {mode} | {a['k']}/{a['n']} | {num(m['calls_per_accepted'])} calls | "
                f"{num(m['latency_s_per_accepted'], 1)} | {num(m['cost_usd_per_accepted'], 5)} |"
            )
    if base:
        a = base["requests_filled"]
        L.append(
            f"| Random + solver filter | {a['k']}/{a['n']} | {num(base['attempts_per_accepted'], 1)} samples | "
            f"{num(base['seconds_per_accepted'], 4)} | 0 |"
        )

    if base:
        L += ["", "### Procedural baseline (random generator + solver filter)", "",
              "| Band | Requests filled | Per-sample acceptance | Samples per accepted | Seconds per accepted |",
              "|---|---|---|---|---|"]
        for band, b in base["by_band"].items():
            L.append(
                f"| {band} | {fmt_rate(b['requests_filled'])} | {fmt_rate(b['per_sample_accept_rate'])} | "
                f"{num(b['attempts_per_accepted'], 1)} | {num(b['seconds_per_accepted'], 4)} |"
            )
        u = base["unfiltered"]
        q = u["optimal_moves_quartiles"]
        L += ["",
              f"Unfiltered random levels (n={u['n']}): {fmt_rate(u['solvable'])} solvable; "
              f"optimal moves of the solvable ones: quartiles {q[0]} / {q[1]} / {q[2]}, "
              f"range {u['optimal_moves_min_max'][0]}-{u['optimal_moves_min_max'][1]}. "
              "Share of all samples landing in each band: "
              + ", ".join(f"{b} {pct(r['rate'])}" for b, r in u["share_in_band"].items()) + "."]

    L += ["", "### Novelty and level features (accepted levels)", "",
          "Nearest-neighbour distance = fraction of cells that differ from the closest "
          "other accepted level of the same size in the same set, minimised over "
          "rotations and reflections. 'Clay matters' = replacing clay with floor "
          "changes solvability or the optimal move count.", "",
          "| Source | Levels | Distinct | Duplicates rejected | NN distance mean (min) | Wall density | Coals | Has dead ends | Unique optimal solution | Uses clay | Clay matters |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]

    def feat_row(name, nov, dup, f):
        return (
            f"| {name} | {nov['n_levels']} | {nov['n_distinct']} | {dup} | "
            f"{num(nov['nearest_neighbour_distance_mean'])} ({num(nov['nearest_neighbour_distance_min'])}) | "
            f"{num(f['wall_density_mean'])} | {num(f['coals_mean'], 1)} | "
            f"{pct(f['levels_with_dead_ends']['rate'])} | {pct(f['unique_optimal_solution']['rate'])} | "
            f"{pct(f['uses_clay']['rate'])} | {pct(f['clay_changes_solution']['rate'])} |"
        )

    for x in llms:
        for mode, m in x["modes"].items():
            if m["novelty"]:
                L.append(feat_row(f"{x['model']}, {mode}", m["novelty"],
                                  m["duplicates_rejected"], m["features_accepted"]))
    if base:
        L.append(feat_row("Random + solver filter", base["novelty"], "n/a",
                          base["features_accepted"]))
    if s.get("llm_partial"):
        L += ["", "### Partial runs (not in the tables above)", ""]
        for x in s["llm_partial"]:
            for mode, m in x["modes"].items():
                L.append(
                    f"- {x['model']}, {mode}: {m['n_requests']} of {m['n_requests_planned']} "
                    f"requests finished; one-shot {fmt_rate(m['per_call_accept_rate_round1'])}, "
                    f"within 5 rounds {fmt_rate(m['accepted_within_max_rounds'])}."
                )
    return "\n".join(L) + "\n"


MIN_REQUESTS_FOR_MAIN_TABLE = 30


def discover_labels() -> list[str]:
    labels = []
    for name in sorted(os.listdir(RAW)):
        for mode in ("repair", "blind"):
            if name.endswith(f"_{mode}.json"):
                label = name[: -len(f"_{mode}.json")]
                if label not in labels:
                    labels.append(label)
    return labels


def summarize_live_patch(label: str) -> dict | None:
    """A run made with --patch: the loop itself applied the solver patch."""
    d = load(f"{label}_repair.json")
    if not d or not d["records"]:
        return None
    recs = list(d["records"].values())
    n = len(recs)
    calls = sum(r["n_calls"] for r in recs)
    acc = [r for r in recs if r["accepted"]]
    return {
        "label": label,
        "model": d["meta"]["model"],
        "n_requests": n,
        "accepted": rate(len(acc), n),
        "accepted_unpatched": rate(sum(1 for r in acc if not r.get("patched")), n),
        "accepted_patched": rate(sum(1 for r in acc if r.get("patched")), n),
        "accepted_at_round1": rate(sum(1 for r in acc if r["accepted_round"] == 1), n),
        "calls": calls,
        "calls_per_accepted": calls / len(acc) if acc else None,
        "cost_usd_total": sum(r["cost_usd"] for r in recs),
        "cost_usd_per_accepted": sum(r["cost_usd"] for r in recs) / len(acc) if acc else None,
        "latency_s_per_accepted": sum(rd["latency_s"] for r in recs for rd in r["rounds"]) / len(acc) if acc else None,
        "by_band": {
            b: rate(sum(1 for r in acc if r["band"] == b), sum(1 for r in recs if r["band"] == b))
            for b in BANDS
        },
        "novelty": hamming_novelty([r["level"] for r in acc]) if acc else None,
        "features_accepted": level_features(recs),
    }


def is_patch_run(label: str) -> bool:
    d = load(f"{label}_repair.json")
    return bool(d and d["meta"].get("patch"))


def main() -> None:
    labels = discover_labels()
    live = [x for x in map(summarize_live_patch, filter(is_patch_run, labels)) if x]
    runs = [x for x in map(summarize_llm, (l for l in labels if not is_patch_run(l))) if x]
    size = lambda x: max(m["n_requests"] for m in x["modes"].values())
    runs.sort(key=lambda x: (x["provider"] != "gemini", x["label"]))
    summary = {
        "bands": BANDS,
        "llm": [x for x in runs if size(x) >= MIN_REQUESTS_FOR_MAIN_TABLE],
        # Runs cut short (for example by a provider quota) are kept out of the
        # tables and reported separately.
        "llm_partial": [x for x in runs if size(x) < MIN_REQUESTS_FOR_MAIN_TABLE],
        "live_patch_runs": live,
        "baseline": summarize_baseline(),
    }
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    md = to_markdown(summary)
    with open(os.path.join("results", "summary.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
