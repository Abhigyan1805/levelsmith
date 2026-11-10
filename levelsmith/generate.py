"""Generate one level from the command line.

  python -m levelsmith.generate --band medium --size 7x7 --provider deepseek --patch
  python -m levelsmith.generate --band hard --size 8x8 --provider random

Prints the accepted level, the solver's optimal solution and difficulty
features. Every model call is appended to results/raw/calls_cli.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from . import baseline
from .agent import run_request
from .judge import BANDS
from .llm import Budget, make_client


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--band", choices=list(BANDS), default="medium")
    ap.add_argument("--size", default="7x7", help="WIDTHxHEIGHT, each 4..10")
    ap.add_argument("--provider", default="random",
                    help="random | deepseek | gemini | gemini:<model>")
    ap.add_argument("--mode", choices=["repair", "blind"], default="repair")
    ap.add_argument("--patch", action="store_true",
                    help="let the solver try a one-edit fix after each rejection")
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--max-usd", type=float, default=0.01)
    ap.add_argument("--thinking-budget", type=int, default=None)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="print the full record as JSON")
    args = ap.parse_args()
    width, height = (int(x) for x in args.size.lower().split("x"))

    if args.provider == "random":
        seed = args.seed if args.seed is not None else random.randrange(2**31)
        rec = baseline.generate(args.band, width, height, seed, set())
    else:
        kw = {}
        if args.provider.startswith("gemini") and args.thinking_budget is not None:
            kw["thinking_budget"] = args.thinking_budget
        if args.provider.startswith("deepseek") and args.no_thinking:
            kw["extra_body"] = {"thinking": {"type": "disabled"}}
        budget = Budget(args.max_rounds, args.max_usd, reserve_usd=args.max_usd / 10)
        client = make_client(args.provider, budget, "results/raw/calls_cli.jsonl", **kw)
        rec = run_request(client, args.mode, args.band, width, height, set(),
                          max_rounds=args.max_rounds, patch=args.patch)
    if args.json:
        print(json.dumps(rec, indent=1))
        return
    if not rec["accepted"]:
        codes = [r["verdict"]["code"] for r in rec.get("rounds", [])]
        print(f"no level accepted ({rec.get('stop_reason', 'attempt cap')}); verdicts: {codes}")
        sys.exit(1)
    f = rec["features"]
    print(rec["level"])
    how = f"{rec['n_calls']} model call(s)" if "n_calls" in rec else f"{rec['attempts']} random sample(s)"
    if rec.get("patched"):
        how += f", solver patch: {rec['rounds'][-1]['patch']}"
    print(f"\n{args.band}: optimal {f['optimal_moves']} moves ({f['solution']}), "
          f"{f['reachable_states']} reachable states, {f['dead_end_states']} dead-end states, "
          f"{f['n_optimal_solutions']} optimal solution(s); {how}")


if __name__ == "__main__":
    main()
