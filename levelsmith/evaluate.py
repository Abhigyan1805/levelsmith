"""Evaluation runner.

  python -m levelsmith.evaluate llm --provider deepseek --mode repair
  python -m levelsmith.evaluate llm --provider gemini --mode blind
  python -m levelsmith.evaluate baseline

Every run uses the same fixed request list (band, width, height), derived
from a master seed. Raw records are written to results/raw/ after every
request, and a run that is interrupted resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import baseline
from .agent import SYSTEM_PROMPT, run_request
from .judge import BANDS
from .llm import Budget, deepseek_balance, make_client

MASTER_SEED = 2026
PER_BAND = 20
SIZES = (6, 7, 8)
RAW_DIR = os.path.join("results", "raw")


def make_requests(master_seed: int = MASTER_SEED, per_band: int = PER_BAND) -> list[dict]:
    reqs = []
    for band in BANDS:
        for i in range(per_band):
            rng = random.Random(f"{master_seed}-{band}-{i}")
            reqs.append(
                {
                    "id": f"{band}-{i:02d}",
                    "band": band,
                    "width": rng.choice(SIZES),
                    "height": rng.choice(SIZES),
                    "seed": rng.randrange(2**31),
                }
            )
    return reqs


def _save(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


FINAL = ("accepted", "max_rounds")


def run_llm(args) -> None:
    label = args.label or args.provider.split(":")[0]
    path = os.path.join(RAW_DIR, f"{label}_{args.mode}.json")
    log_path = os.path.join(RAW_DIR, f"calls_{label}.jsonl")
    requests = make_requests(per_band=args.per_band)
    if args.limit:
        requests = [r for r in requests if int(r["id"].split("-")[1]) < args.limit]

    data = {"meta": {}, "records": {}, "incomplete": []}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    records = data["records"]

    budget = Budget(args.max_calls, args.max_usd, reserve_usd=args.reserve_usd)
    kw = {}
    extra = {}
    if args.provider.startswith("gemini") and args.thinking_budget is not None:
        kw["thinking_budget"] = args.thinking_budget
    if args.provider.startswith("deepseek") and args.no_thinking:
        kw["extra_body"] = {"thinking": {"type": "disabled"}}
        extra["thinking"] = "disabled"
    client = make_client(
        args.provider, budget, log_path,
        max_output_tokens=args.max_output_tokens,
        min_interval_s=args.min_interval, **kw,
    )
    is_deepseek = client.provider == "deepseek"
    meta = data["meta"]
    meta.setdefault("started_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    meta.update(
        {
            "provider": client.provider,
            "model": client.model,
            "mode": args.mode,
            "max_rounds": args.max_rounds,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "thinking_budget": args.thinking_budget,
            "workers": args.workers,
            "master_seed": MASTER_SEED,
            "n_requests": len(requests),
            "system_prompt": SYSTEM_PROMPT,
            "patch": args.patch,
            **extra,
        }
    )
    if is_deepseek and "deepseek_balance_before" not in meta:
        meta["deepseek_balance_before"] = deepseek_balance()

    seen = {r["canonical"] for r in records.values() if r.get("accepted")}
    lock = threading.Lock()
    todo = [r for r in requests if r["id"] not in records]
    print(f"{label}/{args.mode}: {len(records)} done, {len(todo)} to run", flush=True)

    def work(req: dict) -> None:
        rec = run_request(
            client, args.mode, req["band"], req["width"], req["height"], seen,
            max_rounds=args.max_rounds, temperature=args.temperature,
            tag={"request": req["id"], "mode": args.mode}, lock=lock,
            patch=args.patch,
        )
        rec["id"] = req["id"]
        with lock:
            if rec["stop_reason"] in FINAL:
                records[req["id"]] = rec
            else:
                data["incomplete"].append(rec)
            meta["calls_this_run"] = budget.calls
            meta["usd_this_run"] = round(budget.usd, 6)
            _save(path, data)
            print(
                f"  {req['id']} {req['width']}x{req['height']} -> {rec['stop_reason']} "
                f"after {rec['n_calls']} call(s); budget {budget.calls} calls "
                f"${budget.usd:.4f}",
                flush=True,
            )

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, todo))

    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["total_calls_logged"] = meta.get("total_calls_logged", 0) + budget.calls
    meta["total_usd_logged"] = round(meta.get("total_usd_logged", 0.0) + budget.usd, 6)
    if is_deepseek:
        meta["deepseek_balance_after"] = deepseek_balance()
    _save(path, data)
    n_acc = sum(1 for r in records.values() if r["accepted"])
    print(f"done: {n_acc}/{len(records)} accepted, {len(requests) - len(records)} not finished")


def run_baseline(args) -> None:
    path = os.path.join(RAW_DIR, "baseline.json")
    requests = make_requests(per_band=args.per_band)
    seen: set[str] = set()
    records = {}
    for req in requests:
        rec = baseline.generate(
            req["band"], req["width"], req["height"], req["seed"], seen,
            max_attempts=args.max_attempts,
        )
        rec["id"] = req["id"]
        if rec["accepted"]:
            seen.add(rec["canonical"])
        records[req["id"]] = rec

    # Unfiltered sample: what does the random generator produce before filtering?
    rng = random.Random(MASTER_SEED)
    from .judge import judge

    sample = []
    t0 = time.perf_counter()
    for _ in range(args.sample):
        w, h = rng.choice(SIZES), rng.choice(SIZES)
        v = judge(baseline.random_level_text(rng, w, h), "medium")
        sample.append(
            {
                "solvable": v.solvable,
                "optimal_moves": v.solve.optimal_moves if v.solvable else None,
            }
        )
    meta = {
        "master_seed": MASTER_SEED,
        "max_attempts": args.max_attempts,
        "n_requests": len(requests),
        "sample_size": args.sample,
        "sample_seconds": round(time.perf_counter() - t0, 3),
    }
    _save(path, {"meta": meta, "records": records, "unfiltered_sample": sample})
    n_acc = sum(1 for r in records.values() if r["accepted"])
    print(f"baseline: {n_acc}/{len(records)} requests filled")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("llm", help="run the LLM generator on the fixed request list")
    a.add_argument("--provider", required=True, help="gemini | deepseek [:model]")
    a.add_argument("--mode", required=True, choices=["repair", "blind"])
    a.add_argument("--label", default=None)
    a.add_argument("--per-band", type=int, default=PER_BAND)
    a.add_argument("--limit", type=int, default=0, help="only the first N requests per band")
    a.add_argument("--max-rounds", type=int, default=5)
    a.add_argument("--temperature", type=float, default=1.0)
    a.add_argument("--max-calls", type=int, default=320)
    a.add_argument("--max-usd", type=float, default=0.05)
    a.add_argument("--reserve-usd", type=float, default=0.002)
    a.add_argument("--max-output-tokens", type=int, default=2000)
    a.add_argument("--thinking-budget", type=int, default=None)
    a.add_argument("--no-thinking", action="store_true")
    a.add_argument("--patch", action="store_true",
                   help="apply the solver's one-edit patch after each rejection")
    a.add_argument("--min-interval", type=float, default=0.0)
    a.add_argument("--workers", type=int, default=4)
    a.set_defaults(func=run_llm)

    b = sub.add_parser("baseline", help="random generator + solver filter")
    b.add_argument("--per-band", type=int, default=PER_BAND)
    b.add_argument("--max-attempts", type=int, default=2000)
    b.add_argument("--sample", type=int, default=5000)
    b.set_defaults(func=run_baseline)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
