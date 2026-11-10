"""Build the playable page: embed accepted levels into a single index.html.

  python -m levelsmith.build_demo

Reads results/raw/*.json and web/index.template.html, writes index.html at the
repository root. The page has no external dependencies and works from file://.
"""

from __future__ import annotations

import json
import os

from .game import is_solution, parse_level
from .judge import BANDS

RAW = os.path.join("results", "raw")
TEMPLATE = os.path.join("web", "index.template.html")
OUT = "index.html"

# (file, source id, label, note shown under the level list)
SOURCES = [
    ("flashlite_repair.json", "flashlite-repair", "Gemini 3.1 Flash-Lite + solver feedback",
     "Proposed by Gemini 3.1 Flash-Lite and accepted by the solver, with solver feedback after rejections."),
    ("flashlite_blind.json", "flashlite-blind", "Gemini 3.1 Flash-Lite, resampled",
     "Proposed by Gemini 3.1 Flash-Lite and accepted by the solver, resampling without feedback."),
    ("gemini36_repair.json", "gemini36-repair", "Gemini 3.6 Flash (partial run)",
     "Proposed by Gemini 3.6 Flash and accepted by the solver. The run stopped at the free-tier daily quota."),
    ("deepseek_repair.json", "deepseek-repair", "DeepSeek Flash + solver feedback",
     "Proposed by DeepSeek V4.1 Flash (reasoning off) and accepted by the solver, with solver feedback."),
    ("deepseek_blind.json", "deepseek-blind", "DeepSeek Flash, resampled",
     "Proposed by DeepSeek V4.1 Flash (reasoning off) and accepted by the solver, no feedback."),
    ("deepseekpatch_repair.json", "deepseek-patch", "DeepSeek Flash + solver one-edit patch",
     "Proposed by DeepSeek V4.1 Flash; where the solver rejected a proposal, it searched for a single-cell edit that makes it valid."),
    ("baseline.json", "random", "Random generator + solver filter",
     "Random grids kept only if the solver accepts them. No language model involved."),
]


def collect() -> tuple[list[dict], list[dict]]:
    levels, sources = [], []
    band_order = {b: i for i, b in enumerate(BANDS)}
    for fname, sid, label, note in SOURCES:
        path = os.path.join(RAW, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            records = json.load(f)["records"]
        got = []
        for rid, r in records.items():
            if not r.get("accepted"):
                continue
            lvl = parse_level(r["level"])
            feat = r["features"]
            # Never ship a level whose stored solution does not replay.
            assert is_solution(lvl, feat["solution"]), (sid, rid)
            got.append(
                {
                    "id": rid,
                    "src": sid,
                    "band": r["band"],
                    "w": lvl.width,
                    "h": lvl.height,
                    "rows": list(lvl.rows),
                    "opt": feat["optimal_moves"],
                    "sol": feat["solution"],
                }
            )
        got.sort(key=lambda m: (band_order[m["band"]], m["opt"], m["id"]))
        if got:
            levels += got
            sources.append({"id": sid, "label": label, "note": note})
    return levels, sources


def main() -> None:
    levels, sources = collect()
    with open(TEMPLATE) as f:
        html = f.read()
    assert "/*LEVELS*/[]" in html and "/*SOURCES*/[]" in html
    html = html.replace("/*LEVELS*/[]", json.dumps(levels, separators=(",", ":")))
    html = html.replace("/*SOURCES*/[]", json.dumps(sources, separators=(",", ":")))
    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT}: {len(levels)} levels from {len(sources)} sources, {len(html)} bytes")


if __name__ == "__main__":
    main()
