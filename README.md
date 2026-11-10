# levelsmith

levelsmith generates levels for Kiln Slide, a small sliding puzzle designed for this project. A language model proposes a level as a text grid, an exact solver checks whether it can be solved and how many moves the shortest solution takes, and rejected proposals go back to the model with the solver's findings. The problem it addresses: a model can write a grid that looks like a puzzle, but it cannot be trusted to know whether the grid is solvable or how hard it is, and a level generator has to guarantee both. This repository measures how often model proposals are valid, whether solver feedback helps the model repair them, how well the requested difficulty matches the measured one, and how all of that compares with a random generator behind the same solver.

Play the generated levels by opening `index.html` in a browser. It is one static file and works offline.

## The rules

Each move slides the ember in one of four directions until a wall stops it; it collects every coal it passes over, and each wet clay tile it leaves bakes into a wall. The level is solved when every coal is collected and the ember comes to rest on the exit.

Two consequences carry most of the puzzle. The ember cannot stop where it likes, so the exit only counts if something stops the ember on it. Clay works in both directions: crossing it creates a new wall to stop against later, and it can also seal a route the ember still needed.

Levels are plain text, 4 to 10 cells on each side:

```
..~.*.      #  wall          .  floor
......      S  ember start   E  exit
E....S      *  coal (0-5)    ~  wet clay (0-6)
.#....
......      Cells outside the grid are walls.
```

The shortest solution of this level is five moves, `U L R D L`: up to the corner, left across the coal and the clay (which bakes), right until the baked clay stops the ember one cell later, down to the wall at row 4, and left onto the exit.

## How the agent loop works

```
 request: band (easy / medium / hard), width x height
     |
     v
 +---------+  reply   +----------------+  level   +-------------------------+
 |  model  | -------> | strict parser  | -------> | exact solver (BFS over  |
 | propose |          | + size limits  |          | every reachable state)  |
 +---------+          +----------------+          +-------------------------+
     ^                    | malformed                  |
     |                    v                            v
     |          +------------------------------------------------+
     |          | verdict: ok | malformed | unsolvable | trivial |
     |          | wrong_difficulty | duplicate | too_complex     |
     |          +------------------------------------------------+
     |                    |                            |
     |  structured        | rejected, rounds left      | ok
     +--- feedback <------+                            v
          (repair mode)                   accepted level + difficulty features
```

- **Solver.** Breadth-first search over (ember position, coals collected, clay still wet). It explores the full reachable graph, so it returns the optimal move count and one optimal solution, and a backward pass from the goal states gives exact dead-end counts. Features reported per level: optimal moves, number of optimal solutions, states expanded before the first solution, reachable states, mean branching factor, dead-end states and ratio, and "trap moves" (moves off the optimal path that enter a dead end).
- **Difficulty bands.** Defined on the optimal move count: easy 3-5, medium 6-9, hard 10-16. Two moves or fewer is rejected as trivial.
- **Feedback.** A rejected proposal returns a verdict code plus solver output: for an unsolvable level, a move-by-move simulation of the model's own stated solution showing where it goes wrong, whether the exit can ever be rested on, which coals are unreachable, and the list of cells where the ember can come to rest; for a wrong difficulty, the solver's shortest solution and how many moves off the band it is.
- **Two modes.** `repair` continues the conversation with that feedback. `blind` sends the original request again with no feedback. Round 1 is the same in both, so round 1 is the one-shot result, and the comparison between the modes isolates what the feedback adds over simply sampling again.
- **Solver patch (no model call).** After a rejection the solver can also search every single-cell edit of the proposal (move the exit; add or remove one wall, coal or clay tile) and keep one that the judge accepts.
- **Guardrails.** The parser rejects anything that is not a rectangular grid of the six legal characters with exactly one start and one exit, within the size and piece limits and of the requested size. Replies over 6,000 characters are rejected unread. The solver stops at 250,000 states. Every run has a hard cap on the number of calls and on spend, checked before each call, with spend reserved for calls in flight. Every call is logged with tokens, latency, cost, retries and HTTP status (`results/raw/calls_*.jsonl`); keys are read from the environment or a local file and are scrubbed from logged errors.

## Results

Setup: 60 fixed requests (3 bands x 20, sizes 6 to 8 per side, master seed 2026), up to 5 rounds per request, temperature 1.0, run on 2026-11-10. Models with full runs: DeepSeek V4.1 Flash with reasoning off (`deepseek-flash`), and Gemini 3.1 Flash-Lite with a 4,000-token thinking budget. Every proposal, verdict and call is in `results/raw/`; these tables are generated by `python -m levelsmith.report` (`results/summary.md`, `results/summary.json`).

Read this first: the Gemini 3.1 Flash-Lite blind-resampling run stopped at the free tier's 500-requests-per-day cap after 20 requests (19 easy, 1 medium), while its repair run finished 59 of 60. For that model, compare repair with blind in the by-band table on the easy row only; the overall "repair minus blind" figure mixes different request sets and should not be used. The DeepSeek comparison is complete (60 requests in each mode).

### Valid-level rate

Accepted = parses, right size, solvable, not a duplicate, and the solver's optimal move count is inside the requested band. Intervals are 95% Wilson.

| Model | One-shot (round 1) | Repair loop, up to 5 rounds | Blind resampling, up to 5 rounds | Repair minus blind |
|---|---|---|---|---|
| gemini-3.1-flash-lite | 12% (10/80; 7-22%) | 36% (21/59; 25-48%) | 24% (5/21; 11-45%) | +12 pts (-12 to +30) |
| deepseek-flash | 10% (12/120; 6-17%) | 22% (13/60; 13-34%) | 38% (23/60; 27-51%) | -17 pts (-32 to -0) |

By band (one-shot pooled over both runs / repair loop / blind resampling):

| Model | Band | One-shot | Repair loop | Blind resampling |
|---|---|---|---|---|
| gemini-3.1-flash-lite | easy | 15% (6/39; 7-30%) | 55% (11/20; 34-74%) | 26% (5/19; 12-49%) |
| gemini-3.1-flash-lite | medium | 10% (2/21; 3-29%) | 37% (7/19; 19-59%) | 0% (0/2; 0-66%) |
| gemini-3.1-flash-lite | hard | 10% (2/20; 3-30%) | 15% (3/20; 5-36%) | n/a |
| deepseek-flash | easy | 8% (3/40; 3-20%) | 20% (4/20; 8-42%) | 35% (7/20; 18-57%) |
| deepseek-flash | medium | 15% (6/40; 7-29%) | 30% (6/20; 15-52%) | 55% (11/20; 34-74%) |
| deepseek-flash | hard | 8% (3/40; 3-20%) | 15% (3/20; 5-36%) | 25% (5/20; 11-47%) |

Cumulative acceptance by round:

| Model | Mode | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | repair | 15% | 20% | 27% | 31% | 36% |
| gemini-3.1-flash-lite | blind | 5% | 5% | 5% | 14% | 24% |
| deepseek-flash | repair | 13% | 15% | 17% | 18% | 22% |
| deepseek-flash | blind | 7% | 10% | 22% | 28% | 38% |

Per-call acceptance in rounds 2-5 (after a first rejection):

| Model | With solver feedback | Blind resample | Difference |
|---|---|---|---|
| gemini-3.1-flash-lite | 7% (12/181; 4-11%) | 5% (4/78; 2-12%) | +2 pts (-6 to +7) |
| deepseek-flash | 2% (5/202; 1-6%) | 10% (19/200; 6-14%) | -7 pts (-12 to -2) |

### Solver one-edit patch

After a rejection the solver tries every single-cell edit (move the exit, add or remove one wall, coal or clay tile) and keeps one that the judge accepts. No extra model calls. The loop columns replay the logged runs with the patch applied after each rejection.

| Source | One-shot | One-shot + patch | Rejected proposals one edit from valid | Repair loop + patch (5 rounds) | Blind + patch (5 rounds) |
|---|---|---|---|---|---|
| gemini-3.1-flash-lite | 12% (10/80; 7-22%) | 51% (41/80; 40-62%) | 44% (31/70; 33-56%) | 92% (54/59; 82-96%) | 100% (21/21; 85-100%) |
| deepseek-flash | 10% (12/120; 6-17%) | 56% (67/120; 47-64%) | 51% (55/108; 42-60%) | 88% (53/60; 78-94%) | 95% (57/60; 86-98%) |
| Random grids | 10% (31/300; 7-14%) | 68% (203/300; 62-73%) | 64% (172/269; 58-69%) | n/a | n/a |

Mean patch search time: 149 ms per rejected level.

Live run with the patch inside the loop (deepseek-flash, repair mode, new samples): 85% (51/60; 74-92%) accepted, of which 10 without a patch and 41 patched; 60% (36/60; 47-71%) after a single model call; 2.33 calls and $0.00014 per accepted level.

### Why proposals were rejected

| Model | Mode | Proposals | ok | unsolvable | wrong difficulty | trivial | malformed | duplicate | other |
|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | repair | 240 | 21 | 181 | 16 | 0 | 22 | 0 | 0 |
| gemini-3.1-flash-lite | blind | 99 | 5 | 81 | 5 | 2 | 6 | 0 | 0 |
| deepseek-flash | repair | 262 | 13 | 179 | 49 | 5 | 16 | 0 | 0 |
| deepseek-flash | blind | 260 | 23 | 193 | 32 | 1 | 11 | 0 | 0 |

### Difficulty control (one-shot proposals)

Requested band against the solver's optimal move count, for round-1 proposals that were solvable.

| Model | Band (requested moves) | Solvable / proposals | Median optimal | Range | In band, of solvable | Below / above |
|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | easy (3-5) | 7/39 | 3.0 | 3-7 | 86% (6/7; 49-97%) | 0 / 1 |
| gemini-3.1-flash-lite | medium (6-9) | 5/21 | 6.0 | 4-15 | 40% (2/5; 12-77%) | 2 / 1 |
| gemini-3.1-flash-lite | hard (10-16) | 6/20 | 8.5 | 4-16 | 33% (2/6; 10-70%) | 4 / 0 |
| deepseek-flash | easy (3-5) | 8/40 | 4.5 | 2-10 | 38% (3/8; 14-69%) | 1 / 4 |
| deepseek-flash | medium (6-9) | 9/40 | 7.0 | 5-9 | 67% (6/9; 35-88%) | 3 / 0 |
| deepseek-flash | hard (10-16) | 15/40 | 6.0 | 4-13 | 20% (3/15; 7-45%) | 12 / 0 |

- gemini-3.1-flash-lite: Spearman rho between requested band and measured optimal moves = 0.68 (bootstrap 95% CI 0.33 to 0.88, n=18 solvable one-shot proposals); in band given solvable: 56% (10/18; 34-75%).
- deepseek-flash: Spearman rho between requested band and measured optimal moves = 0.27 (bootstrap 95% CI -0.11 to 0.60, n=32 solvable one-shot proposals); in band given solvable: 38% (12/32; 23-55%).

### Cost and latency per accepted level

| Generator | Accepted | Calls or samples per accepted | Seconds per accepted | USD per accepted |
|---|---|---|---|---|
| gemini-3.1-flash-lite, repair | 21/59 | 11.43 calls | 305.2 | 0.02429 |
| gemini-3.1-flash-lite, blind | 5/21 | 19.80 calls | 790.7 | 0.04248 |
| deepseek-flash, repair | 13/60 | 20.15 calls | 23.8 | 0.00129 |
| deepseek-flash, blind | 23/60 | 11.30 calls | 14.0 | 0.00057 |
| Random + solver filter | 60/60 | 12.6 samples | 0.0069 | 0 |

### Procedural baseline (random generator + solver filter)

| Band | Requests filled | Per-sample acceptance | Samples per accepted | Seconds per accepted |
|---|---|---|---|---|
| easy | 100% (20/20; 84-100%) | 9% (20/214; 6-14%) | 10.7 | 0.0057 |
| medium | 100% (20/20; 84-100%) | 10% (20/207; 6-14%) | 10.3 | 0.0059 |
| hard | 100% (20/20; 84-100%) | 6% (20/335; 4-9%) | 16.8 | 0.0092 |

Unfiltered random levels (n=5000): 30% (1502/5000; 29-31%) solvable; optimal moves of the solvable ones: quartiles 4 / 7 / 10, range 1-26. Share of all samples landing in each band: easy 8%, medium 11%, hard 8%.

### Novelty and level features (accepted levels)

Nearest-neighbour distance = fraction of cells that differ from the closest other accepted level of the same size in the same set, minimised over rotations and reflections. 'Clay matters' = replacing clay with floor changes solvability or the optimal move count.

| Source | Levels | Distinct | Duplicates rejected | NN distance mean (min) | Wall density | Coals | Has dead ends | Unique optimal solution | Uses clay | Clay matters |
|---|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite, repair | 21 | 21 | 0 | 0.24 (0.10) | 0.35 | 2.1 | 14% | 67% | 43% | 29% |
| gemini-3.1-flash-lite, blind | 5 | 5 | 0 | 0.42 (0.33) | 0.34 | 1.4 | 0% | 100% | 40% | 0% |
| deepseek-flash, repair | 13 | 13 | 0 | 0.31 (0.12) | 0.48 | 1.9 | 46% | 69% | 92% | 62% |
| deepseek-flash, blind | 23 | 23 | 0 | 0.27 (0.12) | 0.48 | 2.0 | 30% | 74% | 91% | 39% |
| Random + solver filter | 60 | 60 | n/a | 0.34 (0.23) | 0.16 | 1.8 | 60% | 85% | 90% | 48% |

### Partial runs (not in the tables above)

- gemini-3-flash-preview, repair: 17 of 18 requests finished; one-shot 53% (9/17; 31-74%) (one-shot only run); one-shot + solver patch 82% (14/17; 59-94%).
- gemini-3.6-flash, repair: 7 of 60 requests finished; one-shot 86% (6/7; 49-97%), within 5 rounds 100% (7/7; 65-100%); one-shot + solver patch 86% (6/7; 49-97%).

DeepSeek spend: account balance USD 0.65 before the first call and USD 0.59 after the last (the balance API reports whole cents); USD 0.037 in the committed call logs, plus about USD 0.014 of early probe calls whose logs were not kept. The cap for this work was USD 0.20. Gemini calls ran on the free tier; their USD figures are list-price equivalents computed from logged tokens (prices read from the providers' pricing pages on 2026-11-10). Gemini seconds include client-side pacing and back-off under free-tier rate limits, so they overstate model latency; DeepSeek seconds do not.

## What it shows and what it does not

What the numbers support:

- One-shot proposals are mostly invalid. DeepSeek Flash produced an acceptable level on the first try for 12 of 120 requests (10%), which is the same rate as unfiltered random grids (31 of 300, 10%). The usual failure is an unsolvable level, most often because nothing ever stops the ember on the exit.
- For DeepSeek Flash without reasoning, solver feedback did worse than resampling: 13 of 60 requests filled with feedback against 23 of 60 with blind resampling. The transcripts show why: with feedback the model changed about 8% of the cells per round and stayed in the same broken layout (unsolvable followed by unsolvable 124 times, unsolvable followed by accepted once).
- The solver-side one-edit patch helped more than either loop. Run live inside the loop on DeepSeek it filled 51 of 60 requests with 2.3 calls per accepted level; the replay of the logged runs had predicted 53 of 60.
- The random generator behind the same solver filled all 60 requests in about 7 ms each, with no model. On validity, cost and speed it beats every model configuration here. Rejected model proposals were also no closer to valid than rejected random grids (51% against 64% one edit from valid).
- Difficulty control by prompt was weak for DeepSeek (see the Spearman figures); hard requests mostly came back too short.
- Two stronger reasoning models were only measured on small samples because of 20-requests-per-day free quotas (listed under partial runs): they were accepted one-shot on easy requests almost every time and failed more often on medium and hard.

What they do not support:

- Nothing here says whether any of these levels are fun. Difficulty is the optimal move count, a proxy; there was no play-testing.
- This is one game, one prompt (not tuned), two models with full runs and 60 requests, so the intervals are wide and the conclusions are about these configurations only.
- The feedback-versus-resampling question is answered for one model without reasoning. For a reasoning model it is answered only on the easy band, with 20 requests per arm.

## Limitations

- Model sampling is not reproducible bit for bit; the request set and the baseline are. Raw replies are saved.
- Requests ran in parallel, so which of two identical proposals counts as the duplicate depends on arrival order. No duplicates occurred in the reported runs.
- A request cut off by a provider error is rerun from round 1; its earlier calls are in the call log but not in the per-level cost.
- The patch search covers single-cell edits only, and the solver refuses levels above 250,000 states (none of the reported proposals reached that).
- Levels are at most 10x10 with at most 5 coals and 6 clay tiles.
- Planned and not finished in the time available: a complete repair-versus-blind run on a reasoning model with balanced bands, and restricting that comparison in the report to requests present in both arms.


## How to run

Python 3.10 or newer. The game, solver, generator and evaluation use only the standard library; `pytest` is needed for the tests, and `node` is used by two tests that check the web page's JavaScript against the Python rules (they are skipped without it).

```
python -m pytest                              # 71 tests

python -m levelsmith.generate --band hard --size 8x8 --provider random
python -m levelsmith.generate --band medium --size 7x7 --provider deepseek --no-thinking --patch

python -m levelsmith.evaluate baseline        # random generator + solver filter
python -m levelsmith.evaluate llm --provider deepseek --no-thinking --mode repair
python -m levelsmith.evaluate llm --provider deepseek --no-thinking --mode blind
python -m levelsmith.evaluate llm --provider gemini:gemini-3.1-flash-lite --label flashlite \
       --thinking-budget 4000 --max-output-tokens 6000 --mode repair
python -m levelsmith.report                   # results/summary.json and results/summary.md
python -m levelsmith.build_demo               # rebuild index.html from accepted levels
```

Model runs need `GOOGLE_API_KEY`, or `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL`, in the environment or in `~/.config/fm-secrets/llm.env`. Runs resume if interrupted and stop at `--max-calls` or `--max-usd`.

| Path | Contents |
|---|---|
| `levelsmith/game.py` | rules, level format, strict parser |
| `levelsmith/solver.py` | exact BFS solver and difficulty features |
| `levelsmith/judge.py` | verdicts and difficulty bands, shared by every generator |
| `levelsmith/agent.py` | prompt, reply extraction, feedback, propose-verify-repair loop |
| `levelsmith/patch.py` | solver-side one-edit repair search |
| `levelsmith/llm.py` | Gemini and OpenAI-compatible clients, budget caps, call log |
| `levelsmith/baseline.py` | random generator behind the same judge |
| `levelsmith/evaluate.py`, `report.py`, `stats.py` | fixed request set, runs, statistics, tables |
| `web/index.template.html`, `index.html` | the playable page (template and built file) |
| `results/raw/` | every proposal, verdict and call from the reported runs |
| `tests/` | rules, solver, agent loop, patch, statistics, web page |
