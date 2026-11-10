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

<!-- RESULTS -->

## How to run

Python 3.10 or newer. The game, solver, generator and evaluation use only the standard library; `pytest` is needed for the tests, and `node` is used by two tests that check the web page's JavaScript against the Python rules (they are skipped without it).

```
python -m pytest                              # 68 tests

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
