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
