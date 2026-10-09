# EVOLVING WALKERS: evolution study

Four runs of the genetic algorithm in `walkers/core.py` (the Python twin of `web/js/walkers-core.js`):
60 generations, population 80, mutation rate 0.5, default world (gravity 9.8, friction 0.8, restitution 0.2,
10 s per creature, step 1/240 s). Raw per-generation numbers, including the champion of every generation, are in
`results/walkers_evolution.json`. Reproduce with `python -m walkers study` (or `--reuse results/walkers_evolution.json`
to redo only the preset selection).

Fitness is the centre-of-mass travel in metres, less 0.25 per radian of spine rotation beyond the first radian.
"Distance" below is the travel alone. The two differ only when a body rolls or tumbles.

## Best and mean fitness by generation

| Run | Gen 1 best / mean | Gen 10 best | Gen 20 best | Gen 30 best | Gen 40 best | Gen 50 best | Gen 60 best / mean |
|---|---|---|---|---|---|---|---|
| seed 1, flat | 0.080 / -0.021 | 0.261 | 0.395 | 0.763 | 0.835 | 0.835 | 0.835 / 0.092 |
| seed 2, flat | 0.130 / -0.023 | 0.389 | 0.707 | 0.926 | 1.489 | 1.489 | 1.504 / 0.184 |
| seed 3, flat | 0.422 / -0.026 | 1.292 | 1.340 | 1.454 | 1.493 | 1.505 | 1.529 / 0.141 |
| seed 1, hills | 0.080 / -0.021 | 0.261 | 0.366 | 0.450 | 4.808 | 4.808 | 4.808 / 0.390 |

Species count per run at generation 60: 24 (flat seed 1), 29 (flat seed 2), 24 (flat seed 3), 20 (hills seed 1).
No creature exploded (NaN or runaway state) in any run.

## Champions at generation 60

| Run | Nodes | Springs | Muscles | Distance (m) | Fitness | Spine rotation (rad) | Class |
|---|---|---|---|---|---|---|---|
| seed 1, flat | 4 | 5 | 3 | 0.84 | 0.84 | 0.71 | crawler |
| seed 2, flat | 10 | 18 | 16 | 1.50 | 1.50 | 0.96 | inchworm |
| seed 3, flat | 4 | 3 | 2 | 1.53 | 1.53 | 0.42 | inchworm |
| seed 1, hills | 8 | 13 | 8 | 5.73 | 4.81 | 4.70 | inchworm (rolls) |

Classes come from `walkers/study.py:behaviour`: stalled if under 0.5 m; hopper if at least 3 hops; inchworm if the
centre of mass stays under 0.5 m above the ground and it travels at least 1 m; otherwise crawler.

## What emerged

- Every run moved from random flailing (generation 1, mean fitness slightly negative) to bodies that travel a
  metre or more. By generation 30 the flat runs had reached 62% to 95% of their generation-60 best.
- The best flat-ground bodies are small. Seed 3's champion is a 4-node body with 2 muscles that crawls 1.5 m with
  its centre of mass never above 8 cm over the ground. Seed 2's champion is a 10-node, 16-muscle body that
  inches along.
- On hills the champion travelled 5.7 m, but its fitness is 4.8 after the spin penalty: it rolls and tumbles
  rather than walking. It is the furthest body of the study, and it is not a clean gait.
- Speciation kept 13 to 30 species alive; the count rose and fell with the population's topology mix.

## What did not emerge

- **No hoppers.** Across all 63 champions of the four runs, the best has one hop, and none has the three hops the
  hopper rule asks for. A separate 60-generation gaps-terrain probe (seed 1, not in this file) also produced no
  hoppers (best 0.84 m, no hops). Hopping needs a larger reward for airtime, or many more generations, and the
  study did not provide either. The page's HOPPER preset is therefore missing.
- **No running gait.** No champion exceeds 1.5 m on flat ground in 60 generations. Growth had slowed by generation
  40 in every flat run (seed 2 plateaued at 1.489 from generation 40 to 50), so longer runs or a larger population
  would be needed for faster creatures. This is an honest limit of the study, not a tuned result.

## Changes from the brief

- The inchworm test uses the centre of mass's height above the ground under it, measured from first touchdown.
  The first draft measured height above the spawn pose, which is in the air, so every body that drops scored at
  or below zero and the test was trivially true. The fix is in `walkers/study.py:behaviour`.
- The runner preset is the furthest champion of all runs, which is the hills body (5.73 m, rolling). The inchworm
  preset is the furthest low walker on flat ground (1.53 m). Each preset keeps its own terrain field.

## Time per generation

Population 80, 10 s per creature, one process. Measured on the shared laptop with other sessions loading the CPU:
1.7 to 3.1 s per generation (about 1.5 s on a quiet core, the NumPy batch's per-step overhead dominates). The
JavaScript worker is faster per creature (about 3 ms per 10 s run in node), so the page evolves more generations per
second than the CLI.

## Reproduce

    python -m walkers evolve --generations 60 --pop 80 --seed 1
    python -m walkers study                       # all four runs and the presets, several minutes
    python -m walkers study --reuse results/walkers_evolution.json   # presets only
