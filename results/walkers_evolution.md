# EVOLVING WALKERS: evolution study (300 generations)

Four runs of the genetic algorithm in `walkers/core.py` (the Python twin of `web/js/walkers-core.js`): 300 generations,
population 80, mutation rate 0.5, default world (flat or hills; gravity 9.8, friction 0.8, restitution 0.2, 10 s per
creature, step 1/240 s). Per-generation numbers, and the champion code of every generation, are in
`results/walkers_evolution.json`. Reproduce with `python -m walkers study --reuse results/walkers_evolution.json` (the
presets from the saved runs), or `python -m walkers evolve --generations 300 --pop 80 --seed S --terrain T --json PATH`
for one run.

## Fitness (the score being optimised)

- Centre-of-mass travel in x over the run, in metres.
- Minus 1.0 m per radian of spine rotation beyond the first radian. The spine is the pair of nodes farthest apart at
  spawn, and the rotation is the sum of |sin| of its turning angle. A body that tumbles therefore pays a lot.
- Plus 0.5 s times the mean forward speed of the second half of the run (the centre of mass's x-gain from the
  midpoint to the end, divided by the second half's seconds). A body that keeps moving scores more than one that lunges.
  A body that falls into a gap gets no speed bonus.

## Best and mean by generation (score, metres)

| Run | Gen 1 | Gen 30 | Gen 60 | Gen 100 | Gen 150 | Gen 200 | Gen 250 | Gen 300 best (mean) |
|---|---|---|---|---|---|---|---|---|
| seed 1, flat | 0.08 | 0.45 | 0.74 | 2.13 | 4.65 | 5.74 | 11.08 | 14.32 (2.78) |
| seed 2, flat | 0.13 | 0.94 | 0.94 | 1.67 | 3.17 | 7.17 | 8.60 | 11.70 (1.07) |
| seed 3, flat | 0.45 | 0.72 | 1.16 | 2.20 | 4.76 | 4.82 | 10.64 | 14.07 (2.20) |
| seed 1, hills | 0.08 | 0.47 | 0.74 | 1.26 | 2.24 | 2.55 | 2.55 | 3.14 (0.28) |

No creature exploded in any run. Species at generation 300: 23, 26, 28 (flat) and 35 (hills).

## What emerged (champions by score, from the study's behaviour check)

| Champion | Run, generation | Score | Distance (10 s) | Nodes, springs, muscles | Behaviour |
|---|---|---|---|---|---|
| runner | seed 1 flat, gen 299 | 14.32 | 13.72 m (1.4 m/s) | 10, 16, 9 | one hop per run (peak 0.68 m), spine rotation 1.1 rad: a fast skip, not a three-hop hopper |
| inchworm | seed 3 flat, gen 300 | 14.07 | 15.30 m (1.5 m/s) | 11, 19, 5 | no hops; the centre of mass never more than 0.31 m up; spine rotation 3.1 rad |
| (best hills) | seed 1 hills, gen 282 | 3.14 | 3.83 m | 8, 13, 8 | slower, uneven ground |

- Every flat run moved from flailing (generation 1, mean score near zero) to bodies that cover 13 m or more in 10 s.
- The two best flat bodies come from different seeds and body plans: a 10-node skipper, and an 11-node low crawler.
- Growth had not stopped at generation 300: the best score rose about 3 m between generations 250 and 300 in the flat
  runs. The study is not converged, so longer runs would likely go further. No body hops three times, so there is no
  hopper preset.

## Earlier numbers (kept as history)

- `results/walkers_evolution_v1.json` and `.md`: 60 generations, the old spin penalty of 0.25 m per radian and no speed
  bonus. The best flat body there reached 1.5 m; the hills body reached 5.7 m, but it was a tumbler.
- An intermediate run (not committed as results) with the 0.25 penalty and the speed bonus found a two-node body that
  tumbles end over end: 16 full turns (105 rad of spine rotation) and 31 small bounces for 99 m, score 79. Its
  distance came from tumbling, not walking. That is why the spin penalty is now 1.0 m per radian beyond one radian,
  and why a body with more than one full turn of spin is labelled `tumbler` (never a hopper or a runner).

## Friction and push-off

Friction is a sensitive part of the gait. The best flat inchworm of the earlier study travels 0.76 m at friction 0.1,
0.75 m at 0.3, 1.07 m at 0.6, 1.53 m at 0.8 and 0.84 m at 1.2. The stick threshold is the static coefficient
(1.2 times friction) times (1 + restitution) times the normal speed, so a foot grips in proportion to how hard it is
pressed, and push-off works at the default 0.8. Friction is not the limit of the gait.

## Time per generation

About 1.3 to 3 s per generation of 80 creatures for the NumPy batch, on the shared laptop with other sessions loading the
CPU. The JavaScript worker is faster per creature (about 3 ms per 10 s creature in node).

## Reproduce

    python -m walkers study --reuse results/walkers_evolution.json   # presets from the saved runs
    python -m walkers evolve --generations 300 --pop 80 --seed 1 --terrain flat --json /tmp/run.json
