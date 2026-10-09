# Lost Robot benchmark

16 trials per row (2 layouts x 4 floor seeds x 2 robot seeds), 100 steps each. Steps counts the first step with the estimate within 0.5 m of the robot. Time is per filter step in Python on one core; the page runs the same algorithms in JavaScript.

## Particle count (default noise)

| setting | median steps to 0.5 m | failures | err @10 (m) | err @25 | err @50 | err @100 | ms/step | belief modes after step 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 particles | 65 | 11/16 | 3.75 | 4.05 | 4.35 | 3.03 | 4.2 | 1.4 |
| 500 particles | 4 | 4/16 | 2.02 | 2.62 | 1.29 | 1.05 | 8.4 | 1.4 |
| 2000 particles | 2 | 1/16 | 2.01 | 1.56 | 0.20 | 0.40 | 29.7 | 1.4 |

## Noise (2000 particles)

| setting | median steps to 0.5 m | failures | err @10 (m) | err @25 | err @50 | err @100 | ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| sensor sigma 0.15 cells (0.07 m) | 2 | 3/16 | 1.56 | 1.24 | 0.94 | 0.76 | 32.1 |
| sensor sigma 0.3 cells (0.15 m) | 2 | 1/16 | 2.01 | 1.56 | 0.20 | 0.40 | 18.4 |
| sensor sigma 0.6 cells (0.30 m) | 2 | 4/16 | 1.46 | 1.74 | 1.33 | 1.39 | 14.8 |
| odometry scale 0.5 | 2 | 2/16 | 1.27 | 1.46 | 0.51 | 0.70 | 14.3 |
| odometry scale 2.0 | 1 | 3/16 | 1.86 | 2.06 | 1.57 | 0.55 | 22.1 |

## Grid filter vs particle filter (same trials)

| setting | median steps to 0.5 m | failures | err @10 (m) | err @25 | err @50 | err @100 | ms/step | init ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| grid filter (16 headings, 0.25 m bins) | 6 | 0/16 | 1.68 | 1.63 | 0.32 | 0.18 | 40.5 | 255 |
| particle filter, 2000 particles | 2 | 1/16 | 2.01 | 1.56 | 0.20 | 0.40 | 14.8 | - |

## Kidnapped robot (teleported at step 40)

| filter | median steps to recover | failures | ms/step |
|---|---:|---:|---:|
| augmented MCL (injection) | 13 | 3/16 | 16.6 |
| plain MCL (no injection) | 18 | 14/16 | 16.4 |
| grid filter (exact Bayes) | 33 | 1/16 | 46.6 |

Augmented MCL injects 1.19% of particles per step on average when nothing is wrong, and 18.9% per step after the kidnap.

