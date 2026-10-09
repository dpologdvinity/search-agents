# Lost Robot benchmark

16 trials per row (2 layouts x 4 floor seeds x 2 robot seeds), 100 steps each. Steps counts the first step with the estimate within 0.5 m of the robot. Time is per filter step in Python on one core; the page runs the same algorithms in JavaScript.

## Particle count (default noise)

| setting | median steps to 0.5 m | failures | err @10 (m) | err @25 | err @50 | err @100 | ms/step | belief modes after step 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 particles | 12 | 12/16 | 3.65 | 3.27 | 2.74 | 3.73 | 1.0 | 1.4 |
| 500 particles | 2 | 6/16 | 2.01 | 2.14 | 1.86 | 1.79 | 2.0 | 1.4 |
| 2000 particles | 2 | 2/16 | 1.54 | 1.57 | 0.82 | 0.78 | 6.3 | 1.4 |

## Noise (2000 particles)

| setting | median steps to 0.5 m | failures | err @10 (m) | err @25 | err @50 | err @100 | ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| sensor sigma 0.15 cells (0.07 m) | 2 | 1/16 | 1.56 | 1.24 | 1.09 | 0.07 | 8.1 |
| sensor sigma 0.3 cells (0.15 m) | 2 | 2/16 | 1.54 | 1.57 | 0.82 | 0.78 | 6.8 |
| sensor sigma 0.6 cells (0.30 m) | 2 | 4/16 | 1.46 | 1.74 | 0.98 | 1.39 | 6.5 |
| odometry scale 0.5 | 2 | 2/16 | 1.27 | 1.44 | 0.38 | 0.48 | 6.1 |
| odometry scale 2.0 | 1 | 3/16 | 1.86 | 2.06 | 1.57 | 0.82 | 6.1 |

## Grid filter vs particle filter (same trials)

| setting | median steps to 0.5 m | failures | err @10 (m) | err @25 | err @50 | err @100 | ms/step | init ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| grid filter (16 headings, 0.25 m bins) | 6 | 0/16 | 1.68 | 1.63 | 0.32 | 0.18 | 14.5 | 91 |
| particle filter, 2000 particles | 2 | 2/16 | 1.54 | 1.57 | 0.82 | 0.78 | 5.9 | - |

## Kidnapped robot (teleported at step 40)

| filter | median steps to recover | failures | ms/step |
|---|---:|---:|---:|
| augmented MCL (injection) | 11 | 2/16 | 6.1 |
| plain MCL (no injection) | 18 | 14/16 | 5.7 |
| grid filter (exact Bayes) | 33 | 1/16 | 12.0 |

Augmented MCL injects 1.34% of particles per step on average when nothing is wrong, and 13.1% per step after the kidnap.

