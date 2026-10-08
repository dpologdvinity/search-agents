Sensor radius 2 (square), 5 seeded maps per row, 4-connected grid, Manhattan heuristic, unknown cells planned as free. Start top-left, goal bottom-right. Both runs drive the same routes with the same belief changes; expansions are queue pops (D* Lite) or closed cells (A*), averaged over the maps.

| size | density | steps | replans | D* init exp | D* replan exp | A* init exp | A* replan exp | D* total / A* total | D* ms | A* ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 21x21 | 10% | 45 | 15.6 | 441 | 32 | 41 | 384 | 473 / 425 (1.11x) | 14 | 2 |
| 21x21 | 20% | 47 | 24.4 | 441 | 57 | 41 | 551 | 498 / 592 (0.84x) | 7 | 2 |
| 21x21 | 30% | 46 | 30.2 | 441 | 59 | 41 | 718 | 500 / 759 (0.66x) | 6 | 2 |
| 41x41 | 10% | 87 | 29.6 | 1681 | 55 | 81 | 1279 | 1736 / 1360 (1.28x) | 46 | 9 |
| 41x41 | 20% | 100 | 56.4 | 1681 | 169 | 81 | 2353 | 1850 / 2434 (0.76x) | 41 | 8 |
| 41x41 | 30% | 127 | 86.4 | 1681 | 467 | 81 | 3920 | 2148 / 4001 (0.54x) | 63 | 22 |
| 61x61 | 10% | 135 | 47.0 | 3721 | 100 | 121 | 3030 | 3821 / 3151 (1.21x) | 88 | 25 |
| 61x61 | 20% | 147 | 82.2 | 3721 | 241 | 121 | 5076 | 3962 / 5197 (0.76x) | 93 | 23 |
| 61x61 | 30% | 196 | 131.2 | 3721 | 696 | 121 | 8154 | 4417 / 8275 (0.53x) | 182 | 68 |

Cost check: 0 replans where D* Lite and A* disagreed on the cost from the rover's cell (out of 2515 replans across 45 maps).
