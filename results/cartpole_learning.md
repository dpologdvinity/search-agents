# CartPole learning curves

Training episodes per seed, sampled (not greedy) returns. Discount 0.99.
The learned policy shipped is the seed with the best last 50 episodes.

## Reinforce + Baseline

| Seed | Episodes | First 50 mean | Last 50 mean | Trailing-100 mean reaches 475 at episode |
|---:|---:|---:|---:|---:|
| 0 | 1500 | 25.4 | 496.0 | 376 |
| 1 | 1500 | 26.5 | 494.2 | 384 |
| 2 (shipped) | 1500 | 26.0 | 500.0 | 408 |
| 3 | 1500 | 27.1 | 500.0 | 478 |
| 4 | 1500 | 27.2 | 500.0 | 377 |

## Actor-Critic

| Seed | Episodes | First 50 mean | Last 50 mean | Trailing-100 mean reaches 475 at episode |
|---:|---:|---:|---:|---:|
| 0 (shipped) | 1500 | 25.9 | 500.0 | 344 |
| 1 | 1500 | 23.5 | 500.0 | 381 |
| 2 | 1500 | 27.3 | 500.0 | 370 |
| 3 | 1500 | 27.3 | 500.0 | 347 |
| 4 | 1500 | 23.3 | 500.0 | 382 |

## Cross-Entropy Search

| Seed | Episodes | First 50 mean | Last 50 mean | Trailing-100 mean reaches 475 at episode |
|---:|---:|---:|---:|---:|
| 0 (shipped) | 1500 | 38.2 | 500.0 | 391 |
| 1 | 1500 | 13.0 | 484.4 | 388 |
| 2 | 1500 | 51.3 | 500.0 | 528 |
| 3 | 1500 | 35.4 | 500.0 | 299 |
| 4 | 1500 | 58.0 | 492.2 | 329 |
