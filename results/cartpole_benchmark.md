# CartPole benchmark

200 seeded episodes per agent (seeds 10000+i), 500-step cap. Learned policies act greedily.

| Agent | Mean steps | 95% CI | Std | Min | Reached 500 |
|---|---:|---:|---:|---:|---:|
| RANDOM | 22.5 | ±1.6 | 11.8 | 8 | 0.0% |
| PD CONTROLLER | 500.0 | ±0.0 | 0.0 | 500 | 100.0% |
| REINFORCE + BASELINE | 500.0 | ±0.0 | 0.0 | 500 | 100.0% |
| ACTOR-CRITIC | 500.0 | ±0.0 | 0.0 | 500 | 100.0% |
| CROSS-ENTROPY SEARCH | 500.0 | ±0.0 | 0.0 | 500 | 100.0% |
