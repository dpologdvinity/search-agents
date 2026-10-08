Cumulative regret (expected reward lost against the best machine), mean over 100 seeds with a 95% band, 5 machines. Lower is better.

### Bernoulli machines

| Agent | Regret at T=1,000 | Regret at T=10,000 | % best machine, pulls 9,001-10,000 |
| --- | ---: | ---: | ---: |
| greedy | 70.3 ± 23.4 | 679.9 ± 236.0 | 63.0% ± 9.5 |
| epsilon-greedy 0.1 | 44.6 ± 4.7 | 309.9 ± 19.5 | 91.2% ± 1.8 |
| epsilon-greedy decaying | 26.4 ± 7.2 | 161.2 ± 75.3 | 86.0% ± 6.8 |
| UCB1 | 78.6 ± 2.2 | 197.2 ± 9.5 | 95.4% ± 1.3 |
| Thompson sampling | 23.1 ± 2.4 | 35.6 ± 4.1 | 99.6% ± 0.2 |

Lai-Robbins floor: 7.342 ln(t) (mean over machine sets), so 67.6 at T=10,000.

### Gaussian machines (sigma 0.2)

| Agent | Regret at T=1,000 | Regret at T=10,000 | % best machine, pulls 9,001-10,000 |
| --- | ---: | ---: | ---: |
| greedy | 22.4 ± 8.8 | 202.5 ± 87.1 | 80.0% ± 7.9 |
| epsilon-greedy 0.1 | 32.3 ± 2.0 | 282.5 ± 16.7 | 92.1% ± 0.2 |
| epsilon-greedy decaying | 15.2 ± 4.1 | 58.9 ± 36.3 | 94.0% ± 4.7 |
| UCB1 | 8.5 ± 0.6 | 13.0 ± 1.1 | 99.9% ± 0.1 |
| Thompson sampling | 6.5 ± 0.8 | 9.0 ± 0.8 | 99.9% ± 0.0 |

Lai-Robbins floor: 1.448 ln(t) (mean over machine sets), so 13.3 at T=10,000.

### Drifting Bernoulli machines (redrawn every 500 pulls)

| Agent | Regret at T=1,000 | Regret at T=10,000 | % best machine, pulls 9,001-10,000 |
| --- | ---: | ---: | ---: |
| greedy | 169.6 ± 25.4 | 2640.1 ± 96.4 | 26.8% ± 6.1 |
| epsilon-greedy 0.1 | 123.3 ± 14.0 | 2316.1 ± 70.2 | 24.9% ± 5.7 |
| epsilon-greedy decaying | 134.2 ± 18.9 | 2456.0 ± 93.1 | 26.7% ± 6.1 |
| UCB1 | 86.6 ± 5.3 | 978.3 ± 51.8 | 58.5% ± 5.5 |
| Thompson sampling | 91.6 ± 11.1 | 2058.8 ± 72.4 | 27.3% ± 5.6 |
| EXP3 | 241.2 ± 10.8 | 2558.0 ± 64.1 | 25.4% ± 4.3 |
| sliding-window UCB | 92.6 ± 2.2 | 950.5 ± 8.4 | 63.3% ± 2.0 |
