Command: `python -m game2048.tune --evaluate game2048/data/heuristic_weights.json results/game2048_heuristic_weights_v2.json results/game2048_heuristic_weights_v3.json --games 3000 --seed 500000 --out results/game2048_heuristic_greedy_heldout.json`

Greedy one-move lookahead, 3000 games, one batch seed 500000; 95% normal interval.

| Weights file | Smoothness weight | Mean score (95% CI) | Std | Recorded greedy_mean_score |
|---|---|---|---|---|
| `game2048/data/heuristic_weights.json` | -0.129 | 7,074 (6,930 to 7,219) | 4,046 | 7,140.94 |
| `results/game2048_heuristic_weights_v2.json` | 0 | 5,823 (5,709 to 5,936) | 3,168 | 5,736.25 |
| `results/game2048_heuristic_weights_v3.json` | 0 | 5,768 (5,657 to 5,880) | 3,123 | 5,780.90 |
