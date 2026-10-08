12x12 board, 200 games per agent, seeds 50000 onward (the same boards for every agent).

| agent | mean apples | median | max | death rate | starved | full | mean steps |
|---|---|---|---|---|---|---|---|
| random | 0.16 | 0 | 2 | 100.0% | 0 | 0 | 26.9 |
| greedy | 20.12 | 19 | 47 | 100.0% | 0 | 0 | 190.9 |
| planner | 56.13 | 49.5 | 137 | 0.0% | 200 | 0 | 1241.3 |
| evolved | 13.69 | 13 | 26 | 38.5% | 123 | 0 | 457.9 |

Agents:
- random: uniform over left, straight, right; no sensing at all
- greedy: the safe move that ends closest to the food
- planner: BFS path to the food, with a tail-chasing safety check
- evolved: the neural net evolved by the genetic algorithm (the committed champion)
