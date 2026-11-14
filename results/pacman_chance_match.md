Pac-Man agent: Q-learning (learned) (q). Approximate Q-learning: a linear model scores each move from hand-built features (pellet distance, ghost distance, power pellets, and how long the ghosts' real moves let Pac-Man survive). Its weights were learned from self-play.
Ghosts: AI ghosts (A* routes) vs Chance (fixed odds).
Seeds 10000-10199, 200 games per maze, 2 mazes.

| ghosts | win rate | mean score | mean turns | won / lost / timeout |
|---|---|---|---|---|
| AI ghosts (A* routes) | 59.8% | 1324.2 | 208.3 | 239 / 69 / 92 |
| Chance (fixed odds) | 38.0% | 1320.5 | 106.1 | 152 / 248 / 0 |

Chance odds, relative to the ghost's last move: straight 60%, left 15%, right 15%, back 10%.
