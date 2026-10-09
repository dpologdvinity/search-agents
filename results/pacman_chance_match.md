Pac-Man agent: Q-learning (learned) (q). Approximate Q-learning: a linear model scores each move from hand-built features (pellet distance, ghost distance, power pellets, and how long the ghosts' real moves let Pac-Man survive). Its weights were learned from self-play.
Ghosts: AI ghosts (A* routes) vs Chance (fixed odds).
Seeds 10000-10199, 200 games per maze, 2 mazes.

| ghosts | win rate | mean score | mean turns | won / lost / timeout |
|---|---|---|---|---|
| AI ghosts (A* routes) | 64.2% | 1350.3 | 193.9 | 257 / 47 / 96 |
| Chance (fixed odds) | 46.2% | 1513.7 | 110.8 | 185 / 215 / 0 |

Chance odds, relative to the ghost's last move: straight 60%, left 15%, right 15%, back 10%.
