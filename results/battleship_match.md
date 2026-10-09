`PYTHONPATH=. .venv/bin/python -m battleship match --games 100 --seed 1` (AI = probability), and the same with `--ai hunt`.

```
AI: Probability (Bayesian) (probability)  vs  chance: Chance (fixed odds) (chance)
100 races, seed 1: AI won 100 (100.0%), chance won 0 (0.0%). Races cannot be drawn: the first fleet sunk decides.
mean shots fired when the race ended: AI 46.0, chance 45.5
(63.64 s, 636.42 ms per race)

AI: Hunt/target (hunt)  vs  chance: Chance (fixed odds) (chance)
100 races, seed 1: AI won 100 (100.0%), chance won 0 (0.0%). Races cannot be drawn: the first fleet sunk decides.
mean shots fired when the race ended: AI 50.5, chance 50.0
(0.56 s, 5.59 ms per race)
```
