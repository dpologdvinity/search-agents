Command: `PYTHONPATH=. .venv/bin/python -m battleship benchmark --games 100 --seed 1`

Timing column measured on a shared laptop under load (cores 7-11, nice 19); it varies between runs.

```
agent        games  mean shots  median  worst   s/game
random         100        95.4    97.0    100    0.006
hunt           100        51.9    54.0     67    0.007
probability    100        45.9    45.5     64    1.122
(ships occupy 17 of the 100 cells; shots to sink them all, lower is better)
```
