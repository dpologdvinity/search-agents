Command: `PYTHONPATH=. .venv/bin/python -m warehouse benchmark --layout aisles --robots 8 --instances 10`

Timing column measured on a shared laptop under load (nice 19, cores 7-11); it varies between runs.

```
Shelf aisles (aisles), 8 robots, 10 instances (seeds 0-9)
planner                 solved  mean SOC  makespan  collisions  CBS nodes  seconds
----------------------------------------------------------------------------------
Conflict-Based Search    10/10      75.7      17.4        0.00      135.0   0.0561
Prioritized planning     10/10      79.5      17.1        0.00          -   0.0007
Independent A*            0/10      72.9      17.0        3.50          -   0.0006
prioritized is suboptimal on 9/10 instances where both solved (mean extra cost 3.80)
```
