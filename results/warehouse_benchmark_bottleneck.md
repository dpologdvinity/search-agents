Command: `PYTHONPATH=. .venv/bin/python -m warehouse benchmark --layout bottleneck --robots 6 --instances 10`

Timing column measured on a shared laptop under load (nice 19, cores 7-11); it varies between runs.

```
Bottleneck (bottleneck), 6 robots, 10 instances (seeds 0-9)
planner                 solved  mean SOC  makespan  collisions  CBS nodes  seconds
----------------------------------------------------------------------------------
Conflict-Based Search    10/10      52.9      15.5        0.00      693.6   0.3482
Prioritized planning     10/10      55.9      15.7        0.00          -   0.0020
Independent A*            0/10      47.7      14.3        3.80          -   0.0006
prioritized is suboptimal on 5/10 instances where both solved (mean extra cost 3.00)
```
