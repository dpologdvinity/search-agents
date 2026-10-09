Command: `PYTHONPATH=. .venv/bin/python -m routes --compare --cities 12`

Timing column measured on a shared laptop under load (nice 19, cores 7-11); it varies between runs.

```
12 random cities, seed 1
solver                      length   above optimal   seconds
nearest_neighbor_2opt        3.682           1.53%     0.000
simulated_annealing          3.627           0.00%     0.559
genetic_algorithm            3.627           0.00%     1.060
held_karp                    3.627           0.00%     0.071
best tour: genetic_algorithm
+------------------------------------------------------------+
|            .8.                                             |
|           .   .                                            |
|        ...     ...                                         |
|       0           .                                        |
|      .             ...                                     |
|     .                 .             ...3                   |
|     .                  ...    ......  .                    |
|    .                      7...       .                     |
|   .                                 .                      |
|  .                                 .                       |
|  .                                .                        |
| .                                .                         |
|A                                .                          |
|.                               .                           |
|.                              .                            |
| .                            2..........                   |
| .                                       ...........5...    |
| .                                                      ...B|
|  .                                                    ...  |
|  .                                                  ..     |
|  .                                               ...       |
|  .                                             1.          |
|  .                                              .          |
|   .                                              ..        |
|   .                                                .       |
|   .                                                 .      |
|    .                                                 ..    |
|    .                                                   .   |
|    4......................                          ....9  |
|                           .....................6....       |
+------------------------------------------------------------+
```
