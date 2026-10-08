Node budget 20,000 and 10 s per run. Pushes are counted for solved runs; "optimal" means the pushes equal the optimal count.

| search | deadlock pruning | solved | optimal | nodes expanded (solved levels) | nodes expanded (all) | time (all) | states pruned |
|---|---|---:|---:|---:|---:|---:|---:|
| BFS | off | 8/12 | 8/12 | 7,569 | 77,355 | 41.8 s | 0 |
| BFS | on | 10/12 | 10/12 | 22,516 | 38,292 | 32.4 s | 89,045 |
| Greedy (matching) | off | 12/12 | 11/12 | 6,039 | 6,039 | 5.9 s | 0 |
| Greedy (matching) | on | 12/12 | 11/12 | 153 | 153 | 0.3 s | 364 |
| A* (simple) | off | 10/12 | 10/12 | 9,420 | 28,743 | 29.6 s | 0 |
| A* (simple) | on | 12/12 | 12/12 | 10,907 | 10,907 | 11.3 s | 30,981 |
| A* (matching) | off | 12/12 | 12/12 | 269 | 269 | 0.3 s | 0 |
| A* (matching) | on | 12/12 | 12/12 | 154 | 154 | 0.2 s | 328 |

Nodes expanded per level (`-` unsolved within the budget; `*` solved but not optimal):

| level | optimal pushes | BFS | BFS +P | Greedy (matching) | Greedy (matching) +P | A* (simple) | A* (simple) +P | A* (matching) | A* (matching) +P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Warm-up | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Around the corner | 3 | 14 | 5 | 3 | 3 | 3 | 3 | 3 | 3 |
| Pillar | 3 | 9 | 8 | 3 | 3 | 3 | 3 | 3 | 3 |
| Two in a row | 4 | 69 | 13 | 5 | 4 | 6 | 5 | 5 | 4 |
| Side step | 6 | 224 | 101 | 6 | 6 | 9 | 6 | 6 | 6 |
| Crossroads | 8 | 3,317 | 1,106 | 8 | 8 | 8 | 8 | 8 | 8 |
| Open room | 9 | 2,189 | 479 | 9 | 9 | 151 | 140 | 9 | 9 |
| Stacks | 11 | 1,746 | 357 | 46 | 11 | 429 | 178 | 15 | 11 |
| Forklift | 13 | - | 9,015 | 5,895 | 52 | 817 | 330 | 124 | 30 |
| Shelf | 15 | - | 11,431 | 15 | 15 | 7,993 | 2,495 | 15 | 15 |
| Aisle | 18 | - | - | 29* | 22* | - | 4,917 | 61 | 45 |
| Dock | 19 | - | - | 19 | 19 | - | 2,821 | 19 | 19 |
