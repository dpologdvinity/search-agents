Seeds 1000 onward, the same seeds for every agent. Steps are each agent's own unit (see queens/agents.py);
candidate squares scored is the work measure shared by all four. Time is the median over all runs.

| agent | N | runs | solved | steps (median, solved) | squares scored (median, solved) | time (median) | how the other runs ended |
|---|---:|---:|---:|---:|---:|---:|---|
| backtrack | 8 | 1 | 1 | 218 | 113 | 0.0001 s | — |
| backtrack | 16 | 1 | 1 | 20,088 | 10,052 | 0.0263 s | — |
| backtrack | 32 | 1 | 0 | — | — | 2.21 s | step cap 1 |
| backtrack | 128 | 1 | 0 | — | — | 2.28 s | step cap 1 |
| backtrack | 1,000 | 1 | 0 | — | — | 3.25 s | step cap 1 |
| backtrack | 10,000 | 1 | 0 | — | — | 5.77 s | step cap 1 |
| hill | 8 | 200 | 200 | 22 | 1,440 | 0.0002 s | — |
| hill | 16 | 100 | 100 | 170 | 43,392 | 0.0061 s | — |
| hill | 32 | 30 | 30 | 1,024 | 1,049,088 | 0.142 s | — |
| hill | 128 | 3 | 0 | — | — | 15 s | time cap 3 |
| hill | 1,000 | 2 | 0 | — | — | 20.1 s | time cap 2 |
| anneal | 8 | 200 | 200 | 778 | 778 | 0.0016 s | — |
| anneal | 16 | 100 | 100 | 13,420 | 13,420 | 0.0381 s | — |
| anneal | 32 | 30 | 30 | 712,202 | 712,202 | 1.76 s | — |
| anneal | 128 | 5 | 0 | — | — | 20 s | time cap 5 |
| anneal | 1,000 | 3 | 0 | — | — | 30 s | time cap 3 |
| anneal | 10,000 | 2 | 0 | — | — | 55.6 s | step cap 2 |
| minconf | 8 | 200 | 200 | 24 | 4,826 | 0.0026 s | — |
| minconf | 16 | 200 | 200 | 44 | 9,976 | 0.0043 s | — |
| minconf | 32 | 100 | 100 | 44 | 13,554 | 0.0047 s | — |
| minconf | 128 | 100 | 100 | 52 | 26,432 | 0.0088 s | — |
| minconf | 1,000 | 50 | 50 | 54 | 85,032 | 0.0209 s | — |
| minconf | 10,000 | 20 | 20 | 56 | 624,649 | 0.149 s | — |
| minconf | 100,000 | 3 | 3 | 116 | 11,941,980 | 2.05 s | — |
| minconf | 1,000,000 | 1 | 1 | 47 | 50,094,321 | 13.6 s | — |
| hill N=10000 | | | | | | | not run: one step scores 10^8 squares (N^2), too slow to finish a single step in Python |
