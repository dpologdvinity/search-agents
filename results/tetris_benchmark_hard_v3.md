Board 10 rows, piece cap 100000. The 'ga' row uses the weights in results/tetris_cem_weights.json.

| Strategy | Games | Mean lines | 95% CI | Median | Min-max | Hit cap | Topped out | Mean pieces | s per game |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random placement | 30 | 0.0 | ± 0.1 | 0 | 0-1 | 0 / 30 | 30 | 14 | 0.01 |
| hand-picked weights | 30 | 2128.5 | ± 895.6 | 867 | 80-8743 | 0 / 30 | 30 | 5342 | 5.16 |
| CEM-tuned weights (v3) | 30 | 6702.0 | ± 2194.9 | 6305 | 84-24870 | 0 / 30 | 30 | 16776 | 26.72 |
