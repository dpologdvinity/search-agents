# Lights Out 5x5 statistics

Command: `PYTHONPATH=. .venv/bin/python -c "<code in results/lightsout_stats.json>"` (the full code is the "command" field of the JSON)

Claims checked: rank 23 (nullity 2), exactly 4 solutions per solvable board, share of solvable boards 1/4, mean lightest solution, mean heaviest-minus-lightest, time per solve. Boards from random_board: solvable=True boards are solvable by construction, so the uniform sample (solvable=False) is the one that measures the 1/4.
