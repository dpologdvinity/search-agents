"""Hex: connection on a rhombus of hexagons, where the search is Monte Carlo tree search with RAVE.

board.py      the rules, adjacency, connection search, the fill-then-check random playout, and the swap rule
mcts.py       MCTS: plain UCT, and RAVE (all-moves-as-first statistics blended in by beta)
heuristic.py  the baselines: uniform random, and a one-ply shortest-path race
agents.py     the named agents and search budgets, shared by the CLI, the server, and the benchmark
benchmark.py  seeded head-to-head matches with both colours, and search speed
"""
