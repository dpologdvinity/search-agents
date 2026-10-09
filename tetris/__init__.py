"""Tetris: placement search with a weighted board evaluation, and a genetic algorithm that tunes the weights.

board.py     the playfield as row bitmasks: fitting, dropping, locking, clearing lines
pieces.py    the seven tetrominoes and their rotation states
features.py  the nine board features and the hand-picked baseline weights
search.py    enumerate every rotation and column for a piece, score the results, pick the best
game.py      the seeded 7-bag, the policies (random, greedy, lookahead), and the capped game loop
evolve.py    the genetic algorithm that tunes the weights (python -m tetris evolve)
benchmark.py the comparison of four strategies on the same seeded games (python -m tetris benchmark)
terminal.py  turn-based play in the terminal, with the agent's hint (python -m tetris play)
tuned.py     loads tuned.json, the committed tuned weights (cross-entropy v3)
"""
