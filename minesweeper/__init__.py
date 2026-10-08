"""Minesweeper: a constraint-satisfaction and probability agent for the classic mine-finding game.

board.py      the rules: hidden mines, the reveal flood fill, win and loss, and cell names
inference.py  what the agent can prove: single-cell rules, the constraint components of the frontier,
              exact mine counts per component (backtracking with memoization), and exact probabilities
agents.py     four agents of rising strength, and the loop that plays one seeded game
benchmark.py  win rate and guesses per game on fixed seeds, for the three classic sizes
"""
