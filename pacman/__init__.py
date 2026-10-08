"""Pac-Man: a maze with pellets, power pellets, and ghosts, plus baseline and learned agents.

Modules, from the bottom up:
  mazes      hand-drawn layouts, parsed into integer-indexed grids
  search     breadth-first distances and A* routes on the grid
  ghosts     ghost personalities, their targets, and their moves
  engine     the turn-by-turn rules, deterministic given a seed
  features   hand-built features of the position after a Pac-Man move
  lookahead  how long Pac-Man survives after a move, playing the real ghost AI forward
  agents     random and reflex baselines, and the approximate Q-learning policy
  qlearning  the Q-learning update and the self-play training loop
  episode    plays one game with an agent and records every turn
  benchmark  win rate and average score per agent
  terminal   ASCII rendering, human play, and watch mode
"""
