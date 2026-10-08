"""Snake: a neural network whose weights are evolved by a genetic algorithm, against classic planners.

board.py      the rules: a grid, the snake turns left, straight, or right, food, walls, and the body
net.py        what the snake senses (14 relative inputs) and a one-hidden-layer NumPy network
agents.py     random, greedy, a BFS planner with a tail-chasing safety check, and the evolved net
evolve.py     the genetic algorithm: tournament selection, uniform crossover, Gaussian mutation, elitism
benchmark.py  apples, deaths, and survival per agent on the same seeded boards
data/         the committed champion network (JSON), with its training history
"""
