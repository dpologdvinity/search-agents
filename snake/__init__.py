"""Snake: a neural network whose weights are evolved by a genetic algorithm, against classic planners.

board.py      the rules: a grid, the snake turns left, straight, or right, food, walls, and the body
net.py        what the snake senses (17 relative inputs) and a one-hidden-layer NumPy network
evolve.py     the genetic algorithm: tournament selection, uniform crossover, Gaussian mutation, elitism
evaluator.py  the evaluation function: eight features of the position after each move, weighted and summed
cem.py        the cross-entropy method that evolves the evaluator's eight weights
agents.py     random, greedy, a BFS planner with a tail-chasing safety check, the evolved net, and the evaluator
benchmark.py  apples, deaths, and survival per agent on the same seeded boards
data/         the committed net and evaluator weights (JSON), with their training histories
"""
