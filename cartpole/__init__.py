"""CartPole: balance a pole on a cart by pushing left or right, learned with policy gradients in NumPy.

env.py        the physics (Barto, Sutton and Anderson 1983 equations, Euler steps) and the failure rules
nets.py       the small MLPs (policy and critic), their forward pass, hand-written backprop, and Adam
agents.py     the agents that act in the environment: random, PD controller, and trained networks
train.py      REINFORCE with a baseline, actor-critic, and a cross-entropy-method search
benchmark.py  seeded evaluation of every agent, and the committed result files
"""
