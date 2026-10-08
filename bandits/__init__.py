"""Neon casino: multi-armed bandits, the exploration versus exploitation problem.

rng.py        seeded random numbers that the JavaScript page reproduces exactly (mulberry32, gamma, beta)
env.py        the casino: seeded Bernoulli, Gaussian and drifting machines, with the outcome table rolled
agents.py     greedy, epsilon-greedy (fixed and decaying), UCB1, Thompson, EXP3, sliding-window UCB
sim.py        the play loop, scoring a human's pulls, and confidence bands over seeds
benchmark.py  every agent on every casino over many seeds, against the Lai-Robbins floor
__main__.py   the command line: play, watch, benchmark
"""
