"""Blackjack as a Markov decision process: exact basic strategy by dynamic programming, learned by Monte Carlo.

Modules:
- rules: card model and the table rules shared by everything here.
- solver: exact action values and the basic-strategy table, by recursion over the dealer and the player.
- learner: Monte Carlo control, which learns the same table from simulated hands.
- game: the playable engine (shoe, rounds, payouts) and the basic-strategy simulation.
- __main__: the terminal game, `strategy`, and `simulate`.
"""
