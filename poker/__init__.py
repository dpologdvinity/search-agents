"""Poker with imperfect information: Counterfactual Regret Minimization on Kuhn poker and Leduc hold'em.

kuhn.py, leduc.py  the games: deals, betting, showdowns, and the information set each seat sees
betting.py         the fixed-limit betting rules both games share
tree.py            an explicit game tree, exact expected value, and exact best response / exploitability
cfr.py             vanilla CFR and CFR+ (regret matching on the tree, average strategy, checkpoints)
train.py           training runs and the committed files (strategy table, exploitability curves)
strategy.py        the trained table: JSON on disk, sampling for the bot, hint text
bots.py            the CFR bot, three baselines, and the seeded head-to-head runner
chance.py          the chance opponent: fixed action odds, no card and no search
benchmark.py       the benchmark table written to results/
"""
