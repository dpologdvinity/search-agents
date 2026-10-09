"""Grid MDP lab: a grid world solved three ways, with the same numbers as the browser page.

grid.py     the layout language, the parameters, and the presets (cliff, rooms, maze, windy)
mdp.py      the grid as a Markov decision process: the transition table and the slip rule
solve.py    value iteration and policy iteration on that table (exact expectations)
learn.py    tabular Q-learning and SARSA, learning from sampled episodes
__main__.py the command line: solve (exact) and learn (from experience)

The JavaScript port is web/js/mdplab-core.js; tests/test_mdplab_parity.py checks the two agree.
"""

from .grid import PRESETS, Grid, Params, Preset, preset
from .learn import QLearner, optimal_reference
from .mdp import GridMDP
from .solve import PolicyIterator, ValueIterator, greedy_policy, q_table

__all__ = [
    "PRESETS",
    "Grid",
    "GridMDP",
    "Params",
    "Preset",
    "PolicyIterator",
    "QLearner",
    "ValueIterator",
    "greedy_policy",
    "optimal_reference",
    "preset",
    "q_table",
]
