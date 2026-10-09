"""Decision trees and random forests: CART splits, impurity criteria, pruning, bagging, out-of-bag scores.

rng.py      mulberry32, the seeded generator the JavaScript lab reproduces bit for bit
data.py     seeded 2-D presets (blobs, XOR, checkerboard, spirals, nested rings, noisy diagonal) and feature expansion
tree.py     impurity, the exhaustive best-split search, the breadth-first CART builder, cost-complexity pruning
forest.py   bagged trees with random feature subsets, out-of-bag scoring and impurity-based feature importances
cli.py      the command line: grow a tree or train a forest and print the result
"""

from .forest import RandomForest
from .tree import DecisionTree, Node, prune

__all__ = ["DecisionTree", "Node", "RandomForest", "prune"]
