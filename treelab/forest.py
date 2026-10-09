"""Random forest: bagged CART trees, each split tried on a random subset of features.

Each tree gets its own seed drawn from a master stream. The tree draws a bootstrap sample (n rows with
replacement) first, then one feature subset per split it considers. Rows a tree never saw are its out-of-bag
rows: they give an honest accuracy estimate without a separate test set. Feature importance is the
impurity decrease each feature buys, summed over the tree's splits (weighted by sample count) and normalised
per tree, then averaged over trees.
"""

from __future__ import annotations

import math

import numpy as np

from .rng import Mulberry32
from .tree import DecisionTree, preorder


class RandomForest:
    def __init__(self, n_estimators: int = 50, *, criterion: str = "gini", max_depth: int = 5,
                 min_samples_leaf: int = 1, max_features: int | None = None, bootstrap: bool = True,
                 seed: int = 1):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # None means floor(sqrt(number of features))
        self.bootstrap = bootstrap
        self.seed = seed
        self.trees: list[DecisionTree] = []
        self.in_bag: list[np.ndarray] = []
        self.n_classes = 0
        self.n_features = 0
        self.oob_accuracy: float | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None) -> RandomForest:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n = len(y)
        self.n_features = X.shape[1]
        self.n_classes = int(n_classes if n_classes is not None else y.max() + 1)
        mf = self.max_features if self.max_features is not None else max(1, math.isqrt(self.n_features))
        master = Mulberry32(self.seed)
        self.trees, self.in_bag = [], []
        for _ in range(self.n_estimators):
            rng = Mulberry32(int(master.next() * 4294967296.0))
            if self.bootstrap:
                idx = np.array([rng.rand_int(n) for _ in range(n)], dtype=np.int64)
            else:
                idx = np.arange(n, dtype=np.int64)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            tree = DecisionTree(self.criterion, self.max_depth, self.min_samples_leaf,
                                max_features=mf, rng=rng).fit(X, y, self.n_classes, idx=idx)
            self.trees.append(tree)
            self.in_bag.append(mask)
        self._score_oob(X, y)
        self._importances()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average of the trees' leaf class distributions."""
        total = np.zeros((len(X), self.n_classes))
        for tree in self.trees:
            total = total + tree.predict_proba(X)
        return total / len(self.trees)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def _score_oob(self, X: np.ndarray, y: np.ndarray) -> None:
        """Each row is scored by the trees that did not train on it; rows no tree left out are skipped."""
        n = len(y)
        acc = np.zeros((n, self.n_classes))
        seen = np.zeros(n)
        for tree, mask in zip(self.trees, self.in_bag):
            rows = np.flatnonzero(~mask)
            if rows.size:
                acc[rows] += tree.predict_proba(X[rows])
                seen[rows] += 1
        scored = seen > 0
        if scored.any():
            pred = (acc[scored] / seen[scored, None]).argmax(axis=1)
            self.oob_accuracy = float(np.mean(pred == y[scored]))
        else:
            self.oob_accuracy = None

    def _importances(self) -> None:
        total = np.zeros(self.n_features)
        for tree in self.trees:
            imp = np.zeros(self.n_features)
            for node in preorder(tree.root):
                if not node.is_leaf:
                    imp[node.feature] += node.n * node.gain
            s = imp.sum()
            total += imp / s if s > 0 else imp
        self.feature_importances_ = total / len(self.trees)
