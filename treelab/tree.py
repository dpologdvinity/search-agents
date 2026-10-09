"""CART decision trees: greedy splits, Gini or entropy, depth and leaf-size limits, cost-complexity pruning.

A split sends a sample left when x[feature] < threshold. The best split of a node is found by sorting the
node's samples on each candidate feature and testing the midpoint between every pair of distinct values
(exhaustive, so no threshold is missed). Ties are broken deterministically: the earlier feature in the
candidate list wins, then the smaller threshold. The web lab runs the same arithmetic in the same order
(web/js/treelab-core.js), so the two builds agree node for node.

The builder is breadth-first: a FIFO queue holds the frontier, and each step() splits the next node that can
be split. Growing fully gives the same tree as recursion; stepping just lets the lab animate it.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .rng import Mulberry32

EPS = 1e-12  # gains and risks closer than this count as ties
CRITERIA = ("gini", "entropy")


def impurity(counts: Sequence[int], criterion: str = "gini") -> float:
    """Gini 1 - sum p^2, or entropy -sum p log2 p, of a class-count vector. Zero for an empty or pure node."""
    n = sum(counts)
    if n == 0:
        return 0.0
    acc = 0.0
    if criterion == "gini":
        for c in counts:
            p = c / n
            acc = acc + p * p
        return 1.0 - acc
    if criterion != "entropy":
        raise ValueError(f"unknown criterion {criterion!r}")
    for c in counts:
        if c > 0:
            p = c / n
            acc = acc - p * math.log2(p)
    return acc


def _impurity_rows(cnt: np.ndarray, n: np.ndarray, criterion: str) -> np.ndarray:
    """impurity() for many count rows at once; same operations in the same order as the scalar version."""
    p = cnt / n[:, None]
    acc = np.zeros(len(n))
    if criterion == "gini":
        for c in range(cnt.shape[1]):
            pc = p[:, c]
            acc = acc + pc * pc
        return 1.0 - acc
    for c in range(cnt.shape[1]):
        pc = p[:, c]
        acc = acc - pc * np.log2(np.where(pc > 0, pc, 1.0))
    return acc


def best_split(X: np.ndarray, y: np.ndarray, idx: np.ndarray, features: Sequence[int], criterion: str,
               min_leaf: int, n_classes: int) -> tuple[int, float, float] | None:
    """Best (feature, threshold, gain) over the candidate features, or None if no split helps.

    gain is the decrease in weighted impurity, parent minus the size-weighted children. Both children must
    keep at least min_leaf samples. Returns None when no candidate beats EPS.
    """
    m = len(idx)
    ys = y[idx]
    total = np.bincount(ys, minlength=n_classes).astype(np.int64)
    parent = impurity(total.tolist(), criterion)
    best_f, best_gain, best_t = None, 0.0, 0.0
    for f in features:
        vals = X[idx, f]
        order = np.argsort(vals, kind="stable")
        v = vals[order]
        cum = np.cumsum(np.eye(n_classes, dtype=np.int64)[ys[order]], axis=0)
        pos = np.flatnonzero(v[:-1] < v[1:])  # split between sorted rows pos and pos + 1
        nl = pos + 1
        nr = m - nl
        keep = (nl >= min_leaf) & (nr >= min_leaf)
        pos, nl, nr = pos[keep], nl[keep], nr[keep]
        if pos.size == 0:
            continue
        cl = cum[pos]
        cr = total - cl
        gl = _impurity_rows(cl, nl.astype(np.float64), criterion)
        gr = _impurity_rows(cr, nr.astype(np.float64), criterion)
        gain = parent - (nl / m) * gl - (nr / m) * gr
        # Midpoint threshold; if rounding puts it on the upper value, use the upper value so the
        # left side is exactly the samples at or below the lower one.
        lo, hi = v[pos], v[pos + 1]
        thr = (lo + hi) / 2.0
        thr = np.where(lo < thr, thr, hi)
        # First candidate within EPS of this feature's best gain.
        gmax = gain.max()
        k = int(np.flatnonzero(gain >= gmax - EPS)[0])
        g = float(gain[k])
        if best_f is None:
            ok = g > EPS
        else:
            ok = g > best_gain + EPS
        if ok:
            best_f, best_gain, best_t = int(f), g, float(thr[k])
    if best_f is None:
        return None
    return best_f, best_t, best_gain


@dataclass
class Node:
    """One node of a fitted tree. Leaves have left and right set to None."""

    id: int
    depth: int
    counts: list[int]
    impurity: float
    feature: int = -1
    threshold: float = 0.0
    gain: float = 0.0
    left: Node | None = None
    right: Node | None = None
    status: str = "pending"  # pending (in the frontier), split, or leaf
    idx: np.ndarray | None = None  # training rows while the node is in the frontier

    @property
    def n(self) -> int:
        return sum(self.counts)

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    @property
    def label(self) -> int:
        """Majority class; ties go to the lowest class index."""
        best = 0
        for c, k in enumerate(self.counts):
            if k > self.counts[best]:
                best = c
        return best


class Builder:
    """Breadth-first CART. step() splits the next splittable node in the frontier; grow() runs to the end."""

    def __init__(self, X: np.ndarray, y: np.ndarray, n_classes: int, *, criterion: str = "gini",
                 max_depth: int = 4, min_samples_leaf: int = 1, rng: Mulberry32 | None = None,
                 max_features: int | None = None, idx: np.ndarray | None = None):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.int64)
        self.n_classes = int(n_classes)
        self.criterion = criterion
        self.max_depth = int(max_depth)
        self.min_leaf = int(min_samples_leaf)
        self.rng = rng
        self.n_features = self.X.shape[1]
        self.max_features = None if max_features is None else min(int(max_features), self.n_features)
        if self.max_features is not None and self.max_features < self.n_features and rng is None:
            raise ValueError("feature subsampling needs an rng")
        rows = np.arange(len(self.y), dtype=np.int64) if idx is None else np.asarray(idx, dtype=np.int64)
        self._next_id = 0
        self.root = self._make(rows, 0)
        self.queue: deque[Node] = deque([self.root])

    def _make(self, rows: np.ndarray, depth: int) -> Node:
        counts = np.bincount(self.y[rows], minlength=self.n_classes).tolist()
        node = Node(id=self._next_id, depth=depth, counts=counts, impurity=impurity(counts, self.criterion),
                    idx=rows)
        self._next_id += 1
        return node

    def _candidates(self) -> list[int]:
        """All features, or a random subset of max_features drawn by a partial Fisher-Yates shuffle."""
        p = self.n_features
        if self.max_features is None or self.max_features >= p:
            return list(range(p))
        arr = list(range(p))
        for i in range(self.max_features):
            j = i + self.rng.rand_int(p - i)
            arr[i], arr[j] = arr[j], arr[i]
        return arr[: self.max_features]

    def step(self) -> bool:
        """Split the next splittable node in the frontier. Returns False when the frontier is exhausted."""
        while self.queue:
            node = self.queue.popleft()
            rows = node.idx
            node.idx = None
            if node.impurity <= 0 or node.depth >= self.max_depth or node.n < 2 * self.min_leaf:
                node.status = "leaf"
                continue
            best = best_split(self.X, self.y, rows, self._candidates(), self.criterion, self.min_leaf,
                              self.n_classes)
            if best is None:
                node.status = "leaf"
                continue
            f, t, g = best
            mask = self.X[rows, f] < t
            node.feature, node.threshold, node.gain, node.status = f, t, g, "split"
            node.left = self._make(rows[mask], node.depth + 1)
            node.right = self._make(rows[~mask], node.depth + 1)
            self.queue.append(node.left)
            self.queue.append(node.right)
            return True
        return False

    def grow(self) -> Node:
        while self.step():
            pass
        return self.root


class DecisionTree:
    """A CART classifier. fit() grows the whole tree; predict_proba() routes rows to their leaves."""

    def __init__(self, criterion: str = "gini", max_depth: int = 4, min_samples_leaf: int = 1,
                 max_features: int | None = None, rng: Mulberry32 | None = None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng
        self.root: Node | None = None
        self.n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None,
            idx: np.ndarray | None = None) -> DecisionTree:
        y = np.asarray(y, dtype=np.int64)
        self.n_classes = int(n_classes if n_classes is not None else y.max() + 1)
        builder = Builder(X, y, self.n_classes, criterion=self.criterion, max_depth=self.max_depth,
                          min_samples_leaf=self.min_samples_leaf, rng=self.rng, max_features=self.max_features,
                          idx=idx)
        self.root = builder.grow()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros((len(X), self.n_classes))
        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, rows = stack.pop()
            if node.is_leaf:
                c = np.array(node.counts, dtype=np.float64)
                out[rows] = c / c.sum()
            elif rows.size:
                mask = X[rows, node.feature] < node.threshold
                stack.append((node.left, rows[mask]))
                stack.append((node.right, rows[~mask]))
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def n_leaves(self) -> int:
        return n_leaves(self.root)

    def depth(self) -> int:
        return tree_depth(self.root)


def preorder(root: Node) -> Iterator[Node]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        if node.right is not None:
            stack.append(node.right)
            stack.append(node.left)


def n_leaves(root: Node) -> int:
    return sum(1 for node in preorder(root) if node.is_leaf)


def tree_depth(root: Node) -> int:
    return max(node.depth for node in preorder(root))


def copy_tree(root: Node) -> Node:
    """Deep copy of the structure (training rows are dropped; the copy is for display and pruning)."""
    def clone(node: Node) -> Node:
        new = Node(id=node.id, depth=node.depth, counts=list(node.counts), impurity=node.impurity,
                   feature=node.feature, threshold=node.threshold, gain=node.gain, status=node.status)
        if node.left is not None:
            new.left = clone(node.left)
            new.right = clone(node.right)
        return new
    return clone(root)


def _subtree_risk(node: Node, total: int) -> tuple[float, int]:
    """Sum of leaf misclassification risks under node, and the number of leaves."""
    if node.is_leaf:
        return (node.n - max(node.counts)) / total, 1
    lr, ll = _subtree_risk(node.left, total)
    rr, rl = _subtree_risk(node.right, total)
    return lr + rr, ll + rl


def prune(root: Node, alpha: float) -> Node:
    """Cost-complexity (weakest-link) pruning. Returns a pruned copy; the input is not changed.

    Risk R(t) is the misclassification count of node t's majority vote, divided by the root size. For every
    internal node the link strength is g(t) = (R(t) - R(subtree)) / (leaves - 1). The weakest link is collapsed
    while its strength is at most alpha. Larger alpha prunes more, and the leaf count never increases.
    """
    t = copy_tree(root)
    total = t.n
    while True:
        weakest, best_g = None, None
        for node in preorder(t):
            if node.is_leaf:
                continue
            r_node = (node.n - max(node.counts)) / total
            r_sub, leaves = _subtree_risk(node, total)
            g = (r_node - r_sub) / (leaves - 1)
            if best_g is None or g < best_g:
                weakest, best_g = node, g
        if weakest is None or best_g > alpha:
            return t
        weakest.left = None
        weakest.right = None
        weakest.status = "leaf"
        weakest.feature = -1


def link_strengths(root: Node) -> list[float]:
    """g(t) for every internal node, in preorder. Used to pick sensible alpha values for a slider."""
    total = root.n
    out = []
    for node in preorder(root):
        if node.is_leaf:
            continue
        r_node = (node.n - max(node.counts)) / total
        r_sub, leaves = _subtree_risk(node, total)
        out.append((r_node - r_sub) / (leaves - 1))
    return out


def rules_text(root: Node, names: Sequence[str], criterion: str = "gini") -> str:
    """Indented if/else rendering of a tree, with sample counts and impurity at every node."""
    lines: list[str] = []

    def walk(node: Node, pad: str) -> None:
        if node.is_leaf:
            counts = ", ".join(f"{c}:{k}" for c, k in enumerate(node.counts))
            lines.append(f"{pad}-> class {node.label}  (n={node.n}, {criterion}={node.impurity:.3f}, "
                         f"counts [{counts}])")
            return
        lines.append(f"{pad}{names[node.feature]} < {node.threshold:.4f}?  "
                     f"(n={node.n}, {criterion}={node.impurity:.3f})")
        lines.append(f"{pad}  yes:")
        walk(node.left, pad + "    ")
        lines.append(f"{pad}  no:")
        walk(node.right, pad + "    ")

    walk(root, "")
    return "\n".join(lines)
