"""PUCT tree search guided by a policy-value evaluator (the AlphaZero search).

Each simulation descends by maximizing Q(s,a) + U(s,a), where
    U(s,a) = c_puct * P(s,a) * sqrt(N(s) + 1) / (1 + N(s,a)),
(the + 1 gives the first simulation from a node a nonzero exploration bonus),
P is the network's prior, and Q the mean backed-up value. At a new leaf the
network supplies priors for its children and a value estimate, which
replaces the random rollout of plain MCTS. Terminal positions use the true
result.

Search is split into select_leaf() and expand_and_backup() so many trees
(for example, many self-play games) can share one batched network call per
round. Values are always from the perspective of the player to move at the
node that holds them.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .board import COLS, Board

# evaluator(boards) -> (priors (B, 7) summing to 1 over legal moves, values (B,))
Evaluator = Callable[[list[Board]], tuple[np.ndarray, np.ndarray]]


class Node:
    __slots__ = ("board", "prior", "visits", "value_sum", "children", "terminal_value")

    def __init__(self, board: Board):
        self.board = board
        self.prior: np.ndarray | None = None  # set when expanded
        self.visits = np.zeros(COLS, dtype=np.float64)
        self.value_sum = np.zeros(COLS, dtype=np.float64)  # for the player to move here
        self.children: dict[int, Node] = {}
        if board.last_player_won():
            self.terminal_value: float | None = -1.0  # the player to move has lost
        elif board.is_full():
            self.terminal_value = 0.0
        else:
            self.terminal_value = None

    @property
    def expanded(self) -> bool:
        return self.prior is not None

    def total_visits(self) -> float:
        return float(self.visits.sum())


class Tree:
    """One search tree; reusable across moves via advance()."""

    def __init__(self, board: Board, c_puct: float = 2.0,
                 dirichlet_alpha: float = 1.0, noise_fraction: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.root = Node(board)
        self.c_puct = c_puct
        self.alpha = dirichlet_alpha
        self.noise = noise_fraction
        self.rng = rng or np.random.default_rng()

    def select_leaf(self):
        """Descend to an unexpanded or terminal node; returns (leaf, path of (node, action))."""
        node, path = self.root, []
        while node.expanded and node.terminal_value is None:
            n = node.total_visits()
            q = np.divide(node.value_sum, node.visits, out=np.zeros(COLS), where=node.visits > 0)
            u = self.c_puct * node.prior * math.sqrt(n + 1) / (1 + node.visits)
            score = np.where(node.prior > 0, q + u, -np.inf)
            action = int(score.argmax())
            path.append((node, action))
            child = node.children.get(action)
            if child is None:
                child = node.children[action] = Node(node.board.play(action))
            node = child
        return node, path

    def expand_and_backup(self, leaf: Node, path, prior: np.ndarray | None, value: float):
        """Expand leaf with network priors (ignored for terminal leaves) and back up value.

        value is from the perspective of the player to move at leaf.
        """
        if leaf.terminal_value is not None:
            value = leaf.terminal_value
        elif not leaf.expanded:
            if leaf is self.root and self.noise > 0:
                prior = self._with_noise(prior)
            leaf.prior = prior
        for node, action in reversed(path):
            value = -value  # the parent's player is the leaf's opponent
            node.visits[action] += 1
            node.value_sum[action] += value

    def _with_noise(self, prior: np.ndarray) -> np.ndarray:
        legal = prior > 0
        noise = np.zeros(COLS)
        noise[legal] = self.rng.dirichlet([self.alpha] * int(legal.sum()))
        return (1 - self.noise) * prior + self.noise * noise

    def policy(self, temperature: float = 1.0) -> np.ndarray:
        """Move distribution from root visit counts."""
        visits = self.root.visits
        if temperature == 0:
            pi = np.zeros(COLS)
            pi[int(visits.argmax())] = 1.0
            return pi
        v = visits ** (1.0 / temperature)
        return v / v.sum()

    def root_value(self) -> float:
        """Mean value of the root for the player to move, from search."""
        n = self.root.total_visits()
        return float(self.root.value_sum.sum() / n) if n else 0.0

    def advance(self, action: int):
        """Make action the new root, keeping its subtree."""
        child = self.root.children.get(action)
        self.root = child if child is not None else Node(self.root.board.play(action))
        if self.root.expanded and self.noise > 0 and self.root.terminal_value is None:
            self.root.prior = self._with_noise(self.root.prior)


def run_simulations(trees: list[Tree], evaluate: Evaluator, simulations: int):
    """Run `simulations` rounds on every tree, batching all leaf evaluations per round."""
    # Roots need priors before the first selection.
    fresh = [t for t in trees if not t.root.expanded and t.root.terminal_value is None]
    if fresh:
        priors, _ = evaluate([t.root.board for t in fresh])
        for tree, prior in zip(fresh, priors):
            tree.expand_and_backup(tree.root, [], prior, 0.0)
    for _ in range(simulations):
        pending = []
        for tree in trees:
            leaf, path = tree.select_leaf()
            if leaf.terminal_value is not None:
                tree.expand_and_backup(leaf, path, None, 0.0)
            else:
                pending.append((tree, leaf, path))
        if pending:
            priors, values = evaluate([leaf.board for _, leaf, _ in pending])
            for (tree, leaf, path), p, v in zip(pending, priors, values):
                tree.expand_and_backup(leaf, path, p, float(v))


def search(board: Board, evaluate: Evaluator, simulations: int = 400, c_puct: float = 2.0) -> Tree:
    """Single-position search without noise, for play. Returns the tree."""
    tree = Tree(board, c_puct=c_puct)
    run_simulations([tree], evaluate, simulations)
    return tree
