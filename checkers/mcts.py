"""Monte Carlo tree search (UCT) for checkers, with capped, seeded rollouts.

Each iteration does four steps. Selection walks down the tree, picking the child with
the best upper confidence bound (UCB1): its average result plus a bonus for children
tried rarely. Expansion adds one untried move as a new node. Simulation plays a rollout
from that node. Backpropagation adds the result to every node on the path.

A rollout is cut off after ROLLOUT_PLIES plies. Random play is noisy and long games are
slow, so the cut-off position is scored by the same evaluation the tree searches use,
squashed into (0, 1). Rollouts either play a uniform random move (policy "random") or
prefer captures when one exists (policy "capture", a light heuristic).

The search runs a fixed number of iterations rather than a time limit, and the random
generator is seeded. The same position with the same seed therefore gives the same
answer on any machine, which keeps tests and match results reproducible.
"""

from __future__ import annotations

import math
import random
import time

from .board import legal_moves
from .search import SearchInfo, evaluate

EXPLORATION = math.sqrt(2)  # the standard UCB1 constant
ROLLOUT_PLIES = 40  # rollouts stop here and score the position instead
EVAL_SCALE = 300  # evaluation points at which the cut-off score is about 0.12 or 0.88
POLICIES = ("random", "capture")


class _Node:
    """One position in the tree. wins is credited to the side that moved into this node."""

    __slots__ = ("board", "side", "move", "parent", "children", "untried", "visits", "wins", "ply")

    def __init__(self, board, side, move, parent, ply, forced):
        self.board = board
        self.side = side  # the side to move here
        self.move = move  # the move that led here from the parent (None at the root)
        self.parent = parent
        self.children = []
        # Every legal move starts untried; expansion removes them one at a time.
        self.untried = list(legal_moves(board, side, forced))
        self.visits = 0
        self.wins = 0.0
        self.ply = ply


class MCTS:
    """UCT search with a fixed iteration budget and a seeded generator."""

    def __init__(self, iterations=1000, seed=0, policy="capture", rollout_plies=ROLLOUT_PLIES, forced=True):
        if policy not in POLICIES:
            raise ValueError(f"rollout policy must be one of {POLICIES}, not {policy!r}")
        if iterations < 1:
            raise ValueError("MCTS needs at least one iteration")
        self.iterations = iterations
        self.rng = random.Random(seed)
        self.policy = policy
        self.rollout_plies = rollout_plies
        self.forced = forced

    def _rollout(self, board, side) -> float:
        """Play from this position and return the chance that side to move wins (0 to 1).

        The value is always from the point of view of the side to move at the start of the
        rollout. A side with no legal move has lost: 0 if that is the starting side, 1 if
        it is the opponent. Otherwise play runs for at most rollout_plies plies, and then
        the evaluation at that position, taken from the starting side's view, is squashed
        into (0, 1). The squash is a tanh, so a man's worth (100 points) moves the value
        from 0.5 to about 0.66.
        """
        start = side
        for _ in range(self.rollout_plies):
            moves = legal_moves(board, side, self.forced)
            if not moves:
                return 0.0 if side == start else 1.0
            pool = moves
            if self.policy == "capture":
                pool = [m for m in moves if m.captured] or moves
            move = pool[self.rng.randrange(len(pool))]
            board, side = move.result, -side
        return 0.5 + 0.5 * math.tanh(evaluate(board, start) / EVAL_SCALE)

    def _select(self, node):
        """The child with the best UCB1 score. Children always have at least one visit."""
        log_n = math.log(node.visits)
        return max(node.children,
                   key=lambda c: c.wins / c.visits + EXPLORATION * math.sqrt(log_n / c.visits))

    def search(self, board, side) -> SearchInfo:
        """Run the iterations from this position and return the most visited move, with win rates."""
        start = time.perf_counter()
        root = _Node(board, side, None, None, 0, self.forced)
        if not root.untried:
            raise ValueError("no legal moves")
        nodes, deepest, rollouts = 1, 0, 0
        for _ in range(self.iterations):
            node = root
            # Selection: go down while the node is fully expanded and has children.
            while not node.untried and node.children:
                node = self._select(node)
            # Expansion: add one untried move as a child. A node with no moves at all is
            # terminal and is simply simulated, which scores it as a loss for its side.
            if node.untried:
                i = self.rng.randrange(len(node.untried))
                # Swap the chosen move to the end before popping, so removal is O(1).
                node.untried[i], node.untried[-1] = node.untried[-1], node.untried[i]
                m = node.untried.pop()
                child = _Node(m.result, -node.side, m, node, node.ply + 1, self.forced)
                node.children.append(child)
                node = child
                nodes += 1
                deepest = max(deepest, node.ply)
            # Simulation, then backpropagation. value is for the side to move at the node;
            # each node's wins are credited to the side that moved into it, so flip the
            # value at every step up the tree.
            value = self._rollout(node.board, node.side)
            rollouts += 1
            while node is not None:
                node.visits += 1
                node.wins += 1.0 - value
                value = 1.0 - value
                node = node.parent

        # The move played is the most visited child (the most robust choice), with the
        # higher win rate breaking ties. A child's win rate is for the side at the root.
        # Moves are matched by path: on one board a path names exactly one legal move.
        tried = {c.move.path: c for c in root.children}
        scored = []
        for m in legal_moves(board, side, self.forced):
            child = tried.get(m.path)
            scored.append((m, child.wins / child.visits if child is not None else None))
        best = max(root.children, key=lambda c: (c.visits, c.wins / c.visits))
        return SearchInfo(best.move, best.wins / best.visits, deepest, nodes, 0,
                          time.perf_counter() - start, scored, rollouts)
