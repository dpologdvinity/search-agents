"""Monte Carlo tree search with UCT and random rollouts: the no-learning baseline.

Each simulation walks down the tree choosing children by UCB1, adds one new
node, plays the rest of the game at random, and backs the result up. Values
are stored from the perspective of the player who made the move into a node.
"""

from __future__ import annotations

import math
import random
import time

from .board import Board


class _Node:
    __slots__ = ("board", "parent", "move", "children", "untried", "visits", "value", "terminal")

    def __init__(self, board: Board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: list[_Node] = []
        self.visits = 0
        self.value = 0.0  # summed results for the player who moved into this node
        self.terminal = board.last_player_won() or board.is_full()
        self.untried = [] if self.terminal else board.legal_moves()


def _rollout(board: Board, rng: random.Random) -> float:
    """Play randomly to the end. Returns 1 if the player to move in `board` wins, -1 if they lose."""
    if board.last_player_won():
        return -1.0
    sign = 1.0
    while True:
        moves = board.legal_moves()
        if not moves:
            return 0.0
        col = rng.choice(moves)
        if board.is_winning_move(col):
            return sign
        board = board.play(col)
        sign = -sign


class MCTS:
    def __init__(self, simulations: int = 2000, c: float = 1.4,
                 max_seconds: float | None = None, seed: int | None = None):
        self.simulations = simulations
        self.c = c
        self.max_seconds = max_seconds
        self.rng = random.Random(seed)

    def search(self, board: Board) -> dict:
        start = time.perf_counter()
        root = _Node(board)
        for col in board.legal_moves():  # an immediate win needs no search
            if board.is_winning_move(col):
                return {"move": col, "simulations": 0, "visits": {col: 1}, "seconds": 0.0}

        sims = 0
        while sims < self.simulations:
            if self.max_seconds and sims % 64 == 0 and time.perf_counter() - start > self.max_seconds:
                break
            node = root
            # Selection
            while not node.untried and node.children:
                log_n = math.log(node.visits)
                node = max(node.children, key=lambda ch: ch.value / ch.visits
                           + self.c * math.sqrt(log_n / ch.visits))
            # Expansion
            if node.untried:
                col = node.untried.pop(self.rng.randrange(len(node.untried)))
                child = _Node(node.board.play(col), node, col)
                node.children.append(child)
                node = child
            # Simulation: result for the player who moved into `node`
            result = 1.0 if node.board.last_player_won() else -_rollout(node.board, self.rng)
            # Backpropagation, flipping perspective at each level
            while node is not None:
                node.visits += 1
                node.value += result
                result = -result
                node = node.parent
            sims += 1

        best = max(root.children, key=lambda ch: ch.visits)
        return {"move": best.move, "simulations": sims,
                "visits": {ch.move: ch.visits for ch in root.children},
                "seconds": time.perf_counter() - start}
