"""Board helpers shared by the N-Puzzle tests."""

import random

from npuzzle.board import goal, successors


def walk(n, steps, rng: random.Random):
    """Board reached by a random walk from the goal that never undoes a move."""
    board, prev = goal(n), None
    for _ in range(steps):
        options = [b for _, b in successors(board, n) if b != prev]
        prev, board = board, rng.choice(options)
    return board


def random_board(n, rng: random.Random):
    """Solvable board from a walk of 5-40 moves."""
    return walk(n, rng.randint(5, 40), rng)


def solves(start, path):
    """True if playing path from start reaches the goal."""
    from npuzzle.board import replay, size_of

    n = size_of(start)
    return replay(start, path, n) == goal(n)
