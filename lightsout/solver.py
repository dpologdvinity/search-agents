"""Exact Lights Out solver: Gaussian elimination over GF(2).

Model. Let x be the press vector (x[j] = 1 if cell j is pressed) and b the board
(b[i] = 1 if cell i is lit). Cell i ends dark exactly when the presses in its
neighbourhood XOR to its lit state, so the puzzle is the linear system

    A x = b          (all arithmetic mod 2)

where row i of A is the neighbourhood of cell i. A is symmetric: the same matrix
describes what a press does and which presses undo what.

Solving. Gauss-Jordan elimination reduces [A | b] to reduced row echelon form.
  * A row that reduces to zeros on the left but keeps a 1 on the right says 0 = 1,
    so b is not in the column space of A. No press pattern clears such a board,
    and no press sequence can change that: each press adds a column of A, and the
    column space is closed under that. Solvability is an invariant of the game.
  * Otherwise the solutions are x0 plus any combination of the null-space basis,
    where x0 sets every free variable to 0. The null space has dimension
    nullity = n*n - rank, so there are 2**nullity solutions. For 5x5 the rank is
    23 and the nullity is 2, giving 4 solutions.
The minimum-press answer is the lightest of those solutions.

Representation. Each row is one Python int: bit j is column j, and bit n*n is the
right-hand side. Adding rows over GF(2) is XOR, so a row operation is one `^`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

from .board import neighbourhood, to_mask


class Step(NamedTuple):
    """Work done for one column: the row swap that brings a pivot up, the pivot row, and rows cleared.

    `pivot` is None when the column has no pivot, which makes its variable free.
    The pivot row index is the row's position after the swap.
    """

    col: int
    pivot: int | None
    swap: tuple[int, int] | None
    cleared: tuple[int, ...]


class Trace(NamedTuple):
    """Everything needed to replay the elimination: the starting augmented rows and the steps.

    Each start row is a string of '0'/'1' with n*n + 1 characters; the last one is the right-hand side.
    Replaying the steps in order (swap, then XOR the pivot into each cleared row) reproduces the reduction.
    """

    start: tuple[str, ...]
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Solution:
    """Result of solving one board.

    presses: cells of the minimum-press solution, sorted; () if already dark, None if unsolvable.
    solution_sizes: press count of every solution, sorted; empty if unsolvable.
    """

    n: int
    solvable: bool
    rank: int
    nullity: int
    presses: tuple[int, ...] | None
    solution_sizes: tuple[int, ...]
    trace: Trace | None

    @property
    def count(self) -> int:
        """How many distinct press patterns clear the board (2**nullity when solvable, else 0)."""
        return len(self.solution_sizes)


class _Reduced(NamedTuple):
    rows: list[int]  # reduced augmented rows, right-hand side in bit n*n
    pivots: list[int]  # pivot column of each pivot row; pivot row r is rows[r] for r < rank
    rank: int
    steps: list[Step]
    start: tuple[str, ...]


def _row_text(row: int, width: int) -> str:
    return "".join("1" if row >> j & 1 else "0" for j in range(width))


@cache
def _system(n: int) -> tuple[int, ...]:
    """Rows of A. Row i is the neighbourhood of cell i, which is also column i by symmetry."""
    return tuple(neighbourhood(n, i) for i in range(n * n))


def _reduce(n: int, board_mask: int) -> _Reduced:
    """Gauss-Jordan elimination of [A | b] over GF(2), recording every step."""
    N = n * n
    rhs = 1 << N
    rows = [a | (rhs if board_mask >> i & 1 else 0) for i, a in enumerate(_system(n))]
    start = tuple(_row_text(r, N + 1) for r in rows)
    pivots: list[int] = []
    steps: list[Step] = []
    rank = 0
    for col in range(N):
        bit = 1 << col
        # Rows above `rank` already hold pivots. Look for a row at or below `rank` with a 1 here.
        found = next((r for r in range(rank, N) if rows[r] & bit), None)
        if found is None:
            # No pivot: this variable is free. Its value is chosen by the null-space directions.
            steps.append(Step(col, None, None, ()))
            continue
        swap = None
        if found != rank:
            rows[rank], rows[found] = rows[found], rows[rank]
            swap = (rank, found)
        pivot_row = rows[rank]
        # Gauss-Jordan clears this column in every other row, above as well as below.
        # That leaves each pivot column with a single 1, so the solution can be read off directly.
        cleared = tuple(r for r in range(N) if r != rank and rows[r] & bit)
        for r in cleared:
            rows[r] ^= pivot_row
        pivots.append(col)
        steps.append(Step(col, rank, swap, cleared))
        rank += 1
    return _Reduced(rows, pivots, rank, steps, start)


def _null_basis(n: int, red: _Reduced) -> list[int]:
    """Basis of the null space of A, one vector per free column.

    Setting free variable f to 1 (others free at 0) forces each pivot variable to the
    bit of f in its pivot row, because a reduced pivot row reads
    x[pivot] + sum(x[free] for free columns set in that row) = right-hand side.
    """
    pivot_cols = set(red.pivots)
    basis = []
    for f in range(n * n):
        if f in pivot_cols:
            continue
        v = 1 << f
        for r, col in enumerate(red.pivots):
            if red.rows[r] >> f & 1:
                v |= 1 << col
        basis.append(v)
    return basis


def null_space(n: int) -> list[int]:
    """Press patterns that change nothing (A v = 0), as cell bit masks. Its size is the nullity."""
    return _null_basis(n, _reduce(n, 0))


def solve(n: int, board, trace: bool = False) -> Solution:
    """Solve an n x n board: its minimum-press solution, or proof that none exists.

    `board` is n*n values of 0 or 1 in row-major order. With trace=True the result
    carries the elimination steps so a visualizer can replay them.
    """
    N = n * n
    board = tuple(board)
    if len(board) != N or any(v not in (0, 1) for v in board):
        raise ValueError(f"board must be {N} cells of 0 or 1")
    red = _reduce(n, to_mask(board))
    nullity = N - red.rank
    tr = Trace(red.start, tuple(red.steps)) if trace else None

    # Rows past the rank have no pivot, so their left side is all zeros. A 1 on the right
    # there means 0 = 1: the board lies outside the column space and cannot be cleared.
    rhs = 1 << N
    if any(red.rows[r] & rhs for r in range(red.rank, N)):
        return Solution(n, False, red.rank, nullity, None, (), tr)

    # Particular solution x0: free variables 0, so each pivot variable equals its row's right-hand side.
    x0 = 0
    for r, col in enumerate(red.pivots):
        if red.rows[r] & rhs:
            x0 |= 1 << col

    # Every solution is x0 XOR a combination of the null-space basis. Appending one basis vector
    # at a time doubles the list, so the end result holds all 2**nullity solutions.
    solutions = [x0]
    for v in _null_basis(n, red):
        solutions += [s ^ v for s in solutions]

    # Tie-break on the mask so the answer is deterministic when two solutions are equally short.
    best = min(solutions, key=lambda m: (m.bit_count(), m))
    presses = tuple(c for c in range(N) if best >> c & 1)
    sizes = tuple(sorted(m.bit_count() for m in solutions))
    return Solution(n, True, red.rank, nullity, presses, sizes, tr)
