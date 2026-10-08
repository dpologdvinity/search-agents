"""The N-Queens placement and its conflict bookkeeping.

A placement puts one queen in every row: cols[r] is the column of the row-r queen, or -1 while the
row is empty (only backtracking leaves rows empty). Two queens in the same row are impossible by
construction, so a conflict is a pair of queens that share a column or a diagonal.

Three count arrays make every question O(1):
  col[c]             queens in column c
  d1[r + c]          queens on the down-right diagonal through (r, c), indexed 0 .. 2n-2
  d2[r - c + n - 1]  queens on the down-left diagonal through (r, c), indexed 0 .. 2n-2

A line that holds k queens contains C(k, 2) conflicting pairs. So adding a queen to a line that
already holds k queens adds exactly k conflicts, and removing one subtracts the k left behind.
That is the whole update rule, and it is what lets the agents rescore a move without rebuilding.
"""

from __future__ import annotations


class Board:
    """A row-indexed placement with an incrementally kept conflict count.

    `conflicts` is the number of attacking pairs among the placed queens. place and remove keep it
    exact; the tests check it against a brute-force count.
    """

    def __init__(self, n: int, cols: list[int] | None = None):
        if n < 1:
            raise ValueError("the board needs at least one square")
        self.n = n
        self.cols = [-1] * n
        self.col = [0] * n
        self.d1 = [0] * (2 * n - 1)
        self.d2 = [0] * (2 * n - 1)
        self.conflicts = 0
        if cols is not None:
            for r, c in enumerate(cols):
                if c >= 0:
                    self.place(r, c)

    def attacks_if(self, r: int, c: int) -> int:
        """Queens that would attack square (r, c), not counting the queen already in row r.

        Every agent scores moves with this. The three lookups count the queens on the column and
        the two diagonals through (r, c). If row r holds a queen on column c itself, that queen is
        on all three lines, so it is subtracted. The row-r queen never affects another column's
        count, because its lines meet (r, c) only when c is its own column.
        """
        n = self.n
        a = self.col[c] + self.d1[r + c] + self.d2[r - c + n - 1]
        if self.cols[r] == c:
            a -= 3
        return a

    def queen_attacks(self, r: int) -> int:
        """Queens attacking the queen in row r (0 when the row is empty)."""
        c = self.cols[r]
        return self.attacks_if(r, c) if c >= 0 else 0

    def place(self, r: int, c: int) -> None:
        """Put a queen in empty row r, column c. Adds the conflicts it creates."""
        if self.cols[r] >= 0:
            raise ValueError(f"row {r} already has a queen")
        n = self.n
        i1, i2 = r + c, r - c + n - 1
        self.conflicts += self.col[c] + self.d1[i1] + self.d2[i2]
        self.col[c] += 1
        self.d1[i1] += 1
        self.d2[i2] += 1
        self.cols[r] = c

    def remove(self, r: int) -> None:
        """Take the queen out of row r. Its conflicts are subtracted after the counts drop."""
        c = self.cols[r]
        if c < 0:
            raise ValueError(f"row {r} is empty")
        n = self.n
        i1, i2 = r + c, r - c + n - 1
        self.col[c] -= 1
        self.d1[i1] -= 1
        self.d2[i2] -= 1
        self.conflicts -= self.col[c] + self.d1[i1] + self.d2[i2]
        self.cols[r] = -1

    def move(self, r: int, c: int) -> None:
        """Move the row-r queen to column c (placing it if the row is empty)."""
        if self.cols[r] >= 0:
            self.remove(r)
        self.place(r, c)

    def is_solved(self) -> bool:
        return all(c >= 0 for c in self.cols) and self.conflicts == 0


def count_conflicts(cols: list[int]) -> int:
    """Attacking pairs among the placed queens (cols[r] == -1 is an empty row), in O(n).

    Each line holding k queens has C(k, 2) pairs, and no pair can share two lines (same column and
    same diagonal would force the same row), so summing over the lines counts every pair once.
    """
    n = len(cols)
    col = [0] * n
    d1 = [0] * (2 * n - 1)
    d2 = [0] * (2 * n - 1)
    for r, c in enumerate(cols):
        if c >= 0:
            col[c] += 1
            d1[r + c] += 1
            d2[r - c + n - 1] += 1
    return sum(k * (k - 1) // 2 for k in col + d1 + d2)


def brute_conflicts(cols: list[int]) -> int:
    """Attacking pairs by checking every pair of queens. O(k^2): an independent check for the tests."""
    placed = [(r, c) for r, c in enumerate(cols) if c >= 0]
    total = 0
    for i in range(len(placed)):
        r1, c1 = placed[i]
        for j in range(i + 1, len(placed)):
            r2, c2 = placed[j]
            if c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                total += 1
    return total


def is_solution(cols: list[int]) -> bool:
    """True when every row holds a queen and no two queens attack each other."""
    return len(cols) > 0 and all(c >= 0 for c in cols) and count_conflicts(cols) == 0


def render(cols: list[int]) -> str:
    """ASCII board: Q for a queen, '.' for an empty square. Row 1 is at the top, column a at the left."""
    n = len(cols)
    lines = []
    if n <= 26:
        lines.append("   " + " ".join(chr(ord("a") + c) for c in range(n)))
    for r, c in enumerate(cols):
        cells = " ".join("Q" if j == c else "." for j in range(n))
        lines.append(f"{r + 1:>2} {cells}")
    return "\n".join(lines)
