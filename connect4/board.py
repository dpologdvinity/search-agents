"""Connect Four position as a pair of bitboards.

Bit layout (one 7-bit column per board column, bit 0 at the bottom):

     5 12 19 26 33 40 47
     4 11 18 25 32 39 46
     3 10 17 24 31 38 45
     2  9 16 23 30 37 44
     1  8 15 22 29 36 43
     0  7 14 21 28 35 42

The 7th bit of each column (6, 13, ...) stays empty. That spare row stops
shifts used for line detection from wrapping into the next column.

A position stores `mask` (every stone) and `current` (stones of the player
to move). Swapping sides is current ^ mask, so the board never needs to
know which color is which.
"""

from __future__ import annotations

ROWS = 6
COLS = 7
H1 = ROWS + 1  # bits per column, including the spare row

BOTTOM = sum(1 << (c * H1) for c in range(COLS))
FULL = BOTTOM * ((1 << ROWS) - 1)  # every playable cell
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)


def _bottom(col: int) -> int:
    return 1 << (col * H1)


def _top(col: int) -> int:
    return 1 << (col * H1 + ROWS - 1)


def _column(col: int) -> int:
    return ((1 << ROWS) - 1) << (col * H1)


def has_four(stones: int) -> bool:
    """True if `stones` contains four in a row in any direction."""
    for shift in (1, H1, H1 - 1, H1 + 1):  # vertical, horizontal, two diagonals
        pairs = stones & (stones >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


class Board:
    """Immutable Connect Four position."""

    __slots__ = ("current", "mask", "moves")

    def __init__(self, current: int = 0, mask: int = 0, moves: int = 0):
        self.current = current
        self.mask = mask
        self.moves = moves

    @classmethod
    def from_moves(cls, columns) -> Board:
        """Board after playing columns in order, e.g. "4453" or [3, 3, 4, 2].

        Strings use 1-based columns, the convention in Connect Four
        solver test sets; lists use 0-based columns.
        """
        if isinstance(columns, str):
            columns = [int(c) - 1 for c in columns]
        board = cls()
        for col in columns:
            if not board.can_play(col):
                raise ValueError(f"column {col} is full or out of range")
            board = board.play(col)
        return board

    def can_play(self, col: int) -> bool:
        return 0 <= col < COLS and not (self.mask & _top(col))

    def legal_moves(self) -> list[int]:
        return [c for c in range(COLS) if not (self.mask & _top(c))]

    def play(self, col: int) -> Board:
        """Drop a stone in col for the player to move; the opponent moves next."""
        mask = self.mask | (self.mask + _bottom(col))
        return Board(self.current ^ self.mask, mask, self.moves + 1)

    def is_winning_move(self, col: int) -> bool:
        """True if playing col wins immediately for the player to move."""
        stone = (self.mask + _bottom(col)) & _column(col)
        return has_four(self.current | stone)

    def last_player_won(self) -> bool:
        """True if the move that produced this position made four in a row."""
        return has_four(self.current ^ self.mask)

    def is_full(self) -> bool:
        return self.moves == ROWS * COLS

    def is_terminal(self) -> bool:
        return self.last_player_won() or self.is_full()

    def key(self) -> int:
        """Unique integer for the position (current + mask identifies it)."""
        return self.current + self.mask

    def mirror(self) -> Board:
        """Left-right reflection."""
        current = mask = 0
        for c in range(COLS):
            shift = (COLS - 1 - 2 * c) * H1
            col = _column(c)
            if shift >= 0:
                current |= (self.current & col) << shift
                mask |= (self.mask & col) << shift
            else:
                current |= (self.current & col) >> -shift
                mask |= (self.mask & col) >> -shift
        return Board(current, mask, self.moves)

    def grid(self) -> list[list[int]]:
        """Rows top to bottom; 1 = player to move, -1 = opponent, 0 = empty."""
        opponent = self.current ^ self.mask
        rows = []
        for r in range(ROWS - 1, -1, -1):
            row = []
            for c in range(COLS):
                bit = 1 << (c * H1 + r)
                row.append(1 if self.current & bit else -1 if opponent & bit else 0)
            rows.append(row)
        return rows

    def to_move_is_first(self) -> bool:
        """True if the player to move is the one who moved first."""
        return self.moves % 2 == 0

    def __eq__(self, other):
        return isinstance(other, Board) and self.mask == other.mask and self.current == other.current

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        first = self.to_move_is_first()
        symbols = {0: ".", 1: "X" if first else "O", -1: "O" if first else "X"}
        return "\n".join(" ".join(symbols[v] for v in row) for row in self.grid())
