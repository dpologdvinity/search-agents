"""2048 board as a 64-bit integer: 16 cells x 4 bits, each holding a tile's exponent.

Cell (r, c) lives in bits 16*r + 4*c .. +3, so each row is a 16-bit
integer. A tile 2**k is stored as k (0 = empty), which caps tiles at 32768.

Sliding a row depends only on that row's 16 bits, so every possible row
result is precomputed once. A move is then four table lookups; up and down
transpose the board, slide left or right, and transpose back.

Directions: 0 = Up, 1 = Down, 2 = Left, 3 = Right.
"""

from __future__ import annotations

import random

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRECTIONS = ("Up", "Down", "Left", "Right")
ROW_MASK = 0xFFFF


def _row_cells(row: int) -> list[int]:
    return [(row >> (4 * i)) & 0xF for i in range(4)]


def _cells_row(cells) -> int:
    return sum(v << (4 * i) for i, v in enumerate(cells))


def _slide_left(cells):
    """Slide one row toward index 0, merging equal pairs once. Returns (cells, score gained)."""
    tiles = [v for v in cells if v]
    out, score, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] < 15:
            out.append(tiles[i] + 1)
            score += 1 << (tiles[i] + 1)
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return out + [0] * (4 - len(out)), score


def _build_tables():
    left, right, score = [0] * 65536, [0] * 65536, [0] * 65536
    for row in range(65536):
        cells = _row_cells(row)
        moved, gained = _slide_left(cells)
        left[row] = _cells_row(moved)
        rmoved, _ = _slide_left(cells[::-1])
        right[row] = _cells_row(rmoved[::-1])
        score[row] = gained  # merges score the same in either direction
    return left, right, score


LEFT_TABLE, RIGHT_TABLE, SCORE_TABLE = _build_tables()


def transpose(x: int) -> int:
    """Swap rows and columns of the 4x4 nibble grid (standard bit trick)."""
    a1 = x & 0xF0F00F0FF0F00F0F
    a2 = x & 0x0000F0F00000F0F0
    a3 = x & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


def _apply_rows(x: int, table) -> int:
    return (table[x & ROW_MASK]
            | table[(x >> 16) & ROW_MASK] << 16
            | table[(x >> 32) & ROW_MASK] << 32
            | table[(x >> 48) & ROW_MASK] << 48)


def _row_score(x: int) -> int:
    return (SCORE_TABLE[x & ROW_MASK] + SCORE_TABLE[(x >> 16) & ROW_MASK]
            + SCORE_TABLE[(x >> 32) & ROW_MASK] + SCORE_TABLE[(x >> 48) & ROW_MASK])


def move(board: int, direction: int) -> tuple[int, int]:
    """Board after sliding, and the score gained. The board is unchanged if the move is illegal."""
    if direction == LEFT:
        return _apply_rows(board, LEFT_TABLE), _row_score(board)
    if direction == RIGHT:
        return _apply_rows(board, RIGHT_TABLE), _row_score(board)
    t = transpose(board)
    table = LEFT_TABLE if direction == UP else RIGHT_TABLE
    return transpose(_apply_rows(t, table)), _row_score(t)


def legal_moves(board: int) -> list[tuple[int, int, int]]:
    """[(direction, new board, score gained)] for moves that change the board."""
    out = []
    for d in (UP, DOWN, LEFT, RIGHT):
        new, gained = move(board, d)
        if new != board:
            out.append((d, new, gained))
    return out


def empty_cells(board: int) -> list[int]:
    """Indices 0..15 (row-major) of empty cells."""
    return [i for i in range(16) if not (board >> (4 * i)) & 0xF]


def spawn(board: int, rng: random.Random) -> int:
    """Add a 2 (probability 0.9) or 4 (0.1) to a random empty cell."""
    cells = empty_cells(board)
    if not cells:
        return board
    i = rng.choice(cells)
    return board | ((1 if rng.random() < 0.9 else 2) << (4 * i))


def new_game(rng: random.Random) -> int:
    return spawn(spawn(0, rng), rng)


def max_exponent(board: int) -> int:
    return max((board >> (4 * i)) & 0xF for i in range(16))


def to_grid(board: int) -> list[list[int]]:
    """Tile values, row by row: [[0, 2, 4, 0], ...]."""
    return [[(1 << v) if v else 0 for v in _row_cells((board >> (16 * r)) & ROW_MASK)] for r in range(4)]


def from_grid(grid) -> int:
    board = 0
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value:
                k = value.bit_length() - 1
                if 1 << k != value or not 1 <= k <= 15:
                    raise ValueError(f"invalid tile {value}")
                board |= k << (16 * r + 4 * c)
    return board


def game_over(board: int) -> bool:
    return not legal_moves(board)


def play_game(choose, rng: random.Random, max_moves: int | None = None):
    """Play one game with choose(board) -> direction. Returns (final board, score, moves)."""
    board, score, moves = new_game(rng), 0, 0
    while True:
        options = legal_moves(board)
        if not options or (max_moves is not None and moves >= max_moves):
            return board, score, moves
        d = choose(board)
        new, gained = move(board, d)
        if new == board:
            raise ValueError(f"agent chose illegal move {DIRECTIONS[d]}")
        board, score, moves = spawn(new, rng), score + gained, moves + 1
