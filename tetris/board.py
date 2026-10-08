"""The playfield as bitmasks: one int per row, so a whole row is compared or cleared in one operation.

A board is a tuple of H row masks, row 0 at the bottom, bit x of a row = column x filled. Pieces
spawn in the top two rows. A cell at or above row H is outside the box: it is free for movement and
rotation, but a locked piece that reaches above the top is a top-out (game over).

Rules are the usual ones: a row that becomes full is removed and everything above drops by one row.
Pieces here never stop in mid-air, so there is no gravity between placements.
"""

from __future__ import annotations

from .pieces import Rotation

W = 10  # columns
H = 20  # rows of the visible playfield
FULL = (1 << W) - 1  # a complete row


def empty_board(height: int = H) -> tuple[int, ...]:
    return (0,) * height


def fits(board, rot: Rotation, px: int, py: int) -> bool:
    """True when the piece with its bottom-left box corner at (px, py) overlaps nothing and stays in the walls."""
    if px < 0 or px + rot.width > W or py < 0:
        return False
    for i, m in enumerate(rot.masks):
        r = py + i
        if r >= len(board):
            continue  # above the ceiling: nothing can be there yet
        if board[r] & (m << px):
            return False
    return True


def drop_row(board, rot: Rotation, px: int) -> int | None:
    """Bottom row where a piece dropped straight down column px comes to rest.

    The piece starts with its top row in the top row of the board (the spawn height). If it already
    overlaps there, the column is blocked and this returns None. Otherwise step down while the next row
    down is free.
    """
    y = len(board) - rot.height
    if not fits(board, rot, px, y):
        return None
    while y > 0 and fits(board, rot, px, y - 1):
        y -= 1
    return y


def lock(board, rot: Rotation, px: int, py: int) -> tuple[tuple[int, ...], int, int, bool]:
    """Write a piece into the board and clear full rows.

    Returns (new_board, lines, eroded_cells, topped_out). eroded_cells is lines times the number of this
    piece's cells in the cleared rows: the 'eroded piece cells' feature of Dellacherie's set, which rewards
    clearing lines with many cells of the piece just placed. topped_out is True when a cell lands at or
    above H, so the piece locked above the visible field.
    """
    rows = list(board)
    h = len(rows)
    topped = False
    for i, m in enumerate(rot.masks):
        r = py + i
        if r >= h:
            topped = topped or m != 0
            continue
        rows[r] |= m << px
    lines = 0
    kept = []
    for r in rows:
        if r == FULL:
            lines += 1
        else:
            kept.append(r)
    cleared_cells = 0
    if lines:
        # Count this piece's cells that sat in the rows that just vanished.
        for i, m in enumerate(rot.masks):
            if py + i < h and rows[py + i] == FULL:
                cleared_cells += m.bit_count()
    kept.extend([0] * lines)  # empty rows come in at the top
    return tuple(kept), lines, lines * cleared_cells, topped


def render(board, overlays=()) -> str:
    h = len(board)
    """Plain-text picture of a board, top row first, with '#' filled and '.' empty.

    overlays is a sequence of (rot, px, py, char): each piece is drawn over the board with that character.
    The terminal game uses it for the falling piece ('@') and the agent's hint ('o').
    """
    marks: dict[tuple[int, int], str] = {}
    for rot, px, py, ch in overlays:
        for i, m in enumerate(rot.masks):
            for x in range(W):
                if m >> x & 1 and 0 <= py + i < h:
                    marks[(py + i, px + x)] = ch
    lines = []
    for r in range(h - 1, -1, -1):
        row = "".join(marks.get((r, c), "#" if board[r] >> c & 1 else ".") for c in range(W))
        lines.append(f"|{row}|")
    lines.append("+" + "-" * W + "+")
    return "\n".join(lines)
