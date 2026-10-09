"""The board features the evaluation sums, each computed from row bitmasks.

A placement is scored as score = sum(weight_i * feature_i) over these nine features, measured on the
board after the piece locks and full rows clear. The first five come from Dellacherie's set (landing
height, eroded piece cells, row and column transitions, holes, cumulative wells); the last four are
extra board-shape terms (aggregate height, bumpiness, and completed lines on their own).

    landing_height     height of the piece's centre when it lands (bigger = higher up the board)
    eroded_cells       lines cleared times this piece's cells in those lines
    row_transitions    filled/empty changes along each occupied row, counting the walls as filled
    column_transitions filled/empty changes up each column, counting the floor as filled
    holes              empty cells with a filled cell somewhere above them
    wells              cumulative well depth: a well of depth d adds 1 + 2 + ... + d
    aggregate_height   sum of the column heights
    bumpiness          sum of |height difference| between neighbouring columns
    lines              full rows cleared by this placement

Weights are signed. The GA learns them, and HAND_WEIGHTS is a fixed baseline for comparison.
"""

from __future__ import annotations

from .board import FULL, W

FEATURES = (
    "landing_height",
    "eroded_cells",
    "row_transitions",
    "column_transitions",
    "holes",
    "wells",
    "aggregate_height",
    "bumpiness",
    "lines",
)

# Hand-picked baseline: Dellacherie-style signs and sizes on the six classic features, zero on the
# other three. These are the values from the Dellacherie feature set as commonly reported
# (e.g. Thiery & Scherrer 2009); they were not tuned here.
HAND_WEIGHTS = (-1.0, 1.0, -1.0, -1.0, -4.0, -1.0, 0.0, 0.0, 0.0)

WEIGHT_LIMIT = 10.0  # GA genes are kept in [-WEIGHT_LIMIT, WEIGHT_LIMIT]


def _row_transitions(mask: int) -> int:
    """Filled/empty changes across one row, with a filled wall on each side.

    The row is padded as wall, W cells, wall. Bit k of (e ^ (e >> 1)) says whether cell k differs from
    cell k+1, so counting the set bits of the first W+1 positions counts every change in the row.
    """
    e = (mask << 1) | 1 | (1 << (W + 1))
    return ((e ^ (e >> 1)) & ((1 << (W + 1)) - 1)).bit_count()


# A table over all 2**W row patterns, so the per-row count is a single lookup.
ROW_TRANSITIONS = tuple(_row_transitions(m) for m in range(FULL + 1))


def board_features(rows) -> tuple[int, int, int, int, int, int]:
    """Row transitions, column transitions, holes, wells, aggregate height and bumpiness of a board.

    Only the rows up to the highest filled cell ('top') are read; the empty space above is not part of
    the surface. One pass from the top down finds holes, heights, and aggregate height at once.
    """
    top = 0
    for r in range(len(rows) - 1, -1, -1):
        if rows[r]:
            top = r + 1
            break

    row_t = 0
    for r in range(top):
        row_t += ROW_TRANSITIONS[rows[r]]

    # Column transitions start from a filled floor (FULL below row 0) and end at the empty space above
    # 'top'. Each adjacent pair of rows contributes the bits where they differ.
    col_t = 0
    prev = FULL
    for r in range(top):
        col_t += (prev ^ rows[r]).bit_count()
        prev = rows[r]
    col_t += prev.bit_count()  # the last row against the empty row above it (for an empty board: the floor)

    # Walk from the top down. 'seen' is the OR of the rows above the current one, so an empty cell
    # in 'seen' is a hole, and a column first seen at row r has height r + 1.
    seen = 0
    holes = 0
    agg = 0
    heights = [0] * W
    for r in range(top - 1, -1, -1):
        row = rows[r]
        holes += (seen & ~row & FULL).bit_count()
        new = row & ~seen
        while new:
            low = new & -new
            heights[low.bit_length() - 1] = r + 1
            new ^= low
        seen |= row
        agg += seen.bit_count()  # each column counts once per row it reaches, which sums to its height

    bump = 0
    for c in range(W - 1):
        bump += abs(heights[c] - heights[c + 1])

    # Wells: an empty cell with a filled cell on both sides (walls count as filled). Consecutive well
    # cells in one column form a run; the k-th cell of a run adds k, so a run of depth d adds d(d+1)/2.
    wells = 0
    run = [0] * W
    for r in range(top):
        row = rows[r]
        left = ((row << 1) | 1) & FULL  # bit c set when column c-1 is filled (wall for column 0)
        right = (row >> 1) | (1 << (W - 1))  # bit c set when column c+1 is filled (wall past the edge)
        well = ~row & left & right & FULL
        for c in range(W):
            if well >> c & 1:
                run[c] += 1
                wells += run[c]
            else:
                run[c] = 0
    return row_t, col_t, holes, wells, agg, bump


def placement_features(rot, py: int, lines: int, eroded: int, new_board) -> tuple:
    """The nine features for a placement, in the order of FEATURES."""
    row_t, col_t, holes, wells, agg, bump = board_features(new_board)
    landing = py + (rot.height - 1) / 2
    return (landing, eroded, row_t, col_t, holes, wells, agg, bump, lines)


def dot(weights, feats) -> float:
    return sum(w * f for w, f in zip(weights, feats))
