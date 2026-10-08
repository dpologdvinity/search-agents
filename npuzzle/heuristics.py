"""Heuristics estimating the number of moves from a state to the goal.

Every heuristic here is admissible (never overestimates), so A* and IDA*
using them return optimal paths. Each takes (config, n) and returns an int.
"""

from __future__ import annotations

from functools import cache


@cache
def _manhattan_table(n: int) -> tuple[tuple[int, ...], ...]:
    """table[tile][index] = Manhattan distance of tile at index from its goal.

    In the goal, tile t sits at index t. Precomputing the table replaces the
    per-tile config.index() scan, which made each evaluation O(n^4).
    """
    size = n * n
    return tuple(
        tuple(abs(i // n - t // n) + abs(i % n - t % n) for i in range(size))
        for t in range(size)
    )


def manhattan(config: tuple[int, ...], n: int) -> int:
    """Sum of each tile's row and column distance from its goal position.

    Admissible: every move shifts one tile by one square, so it reduces the
    sum by at most 1. The blank is not counted.
    """
    table = _manhattan_table(n)
    return sum(table[tile][i] for i, tile in enumerate(config) if tile)


def _removals_to_order(goal_positions: list[int]) -> int:
    """Fewest tiles to remove so the rest are in increasing order.

    Equals len - (longest increasing subsequence). Lines hold at most n tiles,
    so the O(k^2) LIS is fine.
    """
    k = len(goal_positions)
    if k < 2:
        return 0
    lis = [1] * k
    for i in range(k):
        for j in range(i):
            if goal_positions[j] < goal_positions[i] and lis[j] + 1 > lis[i]:
                lis[i] = lis[j] + 1
    return k - max(lis)


def linear_conflict(config: tuple[int, ...], n: int) -> int:
    """Manhattan distance plus 2 per tile that must leave its line to let others pass.

    Two tiles conflict when both sit in their goal row (or column) but in
    reversed order: one must step out of the line and back, two moves that
    Manhattan distance does not count. Counting removals per line as
    len - LIS (not raw conflicting pairs) keeps the heuristic admissible.
    """
    h = manhattan(config, n)
    extra = 0
    for line in range(n):
        row_goals = []  # goal columns of tiles in this row whose goal row is this row
        col_goals = []  # goal rows of tiles in this column whose goal column is this column
        for k in range(n):
            tile = config[line * n + k]
            if tile and tile // n == line:
                row_goals.append(tile % n)
            tile = config[k * n + line]
            if tile and tile % n == line:
                col_goals.append(tile // n)
        extra += _removals_to_order(row_goals) + _removals_to_order(col_goals)
    return h + 2 * extra


def pattern_database(config: tuple[int, ...], n: int) -> int:
    """Additive 5-5-5 pattern database lookup (4x4 only). See npuzzle.pdb."""
    from .pdb import lookup

    return lookup(config, n)


HEURISTICS = {
    "manhattan": manhattan,
    "linear_conflict": linear_conflict,
    "pdb": pattern_database,
}
