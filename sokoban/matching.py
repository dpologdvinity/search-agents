"""Hungarian algorithm: minimum-cost perfect matching in a square cost matrix, in O(n^3).

Sokoban's matching heuristic needs this. Each box must end on a distinct goal, so the pushes still
to come are at least the cheapest way to pair boxes with goals, where pairing box b with goal g
costs the push distance from b to g. That bound is never more than the true pushes, because a real
solution is one such pairing with each box's pushes at least its push distance.

The implementation keeps dual potentials u (rows) and v (columns) and grows an alternating tree
from one unmatched row at a time, as in the classic Kuhn-Munkres formulation. Costs are integers,
so the result is exact.
"""

from __future__ import annotations

from collections.abc import Sequence

_BIG = 1 << 60


def min_cost_matching(cost: Sequence[Sequence[int]]) -> tuple[int, list[int]]:
    """Return (total cost, match) for a square matrix, where match[row] is the column paired with row."""
    n = len(cost)
    if n == 0:
        return 0, []
    # Index 0 is a sentinel: p[j] is the row matched to column j (0 = none), with 1-based indices.
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [_BIG] * (n + 1)  # cheapest reduced cost from the tree to each column
        used = [False] * (n + 1)
        # Grow the tree of columns reached from row i until a free column is found.
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = _BIG
            j1 = 0
            row = cost[i0 - 1]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]  # reduced cost keeps every entry non-negative
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            # Shift the potentials by delta so that the next tight edge appears.
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Flip the augmenting path back to the root, which adds one matched pair.
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    match = [0] * n
    for j in range(1, n + 1):
        match[p[j] - 1] = j - 1
    total = sum(cost[r][match[r]] for r in range(n))
    return total, match
