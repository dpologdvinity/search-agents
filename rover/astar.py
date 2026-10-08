"""Baseline: plain A* from the rover's cell, run again from scratch every time the map changes.

Between map changes the rover simply follows the path it already has. A map change that does not
touch a wall-state (a cell it has already sensed as free) cannot change any shortest path, so it does
not trigger a search. A change that does trigger one is handled by a complete search that forgets
everything, which is the cost D* Lite is designed to avoid.

Search order: priority (f = g + h, then h, then cell number). The tie-breaks make the search
deterministic, so the JS port expands the same cells. With the Manhattan heuristic (consistent for
unit steps), a cell's first pop is already optimal, so popped-and-closed cells are never reopened.
"""

from __future__ import annotations

import heapq

from .world import neighbours

INF = float("inf")


class AStar:
    """Replans from the rover's cell with A* whenever `on_change` is called; follows the stored path otherwise.

    The interface matches DStarLite, so the explorer can drive either planner. `expansions` counts
    closed cells over the planner's life; `last_expanded` lists them for the most recent search.
    """

    def __init__(self, n: int, start: int, goal: int):
        self.n = n
        self.goal = goal
        self.wall = bytearray(n * n)  # the belief: 1 = known wall (unknown counts as free)
        self.path: list[int] = []     # cells from the rover's cell to the goal; empty if none
        self.expansions = 0
        self.last_expanded: list[int] = []
        self._plan(start)

    def _h(self, a: int, b: int) -> int:
        n = self.n
        return abs(a // n - b // n) + abs(a % n - b % n)

    def _plan(self, pos: int) -> None:
        """Search from pos to the goal over the current belief and store the path."""
        n, goal, wall = self.n, self.goal, self.wall
        N = n * n
        g = [INF] * N
        parent = [-1] * N
        closed = bytearray(N)
        g[pos] = 0
        h0 = self._h(pos, goal)
        heap = [(h0, h0, pos)]         # entries are (f, h, cell)
        order: list[int] = []
        while heap:
            _, _, u = heapq.heappop(heap)
            if closed[u]:              # a later, worse entry for a cell that is already expanded
                continue
            closed[u] = 1
            self.expansions += 1
            order.append(u)
            if u == goal:
                break
            for v in neighbours(n, u):
                if wall[v]:
                    continue
                ng = g[u] + 1
                if ng < g[v]:
                    g[v] = ng
                    parent[v] = u
                    hv = self._h(v, goal)
                    heapq.heappush(heap, (ng + hv, hv, v))
        self.last_expanded = order
        if not closed[goal]:
            self.path = []
            return
        path, cur = [goal], goal
        while cur != pos:              # walk parent links back to the rover's cell
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        self.path = path

    # ── public API (same shape as DStarLite) ─────────────────────────────

    def on_change(self, changes, pos: int) -> None:
        """Record the sensed changes and search again from pos."""
        for cell, wall in changes:
            self.wall[cell] = wall
        self._plan(pos)

    def cost_to_goal(self, pos: int) -> float:
        """Length of the stored path from pos to the goal, or INF if there is none."""
        return len(self.path) - 1 if self.path and self.path[0] == pos else INF

    def next_cell(self, pos: int):
        """The next cell on the stored path. The path is trimmed to start at pos first, since the rover has moved."""
        while self.path and self.path[0] != pos:
            self.path.pop(0)
        if len(self.path) < 2:
            return None
        return self.path[1]
