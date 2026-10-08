"""D* Lite (Koenig and Likhachev, 2002): shortest paths that are repaired, not recomputed, when the map changes.

The idea in one paragraph. The planner searches backwards from the goal. Every cell has
  g(s)   the cost it currently believes it has to reach the goal from s, and
  rhs(s) a one-step lookahead: the best g(s') over its neighbours plus the cost of the step to them.
A cell is "consistent" when g(s) == rhs(s); when every cell is consistent, following the cheapest
neighbour from the rover's cell walks a shortest path. A map change makes some rhs values wrong. Only
those cells are re-examined and pushed onto a priority queue, so the search revisits just the part of
the graph the change affects instead of starting from the goal again.

Two tricks make the search reusable as the rover moves. The queue keys are measured from the rover's
current cell, so the rover can move without rebuilding the queue; the offset `km` absorbs the
distance the rover has moved since the last repair, so older keys stay valid.

Unknown cells are planned as free (cost 1). A cell becomes a wall only when the sensor sees it, and
every edge touching a wall costs infinity. The rover therefore plans optimistically and repairs the
plan as it learns.

Conventions: 4-connected grid, unit step cost, Manhattan-distance heuristic (consistent for unit
costs). "Expansion" means one queue pop that is processed, the same count A* uses.
"""

from __future__ import annotations

import heapq

from .world import neighbours

INF = float("inf")


class DStarLite:
    """Incremental shortest-path planner for one rover on one map.

    Call `on_change` whenever the sensor flips some cells between free and wall. Call `next_cell` to
    pick the rover's next step. `expansions` counts queue pops over the planner's whole life.
    """

    def __init__(self, n: int, start: int, goal: int):
        self.n = n
        self.goal = goal
        self.start = start        # s_start: the rover's current cell, which moves over time
        self.s_last = start       # the rover's cell when the queue was last repaired
        self.km = 0               # accumulated heuristic offset, see the module docstring
        N = n * n
        self.wall = bytearray(N)  # the planner's belief: 1 = known wall (unknown counts as free)
        self.g = [INF] * N
        self.rhs = [INF] * N
        # Queue. Entries are (k1, k2, cell, version). ver[cell] is the version of the live entry, or 0
        # when the cell is not queued. Removing a cell just sets ver to 0 and leaves its heap entry in
        # place; stale entries are skipped when they reach the top. This is cheaper than deleting from
        # the middle of a heap.
        self.heap: list[tuple] = []
        self.ver = [0] * N
        self.version = 0
        self.expansions = 0
        self.last_expanded: list[int] = []  # cells popped in the latest repair, for the page's heat map
        # Initialise: the goal has rhs 0 and is the only queued cell, so the first repair grows the
        # search outward from the goal over an all-free map.
        self.rhs[goal] = 0
        self._insert(goal, self._key(goal))
        self._compute()

    # ── helpers ──────────────────────────────────────────────────────────

    def _h(self, a: int, b: int) -> int:
        """Manhattan distance: admissible and consistent for unit steps on a 4-connected grid."""
        n = self.n
        return abs(a // n - b // n) + abs(a % n - b % n)

    def _cost(self, u: int, v: int):
        """Cost of one step between neighbours u and v: 1, or infinity if either end is a known wall."""
        return INF if (self.wall[u] or self.wall[v]) else 1

    def _key(self, u: int):
        """Queue key. k1 is the best cost estimate through u (measured from the rover); k2 breaks ties.

        min(g, rhs) is the value the cell currently offers. Adding h(start, u) and km makes the key
        an estimate of the total route cost from the rover's cell through u.
        """
        m = min(self.g[u], self.rhs[u])
        return (m + self._h(self.start, u) + self.km, m)

    def _insert(self, u: int, key) -> None:
        self.version += 1
        self.ver[u] = self.version
        heapq.heappush(self.heap, (key[0], key[1], u, self.version))

    def _top(self):
        """Key of the cheapest live queue entry (dropping stale entries first), or (INF, INF) if empty."""
        heap = self.heap
        while heap and self.ver[heap[0][2]] != heap[0][3]:
            heapq.heappop(heap)
        return (heap[0][0], heap[0][1]) if heap else (INF, INF)

    def _pop(self):
        """Remove and return the cheapest live entry as (key, cell)."""
        heap = self.heap
        while True:
            k1, k2, u, v = heapq.heappop(heap)
            if self.ver[u] == v:
                self.ver[u] = 0
                return (k1, k2), u

    def _update(self, u: int) -> None:
        """UpdateVertex from the paper: recompute rhs(u) from its neighbours, then (re)queue it if inconsistent."""
        if u != self.goal:
            best = INF
            for s in neighbours(self.n, u):
                c = self._cost(u, s) + self.g[s]
                if c < best:
                    best = c
            self.rhs[u] = best
        if self.ver[u]:
            self.ver[u] = 0          # drop the old queue entry
        if self.g[u] != self.rhs[u]:
            self._insert(u, self._key(u))

    def _compute(self) -> None:
        """ComputeShortestPath: pop cells until the rover's cell is consistent and nothing cheaper is queued.

        Popping a cell that is overconsistent (g > rhs) lowers g to rhs and updates its neighbours.
        Popping an underconsistent cell (g < rhs) first sets g to infinity and then re-checks the cell
        and its neighbours, which removes the stale value the map change left behind.
        """
        self.last_expanded = []
        while True:
            top = self._top()          # also drops stale entries, so the emptiness test below is exact
            if not self.heap:
                break
            ks = self._key(self.start)
            if not (top < ks or self.rhs[self.start] != self.g[self.start]):
                break
            k_old, u = self._pop()
            self.expansions += 1
            self.last_expanded.append(u)
            k_new = self._key(u)
            if k_old < k_new:
                # The key was measured from an older rover position or km, so refresh it and requeue.
                self._insert(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in neighbours(self.n, u):
                    self._update(s)
            else:
                self.g[u] = INF
                for s in neighbours(self.n, u):
                    self._update(s)
                self._update(u)

    # ── public API ───────────────────────────────────────────────────────

    def on_change(self, changes, pos: int) -> None:
        """Apply sensed changes and repair the plan for a rover now at `pos`.

        `changes` is a list of (cell, is_wall). The rover's move since the last repair changes the
        heuristic offsets, so km grows by h(s_last, pos) before the keys are recomputed.
        """
        self.km += self._h(self.s_last, pos)
        self.s_last = pos
        self.start = pos
        for cell, wall in changes:
            self.wall[cell] = wall
        # Every edge touching a changed cell has a new cost, so both its endpoints need updating.
        for cell, _ in changes:
            self._update(cell)
            for s in neighbours(self.n, cell):
                self._update(s)
        self._compute()

    def cost_to_goal(self, pos: int) -> float:
        """Planned cost from `pos` to the goal (INF if the planner knows no route)."""
        return self.g[pos]

    def next_cell(self, pos: int):
        """The neighbour of `pos` with the lowest c(pos, s) + g(s), or None if there is no route.

        This is a pure lookup: it does not change the planner, so it can also trace the whole path.
        """
        best, best_v = None, INF
        for s in neighbours(self.n, pos):
            v = self._cost(pos, s) + self.g[s]
            if v < best_v:
                best, best_v = s, v
        return best if best_v < INF else None

    def path_from(self, pos: int) -> list[int]:
        """The planned route from pos to the goal, following next_cell. Empty if there is no route."""
        if self.g[pos] == INF:
            return []
        path, cur = [pos], pos
        limit = self.n * self.n
        while cur != self.goal and len(path) <= limit:
            cur = self.next_cell(cur)
            if cur is None:
                return []
            path.append(cur)
        return path
