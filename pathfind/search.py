"""Seven search algorithms on a Grid: BFS, DFS, uniform cost, greedy best-first, A*, weighted A*, and
bidirectional BFS.

Every algorithm returns a Result whose `steps` list records each expansion in order, with the cells
it discovered. The browser replays those steps to animate the search, and the step list is what the
Python and JS parity tests compare.

Shared rules, so runs are deterministic and comparable:
- A cell is expanded when it is popped, and the goal test runs at that moment, so a goal is never
  returned before a cheaper one still in the frontier has been considered.
- Heap entries are (priority, insertion number, cell). The insertion number breaks every tie, so
  equal priorities pop in the order they were pushed.
- Neighbours are visited in the order given by Grid.neighbors (N, E, S, W, then diagonals).
- Each step records g, the cost from start along the tree so far, for every new cell. The browser
  shows g, h and f for any cell from these values.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

from .grid import Grid, heuristic

ALGOS = ("bfs", "dfs", "ucs", "greedy", "astar", "wastar", "bidi")
LABELS = {
    "bfs": "Breadth-first",
    "dfs": "Depth-first",
    "ucs": "Uniform cost (Dijkstra)",
    "greedy": "Greedy best-first",
    "astar": "A*",
    "wastar": "Weighted A*",
    "bidi": "Bidirectional BFS",
}
HEURISTIC_ALGOS = ("greedy", "astar", "wastar")
DEFAULT_WEIGHT = 2.0

NOT_SEEN = -2  # parent value for a cell no search has discovered yet
ROOT = -1  # parent value of the start cell


@dataclass
class Result:
    """Outcome of one search.

    steps: one (cell, side, discovered) tuple per expansion. side is 0 for the forward search and 1
    for the backward search of bidirectional BFS; discovered is a list of (cell, g) pairs.
    """

    algo: str
    found: bool
    path: list[int]
    cost: float | None
    steps: list[tuple[int, int, list[tuple[int, float]]]] = field(default_factory=list)
    generated: int = 0

    @property
    def expanded(self) -> int:
        return len(self.steps)

    @property
    def expansion_order(self) -> list[int]:
        return [s[0] for s in self.steps]

    def frontier_sizes(self) -> list[int]:
        """Frontier size after each expansion: cells discovered so far that are not yet expanded."""
        seen: set[int] = set()
        expanded: set[int] = set()
        sizes = []
        for cell, _side, new in self.steps:
            expanded.add(cell)
            for v, _g in new:
                seen.add(v)
            sizes.append(len(seen - expanded))
        return sizes


def _chain(parent: list[int], v: int) -> list[int]:
    """Follow parent links from v back to the root; returned root first."""
    out = [v]
    while parent[v] != ROOT:
        v = parent[v]
        out.append(v)
    out.reverse()
    return out


def _finish(grid: Grid, algo: str, steps, goal: int, path: list[int], generated: int) -> Result:
    found = bool(steps) and steps[-1][0] == goal
    if not found:
        return Result(algo, False, [], None, steps, generated)
    return Result(algo, True, path, grid.path_cost(path), steps, generated)


def bfs(grid: Grid, start: int, goal: int, diagonal: bool = False) -> Result:
    """Breadth-first: expands by number of steps. Shortest in steps, not in terrain cost."""
    n = len(grid)
    parent = [NOT_SEEN] * n
    g = [0.0] * n
    parent[start] = ROOT
    queue = deque([start])
    steps: list = []
    generated = 0
    while queue:
        u = queue.popleft()
        new: list[tuple[int, float]] = []
        steps.append((u, 0, new))
        if u == goal:
            break
        for v, mult in grid.neighbors(u, diagonal):
            if parent[v] == NOT_SEEN:
                parent[v] = u
                g[v] = g[u] + mult * grid.cells[v]
                queue.append(v)
                new.append((v, g[v]))
                generated += 1
    path = _chain(parent, goal) if steps and steps[-1][0] == goal else []
    return _finish(grid, "bfs", steps, goal, path, generated)


def dfs(grid: Grid, start: int, goal: int, diagonal: bool = False) -> Result:
    """Depth-first: the stack holds (cell, parent, g) entries and the last pushed is expanded first.
    The path it finds is whatever branch it reached the goal down, not a short one."""
    n = len(grid)
    closed = [False] * n
    parent = [ROOT] * n
    g = [0.0] * n
    stack: list[tuple[int, int, float]] = [(start, ROOT, 0.0)]
    steps: list = []
    generated = 0
    while stack:
        u, p, gu = stack.pop()
        if closed[u]:
            continue
        closed[u] = True
        parent[u] = p
        g[u] = gu
        new: list[tuple[int, float]] = []
        steps.append((u, 0, new))
        if u == goal:
            break
        for v, mult in grid.neighbors(u, diagonal):
            if not closed[v]:
                gv = gu + mult * grid.cells[v]
                stack.append((v, u, gv))
                new.append((v, gv))
                generated += 1
    path = _chain(parent, goal) if steps and steps[-1][0] == goal else []
    return _finish(grid, "dfs", steps, goal, path, generated)


def _best_first(grid: Grid, algo: str, start: int, goal: int, *, diagonal: bool, heur: str, weight: float) -> Result:
    """Shared loop for uniform cost (priority g), greedy (priority h) and A* (priority g + w*h).

    A cell's entry is pushed again whenever a cheaper route to it turns up (lazy deletion), and stale
    entries are skipped when popped.
    """
    n = len(grid)
    gx, gy = grid.xy(goal)
    inf = float("inf")
    g = [inf] * n
    parent = [ROOT] * n
    closed = [False] * n

    def priority(v: int, gv: float) -> float:
        if algo == "ucs":
            return gv
        x, y = grid.xy(v)
        h = heuristic(heur, x, y, gx, gy)
        if algo == "greedy":
            return h
        return gv + weight * h

    g[start] = 0.0
    heap: list[tuple[float, int, int]] = [(priority(start, 0.0), 0, start)]
    seq = 0
    steps: list = []
    generated = 0
    while heap:
        _, _, u = heapq.heappop(heap)
        if closed[u]:
            continue
        closed[u] = True
        new: list[tuple[int, float]] = []
        steps.append((u, 0, new))
        if u == goal:
            break
        for v, mult in grid.neighbors(u, diagonal):
            if closed[v]:
                continue
            ng = g[u] + mult * grid.cells[v]
            if ng < g[v]:
                g[v] = ng
                parent[v] = u
                seq += 1
                heapq.heappush(heap, (priority(v, ng), seq, v))
                new.append((v, ng))
                generated += 1
    path = _chain(parent, goal) if steps and steps[-1][0] == goal else []
    return _finish(grid, algo, steps, goal, path, generated)


def uniform_cost(grid: Grid, start: int, goal: int, diagonal: bool = False) -> Result:
    """Dijkstra: expands by path cost so far. Always optimal; the reference the others are judged by."""
    return _best_first(grid, "ucs", start, goal, diagonal=diagonal, heur="manhattan", weight=0.0)


def greedy(grid: Grid, start: int, goal: int, diagonal: bool = False, heur: str = "octile") -> Result:
    """Greedy best-first: expands by the heuristic alone. Fast, and not guaranteed to be optimal."""
    return _best_first(grid, "greedy", start, goal, diagonal=diagonal, heur=heur, weight=1.0)


def astar(grid: Grid, start: int, goal: int, diagonal: bool = False, heur: str = "octile") -> Result:
    """A*: expands by g + h. Optimal when h never overestimates (see grid.heuristic)."""
    return _best_first(grid, "astar", start, goal, diagonal=diagonal, heur=heur, weight=1.0)


def weighted_astar(
    grid: Grid, start: int, goal: int, diagonal: bool = False, heur: str = "octile", weight: float = DEFAULT_WEIGHT
) -> Result:
    """Weighted A*: expands by g + w * h with w >= 1. Its path costs at most w times the optimum."""
    if weight < 1:
        raise ValueError("weight must be at least 1")
    return _best_first(grid, "wastar", start, goal, diagonal=diagonal, heur=heur, weight=weight)


def bidirectional_bfs(grid: Grid, start: int, goal: int, diagonal: bool = False) -> Result:
    """Breadth-first from both ends, one whole level at a time, until the two searches have met.
    It searches in steps, like BFS, so its path is as short in steps as BFS's path.

    The first meeting found is not always the shortest: a later cell in the same level can meet the
    other search closer to its root. So every meeting is scored by its total steps, and the search
    stops only when no shorter path can exist: once the best meeting uses no more steps than the
    two completed levels reach (any path that short has a cell both searches have already found)."""
    n = len(grid)
    parents = ([NOT_SEEN] * n, [NOT_SEEN] * n)  # side 0 points back to start, side 1 forward to goal
    depth = ([0] * n, [0] * n)  # steps from start (side 0) or to goal (side 1)
    g = ([0.0] * n, [0.0] * n)  # side 0: cost start -> cell; side 1: cost cell -> goal
    parents[0][start] = ROOT
    parents[1][goal] = ROOT
    queues = (deque([start]), deque([goal]))
    done = [0, 0]  # whole levels each side has expanded, so its cells are all found within that many steps
    steps: list = []
    generated = 0
    if start == goal:
        steps.append((start, 0, []))
        return Result("bidi", True, [start], 0.0, steps, 0)
    best = -1  # steps in the shortest meeting path found so far
    meet = -1
    while queues[0] or queues[1]:
        if best >= 0 and best <= done[0] + done[1]:
            break
        # Expand the side with the smaller next level; ties go to the forward search.
        if queues[0] and (not queues[1] or len(queues[0]) <= len(queues[1])):
            side = 0
        else:
            side = 1
        # The queue holds exactly one level at this point, so expand all of it before switching sides.
        for _ in range(len(queues[side])):
            u = queues[side].popleft()
            new: list[tuple[int, float]] = []
            steps.append((u, side, new))
            for v, mult in grid.neighbors(u, diagonal):
                if parents[side][v] != NOT_SEEN:
                    continue
                parents[side][v] = u
                depth[side][v] = depth[side][u] + 1
                # The step is entered from u forward, or from v backward, so the cost is of the cell entered.
                entered = v if side == 0 else u
                g[side][v] = g[side][u] + mult * grid.cells[entered]
                queues[side].append(v)
                new.append((v, g[side][v]))
                generated += 1
                if parents[1 - side][v] != NOT_SEEN:
                    total = depth[side][v] + depth[1 - side][v]
                    if best < 0 or total < best:
                        best, meet = total, v
        done[side] += 1
    if best < 0:
        return Result("bidi", False, [], None, steps, generated)
    forward = _chain(parents[0], meet)
    backward = _chain(parents[1], meet)  # goal ... meet, root first at goal
    backward.reverse()  # now meet ... goal
    path = forward + backward[1:]
    return Result("bidi", True, path, grid.path_cost(path), steps, generated)


def run(
    algo: str,
    grid: Grid,
    start: int,
    goal: int,
    *,
    diagonal: bool = False,
    heur: str = "octile",
    weight: float = DEFAULT_WEIGHT,
) -> Result:
    """Run one algorithm by name. heur and weight are used only by the heuristic algorithms."""
    if algo == "bfs":
        return bfs(grid, start, goal, diagonal)
    if algo == "dfs":
        return dfs(grid, start, goal, diagonal)
    if algo == "ucs":
        return uniform_cost(grid, start, goal, diagonal)
    if algo == "greedy":
        return greedy(grid, start, goal, diagonal, heur)
    if algo == "astar":
        return astar(grid, start, goal, diagonal, heur)
    if algo == "wastar":
        return weighted_astar(grid, start, goal, diagonal, heur, weight)
    if algo == "bidi":
        return bidirectional_bfs(grid, start, goal, diagonal)
    raise ValueError(f"unknown algorithm {algo!r}")
