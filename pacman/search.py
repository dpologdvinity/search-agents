"""Shortest paths on a maze: breadth-first distances and A* routes.

Every move costs one step, so:
  - breadth-first search (BFS) from one cell gives the exact walking distance to every
    other cell in one pass. The features and the scared-ghost flight both use it.
  - A* finds a shortest route to one goal. Its heuristic is the Manhattan distance,
    which never overestimates on a grid with unit moves, so the first time the goal
    leaves the queue its route is optimal. It expands far fewer cells than BFS would.
"""

from __future__ import annotations

import heapq
from collections import deque

from .mazes import NO_CELL, Maze

UNREACHABLE = -1


def distances(maze: Maze, source: int) -> list[int]:
    """Walking distance from `source` to every cell; UNREACHABLE for walls and cut-off cells."""
    dist = [UNREACHABLE] * maze.cells
    dist[source] = 0
    queue = deque([source])
    while queue:
        cur = queue.popleft()
        nxt = dist[cur] + 1
        for nb in maze.nbr[cur]:
            if nb != NO_CELL and dist[nb] == UNREACHABLE:
                dist[nb] = nxt
                queue.append(nb)
    return dist


def manhattan(maze: Maze, a: int, b: int) -> int:
    """Grid distance ignoring walls. Used as the A* heuristic and for ghost fallbacks."""
    return abs(maze.row[a] - maze.row[b]) + abs(maze.col[a] - maze.col[b])


def astar(maze: Maze, start: int, goal: int) -> list[int] | None:
    """Shortest route from `start` to `goal`, both included, or None if there is none.

    The queue is ordered by f = g + h, where g is the steps taken so far and h is the
    Manhattan distance to the goal. On ties the cell with the smaller h (nearer the goal)
    is expanded first, which cuts the number of cells explored. Each cell is expanded at
    most once; a stale queue entry for a cell already closed is skipped.
    """
    if start == goal:
        return [start]
    best_g = {start: 0}
    parent = {start: NO_CELL}
    h0 = manhattan(maze, start, goal)
    heap = [(h0, h0, start)]
    closed: set[int] = set()
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur == goal:
            route = [cur]
            while parent[route[-1]] != NO_CELL:
                route.append(parent[route[-1]])
            route.reverse()
            return route
        if cur in closed:
            continue
        closed.add(cur)
        g = best_g[cur] + 1
        for nb in maze.nbr[cur]:
            if nb == NO_CELL or nb in closed:
                continue
            if g < best_g.get(nb, g + 1):
                best_g[nb] = g
                parent[nb] = cur
                h = manhattan(maze, nb, goal)
                heapq.heappush(heap, (g + h, h, nb))
    return None
