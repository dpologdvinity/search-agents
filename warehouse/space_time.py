"""Single-robot planning in space-time: A* over (cell, time) states, honouring reservations.

The low level of CBS and the core of prioritized planning. A robot's plan is searched in the
time-expanded graph, where a state is a cell at a time step and each step is a move to a
neighbour or a wait. Anything another robot (or a CBS constraint) forbids is kept in a
``Reservations`` table:

- vertex block ``(cell, t)``: the robot may not be on ``cell`` at time ``t``;
- edge block ``(src, dst, t)``: the robot may not move ``src -> dst`` arriving at time ``t``
  (this is what forbids swapping with another robot);
- parked cell: another robot has reached its goal and stays there forever from time ``t0``.

A robot's arrival is valid only if its goal stays free afterwards. Otherwise the robot would
arrive, be pushed off, or block a parked robot later, so the search keeps going.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass

from .conflicts import Path
from .grid import Grid


@dataclass(frozen=True)
class VertexConstraint:
    """Robot ``robot`` may not be on ``cell`` at time ``t``."""

    robot: int
    cell: int
    t: int

    def to_dict(self) -> dict:
        return {"kind": "vertex", "robot": self.robot, "cell": self.cell, "t": self.t}


@dataclass(frozen=True)
class EdgeConstraint:
    """Robot ``robot`` may not move ``src -> dst`` so that it arrives at ``dst`` at time ``t``."""

    robot: int
    src: int
    dst: int
    t: int

    def to_dict(self) -> dict:
        return {"kind": "edge", "robot": self.robot, "src": self.src, "dst": self.dst, "t": self.t}


class Reservations:
    """The set of forbidden (cell, time) and (move, time) facts that one robot's search must respect."""

    def __init__(self, constraints: Iterable[VertexConstraint | EdgeConstraint] = ()):
        self.vertex: set[tuple[int, int]] = set()
        self.edge: set[tuple[int, int, int]] = set()
        self.parked: dict[int, int] = {}  # cell -> first time step from which it is blocked for good
        self.last_vertex: dict[int, int] = {}  # cell -> latest time with a vertex block (for the goal test)
        self._max_t = 0
        for c in constraints:
            if isinstance(c, VertexConstraint):
                self.add_vertex(c.cell, c.t)
            else:
                self.add_edge(c.src, c.dst, c.t)

    def add_vertex(self, cell: int, t: int) -> None:
        self.vertex.add((cell, t))
        if t > self.last_vertex.get(cell, -1):
            self.last_vertex[cell] = t
        self._max_t = max(self._max_t, t)

    def add_edge(self, src: int, dst: int, t: int) -> None:
        self.edge.add((src, dst, t))
        self._max_t = max(self._max_t, t)

    def add_path(self, path: Path) -> None:
        """Reserve everything another robot does along its path, then its parked goal cell."""
        for t, c in enumerate(path):
            self.add_vertex(c, t)
            if t > 0 and path[t - 1] != c:
                # The other robot moved path[t-1] -> c. Forbid the reverse move at time t,
                # which would be a swap with it.
                self.add_edge(c, path[t - 1], t)
        end = len(path) - 1
        self.parked[path[-1]] = end
        self._max_t = max(self._max_t, end)

    def blocked_vertex(self, cell: int, t: int) -> bool:
        if (cell, t) in self.vertex:
            return True
        parked_from = self.parked.get(cell)
        return parked_from is not None and t >= parked_from

    def blocked_edge(self, src: int, dst: int, t: int) -> bool:
        return (src, dst, t) in self.edge

    def can_stay_from(self, cell: int, t: int) -> bool:
        """True if a robot arriving on ``cell`` at ``t`` can stay there for the rest of time."""
        return cell not in self.parked and self.last_vertex.get(cell, -1) < t

    def max_time(self) -> int:
        """Last time step at which anything is forbidden. Beyond it, only distance matters."""
        return self._max_t


def space_time_astar(grid: Grid, start: int, goal: int, res: Reservations,
                     horizon: int | None = None) -> tuple[Path | None, int]:
    """Earliest-arrival path from ``start`` to ``goal`` that respects ``res``.

    Returns ``(path, expanded)``: the cells at times 0..T (the path ends on arrival), or None if
    no path exists within ``horizon`` steps, and the number of states expanded.

    Why it is optimal: the heuristic is the static walking distance, which never overestimates
    and drops by at most one per step, so it is consistent. States pop in order of
    ``f = t + h``, and waiting costs one step without reducing ``h``. The first valid goal
    state popped therefore has the smallest arrival time. A goal state with a later block
    is not valid, so the search continues past it.

    Why ``horizon`` is enough by default: after ``res.max_time()`` nothing is forbidden, so a robot
    that is still alive then can walk the rest of the way in at most ``grid.size`` steps.
    """
    dist = grid.distances_to(goal)
    if dist[start] < 0:
        return None, 0
    if horizon is None:
        horizon = res.max_time() + grid.size + 1
    # Heap entries: (f, -t, cell, t). Ties go to the deeper state (larger t), which reaches the
    # goal with fewer expansions.
    heap = [(dist[start], 0, start, 0)]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(start, 0): None}
    expanded = 0
    while heap:
        _, _, cell, t = heapq.heappop(heap)
        expanded += 1
        if cell == goal and res.can_stay_from(goal, t):
            return _walk_back(parent, cell, t), expanded
        nt = t + 1
        if nt > horizon:
            continue
        # Try waiting first, then each move. Blocked states are dropped here, not when popped.
        for nxt in (cell, *grid.neighbors[cell]):
            if dist[nxt] < 0 or (nxt, nt) in parent:
                continue
            if res.blocked_vertex(nxt, nt):
                continue
            if nxt != cell and res.blocked_edge(cell, nxt, nt):
                continue
            parent[(nxt, nt)] = (cell, t)
            heapq.heappush(heap, (nt + dist[nxt], -nt, nxt, nt))
    return None, expanded


def _walk_back(parent: dict, cell: int, t: int) -> Path:
    """Rebuild the cell sequence by following parent pointers from the goal state."""
    cells = [cell]
    state = (cell, t)
    while parent[state] is not None:
        state = parent[state]
        cells.append(state[0])
    cells.reverse()
    return tuple(cells)
