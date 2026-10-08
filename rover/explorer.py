"""The rover loop: sense, replan when the map changes, then take one step.

Each tick does three things, in this order:
  1. Sense. Every cell inside the sensor square is copied from the true map into the rover's belief.
     Only cells whose wall state differs from the belief count as changes (a cell that turns out to be
     free was already planned as free, so nothing needs repairing).
  2. Replan. If there are changes, both planners are told about them. D* Lite repairs its queue; A*
     searches from scratch. Both are always updated, so their costs and expansion counts can be
     compared on exactly the same sequence of beliefs.
  3. Move. The driving planner picks the next cell. The rover moves only into cells it believes are
     free. The sensor always covers the adjacent cells, so that belief is also true.

Because both planners see the same changes at the same moments, their plans have the same cost at every
replan. The explorer checks this (`mismatches`), and the tests check it against a fresh A* search.

The user can edit the true map at any time with `edit`, which senses and replans at once without moving.
That is how the page makes the rover replan when a wall is dropped in front of it.
"""

from __future__ import annotations

import time

from .astar import AStar
from .dstar import DStarLite
from .world import World, sensed_cells

PLANNERS = ("dstar", "astar")


class Explorer:
    """One rover exploring one World. Both planners run on the same belief; `driver` picks who steers."""

    def __init__(self, world: World, radius: int = 2, driver: str = "dstar"):
        if driver not in PLANNERS:
            raise ValueError(f"driver must be one of {PLANNERS}")
        if radius < 1:
            raise ValueError("the sensor radius must be at least 1 so the rover sees the cells it steps onto")
        self.world = world
        self.n = world.n
        self.radius = radius
        self.driver = driver
        N = self.n * self.n
        self.belief = bytearray(N)   # 1 = wall as the rover currently believes it
        self.seen = bytearray(N)     # 1 = the sensor has covered this cell at least once
        self.pos = world.start
        self.steps = 0
        self.replans = 0             # ticks on which the sensor changed something
        self.done = False
        self.stuck = False           # true when the driver has no route and the rover waits
        self.mismatches = 0          # replans where D* Lite and A* disagree on the cost (should stay 0)
        self.seconds = {"dstar": 0.0, "astar": 0.0}  # time spent inside each planner
        self.last_changes: list[tuple[int, int]] = []  # changes applied by the latest replan
        # Both planners plan on an all-free map here; the first sensing then arrives as changes.
        # Their work is counted in `expansions` like any other replan, and kept apart in `init_expansions`.
        self.planners = {}
        self.init_seconds = {}
        for name, cls in (("dstar", DStarLite), ("astar", AStar)):
            t0 = time.perf_counter()
            self.planners[name] = cls(self.n, world.start, world.goal)
            self.init_seconds[name] = time.perf_counter() - t0
            self.seconds[name] += self.init_seconds[name]
        self.init_expansions = {name: p.expansions for name, p in self.planners.items()}

    # ── state readers ────────────────────────────────────────────────────

    def expansions(self, name: str) -> int:
        return self.planners[name].expansions

    def path(self, name: str | None = None) -> list[int]:
        """Route the planner would take from the rover's cell (the driver by default)."""
        name = name or self.driver
        if name == "dstar":
            return self.planners["dstar"].path_from(self.pos)
        p = self.planners["astar"]
        return list(p.path) if p.path and p.path[0] == self.pos else []

    # ── the loop ─────────────────────────────────────────────────────────

    def sense(self) -> list[tuple[int, int]]:
        """Copy the sensor's view into the belief. Returns the cells whose wall state changed."""
        changes = []
        world_walls = self.world.walls
        for c in sensed_cells(self.n, self.pos, self.radius):
            self.seen[c] = 1
            if world_walls[c] != self.belief[c]:
                self.belief[c] = world_walls[c]
                changes.append((c, world_walls[c]))
        return changes

    def refresh(self) -> int:
        """Sense, and if anything changed, tell both planners. Returns the number of changed cells."""
        changes = self.sense()
        if not changes:
            return 0
        self.replans += 1
        self.last_changes = changes
        for name in PLANNERS:
            t0 = time.perf_counter()
            self.planners[name].on_change(changes, self.pos)
            self.seconds[name] += time.perf_counter() - t0
        # Cross-check: both planners now report the cost from the rover's cell. They must agree.
        a, b = self.planners["dstar"].cost_to_goal(self.pos), self.planners["astar"].cost_to_goal(self.pos)
        if a != b:
            self.mismatches += 1
        return len(changes)

    def tick(self) -> None:
        """One step of the rover loop: sense and replan, then move one cell (or wait if there is no route)."""
        if self.done:
            return
        self.refresh()
        if self.pos == self.world.goal:
            self.done = True
            return
        planner = self.planners[self.driver]
        t0 = time.perf_counter()
        nxt = planner.next_cell(self.pos)
        self.seconds[self.driver] += time.perf_counter() - t0
        if nxt is None:
            self.stuck = True        # no known route: wait for the map to change
            return
        # The sensor covers every neighbour, so the belief is true for nxt. Walking into a wall
        # would be a bug in the planner or the sensor.
        assert not self.world.walls[nxt], "the rover stepped into a wall"
        self.stuck = False
        self.pos = nxt
        self.steps += 1
        if self.pos == self.world.goal:
            self.done = True

    def edit(self, cell: int, wall: bool) -> bool:
        """The user changes the true map. The rover senses and replans now if the change is in view.

        The rover's own cell and the goal cannot be edited. Returns False for those, True otherwise.
        """
        if cell in (self.pos, self.world.goal):
            return False
        self.world.set_wall(cell, wall)
        self.refresh()
        return True

    def run(self, max_ticks: int | None = None) -> Explorer:
        """Tick until the goal is reached, or until max_ticks (default 20 cells per map cell)."""
        limit = max_ticks if max_ticks is not None else 20 * self.n * self.n
        for _ in range(limit):
            if self.done:
                break
            self.tick()
        return self
