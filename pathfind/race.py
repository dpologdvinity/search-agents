"""Run several algorithms on one map and compare them, plus the ASCII renderer the CLI prints with.

The optimal cost is the uniform-cost (Dijkstra) answer on the same map and movement rules. An
algorithm is marked optimal when its path cost matches that to within floating-point noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from .grid import SWAMP, WALL, Grid
from .mazes import Maze
from .search import ALGOS, DEFAULT_WEIGHT, Result, run, uniform_cost

EPS = 1e-9


@dataclass
class Row:
    """One line of the leaderboard."""

    algo: str
    result: Result
    optimal: float | None

    @property
    def is_optimal(self) -> bool:
        return self.result.found and self.optimal is not None and self.result.cost <= self.optimal + EPS


def race(
    maze: Maze,
    algos: tuple[str, ...] = ALGOS,
    *,
    diagonal: bool = False,
    heur: str = "octile",
    weight: float = DEFAULT_WEIGHT,
) -> tuple[float | None, list[Row]]:
    """Run each algorithm on the maze. Returns (optimal cost or None if no path, rows in the order given)."""
    ref = uniform_cost(maze.grid, maze.start, maze.goal, diagonal)
    optimal = ref.cost if ref.found else None
    rows = []
    for algo in algos:
        res = run(algo, maze.grid, maze.start, maze.goal, diagonal=diagonal, heur=heur, weight=weight)
        rows.append(Row(algo, res, optimal))
    return optimal, rows


def table(rows: list[Row], optimal: float | None) -> str:
    """Plain-text leaderboard: expansions, path cost and whether the cost is optimal."""
    head = f"{'ALGO':<8} {'FOUND':<6} {'EXPANDED':>9} {'GENERATED':>10} {'PATH':>6} {'COST':>10}  OPTIMAL"
    lines = [head, "-" * len(head)]
    for row in rows:
        r = row.result
        cost = "-" if r.cost is None else f"{r.cost:.2f}"
        if not r.found:
            mark = "no path"
        else:
            mark = "yes" if row.is_optimal else f"no (+{r.cost - optimal:.2f})"
        lines.append(
            f"{row.algo:<8} {'yes' if r.found else 'no':<6} {r.expanded:>9} {r.generated:>10} "
            f"{len(r.path):>6} {cost:>10}  {mark}"
        )
    if optimal is None:
        lines.append("no route exists between start and goal")
    else:
        lines.append(f"optimal cost (Dijkstra reference): {optimal:.2f}")
    return "\n".join(lines)


def render(grid: Grid, start: int, goal: int, path: list[int], expanded: list[int] | None = None) -> str:
    """ASCII map. '#' wall, '~' swamp, '.' open, ':' expanded, '*' path, 'S' start, 'G' goal."""
    on_path = set(path)
    seen = set(expanded or ())
    out = []
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
            i = grid.index(x, y)
            if i == start:
                ch = "S"
            elif i == goal:
                ch = "G"
            elif i in on_path:
                ch = "*"
            elif grid.cells[i] == WALL:
                ch = "#"
            elif i in seen:
                ch = ":"
            elif grid.cells[i] == SWAMP:
                ch = "~"
            else:
                ch = "."
            row.append(ch)
        out.append("".join(row))
    return "\n".join(out)
