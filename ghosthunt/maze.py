"""The board: a perfect maze on an odd grid, its open cells, and the distance between every pair of them.

generate()   a seeded perfect maze (recursive backtracker; every choice comes from mulberry32, so the page
             draws the same maze for the same seed)
Maze         the open cells numbered 0..K-1, the open neighbour in each direction, and all-pairs BFS distances
render()     an ASCII picture of the maze for the terminal

Directions are numbered 0..3 = N, E, S, W (rows grow downward). Neighbour lists, tie-breaks and the JavaScript
port all use this order, so every decision the code makes is reproducible.

Why a perfect maze: it has exactly one route between any two open cells, so the distance the sonar measures
is the length of the corridor path, not a straight-line guess. A patrol ghost can then follow corridors.
"""

from __future__ import annotations

from collections import deque

from bandits.rng import Rng

DR = (-1, 0, 1, 0)  # row step for N, E, S, W
DC = (0, 1, 0, -1)  # column step for N, E, S, W
DIR_CHARS = "NESW"
SIZES = (11, 15, 21)  # odd side lengths offered by the page and the CLI


def generate(n: int, seed: int) -> list[int]:
    """A perfect maze of side n (odd, at least 5) as a flat row-major list; 1 is a wall, 0 is open.

    Rooms sit at odd coordinates. The walk starts at room (1, 1), and from the current room it picks one of the
    unvisited rooms two steps away, knocks out the wall between them, and moves there. When no room is left
    unvisited it backtracks. Every room is reached exactly once, so the result is a tree: one path between any
    two open cells.
    """
    if n < 5 or n % 2 == 0:
        raise ValueError("maze side must be an odd number, at least 5")
    rng = Rng(seed)
    walls = [1] * (n * n)
    walls[n + 1] = 0  # room (1, 1)
    stack = [(1, 1)]
    while stack:
        r, c = stack[-1]
        options = []
        for d in range(4):
            r2, c2 = r + 2 * DR[d], c + 2 * DC[d]
            # A room is still unvisited while its cell is a wall; the outer ring stays solid.
            if 0 < r2 < n - 1 and 0 < c2 < n - 1 and walls[r2 * n + c2]:
                options.append((d, r2, c2))
        if not options:
            stack.pop()
            continue
        d, r2, c2 = options[rng.index(len(options))]
        walls[(r + DR[d]) * n + (c + DC[d])] = 0  # the passage between the two rooms
        walls[r2 * n + c2] = 0
        stack.append((r2, c2))
    return walls


class Maze:
    """A maze with its open cells numbered 0..K-1 (row-major order) and all-pairs shortest distances.

    step[k][d] is the open index of the neighbour of cell k in direction d, or -1 for a wall or the edge.
    dist[a][b] is the number of steps between open cells a and b along the corridors (the maze is a tree,
    so this is the only route length). The start is room (1, 1), the top-left room.
    """

    def __init__(self, n: int, walls):
        self.n = n
        self.walls = tuple(walls)
        self.cells = [c for c in range(n * n) if not walls[c]]
        self.K = len(self.cells)
        self.k_of = [-1] * (n * n)
        for k, c in enumerate(self.cells):
            self.k_of[c] = k
        self.step: list[list[int]] = []
        for c in self.cells:
            r, col = divmod(c, n)
            row = []
            for d in range(4):
                r2, c2 = r + DR[d], col + DC[d]
                inside = 0 <= r2 < n and 0 <= c2 < n
                row.append(self.k_of[r2 * n + c2] if inside else -1)
            self.step.append(row)
        self.dist = [self._bfs(k) for k in range(self.K)]
        self.diameter = max(max(row) for row in self.dist)  # the largest reading the sonar can report
        self.start = self.k_of[n + 1]
        self.cache: dict = {}  # memo for motion tables that do not depend on the player

    def _bfs(self, src: int) -> list[int]:
        dist = [-1] * self.K
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in self.step[u]:
                if v >= 0 and dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def cell(self, k: int) -> int:
        """Flat grid index of open cell k."""
        return self.cells[k]


def render(maze: Maze, marks: dict[int, str] | None = None) -> str:
    """ASCII picture of the maze. `marks` maps open-cell indices to a one-character label (player, ghosts)."""
    marks = marks or {}
    n = maze.n
    rows = []
    for r in range(n):
        line = []
        for c in range(n):
            cell = r * n + c
            if maze.walls[cell]:
                line.append("#")
            else:
                line.append(marks.get(maze.k_of[cell], "."))
        rows.append("".join(line))
    return "\n".join(rows)
