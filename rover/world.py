"""The hidden world the rover explores: a square grid of walls, seeded map generation, and the sensor.

Cells are numbered row-major: cell c is row c // n, column c % n. The start is the top-left cell and
the goal the bottom-right one. The rover never reads `walls` directly. It learns a cell's state only
when the sensor covers that cell, so the planner always works from a belief about the map.

The generator uses mulberry32, a 32-bit PRNG whose arithmetic is written so that the JavaScript port
(web/js/rover_core.js) produces the same bits. The web page and the CLI therefore draw the same map
for the same (size, density, seed), and a parity test checks it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

M32 = 0xFFFFFFFF


class Mulberry32:
    """Seeded PRNG returning floats in [0, 1). Each step is 32-bit arithmetic, mirrored exactly in JS.

    The JS version uses Math.imul, which keeps the low 32 bits of a product. Python does the same with
    `& M32`, so the two streams match bit for bit.
    """

    def __init__(self, seed: int):
        self.a = seed & M32

    def next(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & M32
        t = self.a
        # Two rounds of multiply-xorshift mix the bits; "1 | t" and "61 | t" keep the multipliers odd.
        t = ((t ^ (t >> 15)) * (1 | t)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296


@dataclass
class World:
    """The true map: `walls[c] == 1` for a wall. The rover never reads this directly, only through the sensor."""

    n: int
    walls: bytearray
    start: int = 0
    goal: int = -1

    def __post_init__(self):
        if self.goal < 0:
            self.goal = self.n * self.n - 1

    def set_wall(self, cell: int, wall: bool) -> None:
        self.walls[cell] = 1 if wall else 0


def neighbours(n: int, cell: int):
    """The up to four orthogonal neighbours of `cell`, in the fixed order N, E, S, W.

    The fixed order is what makes tie-breaking deterministic, so the JS port expands the same cells.
    """
    r, c = divmod(cell, n)
    if r > 0:
        yield cell - n
    if c < n - 1:
        yield cell + 1
    if r < n - 1:
        yield cell + n
    if c > 0:
        yield cell - 1


def reachable(n: int, walls, start: int, goal: int) -> bool:
    """True if a plain breadth-first search finds a wall-free path from start to goal."""
    seen = bytearray(n * n)
    seen[start] = 1
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if u == goal:
            return True
        for v in neighbours(n, u):
            if not seen[v] and not walls[v]:
                seen[v] = 1
                queue.append(v)
    return False


def generate(n: int, density: float, seed: int) -> bytearray:
    """A seeded random map: each cell is a wall with probability `density`, but the corners stay open.

    A map is redrawn (from the same PRNG stream) until the goal is reachable, so the rover always has a
    possible route. Redrawing is rare at the densities the page offers (up to 30%).
    """
    n = int(n)
    start, goal = 0, n * n - 1
    rng = Mulberry32(seed)
    while True:
        walls = bytearray(n * n)
        for c in range(n * n):
            if c != start and c != goal and rng.next() < density:
                walls[c] = 1
        if reachable(n, walls, start, goal):
            return walls


def sensed_cells(n: int, cell: int, radius: int):
    """Cells inside the sensor's square of half-width `radius` around `cell` (a square, not a disc)."""
    r0, c0 = divmod(cell, n)
    for r in range(max(0, r0 - radius), min(n - 1, r0 + radius) + 1):
        for c in range(max(0, c0 - radius), min(n - 1, c0 + radius) + 1):
            yield r * n + c


def render(n: int, walls, seen, pos: int, goal: int, path=()) -> str:
    """ASCII picture of the map as the rover knows it.

    '#' known wall, '.' known free, ' ' not yet sensed, 'R' rover, 'G' goal, '*' planned path.
    The true map is never drawn, so the picture shows only what the rover has learned.
    """
    on_path = set(path)
    rows = []
    for r in range(n):
        line = []
        for c in range(n):
            i = r * n + c
            if i == pos:
                ch = "R"
            elif i == goal:
                ch = "G"
            elif not seen[i]:
                ch = " "
            elif walls[i]:
                ch = "#"
            elif i in on_path:
                ch = "*"
            else:
                ch = "."
            line.append(ch)
        rows.append("".join(line))
    return "\n".join(rows)
