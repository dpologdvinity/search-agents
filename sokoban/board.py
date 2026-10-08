"""Sokoban rules: parsing levels, the padded grid, walking, pushing, and rendering.

A level is a rectangle of walls, floor, goals, boxes and one player. The grid is padded with a
ring of walls, so every cell has four neighbours and the search never needs bounds checks. A cell
is an index i = row * width + column into a flat list, and the four directions are offsets:
up = -width, down = +width, left = -1, right = +1.

Move letters: w, s, a, d walk up, down, left, right. The uppercase letter pushes: the player steps
into a box and the box slides one cell further. A push needs the box's far side to be floor with
no other box on it.

Level symbols (the common text format): '#' wall, ' ' floor, '.' goal, '$' box, '*' box on a goal,
'@' player, '+' player on a goal.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace

WALL, FLOOR, GOAL, BOX, BOX_ON_GOAL, PLAYER, PLAYER_ON_GOAL = "#", " ", ".", "$", "*", "@", "+"
LETTERS = "wsad"  # up, down, left, right; the uppercase letter is a push in the same direction
SYMBOLS = " .$*@+"


@dataclass(frozen=True)
class Level:
    """A puzzle in padded flat-index form. Frozen and hashable, so the precomputed tables can be cached.

    `floor[i]` is 1 where a box or the player may stand. `goals` and `start_boxes` are sorted tuples.
    `text` is the level as typed (without the padding), which the CLI and page show.
    """

    name: str
    width: int
    height: int
    floor: bytes
    goals: tuple[int, ...]
    start_player: int
    start_boxes: tuple[int, ...]
    text: str

    @property
    def offsets(self) -> tuple[int, int, int, int]:
        """Index offsets for the four move letters, in the order of LETTERS."""
        return (-self.width, self.width, -1, 1)

    def rc(self, i: int) -> tuple[int, int]:
        """(row, column) of a flat index, in padded coordinates."""
        return divmod(i, self.width)

    def at(self, row: int, col: int) -> int:
        """Flat index of a padded (row, column)."""
        return row * self.width + col

    def with_state(self, player: int, boxes: Iterable[int]) -> Level:
        """The same level with a different starting position, used to plan from mid-game."""
        return replace(self, start_player=player, start_boxes=tuple(sorted(boxes)))


def parse_level(text: str, name: str = "") -> Level:
    """Read a level from its text. Raises ValueError with a readable message for a broken level.

    Trailing spaces on each line are dropped (they can only be floor outside the walls), and the
    padding ring is added so that floor outside the text is wall.
    """
    lines = [line.rstrip(" \r") for line in text.strip("\n").split("\n")]
    if not any(lines):
        raise ValueError("the level is empty")
    cols = max(len(line) for line in lines)
    width, height = cols + 2, len(lines) + 2
    floor = bytearray(width * height)
    goals, boxes, players = [], [], []
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == WALL:
                continue
            if ch not in SYMBOLS:
                raise ValueError(f"unexpected character {ch!r} in row {r + 1}: use # $ . * @ + and space")
            i = (r + 1) * width + (c + 1)
            floor[i] = 1
            if ch in (GOAL, BOX_ON_GOAL, PLAYER_ON_GOAL):
                goals.append(i)
            if ch in (BOX, BOX_ON_GOAL):
                boxes.append(i)
            if ch in (PLAYER, PLAYER_ON_GOAL):
                players.append(i)
    if len(players) != 1:
        raise ValueError(f"a level needs exactly one player, found {len(players)}")
    if not boxes:
        raise ValueError("a level needs at least one box")
    if len(boxes) != len(goals):
        raise ValueError(f"a level needs one goal per box: {len(boxes)} boxes, {len(goals)} goals")
    return Level(
        name=name,
        width=width,
        height=height,
        floor=bytes(floor),
        goals=tuple(sorted(goals)),
        start_player=players[0],
        start_boxes=tuple(sorted(boxes)),
        text="\n".join(lines),
    )


def is_solved(level: Level, boxes: Iterable[int]) -> bool:
    """True when every box sits on a goal (the goal count equals the box count)."""
    return tuple(sorted(boxes)) == level.goals


def reachable(level: Level, start: int, boxes: set[int] | frozenset[int]) -> list[int]:
    """Every floor cell the player can walk to from `start` without crossing a box, in BFS order."""
    floor = level.floor
    offs = level.offsets
    seen = bytearray(len(floor))
    seen[start] = 1
    order = [start]
    head = 0
    while head < len(order):
        cell = order[head]
        head += 1
        for off in offs:
            nxt = cell + off
            if floor[nxt] and not seen[nxt] and nxt not in boxes:
                seen[nxt] = 1
                order.append(nxt)
    return order


def canonical_player(level: Level, start: int, boxes: set[int] | frozenset[int]) -> int:
    """The smallest cell the player can reach. Two player positions in one region give the same state."""
    return min(reachable(level, start, boxes))


def walk_path(level: Level, start: int, goal: int, boxes: set[int] | frozenset[int]) -> str | None:
    """Walk letters (lowercase) from `start` to `goal` around the boxes, or None if unreachable."""
    if start == goal:
        return ""
    offs = level.offsets
    floor = level.floor
    parent = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for k, off in enumerate(offs):
            nxt = cell + off
            if not floor[nxt] or nxt in boxes or nxt in parent:
                continue
            parent[nxt] = (cell, k)
            if nxt == goal:
                letters = []
                node = goal
                while parent[node] is not None:
                    prev, kk = parent[node]
                    letters.append(LETTERS[kk])
                    node = prev
                return "".join(reversed(letters))
            queue.append(nxt)
    return None


def apply_move(level: Level, player: int, boxes: set[int], letter: str) -> tuple[int, set[int], bool] | None:
    """Apply one move letter (walk or push) to a position.

    Returns (player, boxes, pushed) for a legal move, or None if the move is blocked. Boxes is a new
    set; the caller's set is not changed.
    """
    k = LETTERS.index(letter.lower())
    off = level.offsets[k]
    target = player + off
    if not level.floor[target]:
        return None
    if target not in boxes:
        return target, boxes, False
    beyond = target + off
    if not level.floor[beyond] or beyond in boxes:
        return None
    new_boxes = (boxes - {target}) | {beyond}
    return target, new_boxes, True


def render(level: Level, player: int, boxes: Iterable[int]) -> list[str]:
    """The interior of the grid as text rows, using the same symbols as the level format."""
    occ = set(boxes)
    goals = set(level.goals)
    # Each row is drawn only as far as its own text: the padding beyond a short row is not part of the level.
    text_rows = level.text.split("\n")
    lines = []
    for r in range(1, level.height - 1):
        row = []
        row_len = len(text_rows[r - 1]) if r - 1 < len(text_rows) else 0
        for c in range(1, 1 + row_len):
            i = r * level.width + c
            if not level.floor[i]:
                row.append(WALL)
            elif i == player:
                row.append(PLAYER_ON_GOAL if i in goals else PLAYER)
            elif i in occ:
                row.append(BOX_ON_GOAL if i in goals else BOX)
            else:
                row.append(GOAL if i in goals else FLOOR)
        lines.append("".join(row).rstrip())
    return lines


def push_letters(moves: str) -> int:
    """Number of pushes in a move string (the uppercase letters)."""
    return sum(1 for ch in moves if ch.isupper())
