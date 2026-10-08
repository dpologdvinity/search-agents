"""The Snake rules: a square grid, a snake that turns left, goes straight, or turns right, and food.

The engine is deliberately small and plain Python, so the same rules can be read here, mirrored
in web/js/snake-core.js, and checked by the tests. Coordinates are (x, y) with x = column and
y = row, y growing downward, so "north" is (0, -1).

Rules:
  * The snake starts with 3 segments heading east, in the middle of the board.
  * Each step it turns left, goes straight, or turns right. It cannot reverse, so there is no
    fourth action.
  * Hitting a wall or its own body ends the game. Moving into the cell the tail is leaving is
    allowed, because the tail moves away in the same step; when the snake eats, the tail stays,
    so that cell blocks.
  * Eating food adds one segment and puts new food on a uniformly random empty cell.
  * Going too long without food (2 * cells steps, two laps of the board) ends the game as "starved". Without this cap a
    looping network would play forever.
  * Filling the whole board ends the game as "full", the only way to win.
"""

from __future__ import annotations

import random

BOARD = 12  # side length of the square board used by training, the benchmark, and the page
START_LENGTH = 3

# Headings in clockwise order: 0 north, 1 east, 2 south, 3 west. Turning right adds 1 to the heading.
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
HEADING_NAMES = ("north", "east", "south", "west")

# Actions are relative to the heading, so the policy never needs to know the compass.
LEFT, STRAIGHT, RIGHT = 0, 1, 2
ACTION_NAMES = ("left", "straight", "right")


def turn(heading: int, action: int) -> int:
    """The heading after taking `action`: left subtracts one quarter turn, right adds one."""
    return (heading + {LEFT: -1, STRAIGHT: 0, RIGHT: 1}[action]) % 4


def action_toward(heading: int, direction: int) -> int | None:
    """The relative action that points the snake at absolute `direction`, or None for a reversal."""
    diff = (direction - heading) % 4
    return {0: STRAIGHT, 1: RIGHT, 3: LEFT}.get(diff)


class Game:
    """One game of Snake on a `size` x `size` board, with food placed by a seeded RNG.

    `body[0]` is the head and `body[-1]` the tail. `cause` is None while the game runs, then one of
    "wall", "body", "starved", or "full".
    """

    def __init__(self, seed: int = 0, size: int = BOARD):
        self.size = size
        self.rng = random.Random(seed)
        cx, cy = size // 2, size // 2
        self.body: list[tuple[int, int]] = [(cx - i, cy) for i in range(START_LENGTH)]
        self.heading = 1  # east
        self.apples = 0
        self.steps = 0
        self.idle = 0  # steps since the last apple
        self.idle_cap = 2 * size * size
        self.cause: str | None = None
        self.food: tuple[int, int] | None = self._place_food()

    @classmethod
    def from_state(cls, size, body, heading, food, apples=0, steps=0) -> Game:
        """Rebuild a game from a position, for the web API. The food RNG is not used for that position."""
        game = cls(0, size)
        game.body = [tuple(c) for c in body]
        game.heading = heading
        game.food = tuple(food) if food is not None else None
        game.apples = apples
        game.steps = steps
        return game

    @property
    def alive(self) -> bool:
        return self.cause is None

    @property
    def length(self) -> int:
        return len(self.body)

    def _place_food(self) -> tuple[int, int] | None:
        """Food goes on a random empty cell. Returns None when the snake fills the board."""
        taken = set(self.body)
        free = [(x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in taken]
        return self.rng.choice(free) if free else None

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.size and 0 <= y < self.size

    def next_cell(self, action: int) -> tuple[int, int]:
        """Where the head would go if the snake took `action`. Does not move anything."""
        dx, dy = DIRS[turn(self.heading, action)]
        hx, hy = self.body[0]
        return hx + dx, hy + dy

    def would_die(self, action: int) -> bool:
        """True if `action` ends the game (wall or body). The baselines use this to avoid suicide."""
        cell = self.next_cell(action)
        if not self.in_bounds(cell):
            return True
        return self._blocks(cell, eats=cell == self.food)

    def _blocks(self, cell: tuple[int, int], eats: bool) -> bool:
        # When the snake does not eat, the tail moves away this step, so the last segment is free.
        if eats:
            return cell in self.body
        return cell in self.body and cell != self.body[-1]

    def step(self, action: int) -> bool:
        """Take one action. Returns True while the game goes on. A finished game ignores further steps."""
        if not self.alive:
            return False
        cell = self.next_cell(action)  # uses the heading before the turn
        self.heading = turn(self.heading, action)
        self.steps += 1
        if not self.in_bounds(cell):
            self.cause = "wall"
            return False
        eats = cell == self.food
        if self._blocks(cell, eats):
            self.cause = "body"
            return False
        self.body.insert(0, cell)
        if eats:
            self.apples += 1
            self.idle = 0
            self.food = self._place_food()
            if self.food is None:
                self.cause = "full"
                return False
        else:
            self.body.pop()
            self.idle += 1
            if self.idle >= self.idle_cap:
                self.cause = "starved"
                return False
        return True

    def play(self, policy, max_steps: int | None = None) -> None:
        """Run `policy(game) -> action` until the game ends, or until `max_steps` moves have been made."""
        while self.alive and (max_steps is None or self.steps < max_steps):
            self.step(policy(self))
