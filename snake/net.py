"""The policy network: what the snake senses, and a small NumPy net that turns it into a turn.

The snake senses the board in its own frame (ahead, left, right), so one learned rule works
whichever way it faces. Seventeen inputs:

  0-2   danger ahead, left, right: 1 if the adjacent cell is a wall or a body segment, else 0
  3-5   free run ahead, left, right: empty cells before the first obstacle, over size - 1
  6-7   food: forward and rightward offset from the head, over size - 1, clipped to [-1, 1]
  8-9   tail: the same offsets, so the net can learn to chase its tail when boxed in
  10-13 heading one-hot: north, east, south, west
  14-16 room ahead, left, right: how many empty cells can be reached by walking from the
        adjacent cell in that direction, counted up to FLOOD_CAP and divided by it. This is the
        flood fill that lets the net see a pocket it would trap itself in, which the straight
        run cannot show.

The net is one tanh hidden layer and a linear layer with three outputs (left, straight, right).
The action is the largest output. A champion has 17*16 + 16 + 16*3 + 3 = 339 numbers, small
enough to commit as JSON and run in the browser with the same arithmetic.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from .board import DIRS, Game

INPUT_NAMES = (
    "danger ahead", "danger left", "danger right",
    "free ahead", "free left", "free right",
    "food forward", "food right",
    "tail forward", "tail right",
    "heading N", "heading E", "heading S", "heading W",
    "room ahead", "room left", "room right",
)
OUTPUT_NAMES = ("left", "straight", "right")
N_IN = len(INPUT_NAMES)
N_HIDDEN = 16
N_OUT = len(OUTPUT_NAMES)
N_PARAMS = N_IN * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUT + N_OUT
# The room count stops at this many cells: enough to see a pocket, and it bounds the cost of a flood fill.
FLOOD_CAP = 30

CHAMPION_PATH = Path(__file__).resolve().parent / "data" / "champion.json"


def room(occupied: set, size: int, start: tuple[int, int], cap: int = FLOOD_CAP) -> int:
    """Empty cells reachable from `start` (counting it), stopping at `cap`. Returns 0 if `start` is blocked.

    Breadth-first search over empty cells; the body is the wall. The cap keeps this cheap: a snake only
    needs to know whether a pocket is small, not how big the open board is.
    """
    x, y = start
    if not (0 <= x < size and 0 <= y < size) or start in occupied:
        return 0
    seen = {start}
    queue = deque([start])
    while queue and len(seen) < cap:
        cx, cy = queue.popleft()
        for dx, dy in DIRS:
            nxt = (cx + dx, cy + dy)
            if nxt in seen or nxt in occupied or not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            seen.add(nxt)
            queue.append(nxt)
            if len(seen) >= cap:
                break
    return len(seen)


def features(game: Game) -> np.ndarray:
    """The 17 inputs above for the current position. Works in the snake's frame.

    The three relative directions are unit steps on the grid: `ahead` is the heading, `left` and `right`
    are a quarter turn either side. Cell tests use the whole body, the tail included. That is more cautious
    than the rules (the tail usually moves away), and the web code does the same.
    """
    size = game.size
    occupied = set(game.body)
    hx, hy = game.body[0]
    heading = game.heading
    ax, ay = DIRS[heading]
    rx, ry = DIRS[(heading + 1) % 4]  # right of the heading
    scale = float(size - 1)
    out = [0.0] * N_IN
    for k, (dx, dy) in enumerate(((ax, ay), (-rx, -ry), (rx, ry))):  # ahead, left, right
        # Walk from the head until the first obstacle. A run of 0 means the adjacent cell is blocked,
        # which is exactly the danger input.
        run, cx, cy = 0, hx, hy
        while True:
            cx += dx
            cy += dy
            if not (0 <= cx < size and 0 <= cy < size) or (cx, cy) in occupied:
                break
            run += 1
        out[k] = 1.0 if run == 0 else 0.0
        out[3 + k] = run / scale
        out[14 + k] = room(occupied, size, (hx + dx, hy + dy)) / FLOOD_CAP

    def offset(cell):
        # Offsets are projected on the heading (forward) and on the right-hand unit vector (right).
        ox, oy = cell[0] - hx, cell[1] - hy
        fwd = (ox * ax + oy * ay) / scale
        side = (ox * rx + oy * ry) / scale
        return max(-1.0, min(1.0, fwd)), max(-1.0, min(1.0, side))

    if game.food is not None:
        out[6], out[7] = offset(game.food)
    out[8], out[9] = offset(game.body[-1])
    out[10 + heading] = 1.0
    return np.array(out)


class Net:
    """A one-hidden-layer tanh network with the weights stored as NumPy arrays.

    Shapes: w1 (N_IN, N_HIDDEN), b1 (N_HIDDEN,), w2 (N_HIDDEN, N_OUT), b2 (N_OUT,).
    """

    def __init__(self, w1, b1, w2, b2):
        self.w1 = np.asarray(w1, dtype=np.float64).reshape(N_IN, N_HIDDEN)
        self.b1 = np.asarray(b1, dtype=np.float64).reshape(N_HIDDEN)
        self.w2 = np.asarray(w2, dtype=np.float64).reshape(N_HIDDEN, N_OUT)
        self.b2 = np.asarray(b2, dtype=np.float64).reshape(N_OUT)

    @classmethod
    def from_genome(cls, genome) -> Net:
        """Unpack the flat vector the evolver works on (length N_PARAMS) into the four arrays."""
        g = np.asarray(genome, dtype=np.float64)
        if g.shape != (N_PARAMS,):
            raise ValueError(f"a genome has {N_PARAMS} numbers, got {g.shape}")
        a = N_IN * N_HIDDEN
        b = a + N_HIDDEN
        c = b + N_HIDDEN * N_OUT
        return cls(g[:a], g[a:b], g[b:c], g[c:])

    def genome(self) -> np.ndarray:
        """The inverse of from_genome: the flat vector the evolver mutates."""
        return np.concatenate([self.w1.ravel(), self.b1, self.w2.ravel(), self.b2])

    def forward(self, x) -> tuple[np.ndarray, np.ndarray]:
        """Hidden activations and output logits for one input vector."""
        h = np.tanh(x @ self.w1 + self.b1)
        return h, h @ self.w2 + self.b2

    def act(self, game: Game) -> int:
        """The action with the largest logit. Ties go to the lower index, as in the web code."""
        _, logits = self.forward(features(game))
        return int(np.argmax(logits))

    def to_json(self, **meta) -> dict:
        """Weights rounded to 4 decimals (the page runs the same arithmetic) plus any metadata."""
        def r(a):
            return np.round(a, 4).tolist()
        return {**meta, "layers": [N_IN, N_HIDDEN, N_OUT],
                "w1": r(self.w1), "b1": r(self.b1), "w2": r(self.w2), "b2": r(self.b2)}

    @classmethod
    def from_json(cls, data: dict) -> Net:
        return cls(data["w1"], data["b1"], data["w2"], data["b2"])


def load_champion(path: Path = CHAMPION_PATH) -> tuple[Net, dict]:
    """The committed champion network and its metadata (generation, fitness, seeds)."""
    data = json.loads(Path(path).read_text())
    meta = {k: v for k, v in data.items() if k not in ("w1", "b1", "w2", "b2")}
    return Net.from_json(data), meta
