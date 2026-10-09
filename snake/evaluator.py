"""The evaluation-function agent: score each safe move by a weighted sum of features of the position it leads to.

No network here. For every move that does not end the game, the snake imagines taking it, reads eight
features of the resulting position, and takes the move with the largest weighted sum. The features are
hand-built (what a snake player would look at: food, room to move, an escape route to the tail); the
eight weights are the only thing that is learned, by the cross-entropy method in snake/cem.py.

The features, all in [0, 1], for the position after the move:

  0 apple            1 if the move eats
  1 food near        1 if it eats; else 1 - d / (2 * size) for the BFS distance d to the food (0 if no route)
  2 food reachable   1 if it eats or the food can be reached through empty cells
  3 area             empty cells reachable from the head, over all empty cells
  4 room for body    the reachable area over the snake's length, capped at 1: can the pocket hold the body?
  5 tail reachable   1 if the head can reach the tail through empty cells (the tail moves away, so it is a goal)
  6 exits            empty neighbours of the head, over 4
  7 length           the snake's length over the number of cells

A move that ends the game is never chosen. If every move ends the game the snake goes straight.
The evaluator in web/js/snake-core.js mirrors this module; tests/test_snake_eval_parity.py checks them.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from .board import DIRS, LEFT, RIGHT, STRAIGHT, Game

FEATURE_NAMES = (
    "apple", "food near", "food reachable", "area", "room for body", "tail reachable", "exits", "length",
)
N_FEATURES = len(FEATURE_NAMES)
# Candidate order matters only for ties: straight first, as in the greedy baseline.
ACTION_ORDER = (STRAIGHT, LEFT, RIGHT)

WEIGHTS_PATH = Path(__file__).resolve().parent / "data" / "evaluator.json"


def move_features(game: Game, action: int) -> list[float]:
    """The eight features of the position after `action`. Does not change the game.

    The snake's body after the move is `body2`: the new head, then the old body, minus the tail unless
    the snake ate (eating keeps the tail in place). The tail of `body2` is the goal of the escape check.
    """
    size = game.size
    head = game.next_cell(action)
    eats = head == game.food
    body2 = [head] + (game.body if eats else game.body[:-1])
    cells = size * size
    occupied = set(body2)
    tail = body2[-1]
    blocked = set(body2[:-1])  # the tail may be entered as a goal, but not passed through
    n_empty = cells - len(body2)

    # One breadth-first search from the head over empty cells gives the food distance, the reachable
    # area, and whether the tail can be reached.
    dist = {head: 0}
    queue = deque([head])
    tail_ok = False
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in DIRS:
            nxt = (cx + dx, cy + dy)
            if nxt in dist or not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if nxt == tail:
                tail_ok = True
                continue
            if nxt in blocked:
                continue
            dist[nxt] = dist[(cx, cy)] + 1
            queue.append(nxt)
    reach = len(dist) - 1  # the head itself is not an empty cell

    if eats:
        food_near, food_reach = 1.0, 1.0
    elif game.food is not None and game.food in dist:
        food_near = max(0.0, 1.0 - dist[game.food] / (2 * size))
        food_reach = 1.0
    else:
        food_near, food_reach = 0.0, 0.0

    hx, hy = head
    exits = sum(
        1 for dx, dy in DIRS
        if 0 <= hx + dx < size and 0 <= hy + dy < size and (hx + dx, hy + dy) not in occupied
    )
    length = len(body2)
    return [
        1.0 if eats else 0.0,
        food_near,
        food_reach,
        reach / n_empty if n_empty > 0 else 0.0,
        min(1.0, reach / length),
        1.0 if tail_ok else 0.0,
        exits / 4.0,
        length / cells,
    ]


def move_table(game: Game, weights) -> list[dict]:
    """Every move with its features, score, and whether it ends the game. For the page's explanation."""
    w = np.asarray(weights, dtype=np.float64)
    rows = []
    for action in ACTION_ORDER:
        dies = game.would_die(action)
        feats = move_features(game, action)
        score = float(w @ np.asarray(feats)) if not dies else None
        rows.append({"action": action, "features": feats, "score": score, "dies": dies})
    return rows


def choose(game: Game, weights) -> int:
    """The safe move with the largest weighted feature sum. Ties keep the earlier move in ACTION_ORDER."""
    w = np.asarray(weights, dtype=np.float64)
    best_action, best_score = None, -np.inf
    for action in ACTION_ORDER:
        if game.would_die(action):
            continue
        score = float(w @ np.asarray(move_features(game, action)))
        if score > best_score:
            best_action, best_score = action, score
    return STRAIGHT if best_action is None else best_action


def evaluator(weights):
    """The policy for a weight vector: `choose` with the weights fixed."""
    w = np.asarray(weights, dtype=np.float64)

    def policy(game: Game) -> int:
        return choose(game, w)

    return policy


def load_weights(path: Path = WEIGHTS_PATH) -> tuple[np.ndarray, dict]:
    """The committed weight vector and its metadata (training settings, seeds, fitness, log)."""
    data = json.loads(Path(path).read_text())
    w = np.asarray(data["weights"], dtype=np.float64)
    if w.shape != (N_FEATURES,):
        raise ValueError(f"expected {N_FEATURES} weights, got {w.shape}")
    meta = {k: v for k, v in data.items() if k != "weights"}
    return w, meta
