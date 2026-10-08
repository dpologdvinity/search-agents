"""The four agents: random, greedy, a BFS path planner with a tail-chasing safety check, and the evolved net.

Every agent is a function `policy(game) -> action`, so the benchmark, the watch command, and the
evolver all drive them the same way. The planner is the strong classic baseline; the other two
show what a snake gets from doing nothing clever (random), and from always heading for the food
(greedy).
"""

from __future__ import annotations

import random
from collections import deque

from .board import DIRS, LEFT, RIGHT, STRAIGHT, Game, action_toward
from .net import Net, load_champion

Cell = tuple[int, int]


def random_policy(seed: int = 0):
    """Uniform over left, straight, right. Its own RNG, so a game is repeatable from its seed."""
    rng = random.Random(seed)

    def policy(game: Game) -> int:
        return rng.randrange(3)

    return policy


def greedy(game: Game) -> int:
    """Take the safe move whose head ends closest (Manhattan) to the food. Ties prefer going straight."""
    if game.food is None:
        return STRAIGHT
    best_action, best_dist = STRAIGHT, None
    for action in (STRAIGHT, LEFT, RIGHT):  # straight first, so ties keep the snake going straight
        if game.would_die(action):
            continue
        x, y = game.next_cell(action)
        dist = abs(x - game.food[0]) + abs(y - game.food[1])
        if best_dist is None or dist < best_dist:
            best_action, best_dist = action, dist
    return best_action


def _bfs(game: Game, start: Cell, goal: Cell, blocked: set[Cell]) -> list[Cell] | None:
    """Shortest path from `start` to `goal` over empty cells, as the cells after `start`. None if unreachable.

    `blocked` is the set of cells that cannot be entered. The goal may be entered even if it is
    not otherwise free, which is how the tail becomes a valid target.
    """
    if start == goal:
        return []
    parent: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for dx, dy in DIRS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in parent or not game.in_bounds(nxt):
                continue
            if nxt != goal and nxt in blocked:
                continue
            parent[nxt] = cell
            if nxt == goal:
                path = [nxt]
                while parent[path[-1]] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append(nxt)
    return None


def _reachable(game: Game, start: Cell, blocked: set[Cell]) -> int:
    """How many empty cells can be reached from `start`. Used to pick the roomiest move when boxed in."""
    seen = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for dx, dy in DIRS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in seen or not game.in_bounds(nxt) or nxt in blocked:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return len(seen)


def planner_path(game: Game) -> list[Cell] | None:
    """The planner's chosen route: a path to the food if it is safe to take, else a path to the tail.

    Safety rule (tail-chasing): once the snake has eaten along the food path, it must still be able to
    reach its own tail. If it can, the food path is taken. Otherwise the planner follows its tail, which
    always moves away, until the food path opens up. Returns None when no route exists at all.
    """
    body = game.body
    head, tail = body[0], body[-1]
    # Everything except the tail is an obstacle: the tail will have moved by the time the snake gets there.
    blocked = set(body[:-1])
    if game.food is not None:
        path = _bfs(game, head, game.food, blocked)
        if path is not None:
            # After eating, the path cells and the old tail are all body; the tail stays in place,
            # so the snake needs a route from the food to that tail through the new body.
            new_blocked = set(body) | set(path)
            new_blocked.discard(tail)
            if _bfs(game, game.food, tail, new_blocked) is not None:
                return path
    return _bfs(game, head, tail, blocked)


def planner(game: Game) -> int:
    """The BFS planner's action. If no route exists, pick the safe move with the most room, else go straight."""
    path = planner_path(game)
    if path:
        action = action_toward(game.heading, _direction(game.body[0], path[0]))
        if action is not None and not game.would_die(action):
            return action
    # Boxed in: choose the safe move that leaves the largest empty region to live in.
    best_action, best_room = STRAIGHT, -1
    for action in (STRAIGHT, LEFT, RIGHT):
        if game.would_die(action):
            continue
        cell = game.next_cell(action)
        room = _reachable(game, cell, set(game.body[:-1]))
        if room > best_room:
            best_action, best_room = action, room
    return best_action


def _direction(a: Cell, b: Cell) -> int:
    """Index in DIRS of the unit step from a to b (b must be a neighbour of a)."""
    return DIRS.index((b[0] - a[0], b[1] - a[1]))


def evolved(net: Net):
    """The policy for a trained network: the action is the argmax of its three outputs."""

    def policy(game: Game) -> int:
        return net.act(game)

    return policy


def champion_policy():
    """The committed champion, loaded from snake/data/champion.json."""
    net, _ = load_champion()
    return evolved(net)


# name -> factory(seed) -> policy. The factories take a seed so random can be repeated per game.
AGENT_NAMES = ("random", "greedy", "planner", "evolved")
DESCRIPTIONS = {
    "random": "uniform over left, straight, right; no sensing at all",
    "greedy": "the safe move that ends closest to the food",
    "planner": "BFS path to the food, with a tail-chasing safety check",
    "evolved": "the neural net evolved by the genetic algorithm (the committed champion)",
}


def make_policy(name: str, seed: int = 0, net: Net | None = None):
    """A fresh policy for one game. `net` is used only by "evolved"; the champion is loaded if it is None."""
    if name == "random":
        return random_policy(seed)
    if name == "greedy":
        return greedy
    if name == "planner":
        return planner
    if name == "evolved":
        return evolved(net if net is not None else load_champion()[0])
    raise KeyError(f"unknown agent {name!r}; choose from {', '.join(AGENT_NAMES)}")

