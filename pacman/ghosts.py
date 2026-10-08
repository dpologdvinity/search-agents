"""Ghost AI: each ghost picks a target cell, A* plans a route to it, and it takes one step.

Personalities echo the roles of the classic arcade ghosts:
  chaser    targets Pac-Man's cell. The most direct threat.
  ambusher  targets the cell AMBUSH_LEAD steps ahead of Pac-Man in the direction he last
            moved, so it tries to cut him off instead of trailing him.
  scatter   chases from afar, but within SCATTER_RADIUS it retreats to its home corner,
            so Pac-Man can lure it away and then slip past.

A scared ghost (after a power pellet) ignores its target and steps to the neighbour that
is farthest from Pac-Man by walking distance. Ties are broken with the game's RNG, so a
seeded game replays exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .mazes import NO_CELL, Maze
from .search import astar, manhattan

PERSONALITIES = ("chaser", "ambusher", "scatter", "chaser")  # ghost i gets PERSONALITIES[i % 4]
PERSONALITY_DOCS = {
    "chaser": "Heads straight for Pac-Man.",
    "ambusher": "Aims for the cell a few steps ahead of Pac-Man, to cut him off.",
    "scatter": "Chases from afar, but retreats to its corner when it gets close.",
}
AMBUSH_LEAD = 4
SCATTER_RADIUS = 4


@dataclass(frozen=True)
class Ghost:
    """One ghost. `home` is where it respawns after being eaten; `scared` counts down the turns of fright left."""

    pos: int
    personality: str
    home: int
    scared: int = 0


def personality_for(index: int) -> str:
    return PERSONALITIES[index % len(PERSONALITIES)]


def target_for(maze: Maze, ghost: Ghost, index: int, pac: int, facing: int) -> int:
    """The cell this ghost is trying to reach this turn.

    `pac` is Pac-Man's cell after his move this turn, and `facing` the action index of
    that move (-1 before he has moved). The ambusher walks forward from `pac` in the
    facing direction, stopping early at a wall, so its target is always an open cell.
    """
    if ghost.personality == "chaser":
        return pac
    if ghost.personality == "ambusher":
        cell = pac
        if facing >= 0:
            for _ in range(AMBUSH_LEAD):
                nxt = maze.nbr[cell][facing]
                if nxt == NO_CELL:
                    break
                cell = nxt
        return cell
    # scatter: close to Pac-Man it gives up the chase and goes to its corner
    if manhattan(maze, ghost.pos, pac) <= SCATTER_RADIUS:
        return maze.corners[index % len(maze.corners)]
    return pac


def choose_move(
    maze: Maze,
    ghost: Ghost,
    index: int,
    pac: int,
    facing: int,
    rng: random.Random,
    pac_dist: list[int],
) -> tuple[int, int | None, tuple[int, ...]]:
    """Return (next cell, target or None when fleeing, the route the ghost planned).

    `pac_dist` is the BFS distance map from Pac-Man's cell, shared by all scared ghosts
    in a turn so it is computed once.
    """
    options = [nb for nb in maze.nbr[ghost.pos] if nb != NO_CELL]
    if ghost.scared:
        best = max(pac_dist[nb] for nb in options)
        nxt = rng.choice([nb for nb in options if pac_dist[nb] == best])
        return nxt, None, (ghost.pos, nxt)

    target = target_for(maze, ghost, index, pac, facing)
    route = astar(maze, ghost.pos, target)
    if route is not None and len(route) > 1:
        return route[1], target, tuple(route)
    # Only reached when the target is the ghost's own cell: step toward Pac-Man instead.
    nxt = min(options, key=lambda nb: (manhattan(maze, nb, pac), nb))
    return nxt, target, (ghost.pos, nxt)
