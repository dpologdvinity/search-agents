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

The chance policy (`chance_move`) replaces all of that with a fixed table of odds: every
turn the ghost keeps going, turns left or right, or reverses, by the odds in CHANCE_ODDS.
It does not look at Pac-Man, the pellets or the routes at all, so it is the opponent to
beat when you want to know how much of a score comes from reading the ghosts.
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


GHOST_POLICIES = ("ai", "chance")  # "ai": A* routes and personalities (the default). "chance": fixed odds.
GHOST_LABELS = {"ai": "AI ghosts (A* routes)", "chance": "Chance (fixed odds)"}
GHOST_DOCS = {
    "ai": (
        "Each ghost picks a target (Pac-Man, a cell ahead of him, or its corner) and plans a shortest "
        "route to it with A*. Scared ghosts flee."
    ),
    "chance": (
        "No planning: each ghost keeps going 60% of the time, turns left 15%, turns right 15%, and "
        "reverses 10%, renormalised over the directions that are open."
    ),
}
NO_HEADING = -1  # a ghost that has not moved yet has no direction to keep going in

# Fixed odds for a chance ghost, as the share of each turn's choice that goes each way, relative
# to the direction the ghost last moved (its heading). Why these numbers: a ghost mostly keeps
# going, turns left and right about equally often, and reverses least, since turning back is
# the least common move of a wandering walker. They were set by hand, not fitted to any data,
# and they sum to 1. Over the open directions only, the shares are renormalised (see chance_move).
CHANCE_ODDS = {"straight": 0.60, "left": 0.15, "right": 0.15, "back": 0.10}
# Name of each turn by its clockwise offset from the heading. ACTIONS is N, E, S, W, so one step
# round that list is a right turn and three steps is a left turn.
_RELATIVE = ("straight", "right", "back", "left")


@dataclass(frozen=True)
class Ghost:
    """One ghost. `home` is where it respawns after being eaten; `scared` counts down the turns of fright left.

    `heading` is the index (into ACTIONS) of the direction the ghost last moved, or NO_HEADING
    before its first move. Only the chance policy reads it; the A* ghosts ignore it.
    """

    pos: int
    personality: str
    home: int
    scared: int = 0
    heading: int = NO_HEADING


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


def chance_move(maze: Maze, ghost: Ghost, rng: random.Random) -> tuple[int, int]:
    """Pick the next cell by the fixed odds, with no search. Returns (next cell, direction index).

    The candidates are the directions that do not run into a wall. Each gets its share of
    CHANCE_ODDS, read relative to the ghost's heading (straight on, a turn, or back), and the
    shares are renormalised over the candidates only, so a ghost in a corridor with one way
    out must take it. A ghost with no heading yet gives every candidate the same weight.
    One draw from `rng` makes the choice, so a seeded game replays exactly.
    """
    nbrs = maze.nbr[ghost.pos]
    open_dirs = [d for d in range(len(nbrs)) if nbrs[d] != NO_CELL]
    if ghost.heading == NO_HEADING:
        weights = [1.0] * len(open_dirs)
    else:
        weights = [CHANCE_ODDS[_RELATIVE[(d - ghost.heading) % len(nbrs)]] for d in open_dirs]
    # Scale one uniform draw to the total weight, then walk the candidates until it is used up.
    roll = rng.random() * sum(weights)
    for d, w in zip(open_dirs, weights, strict=True):
        roll -= w
        if roll < 0:
            return nbrs[d], d
    # Only floating-point rounding can leave `roll` non-negative here; the last candidate takes it.
    return nbrs[open_dirs[-1]], open_dirs[-1]
