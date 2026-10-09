"""Autopilot: a simple policy that trusts the belief, walks to the most likely cell, and busts when sure.

Each turn it looks at every live ghost's belief about its cell and takes the single most probable cell over all
ghosts (the most confident peak):
- if that peak holds more than BUST_THRESHOLD of the ghost's probability and it is the player's cell or next to
  it, bust it (the trap fires at the peak);
- otherwise step along a shortest corridor path toward the peak;
- if the player already stands on the peak but is not sure enough, walk toward the second-best cell instead.
  Sonar from a different corridor triangulates the ghost, so this is how the belief gets sharper.

The policy only reads beliefs, so it works with either filter: the exact forward algorithm or the particle
approximation. Comparing their bust times shows how much the approximation costs in play.
"""

from __future__ import annotations

from .maze import DIR_CHARS

BUST_THRESHOLD = 0.5  # the peak must hold more than this fraction of the ghost's probability to be busted


def _argmax(values: list[float], skip: int = -1) -> int:
    """Index of the largest value, lowest index on ties. `skip` is ignored (used to look for the second peak)."""
    best, best_i = -1.0, -1
    for i, v in enumerate(values):
        if i != skip and v > best:
            best, best_i = v, i
    return best_i


def first_step(maze, p: int, target: int) -> int:
    """Direction of the first step of a shortest corridor path from open cell p to open cell target (-1 if none).

    Neighbours are tried in N, E, S, W order, so ties always break the same way.
    """
    here = maze.dist[p][target]
    for d in range(4):
        q = maze.step[p][d]
        if q >= 0 and maze.dist[q][target] == here - 1:
            return d
    return -1


def choose(game, threshold: float = BUST_THRESHOLD) -> str:
    """The action to play this turn, as one of the strings in game.ACTIONS."""
    maze, p = game.maze, game.p
    best = None  # (confidence, ghost id, cell, marginal)
    for gh in game.ghosts:
        if not gh.live:
            continue
        marg = game.marginal(gh.id)
        k = _argmax(marg)
        if best is None or marg[k] > best[0]:
            best = (marg[k], gh.id, k, marg)
    if best is None:
        return "."
    conf, _, k, marg = best
    if conf > threshold:
        if k == p:
            return "b."
        d = first_step(maze, p, k)
        if d >= 0 and maze.step[p][d] == k:  # the peak is a neighbour: fire at it
            return "b" + DIR_CHARS[d]
    if k == p:  # already standing on the peak: go and look from the second-best cell
        k = _argmax(marg, skip=p)
    d = first_step(maze, p, k)
    if d < 0:  # the target is unreachable (cannot happen in a connected maze): wait
        return "."
    return DIR_CHARS[d]
