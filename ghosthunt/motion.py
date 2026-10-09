"""Ghost motion models: the transition distributions the hidden Markov model is built from.

A ghost's hidden state is s = 4*k + h: the open cell k it is in and the heading h (0..3 = N, E, S, W) of its
last move. The heading is part of the state so that the patrol model can follow corridors; the other models
simply ignore it. The model is a known stochastic process, so the agent knows every probability below.

random   stays put with probability STAY_P, otherwise moves to a uniformly chosen open neighbour.
lurker   weighs staying and each open neighbour by exp(-BETA * distance to the player), so it drifts toward
         the player. Because the player's cell is known, the probabilities are known too: the model is an HMM
         whose transitions depend on the (observed) player position, and the forward algorithm handles that.
patrol   keeps its heading with probability STRAIGHT_P when the corridor continues, otherwise turns into one of
         the side openings (never back, unless it is at a dead end). A patrol ghost never waits.

transitions() returns the successors of one state with probabilities that sum to 1. Each entry is ordered by
direction, so the arithmetic (and the JavaScript port) visits them in the same order.
"""

from __future__ import annotations

import math

MODELS = ("random", "lurker", "patrol")
STAY_P = 0.1       # random: chance of not moving on a turn
BETA = 1.0         # lurker: strength of the pull toward the player (per step of distance)
STRAIGHT_P = 0.8   # patrol: chance to keep going straight when the corridor continues


def _random(maze, k: int, h: int) -> list[tuple[int, float]]:
    opts = [(d, maze.step[k][d]) for d in range(4) if maze.step[k][d] >= 0]
    if not opts:  # an isolated cell (cannot happen in a perfect maze, but keep the model total)
        return [(4 * k + h, 1.0)]
    out = [(4 * k + h, STAY_P)]
    share = (1.0 - STAY_P) / len(opts)
    for d, k2 in opts:
        out.append((4 * k2 + d, share))
    return out


def _lurker(maze, k: int, h: int, player: int) -> list[tuple[int, float]]:
    # Weight of each option is exp(-BETA * distance to the player). Staying is one option among the rest.
    opts = [(4 * k + h, maze.dist[k][player], None)]
    for d in range(4):
        k2 = maze.step[k][d]
        if k2 >= 0:
            opts.append((4 * k2 + d, maze.dist[k2][player], d))
    weights = [math.exp(-BETA * dist) for _, dist, _ in opts]
    total = sum(weights)
    return [(s2, w / total) for (s2, _, _), w in zip(opts, weights)]


def _patrol(maze, k: int, h: int) -> list[tuple[int, float]]:
    # Corridor rule: keep the heading when the way ahead is open; at a junction, take a side opening; in a
    # dead end, turn around. The reverse direction is only taken when nothing else is open.
    open_dirs = {d for d in range(4) if maze.step[k][d] >= 0}
    back = (h + 2) % 4
    fwd_ok = h in open_dirs
    turns = [d for d in range(4) if d in open_dirs and d != h and d != back]
    back_ok = back in open_dirs
    out: list[tuple[int, float]] = []
    if fwd_ok:
        if turns:
            out.append((4 * maze.step[k][h] + h, STRAIGHT_P))
            for d in turns:
                out.append((4 * maze.step[k][d] + d, (1.0 - STRAIGHT_P) / len(turns)))
        elif back_ok:  # a corridor: straight on, or turn back
            out.append((4 * maze.step[k][h] + h, STRAIGHT_P))
            out.append((4 * maze.step[k][back] + back, 1.0 - STRAIGHT_P))
        else:
            out.append((4 * maze.step[k][h] + h, 1.0))
    elif turns:  # the way ahead is a wall (only possible for an arbitrary starting heading)
        for d in turns:
            out.append((4 * maze.step[k][d] + d, 1.0 / len(turns)))
    elif back_ok:  # dead end: turn around
        out.append((4 * maze.step[k][back] + back, 1.0))
    else:
        out.append((4 * k + h, 1.0))
    return out


def transitions(maze, model: str, s: int, player: int) -> list[tuple[int, float]]:
    """Successor states of s with their probabilities. `player` is the open index of the player's cell."""
    k, h = divmod(s, 4)
    if model == "random":
        return _random(maze, k, h)
    if model == "lurker":
        return _lurker(maze, k, h, player)
    if model == "patrol":
        return _patrol(maze, k, h)
    raise ValueError(f"unknown motion model {model!r}")


def table(maze, model: str, player: int) -> list[list[tuple[int, float]]]:
    """transitions() for every state. The random and patrol tables do not depend on the player, so they are
    computed once per maze; the lurker table is rebuilt whenever the player moves."""
    if model != "lurker":
        key = ("motion", model)
        if key not in maze.cache:
            maze.cache[key] = [transitions(maze, model, s, player) for s in range(4 * maze.K)]
        return maze.cache[key]
    return [transitions(maze, model, s, player) for s in range(4 * maze.K)]
