"""Survival lookahead: how many turns Pac-Man stays alive after a move, if the ghosts move as they really do.

A static feature cannot see a trap that is two or three turns away: a corridor that looks
open from where Pac-Man stands but has a ghost waiting at its far end. Here the game itself
is run forward. For each legal move, the ghosts' real AI (ghosts.choose_move) answers, Pac-Man
tries every move again, and so on, up to SURVIVAL_HORIZON turns. The depth reached is the
feature: 0 means the move is fatal at once, the horizon means Pac-Man can keep out of the
ghosts' reach for that long.

The search is exact for the ghost AI, which is deterministic except for the tie-break
between equally good flight squares of a scared ghost; that tie-break uses a fixed seed
here, so it can differ from the real game's choice. A position that survives to the
horizon counts as fully safe, and a won game counts as surviving too. Positions are memoised
on everything that decides the future (Pac-Man's cell and last move, every ghost, and how
many pellets remain), so the search stays cheap enough to run for every move of every turn.
"""

from __future__ import annotations

from .engine import Game, State, legal_actions

SURVIVAL_HORIZON = 6


def _alive(game: Game, state: State, plies: int, memo: dict) -> int:
    """How many of the next `plies` turns a living position survives, under Pac-Man's best play."""
    if plies == 0 or state.over:
        return plies
    key = (state.pac, state.facing, state.ghosts, len(state.pellets), plies)
    if key in memo:
        return memo[key]
    best = 0
    for action in legal_actions(state):
        game.state = state
        turn = game.step(action)
        if "death" in turn.events:
            continue
        best = max(best, 1 + _alive(game, game.state, plies - 1, memo))
        if best == plies:  # cannot do better than surviving the whole horizon
            break
    memo[key] = best
    return best


def survival_depths(state: State, horizon: int = SURVIVAL_HORIZON) -> dict[int, int]:
    """For each legal move: the turns (1 to `horizon`) Pac-Man survives if he takes it, 0 if it is fatal.

    The move itself counts as the first turn, so a move that survives is at least 1.
    `state` must be in play; the moves are the keys of the result.
    """
    game = Game(state.maze, seed=0)
    memo: dict = {}
    depths = {}
    for action in legal_actions(state):
        game.state = state
        turn = game.step(action)
        if "death" in turn.events:
            depths[action] = 0
        else:
            depths[action] = 1 + _alive(game, game.state, horizon - 1, memo)
    return depths
