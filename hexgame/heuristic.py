"""Two baseline players for Hex: uniform random, and a one-ply shortest-path heuristic.

The shortest-path heuristic measures each side's "distance to connection": the fewest empty cells
a chain still needs, where its own stones cost 0, empty cells cost 1, and opposing stones are walls.
Hex is decided by this race, so the heuristic plays the move that most lengthens the opponent's
distance and most shortens its own. The distance is a 0-1 BFS (a deque shortest-path search).
It is not a full two-distance or resistance model, so it is a fair but weak benchmark opponent.
"""

from __future__ import annotations

import random
from collections import deque
from functools import cache

from .board import EMPTY, _edges, neighbours, other

INF = 10**6


@cache
def _goal_cells(n: int, player: int) -> tuple[int, ...]:
    _, is_goal = _edges(n, player)
    return tuple(i for i in range(n * n) if is_goal[i])


def distance(n: int, cells, player: int) -> int:
    """Fewest empty cells `player` must still fill to connect. 0 means already connected; INF means walled off."""
    starts, _ = _edges(n, player)
    nb = neighbours(n)
    opp = other(player)
    dist = [INF] * (n * n)
    dq: deque[int] = deque()
    for s in starts:
        if cells[s] == opp:
            continue
        w = 0 if cells[s] == player else 1
        dist[s] = w
        if w == 0:
            dq.appendleft(s)
        else:
            dq.append(s)
    while dq:
        u = dq.popleft()
        d = dist[u]
        for v in nb[u]:
            if cells[v] == opp:
                continue
            w = 0 if cells[v] == player else 1
            if d + w < dist[v]:
                dist[v] = d + w
                # Zero-cost steps go to the front and cost-one steps to the back, which keeps the deque sorted.
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
    return min(dist[g] for g in _goal_cells(n, player))


def shortest_path_move(n: int, cells, to_move: int, rng: random.Random) -> int:
    """Play the empty cell that maximises (opponent distance) - (own distance) after the move.

    Ties are broken at random. This is one ply deep, so it does not see that a block may be answered.
    """
    opp = other(to_move)
    best, best_score = [], None
    for m in range(n * n):
        if cells[m] != EMPTY:
            continue
        cells[m] = to_move
        score = distance(n, cells, opp) - distance(n, cells, to_move)
        cells[m] = EMPTY
        if best_score is None or score > best_score:
            best, best_score = [m], score
        elif score == best_score:
            best.append(m)
    return rng.choice(best)


def random_move(n: int, cells, to_move: int, rng: random.Random) -> int:
    """A uniformly random empty cell."""
    return rng.choice([i for i, v in enumerate(cells) if v == EMPTY])
