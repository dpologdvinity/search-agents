"""Hand-built features for approximate Q-learning.

The Q-value of an action is a weighted sum of features of the position it leads to:
    Q(s, a) = w . f(s, a)
f describes the position right after Pac-Man's move and before the ghosts move, which
is the moment he has to decide with. The features are chosen so a linear model can
express the trade-offs that matter: eat pellets and power pellets, stay away from
active ghosts, chase scared ghosts, avoid dead ends, and keep clear of traps that only
show up a few turns later (the survival feature, see lookahead.py). The weights are
learned, so nothing here encodes how strongly each feature should count.

Distances are walking distances from the destination cell (one BFS), so walls and
corridors count the way they do for a ghost.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import State, legal_actions
from .lookahead import SURVIVAL_HORIZON, survival_depths
from .mazes import NO_CELL
from .search import UNREACHABLE, distances

FEATURE_NAMES = (
    "bias",
    "pellet",
    "power",
    "food",
    "capsule",
    "ghost_near",
    "ghost_adjacent",
    "caught",
    "scared_near",
    "eat_ghost",
    "openness",
    "trap",
    "survival",
)
FEATURE_DOCS = {
    "bias": "constant 1; learns the typical value of a turn",
    "pellet": "1 if the move eats a pellet",
    "power": "1 if the move eats a power pellet",
    "food": "walking distance to the nearest pellet / 10 (0 when none remain)",
    "capsule": "walking distance to the nearest power pellet / 10 (0 when none remain)",
    "ghost_near": "sum over active ghosts of max(0, 1 - d / 6), d = walking distance from the move",
    "ghost_adjacent": "1 if an active ghost is one step from the move",
    "caught": "1 if an active ghost stands on the destination (fatal)",
    "scared_near": "sum over scared ghosts of max(0, 1 - d / 8) for d >= 1",
    "eat_ghost": "1 if the move lands on a scared ghost",
    "openness": "open neighbours of the destination / 4 (low values are dead ends)",
    "trap": "ghost_near * (1 - openness): a ghost closing in on a dead end",
    "survival": "turns Pac-Man stays alive after the move if the ghosts move as they really do, / 6",
}
GHOST_RANGE = 6
SCARED_RANGE = 8
DISTANCE_SCALE = 10


@dataclass(frozen=True)
class Successor:
    """Facts about the position after one candidate move, from which features and reflex scores are built."""

    pac: int
    dist: list[int]  # walking distance from `pac` to every cell
    food: int | None  # distance to the nearest pellet, None if none remain
    capsule: int | None  # distance to the nearest power pellet, None if none remain
    active: tuple[int, ...]  # distances to ghosts that can hurt Pac-Man
    scared: tuple[int, ...]  # distances to ghosts that Pac-Man can eat
    eats_pellet: bool
    eats_power: bool
    eats_ghost: bool
    openness: float


def successor(state: State, action: int) -> Successor:
    """The position after Pac-Man takes `action`, before any ghost moves."""
    maze = state.maze
    pac = maze.nbr[state.pac][action]
    dist = distances(maze, pac)
    food = min((dist[p] for p in state.pellets), default=None)
    capsule = min((dist[p] for p in state.pellets if p in maze.powers), default=None)
    active, scared = [], []
    for g in state.ghosts:
        d = dist[g.pos]
        (scared if g.scared else active).append(d)
    open_neighbours = sum(1 for nb in maze.nbr[pac] if nb != NO_CELL)
    return Successor(
        pac=pac,
        dist=dist,
        food=food,
        capsule=capsule,
        active=tuple(active),
        scared=tuple(scared),
        eats_pellet=pac in state.pellets and pac not in maze.powers,
        eats_power=pac in state.pellets and pac in maze.powers,
        eats_ghost=any(d == 0 for d in scared),
        openness=open_neighbours / 4,
    )


def _scaled_distance(d: int | None) -> float:
    """Walking distance divided by DISTANCE_SCALE; 0 when there is no target.

    Linear rather than 1 / (1 + d): an inverse distance flattens out for far targets, so
    the model could not tell a pellet 15 steps away from one 25 steps away and never
    learned to head for distant clusters. A linear distance keeps a constant gradient.
    """
    return 0.0 if d is None or d == UNREACHABLE else d / DISTANCE_SCALE


def features_of(s: Successor, survival: int) -> tuple[float, ...]:
    """The feature vector in FEATURE_NAMES order. `survival` is the move's survival depth (see lookahead.py)."""
    ghost_near = sum(max(0.0, 1 - d / GHOST_RANGE) for d in s.active)
    return (
        1.0,
        float(s.eats_pellet),
        float(s.eats_power),
        _scaled_distance(s.food),
        _scaled_distance(s.capsule),
        ghost_near,
        float(any(d == 1 for d in s.active)),
        float(any(d == 0 for d in s.active)),
        sum(max(0.0, 1 - d / SCARED_RANGE) for d in s.scared if d > 0),
        float(s.eats_ghost),
        s.openness,
        ghost_near * (1 - s.openness),
        survival / SURVIVAL_HORIZON,
    )


def state_features(state: State) -> dict[int, tuple[float, ...]]:
    """f(s, a) for every legal action of `state`. The survival search is shared by all of them."""
    depths = survival_depths(state)
    return {a: features_of(successor(state, a), depths[a]) for a in legal_actions(state)}
