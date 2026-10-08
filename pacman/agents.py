"""Pac-Man agents: two baselines and the approximate Q-learning policy.

Every agent maps a State to a Decision. The Decision carries the chosen action and, for
the Q-agent, the Q-value and feature vector of every legal action, so the web page can
show why it chose what it did. Agents that need randomness take a `random.Random`; the
episode runner gives them their own generator, separate from the ghosts'.

  RandomAgent  picks uniformly among legal moves. The floor any learned policy must beat.
  ReflexAgent  a hand-written greedy rule: head for the nearest pellet, and keep out of
               the reach of active ghosts.
  QAgent       picks the action with the highest learned Q-value, w . f(s, a).
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .engine import State, legal_actions
from .features import FEATURE_NAMES, Successor, features_of, successor

WEIGHTS_PATH = Path(__file__).parent / "data" / "weights.json"


@dataclass(frozen=True)
class Decision:
    """An agent's choice for one turn. `values` and `features` are keyed by action index (Q-agent only)."""

    action: int
    values: dict[int, float] | None = None
    features: dict[int, tuple[float, ...]] | None = None


def greedy_action(values: dict[int, float], rng: random.Random) -> int:
    """Highest-value action; exact ties are broken at random so the agent does not always pick the same one."""
    best = max(values.values())
    return rng.choice([a for a, v in values.items() if v == best])


class RandomAgent:
    name = "random"

    def act(self, state: State, rng: random.Random) -> Decision:
        return Decision(rng.choice(legal_actions(state)))


def reflex_score(s: Successor) -> float:
    """Greedy score of a move: nearer pellets are better, active ghosts are to be avoided.

    A move that lands on an active ghost is fatal and scores -1000. A move one step from
    an active ghost loses 80 points, two steps loses 40. Eating a scared ghost is worth
    +50. Otherwise the score is minus the distance to the nearest pellet.
    """
    if any(d == 0 for d in s.active):
        return -1000.0
    score = 0.0
    score -= sum((3 - d) * 40 for d in s.active if d in (1, 2))
    if s.eats_ghost:
        score += 50
    score -= s.food or 0
    return score


class ReflexAgent:
    name = "reflex"

    def act(self, state: State, rng: random.Random) -> Decision:
        scores = {a: reflex_score(successor(state, a)) for a in legal_actions(state)}
        return Decision(greedy_action(scores, rng))


class QAgent:
    """Greedy policy over a linear Q-function. `weights` may be shared with a trainer that updates it in place."""

    name = "q"

    def __init__(self, weights: Sequence[float]):
        if len(weights) != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} weights, got {len(weights)}")
        self.weights = weights

    def evaluate(self, state: State) -> tuple[dict[int, float], dict[int, tuple[float, ...]]]:
        """Q-value and feature vector for every legal action in `state`."""
        feats = {a: features_of(successor(state, a)) for a in legal_actions(state)}
        values = {a: sum(w * f for w, f in zip(self.weights, fv, strict=True)) for a, fv in feats.items()}
        return values, feats

    def act(self, state: State, rng: random.Random, epsilon: float = 0.0) -> Decision:
        """Greedy action, or with probability `epsilon` a uniformly random one (used while training)."""
        values, feats = self.evaluate(state)
        if epsilon and rng.random() < epsilon:
            action = rng.choice(list(values))
        else:
            action = greedy_action(values, rng)
        return Decision(action, values, feats)


@dataclass(frozen=True)
class QWeights:
    weights: tuple[float, ...]
    trained: dict


def save_weights(path: Path, weights: Sequence[float], trained: dict) -> None:
    """Write weights and their training metadata as JSON, with the feature names next to the numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": list(FEATURE_NAMES),
        "weights": [round(float(w), 6) for w in weights],
        "trained": trained,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_weights(path: Path = WEIGHTS_PATH) -> QWeights:
    payload = json.loads(path.read_text())
    if tuple(payload["features"]) != FEATURE_NAMES:
        raise ValueError("weights were trained with a different feature set; retrain with python -m pacman train")
    return QWeights(tuple(payload["weights"]), payload.get("trained", {}))


@lru_cache(maxsize=1)
def load_weights() -> QWeights:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"{WEIGHTS_PATH} missing; train it with: python -m pacman train")
    return read_weights(WEIGHTS_PATH)


AGENTS = ("q", "reflex", "random")
DESCRIPTIONS = {
    "q": (
        "Approximate Q-learning: a linear model scores each move from hand-built features (pellet "
        "distance, ghost distance, power pellets). Its weights were learned from self-play."
    ),
    "reflex": "Greedy rule: go for the nearest pellet, but step away from any ghost within two cells.",
    "random": "Picks a legal move at random. The floor that every other agent should beat.",
}


def make_agent(name: str):
    """Build the named agent. The Q-agent loads the committed weights."""
    if name == "random":
        return RandomAgent()
    if name == "reflex":
        return ReflexAgent()
    if name == "q":
        return QAgent(load_weights().weights)
    raise KeyError(f"unknown agent {name!r}; choose from {', '.join(AGENTS)}")
