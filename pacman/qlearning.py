"""Approximate Q-learning for Pac-Man: learn the feature weights from self-play.

Q(s, a) is linear in the features, so learning means adjusting one weight per feature.
After each turn the agent compares what it expected with what it observed, and nudges
the weights of the features that produced that decision:

    target = r + gamma * max_a' Q(s', a')      (just r when the game ended)
    delta  = target - Q(s, a)                  the temporal-difference error
    w_i   += alpha * delta * f_i(s, a)         each weight moves in proportion to its feature

The max is over the legal actions in the next position, so the update learns the value
of the best play from there (Watkins' Q-learning, off-policy), even though the agent
explores with epsilon-greedy moves. A weight only moves on turns where its feature is
non-zero: rare features such as the power pellet learn slowly, while the bias, which is
always on, moves on every turn.

Training plays games on the given mazes with ghosts that follow their normal AI. Epsilon
falls linearly from `epsilon[0]` to `epsilon[1]` over the first 80 percent of episodes, so
the late games mostly exploit what was learned.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .agents import greedy_action
from .engine import MAX_TURNS, Game, learning_reward, legal_actions
from .features import FEATURE_NAMES, features
from .mazes import get

DEFAULT_MAZES = ("lanes", "vault")


def dot(w: Sequence[float], f: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(w, f, strict=True))


def td_update(
    w: list[float],
    feats: Sequence[float],
    reward: float,
    next_feats: Sequence[Sequence[float]],
    alpha: float,
    gamma: float,
) -> float:
    """One Q-learning update of `w` for the features `feats` of the action just taken.

    `next_feats` holds the feature vector of every legal action in the next position,
    and is empty when the game ended. Returns the TD error.
    """
    target = reward
    if next_feats:
        # Computed before any weight changes, so the bootstrapped value uses the old weights.
        target += gamma * max(dot(w, f) for f in next_feats)
    error = target - dot(w, feats)
    for i, f in enumerate(feats):
        w[i] += alpha * error * f
    return error


@dataclass
class TrainResult:
    weights: list[float]
    history: list[dict] = field(default_factory=list)


def train(
    episodes: int,
    seed: int = 0,
    mazes: Sequence[str] = DEFAULT_MAZES,
    alpha: tuple[float, float] = (0.01, 0.001),
    gamma: float = 0.9,
    epsilon: tuple[float, float] = (0.3, 0.02),
    max_turns: int = MAX_TURNS,
    log_every: int = 25,
    on_log: Callable[[dict], None] | None = None,
) -> TrainResult:
    """Learn weights from `episodes` self-play games, cycling through `mazes`.

    `alpha` and `epsilon` are (start, end) pairs, interpolated linearly over the run. A
    step size that falls over time lets the weights settle, where a constant one keeps
    them oscillating. The trainer's own RNG chooses each game's seed, so a run with the
    same arguments always produces the same weights. `on_log` receives a summary every
    `log_every` episodes.
    """
    rng = random.Random(seed)
    weights = [0.0] * len(FEATURE_NAMES)
    result = TrainResult(weights=weights)
    recent: list[tuple[int, bool]] = []
    for ep in range(episodes):
        progress = min(1.0, ep / max(1, 0.8 * episodes))
        eps = epsilon[0] + (epsilon[1] - epsilon[0]) * progress
        step_size = alpha[0] + (alpha[1] - alpha[0]) * (ep / max(1, episodes - 1))
        maze = get(mazes[ep % len(mazes)])
        game = Game(maze, seed=rng.randrange(2**31), max_turns=max_turns)
        feats_now = {a: features(game.state, a) for a in legal_actions(game.state)}
        while not game.state.over:
            values = {a: dot(weights, f) for a, f in feats_now.items()}
            if rng.random() < eps:
                action = rng.choice(list(feats_now))
            else:
                action = greedy_action(values, rng)
            turn = game.step(action)
            reward = learning_reward(turn)
            if game.state.over:
                next_feats: dict[int, tuple[float, ...]] = {}
            else:
                next_feats = {a: features(game.state, a) for a in legal_actions(game.state)}
            update_target = list(next_feats.values())
            td_update(weights, feats_now[action], reward, update_target, step_size, gamma)
            feats_now = next_feats  # features do not depend on weights, so they carry over
        recent.append((game.state.points, game.state.status == "won"))
        if (ep + 1) % log_every == 0 or ep + 1 == episodes:
            record = {
                "episodes": ep + 1,
                "epsilon": round(eps, 4),
                "mean_score": round(sum(p for p, _ in recent) / len(recent), 2),
                "win_rate": round(sum(w for _, w in recent) / len(recent), 4),
                "weights": [round(w, 4) for w in weights],
            }
            result.history.append(record)
            recent = []
            if on_log:
                on_log(record)
    return result
