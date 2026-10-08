"""The agents that play CartPole. Every agent has act(obs) -> 0 (left) or 1 (right).

random     uniform coin flips. The floor every other agent should beat.
pd         a hand-tuned linear controller: push toward where the pole is falling, with small
           corrections for the cart's position and velocity. No learning.
reinforce  REINFORCE with a baseline. A trained MLP policy, acting by argmax (greedy).
actor_critic  the same policy shape plus a learned value network. Acts greedily; the critic is
           used only to report its value estimate to the page.
cem        cross-entropy search. A linear policy: push right when w . obs + b > 0.

Trained agents load their weights from cartpole/data/*.npz. Those files are produced by
`python -m cartpole train` and committed, so serving and benchmarking need no training.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .env import LEFT, RIGHT
from .nets import load_npz as _load_npz
from .nets import normalise, softmax

AGENT_NAMES = ("random", "pd", "reinforce", "actor_critic", "cem")
LABELS = {
    "random": "RANDOM",
    "pd": "PD CONTROLLER",
    "reinforce": "REINFORCE + BASELINE",
    "actor_critic": "ACTOR-CRITIC",
    "cem": "CROSS-ENTROPY SEARCH",
}
DATA_DIR = Path(__file__).resolve().parent / "data"

# PD gains on (angle, angular velocity, cart position, cart velocity). Chosen by hand after a
# coarse sweep over tuning seeds 9000-9099, which are not benchmark seeds. The push goes right
# when the sum is positive, so a pole leaning right (theta > 0) gets a push to the right.
PD_GAINS = (30.0, 6.0, 0.1, 0.5)


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, obs) -> int:
        return self.rng.choice((LEFT, RIGHT))


class PDAgent:
    def __init__(self, gains=PD_GAINS):
        self.gains = gains

    def act(self, obs) -> int:
        x, x_dot, theta, theta_dot = obs
        kt, kw, kx, kv = self.gains
        u = kt * theta + kw * theta_dot + kx * x + kv * x_dot
        return RIGHT if u > 0 else LEFT

    def control(self, obs) -> float:
        """The signed control value u before the sign is taken, for the page's readout."""
        x, x_dot, theta, theta_dot = obs
        kt, kw, kx, kv = self.gains
        return kt * theta + kw * theta_dot + kx * x + kv * x_dot


class NetAgent:
    """Greedy agent for an MLP policy (and, optionally, the critic). Shared by REINFORCE and actor-critic."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.W1, self.b1 = weights["W1"], weights["b1"]
        self.W2, self.b2 = weights["W2"], weights["b2"]
        self.critic = None
        if "v_W1" in weights:
            self.critic = (weights["v_W1"], weights["v_b1"], weights["v_W2"], weights["v_b2"])
            self.value_scale = float(weights["value_scale"][0])

    def logits(self, obs) -> np.ndarray:
        h = np.tanh(self.W1 @ normalise(obs) + self.b1)
        return self.W2 @ h + self.b2

    def probs(self, obs) -> tuple[float, float]:
        """(P(left), P(right)) under the policy."""
        p = softmax(self.logits(obs)[None, :])[0]
        return float(p[0]), float(p[1])

    def value(self, obs) -> float | None:
        """The critic's estimate of the discounted return from obs, or None for REINFORCE."""
        if self.critic is None:
            return None
        W1, b1, W2, b2 = self.critic
        h = np.tanh(W1 @ normalise(obs) + b1)
        return float(self.value_scale * (W2 @ h + b2)[0])

    def act(self, obs) -> int:
        z = self.logits(obs)
        return RIGHT if z[1] > z[0] else LEFT


class LinearAgent:
    """Cross-entropy policy: five numbers, the last one a bias. Pushes right when the score is positive."""

    def __init__(self, theta):
        self.w = np.asarray(theta[:4], dtype=np.float64)
        self.b = float(theta[4])

    def act(self, obs) -> int:
        return RIGHT if float(self.w @ normalise(obs)) + self.b > 0 else LEFT


def load_weights(name: str, data_dir: Path = DATA_DIR) -> dict[str, np.ndarray]:
    path = data_dir / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"no trained weights at {path}; run: python -m cartpole train --algo {name}")
    return _load_npz(path)


def load_agent(name: str, seed: int = 0, data_dir: Path = DATA_DIR):
    """Build an agent by name. The seed only matters for the random agent."""
    if name == "random":
        return RandomAgent(seed)
    if name == "pd":
        return PDAgent()
    if name in ("reinforce", "actor_critic"):
        return NetAgent(load_weights(name, data_dir))
    if name == "cem":
        return LinearAgent(load_weights("cem", data_dir)["theta"])
    raise ValueError(f"unknown agent {name!r}; choose from {', '.join(AGENT_NAMES)}")
