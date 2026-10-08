"""Cart-pole physics, written from the classic equations so there is no gym dependency.

A cart slides on a track and a pole hinged to it hangs from the top. Each step the agent
pushes the cart left or right with a fixed force. The episode ends when the pole falls past
12 degrees or the cart leaves |x| > 2.4, and is cut off (truncated) at 500 steps. Every step
gives reward 1, as in gym's CartPole, so the return of an episode equals the number of steps.

State is (x, x_dot, theta, theta_dot). theta is measured from vertical, in radians. Positive
theta means the top of the pole leans to the right, so pushing right (action 1) corrects a
pole that leans right.
"""

from __future__ import annotations

import math
import random

GRAVITY = 9.8          # m/s^2
MASS_CART = 1.0        # kg
MASS_POLE = 0.1        # kg
TOTAL_MASS = MASS_CART + MASS_POLE
HALF_LENGTH = 0.5      # m, distance from the hinge to the pole's centre of mass
POLE_MASS_LENGTH = MASS_POLE * HALF_LENGTH
FORCE = 10.0           # N, the magnitude of every push
TAU = 0.02             # s per step (50 Hz control)
THETA_LIMIT = math.radians(12)
X_LIMIT = 2.4          # m, the track is |x| <= 2.4
MAX_STEPS = 500        # the episode is truncated here and counts as a full balance
INIT_RANGE = 0.05      # every state variable starts uniform in [-0.05, 0.05]

LEFT, RIGHT = 0, 1
ACTIONS = (LEFT, RIGHT)


class CartPole:
    """One episode at a time. Call reset(seed), then step(action) until terminated or truncated."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.x = self.x_dot = self.theta = self.theta_dot = 0.0
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.reset(seed)

    def reset(self, seed: int | None = None) -> tuple[float, float, float, float]:
        """Start a new episode. A seed makes the start state reproducible; None keeps the current stream."""
        if seed is not None:
            self.rng = random.Random(seed)
        r = self.rng.uniform
        self.x, self.x_dot = r(-INIT_RANGE, INIT_RANGE), r(-INIT_RANGE, INIT_RANGE)
        self.theta, self.theta_dot = r(-INIT_RANGE, INIT_RANGE), r(-INIT_RANGE, INIT_RANGE)
        self.steps = 0
        self.terminated = self.truncated = False
        return self.obs()

    def obs(self) -> tuple[float, float, float, float]:
        return (self.x, self.x_dot, self.theta, self.theta_dot)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def step(self, action: int) -> tuple[tuple[float, float, float, float], float, bool, bool]:
        """Advance one control step. Returns (obs, reward, terminated, truncated).

        With temp the cart acceleration that the push alone would produce, the pole's angular
        acceleration couples gravity (g sin theta) against the cart's acceleration (cos theta *
        temp). The cart is then pushed back by the reaction force of the pole.
        """
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        force = FORCE if action == RIGHT else -FORCE
        cos_t, sin_t = math.cos(self.theta), math.sin(self.theta)
        # Force on the cart, including the centripetal term from the pole's rotation.
        temp = (force + POLE_MASS_LENGTH * self.theta_dot * self.theta_dot * sin_t) / TOTAL_MASS
        # Angular acceleration of the pole (the 4/3 comes from the pole's moment of inertia about its end).
        theta_acc = (GRAVITY * sin_t - cos_t * temp) / (
            HALF_LENGTH * (4.0 / 3.0 - MASS_POLE * cos_t * cos_t / TOTAL_MASS))
        # Cart acceleration once the pole's reaction on the cart is subtracted.
        x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_t / TOTAL_MASS
        # Explicit Euler, positions first: x and theta advance with the velocities from the start of the step.
        self.x += TAU * self.x_dot
        self.x_dot += TAU * x_acc
        self.theta += TAU * self.theta_dot
        self.theta_dot += TAU * theta_acc
        self.steps += 1

        self.terminated = abs(self.x) > X_LIMIT or abs(self.theta) > THETA_LIMIT
        self.truncated = (not self.terminated) and self.steps >= MAX_STEPS
        return self.obs(), 1.0, self.terminated, self.truncated


def run_episode(agent, seed: int, max_steps: int = MAX_STEPS, record: bool = False):
    """Play one episode with agent.act(obs) -> action.

    Returns the step count, or (steps, trajectory) when record is set. A trajectory is a list
    with one dict per step: the observation the agent saw, the action it took, and the state
    after the step. The web page uses the same layout when it replays a rollout.
    """
    env = CartPole(seed)
    obs = env.reset(seed)
    traj = []
    while not env.done and env.steps < max_steps:
        action = agent.act(obs)
        seen = obs
        obs, _, _, _ = env.step(action)
        if record:
            traj.append({"obs": list(seen), "action": action, "next": list(obs)})
    return (env.steps, traj) if record else env.steps
