"""Learning from experience: tabular Q-learning and SARSA, with the agent walking the grid.

The learner never sees the model. It acts, observes a next cell and a payout, and nudges one table entry
Q(s, a) toward a target built from that single step:

    Q-learning:  target = r + gamma * max_a' Q(s', a')          (off-policy: the best next action)
    SARSA:       target = r + gamma * Q(s', a')                (on-policy: the action it will really take)
    terminal s': target = r                                      (nothing follows the episode's end)
    update:      Q(s, a) += alpha * (target - Q(s, a))

Actions are chosen epsilon-greedily: with probability epsilon a uniform random action, otherwise the greedy
action, with exact ties broken uniformly at random. epsilon is multiplied by `decay` after each episode, down
to eps_min. Randomness comes from the shared mulberry32 stream, so a seed fixes every episode, and the
JavaScript port reproduces the same trajectories.

Random draws happen in a fixed order per step: one uniform for the exploration test (plus one index if
exploring), one uniform for the slip, and for ties one index. Both implementations follow this order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bandits.rng import Rng

from .grid import TIE
from .mdp import GridMDP
from .solve import greedy_policy, q_table, reachable


@dataclass
class Episode:
    """What one episode did: its undiscounted return, its length, and the cells visited in order."""

    ret: float
    steps: int
    path: list[int] = field(default_factory=list)
    reached_terminal: bool = False
    fell: bool = False  # True when the episode ended in a pit (a - cell)


class QLearner:
    """A tabular agent that learns Q(s, a) from episodes. `algo` is "q" or "sarsa"."""

    def __init__(self, mdp: GridMDP, *, algo: str = "q", seed: int = 1, alpha: float | None = None,
                 epsilon: float | None = None, decay: float | None = None, eps_min: float | None = None,
                 max_steps: int | None = None):
        if algo not in ("q", "sarsa"):
            raise ValueError("algo must be 'q' or 'sarsa'")
        p = mdp.params
        self.mdp = mdp
        self.algo = algo
        self.alpha = p.alpha if alpha is None else alpha
        self.epsilon0 = p.epsilon if epsilon is None else epsilon
        self.decay = p.decay if decay is None else decay
        self.eps_min = p.eps_min if eps_min is None else eps_min
        self.max_steps = p.max_steps if max_steps is None else max_steps
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.rng = Rng(seed)
        self.Q = [[0.0] * 4 for _ in range(mdp.n)]
        self.eps = self.epsilon0
        self.episodes = 0
        self.returns: list[float] = []
        self.last: Episode | None = None

    def greedy(self, s: int) -> int:
        """The greedy action at s, with exact-ish ties (within TIE) to the lowest index. No randomness."""
        qs = self.Q[s]
        top = max(qs)
        for a in range(4):
            if qs[a] >= top - TIE:
                return a
        return 0

    def choose(self, s: int) -> int:
        """Epsilon-greedy action at s. Ties among the greedy actions are broken at random."""
        if self.rng.uniform() < self.eps:
            return self.rng.index(4)
        qs = self.Q[s]
        top = max(qs)
        ties = [a for a in range(4) if qs[a] >= top - TIE]
        if len(ties) == 1:
            return ties[0]
        return ties[self.rng.index(len(ties))]

    def episode(self) -> Episode:
        """Run one episode from the start until a terminal cell or max_steps. Learns on every step."""
        mdp = self.mdp
        p = mdp.params
        g = p.gamma
        Q = self.Q
        s = mdp.start
        a = self.choose(s)
        ret = 0.0
        steps = 0
        path = [s]
        ended = False
        while True:
            # The slip draw decides the direction actually moved, then the table says where that leads.
            d = mdp.direction_for(a, self.rng.uniform())
            dest, pay = mdp.res[s][d]
            r = p.living + pay
            ret += r
            steps += 1
            path.append(dest)
            done = mdp.terminal[dest]
            a_next = -1
            if done:
                target = r
            elif self.algo == "sarsa":
                # SARSA picks the next action now, so its target uses the action it will actually take.
                a_next = self.choose(dest)
                target = r + g * Q[dest][a_next]
            else:
                target = r + g * max(Q[dest])
            Q[s][a] += self.alpha * (target - Q[s][a])
            if done or steps >= self.max_steps:
                ended = done
                break
            s = dest
            a = a_next if self.algo == "sarsa" else self.choose(s)
        # Exploration decays once per episode, down to the floor.
        self.eps = max(self.eps_min, self.eps * self.decay)
        self.episodes += 1
        self.returns.append(ret)
        ep = Episode(ret=ret, steps=steps, path=path, reached_terminal=ended,
                     fell=ended and mdp.kinds[path[-1]] == "-")
        self.last = ep
        return ep

    def run(self, episodes: int) -> list[float]:
        """Run several episodes. Returns their returns."""
        return [self.episode().ret for _ in range(episodes)]

    def policy(self) -> list[int]:
        """The greedy policy from the learned table; -1 for non-states."""
        pi = [-1] * self.mdp.n
        for s in self.mdp.states:
            pi[s] = self.greedy(s)
        return pi

    def agreement(self, optimal_pi: list[int], optimal_Q: list[list[float]], slack: float = 1e-6
                  ) -> tuple[float, list[bool]]:
        """How often the learned greedy action is optimal, over the states the optimal policy reaches.

        A state agrees when the learned action equals the optimal one, or when its exact optimal value is
        within `slack` of the best (so two equally good moves both count). Dead ends that no optimal route
        visits are left out of the fraction, because nothing the agent does there is ever tested. Returns
        the fraction over those states and a per-cell list of flags (False outside them).
        """
        scope = reachable(self.mdp, optimal_pi)
        flags = [False] * self.mdp.n
        ok = 0
        for s in scope:
            g = self.greedy(s)
            best = max(optimal_Q[s])
            same = g == optimal_pi[s] or optimal_Q[s][g] >= best - slack
            flags[s] = same
            ok += same
        return (ok / len(scope) if scope else 1.0), flags


def optimal_reference(mdp: GridMDP, V: list[float]) -> tuple[list[int], list[list[float]]]:
    """The optimal policy and Q table for values V, the reference the learner is compared with."""
    return greedy_policy(mdp, V), q_table(mdp, V)
