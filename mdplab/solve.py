"""Planning with a known model: value iteration and policy iteration.

Both solvers need the transition table in mdp.py, so they compute exact expectations over slips rather
than sampling. They are written so the JavaScript port performs the same floating-point operations in the
same order: the results agree to the last bit, which the parity test relies on.

Sweeps are synchronous (Jacobi): every new value is computed from the previous sweep's values. A sweep
therefore moves information exactly one step, which is what makes the values spread out like heat on the page.

Bellman backup for state s and action a, with V the current values:
    Q(s, a) = sum over (p, s', r) in P(s, a) of  p * (r + gamma * V(s'))
Terminal s' have V = 0 because their entries are never updated.
"""

from __future__ import annotations

from bandits.rng import Rng  # the shared mulberry32 stream, also used by the JavaScript port

from .grid import ARROWS, TIE
from .mdp import GridMDP


def backup(mdp: GridMDP, V: list[float], s: int) -> list[float]:
    """Q(s, a) for all four actions, from values V."""
    g = mdp.params.gamma
    row = []
    for a in range(4):
        q = 0.0
        for prob, dest, r in mdp.P[s][a]:
            q += prob * (r + g * V[dest])
        row.append(q)
    return row


def best_action(qs: list[float]) -> int:
    """The lowest-index action whose value is within TIE of the best. Ties go to the lower index."""
    top = max(qs)
    for a, q in enumerate(qs):
        if q >= top - TIE:
            return a
    return 0  # unreachable: the maximum always satisfies the test


def greedy_policy(mdp: GridMDP, V: list[float]) -> list[int]:
    """The greedy action for each state from values V; -1 for non-states (walls, terminals, cliffs)."""
    pi = [-1] * mdp.n
    for s in mdp.states:
        pi[s] = best_action(backup(mdp, V, s))
    return pi


def q_table(mdp: GridMDP, V: list[float]) -> list[list[float]]:
    """Q(s, a) for every state; non-states get four zeros so the list can be indexed by cell."""
    table = [[0.0] * 4 for _ in range(mdp.n)]
    for s in mdp.states:
        table[s] = backup(mdp, V, s)
    return table


def reachable(mdp: GridMDP, pi: list[int]) -> list[int]:
    """States the agent can reach from the start by following pi, counting any slip with positive probability.

    This is the set of states that matter for the start: a learned policy is judged on these, since a dead
    end that no sensible route visits never earns any feedback to correct it.
    """
    seen = {mdp.start}
    stack = [mdp.start]
    while stack:
        s = stack.pop()
        for _prob, dest, _r in mdp.P[s][pi[s]]:
            if dest not in seen and not mdp.terminal[dest]:
                seen.add(dest)
                stack.append(dest)
    return sorted(seen)


def arrow(a: int) -> str:
    """The arrow glyph for action a, or a dot when a is -1 (no action)."""
    return ARROWS[a] if a >= 0 else "·"


class ValueIterator:
    """Value iteration, one sweep at a time.

    Starts from V = 0. Each sweep sets V(s) = max_a Q(s, a) for every state. `residual` is the largest
    change in any value in the last sweep, which is the Bellman residual of the previous values. Under
    a discount below 1 it shrinks at least by a factor gamma per sweep.
    """

    def __init__(self, mdp: GridMDP):
        self.mdp = mdp
        self.V = [0.0] * mdp.n
        self.sweeps = 0
        self.residual = float("inf")

    def sweep(self) -> float:
        """One synchronous sweep. Returns the residual of this sweep."""
        old = self.V
        new = list(old)
        res = 0.0
        for s in self.mdp.states:
            best = max(backup(self.mdp, old, s))
            res = max(res, abs(best - old[s]))
            new[s] = best
        self.V = new
        self.sweeps += 1
        self.residual = res
        return res

    def solve(self, eps: float, max_sweeps: int = 100_000) -> int:
        """Sweep until the residual is below eps (or max_sweeps). Returns the number of sweeps run."""
        while self.residual >= eps and self.sweeps < max_sweeps:
            self.sweep()
        return self.sweeps

    def policy(self) -> list[int]:
        return greedy_policy(self.mdp, self.V)


class PolicyIterator:
    """Policy iteration: alternate evaluating the current policy and improving it greedily.

    The starting policy is one action per state drawn from a seeded stream, so it is usually poor. Each
    evaluation sweep sets V(s) = sum over P(s, pi(s)) of p (r + gamma V(s')), again synchronously, and
    the values are kept between rounds. An improvement step switches a state to a better action only
    when the new one beats the current one by more than TIE, which guarantees termination. A round that
    changes no action means the policy is stable, and so optimal.
    """

    def __init__(self, mdp: GridMDP, seed: int = 1):
        self.mdp = mdp
        self.V = [0.0] * mdp.n
        self.pi = [-1] * mdp.n
        rng = Rng(seed)
        for s in mdp.states:
            self.pi[s] = rng.index(4)
        self.eval_sweeps = 0
        self.rounds = 0  # improvement steps taken
        self.residual = float("inf")
        self.stable = False
        self.last_changed = -1

    def eval_sweep(self) -> float:
        """One synchronous evaluation sweep for the current policy. Returns its residual."""
        g = self.mdp.params.gamma
        old = self.V
        new = list(old)
        res = 0.0
        for s in self.mdp.states:
            q = 0.0
            for prob, dest, r in self.mdp.P[s][self.pi[s]]:
                q += prob * (r + g * old[dest])
            res = max(res, abs(q - old[s]))
            new[s] = q
        self.V = new
        self.eval_sweeps += 1
        self.residual = res
        return res

    def evaluate(self, eps: float, max_sweeps: int = 5000) -> int:
        """Evaluate until the residual is below eps. Returns the sweeps this phase used.

        The cap matters for undiscounted problems: a deterministic policy that loops never settles, and
        the cap stops the loop. The improvement step still runs on the capped values.
        """
        used = 0
        self.residual = float("inf")
        while self.residual >= eps and used < max_sweeps:
            self.eval_sweep()
            used += 1
        return used

    def improve(self) -> int:
        """Greedy improvement against the current values. Returns how many states changed action."""
        changed = 0
        for s in self.mdp.states:
            qs = backup(self.mdp, self.V, s)
            top = max(qs)
            # Keep the current action if it is still within TIE of the best.
            if qs[self.pi[s]] >= top - TIE:
                continue
            self.pi[s] = best_action(qs)
            changed += 1
        self.rounds += 1
        self.last_changed = changed
        self.stable = changed == 0
        return changed

    def solve(self, eps: float, max_rounds: int = 500, max_sweeps: int = 5000) -> dict:
        """Run evaluate/improve rounds until the policy is stable. Returns a summary dict."""
        capped = False
        while not self.stable and self.rounds < max_rounds:
            used = self.evaluate(eps, max_sweeps)
            capped = capped or used >= max_sweeps
            self.improve()
        return {"rounds": self.rounds, "eval_sweeps": self.eval_sweeps, "stable": self.stable, "capped": capped}

    def policy(self) -> list[int]:
        return list(self.pi)
