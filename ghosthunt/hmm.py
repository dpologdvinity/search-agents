"""Exact inference for one ghost: the forward algorithm (filtering) and Viterbi (the most likely path).

The ghost is a hidden Markov model. Its hidden state is s = 4*k + h (cell and heading, see motion.py); the
readings are the sonar pings. Filtering keeps the belief b_t(s) = P(state at t | readings 0..t):

    predict:  b'(s') = sum over s of b(s) * P(s' | s, player_t)          (the motion model)
    update:   b(s') <- b'(s') * P(reading_t | distance(cell(s'), player_t)), then normalise  (the sonar)

Each turn is one predict followed by one update, which is the forward algorithm. Its cost is linear in the
number of states times the number of successors, so the whole board is cheap enough to do exactly.

Viterbi runs the same recursion with max instead of sum, in log space, and keeps back-pointers. After a ghost
is busted it recovers the single most likely sequence of hidden states, which the page replays against the
true path.
"""

from __future__ import annotations

import math

from . import motion


class ExactFilter:
    """Belief over one ghost's hidden states, updated exactly by the forward recursion.

    `prior` is a list of length 4K whose entries sum to 1. `emis` is the sonar table from sonar.emission_table.
    """

    def __init__(self, maze, model: str, prior: list[float], emis: list[list[float]]):
        self.maze = maze
        self.model = model
        self.emis = emis
        self.belief = list(prior)

    def predict(self, player: int) -> None:
        """Push the belief through the motion model. The player's cell matters only for the lurker."""
        trans = motion.table(self.maze, self.model, player)
        new = [0.0] * len(self.belief)
        for s, p in enumerate(self.belief):
            if p == 0.0:
                continue
            for s2, q in trans[s]:
                new[s2] += p * q
        self.belief = new

    def update(self, reading: int, player: int) -> None:
        """Multiply by the sonar likelihood of each state, then normalise so the belief sums to 1."""
        dist = self.maze.dist
        emis = self.emis
        new = [p * emis[dist[s >> 2][player]][reading] for s, p in enumerate(self.belief)]
        total = sum(new)
        if total > 0.0:
            self.belief = [x / total for x in new]
        else:
            # Numerical guard: every state underflowed. Restart from the sonar alone rather than divide by zero.
            like = [emis[dist[s >> 2][player]][reading] for s in range(len(self.belief))]
            z = sum(like)
            self.belief = [x / z for x in like]

    def marginal(self) -> list[float]:
        """Probability of each open cell: the belief summed over the four headings."""
        out = [0.0] * self.maze.K
        for s, p in enumerate(self.belief):
            out[s >> 2] += p
        return out


def forward(maze, model: str, prior, emis, players: list[int], readings: list[int]) -> list[list[float]]:
    """Beliefs over states after each reading, for a known list of player cells and readings (used by tests).

    players[t] and readings[t] are the player's cell and the reading at time t. The result has one belief per
    time, the same as running ExactFilter turn by turn.
    """
    f = ExactFilter(maze, model, prior, emis)
    f.update(readings[0], players[0])
    out = [list(f.belief)]
    for t in range(1, len(readings)):
        f.predict(players[t])
        f.update(readings[t], players[t])
        out.append(list(f.belief))
    return out


def viterbi(maze, model: str, prior, emis, players: list[int], readings: list[int]) -> list[int]:
    """The most likely hidden state at each time given all readings (Viterbi, in log space).

    delta[s] is the best log-probability of any state sequence ending in s. Each step keeps, for every new
    state, the predecessor that maximises delta + log transition, then adds the log sonar likelihood. Back-pointers
    record those predecessors so the best sequence can be read off from the last time backwards.
    Impossible transitions and zero prior entries are -inf and never win.
    """
    NEG = -math.inf
    S = len(prior)
    dist = maze.dist

    def log_emis(s, t):
        return math.log(emis[dist[s >> 2][players[t]]][readings[t]])

    delta = [math.log(p) + log_emis(s, 0) if p > 0.0 else NEG for s, p in enumerate(prior)]
    back: list[list[int]] = []
    for t in range(1, len(readings)):
        trans = motion.table(maze, model, players[t])
        new = [NEG] * S
        ptr = [-1] * S
        for s, d in enumerate(delta):
            if d == NEG:
                continue
            for s2, q in trans[s]:
                cand = d + math.log(q)
                if cand > new[s2]:  # strict '>' keeps the lowest-index predecessor on ties
                    new[s2] = cand
                    ptr[s2] = s
        for s2 in range(S):
            if new[s2] > NEG:
                new[s2] += log_emis(s2, t)
        delta = new
        back.append(ptr)
    # Read the path backwards from the best final state.
    last = max(range(S), key=lambda s: (delta[s], -s))
    path = [last]
    for ptr in reversed(back):
        path.append(ptr[path[-1]])
    path.reverse()
    return path
