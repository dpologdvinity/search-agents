"""A grid as a Markov decision process: states, actions, and the transition table P(s', r | s, a).

Moves. The agent tries to move in the chosen direction. With probability 1 - slip it goes that way; with
probability slip it slides, going one of the two perpendicular directions (half the slip mass each).
A move into a wall or off the edge leaves the agent in place. Every step pays `living`, plus the cell's
payout when the agent enters a + or - cell (the episode then ends) or a C cell (the agent is sent to S).

Values. Terminal cells have value 0: the episode is over, so nothing follows. The table stores, for
each state and action, the list of (probability, next state, reward) triples the solvers sum over.

`resolve` and `direction_for` are the same rules the Q-learning sampler uses. Keeping them here means the
model and the sampler cannot disagree about the physics.
"""

from __future__ import annotations

from .grid import CLIFF, DELTAS, EMPTY, PIT, REWARD, WALL, Grid, Params


class GridMDP:
    """The MDP for one layout and one parameter set. Built once; solvers read its tables.

    Attributes (all lists indexed by cell, length n = width * height):
        kinds      the cell characters
        terminal   True for + and - cells
        is_state   True for cells the agent can stand on (not wall, terminal, or cliff)
        states     indices of the states, in increasing order (solvers sweep in this order)
        res        res[s][d] = (destination, cell payout) for a move from s in direction d
        P          P[s][a] = tuple of (probability, destination, reward) for action a from s
    """

    def __init__(self, grid: Grid, params: Params):
        params.validate()
        self.grid = grid
        self.params = params
        self.width = grid.width
        self.height = grid.height
        self.n = grid.width * grid.height
        self.kinds = grid.flat
        self.start = grid.start
        self.terminal = [k in (REWARD, PIT) for k in self.kinds]
        self.is_state = [k not in (WALL, REWARD, PIT, CLIFF) for k in self.kinds]
        self.states = [i for i in range(self.n) if self.is_state[i]]
        # Payout for entering a cell. Empty cells and the start pay nothing.
        pay = {REWARD: params.reward, PIT: params.pit, CLIFF: params.cliff}
        self.cell_pay = [pay.get(k, 0.0) for k in self.kinds]
        self.res: list[list[tuple[int, float]] | None] = [None] * self.n
        self.P: list[list[tuple[tuple[float, int, float], ...]] | None] = [None] * self.n
        for s in self.states:
            self.res[s] = [self._resolve(s, d) for d in range(4)]
        for s in self.states:
            self.P[s] = [self._transitions(s, a) for a in range(4)]

    def _resolve(self, s: int, d: int) -> tuple[int, float]:
        """Where a deterministic move from s in direction d ends up, and the payout for the cell it enters."""
        w, h = self.width, self.height
        x, y = s % w, s // w
        dx, dy = DELTAS[d]
        nx, ny = x + dx, y + dy
        # Off the edge or into a wall: the agent stays where it is, and pays nothing extra.
        if not (0 <= nx < w and 0 <= ny < h) or self.kinds[ny * w + nx] == WALL:
            return s, 0.0
        t = ny * w + nx
        if self.kinds[t] == CLIFF:
            # Falling sends the agent back to the start, but the payout is still the cliff's.
            return self.start, self.cell_pay[t]
        return t, self.cell_pay[t]

    def _transitions(self, s: int, a: int) -> tuple[tuple[float, int, float], ...]:
        """The (probability, next state, reward) triples for action a from state s.

        The intended direction gets 1 - slip. Each perpendicular direction gets slip / 2; zero-probability
        entries are dropped so slip = 0 gives a single outcome. Order: intended, then left-slip, then
        right-slip, matching direction_for.
        """
        p = self.params.slip
        outcomes = ((1.0 - p, a), (p / 2.0, (a + 3) % 4), (p / 2.0, (a + 1) % 4))
        out = []
        for prob, d in outcomes:
            if prob <= 0.0:
                continue
            dest, pay = self.res[s][d]
            out.append((prob, dest, self.params.living + pay))
        return tuple(out)

    def direction_for(self, a: int, u: float) -> int:
        """The direction actually moved, given the intended action a and a uniform draw u in [0, 1).

        This is the sampler's view of the same distribution `_transitions` describes: u below 1 - slip
        keeps the intended direction, the next slip/2 of the range goes left of it, the rest goes right.
        """
        p = self.params.slip
        if u < 1.0 - p:
            return a
        if u < 1.0 - p / 2.0:
            return (a + 3) % 4
        return (a + 1) % 4

    def check_reachable(self) -> bool:
        """True when every state can reach a terminal cell by some sequence of moves (ignoring slips)."""
        goal_ok = set(i for i in range(self.n) if self.terminal[i])
        # Work backwards from the terminals over the reverse of the deterministic moves.
        changed = True
        reach = set(goal_ok)
        while changed:
            changed = False
            for s in self.states:
                if s in reach:
                    continue
                if any(dest in reach for dest, _ in self.res[s]):
                    reach.add(s)
                    changed = True
        return all(s in reach for s in self.states)

    def describe(self) -> dict:
        """A small summary used by the CLI header and the API."""
        return {
            "width": self.width,
            "height": self.height,
            "states": len(self.states),
            "terminals": sum(self.terminal),
            "empty": self.kinds.count(EMPTY),
        }
