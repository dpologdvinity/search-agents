"""The Ghost Hunt board: invisible ghosts, a player who hears sonar pings, and the rules that tie them together.

Rules
- A seeded maze is built, and the player starts in the top-left room.
- Each ghost starts at a random open cell at least 4 steps from the player. Its heading is random too.
- A turn is one player action, then the ghosts move, then the sonar pings every live ghost.
  Actions: move N/E/S/W or stay ".", or bust a cell: "bN", "bE", "bS", "bW" (an adjacent cell) or "b." (the
  player's own cell). A bust removes the ghost if it is on that cell when the action resolves; a miss wastes
  the turn. A bust does not move the player.
- The game ends when every ghost is busted or after max_turns. Fewer turns is better.

Hidden and known information
- The ghosts' cells, headings and the sonar noise are hidden. The maze, the motion models, the sonar table,
  the prior and the player's own path are known. Every filter uses only what the player knows.

Randomness: each ghost has its own mulberry32 stream for its start, its motion and its pings, and each filter has
another. Ghosts draw exactly one uniform per turn while they are live, so two runs with the same seed see the same
ghost movement and the same pings, even when the player plays differently (common random numbers). This is
what lets the benchmark compare filters fairly.
"""

from __future__ import annotations

from bandits.rng import Rng

from . import motion, sonar
from .hmm import ExactFilter, viterbi
from .maze import Maze, generate
from .particles import ParticleFilter

ACTIONS = ("N", "E", "S", "W", ".", "bN", "bE", "bS", "bW", "b.")
DIR_OF = {"N": 0, "E": 1, "S": 2, "W": 3, ".": 4}
MIN_START_DISTANCE = 4  # ghosts never start this close to the player (in steps)
TURN_CAP = 300          # a game ends after this many turns if some ghost is still free
FILTERS = ("exact", "particles")


def _pick(options: list[tuple[int, float]], u: float) -> int:
    """Sample a successor: the first option whose cumulative probability exceeds the uniform u."""
    acc = 0.0
    for s2, q in options:
        acc += q
        if u < acc:
            return s2
    return options[-1][0]


class Ghost:
    """One hidden ghost: its true state, its own random stream, its belief, and what the player has heard."""

    def __init__(self, gid: int, x: int, rng: Rng, filt):
        self.id = gid
        self.x = x                 # true hidden state 4*k + h
        self.rng = rng             # world stream: motion draws and sonar draws
        self.filter = filt         # ExactFilter or ParticleFilter over this ghost's states
        self.live = True
        self.bust_turn: int | None = None
        self.true: list[int] = [x >> 2]  # true open cell at every turn it was live (ends at the bust turn)
        self.readings: list[int] = []    # sonar readings, one per turn it was live

    @property
    def k(self) -> int:
        return self.x >> 2


class Game:
    """One game: the maze, the player's position and the ghosts with their filters.

    filt="exact" runs the forward algorithm; filt="particles" uses `particles` samples per ghost.
    """

    def __init__(self, size: int = 15, ghosts: int = 2, model: str = "random", sigma: float = 1.0,
                 seed: int = 1, filt: str = "exact", particles: int = 200, max_turns: int = TURN_CAP):
        if model not in motion.MODELS:
            raise ValueError(f"model must be one of {motion.MODELS}")
        if filt not in FILTERS:
            raise ValueError(f"filter must be one of {FILTERS}")
        if not 1 <= ghosts <= 3:
            raise ValueError("ghosts must be 1, 2 or 3")
        self.size, self.model, self.sigma, self.seed = size, model, sigma, seed
        self.filt, self.particles, self.max_turns = filt, particles, max_turns
        self.maze = Maze(size, generate(size, seed))
        self.emis = sonar.emission_table(sigma, self.maze.diameter)
        eligible = [4 * k + h for k in range(self.maze.K) for h in range(4)
                    if self.maze.dist[k][self.maze.start] >= MIN_START_DISTANCE]
        self.eligible = eligible
        # The prior is what every filter starts from: uniform over eligible (cell, heading) states.
        self.prior = [0.0] * (4 * self.maze.K)
        for s in eligible:
            self.prior[s] = 1.0 / len(eligible)
        self.p = self.maze.start
        self.players = [self.p]
        self.t = 0
        self.ghosts: list[Ghost] = []
        for g in range(ghosts):
            rng = Rng(seed + 7919 * (g + 1))
            x = eligible[rng.index(len(eligible))]  # where the ghost really starts
            if filt == "exact":
                f = ExactFilter(self.maze, model, self.prior, self.emis)
            else:
                f = ParticleFilter(self.maze, model, self.prior, self.emis, particles,
                                   Rng(seed + 7919 * (g + 1) + 0x10000))
            self.ghosts.append(Ghost(g, x, rng, f))
        for gh in self.ghosts:  # the first ping, before anyone has acted
            self._ping(gh)

    # ── rules ──────────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return all(not gh.live for gh in self.ghosts) or self.t >= self.max_turns

    @property
    def busted(self) -> int:
        return sum(1 for gh in self.ghosts if not gh.live)

    def legal_actions(self) -> list[str]:
        out = []
        for d, ch in enumerate("NESW"):
            if self.maze.step[self.p][d] >= 0:
                out.append(ch)
                out.append("b" + ch)
        return [".", "b."] + out

    def act(self, action: str) -> dict:
        """Play one turn. Returns what happened: the busted ghosts and the sonar readings of this turn."""
        if self.done:
            raise RuntimeError("the game is over")
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        p = self.p
        busted_now = []
        if action.startswith("b"):
            d = DIR_OF[action[1:]]
            target = p if d == 4 else self.maze.step[p][d]
            if target < 0:
                raise ValueError("no open cell in that direction")
            for gh in self.ghosts:  # the bust resolves against the ghosts' current cells, before they move
                if gh.live and gh.k == target:
                    gh.live = False
                    gh.bust_turn = self.t
                    busted_now.append(gh.id)
        else:
            d = DIR_OF[action]
            if d != 4:
                q = self.maze.step[p][d]
                if q < 0:
                    raise ValueError("that way is a wall")
                p = q
        self.p = p
        self.t += 1
        self.players.append(p)
        readings = {}
        for gh in self.ghosts:
            if not gh.live:
                continue
            # Ghosts move, then the sonar pings from the player's new cell. The filter does the same steps
            # in the same order: predict with the motion model, then update with the reading.
            gh.x = _pick(motion.transitions(self.maze, self.model, gh.x, p), gh.rng.uniform())
            gh.true.append(gh.k)
            r = self._ping(gh)
            readings[gh.id] = r
        return {"t": self.t, "action": action, "busted": busted_now, "readings": readings}

    def _ping(self, gh: Ghost) -> int:
        """Draw a noisy reading of the distance from the player to the ghost and feed it to the filter."""
        d = self.maze.dist[gh.k][self.p]
        r = sonar.sample_reading(self.emis[d], gh.rng.uniform())
        gh.readings.append(r)
        if self.t > 0:  # the first ping has no motion before it: the prior is already the belief about time 0
            gh.filter.predict(self.p)
        gh.filter.update(r, self.p)
        return r

    # ── views for the agent, the page and the tests ────────────────

    def marginal(self, gid: int) -> list[float]:
        """Current belief about ghost gid's cell (length K). Busted ghosts keep their last belief."""
        return self.ghosts[gid].filter.marginal()

    def true_cell(self, gid: int) -> int:
        return self.ghosts[gid].k

    def viterbi_path(self, gid: int) -> list[int]:
        """Most likely cell sequence of ghost gid up to its bust (or to now, if it is still live).

        Returns open-cell indices, one per turn from 0 to the last turn it was observed.
        """
        gh = self.ghosts[gid]
        n = len(gh.readings)
        states = viterbi(self.maze, self.model, self.prior, self.emis, self.players[:n], gh.readings)
        return [s >> 2 for s in states]

    def trace(self, gid: int) -> dict:
        """Everything the page needs to replay one ghost: its true cells, the Viterbi cells and the readings."""
        gh = self.ghosts[gid]
        return {"true": list(gh.true), "viterbi": self.viterbi_path(gid), "readings": list(gh.readings),
                "bust_turn": gh.bust_turn}
