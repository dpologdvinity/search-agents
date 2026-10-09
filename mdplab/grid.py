"""Grid worlds for the MDP lab: the layout, the parameters, and the presets.

A grid is a rectangle read top row first. Each cell is one character:

    .  empty                  #  wall: never entered; a move into it leaves the agent where it stood
    S  start (exactly one)    +  reward cell: terminal, pays params.reward
    -  pit: terminal, pays params.pit                     C  cliff: not terminal, pays params.cliff
                                                             and sends the agent back to S

The cliff is the cliff-walking rule from Sutton and Barto (2018, example 6.6): falling costs a lot
but the episode goes on, so the agent learns by falling. Everything else ends the episode.

Actions are numbered 0 to 3 in the order up, right, down, left, the same order as the JavaScript port.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

ACTIONS = ("up", "right", "down", "left")
UP, RIGHT, DOWN, LEFT = range(4)
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # (dx, dy) for up, right, down, left
ARROWS = ("↑", "→", "↓", "←")

EMPTY, WALL, START, REWARD, PIT, CLIFF = ".", "#", "S", "+", "-", "C"
KINDS = (EMPTY, WALL, START, REWARD, PIT, CLIFF)

# Two action values count as tied when they differ by less than this. It keeps the greedy choice stable
# against last-bit rounding, and the JavaScript port uses the same constant.
TIE = 1e-9


@dataclass(frozen=True)
class Params:
    """Everything a solver needs besides the layout. Defaults are the lab's starting values.

    gamma       discount per step
    slip        probability that a move goes sideways (one of the two perpendicular directions, half each)
    living      reward added on every step, including a bump into a wall
    reward      payout for entering a + cell
    pit         payout for entering a - cell
    cliff       payout for stepping on a C cell (the agent then returns to the start)
    alpha       Q-learning step size
    epsilon     starting exploration rate; the rate is multiplied by decay after each episode
    decay       per-episode multiplier on epsilon
    eps_min     floor for epsilon
    max_steps   episode length cap for Q-learning and SARSA
    """

    gamma: float = 0.95
    slip: float = 0.1
    living: float = -0.04
    reward: float = 1.0
    pit: float = -1.0
    cliff: float = -100.0
    alpha: float = 0.1
    epsilon: float = 0.1
    decay: float = 0.999
    eps_min: float = 0.01
    max_steps: int = 500

    def validate(self) -> None:
        """Raise ValueError when a parameter is outside the range the solvers assume."""
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.slip <= 0.5:
            raise ValueError("slip must be in [0, 0.5]; above 0.5 the intended move is less likely than a slip")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if not 0.0 <= self.eps_min <= 1.0:
            raise ValueError("eps_min must be in [0, 1]")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")


@dataclass(frozen=True)
class Grid:
    """A layout: rows of cell characters, top row first. Validated on construction."""

    rows: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("a grid needs at least one row")
        width = len(self.rows[0])
        if width == 0 or any(len(r) != width for r in self.rows):
            raise ValueError("every row must have the same non-zero length")
        bad = {c for r in self.rows for c in r} - set(KINDS)
        if bad:
            raise ValueError(f"unknown cell characters: {sorted(bad)}")
        if sum(r.count(START) for r in self.rows) != 1:
            raise ValueError("a grid needs exactly one start cell (S)")

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def flat(self) -> str:
        """All cells, row by row. Cell (x, y) is at index y * width + x."""
        return "".join(self.rows)

    @property
    def start(self) -> int:
        return self.flat.index(START)


@dataclass(frozen=True)
class Preset:
    """A named starting point: a layout plus the parameters that make its lesson visible."""

    key: str
    label: str
    blurb: str
    rows: tuple[str, ...]
    params: Params

    def grid(self) -> Grid:
        return Grid(self.rows)


# The cliff keeps Sutton and Barto's settings (living reward -1, no slip, a constant epsilon of 0.1 and
# alpha 0.5), so SARSA and Q-learning differ only in their update target. The discount is 0.99 rather than
# their 1: with gamma = 1 a deterministic starting policy can loop forever and policy iteration never settles.
# The maze's pit sits on a shortcut, so the exact policy must choose between a long safe route and a short
# risky one; the windy corridor's slip pushes the agent sideways into the pits.
PRESETS: dict[str, Preset] = {
    p.key: p
    for p in (
        Preset(
            key="cliff",
            label="Cliff walk",
            blurb=("Sutton and Barto's cliff. Walk along the top edge or along the cliff? "
                   "Q-learning hugs the edge; SARSA keeps away from it."),
            rows=(
                "............",
                "............",
                "............",
                "SCCCCCCCCCC+",
            ),
            params=Params(gamma=0.99, slip=0.0, living=-1.0, reward=0.0, cliff=-100.0,
                          alpha=0.5, epsilon=0.1, decay=1.0, eps_min=0.1, max_steps=500),
        ),
        Preset(
            key="rooms",
            label="Four rooms",
            blurb="Four rooms joined by doorways, a goal in the far corner and a pit by the door. Wind on.",
            rows=(
                "###########",
                "#....#...+#",
                "#....#....#",
                "#.........#",
                "#....#....#",
                "##.#####.##",
                "#....#....#",
                "#.........#",
                "#....#..-.#",
                "#S...#....#",
                "###########",
            ),
            params=Params(gamma=0.95, slip=0.1, living=-0.04, reward=1.0, pit=-1.0),
        ),
        Preset(
            key="maze",
            label="Maze with a shortcut",
            blurb="A maze whose shortest route runs through a pit. The lab finds the long safe way round.",
            rows=(
                "#########",
                "#S..#...#",
                "#.#.#.#.#",
                "#.#...#.#",
                "#.#####.#",
                "#....-..#",
                "###.#.#.#",
                "#...#.#+#",
                "#########",
            ),
            params=Params(gamma=0.95, slip=0.1, living=-0.04, reward=1.0, pit=-1.0),
        ),
        Preset(
            key="windy",
            label="Windy corridor",
            blurb="A corridor with pits in its middle row. The wind (slip 0.3) pushes you sideways into them.",
            rows=(
                "##############",
                "#S...........#",
                "#.-.-.-.-.-.+#",
                "#............#",
                "##############",
            ),
            params=Params(gamma=0.95, slip=0.3, living=-0.04, reward=1.0, pit=-1.0),
        ),
    )
}


def preset(key: str, **overrides) -> Preset:
    """Return a preset, with any Params fields overridden by keyword (e.g. slip=0.2)."""
    if key not in PRESETS:
        raise KeyError(f"unknown preset {key!r}; choose from {', '.join(PRESETS)}")
    p = PRESETS[key]
    if overrides:
        p = replace(p, params=replace(p.params, **overrides))
    return p
