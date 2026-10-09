"""Hex agents behind one interface, shared by the command line, the web server, and the benchmark.

Every agent is a name in AGENTS. `choose` returns (move, analysis): the move (a cell, or SWAP) and a
dict describing how the agent decided, which the web page shows as a heat map and principal variation.
`player` gives the plain (n, cells, to_move, rng) -> cell callable the benchmark plays with.
"""

from __future__ import annotations

import random

from .board import ACROSS, DOWN, EMPTY, SWAP, Board, label, other
from .heuristic import random_move, shortest_path_move
from .mcts import search

AGENTS = {
    "rave": "RAVE-MCTS: UCT with all-moves-as-first statistics blended in by beta.",
    "uct": "Plain UCT: Monte Carlo tree search with random fill playouts and no AMAF.",
    "shortest": "One-ply shortest-path race: the move that most lengthens the opponent's distance.",
    "random": "Uniform random legal move.",
    "chance": "Chance (fixed odds): a random empty cell weighted 4/3/2/1 by hex ring from the centre "
              "(rings 0/1/2/3+), renormalised over empty cells. No search, evaluation or lookahead.",
}

# Fixed weight for the chance opponent, by hex distance from the centre cell (ring 0 is the centre,
# ring 1 its six neighbours, and so on). Rings 3 and beyond share the last weight. Centre cells lie
# on the most possible chains, so the weights favour the middle the way a casual player might, but
# every empty cell keeps a non-zero chance, which is what makes it a fixed-odds baseline.
CHANCE_RING_WEIGHTS = (4, 3, 2, 1)

# Search budget per strength level, per move. Seconds cap the wall time so one request stays bounded.
LEVELS = {
    1: {"sims": 300, "seconds": 0.4},
    2: {"sims": 1000, "seconds": 0.9},
    3: {"sims": 3000, "seconds": 1.8},
}
HINT_SIMS = 3000  # the CLI's hint search


def hex_ring(n: int, cell: int) -> int:
    """Hex distance from the centre cell to `cell`, on the same six-neighbour grid as neighbours().

    The steps are (0, +-1), (+-1, 0), (-1, +1) and (+1, -1). Two cells whose row and column offsets
    have the same sign need |dr| + |dc| steps (the diagonal is not a neighbour); otherwise the
    distance is max(|dr|, |dc|).
    """
    r, c = divmod(cell, n)
    m = (n - 1) // 2  # the centre row and column; for even n this is the upper of the two middles
    dr, dc = r - m, c - m
    return max(abs(dr), abs(dc)) if dr * dc <= 0 else abs(dr) + abs(dc)


def chance_odds(n: int, cells) -> list[float]:
    """Probability of each cell under the ring weights, renormalised over the empty cells.

    Occupied cells get 0. The list has n*n entries and sums to 1 while any cell is empty.
    """
    weights = []
    for i, v in enumerate(cells):
        if v != EMPTY:
            weights.append(0)
        else:
            weights.append(CHANCE_RING_WEIGHTS[min(hex_ring(n, i), len(CHANCE_RING_WEIGHTS) - 1)])
    total = sum(weights)
    return [w / total for w in weights]


def chance_move(n: int, cells, rng: random.Random) -> int:
    """Sample one empty cell from chance_odds. Swap is never chosen: the chance player only places stones."""
    return rng.choices(range(n * n), weights=chance_odds(n, cells))[0]


def _mcts_move(board: Board, agent: str, sims: int, seconds, seed):
    """Run the search for the mover. Returns (cell, search result)."""
    res = search(board.n, board.cells, board.to_move, sims=sims, seconds=seconds,
                 rave=(agent == "rave"), seed=seed)
    return res["move"], res


def choose(board: Board, agent: str = "rave", *, level: int = 2, sims: int | None = None,
           seconds: float | None = None, seed: int | None = None) -> tuple[int, dict]:
    """The agent's move for `board`, with an analysis dict explaining the choice.

    At the second player's first turn with the swap rule on, the agent compares two searches: one for
    staying (the second player plays on), one for swapping (the second player takes the stone and the
    first player moves next). It swaps when its win chance is higher after the swap.
    """
    if agent not in AGENTS:
        raise ValueError(f"unknown agent {agent!r}")
    rng = random.Random(seed)
    if agent in ("shortest", "random"):
        cells = list(board.cells)
        fn = shortest_path_move if agent == "shortest" else random_move
        move = fn(board.n, cells, board.to_move, rng)
        return move, {"agent": agent, "sims": 0, "playouts": 0, "seconds": 0.0}
    if agent == "chance":
        # No search and no swap: the odds are the whole decision, and the analysis carries them.
        cells = list(board.cells)
        move = chance_move(board.n, cells, rng)
        return move, {"agent": agent, "sims": 0, "playouts": 0, "seconds": 0.0,
                      "odds": [round(p, 4) for p in chance_odds(board.n, cells)], "swap": None}

    budget = LEVELS[level]
    sims = budget["sims"] if sims is None else sims
    seconds = budget["seconds"] if seconds is None else seconds
    swap_note = None
    if board.swap_rule and len(board.moves) == 1 and board.to_move == ACROSS:
        # The second player's first turn: estimate win chance for staying and for swapping.
        stay = search(board.n, board.cells, board.to_move, sims=sims, seconds=seconds,
                      rave=(agent == "rave"), seed=rng.randrange(2**32))
        swapped = list(board.cells)
        first = board.moves[0]
        swapped[first] = other(swapped[first])
        after = search(board.n, swapped, DOWN, sims=sims, seconds=seconds,
                       rave=(agent == "rave"), seed=rng.randrange(2**32))
        win_stay = stay["value"]
        win_swap = round(1 - after["value"], 4)  # `after` is from DOWN's view, to move after the swap
        swap_note = {"stay": win_stay, "swap": win_swap, "chose": "swap" if win_swap > win_stay else "play"}
        if win_swap > win_stay:
            return SWAP, {"agent": agent, "sims": stay["sims"] + after["sims"], "playouts":
                          stay["playouts"] + after["playouts"], "seconds": stay["seconds"] + after["seconds"],
                          "swap": swap_note, "value": win_swap, "pv": [], "visits": None,
                          "win_rate": None, "rave_rate": None, "rave": agent == "rave"}
        res = stay
        move = stay["move"]
    else:
        move, res = _mcts_move(board, agent, sims, seconds, rng.randrange(2**32))
    analysis = dict(res)
    analysis["agent"] = agent
    analysis["swap"] = swap_note
    return move, analysis


def player(name: str, sims: int | None = None):
    """A plain player for the benchmark: (n, cells, to_move, rng) -> cell, no swap, no time cap."""
    if name in ("rave", "uct"):
        budget = sims if sims is not None else LEVELS[2]["sims"]

        def play(n, cells, to_move, rng):
            res = search(n, cells, to_move, sims=budget, rave=(name == "rave"), seed=rng.randrange(2**32))
            return res["move"]
        return play
    if name == "shortest":
        return lambda n, cells, to_move, rng: shortest_path_move(n, list(cells), to_move, rng)
    if name == "random":
        return lambda n, cells, to_move, rng: random_move(n, cells, to_move, rng)
    if name == "chance":
        return lambda n, cells, to_move, rng: chance_move(n, cells, rng)
    raise ValueError(f"unknown agent {name!r}")


def describe_move(n: int, move: int, analysis: dict) -> str:
    """One line for the CLI: the move, and the visits and win rate it earned."""
    if move == SWAP:
        s = analysis.get("swap") or {}
        return f"swap (win chance {s.get('swap')} after swapping, {s.get('stay')} if it stays)"
    text = label(n, move)
    if analysis.get("agent") == "chance":
        return f"{text}  (fixed odds {100 * analysis['odds'][move]:.1f}%)"
    if analysis.get("visits"):
        v = analysis["visits"][move]
        wr = analysis["win_rate"][move]
        text += f"  ({v} visits, {round(100 * wr)}% win)" if wr is not None else f"  ({v} visits)"
    return text
