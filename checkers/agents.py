"""The checkers agents and their strength levels, shared by the terminal game, the web API and the match runner.

make_agent() builds a searcher from its registry name and level. choose() runs one agent
on one position and returns its move plus a summary of the search (depth reached,
positions examined, branches pruned or rollouts run, and a score for every root move),
already in JSON-friendly form.
"""

from __future__ import annotations

import math
import time
import zlib

from .board import RED, WHITE, geometry, legal_moves, notation
from .chance import Chance, odds_text
from .mcts import MCTS
from .search import WIN, AlphaBeta, Greedy, Minimax, RandomMove

# Level -> budget. Alpha-beta gets a time limit (iterative deepening goes as deep as it
# can in that time). Match mode counts nodes instead, so a match replays the same way on
# any machine (about 10,000-20,000 nodes a second here, so the two budgets are alike).
ALPHABETA_SECONDS = {1: 0.2, 2: 0.8, 3: 2.0}
ALPHABETA_NODES = {1: 3_000, 2: 12_000, 3: 40_000}
# Plain minimax has no time limit, so its depth is fixed per (board size, forced captures).
# Optional captures add many moves per position, so they need shallower depths. The
# depths were chosen so the slowest of three midgame positions takes about 3 s or less
# here: 8x8 optional depth 4 took 1.1 s and 12x12 forced depth 4 took 2.9 s, so the 12x12
# forced table stops at 3. Optional captures on 12x12 at depth 3 took 7 s, so they stop at 2.
MINIMAX_DEPTH = {
    (8, True): {1: 2, 2: 4, 3: 5},
    (8, False): {1: 2, 2: 3, 3: 4},
    (10, True): {1: 2, 2: 3, 3: 4},
    (10, False): {1: 2, 2: 2, 3: 3},
    (12, True): {1: 2, 2: 3, 3: 3},
    (12, False): {1: 1, 2: 2, 3: 2},
}
# MCTS is iteration-budgeted at every level, so its answer is deterministic. The counts
# fall with board size because a rollout's legal-move generation costs more on bigger
# boards: about 3.5 ms an iteration on 12x12, so level 3 stays near 2 s there.
MCTS_ITERATIONS = {
    8: {1: 300, 2: 1000, 3: 2000},
    10: {1: 200, 2: 500, 3: 1000},
    12: {1: 150, 2: 350, 3: 500},
}

AGENTS = ("alphabeta", "minimax", "mcts", "greedy", "random", "chance")
DESCRIPTIONS = {
    "alphabeta": (
        "Alpha-beta search: assumes you will always answer with the reply that is worst for it, and skips "
        "lines it can prove are worse than one already found. Iterative deepening, a transposition table, "
        "and best-move-first ordering let it look deeper in the same time."
    ),
    "minimax": (
        "Plain minimax to a fixed depth: the same reasoning, but it examines every line. Compare its node "
        "count with alpha-beta's to see how much pruning saves."
    ),
    "mcts": (
        "Monte Carlo tree search (UCT): plays random games from the positions it considers, cut short after "
        "a few plies and scored by the evaluation, and favours the moves that win the most of them. It "
        "reports a win rate rather than a score, and it is seeded, so the same position always gets the same answer."
    ),
    "greedy": (
        "Greedy: looks one move ahead and plays whatever leaves the position that scores best for it. It "
        "does not see your reply, so it walks into traps that alpha-beta avoids."
    ),
    "random": "Random: plays a uniformly chosen legal move. The baseline the other agents should beat.",
    "chance": (
        f"Chance (fixed odds): no search and no reasoning. Each legal move gets a fixed weight by its type, "
        f"{odds_text()}; the weights are renormalised over the legal moves and one is sampled. "
        "It is the checkers counterpart of the 2048 tile spawner."
    ),
}


def make_agent(agent: str, level: int, size: int = 8, *, forced: bool = True, seed: int = 0,
               by_nodes: bool = False):
    """A searcher for this agent at this level on a size x size board.

    by_nodes swaps alpha-beta's time limit for a node budget, so the match runner gets
    the same answers on any machine. The other agents already have deterministic budgets.
    """
    if agent == "alphabeta":
        if by_nodes:
            return AlphaBeta(max_seconds=math.inf, max_nodes=ALPHABETA_NODES[level], forced=forced)
        return AlphaBeta(max_seconds=ALPHABETA_SECONDS[level], forced=forced)
    if agent == "minimax":
        return Minimax(MINIMAX_DEPTH[(size, forced)][level], forced=forced)
    if agent == "mcts":
        return MCTS(iterations=MCTS_ITERATIONS[size][level], seed=seed, policy="capture", forced=forced)
    if agent == "greedy":
        return Greedy(forced=forced)
    if agent == "random":
        return RandomMove(seed=seed, forced=forced)
    if agent == "chance":
        return Chance(seed=seed, forced=forced)
    raise ValueError(f"unknown agent {agent!r}; choose from {', '.join(AGENTS)}")


# The names the page and the CLI show for each side. Keeping one function for the wording means both say the same thing.
SIDE_NAMES = {RED: "Red", WHITE: "White"}


def agent_label(agent: str, level: int, size: int = 8, *, forced: bool = True, by_nodes: bool = False) -> str:
    """Name the algorithm and its budget, e.g. "MCTS (1,000 playouts)" or "minimax, depth 4".

    Alpha-beta is named by its time budget, because the depth it reaches changes from one position to
    the next (the analysis shows the depth of each move). by_nodes names its node budget instead, as the
    match runner uses.
    """
    if agent == "alphabeta":
        budget = f"{ALPHABETA_NODES[level]:,} nodes" if by_nodes else f"{ALPHABETA_SECONDS[level]:.1f} s"
        return f"alpha-beta search, iterative deepening, {budget}"
    if agent == "minimax":
        return f"minimax, depth {MINIMAX_DEPTH[(size, forced)][level]}"
    if agent == "mcts":
        return f"MCTS ({MCTS_ITERATIONS[size][level]:,} playouts)"
    if agent == "greedy":
        return "greedy, one ply"
    if agent == "random":
        return "random"
    if agent == "chance":
        return "chance (fixed odds)"
    raise ValueError(f"unknown agent {agent!r}")


def matchup_text(players: dict, levels: dict, size: int = 8, *, forced: bool = True, by_nodes: bool = False) -> str:
    """The line that says who plays, shared by the page and the CLI.

    One AI against a human: "White: alpha-beta search, iterative deepening, 0.8 s".
    Two AIs: "Red: MCTS (1,000 playouts) vs White: alpha-beta search, iterative deepening, 0.8 s".
    players maps RED and WHITE to an agent name or "human"; levels maps them to a strength.
    """
    parts = [f"{SIDE_NAMES[side]}: {agent_label(players[side], levels[side], size, forced=forced, by_nodes=by_nodes)}"
             for side in (RED, WHITE) if players[side] != "human"]
    return " vs ".join(parts) if parts else "human vs human"


def move_json(m) -> dict:
    """A Move as plain lists, plus its standard notation (squares numbered from 1)."""
    return {"path": list(m.path), "captured": list(m.captured), "result": list(m.result), "notation": notation(m)}


def describe(score):
    """Turn a raw search score into something readable.

    Scores within 200 of WIN mean a forced win (or loss) was found; the distance
    from WIN is how many plies away it is. A float is an MCTS win rate, so it is
    reported as a percentage. Anything else is a heuristic score.
    """
    if score is None:
        return None
    if isinstance(score, float):
        return {"result": "rate", "percent": round(100 * score, 1)}
    if score >= WIN - 200:
        return {"result": "win", "plies": WIN - score}
    if score <= -(WIN - 200):
        return {"result": "loss", "plies": WIN + score}
    return {"result": "eval", "score": score}


def _position_seed(board, turn: int) -> int:
    """A seed that depends only on the position, so the same position always gets the same random choices."""
    return zlib.crc32(repr((board, turn)).encode())


def choose(board, turn: int, agent: str = "alphabeta", level: int = 2, *, forced: bool = True,
           seed: int | None = None, by_nodes: bool = False) -> dict:
    """Pick a move for `turn` (+1 red, -1 white) with the named agent at the given level.

    The board size comes from the board itself. Returns {"move", "analysis", "reply"}:
    the chosen move, what the search saw, and the opponent's legal replies (so a client
    doesn't need a second request). seed defaults to a value derived from the position.
    """
    if seed is None:
        seed = _position_seed(board, turn)
    searcher = make_agent(agent, level, geometry(board).n, forced=forced, seed=seed, by_nodes=by_nodes)
    start = time.perf_counter()
    info = searcher.search(board, turn)
    reply_moves = legal_moves(info.move.result, -turn, forced)
    return {
        "move": move_json(info.move),
        "analysis": {
            "agent": agent,
            "depth": info.depth,
            "nodes": info.nodes,
            "cutoffs": info.cutoffs,
            "rollouts": info.rollouts,
            "best": describe(info.score),
            # Root moves best-first; unscored moves go last.
            "root": [{"notation": notation(m), "path": list(m.path), "score": describe(sc)}
                     for m, sc in sorted(info.root, key=lambda x: -(x[1] if x[1] is not None else -WIN))],
            "seconds": round(time.perf_counter() - start, 3),
        },
        "reply": [move_json(m) for m in reply_moves],
    }
