"""Monte Carlo tree search for Hex: plain UCT, and RAVE (rapid action value estimation).

Both variants run the same four steps per simulation:

1. Selection: from the root, pick children by a score until reaching a node with an untried move.
2. Expansion: add one child for an untried move.
3. Simulation: finish the game at random (board.fill_playout). This is the only place the win is decided.
4. Backpropagation: add the result to every node on the path.

Plain UCT scores a child by its own win rate plus an exploration bonus, sqrt(ln N / n).

RAVE (Gelly and Silver) adds AMAF statistics, short for all-moves-as-first. In a playout, a cell that a
player took after some node is credited to that node's move list as if the player had made it first,
whether or not it was a child. Each node keeps, per cell, how often that cell was played by the node's
player in a simulation through the node and how often that player then won. These estimates are noisy
but have far more samples early on. The child score blends the two estimates:

    Q = (1 - beta) * Q_uct + beta * Q_rave,   beta = sqrt(k / (3 n + k))

where n is the child's visit count. beta is near 1 while the child has few visits (trust AMAF) and falls
toward 0 as visits grow (trust the exact estimate). k sets how fast that happens. RAVE also orders
expansion: an untried move with the best AMAF rate is expanded first, instead of a random one.
"""

from __future__ import annotations

import math
import random
import time

from .board import EMPTY, connect, fill_playout, other

EXPLORE = 0.6  # exploration constant for win rates in [0, 1]
RAVE_K = 300.0  # the equivalence parameter k; beta = sqrt(k / (3n + k)) is 1/2 at n = k = 300 visits


class _Node:
    """One position in the tree. `mover` made the move that reached it; `to_move` chooses the next move."""

    __slots__ = ("cell", "mover", "to_move", "children", "untried", "visits", "wins",
                 "rave_n", "rave_w", "terminal")

    def __init__(self, n2: int, cell: int, mover: int, to_move: int, untried: list[int], terminal: bool):
        self.cell = cell
        self.mover = mover
        self.to_move = to_move
        self.children: list[_Node] = []
        self.untried = untried
        self.visits = 0
        self.wins = 0          # wins for `mover`
        self.rave_n = [0] * n2  # AMAF: times `to_move` played each cell in a simulation through this node
        self.rave_w = [0] * n2  # ...and how often `to_move` then won
        self.terminal = terminal


def _select(node: _Node, c: float, k: float, rave: bool) -> _Node:
    """The child with the best blended score. Q is from the chooser's point of view, like the stats."""
    log_n = math.log(node.visits)
    best, best_v = None, -1.0
    for ch in node.children:
        n = ch.visits
        q = ch.wins / n
        if rave and node.rave_n[ch.cell]:
            q_rave = node.rave_w[ch.cell] / node.rave_n[ch.cell]
            beta = math.sqrt(k / (3.0 * n + k))
            q = (1.0 - beta) * q + beta * q_rave
        v = q + c * math.sqrt(log_n / n)
        if v > best_v:
            best, best_v = ch, v
    return best


def _pick_untried(node: _Node, rave: bool, rng: random.Random) -> int:
    """Index in node.untried of the move to expand. UCT picks at random; RAVE picks the best AMAF rate."""
    if not rave:
        return rng.randrange(len(node.untried))
    # Laplace-smoothed rate so unseen moves start at 1/2; random tie-break keeps the order unbiased.
    best_i, best_key = 0, None
    for i, m in enumerate(node.untried):
        key = ((node.rave_w[m] + 1) / (node.rave_n[m] + 2), rng.random())
        if best_key is None or key > best_key:
            best_i, best_key = i, key
    return best_i


def search(n: int, cells, to_move: int, *, sims: int = 1000, seconds: float | None = None,
           rave: bool = True, seed: int | None = None, c: float = EXPLORE, k: float = RAVE_K) -> dict:
    """Search the position `cells` (0/1/2 per cell, 1 = DOWN) for `to_move`.

    Stops after `sims` simulations or `seconds` of wall time, whichever comes first. Returns the
    move with the most visits (the robust choice), the visit count and win rate of every root child,
    the RAVE rate of every root move, the win probability the search gives `to_move`, and the
    principal variation (the most-visited line from the root).
    """
    rng = random.Random(seed)
    n2 = n * n
    board = list(cells)  # the stones along the current tree path are placed here and taken back afterwards
    root = _Node(n2, -1, 0, to_move, [i for i in range(n2) if board[i] == EMPTY], False)
    start = time.perf_counter()
    done = nodes = playouts = 0
    while done < sims:
        if seconds is not None and done % 32 == 0 and time.perf_counter() - start > seconds:
            break
        node = root
        path = [root]   # the nodes from the root to the leaf of this simulation
        moves = []      # the cells played along the tree part of this simulation
        # 1. Selection: descend while the node has no untried moves and does have children.
        while not node.untried and node.children:
            node = _select(node, c, k, rave)
            board[node.cell] = node.mover
            path.append(node)
            moves.append(node.cell)
        # 2. Expansion: add one child for an untried move.
        if node.untried:
            cell = node.untried.pop(_pick_untried(node, rave, rng))
            board[cell] = node.to_move
            mover = node.to_move
            done_now = connect(n, board, mover) is not None  # did this move connect the mover?
            untried = [] if done_now else [i for i in range(n2) if board[i] == EMPTY]
            child = _Node(n2, cell, mover, other(mover), untried, done_now)
            node.children.append(child)
            nodes += 1
            node = child
            path.append(node)
            moves.append(cell)
        # 3. Simulation: a finished game has a known winner; otherwise fill the board at random.
        if node.terminal:
            winner, order = node.mover, []
        else:
            winner, order = fill_playout(n, board[:], node.to_move, rng)
            playouts += 1
        # 4. Backpropagation: each node's visit and win counts, then the AMAF counts.
        for nd in path:
            nd.visits += 1
            if nd.mover == winner:
                nd.wins += 1
        if rave:
            # Every cell played after path[d] by path[d].to_move, at positions d, d+2, ... of this sequence,
            # gets one AMAF sample. Moves alternate with no swaps inside the search, so the parity is exact.
            seq = moves + order
            for depth, nd in enumerate(path):
                won = winner == nd.to_move
                rn, rw = nd.rave_n, nd.rave_w
                for j in range(depth, len(seq), 2):
                    m = seq[j]
                    rn[m] += 1
                    if won:
                        rw[m] += 1
        # Take the simulation's stones back off the board, leaving it as it was at the root.
        for m in moves:
            board[m] = EMPTY
        done += 1
    elapsed = time.perf_counter() - start

    if not root.children:  # only possible with sims=0 on a finished board
        return {"move": None, "sims": 0, "playouts": 0, "nodes": 0, "seconds": elapsed}
    best = max(root.children, key=lambda ch: ch.visits)
    visits = [0] * n2
    win_rate = [None] * n2
    for ch in root.children:
        visits[ch.cell] = ch.visits
        win_rate[ch.cell] = round(ch.wins / ch.visits, 4) if ch.visits else None
    rave_rate = [round(root.rave_w[i] / root.rave_n[i], 4) if root.rave_n[i] else None for i in range(n2)]
    pv = []
    node = best
    while node is not None:
        pv.append(node.cell)
        node = max(node.children, key=lambda ch: ch.visits) if node.children else None
    return {
        "move": best.cell,
        "sims": done,
        "playouts": playouts,  # simulations that ended in a random fill (the rest hit a finished game)
        "nodes": nodes + 1,
        "seconds": elapsed,
        "visits": visits,
        "win_rate": win_rate,
        "rave_rate": rave_rate,
        "value": round(best.wins / best.visits, 4),  # win probability for to_move, from the most-visited move
        "pv": pv,
        "rave": rave,
    }
