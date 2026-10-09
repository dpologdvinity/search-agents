"""Board rules, plain minimax, and an alpha-beta search that records the tree it explored.

A board is a 9-character string, row by row: '.' is empty, 'X' and 'O' are the marks. X moves first.
Values are from X's point of view: 1 means X wins with perfect play, 0 a draw, -1 an O win.

The search visits children in square order 0..8 and records the first two plies (the agent's candidate
moves and the replies to them), so the page can draw the tree. Records carry the running counts of
positions searched and moves pruned, so the page can show the counters growing as the tree is drawn.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
EMPTY_BOARD = "........."
# A sentinel outside the value range -1..1, so the first child always improves on it.
INF = 2
# Records are kept for plies 0 and 1 only: the agent's moves (ply 1) and the replies to them (ply 2).
MAX_PLY = 2


def winner(board: str) -> str:
    """Return 'X' or 'O' if that side has three in a row, else ''."""
    for a, b, c in LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    return ""


def legal_moves(board: str) -> list[int]:
    """Empty squares in index order (the order the search tries them)."""
    return [i for i, ch in enumerate(board) if ch == "."]


def other(turn: str) -> str:
    return "O" if turn == "X" else "X"


def minimax(board: str, turn: str) -> int:
    """Plain minimax with no pruning. The reference the alpha-beta search is checked against."""
    w = winner(board)
    if w:
        return 1 if w == "X" else -1
    moves = legal_moves(board)
    if not moves:
        return 0
    values = [minimax(board[:m] + turn + board[m + 1 :], other(turn)) for m in moves]
    return max(values) if turn == "X" else min(values)


@dataclass
class Search:
    """Result of searching one position.

    move: the best square for the side to move (None when the game is already over).
    value: the minimax value from X's view.
    searched: positions visited (every call into the search, leaves included).
    pruned: legal moves skipped because a cutoff made them unnecessary.
    tree: the recorded plies, a list of records {m, v, b, pruned, kids, s, p}. v is the record's value
          (None for pruned moves); b is '' when v is exact, 'hi' when the true value is at most v, 'lo'
          when it is at least v (the search stopped early, so v is only a bound); s and p are the running
          searched and pruned counts when the record was finished.
    """

    move: int | None
    value: int
    searched: int
    pruned: int
    tree: list[dict] = field(default_factory=list)


def _node(board: str, turn: str, alpha: int, beta: int, ply: int, kids: list[dict], stats: list[int]):
    """Alpha-beta over one node. Returns (value, exact, best_move_or_None).

    kids receives the records for this node's children while ply < MAX_PLY. stats is [searched, pruned].
    """
    stats[0] += 1
    w = winner(board)
    if w:
        return (1 if w == "X" else -1), True, None
    moves = legal_moves(board)
    if not moves:
        return 0, True, None
    keep = ply < MAX_PLY
    maximize = turn == "X"
    a0, b0 = alpha, beta
    v = -INF if maximize else INF
    best = None
    for k, m in enumerate(moves):
        child = board[:m] + turn + board[m + 1 :]
        rec = {"m": m, "v": None, "b": "", "pruned": False, "kids": [], "s": 0, "p": 0}
        cv, cexact, _ = _node(child, other(turn), alpha, beta, ply + 1, rec["kids"], stats)
        rec["v"] = cv
        # The child was searched with the window (alpha, beta) as it stood at the call. A value at or
        # below alpha, or at or above beta, is only a bound.
        rec["b"] = "" if cexact else ("hi" if cv <= alpha else "lo")
        rec["s"], rec["p"] = stats
        if keep:
            kids.append(rec)
        if (maximize and cv > v) or (not maximize and cv < v):
            v, best = cv, m
        if maximize:
            alpha = max(alpha, v)
        else:
            beta = min(beta, v)
        if alpha >= beta:
            # Cutoff: the remaining siblings can not change this node's value, so they are never searched.
            rest = moves[k + 1 :]
            stats[1] += len(rest)
            if keep:
                for r in rest:
                    kids.append({"m": r, "v": None, "b": "", "pruned": True, "kids": [], "s": stats[0], "p": stats[1]})
            break
    return v, a0 < v < b0, best


def search(board: str, turn: str) -> Search:
    """Alpha-beta minimax for the side to move, with the first two plies recorded."""
    if winner(board) or not legal_moves(board):
        return Search(None, _node(board, turn, -INF, INF, 0, [], [0, 0])[0], 1, 0, [])
    stats = [0, 0]
    tree: list[dict] = []
    value, _, move = _node(board, turn, -INF, INF, 0, tree, stats)
    return Search(move, value, stats[0], stats[1], tree)


def reachable_positions(include_finished: bool = False) -> list[str]:
    """Every board reachable from the empty board that is still in play (no win, not full), sorted.

    With include_finished, the boards that end the game (a win or a full board) are included too.
    """
    seen = {EMPTY_BOARD}
    queue = deque([EMPTY_BOARD])
    while queue:
        board = queue.popleft()
        if winner(board):
            continue
        turn = "X" if board.count("X") == board.count("O") else "O"
        for m in legal_moves(board):
            nxt = board[:m] + turn + board[m + 1 :]
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    if include_finished:
        return sorted(seen)
    return sorted(b for b in seen if not winner(b) and "." in b)
