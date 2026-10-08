"""Retrograde analysis: build an exact distance-to-mate (DTM) table by working backwards from checkmate.

Forward search asks "what can I do from here?" and needs the whole game tree. Retrograde analysis asks
the reverse question, and that is what makes a tablebase cheap: every position's value is settled by
its successors, and the search starts from the positions that are already decided.

  1. Checkmate (side to move in check, no legal moves) is a loss in 0 plies. Stalemate is a draw.
  2. Every other legal position starts UNKNOWN. Its counter is its number of legal moves.
  3. Process decided positions in order of DTM (a FIFO queue keeps that order, because each new entry
     is one ply deeper than the entry that produced it). For a decided position S, look at each
     predecessor P (a position from which one move reaches S):
       - If S is a loss for its side to move, P's side wins by making that move: P = win, DTM = DTM(S) + 1.
         The first such S gives the shortest win, because S's are processed in DTM order.
       - If S is a win for its side to move, the move P -> S is bad for P's side. Decrement P's counter.
         When the counter reaches 0, every move from P loses, and the last S processed (the deepest, since
         the queue is in order) gives the longest resistance: P = loss, DTM = DTM(S) + 1.
  4. Whatever is still UNKNOWN can never be forced to mate: it is a draw.

Captures are not in the table (king against king is a draw), so a capture counts as a legal move that never
decrements a counter. That keeps the defending side from being declared lost by mistake.

The result is stored as uint16 codes per position (see tablebase.py for the encoding). The parity of the
code gives the outcome: odd = the side to move wins, even = it loses.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from .rules import BLACK, WHITE, apply_move, in_check, is_legal, legal_moves, predecessors

# Codes for positions that are not wins or losses. Real DTM values are small (below 100 plies).
UNKNOWN = 65533  # not decided yet; left only while the search runs
DRAW = 65534
ILLEGAL = 65535
N_CODES = 64 * 64 * 64 * 2


def index(pos) -> int:
    """Position to table index: (wk * 64 + wp) * 64 + bk, then times 2 plus the side to move."""
    wk, wp, bk, stm = pos
    return ((wk * 64 + wp) * 64 + bk) * 2 + stm


def position(i: int):
    """Inverse of index()."""
    stm = i & 1
    i >>= 1
    bk = i & 63
    i >>= 6
    return (i >> 6, i & 63, bk, stm)


def solve(piece: str) -> tuple[np.ndarray, float, int]:
    """Build the table for king and `piece` against a lone king.

    Returns (codes, seconds, legal positions). codes is a uint16 array indexed by index(pos).
    """
    t0 = time.perf_counter()
    values = [ILLEGAL] * N_CODES
    count = [0] * N_CODES  # legal moves left to resolve, per position (captures included)
    queue: deque = deque()
    legal = 0

    # Pass 1: find every legal position, count its moves, and seed the checkmates.
    for wk in range(64):
        for wp in range(64):
            if wp == wk:
                continue
            for bk in range(64):
                if bk == wk or bk == wp:
                    continue
                for stm in (WHITE, BLACK):
                    pos = (wk, wp, bk, stm)
                    if not is_legal(piece, pos):
                        continue
                    legal += 1
                    i = index(pos)
                    moves = legal_moves(piece, pos)
                    if moves:
                        values[i] = UNKNOWN
                        count[i] = len(moves)
                    elif in_check(piece, pos):
                        values[i] = 0  # checkmate: the side to move has lost, zero plies from now
                        queue.append(pos)
                    else:
                        values[i] = DRAW  # stalemate

    # Pass 2: backward propagation in DTM order.
    while queue:
        s = queue.popleft()
        d = values[index(s)]
        s_is_loss = d % 2 == 0  # the side to move in s is mated (even DTM)
        for p in predecessors(piece, s):
            j = index(p)
            if values[j] != UNKNOWN:
                continue  # already decided, or a draw we will not revisit
            if s_is_loss:
                values[j] = d + 1  # p's side makes the move into s and mates
                queue.append(p)
            else:
                count[j] -= 1
                if count[j] == 0:  # every move from p now loses; the last one is the longest
                    values[j] = d + 1
                    queue.append(p)

    # Pass 3: what is left cannot be forced either way.
    codes = np.array([DRAW if v == UNKNOWN else v for v in values], dtype=np.uint16)
    return codes, time.perf_counter() - t0, legal


def legal_positions(piece: str):
    """Yield every legal position for king and `piece` against a lone king, both sides to move."""
    for wk in range(64):
        for wp in range(64):
            if wp == wk:
                continue
            for bk in range(64):
                if bk == wk or bk == wp:
                    continue
                for stm in (WHITE, BLACK):
                    pos = (wk, wp, bk, stm)
                    if is_legal(piece, pos):
                        yield pos


def check_positions(piece: str, codes: np.ndarray, positions) -> list[str]:
    """Check table entries against the definition, one position at a time, without using the solver.

    For a position P with side to move X and value v:
      - a win in d plies (v = d odd) needs a move to a loss for the opponent in d - 1 plies, and no move
        to a loss for the opponent that is faster;
      - a loss in d plies (v = d even) needs every move to be a win for the opponent, none drawn, with
        the slowest of them at d - 1 plies;
      - a draw needs at least one drawing move and no winning move;
      - a checkmate is a loss in 0, and a stalemate is a draw.
    Returns a list of problems (empty when every position passes).
    """
    problems = []
    for pos in positions:
        v = int(codes[index(pos)])
        moves = legal_moves(piece, pos)
        if not moves:
            want = 0 if in_check(piece, pos) else DRAW
            if v != want:
                problems.append(f"terminal {pos}: stored {v}, expected {want}")
            continue
        # The mover's view of each move: ("win", plies), ("draw", None) or ("loss", plies).
        results = []
        for m in moves:
            s = apply_move(pos, m)
            if s is None:
                results.append(("draw", None))
                continue
            w = int(codes[index(s)])
            if w == DRAW:
                results.append(("draw", None))
            elif w % 2 == 0:
                results.append(("win", w + 1))
            else:
                results.append(("loss", w + 1))
        wins = [d for k, d in results if k == "win"]
        losses = [d for k, d in results if k == "loss"]
        draws = sum(1 for k, _ in results if k == "draw")
        if v == DRAW:
            if wins or not draws:
                problems.append(f"draw {pos}: has a winning move or no drawing move")
        elif v % 2 == 1:
            if not wins or min(wins) != v:
                problems.append(f"win {pos} in {v}: fastest winning move is {min(wins) if wins else None}")
        else:
            if draws or wins or not losses or max(losses) != v:
                problems.append(f"loss {pos} in {v}: slowest reply is {max(losses) if losses else None}")
    return problems
