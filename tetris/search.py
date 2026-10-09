"""Placement search: try every rotation and column for the falling piece, score each result, take the best.

A placement is a rotation index, a column, and the piece's resulting board. The piece is dropped
straight down from the spawn row, which is how Dellacherie-style bots assume it moves. Sliding a piece
sideways around overhangs is ignored: the search only asks which resting places are reachable in
principle. On an empty board the counts are 9 for O, 17 for I, S and Z, and 34 for T, J and L.

With one-piece lookahead the search looks at the next (preview) piece too. For each of the best few
one-piece candidates (top_k, a beam), it finds the best placement of the preview piece on the resulting
board, and adds the two scores. This costs about top_k times more than a greedy move, so top_k is kept
small.
"""

from __future__ import annotations

from dataclasses import dataclass

from .board import W, drop_row, lock
from .features import dot, placement_features
from .pieces import ROTATIONS

# Score given to a candidate after which the preview piece has no legal placement (a forced game over).
GAME_OVER_SCORE = -1e9


@dataclass(frozen=True)
class Move:
    """One legal placement of one piece, with everything the UI and the game need."""

    piece: str
    rot: int  # index into ROTATIONS[piece]
    x: int  # bottom-left column of the piece's box
    y: int  # bottom row of the piece's box
    lines: int
    features: tuple
    score: float  # one-piece score: weights . features
    board: tuple  # the board after the piece locks and full rows clear
    total: float  # the score used for the choice: equals score, or score plus the preview's best with lookahead


def legal_moves(board, piece: str, weights) -> list[Move]:
    """Every resting placement of `piece` on `board`, scored with `weights`."""
    out: list[Move] = []
    for ri, rot in enumerate(ROTATIONS[piece]):
        for px in range(W - rot.width + 1):
            py = drop_row(board, rot, px)
            if py is None:
                continue  # the spawn position in this column is already blocked
            new_board, lines, eroded, _ = lock(board, rot, px, py)
            feats = placement_features(rot, py, lines, eroded, new_board)
            s = dot(weights, feats)
            out.append(Move(piece, ri, px, py, lines, feats, s, new_board, s))
    return out


def best_move(board, piece: str, weights, preview: str | None = None, top_k: int = 6):
    """Choose a placement. Returns (chosen Move or None, every one-piece candidate).

    None means no placement is legal and the game is over. The candidate list holds one-piece scores
    (what the UI draws as ghosts). With a preview, the chosen Move carries its lookahead value in `total`.
    """
    moves = legal_moves(board, piece, weights)
    if not moves:
        return None, moves
    if preview is None:
        return max(moves, key=lambda m: m.score), moves

    ranked = sorted(moves, key=lambda m: -m.score)[:top_k]
    best = None
    for m in ranked:
        nxt = legal_moves(m.board, preview, weights)
        bonus = max(n.score for n in nxt) if nxt else GAME_OVER_SCORE
        total = m.score + bonus
        # The candidate's total replaces its one-piece score for the ranking the UI draws.
        m2 = Move(m.piece, m.rot, m.x, m.y, m.lines, m.features, m.score, m.board, total)
        if best is None or total > best.total:
            best = m2
    return best, moves
