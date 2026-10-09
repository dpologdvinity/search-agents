"""Tablebase files and lookups: what the solved table says about a position, and how to play it.

A table is one uint16 per (white king, white piece, black king, side to move) tuple: 2 * 64**3 = 524,288
entries, 1 MiB raw, which compresses to a few hundred kilobytes. Codes (see retro.py):

    ILLEGAL (65535)  the tuple is not a position that can occur
    DRAW    (65534)  neither side can force mate
    otherwise        the distance to mate in plies. An odd value means the side to move wins in that many
                     plies; an even value means the side to move is lost and is mated in that many plies.

Lookups are array indexing, so the server answers without any search. Each file is loaded on first use.

The policy for playing: a winning side picks the move with the smallest distance to mate (win fast), a
losing side picks the largest (resist as long as possible), and a drawn position stays drawn. Ranking a move
by the mover's own view (win in d beats a draw, which beats a loss in d) gives both behaviours from one rule.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .retro import DRAW, ILLEGAL, N_CODES, index, position, solve
from .rules import BLACK, WHITE, apply_move, in_check, is_legal, legal_moves, move_name, square_name, to_fen

DATA = Path(__file__).resolve().parent / "data"


def file_name(piece: str) -> str:
    """Data file for king and `piece` against a lone king, e.g. kqk.npz."""
    return f"k{piece.lower()}k.npz"


@dataclass(frozen=True)
class MoveInfo:
    """One legal move, annotated with its result as the mover sees it.

    outcome is 'win', 'draw' or 'loss' for the side that moved; plies is the distance to mate from the
    position after the move, counted from the mover's own move (1 = mate in one), or None for a draw.
    check says the move leaves the other side in check (mate implies check); the chance opponent reads it.
    """

    move: tuple[str, int, int]
    san: str
    uci: str
    outcome: str
    plies: int | None
    mate: bool  # the move gives checkmate
    check: bool = False  # the move gives check

    @property
    def moves(self) -> int | None:
        """Mate in this many of the mover's moves, or None."""
        return None if self.plies is None else (self.plies + 1) // 2


def describe(code: int) -> tuple[str, int | None]:
    """(outcome, plies) for a stored code, from the side to move's point of view."""
    if code == ILLEGAL:
        return "illegal", None
    if code == DRAW:
        return "draw", None
    return ("win", code) if code % 2 else ("loss", code)


def rank_key(info: MoveInfo) -> tuple:
    """Sort key: higher is better for the mover. Wins prefer fewer plies, losses prefer more."""
    if info.outcome == "win":
        return (2, -info.plies)
    if info.outcome == "draw":
        return (1, 0)
    return (0, info.plies)


class Tablebase:
    """A solved endgame: the codes array plus helpers that read it."""

    def __init__(self, piece: str, codes: np.ndarray, build_seconds: float):
        self.piece = piece
        self.codes = codes
        self.build_seconds = build_seconds
        self._summary = None

    @property
    def name(self) -> str:
        return f"K{self.piece}K"

    def value(self, pos) -> int:
        """Stored code of a position (side to move included)."""
        return int(self.codes[index(pos)])

    def result(self, pos) -> tuple[str, int | None]:
        """(outcome, plies) for the side to move in pos."""
        return describe(self.value(pos))

    def move_infos(self, pos) -> list[MoveInfo]:
        """Every legal move from pos, each with its outcome and distance to mate."""
        infos = []
        for move in legal_moves(self.piece, pos):
            after = apply_move(pos, move)
            if after is None:
                # The black king took the piece: king against king.
                outcome, plies, mate, check = "draw", None, False, False
            else:
                check = in_check(self.piece, after)
                # The codes are the opponent's view after the move, so flip them: a loss for them is a win for us.
                # A code of 0 is checkmate, the opponent to move and mated.
                outcome, theirs = describe(self.value(after))
                plies = None
                if outcome != "draw":
                    plies = theirs + 1
                    outcome = "win" if outcome == "loss" else "loss"
                mate = theirs == 0
            san = move_name(self.piece, pos, move, after)
            if mate:
                san += "#"
            uci = square_name(move[1]) + square_name(move[2])
            infos.append(MoveInfo(move, san, uci, outcome, plies, mate, check))
        return infos

    def best_move(self, pos) -> MoveInfo | None:
        """The move this table plays for the side to move: fastest win, longest defence, or any drawing move."""
        infos = self.move_infos(pos)
        return max(infos, key=rank_key) if infos else None

    def king_row(self, wk: int, wp: int, stm: int) -> np.ndarray:
        """Codes for the black king on each of the 64 squares, with the white king and piece fixed.

        This is the heat map's data. Squares where the black king would overlap a piece or touch the
        white king are ILLEGAL.
        """
        base = (wk * 64 + wp) * 64
        return self.codes[(base + np.arange(64)) * 2 + stm]

    def summary(self) -> dict:
        """Counts and distance-to-mate histograms over every legal position. Computed once, then cached."""
        if self._summary is None:
            self._summary = _summarise(self)
        return self._summary

    def random_position(self, rng, stm: int | None = None, want: str = "any", max_tries: int = 200_000):
        """A random legal position with the given side to move, or ValueError if none is found.

        want='strong_wins' keeps only positions the side with the piece wins, 'draw' keeps only drawn ones,
        and 'any' takes anything legal. This is rejection sampling: about 70% of the index space is legal
        (368,452 of 524,288 for KQK), and the bound stops impossible requests (a drawn position with White to
        move, say) from looping forever.
        """
        for _ in range(max_tries):
            wk, wp, bk = rng.randrange(64), rng.randrange(64), rng.randrange(64)
            side = rng.randrange(2) if stm is None else stm
            pos = (wk, wp, bk, side)
            if not is_legal(self.piece, pos):
                continue
            outcome, _ = self.result(pos)
            if want == "any":
                return pos
            if want == "draw" and outcome == "draw":
                return pos
            if want == "strong_wins":
                # The strong side wins if White is to move and the code is odd (a win for the mover),
                # or Black is to move and the code is even (Black is mated).
                if outcome == "win" and side == WHITE or outcome == "loss" and side == BLACK:
                    return pos
        raise ValueError(f"no {want} position with side {stm} found")


def _summarise(tb: Tablebase) -> dict:
    """Vectorised pass over the codes: counts by side to move and result, and DTM histograms in moves."""
    codes = tb.codes
    idx = np.arange(N_CODES, dtype=np.int64)
    stm = idx & 1
    legal = codes != ILLEGAL
    decided = legal & (codes != DRAW)
    win = decided & (codes % 2 == 1)
    loss = decided & (codes % 2 == 0)
    draw = legal & (codes == DRAW)
    # Histograms in moves: a win in d plies is mate in (d + 1) // 2 moves; a loss in d plies is mated in d // 2.
    win_hist = np.bincount((codes[win].astype(np.int64) + 1) // 2, minlength=1)
    loss_hist = np.bincount(codes[loss].astype(np.int64) // 2, minlength=1)
    longest = int(np.argmax(np.where(win, codes.astype(np.int64), -1)))
    pos = position(longest)
    return {
        "legal": int(legal.sum()),
        "legal_white_to_move": int((legal & (stm == WHITE)).sum()),
        "legal_black_to_move": int((legal & (stm == BLACK)).sum()),
        "win": int(win.sum()),
        "loss": int(loss.sum()),
        "draw": int(draw.sum()),
        "win_white_to_move": int((win & (stm == WHITE)).sum()),
        "win_black_to_move": int((win & (stm == BLACK)).sum()),
        "loss_white_to_move": int((loss & (stm == WHITE)).sum()),
        "loss_black_to_move": int((loss & (stm == BLACK)).sum()),
        "draw_white_to_move": int((draw & (stm == WHITE)).sum()),
        "draw_black_to_move": int((draw & (stm == BLACK)).sum()),
        "max_win_moves": int(len(win_hist) - 1),
        "max_loss_moves": int(len(loss_hist) - 1),
        "max_win_plies": int(codes[win].max()) if win.any() else None,
        "max_loss_plies": int(codes[loss].max()) if loss.any() else None,
        "win_histogram": [int(x) for x in win_hist],  # index = mate in that many moves (win for the side to move)
        "loss_histogram": [int(x) for x in loss_hist],  # index = mated in that many moves
        "longest_win_example": to_fen(tb.piece, pos),
        "longest_win_example_moves": int((int(codes[longest]) + 1) // 2),
    }


@functools.cache
def load(piece: str) -> Tablebase:
    """The solved table for `piece`, read from the committed data file on first use."""
    path = DATA / file_name(piece)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run python -m endgame build")
    with np.load(path) as f:
        return Tablebase(piece, f["codes"].astype(np.uint16), float(f["build_seconds"]))


def build(piece: str, out: Path | None = None) -> Tablebase:
    """Solve the endgame from scratch and write it to the data directory (or `out`)."""
    codes, seconds, _ = solve(piece)
    tb = Tablebase(piece, codes, seconds)
    path = out or DATA / file_name(piece)
    np.savez_compressed(path, codes=codes, build_seconds=np.float64(seconds))
    return tb
