"""Matches between the tablebase and the chance opponent, played with no prompts and no display.

A match is N random winning positions (the side with the piece wins with best play). Every position is
played twice, once with the tablebase as the strong side (White, with the piece) and once with it as the
lone black king, so both seats see the same starting positions and the chance opponent gets the other seat.

play_out is the interactive game's loop without the input: the tablebase plays its own best move, and the
chance opponent picks by the fixed odds. Each game ends by checkmate, by a stalemate or a capture (a draw),
or after MAX_PLIES plies, which is reported as unfinished rather than as a draw.
"""

from __future__ import annotations

import random
from collections import Counter

from . import chance
from .rules import BLACK, WHITE, apply_move, in_check
from .tablebase import Tablebase, load

# Long enough for any forced mate in these tables (the longest is 31 plies in KRK, see results/endgame_stats.json),
# so a tablebase win always finishes under the cap. The cap only stops a chance opponent that holds out in an
# endless line, and those games are counted as unfinished.
MAX_PLIES = 300

SEATS = {"strong": WHITE, "weak": BLACK}
_SIDE = {WHITE: "White", BLACK: "Black"}


def play_out(tb: Tablebase, pos, tb_side: int, rng: random.Random, max_plies: int = MAX_PLIES) -> tuple[str, int]:
    """Play from pos to the end. Returns (result, plies), with the result from the tablebase's side.

    result is 'win', 'loss', 'draw', or 'unfinished' (the cap was reached). Checkmate is recognised as no
    legal move with the side to move in check. A stalemate, or a capture of the piece, is a draw.
    """
    plies = 0
    while plies < max_plies:
        infos = tb.move_infos(pos)
        if not infos:
            if in_check(tb.piece, pos):
                # The side to move is mated, so the side that just moved won.
                winner = 1 - pos[3]
                return ("win" if winner == tb_side else "loss"), plies
            return "draw", plies
        if pos[3] == tb_side:
            info = tb.best_move(pos)
        else:
            info = chance.pick(infos, pos, rng)
        nxt = apply_move(pos, info.move)
        plies += 1
        if nxt is None:
            # The black king took the piece: king against king.
            return "draw", plies
        pos = nxt
    return "unfinished", plies


def run_match(piece: str, games: int, seed: int = 0, seats: tuple[str, ...] = ("strong", "weak")) -> dict:
    """Play `games` random winning positions in each seat and tally the results.

    Position i is drawn from random.Random(seed + i) and the chance opponent's rolls from a second generator
    with the same seed, so a rerun with the same arguments gives the same numbers.
    """
    tb = load(piece)
    report = {
        "piece": piece,
        "name": tb.name,
        "games": games,
        "seed": seed,
        "tablebase": chance.TABLEBASE_LABEL,
        "chance": chance.LABEL,
        "max_plies": MAX_PLIES,
        "seats": [],
    }
    for seat in seats:
        tb_side = SEATS[seat]
        tally = Counter()
        win_plies, loss_plies = [], []
        for i in range(games):
            pos = tb.random_position(random.Random(seed + i), want="strong_wins")
            result, plies = play_out(tb, pos, tb_side, random.Random(seed + i))
            tally[result] += 1
            if result == "win":
                win_plies.append(plies)
            elif result == "loss":
                loss_plies.append(plies)
        report["seats"].append({
            "seat": seat,
            "tablebase_side": _SIDE[tb_side],
            "chance_side": _SIDE[1 - tb_side],
            "wins": tally["win"],
            "draws": tally["draw"],
            "losses": tally["loss"],
            "unfinished": tally["unfinished"],
            "mean_win_moves": _mean_moves(win_plies),
            "mean_loss_moves": _mean_moves(loss_plies),
        })
    return report


def _mean_moves(plies: list[int]) -> float | None:
    """Mean length of these games in moves (two plies make a move), or None when there are none."""
    if not plies:
        return None
    return round(sum((p + 1) // 2 for p in plies) / len(plies), 2)


def match_lines(report: dict) -> list[str]:
    """Text for the CLI: the algorithms, then one line per seat with the tablebase's results."""
    lines = [
        f"{report['name']}: {report['tablebase']} vs {report['chance']}, {report['games']} positions per seat, "
        f"seed {report['seed']}",
    ]
    for s in report["seats"]:
        total = s["wins"] + s["draws"] + s["losses"] + s["unfinished"]
        lines.append(
            f"  tablebase as {s['tablebase_side']} ({s['seat']} seat), chance as {s['chance_side']}: "
            f"{s['wins']} won, {s['draws']} drawn, {s['losses']} lost, {s['unfinished']} unfinished of {total}"
        )
        if s["mean_win_moves"] is not None:
            lines.append(f"    mean length of the tablebase's wins: {s['mean_win_moves']} moves")
        if s["mean_loss_moves"] is not None:
            lines.append(f"    mean length of the tablebase's losses: {s['mean_loss_moves']} moves")
    return lines
