"""Play the solved endgames in the terminal, analyze a position, or print the tablebase statistics.

    python -m endgame                         # KQK: a random winning position, you are the lone king
    python -m endgame --piece R --as strong   # KRK: you have the rook and king; the agent defends
    python -m endgame --fen "8/8/8/8/2k5/8/1R6/K7 w - - 0 1"   # set up a position (the piece is read from the FEN)
    python -m endgame analyze "<FEN>"         # every move with its distance to mate, and the agent's choice
    python -m endgame stats [--piece Q] [--json]   # counts by result, DTM histograms, build time
    python -m endgame build [--piece Q]       # re-solve the tables into endgame/data and verify them
    python -m endgame match --games 100 --piece Q   # tablebase vs the chance opponent, no prompts
    python -m endgame --opponent chance       # the agent picks its moves by fixed odds, not the table

While playing, type a move (Qd4, Kc3, or d1d4 for a move that needs its from square), h for a hint with
every legal move's distance to mate, or q to quit. The countdown above the board is the number of moves the
side with the piece needs, with best play from both sides, so it drops by one on each best move.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys

from . import chance
from .match import match_lines, run_match
from .retro import check_positions, legal_positions
from .rules import BLACK, PIECES, WHITE, apply_move, in_check, parse_fen, parse_square
from .tablebase import MoveInfo, Tablebase, build, load, rank_key

_SIDE = {WHITE: "White", BLACK: "Black"}
# A move as typed: optional piece letter, then one square (destination) or two (from, to).
_MOVE_TEXT = re.compile(r"^([kqr]?)([a-h][1-8])(?:([a-h][1-8]))?$")


def render(piece: str, pos) -> str:
    """ASCII board: white pieces in capitals (K, the piece), black king as k, rank 8 at the top."""
    wk, wp, bk, _ = pos
    sym = {wk: "K", wp: piece, bk: "k"}
    lines = []
    for rank in range(7, -1, -1):
        cells = [sym.get(rank * 8 + f, ".") for f in range(8)]
        lines.append(f"  {rank + 1} " + " ".join(cells))
    lines.append("    a b c d e f g h")
    return "\n".join(lines)


def describe_info(info: MoveInfo) -> str:
    """How a move reads to the side that made it: mate in N, a draw, or mated in N at most."""
    if info.outcome == "draw":
        return "draw"
    if info.outcome == "win":
        return f"mate in {info.moves}"
    return f"mated in {info.moves} at most"


def strong_mate_in(tb: Tablebase, pos) -> int | None:
    """Moves the side with the piece needs to mate from pos, or None if the position is drawn.

    The code counts plies from the side to move. When Black is to move, the code is Black's loss and
    the mate still comes that many plies from here, so the same formula covers both sides to move.
    """
    outcome, plies = tb.result(pos)
    if outcome == "draw" or plies is None:
        return None
    return (plies + 1) // 2


def status_line(tb: Tablebase, pos) -> str:
    """The countdown shown above every board."""
    n = strong_mate_in(tb, pos)
    if n is None:
        return "Drawn: neither side can force mate from here."
    if n == 0:
        return "Checkmate."
    return f"Mate in {n} for the king and {tb.piece} with best play from both sides."


def judge(chosen: MoveInfo, best: MoveInfo, strong: bool) -> str | None:
    """Commentary on the strong side's move: best, slower than the fastest win, or a blunder.

    Only the strong side is judged. The lone king's job is to survive, so its moves get no verdict.
    """
    if not strong:
        return None
    if rank_key(chosen) == rank_key(best):
        return "best move."
    if chosen.outcome == "win":
        return f"winning, but not the fastest: {best.san} was mate in {best.moves}."
    if best.outcome == "win":
        return f"blunder: this gives up the win. {best.san} was mate in {best.moves}."
    return "blunder: this loses."


def parse_move(text: str, infos: list[MoveInfo]) -> MoveInfo:
    """Find the legal move a player typed: 'Qd4', 'Kc3', 'Kxd1', or 'd1d4'. Raises ValueError with the reason."""
    cleaned = text.strip().lower()
    for junk in ("x", "-", "+", "#", " "):
        cleaned = cleaned.replace(junk, "")
    m = _MOVE_TEXT.match(cleaned)
    if not m:
        raise ValueError(f"not a move: {text.strip()!r} (try Qd4, Kc3 or d1d4)")
    letter, first, second = m.group(1), m.group(2), m.group(3)
    from_sq, to_sq = (parse_square(first), parse_square(second)) if second else (None, parse_square(first))
    matches = []
    for info in infos:
        mover, src, dst = info.move
        if dst != to_sq or (from_sq is not None and src != from_sq):
            continue
        if letter and mover.lower() != letter:
            continue
        matches.append(info)
    if not matches:
        raise ValueError(f"{text.strip()!r} is not a legal move here")
    if len(matches) > 1:
        raise ValueError(f"{text.strip()!r} is ambiguous: give the from square too, e.g. d1d4")
    return matches[0]


def hint_lines(tb: Tablebase, pos) -> list[str]:
    """Every legal move with its distance to mate, best first. The best one is marked."""
    infos = sorted(tb.move_infos(pos), key=rank_key, reverse=True)
    if not infos:
        return []
    return [f"  {info.san:<8}{describe_info(info):<24}{'<- best' if info is infos[0] else ''}".rstrip()
            for info in infos]


def play(tb: Tablebase, pos, human: int, read=input, out=print, opponent: str = chance.TABLEBASE_KEY,
         seed: int | None = None) -> str:
    """Interactive game from pos. `human` is BLACK (the lone king) or WHITE (king and piece).

    The agent answers every time it is its turn: with the table's move when opponent is 'tablebase', or with
    a move drawn from the chance odds when it is 'chance' (seeded by `seed`, so a game can be repeated).
    Returns how the game ended: 'checkmate', 'stalemate', 'draw' (a capture of the piece, or a drawn
    position), or 'quit'.
    """
    agent = 1 - human
    piece = tb.piece
    strong = human == WHITE
    rng = random.Random(seed)
    agent_name = chance.LABEL if opponent == chance.KEY else chance.TABLEBASE_LABEL
    if strong:
        out(f"K{piece}K. You have the king and the {piece}: mate as fast as you can. The agent defends.")
    else:
        out(f"K{piece}K. You are the lone king: survive as long as you can. The agent has the king and the {piece}.")
    out(f"Agent: {agent_name}." + (" It draws each move from fixed odds: captures 4, checks 2, king steps toward "
                                  "the centre 2, other moves 1." if opponent == chance.KEY else ""))
    out("Type a move (Qd4, Kc3, d1d4), h for a hint with every move's distance to mate, q to quit.")
    while True:
        out("")
        out(render(piece, pos))
        out(f"{_SIDE[pos[3]]} to move. {status_line(tb, pos)}")
        infos = tb.move_infos(pos)
        if not infos:
            if in_check(piece, pos):
                out("Checkmate. " + ("The agent won." if pos[3] == human else "You won."))
                return "checkmate"
            out("Stalemate: a draw.")
            return "stalemate"

        if pos[3] == agent:
            best = tb.best_move(pos) if opponent == chance.TABLEBASE_KEY else chance.pick(infos, pos, rng)
            out(f"agent plays {best.san}  ({describe_info(best)})")
            nxt = apply_move(pos, best.move)
            if nxt is None:
                out("The agent took the piece: king against king, a draw.")
                return "draw"
            pos = nxt
            continue

        try:
            text = read("> ")
        except EOFError:
            return "quit"
        text = text.strip()
        if text.lower() in ("q", "quit", "exit"):
            return "quit"
        if text.lower() == "h":
            out("hint: every legal move, as the side that makes it sees it")
            out("\n".join(hint_lines(tb, pos)))
            continue
        try:
            chosen = parse_move(text, infos)
        except ValueError as e:
            out(str(e))
            continue
        best = max(infos, key=rank_key)
        out(f"you play {chosen.san}  ({describe_info(chosen)})")
        note = judge(chosen, best, strong)
        if note:
            out(note)
        nxt = apply_move(pos, chosen.move)
        if nxt is None:
            out("You took the piece: king against king, a draw.")
            return "draw"
        pos = nxt


def analyze_lines(tb: Tablebase, pos) -> list[str]:
    """Lines for `analyze`: the board, the position's value, every move, and the move the agent would play."""
    lines = [render(tb.piece, pos), f"{_SIDE[pos[3]]} to move. {status_line(tb, pos)}"]
    outcome, plies = tb.result(pos)
    if plies is not None:
        lines.append(f"Value: {outcome} in {plies} plies (distance to mate with best play on both sides).")
    lines.extend(hint_lines(tb, pos))
    best = tb.best_move(pos)
    if best is not None:
        lines.append(f"agent plays {best.san}: {describe_info(best)}")
    return lines


def _histogram(hist: list[int], label: str, width: int = 36) -> list[str]:
    """Text histogram: one line per move count, the bar scaled to the largest bucket."""
    peak = max(hist) if hist and max(hist) else 1
    lines = []
    for k, count in enumerate(hist):
        if k == 0 or count == 0:
            continue
        bar = "#" * max(1, round(width * count / peak))
        lines.append(f"  {label} {k:>2}  {count:>9,}  {bar}")
    return lines


def stats_lines(tb: Tablebase) -> list[str]:
    """Stats for one table: positions by result and side to move, longest mate, DTM histograms, build time."""
    s = tb.summary()
    lines = [
        f"{tb.name}: {s['legal']:,} legal positions "
        f"({s['legal_white_to_move']:,} White to move, {s['legal_black_to_move']:,} Black to move)",
        f"  White to move: {s['win_white_to_move']:,} won, {s['draw_white_to_move']:,} drawn, "
        f"{s['loss_white_to_move']:,} lost",
        f"  Black to move: {s['loss_black_to_move']:,} lost, {s['draw_black_to_move']:,} drawn, "
        f"{s['win_black_to_move']:,} won",
        f"  Totals: {s['win']:,} won, {s['loss']:,} lost, {s['draw']:,} drawn",
        f"  Longest forced mate: {s['max_win_moves']} moves ({s['max_win_plies']} plies), e.g. "
        f"{s['longest_win_example']}",
        f"  Longest defence: mated in {s['max_loss_moves']} moves ({s['max_loss_plies']} plies)",
        f"  Built in {tb.build_seconds:.2f} s",
        "  DTM histogram in moves: wins for the side to move, then losses for the side to move",
    ]
    lines.extend(_histogram(s["win_histogram"], "win "))
    lines.extend(_histogram(s["loss_histogram"], "lost"))
    return lines


def stats_json(tables: list[Tablebase]) -> str:
    """Stats for every table as one JSON document."""
    doc = {}
    for tb in tables:
        s = dict(tb.summary())
        s["build_seconds"] = round(tb.build_seconds, 3)
        doc[tb.name] = s
    return json.dumps(doc, indent=2)


def _pieces(choice: str) -> list[str]:
    return list(PIECES) if choice == "both" else [choice]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("analyze", "stats", "build", "match"):
        return _subcommand(argv[0], argv[1:])
    if argv and argv[0] == "play":
        argv = argv[1:]
    parser = argparse.ArgumentParser(prog="python -m endgame",
                                     description="Play the solved king and queen / king and rook endgames.")
    parser.add_argument("--piece", choices=PIECES, default="Q",
                        help="the white piece for a random position (default Q)")
    parser.add_argument("--as", dest="side", choices=("weak", "strong"), default="weak",
                        help="weak: you are the lone king and try to survive; strong: you have the piece")
    parser.add_argument("--fen", help="set up this position instead of a random one (the piece is read from it)")
    parser.add_argument("--seed", type=int, help="seed the random position, for a repeatable game")
    parser.add_argument("--opponent", choices=(chance.TABLEBASE_KEY, chance.KEY), default=chance.TABLEBASE_KEY,
                        help="tablebase: the agent plays the exact best move; chance: it draws moves from fixed odds")
    args = parser.parse_args(argv)

    if args.fen:
        try:
            piece, pos = parse_fen(args.fen)
        except ValueError as e:
            parser.error(str(e))
    else:
        piece = args.piece
        pos = None
    tb = load(piece)
    if pos is None:
        # A random position the strong side wins, so there is a mate to count down to, with either side to move.
        pos = tb.random_position(random.Random(args.seed), want="strong_wins")
    play(tb, pos, WHITE if args.side == "strong" else BLACK, opponent=args.opponent, seed=args.seed)
    return 0


def _subcommand(name: str, argv: list[str]) -> int:
    if name == "analyze":
        parser = argparse.ArgumentParser(prog="python -m endgame analyze",
                                         description="Print a position's value and every move's distance to mate.")
        parser.add_argument("fen", help="FEN with a white king, a white queen or rook, and a black king")
        args = parser.parse_args(argv)
        try:
            piece, pos = parse_fen(args.fen)
        except ValueError as e:
            parser.error(str(e))
        print("\n".join(analyze_lines(load(piece), pos)))
        return 0

    if name == "stats":
        parser = argparse.ArgumentParser(prog="python -m endgame stats",
                                         description="Positions by result, DTM histograms, and build time.")
        parser.add_argument("--piece", choices=("Q", "R", "both"), default="both")
        parser.add_argument("--json", action="store_true", help="print JSON instead of text")
        args = parser.parse_args(argv)
        tables = [load(p) for p in _pieces(args.piece)]
        if args.json:
            print(stats_json(tables))
        else:
            for tb in tables:
                print("\n".join(stats_lines(tb)))
                print()
        return 0

    if name == "match":
        parser = argparse.ArgumentParser(prog="python -m endgame match",
                                         description="Tablebase against the chance opponent over random winning "
                                                     "positions, with no prompts.")
        parser.add_argument("--piece", choices=("Q", "R", "both"), default="Q")
        parser.add_argument("--games", type=int, default=100, help="positions per seat (default 100)")
        parser.add_argument("--seed", type=int, default=0, help="seed for the positions and the chance rolls")
        parser.add_argument("--as", dest="seat", choices=("strong", "weak", "both"), default="both",
                            help="the tablebase's seat: strong (White, with the piece), weak (the lone king), or both")
        parser.add_argument("--json", action="store_true", help="print the report as JSON instead of text")
        args = parser.parse_args(argv)
        seats = ("strong", "weak") if args.seat == "both" else (args.seat,)
        reports = [run_match(p, args.games, seed=args.seed, seats=seats) for p in _pieces(args.piece)]
        if args.json:
            print(json.dumps(reports[0] if len(reports) == 1 else reports, indent=2))
        else:
            print("\n\n".join("\n".join(match_lines(r)) for r in reports))
        return 0

    parser = argparse.ArgumentParser(prog="python -m endgame build",
                                     description="Solve the tables from scratch, verify every entry, "
                                                 "and write endgame/data.")
    parser.add_argument("--piece", choices=("Q", "R", "both"), default="both")
    args = parser.parse_args(argv)
    status = 0
    for piece in _pieces(args.piece):
        tb = build(piece)
        problems = check_positions(piece, tb.codes, legal_positions(piece))
        print(f"{tb.name}: solved in {tb.build_seconds:.2f} s; verification found {len(problems)} problems")
        for p in problems[:5]:
            print("  " + p)
        status = status or (1 if problems else 0)
    return status


if __name__ == "__main__":
    sys.exit(main())
