"""Play checkers in the terminal against alpha-beta or minimax, or watch two agents play.

    python -m checkers                              # you (red, moving first) vs alphabeta
    python -m checkers --agent minimax --level 3 --white
    python -m checkers --watch alphabeta minimax

During your turn, type a move such as 22-18 or 23x14x5, its number from the list,
h for a hint, or q to quit.
"""

from __future__ import annotations

import argparse
import sys

from .agents import DESCRIPTIONS, choose
from .board import RED, START, WHITE, Move, index, legal_moves, notation

NAMES = {"alphabeta": "alpha-beta", "minimax": "minimax"}
SIDES = {RED: "red", WHITE: "white"}
PIECES = {1: "r", 2: "R", -1: "w", -2: "W"}

# A draw is declared after this many plies (one move by either side) with no capture
# and no man moving. Without a limit two kings can shuffle forever, and watch mode
# would never end.
QUIET_PLIES = 80


def render(board) -> str:
    """ASCII board, row 0 (squares 1-4) at the top. Empty dark squares show their number.

    Light squares are blank. Pieces are r/R for red man/king and w/W for white.
    """
    line = "+" + "----+" * 8
    rows = [line]
    for r in range(8):
        cells = []
        for c in range(8):
            i = index(r, c)
            if i < 0:  # light square
                cells.append("    ")
            elif board[i]:
                cells.append(f"{PIECES[board[i]]:^4}")
            else:  # the number is what the player types in notation
                cells.append(f"{i + 1:^4}")
        rows.append("|" + "|".join(cells) + "|")
        rows.append(line)
    return "\n".join(rows)


def count(n: int, one: str, many: str) -> str:
    """"1 ply", "2 plies", "1,234 positions": the number with the noun in the right form."""
    return f"{n:,} {one if n == 1 else many}"


def verdict(best) -> str:
    """The search's conclusion in words; best comes from agents.describe()."""
    if best is None:
        return "no score (ran out of time)"
    if best["result"] == "eval":
        return f"evaluation {best['score']:+d}"
    word = "win" if best["result"] == "win" else "loss"
    return f"forced {word} in {count(best['plies'], 'ply', 'plies')}"


def summary(analysis) -> str:
    """Depth, positions examined, cutoffs, and verdict for one search."""
    return (f"depth {analysis['depth']}, {count(analysis['nodes'], 'position', 'positions')}, "
            f"{count(analysis['cutoffs'], 'cutoff', 'cutoffs')}; {verdict(analysis['best'])}")


def as_move(move_json: dict) -> Move:
    """Rebuild a Move from the plain-list form that choose() returns."""
    return Move(tuple(move_json["path"]), tuple(move_json["captured"]), tuple(move_json["result"]))


def parse_move(text: str, moves: list[Move]) -> Move | None:
    """The legal move the player typed: its number in the list, or its notation.

    Notation is compared by path, so "23x14x5" and "23-14-5" name the same
    squares. A path fixes its captures, so the path alone identifies a move.
    Returns None if the text names no legal move.
    """
    if text.isdigit():  # a number is an index into the printed list, never a square
        n = int(text)
        return moves[n - 1] if 1 <= n <= len(moves) else None
    try:
        path = tuple(int(part) - 1 for part in text.replace("x", "-").split("-"))
    except ValueError:
        return None
    for move in moves:
        if move.path == path:
            return move
    return None


def ask_move(board, turn: int, agent: str, level: int, moves: list[Move], read=input) -> Move | None:
    """Prompt until the player enters a legal move; None means quit.

    `agent` is the opponent's algorithm, reused for hints so a hint shows what it would play.
    """
    print("Legal moves: " + "   ".join(f"{i}. {notation(m)}" for i, m in enumerate(moves, 1)))
    while True:
        text = read("Your move (number or notation, h = hint, q = quit): ").strip().lower()
        if text == "q":
            return None
        if text == "h":
            hint = choose(board, turn, agent, level)
            print(f"  hint: {hint['move']['notation']} ({summary(hint['analysis'])})")
            continue
        move = parse_move(text, moves)
        if move is not None:
            return move
        print("  not a legal move: type a number from the list, a move such as 22-18, h, or q")


def play(players: dict, level: int, read=input, board=START) -> int | None:
    """Run one game. players maps RED and WHITE to "human" or an agent name.

    Returns the winning side (RED or WHITE), 0 for a draw, or None if the human quit.
    A side with no legal move loses, which is how captures and blockades end a game.
    """
    turn, quiet = RED, 0  # quiet counts plies since the last capture or man move
    print("Red: " + ("you" if players[RED] == "human" else NAMES[players[RED]])
          + "   White: " + ("you" if players[WHITE] == "human" else NAMES[players[WHITE]]))
    print("Pieces: r red man, R red king, w white man, W white king. Squares are numbered 1-32.")
    while True:
        print("\n" + render(board))
        moves = legal_moves(board, turn)
        if not moves:
            print(f"{SIDES[turn].capitalize()} has no legal moves.")
            return -turn
        if quiet >= QUIET_PLIES:
            return 0
        player = players[turn]
        if player == "human":
            move = ask_move(board, turn, players[-turn], level, moves, read)
            if move is None:
                return None
        else:
            result = choose(board, turn, player, level)
            move = as_move(result["move"])
            print(f"  {SIDES[turn].capitalize()} ({NAMES[player]}) plays {result['move']['notation']}: "
                  f"{summary(result['analysis'])}")
        # Captures and man moves reset the draw count; a king stepping without capturing does not.
        quiet = 0 if move.captured or abs(board[move.path[0]]) == 1 else quiet + 1
        board = move.result
        turn = -turn


def main(argv=None, read=input) -> int:
    """Parse the options, set up the players, and play one game. Returns the exit status."""
    parser = argparse.ArgumentParser(prog="python -m checkers", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=sorted(DESCRIPTIONS), default="alphabeta",
                        help="opponent (default alphabeta)")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=2, help="search budget (default 2)")
    parser.add_argument("--white", action="store_true", help="you play white and move second")
    parser.add_argument("--watch", nargs=2, metavar=("FIRST", "SECOND"), choices=sorted(DESCRIPTIONS),
                        help="watch two agents play each other; FIRST plays red")
    args = parser.parse_args(argv)

    if args.watch:
        players = {RED: args.watch[0], WHITE: args.watch[1]}
    elif args.white:
        players = {RED: args.agent, WHITE: "human"}
    else:
        players = {RED: "human", WHITE: args.agent}

    try:
        winner = play(players, args.level, read)
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
        return 0
    if winner is None:
        print("You quit.")
    elif winner == 0:
        print(f"Draw: {QUIET_PLIES} plies without a capture or a man move.")
    else:
        who = "You win" if players[winner] == "human" else f"{NAMES[players[winner]].capitalize()} wins"
        print(f"{who} ({SIDES[winner]}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
