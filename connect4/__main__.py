"""Play Connect Four in the terminal against any of the agents, or watch two agents play.

    python -m connect4                         # you (X) vs AlphaZero, you move first
    python -m connect4 --agent minimax --level 3 --ai-first
    python -m connect4 --watch alphazero minimax

During your turn, type a column number (1-7), "h" for a hint, or "q" to quit.
"""

from __future__ import annotations

import argparse
import sys

from .agents import AGENTS
from .board import COLS, Board

NAMES = {"alphazero": "AlphaZero", "minimax": "Minimax", "mcts": "MCTS"}


def render(board: Board, first_symbol="X", second_symbol="O") -> str:
    """ASCII board, top row first. The player who moved first is always X."""
    # Board.grid() is relative to the player to move (1 = to move, -1 = other);
    # convert that back to fixed symbols for whoever moved first.
    to_move_is_first = board.to_move_is_first()
    rows = []
    for row in board.grid():
        cells = []
        for v in row:
            if v == 0:
                cells.append(".")
            elif (v == 1) == to_move_is_first:
                cells.append(first_symbol)
            else:
                cells.append(second_symbol)
        rows.append("| " + " ".join(cells) + " |")
    return "\n".join(rows + ["+" + "-" * (2 * COLS + 1) + "+", "  " + " ".join(str(c + 1) for c in range(COLS))])


def describe(result: dict) -> str:
    """One line explaining how an agent chose its move."""
    a = result["analysis"]
    if a["kind"] == "alphazero":
        return (f"{a['simulations']} simulations; estimates its chance of winning at "
                f"{100 * a['win_probability']:.0f}%")
    if a["kind"] == "minimax":
        best = a["best"]
        verdict = {"win": "a forced win", "loss": "a forced loss"}.get(best["result"], f"score {best.get('score')}")
        return f"searched {a['depth']} plies, {a['nodes']:,} positions; sees {verdict}"
    return f"{a['simulations']} random playouts"


def ask_column(board: Board, agent: str, level: int, read=input) -> int | None:
    """Prompt until the player enters a legal column; None means quit."""
    while True:
        text = read("Your move (1-7, h = hint, q = quit): ").strip().lower()
        if text == "q":
            return None
        if text == "h":
            hint = AGENTS[agent](board, level)
            print(f"  hint: column {hint['move'] + 1} ({describe(hint)})")
            continue
        if text.isdigit() and board.can_play(int(text) - 1):
            return int(text) - 1
        print("  that column is full or not a column number")


def play(players: dict, level: int, read=input) -> int:
    """Run one game. players maps 1 (first) and 2 (second) to "human" or an agent name.

    Returns 1 or 2 for the winner, 0 for a draw, or -1 if the human quit.
    """
    board = Board()
    while True:
        number = 1 if board.to_move_is_first() else 2
        player = players[number]
        print("\n" + render(board))
        if player == "human":
            col = ask_column(board, players[3 - number], level, read)
            if col is None:
                return -1
        else:
            result = AGENTS[player](board, level)
            col = result["move"]
            print(f"\n{NAMES[player]} ({'X' if number == 1 else 'O'}) plays column {col + 1}: {describe(result)}")
        if board.is_winning_move(col):
            print("\n" + render(board.play(col)))
            return number
        board = board.play(col)
        if board.is_full():
            print("\n" + render(board))
            return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m connect4", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=sorted(AGENTS), default="alphazero", help="opponent (default alphazero)")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=2, help="search budget (default 2)")
    parser.add_argument("--ai-first", action="store_true", help="let the agent move first")
    parser.add_argument("--watch", nargs=2, metavar=("FIRST", "SECOND"), choices=sorted(AGENTS),
                        help="watch two agents play each other")
    args = parser.parse_args(argv)

    if args.watch:
        players = {1: args.watch[0], 2: args.watch[1]}
    elif args.ai_first:
        players = {1: args.agent, 2: "human"}
    else:
        players = {1: "human", 2: args.agent}

    try:
        winner = play(players, args.level)
    except FileNotFoundError as e:  # AlphaZero weights not trained yet
        print(e, file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
        return 0
    if winner == -1:
        print("You quit.")
    elif winner == 0:
        print("Draw: the board is full.")
    else:
        who = "You win" if players[winner] == "human" else f"{NAMES[players[winner]]} wins"
        print(f"{who} ({'X' if winner == 1 else 'O'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
