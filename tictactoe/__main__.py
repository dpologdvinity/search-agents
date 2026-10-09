"""Command line for tic-tac-toe.

python -m tictactoe              play X against the minimax agent (O); squares are 1-9, keypad layout
python -m tictactoe stats        count the positions and show that perfect play is a draw
python -m tictactoe positions    JSON for every reachable position (used by the JS parity test)
"""

from __future__ import annotations

import json
import sys

from .core import EMPTY_BOARD, legal_moves, minimax, reachable_positions, search, winner

SYMBOL = {"X": "X", "O": "O", ".": " "}


def render(board: str) -> str:
    """A 3x3 grid with the keypad numbers in the empty squares, so the player knows what to type."""
    rows = []
    for r in range(3):
        cells = [SYMBOL[board[3 * r + c]] if board[3 * r + c] != "." else str(3 * r + c + 1) for c in range(3)]
        rows.append(" " + " | ".join(cells))
    return "\n---+---+---\n".join(rows)


def play() -> None:
    board = EMPTY_BOARD
    print("You are X and move first. Type a square number 1-9. The agent is O.")
    while True:
        print()
        print(render(board))
        if winner(board):
            print(f"{winner(board)} wins. Minimax does not lose, so one of us has a bug.")
            return
        if not legal_moves(board):
            print("Draw. Perfect play always ends here.")
            return
        try:
            square = int(input("your move > ")) - 1
        except (EOFError, KeyboardInterrupt):
            print()
            return
        except ValueError:
            print("Type a number from 1 to 9.")
            continue
        if not 0 <= square <= 8 or board[square] != ".":
            print("That square is taken or off the board.")
            continue
        board = board[:square] + "X" + board[square + 1 :]
        if winner(board) or not legal_moves(board):
            continue
        result = search(board, "O")
        board = board[: result.move] + "O" + board[result.move + 1 :]
        print(f"agent plays {result.move + 1}  (searched {result.searched:,} positions, pruned {result.pruned:,})")


def stats() -> None:
    in_play = reachable_positions()
    boards = reachable_positions(include_finished=True)
    # Count every game-tree node (no pruning, no shortcuts) by walking the tree from the empty board.
    nodes = 0

    def count(board: str, turn: str) -> None:
        nonlocal nodes
        nodes += 1
        if winner(board) or not legal_moves(board):
            return
        for m in legal_moves(board):
            count(board[:m] + turn + board[m + 1 :], "O" if turn == "X" else "X")

    count(EMPTY_BOARD, "X")
    print(f"positions:       {len(boards):,} distinct boards reachable ({len(in_play):,} still in play)")
    print(f"game-tree nodes: {nodes:,} (every move sequence, no pruning)")
    print(f"perfect play: {'draw' if minimax(EMPTY_BOARD, 'X') == 0 else 'not a draw'}")
    result = search(EMPTY_BOARD, "X")
    print(f"alpha-beta from the empty board: searched {result.searched:,}, pruned {result.pruned:,}")


def positions() -> None:
    out = []
    for board in reachable_positions():
        turn = "X" if board.count("X") == board.count("O") else "O"
        r = search(board, turn)
        out.append(
            {
                "board": board,
                "turn": turn,
                "move": r.move,
                "value": r.value,
                "searched": r.searched,
                "pruned": r.pruned,
                "tree": r.tree,
            }
        )
    json.dump(out, sys.stdout, separators=(",", ":"))


def main(argv: list[str]) -> None:
    command = argv[1] if len(argv) > 1 else "play"
    if command in ("-h", "--help", "help"):
        print(__doc__.strip())
        return
    commands = {"play": play, "stats": stats, "positions": positions}
    if command not in commands:
        raise SystemExit(f"unknown command {command!r}: use play, stats or positions")
    commands[command]()


if __name__ == "__main__":
    main(sys.argv)
