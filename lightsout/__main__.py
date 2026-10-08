"""Play Lights Out in the terminal, or print the solution for a board.

    python -m lightsout                      # 5x5, a random solvable board
    python -m lightsout -n 7 --any          # 7x7, any board (it may be unsolvable)
    python -m lightsout --solve 1100110...  # minimum-press solution for a board, then exit

Cells are typed as "b3" (column b, row 3) or "3 2" (row 3, column 2). During play,
"h" shows the solver's next press, "s" shows the whole minimum solution, and "q" quits.
"""

from __future__ import annotations

import argparse
import math
import random
import sys

from .board import DEFAULT_N, MAX_N, MIN_N, apply_presses, is_dark, label, parse_cell, random_board
from .solver import Solution, solve


def render(n: int, board) -> str:
    """ASCII board with column letters across the top and row numbers down the side. '*' is lit, '.' dark."""
    lines = ["    " + " ".join(chr(ord("a") + c) for c in range(n))]
    for r in range(n):
        cells = " ".join("*" if board[r * n + c] else "." for c in range(n))
        lines.append(f"{r + 1:>3} {cells}")
    return "\n".join(lines)


def render_presses(n: int, presses) -> str:
    """The same grid layout, with 'x' on every cell that should be pressed."""
    marked = set(presses)
    lines = ["    " + " ".join(chr(ord("a") + c) for c in range(n))]
    for r in range(n):
        cells = " ".join("x" if r * n + c in marked else "." for c in range(n))
        lines.append(f"{r + 1:>3} {cells}")
    return "\n".join(lines)


def describe_unsolvable(sol: Solution) -> str:
    """Explain why a board cannot be cleared, in terms the player can check."""
    cells = sol.n * sol.n
    return (f"This board cannot be cleared. The press matrix has rank {sol.rank} of {cells}, so only "
            f"2^{sol.rank} of the 2^{cells} boards can be reached by presses, and this one is not among them. "
            "Each press moves a board within its class, so no press sequence will ever reach dark.")


def describe_counts(sol: Solution) -> str:
    """Summary of the solution space: how many solutions there are and which press counts occur."""
    seen = sorted(set(sol.solution_sizes))
    noun = "solution" if sol.count == 1 else "solutions"
    return (f"rank {sol.rank} of {sol.n * sol.n}, nullity {sol.nullity}: {sol.count} {noun}, "
            f"press counts {', '.join(str(s) for s in seen)}")


def print_solution(n: int, board, out=print) -> None:
    """Show a board, whether it can be cleared, and its minimum-press solution."""
    sol = solve(n, board)
    out(render(n, board))
    if not sol.solvable:
        out(describe_unsolvable(sol))
        return
    out(describe_counts(sol))
    if not sol.presses:
        out("already dark: no presses needed")
        return
    out(f"minimum solution: {len(sol.presses)} presses: " + " ".join(label(n, c) for c in sol.presses))
    out(render_presses(n, sol.presses))


def play(n: int, board, read=input, out=print) -> int:
    """Interactive game on `board`. Returns the number of presses made, or -1 if the board cannot be cleared.

    The game ends when every light is dark, the player types "q", or input runs out.
    """
    board = tuple(board)
    first = solve(n, board)
    if not first.solvable:
        out(render(n, board))
        out(describe_unsolvable(first))
        return -1
    best = len(first.presses)
    out(f"Lights Out {n}x{n}. Minimum presses from this board: {best}. ({describe_counts(first)})")
    out("Type a cell (b3 = column b, row 3; or 3 2 = row 3, column 2). h = hint, s = solution, q = quit.")
    moves = 0
    while True:
        out("\n" + render(n, board))
        try:
            text = read("> ").strip().lower()
        except EOFError:
            return moves
        if text in ("q", "quit", "exit"):
            return moves
        if text == "h":
            sol = solve(n, board)
            out(f"hint: press {label(n, sol.presses[0])}  ({len(sol.presses)} presses left in the minimum solution)")
            continue
        if text == "s":
            sol = solve(n, board)
            out("solution: " + " ".join(label(n, c) for c in sol.presses) + f"  ({len(sol.presses)} presses)")
            out(render_presses(n, sol.presses))
            continue
        cell = parse_cell(n, text)
        if cell is None:
            out(f"'{text}' is not a cell on this {n}x{n} board")
            continue
        board = apply_presses(n, board, [cell])
        moves += 1
        if is_dark(board):
            out("\n" + render(n, board))
            out(f"Solved in {moves} presses. The minimum from this board was {best}.")
            return moves


_ON, _OFF = "1#", "0."
_SEPARATORS = " \t\n/|,"


def parse_board_text(text: str) -> tuple[int, tuple[int, ...]]:
    """Read a board given as cells in row-major order: '1' or '#' lit, '0' or '.' dark.

    Spaces, '/', '|' and ',' are ignored, so "10/01" works for a 2x2 grid. Returns (n, cells).
    """
    cells = []
    for ch in text:
        if ch in _SEPARATORS:
            continue
        if ch in _ON:
            cells.append(1)
        elif ch in _OFF:
            cells.append(0)
        else:
            raise ValueError(f"unexpected character {ch!r}: use 0/1 (or . and #) for cells")
    n = math.isqrt(len(cells))
    if n * n != len(cells) or not MIN_N <= n <= MAX_N:
        raise ValueError(f"need {MIN_N * MIN_N} to {MAX_N * MAX_N} cells in a square, got {len(cells)}")
    return n, tuple(cells)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Play Lights Out, or print the solution for a board.")
    parser.add_argument("-n", "--size", type=int, help=f"board size, {MIN_N}..{MAX_N} (default {DEFAULT_N}); "
                        "with --solve the size is read from the board")
    parser.add_argument("--any", action="store_true",
                        help="draw any random board, which may be unsolvable, instead of a solvable one")
    parser.add_argument("--solve", metavar="BOARD", help="print the minimum-press solution for BOARD and exit")
    parser.add_argument("--seed", type=int, help="seed the random board, for a repeatable game")
    args = parser.parse_args(argv)

    if args.solve is not None:
        try:
            n, board = parse_board_text(args.solve)
        except ValueError as e:
            parser.error(str(e))
        if args.size is not None and args.size != n:
            parser.error(f"the board has {n}x{n} cells but --size is {args.size}")
        print_solution(n, board)
        return 0

    n = args.size if args.size is not None else DEFAULT_N
    if not MIN_N <= n <= MAX_N:
        parser.error(f"size must be between {MIN_N} and {MAX_N}")
    board = random_board(n, random.Random(args.seed), solvable=not args.any)
    play(n, board)
    return 0


if __name__ == "__main__":
    sys.exit(main())
