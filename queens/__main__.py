"""Place queens yourself, solve a board with any agent, or run the benchmark.

    python -m queens play                          # 8x8: place the queens yourself
    python -m queens solve -n 8 --agent backtrack  # one agent, the board printed (n up to 40)
    python -m queens solve -n 1000000 --agent minconf --seed 1
    python -m queens benchmark [--quick] [--out results]

In play, a square is typed as its column letter and row number: "c3" is column c, row 3. Each row holds
one queen, so a square in a row that already has a queen moves it, and the same square again removes it.
A queen marked ! is in conflict. "h" shows the min-conflicts move, "s" solves the board from scratch,
and "q" quits.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from .agents import AGENT_NAMES, min_conflicts, run
from .benchmark import FULL_PLAN, QUICK_PLAN, run_benchmark, write_results
from .board import Board, is_solution, render

PLAY_N = 8
BOARD_PRINT_MAX = 40  # bigger boards print a summary only


def draw(board: Board) -> str:
    """The board with each queen marked: Q for a safe queen, ! for one under attack, '.' for an empty square."""
    n = board.n
    lines = ["   " + " ".join(chr(ord("a") + c) for c in range(n))]
    for r in range(n):
        cells = []
        for c in range(n):
            if board.cols[r] == c:
                cells.append("!" if board.queen_attacks(r) else "Q")
            else:
                cells.append(".")
        lines.append(f"{r + 1:>2} " + " ".join(cells))
    return "\n".join(lines)


def parse_square(text: str, n: int) -> tuple[int, int] | None:
    """'c3' -> (row 2, column 2), 0-based. Returns None when the text is not a square on the board."""
    text = text.strip().lower()
    if len(text) < 2 or not text[0].isalpha() or not text[1:].isdigit():
        return None
    c = ord(text[0]) - ord("a")
    r = int(text[1:]) - 1
    if 0 <= r < n and 0 <= c < n:
        return r, c
    return None


def hint(board: Board) -> str:
    """The next min-conflicts move, in words. It names the queen to move and the column with the fewest attacks."""
    n = board.n
    conflicted = [r for r in range(n) if board.cols[r] >= 0 and board.queen_attacks(r) > 0]
    if conflicted:
        # Min-conflicts repairs a conflicted queen; pick the most attacked one, first row on a tie.
        r = max(conflicted, key=lambda row: (board.queen_attacks(row), -row))
        here = board.queen_attacks(r)
        best_c = min(range(n), key=lambda c: (board.attacks_if(r, c), c))
        best = board.attacks_if(r, best_c)
        return (f"min-conflicts: move the row {r + 1} queen from {chr(97 + board.cols[r])}{r + 1} "
                f"({here} attack{'s' if here != 1 else ''}) to {chr(97 + best_c)}{r + 1} "
                f"({best} attack{'s' if best != 1 else ''} there)")
    empty = [r for r in range(n) if board.cols[r] < 0]
    if empty:
        r = empty[0]
        best_c = min(range(n), key=lambda c: (board.attacks_if(r, c), c))
        return (f"no conflicts yet: row {r + 1} needs a queen; {chr(97 + best_c)}{r + 1} is attacked by "
                f"{board.attacks_if(r, best_c)} queen{'s' if board.attacks_if(r, best_c) != 1 else ''}")
    return "every row has a queen and nothing attacks anything"


def play(n: int = PLAY_N, read=input, out=print, seed: int | None = None) -> int:
    """Interactive game. Returns the number of moves made when the player solves it (or quits)."""
    board = Board(n)
    moves = 0
    out(f"N-Queens {n}x{n}. Place one queen in every row so that no two attack each other.")
    out("Type a square (c3 = column c, row 3). h = hint, s = solve from scratch, q = quit.")
    while True:
        out("\n" + draw(board))
        out(f"queens {sum(1 for c in board.cols if c >= 0)}/{n}, conflicts {board.conflicts}")
        if board.is_solved():
            out(f"Solved in {moves} moves.")
            return moves
        try:
            text = read("> ").strip().lower()
        except EOFError:
            return moves
        if text in ("q", "quit", "exit"):
            return moves
        if text == "h":
            out(hint(board))
            continue
        if text == "s":
            res = min_conflicts(n, random.Random(seed))
            out("min-conflicts solution (" + f"{res.steps} repairs" + "):")
            out(render(res.cols))
            continue
        square = parse_square(text, n)
        if square is None:
            out(f"'{text}' is not a square on this {n}x{n} board")
            continue
        r, c = square
        if board.cols[r] == c:
            board.remove(r)  # the same square again takes the queen back
        else:
            board.move(r, c)
        moves += 1


def solve_summary(n: int, agent: str, seed: int | None, time_limit: float | None) -> tuple[str, object]:
    kwargs = {"time_limit": time_limit} if time_limit is not None else {}
    res = run(agent, n, seed, **kwargs)
    lines = [
        f"agent {agent}, N = {n:,}, seed {seed}",
        f"  {'solved' if res.solved else 'not solved'} ({res.reason}) in {res.seconds:.3f} s",
        f"  steps {res.steps:,}, candidate squares scored {res.evaluations:,}, restarts {res.restarts:,}",
    ]
    if res.solved:
        lines.append(f"  checked: {'no two queens attack each other' if is_solution(res.cols) else 'INVALID'}")
    elif res.reason == "exhausted":
        # Backtracking searched every row and found no board, so there are no conflicts to count: it is a proof.
        lines.append("  no solution exists")
    else:
        lines.append(f"  conflicts left: {res.conflicts:,}")
    return "\n".join(lines), res


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m queens",
                                     description="N-Queens: four agents on one board, and the benchmark.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("play", help="place the queens yourself on an 8x8 board")
    p_solve = sub.add_parser("solve", help="solve an N-Queens board with one agent")
    p_solve.add_argument("-n", type=int, default=8, help="board size (default 8)")
    p_solve.add_argument("--agent", choices=AGENT_NAMES, default="minconf")
    p_solve.add_argument("--seed", type=int, default=1)
    p_solve.add_argument("--time-limit", type=float, default=60.0, help="seconds before giving up (default 60)")
    p_solve.add_argument("--board", action="store_true", help=f"print the board (sizes up to {BOARD_PRINT_MAX})")
    p_bench = sub.add_parser("benchmark", help="success rate, steps and time for every agent and size")
    p_bench.add_argument("--quick", action="store_true", help="fewer trials and shorter caps")
    p_bench.add_argument("--out", default="results", help="directory for queens_benchmark.json and .md")
    args = parser.parse_args(argv)

    if args.cmd == "play":
        play()
        return 0
    if args.cmd == "solve":
        if args.n < 1:
            parser.error("the board needs at least one square")
        text, res = solve_summary(args.n, args.agent, args.seed, args.time_limit)
        print(text)
        if args.board and args.n <= BOARD_PRINT_MAX and res.solved:
            print(render(res.cols))
        return 0 if res.solved else 1
    if args.cmd == "benchmark":
        plan = QUICK_PLAN if args.quick else FULL_PLAN

        def progress(agent, n, summary):
            print(f"{agent:9s} N={n:>9,}  solved {summary['solved']}/{summary['runs']}  "
                  f"median time {summary['seconds_median']:.3g} s", flush=True)

        t0 = time.perf_counter()
        data = run_benchmark(plan, progress=progress)
        json_path, md_path = write_results(data, args.out)
        print(f"wrote {json_path} and {md_path} in {time.perf_counter() - t0:.0f} s")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
