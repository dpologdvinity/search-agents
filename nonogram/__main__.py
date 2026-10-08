"""Play nonograms in the terminal, solve a puzzle with any method, draw random puzzles, or run the benchmark.

    python -m nonogram                           # play the first library picture
    python -m nonogram play --puzzle rocket      # a library picture
    python -m nonogram play --random 12x12 --seed 7
    python -m nonogram solve --puzzle cat --method sat --stats
    python -m nonogram random 10x10 --seed 3     # clues only; add --solve to show the picture
    python -m nonogram list
    python -m nonogram benchmark --write         # the run committed under results/

In play, cells are typed by row and column, counted from 1: "f 3 4" fills row 3, column 4; "x 3 4" crosses
it out; "e 3 4" clears it; "3 4" fills. "h" shows the hint (one cell that line solving forces, and why),
"s" shows the solution, "k" checks the marks, and "q" quits.
"""

from __future__ import annotations

import argparse
import sys

from .benchmark import run, to_markdown, write_results
from .generate import NoUniquePicture, random_puzzle
from .library import get, library
from .puzzle import EMPTY, FILLED, UNKNOWN, Puzzle, clues_of, empty_grid, picture_text, render
from .solvers import METHODS, hint, solve, verify

HELP = ("commands: f r c fill | x r c cross | e r c clear | r c fill | h hint | s solution | k check | q quit\n"
        "rows and columns are counted from 1")


def _parse_size(text: str) -> tuple[int, int]:
    try:
        rows, cols = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"size must look like 10x12, got {text!r}") from None
    return rows, cols


def pick_puzzle(args) -> Puzzle:
    """The puzzle named by --puzzle, or a random one from --random RxC and --seed."""
    if getattr(args, "random", None):
        rows, cols = args.random
        puzzle, seed, attempts = random_puzzle(rows, cols, seed=args.seed)
        print(f"random {rows}x{cols}, seed {seed}, unique after {attempts} tries")
        return puzzle
    pid = getattr(args, "puzzle", None)
    if pid is None:
        return library()[0]
    puzzle = get(pid)
    if puzzle is None:
        names = ", ".join(p.id for p in library())
        raise SystemExit(f"no puzzle called {pid!r}; choose from: {names}")
    return puzzle


def play(puzzle: Puzzle, read=input, out=print) -> bool:
    """Interactive game. Returns True when the marks match the clues (the puzzle is solved)."""
    grid = empty_grid(puzzle.rows, puzzle.cols)
    out(f"{puzzle.name}: {puzzle.rows}x{puzzle.cols}. {HELP}")
    out(render(puzzle, grid))
    while True:
        try:
            text = read("> ").strip().lower()
        except EOFError:
            return False
        parts = text.split()
        if not parts:
            continue
        if parts[0] in ("q", "quit", "exit"):
            return False
        if parts[0] == "h":
            h = hint(puzzle, grid)
            out(h["message"] if not h["found"] else f"hint: {h['message']}\n  {h['reason']}")
            continue
        if parts[0] == "s":
            res = solve(puzzle, "hybrid")
            if res.solution is None:
                out(f"no solution: {res.status}")
            else:
                out(render(puzzle, res.solution, solved=True))
            return False
        if parts[0] == "k":
            out(_check_text(puzzle, grid))
            continue
        action, coords = "f", parts
        if parts[0] in ("f", "x", "e"):
            action, coords = parts[0], parts[1:]
        try:
            r, c = (int(v) - 1 for v in coords[:2])
        except ValueError:
            out(f"could not read '{text}'. {HELP}")
            continue
        if not (0 <= r < puzzle.rows and 0 <= c < puzzle.cols):
            out(f"row {r + 1}, column {c + 1} is off the {puzzle.rows}x{puzzle.cols} board")
            continue
        grid[r][c] = {"f": FILLED, "x": EMPTY, "e": UNKNOWN}[action]
        out(render(puzzle, grid))
        filled = [[1 if v == FILLED else 0 for v in row] for row in grid]
        if verify(puzzle, filled):
            out("solved: every row and column matches its clue")
            return True


def _check_text(puzzle: Puzzle, grid) -> str:
    """Which rows and columns already match their clue, and which do not."""
    filled = [[1 if v == FILLED else 0 for v in row] for row in grid]
    rows, cols = clues_of(filled)
    bad_rows = [str(i + 1) for i, (a, b) in enumerate(zip(rows, puzzle.row_clues))
                if a != b and all(v != UNKNOWN for v in grid[i])]
    bad_cols = [str(j + 1) for j, (a, b) in enumerate(zip(cols, puzzle.col_clues))
                if a != b and all(grid[i][j] != UNKNOWN for i in range(puzzle.rows))]
    done_rows = sum(a == b for a, b in zip(rows, puzzle.row_clues))
    done_cols = sum(a == b for a, b in zip(cols, puzzle.col_clues))
    text = f"rows matching their clue: {done_rows}/{puzzle.rows}, columns: {done_cols}/{puzzle.cols}"
    if bad_rows or bad_cols:
        text += f"; complete but wrong: rows {', '.join(bad_rows) or '-'}, columns {', '.join(bad_cols) or '-'}"
    return text


def cmd_solve(args) -> int:
    puzzle = pick_puzzle(args)
    # No budget here: the command runs on the user's own machine, so it may take as long as it needs.
    res = solve(puzzle, args.method)
    print(f"{puzzle.name} ({puzzle.rows}x{puzzle.cols}) by {args.method}: {res.status}")
    if res.solution is not None:
        print(render(puzzle, res.solution, solved=True))
    if res.partial is not None and res.solution is None:
        print(render(puzzle, res.partial))
    if res.second is not None:
        print("a second solution:")
        print(picture_text(res.second))
    if args.stats:
        s = res.stats
        print(f"seconds {s.seconds:.4f}  decisions {s.decisions}  conflicts {s.conflicts}  "
              f"propagations {s.propagations}  backtracks {s.backtracks}  sweeps {s.sweeps}")
        if s.variables is not None:
            print(f"variables {s.variables}  clauses {s.clauses}  pure eliminated {s.pure_eliminated}")
    return 0 if res.status in ("unique", "undecided", "multiple") else 1


def cmd_random(args) -> int:
    rows, cols = args.size
    puzzle, seed, attempts = random_puzzle(rows, cols, seed=args.seed)
    print(f"seed {seed}, unique after {attempts} tries")
    print(render(puzzle))
    if args.solve:
        print(render(puzzle, solve(puzzle, "hybrid").solution, solved=True))
    return 0


def cmd_list(_args) -> int:
    for p in library():
        print(f"{p.id:9s} {p.rows}x{p.cols:<3d} {p.name}")
    return 0


def cmd_benchmark(args) -> int:
    result = run(seeds=args.seeds)
    print(to_markdown(result))
    if args.write:
        json_path, md_path = write_results(result)
        print(f"wrote {json_path} and {md_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Nonograms: play, solve, generate, benchmark.")
    sub = parser.add_subparsers(dest="cmd")

    p_play = sub.add_parser("play", help="play a puzzle in the terminal")
    p_play.add_argument("--puzzle", help="library id (see `list`)")
    p_play.add_argument("--random", type=_parse_size, metavar="RxC", help="a random unique picture of this size")
    p_play.add_argument("--seed", type=int, help="seed for --random")

    p_solve = sub.add_parser("solve", help="solve a puzzle and print the picture")
    p_solve.add_argument("--puzzle", help="library id (see `list`)")
    p_solve.add_argument("--random", type=_parse_size, metavar="RxC")
    p_solve.add_argument("--seed", type=int)
    p_solve.add_argument("--method", choices=METHODS, default="hybrid")
    p_solve.add_argument("--stats", action="store_true", help="print the search counters")

    p_rand = sub.add_parser("random", help="draw a random puzzle with a unique solution")
    p_rand.add_argument("size", type=_parse_size, metavar="RxC")
    p_rand.add_argument("--seed", type=int)
    p_rand.add_argument("--solve", action="store_true", help="also print the picture")

    sub.add_parser("list", help="list the library puzzles")

    p_bench = sub.add_parser("benchmark", help="line solving, hybrid, and SAT on the benchmark set")
    p_bench.add_argument("--seeds", type=int, default=10, help="random puzzles per size (default 10)")
    p_bench.add_argument("--write", action="store_true", help="write results/nonogram_benchmark.{json,md}")

    args = parser.parse_args(argv)
    if args.cmd is None:
        args.cmd, args.puzzle, args.random, args.seed = "play", None, None, None
    try:
        if args.cmd == "play":
            play(pick_puzzle(args))
            return 0
        if args.cmd == "solve":
            return cmd_solve(args)
        if args.cmd == "random":
            return cmd_random(args)
        if args.cmd == "list":
            return cmd_list(args)
        return cmd_benchmark(args)
    except NoUniquePicture as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
