"""Command line for the pathfinding package.

    python -m pathfind race --size 41 --maze prim --seed 3
        runs every algorithm on one map, prints the leaderboard, then renders each path

    python -m pathfind show --algo astar --size 41 --maze prim --seed 3 --trace 6
        runs one algorithm, prints its first expansions with g, h and f, then renders its search

Maps and results are identical to the browser lab for the same flags.
"""

from __future__ import annotations

import argparse
import sys

from .grid import HEURISTICS, heuristic
from .mazes import KINDS, make_maze
from .race import race, render, table
from .search import ALGOS, DEFAULT_WEIGHT, LABELS, run


def _int_at_least(low: int):
    """argparse type factory: an integer of at least low. Smaller values give tracebacks or meaningless maps."""

    def parse(text: str) -> int:
        value = int(text)
        if value < low:
            raise argparse.ArgumentTypeError(f"must be at least {low}, got {value}")
        return value

    return parse


def _percent(text: str) -> int:
    """argparse type for a percentage: an integer from 0 to 100."""
    value = int(text)
    if not 0 <= value <= 100:
        raise argparse.ArgumentTypeError(f"must be between 0 and 100, got {value}")
    return value


def _weight(text: str) -> float:
    """argparse type for weighted A*'s w: a number of at least 1 (NaN is rejected too)."""
    value = float(text)
    if not value >= 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {text}")
    return value


def _add_map_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--size", type=_int_at_least(5), default=41, help="map width and height in cells, at least 5 (default 41)"
    )
    p.add_argument("--maze", choices=KINDS, default="prim", help="map generator (default prim)")
    p.add_argument("--seed", type=int, default=1, help="seed for the generator (default 1)")
    p.add_argument("--swamp", type=_percent, default=0, help="percent of open cells that are swamp, cost 5")
    p.add_argument("--density", type=_percent, default=28, help="wall percent for --maze scatter (default 28)")
    p.add_argument("--diagonal", action="store_true", help="allow diagonal moves (8-connected)")
    p.add_argument(
        "--heuristic",
        choices=HEURISTICS,
        default="octile",
        help="heuristic for greedy, A* and weighted A* (default octile)",
    )
    p.add_argument(
        "--weight",
        type=_weight,
        default=DEFAULT_WEIGHT,
        help=f"w for weighted A*, at least 1 (default {DEFAULT_WEIGHT:g})",
    )


def _maze(args: argparse.Namespace):
    return make_maze(args.maze, args.size, args.size, args.seed, density=args.density, swamp=args.swamp)


def cmd_race(args: argparse.Namespace) -> int:
    maze = _maze(args)
    algos = tuple(a.strip() for a in args.algos.split(",")) if args.algos else ALGOS
    for a in algos:
        if a not in ALGOS:
            print(f"unknown algorithm {a!r}; choose from {', '.join(ALGOS)}", file=sys.stderr)
            return 2
    optimal, rows = race(maze, algos, diagonal=args.diagonal, heur=args.heuristic, weight=args.weight)
    print(
        f"map {args.size}x{args.size}  maze={args.maze}  seed={args.seed}  swamp={args.swamp}%  "
        f"diagonal={'on' if args.diagonal else 'off'}  heuristic={args.heuristic}  w={args.weight:g}"
    )
    print(table(rows, optimal))
    if not args.no_render:
        for row in rows:
            print()
            print(f"{row.algo}: {LABELS[row.algo]}")
            print(render(maze.grid, maze.start, maze.goal, row.result.path))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    maze = _maze(args)
    res = run(
        args.algo, maze.grid, maze.start, maze.goal, diagonal=args.diagonal, heur=args.heuristic, weight=args.weight
    )
    optimal, _ = race(maze, ("ucs",), diagonal=args.diagonal)
    print(
        f"{args.algo}: {LABELS[args.algo]}  found={'yes' if res.found else 'no'}  "
        f"expanded={res.expanded}  generated={res.generated}  "
        f"cost={'-' if res.cost is None else f'{res.cost:.2f}'}  "
        f"optimal={'-' if optimal is None else f'{optimal:.2f}'}"
    )
    if args.trace and args.algo in ("greedy", "astar", "wastar"):
        print("first expansions (g = cost so far, h = estimate to goal, f = g + w*h):")
        for cell, g, h, f in _trace(res, maze, args):
            x, y = maze.grid.xy(cell)
            print(f"  ({x:>2},{y:>2})  g={g:7.2f}  h={h:7.2f}  f={f:7.2f}")
    print(render(maze.grid, maze.start, maze.goal, res.path, res.expansion_order))
    return 0


def _trace(res, maze, args) -> list[tuple[int, float, float, float]]:
    """(cell, g, h, f) for the first --trace expansions. g is the best value recorded for the cell
    before it was expanded, which for these algorithms is its final g at that moment."""
    gx, gy = maze.grid.xy(maze.goal)
    g_known = {maze.start: 0.0}
    out = []
    for cell, _side, new in res.steps[: args.trace]:
        x, y = maze.grid.xy(cell)
        g = g_known[cell]
        h = heuristic(args.heuristic, x, y, gx, gy)
        f = g + (args.weight if args.algo == "wastar" else 1.0) * h
        out.append((cell, g, h, f))
        for v, gv in new:
            g_known[v] = gv
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pathfind", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_race = sub.add_parser("race", help="run every algorithm on one map and compare")
    _add_map_args(p_race)
    p_race.add_argument("--algos", default="", help=f"comma list from {','.join(ALGOS)} (default all)")
    p_race.add_argument("--no-render", action="store_true", help="print only the table")
    p_race.set_defaults(func=cmd_race)

    p_show = sub.add_parser("show", help="run one algorithm and render its search")
    _add_map_args(p_show)
    p_show.add_argument("--algo", choices=ALGOS, default="astar")
    p_show.add_argument("--trace", type=_int_at_least(0), default=0, help="print g, h, f for the first N expansions")
    p_show.set_defaults(func=cmd_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
