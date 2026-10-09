"""Play Sokoban in the terminal, watch the solver, or benchmark the search rules.

    python -m sokoban                          # play level 1 (w a s d move and push, see the help line)
    python -m sokoban play --level 5           # start at level 5
    python -m sokoban watch --level 9          # the solver's plan, animated in the terminal
    python -m sokoban solve --level 9 --algorithm bfs --no-prune --budget 50000
    python -m sokoban levels                   # the built-in levels with their optimal push counts
    python -m sokoban benchmark                # every search rule on every level (about a few minutes)
    python -m sokoban benchmark --out results/sokoban_benchmark

Level symbols: '#' wall, ' ' floor, '.' goal, '$' box, '*' box on a goal, '@' player, '+' player on a goal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmark import render_table, run_benchmark, write_results
from .board import render
from .heuristics import MATCHING, NONE, SIMPLE
from .levels import all_levels, get_level
from .play import Game, animate, level_list, play
from .search import ALGORITHMS, ASTAR, BFS, DEFAULT_NODES, DEFAULT_SECONDS, solve

HEURISTIC_CHOICES = (SIMPLE, MATCHING)


def _solve_command(args) -> int:
    level = get_level(args.level)
    heuristic = NONE if args.algorithm == BFS else args.heuristic
    res = solve(level, args.algorithm, heuristic, not args.no_prune, node_budget=args.budget,
                time_limit=args.seconds)
    print("\n".join(render(level, level.start_player, level.start_boxes)))
    print(f"{args.algorithm} ({heuristic}, deadlock pruning {'off' if args.no_prune else 'on'}): {res.status}")
    print(f"expanded {res.expanded:,}  generated {res.generated:,}  pruned {res.pruned:,}  time {res.seconds:.3f} s")
    if res.solved:
        print(f"pushes {res.pushes}  moves {len(res.moves)}")
        print(res.moves)
    return 0 if res.solved else 1


def _watch_command(args) -> int:
    level = get_level(args.level)
    res = solve(level, ASTAR, MATCHING, True, node_budget=DEFAULT_NODES, time_limit=DEFAULT_SECONDS)
    if not res.solved:
        print(f"no solution: status {res.status}")
        return 1
    game = Game.start(level)
    print("\n".join(render(level, game.player, game.boxes)))
    print(f"plan: {res.pushes} pushes, {len(res.moves)} moves (A* with the matching heuristic)")
    animate(game, res.moves, print, args.delay)
    print(f"solved in {game.pushes} pushes, {len(game.moves)} moves")
    return 0


def _benchmark_command(args) -> int:
    runs = run_benchmark(node_budget=args.budget, time_limit=args.seconds)
    print(render_table(runs, args.budget, args.seconds))
    if args.out:
        write_results(Path(args.out), runs, args.budget, args.seconds)
        print(f"\nwrote {Path(args.out).with_suffix('.json')} and {Path(args.out).with_suffix('.md')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sokoban",
                                     description="Sokoban: search with push-level pruning and matching bounds.")
    sub = parser.add_subparsers(dest="command")

    p_play = sub.add_parser("play", help="play in the terminal (the default)")
    p_play.add_argument("--level", type=int, default=1)

    p_watch = sub.add_parser("watch", help="animate the solver's plan for a level")
    p_watch.add_argument("--level", type=int, default=1)
    p_watch.add_argument("--delay", type=float, default=0.15, help="seconds between frames")

    p_solve = sub.add_parser("solve", help="run one search and print the plan")
    p_solve.add_argument("--level", type=int, default=1)
    p_solve.add_argument("--algorithm", choices=ALGORITHMS, default=ASTAR)
    p_solve.add_argument("--heuristic", choices=HEURISTIC_CHOICES, default=MATCHING,
                         help="for greedy and A* (BFS has none)")
    p_solve.add_argument("--no-prune", action="store_true", help="turn deadlock pruning off")
    p_solve.add_argument("--budget", type=int, default=DEFAULT_NODES, help="node budget")
    p_solve.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="time limit")

    sub.add_parser("levels", help="list the built-in levels")

    p_bench = sub.add_parser("benchmark", help="compare the search rules on every level")
    p_bench.add_argument("--budget", type=int, default=DEFAULT_NODES, help="node budget per run")
    p_bench.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="time limit per run")
    p_bench.add_argument("--out", help="path prefix for the .json and .md results")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in (None, "play"):
        level = getattr(args, "level", 1)
        if not 1 <= level <= len(all_levels()):
            parser.error(f"level must be between 1 and {len(all_levels())}")
        play(level)
        return 0
    if args.command == "levels":
        print("\n".join(level_list()))
        return 0
    if args.command == "watch":
        return _watch_command(args)
    if args.command == "solve":
        return _solve_command(args)
    if args.command == "benchmark":
        return _benchmark_command(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
