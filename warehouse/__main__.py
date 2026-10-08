"""Plan robot routes through a warehouse in the terminal, or benchmark the planners.

    python -m warehouse                                  # menu for layout and robot count, CBS, animated
    python -m warehouse --layout bottleneck --robots 6 --planner prioritized --delay 0.1
    python -m warehouse --planner independent --no-animate   # only the result, collisions shown
    python -m warehouse benchmark --layout bottleneck --robots 6 --instances 10

Map symbols: '#' shelf, '.' floor, A, B, C... robots, a, b, c... their goals (once a robot stands
on its goal it shows as its capital letter), '!' two robots on one cell (independent A* collides).
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time

from . import LAYOUT_INFO, LAYOUTS, PLANNER_INFO, PLANNERS, PlanResult, Problem, random_instance, solve
from .cbs import cbs
from .conflicts import position

DEFAULT_LAYOUT = "aisles"
DEFAULT_ROBOTS = 4
MAX_ROBOTS = 26  # one letter per robot
# The CBS budgets are the core function's defaults, so the CLI and the library never disagree.
CBS_NODES = inspect.signature(cbs).parameters["max_nodes"].default
CBS_SECONDS = inspect.signature(cbs).parameters["max_seconds"].default


def letter(i: int) -> str:
    return chr(ord("A") + i)


def render(problem: Problem, paths, t: int) -> str:
    """The map at time step ``t``: shelves, floor, robots at their positions, and their goals."""
    grid = problem.grid
    where: dict[int, list[int]] = {}
    for i, path in enumerate(paths):
        where.setdefault(position(path, t), []).append(i)
    goal_of = {g: i for i, g in enumerate(problem.goals)}
    lines = []
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
            c = grid.cell(x, y)
            if not grid.free[c]:
                row.append("#")
            elif c in where:
                row.append("!" if len(where[c]) > 1 else letter(where[c][0]))
            elif c in goal_of:
                row.append(letter(goal_of[c]).lower())
            else:
                row.append(".")
        lines.append("".join(row))
    return "\n".join(lines)


def describe(result: PlanResult) -> list[str]:
    """Stats lines printed under the map. CBS also reports its search effort."""
    lines = [
        f"planner:    {PLANNER_INFO[result.planner]['title']}",
        f"status:     {result.status}",
    ]
    if result.paths is not None:
        lines.append(f"sum of costs: {result.sum_of_costs}   makespan: {result.makespan}   "
                     f"collisions: {len(result.conflicts)}")
    lines.append(f"seconds:    {result.seconds:.3f}")
    if result.planner == "cbs":
        lines.append(f"CBS: high-level nodes {result.high_nodes}, expanded {result.expanded_nodes}, "
                     f"conflicts resolved {result.conflicts_resolved}, low-level states {result.low_expansions:,}")
    return lines


def ask(prompt: str, default: str, read=input) -> str:
    """Read one line from the user; an empty answer means the default."""
    text = read(f"{prompt} [{default}]: ").strip()
    return text or default


def pick_menu(read=input) -> tuple[str, int]:
    """Numbered menus for layout and robot count, used only when run from an interactive terminal."""
    names = list(LAYOUTS)
    for i, name in enumerate(names, 1):
        print(f"  {i}. {LAYOUT_INFO[name]['title']} - {LAYOUT_INFO[name]['blurb']}")
    picked = ask("Layout (number)", str(names.index(DEFAULT_LAYOUT) + 1), read)
    layout = names[int(picked) - 1] if picked.isdigit() and 1 <= int(picked) <= len(names) else DEFAULT_LAYOUT
    count = ask(f"Robots (1-{MAX_ROBOTS})", str(DEFAULT_ROBOTS), read)
    robots = int(count) if count.isdigit() and 1 <= int(count) <= MAX_ROBOTS else DEFAULT_ROBOTS
    return layout, robots


def animate(problem: Problem, result: PlanResult, delay: float) -> None:
    """Print one frame per time step. Clears the screen between frames only on a real terminal."""
    tty = sys.stdout.isatty()
    for t in range(result.makespan + 1):
        if tty:
            print("\033[2J\033[H", end="")
        print(f"t = {t} / {result.makespan}")
        print(render(problem, result.paths, t))
        if t < result.makespan:
            time.sleep(delay)
        else:
            print()


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m warehouse", description="Plan warehouse robot routes.")
    parser.add_argument("--layout", choices=list(LAYOUTS), help=f"map (default {DEFAULT_LAYOUT})")
    parser.add_argument("--robots", type=_robot_count, help=f"robots, 1-{MAX_ROBOTS} (default {DEFAULT_ROBOTS})")
    parser.add_argument("--seed", type=int, default=1, help="instance seed (default 1)")
    parser.add_argument("--planner", choices=PLANNERS, default="cbs")
    parser.add_argument("--no-animate", action="store_true", help="print only the stats and the final map")
    parser.add_argument("--delay", type=float, default=0.25, help="seconds per frame (default 0.25)")
    parser.add_argument("--max-nodes", type=int, default=CBS_NODES, help=f"CBS node budget (default {CBS_NODES})")
    parser.add_argument("--max-seconds", type=float, default=CBS_SECONDS,
                        help=f"CBS time budget in seconds (default {CBS_SECONDS})")
    args = parser.parse_args(argv)

    layout, robots = args.layout, args.robots
    if sys.stdin.isatty() and (layout is None or robots is None):
        menu_layout, menu_robots = pick_menu()
        layout = layout or menu_layout
        robots = robots or menu_robots
    layout = layout or DEFAULT_LAYOUT
    robots = robots or DEFAULT_ROBOTS

    try:
        # Generate with the same CBS budget that solves below, so the instance is known to fit it.
        problem = random_instance(layout, robots, args.seed, max_nodes=args.max_nodes, max_seconds=args.max_seconds)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    limits = {"max_nodes": args.max_nodes, "max_seconds": args.max_seconds} if args.planner == "cbs" else {}
    result = solve(problem, args.planner, **limits)

    print(f"layout: {LAYOUT_INFO[layout]['title']} ({layout}), {robots} robots, seed {args.seed}")
    if args.no_animate or result.paths is None:
        for line in describe(result):
            print(line)
        if result.paths is not None:
            print(render(problem, result.paths, result.makespan))
        else:
            print("No collision-free plan was produced. Try another seed, a larger budget, or another planner.")
        return 0 if result.paths is not None else 1
    animate(problem, result, args.delay)
    for line in describe(result):
        print(line)
    return 0


def _robot_count(text: str) -> int:
    value = int(text)
    if not 1 <= value <= MAX_ROBOTS:
        raise argparse.ArgumentTypeError(f"robots must be between 1 and {MAX_ROBOTS}")
    return value


def benchmark(argv: list[str]) -> int:
    """Run every planner on the same random instances and print one comparison table.

    Instances are seeds ``seed .. seed + instances - 1``. Generation (which itself runs CBS to
    confirm solvability) happens before the timers start, so the seconds column measures planning only.
    Generation uses the same CBS budget as the comparison, so every generated instance is one CBS can solve.
    """
    parser = argparse.ArgumentParser(prog="python -m warehouse benchmark")
    parser.add_argument("--layout", choices=list(LAYOUTS), default=DEFAULT_LAYOUT)
    parser.add_argument("--robots", type=_robot_count, default=DEFAULT_ROBOTS)
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=CBS_NODES)
    parser.add_argument("--max-seconds", type=float, default=CBS_SECONDS)
    args = parser.parse_args(argv)

    problems = []
    for seed in range(args.seed, args.seed + args.instances):
        try:
            problems.append(random_instance(args.layout, args.robots, seed,
                                            max_nodes=args.max_nodes, max_seconds=args.max_seconds))
        except ValueError:
            pass  # no solvable draw in the attempt budget; counted as skipped below
    if not problems:
        print("no solvable instances generated", file=sys.stderr)
        return 2

    results: dict[str, list[PlanResult]] = {p: [] for p in PLANNERS}
    for problem in problems:
        for planner in PLANNERS:
            limits = {"max_nodes": args.max_nodes, "max_seconds": args.max_seconds} if planner == "cbs" else {}
            results[planner].append(solve(problem, planner, **limits))

    n = len(problems)
    skipped = args.instances - n
    print(f"{LAYOUT_INFO[args.layout]['title']} ({args.layout}), {args.robots} robots, {n} instances "
          f"(seeds {args.seed}-{args.seed + args.instances - 1}"
          f"{f', {skipped} skipped' if skipped else ''})")
    header = (f"{'planner':<22}{'solved':>8}{'mean SOC':>10}{'makespan':>10}"
              f"{'collisions':>12}{'CBS nodes':>11}{'seconds':>9}")
    print(header)
    print("-" * len(header))
    for planner in PLANNERS:
        rs = results[planner]
        # "solved" counts collision-free plans. Independent plans always exist, but may collide.
        solved = sum(r.status == "solved" for r in rs)
        with_paths = [r for r in rs if r.paths is not None]
        soc = sum(r.sum_of_costs for r in with_paths) / len(with_paths) if with_paths else float("nan")
        span = sum(r.makespan for r in with_paths) / len(with_paths) if with_paths else float("nan")
        coll = sum(len(r.conflicts) for r in rs) / n
        nodes = f"{sum(r.high_nodes for r in rs) / n:.1f}" if planner == "cbs" else "-"
        secs = sum(r.seconds for r in rs) / n
        print(f"{PLANNER_INFO[planner]['title']:<22}{f'{solved}/{n}':>8}{soc:>10.1f}{span:>10.1f}"
              f"{coll:>12.2f}{nodes:>11}{secs:>9.4f}")

    both = [(c, p) for c, p in zip(results["cbs"], results["prioritized"])
            if c.status == "solved" and p.status == "solved"]
    if both:
        worse = sum(p.sum_of_costs > c.sum_of_costs for c, p in both)
        extra = sum(p.sum_of_costs - c.sum_of_costs for c, p in both) / len(both)
        print(f"prioritized is suboptimal on {worse}/{len(both)} instances where both solved "
              f"(mean extra cost {extra:.2f})")
    failed = sum(r.status == "failed" for r in results["prioritized"])
    if failed:
        print(f"prioritized failed to route a robot on {failed}/{n} instances")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["benchmark"]:
        return benchmark(argv[1:])
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
