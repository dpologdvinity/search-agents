"""Solve a traveling-salesman map in the terminal: random cities, or your own CSV file.

    python -m routes                                 # 20 random cities, every solver compared
    python -m routes --cities 12 --seed 7 --compare
    python -m routes --solver simulated_annealing --iterations 200000
    python -m routes --file cities.csv               # one "x,y" per line; a header line is skipped

Each solver reports its tour length and time. The best tour is drawn on a character grid:
cities are labelled 0-9, then A-Z and a-z (cities past 62 show as *), and the route is dots.
"""

from __future__ import annotations

import argparse
import csv
import random
import string
import sys
import time

from .tsp import (
    HELD_KARP_LIMIT,
    SOLVERS,
    Result,
    genetic_algorithm,
    held_karp,
    nearest_neighbor_2opt,
    simulated_annealing,
)

LABELS = string.digits + string.ascii_uppercase + string.ascii_lowercase


def random_cities(n: int, seed: int) -> list[tuple[float, float]]:
    """n uniform random points in the unit square, the same coordinates the web page uses."""
    rng = random.Random(seed)
    return [(rng.random(), rng.random()) for _ in range(n)]


def load_cities(path: str) -> list[tuple[float, float]]:
    """Cities from a CSV file with one x,y pair per line.

    Blank lines are skipped. A first line that is not numbers (a header such as x,y) is skipped
    too; a bad line anywhere else is an error, so a typo is not silently dropped from the map.
    """
    cities = []
    with open(path, newline="") as f:
        for number, row in enumerate(csv.reader(f), 1):
            if not row:
                continue
            try:
                cities.append((float(row[0]), float(row[1])))
            except (ValueError, IndexError):
                if number == 1:
                    continue
                raise ValueError(f"line {number}: expected x,y but found {','.join(row)!r}") from None
    return cities


def given(**options) -> dict:
    """The options the user set; None means "not set", so the solver keeps its own default."""
    return {k: v for k, v in options.items() if v is not None}


def solve(name: str, cities, args) -> tuple[Result, float]:
    """Run one solver with the command-line settings. Returns its result and the seconds it took."""
    start = time.perf_counter()
    if name == "held_karp":
        result = held_karp(cities)
    elif name == "simulated_annealing":
        result = simulated_annealing(cities, **given(iterations=args.iterations, cooling=args.cooling, seed=args.seed))
    elif name == "genetic_algorithm":
        result = genetic_algorithm(cities, **given(population_size=args.population, generations=args.generations,
                                                   mutation_rate=args.mutation, seed=args.seed))
    else:  # nearest_neighbor_2opt is the only solver left, and it takes no settings
        result = nearest_neighbor_2opt(cities)
    return result, time.perf_counter() - start


def compare(cities, args) -> list[tuple[str, Result, float]]:
    """Run every solver on the map. held_karp joins only when the map is small enough to solve exactly."""
    names = [*SOLVERS, "held_karp"] if len(cities) <= HELD_KARP_LIMIT else list(SOLVERS)
    return [(name, *solve(name, cities, args)) for name in names]


def table(rows, optimum: float | None) -> str:
    """Solver results as a text table. The last column is how far above the exact optimum each tour is."""
    lines = [f"{'solver':<24}{'length':>10}{'above optimal':>16}{'seconds':>10}"]
    for name, result, seconds in rows:
        # Summing the same edges in another order can leave a tour a rounding error below the
        # optimum, so the gap is clamped at zero rather than printing "-0.00%".
        gap = "-" if optimum is None else f"{max(0.0, 100 * (result.length / optimum - 1)):.2f}%"
        lines.append(f"{name:<24}{result.length:>10.3f}{gap:>16}{seconds:>10.3f}")
    return "\n".join(lines)


def plot(cities, tour, width: int = 60, height: int = 30) -> str:
    """Draw a tour on a character grid: the route as dots, then each city's label on top.

    Each axis is stretched to fill the grid. Rows count from the top, so larger y is higher on
    the page. Each edge is walked one character at a time, so the dots form an unbroken line
    from one city to the next.
    """
    xs = [x for x, _ in cities]
    ys = [y for _, y in cities]
    x0, y0, y1 = min(xs), min(ys), max(ys)
    span_x = (max(xs) - x0) or 1.0  # the "or 1.0" keeps a map with one shared coordinate from dividing by zero
    span_y = (y1 - y0) or 1.0
    cells = [(round((y1 - y) / span_y * (height - 1)), round((x - x0) / span_x * (width - 1))) for x, y in cities]

    grid = [[" "] * width for _ in range(height)]
    edges = zip(tour, tour[1:] + tour[:1])  # the last city links back to the first, closing the tour
    for a, b in edges:
        (r1, c1), (r2, c2) = cells[a], cells[b]
        steps = max(abs(r2 - r1), abs(c2 - c1))
        for s in range(steps + 1):
            t = s / steps if steps else 0.0
            grid[round(r1 + (r2 - r1) * t)][round(c1 + (c2 - c1) * t)] = "."
    for i, (r, c) in enumerate(cells):
        grid[r][c] = LABELS[i] if i < len(LABELS) else "*"

    border = "+" + "-" * width + "+"
    return "\n".join([border, *("|" + "".join(row) + "|" for row in grid), border])


def main(argv=None) -> int:
    """Parse the options, build or load the map, and run the solvers. Returns the exit status."""
    parser = argparse.ArgumentParser(prog="python -m routes", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cities", type=int, default=20, help="random map size (default 20)")
    parser.add_argument("--file", help="load cities from a CSV of x,y lines instead of a random map")
    parser.add_argument("--seed", type=int, default=1, help="random seed for the map and the solvers (default 1)")
    parser.add_argument("--solver", choices=[*SOLVERS, "held_karp"], help="run one solver instead of comparing all")
    parser.add_argument("--compare", action="store_true",
                        help="run every solver and print a table (the default when --solver is not given)")
    parser.add_argument("--iterations", type=int, help="simulated annealing: 2-opt moves to try")
    parser.add_argument("--cooling", type=float, help="simulated annealing: temperature multiplier per move")
    parser.add_argument("--population", type=int, help="genetic algorithm: tours per generation")
    parser.add_argument("--generations", type=int, help="genetic algorithm: number of generations")
    parser.add_argument("--mutation", type=float, help="genetic algorithm: chance that a child is mutated")
    args = parser.parse_args(argv)

    if args.file:
        try:
            cities = load_cities(args.file)
        except (OSError, ValueError) as e:
            parser.error(str(e))
        print(f"{len(cities)} cities from {args.file}")
    else:
        cities = random_cities(args.cities, args.seed)
        print(f"{len(cities)} random cities, seed {args.seed}")
    if len(cities) < 3:
        parser.error(f"a tour needs at least 3 cities, got {len(cities)}")

    if args.solver and not args.compare:
        if args.solver == "held_karp" and len(cities) > HELD_KARP_LIMIT:
            parser.error(f"held_karp handles at most {HELD_KARP_LIMIT} cities")
        result, seconds = solve(args.solver, cities, args)
        print(f"{args.solver}: length {result.length:.3f}, {result.steps:,} steps, {seconds:.3f} s")
        print(plot(cities, result.tour))
        return 0

    rows = compare(cities, args)
    optimum = next((result.length for name, result, _ in rows if name == "held_karp"), None)
    print(table(rows, optimum))
    if optimum is None:
        print(f"held_karp skipped: it solves at most {HELD_KARP_LIMIT} cities exactly")
    name, best, _ = min(rows, key=lambda row: row[1].length)
    print(f"\nbest tour: {name}\n{plot(cities, best.tour)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
