"""Watch a rover explore an unknown maze and replan as it finds walls.

    python -m rover                          # 21x21 map, 20% walls, D* Lite drives, animated in the terminal
    python -m rover --driver astar           # the same map, steered by A* replanned from scratch
    python -m rover --size 41 --density 0.3 --seed 7 --radius 3 --delay 0.03
    python -m rover benchmark                # the seeded benchmark committed under results/

In the picture, '#' is a wall the rover has sensed, '.' a free cell it has sensed, ' ' a cell it has not
seen yet (the rover does not know what is there), 'R' the rover, 'G' the goal, '*' the planned route. The
rover senses a square of cells around itself each step. A map is drawn from its (size, density, seed), so
the same arguments give the same maze here, on the web page, and in the tests.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .explorer import PLANNERS, Explorer
from .world import World, generate, render

ROOT = Path(__file__).resolve().parent.parent
CLEAR = "\033[H\033[2J"


def status_line(ex: Explorer) -> str:
    """One line of counters under the map: progress, replans, and both planners' work so far."""
    state = "GOAL" if ex.done else ("WAITING (no known route)" if ex.stuck else "exploring")
    return (f"step {ex.steps:4d}  {state:<24} replans {ex.replans:4d}   "
            f"D* Lite exp {ex.expansions('dstar'):7d}   A* exp {ex.expansions('astar'):7d}   "
            f"driver {ex.driver}")


def play(size: int, density: float, seed: int, radius: int, driver: str, delay: float, clear: bool,
         out=sys.stdout) -> Explorer:
    """Animate one exploration. Each frame shows the map as the rover knows it and the counters."""
    world = World(size, generate(size, density, seed))
    ex = Explorer(world, radius=radius, driver=driver)
    tty = clear and out.isatty()
    while True:
        path = ex.path()
        frame = [status_line(ex), render(size, world.walls, ex.seen, ex.pos, world.goal, path)]
        if tty:
            out.write(CLEAR)
        out.write("\n".join(frame) + "\n")
        out.flush()
        if ex.done:
            break
        ex.tick()
        if delay and tty:
            time.sleep(delay)  # pause only when animating on a terminal; piped output prints every frame at once
    out.write(f"\nreached the goal in {ex.steps} steps with {ex.replans} replans. "
              f"D* Lite expanded {ex.expansions('dstar')} cells and A* {ex.expansions('astar')}.\n")
    return ex


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Rover exploring an unknown maze: D* Lite (incremental replanning) versus A* from scratch.")
    parser.add_argument("command", nargs="?", default="play", choices=["play", "benchmark"],
                        help="play (default) animates one route; benchmark runs the seeded comparison")
    parser.add_argument("--size", type=int, default=21, help="grid side length, 5..81 (default 21)")
    parser.add_argument("--density", type=float, default=0.2, help="fraction of walls, 0..0.4 (default 0.2)")
    parser.add_argument("--seed", type=int, default=1, help="seed for the map (default 1)")
    parser.add_argument("--radius", type=int, default=2, help="sensor half-width in cells, 1..4 (default 2)")
    parser.add_argument("--driver", choices=PLANNERS, default="dstar",
                        help="the planner that steers the rover; the other one replans alongside it")
    parser.add_argument("--delay", type=float, default=0.06, help="seconds between frames on a terminal (default 0.06)")
    parser.add_argument("--no-clear", action="store_true", help="print frames one after another, without clearing")
    parser.add_argument("--sizes", default=None, help="benchmark: comma-separated sizes (default 21,41,61)")
    parser.add_argument("--densities", default=None, help="benchmark: comma-separated densities (default 0.1,0.2,0.3)")
    parser.add_argument("--seeds", type=int, default=None, help="benchmark: maps per cell (default 5)")
    parser.add_argument("--out", default=str(ROOT / "results"), help="benchmark: directory for the results files")
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        from . import benchmark

        sizes = [int(x) for x in args.sizes.split(",")] if args.sizes else benchmark.SIZES
        densities = [float(x) for x in args.densities.split(",")] if args.densities else benchmark.DENSITIES
        seeds = args.seeds if args.seeds is not None else benchmark.SEEDS
        result = benchmark.run(sizes, densities, seeds, radius=args.radius)
        benchmark.write(result, Path(args.out))
        print(benchmark.markdown(result))
        return 0

    if not 5 <= args.size <= 81:
        parser.error("--size must be between 5 and 81")
    if not 0 <= args.density <= 0.4:
        parser.error("--density must be between 0 and 0.4")
    if not 1 <= args.radius <= 4:
        parser.error("--radius must be between 1 and 4")
    play(args.size, args.density, args.seed, args.radius, args.driver, args.delay, not args.no_clear)
    return 0


if __name__ == "__main__":
    sys.exit(main())
