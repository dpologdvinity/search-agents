"""Benchmark: shots to sink the whole fleet, per agent, on the same random fleets.

    python -m battleship benchmark                 # 10 fleets, seed 1
    python -m battleship benchmark --games 100 --seed 7

Every agent plays the same fleets, so the comparison is paired: differences come from the strategies, not from
luck in the fleet draw. Each agent's own generator is seeded from the game index, so runs repeat exactly.
"""

from __future__ import annotations

import argparse
import statistics
import time
from random import Random

from .agents import AGENTS, make_agent
from .board import CELLS, FLEET, random_fleet
from .game import play_out

ORDER = ("random", "hunt", "probability")


def run(games: int, seed: int = 1, names=ORDER) -> dict[str, dict]:
    """Play `games` fleets with each named agent. Returns {name: {"shots": [...], "seconds": total}}."""
    results = {name: {"shots": [], "seconds": 0.0} for name in names}
    for g in range(games):
        fleet = random_fleet(Random(seed * 1_000_003 + g))
        for i, name in enumerate(names):
            agent = make_agent(name, Random(seed * 1_000_003 + g * 31 + i))
            start = time.perf_counter()
            results[name]["shots"].append(play_out(agent, fleet))
            results[name]["seconds"] += time.perf_counter() - start
    return results


def table(results: dict[str, dict]) -> str:
    """Plain-text table: one row per agent."""
    rows = [f"{'agent':<12}{'games':>6}{'mean shots':>12}{'median':>8}{'worst':>7}{'s/game':>9}"]
    for name, r in results.items():
        shots = r["shots"]
        rows.append(f"{name:<12}{len(shots):>6}{statistics.fmean(shots):>12.1f}{statistics.median(shots):>8.1f}"
                    f"{max(shots):>7}{r['seconds'] / len(shots):>9.3f}")
    rows.append(f"(ships occupy {sum(FLEET)} of the {CELLS} cells; shots to sink them all, lower is better)")
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m battleship benchmark", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=10, help="fleets to play (default 10)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    parser.add_argument("--agents", nargs="+", choices=sorted(AGENTS), default=list(ORDER),
                        help="agents to compare (default: all three)")
    args = parser.parse_args(argv)
    print(table(run(args.games, args.seed, tuple(args.agents))))
    return 0
