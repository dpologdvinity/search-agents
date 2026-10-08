"""Generate a benchmark set of uniformly random 15-puzzle states with optimal costs.

Optimal costs come from IDA* with the 5-5-5 pattern database, solved in
parallel. Uniformly random states average about 53 moves, so this takes a
while in pure Python.

    python -m npuzzle.testset --count 100 --seed 0 --workers 10
"""

from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

from .board import is_solvable
from .heuristics import pattern_database
from .iterative import ida_star

DATA = Path(__file__).parent / "data"


def random_states(count: int, seed: int, n: int = 4) -> list[tuple[int, ...]]:
    rng = random.Random(seed)
    states = []
    while len(states) < count:
        config = list(range(n * n))
        rng.shuffle(config)
        if is_solvable(tuple(config), n):
            states.append(tuple(config))
    return states


def _solve(config):
    start = time.perf_counter()
    result = ida_star(config, heuristic=pattern_database)
    return {
        "board": list(config),
        "optimal_cost": result.cost,
        "expanded": result.expanded,
        "seconds": round(time.perf_counter() - start, 2),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args(argv)

    out = DATA / f"testset_15_seed{args.seed}.json"
    states = random_states(args.count, args.seed)
    # Resume: keep boards already solved by an earlier run.
    done = {}
    if out.exists():
        done = {tuple(r["board"]): r for r in json.loads(out.read_text())["states"]}
    todo = [s for s in states if s not in done]
    print(f"{len(done)} already solved, {len(todo)} to go", flush=True)

    def save():
        # Keep the generation order, whatever order instances finish in.
        rows = [done[s] for s in states if s in done]
        out.write_text(json.dumps({"seed": args.seed, "states": rows}, indent=1))

    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(_solve, todo):
            done[tuple(row["board"])] = row
            save()
            print(f"{len(done)}/{len(states)} cost={row['optimal_cost']} "
                  f"nodes={row['expanded']:,} {row['seconds']}s", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
