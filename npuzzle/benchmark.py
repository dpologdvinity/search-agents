"""Benchmark N-Puzzle algorithms and heuristics on fixed board sets.

    python -m npuzzle.benchmark 8puzzle
    python -m npuzzle.benchmark 15puzzle --seconds 60

Writes per-board rows to results/npuzzle_<suite>.json and prints a
Markdown summary table. Each configuration runs one board at a time, so
times are comparable within a run.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict
from functools import partial
from pathlib import Path

from . import ALGORITHMS
from .batched import batch_heuristic, batch_weighted_a_star
from .board import is_solvable
from .heuristics import linear_conflict, manhattan, pattern_database
from .iterative import ida_star
from .search import SearchLimits, a_star, bfs
from .testset import DATA

RESULTS = Path("results")


def random_boards(count, n, seed):
    rng = random.Random(seed)
    boards = []
    while len(boards) < count:
        config = list(range(n * n))
        rng.shuffle(config)
        if is_solvable(tuple(config), n):
            boards.append(tuple(config))
    return boards


def configs_8puzzle():
    return {
        **ALGORITHMS,
        "astar+lc": partial(a_star, heuristic=linear_conflict),
        "idastar+lc": partial(ida_star, heuristic=linear_conflict),
    }


def configs_15puzzle():
    configs = {
        "idastar+manhattan": partial(ida_star, heuristic=manhattan),
        "idastar+lc": partial(ida_star, heuristic=linear_conflict),
        "idastar+pdb": partial(ida_star, heuristic=pattern_database),
    }
    # g_weight 1.0 is close to optimal A* and as slow; the test set already
    # records optimal IDA* + PDB results, so the sweep covers faster settings.
    for h in ("neural", "pdb"):
        for size in (100, 1000):
            for w in (0.8, 0.6):
                configs[f"bwas+{h} b={size} w={w}"] = partial(
                    batch_weighted_a_star, batch_heuristic=batch_heuristic(h, 4), batch_size=size, g_weight=w)
    return configs


def run_suite(boards, optimal, configs, limits, only=None):
    rows = []
    for name, solve in configs.items():
        if only and not any(o in name for o in only):
            continue
        for i, config in enumerate(boards):
            r = solve(config, limits=limits)
            row = asdict(r)
            del row["path"]
            row.update(config=name, board=i, optimal_cost=optimal[i])
            rows.append(row)
        print(f"done: {name}", flush=True)
    return rows


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r["config"], []).append(r)
    lines = [
        "| Configuration | Solved | Mean nodes | Mean time (s) | Mean cost | Optimal paths | Mean gap |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, rs in by.items():
        solved = [r for r in rs if r["status"] == "solved"]
        if not solved:
            lines.append(f"| {name} | 0/{len(rs)} | - | - | - | - | - |")
            continue
        gap = [(r["cost"] - r["optimal_cost"]) / r["optimal_cost"] for r in solved]
        lines.append(
            f"| {name} | {len(solved)}/{len(rs)} "
            f"| {statistics.mean(r['expanded'] for r in solved):,.0f} "
            f"| {statistics.mean(r['seconds'] for r in solved):.3f} "
            f"| {statistics.mean(r['cost'] for r in solved):.1f} "
            f"| {sum(r['cost'] == r['optimal_cost'] for r in solved)}/{len(solved)} "
            f"| {100 * statistics.mean(gap):.1f}% |"
        )
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("suite", choices=["8puzzle", "15puzzle"])
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seconds", type=float, default=60, help="time limit per board")
    p.add_argument("--max-nodes", type=int, default=5_000_000)
    p.add_argument("--only", nargs="*", help="run configurations whose name contains any of these")
    args = p.parse_args(argv)

    limits = SearchLimits(max_nodes=args.max_nodes, max_seconds=args.seconds)
    if args.suite == "8puzzle":
        boards = random_boards(args.count, 3, seed=0)
        optimal = [bfs(b).cost for b in boards]
        configs = configs_8puzzle()
    else:
        data = json.loads((DATA / "testset_15_seed0.json").read_text())["states"][: args.count]
        boards = [tuple(r["board"]) for r in data]
        optimal = [r["optimal_cost"] for r in data]
        configs = configs_15puzzle()

    rows = run_suite(boards, optimal, configs, limits, args.only)
    RESULTS.mkdir(exist_ok=True)
    suffix = "" if not args.only else "_" + "_".join(args.only).replace(" ", "")
    out = RESULTS / f"npuzzle_{args.suite}{suffix}.json"
    out.write_text(json.dumps(rows, indent=1))
    table = summarize(rows)
    (RESULTS / f"npuzzle_{args.suite}{suffix}.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
