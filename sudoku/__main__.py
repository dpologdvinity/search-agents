"""Solve Sudoku puzzles or benchmark the solvers.

    python -m sudoku solve 003020600900305001001806400008102900700000008006708200002609500800203009005010300
    python -m sudoku benchmark
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .generate import load_puzzles
from .solver import SOLVERS

# Well-known hard puzzles published by their authors.
HARD = {
    "inkala_2012": "800000000003600000070090000050007000000045700000100030001000068068000400000000000",
    "ai_escargot": "100007090030020008009600500005300900010080002600004000300000010040000007007000300",
    "norvig_hard1": "400000805030000000000700000020000060000080400000010000000603070500200000104000000",
    "platinum_blonde": "000000012000000003002300400001800005060070800000009000008500000900040500470006000",
}


def benchmark(max_nodes):
    puzzles = [row["puzzle"] for row in load_puzzles()]
    rows = []
    lines = ["| Solver | Set | Solved | Mean nodes | Max nodes | Mean backtracks | Total time (s) |",
             "|---|---|---|---|---|---|---|"]
    for name, solve in SOLVERS.items():
        for label, items in ((f"{len(puzzles)} generated puzzles", puzzles), ("4 hard puzzles", list(HARD.values()))):
            results = [solve(p, max_nodes=max_nodes) for p in items]
            solved = [r for r in results if r.status == "solved"]
            rows += [{"solver": name, "set": label, "status": r.status, "nodes": r.nodes,
                      "backtracks": r.backtracks, "seconds": r.seconds} for r in results]
            lines.append(
                f"| {name} | {label} | {len(solved)}/{len(items)} "
                f"| {statistics.mean(r.nodes for r in results):,.0f} | {max(r.nodes for r in results):,} "
                f"| {statistics.mean(r.backtracks for r in results):,.0f} "
                f"| {sum(r.seconds for r in results):.2f} |")
            print(lines[-1], flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/sudoku.json").write_text(json.dumps(rows, indent=1))
    Path("results/sudoku.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sudoku")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("puzzle")
    s.add_argument("--solver", choices=sorted(SOLVERS), default="propagate")
    b = sub.add_parser("benchmark")
    b.add_argument("--max-nodes", type=int, default=2_000_000)
    args = p.parse_args(argv)
    if args.cmd == "benchmark":
        benchmark(args.max_nodes)
        return 0
    result = SOLVERS[args.solver](args.puzzle)
    if result.status != "solved":
        print(result.status, file=sys.stderr)
        return 1
    print(result.solution)
    print(f"{result.nodes} nodes, {result.backtracks} backtracks, {result.seconds:.3f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
