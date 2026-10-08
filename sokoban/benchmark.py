"""Compare the search configurations on every built-in level.

The configurations are the four search rules (BFS, greedy with the matching heuristic, A* with the
simple heuristic, A* with the matching heuristic), each with and without deadlock pruning. Every run
gets the same node budget and time limit. A run that hits either is reported as unsolved, not as a
guess, so a column of "budget" marks a configuration that could not finish.

Reported per configuration: levels solved within the budget, the nodes expanded on each level, the
time, whether the plan is optimal (same pushes as A* with the matching heuristic), and the pushes.

    python -m sokoban benchmark                       # table on the terminal
    python -m sokoban benchmark --out results/sokoban_benchmark   # also .json and .md files
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

from .heuristics import MATCHING, NONE, SIMPLE
from .levels import NAMES, OPTIMAL_PUSHES, all_levels
from .search import ASTAR, BFS, DEFAULT_NODES, DEFAULT_SECONDS, GREEDY, Result, solve

# (label, algorithm, heuristic). Prune on and off is run for each of these.
CONFIGS = (
    ("BFS", BFS, NONE),
    ("Greedy (matching)", GREEDY, MATCHING),
    ("A* (simple)", ASTAR, SIMPLE),
    ("A* (matching)", ASTAR, MATCHING),
)


@dataclass
class Run:
    config: str
    prune: bool
    level: str
    status: str
    expanded: int
    generated: int
    pruned: int
    seconds: float
    pushes: int | None
    optimal: bool  # the plan has the optimal push count (A* with matching found it)


def run_benchmark(node_budget: int = DEFAULT_NODES, time_limit: float = DEFAULT_SECONDS, levels=None) -> list[Run]:
    """Run every configuration on every level (or the given subset). Returns one Run per pair."""
    levels = all_levels() if levels is None else levels
    runs: list[Run] = []
    for name, algo, heur in CONFIGS:
        for prune in (False, True):
            for lv in levels:
                r: Result = solve(lv, algo, heur, prune, node_budget=node_budget, time_limit=time_limit)
                optimal = r.solved and r.pushes == OPTIMAL_PUSHES[lv.name]
                runs.append(Run(name, prune, lv.name, r.status, r.expanded, r.generated, r.pruned,
                                round(r.seconds, 4), r.pushes, optimal))
    return runs


def summarize(runs: list[Run]) -> list[dict]:
    """One row per (configuration, pruning) with the totals the table shows."""
    rows = []
    for name, _, _ in CONFIGS:
        for prune in (False, True):
            group = [r for r in runs if r.config == name and r.prune == prune]
            solved = [r for r in group if r.status == "solved"]
            rows.append({
                "config": name,
                "prune": prune,
                "solved": len(solved),
                "levels": len(group),
                "optimal": sum(1 for r in group if r.optimal),
                "expanded_solved": sum(r.expanded for r in solved),
                "expanded_all": sum(r.expanded for r in group),
                "seconds": round(sum(r.seconds for r in group), 2),
                "pruned": sum(r.pruned for r in group),
            })
    return rows


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def render_table(runs: list[Run], budget: int, seconds: float) -> str:
    """Markdown: the summary table, then a per-level table of nodes expanded for each configuration."""
    lines = [
        f"Node budget {_fmt_int(budget)} and {seconds:g} s per run. Pushes are counted for solved runs; "
        "\"optimal\" means the pushes equal the optimal count.",
        "",
        "| search | deadlock pruning | solved | optimal | nodes expanded (solved levels) "
        "| nodes expanded (all) | time (all) | states pruned |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summarize(runs):
        lines.append(
            f"| {row['config']} | {'on' if row['prune'] else 'off'} | {row['solved']}/{row['levels']} | "
            f"{row['optimal']}/{row['levels']} | {_fmt_int(row['expanded_solved'])} | "
            f"{_fmt_int(row['expanded_all'])} | {row['seconds']:.1f} s | {_fmt_int(row['pruned'])} |"
        )
    lines += ["", "Nodes expanded per level (`-` unsolved within the budget; `*` solved but not optimal):", ""]
    header = "| level | optimal pushes | " + " | ".join(
        f"{name} {'+P' if prune else ''}".strip() for name, _, _ in CONFIGS for prune in (False, True)) + " |"
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * (2 * len(CONFIGS)))
    for lv in NAMES:
        cells = []
        for name, _, _ in CONFIGS:
            for prune in (False, True):
                r = next(x for x in runs if x.config == name and x.prune == prune and x.level == lv)
                if r.status != "solved":
                    cells.append("-")
                else:
                    cells.append(f"{_fmt_int(r.expanded)}{'' if r.optimal else '*'}")
        lines.append(f"| {lv} | {OPTIMAL_PUSHES[lv]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results(path: Path, runs: list[Run], budget: int, seconds: float) -> None:
    """Write `<path>.json` (every run) and `<path>.md` (the tables)."""
    payload = {
        "node_budget": budget,
        "time_limit_seconds": seconds,
        "machine": platform.platform(),
        "python": platform.python_version(),
        "summary": summarize(runs),
        "runs": [asdict(r) for r in runs],
    }
    path.with_suffix(".json").write_text(json.dumps(payload, indent=1) + "\n")
    path.with_suffix(".md").write_text(render_table(runs, budget, seconds) + "\n")
