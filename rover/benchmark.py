"""Benchmark: D* Lite against A* replanned from scratch, on seeded random maps of several sizes and densities.

    python -m rover benchmark                       # the full run committed under results/
    python -m rover benchmark --sizes 21 --seeds 1  # a quick check (use --out /tmp/... to keep results/ untouched)

Each map is explored twice: once steered by D* Lite and once steered by A*. In both runs both planners
see the same belief changes, so each run reports the work of both planners on one route. A map is
identified by (size, density, seed), and the seed fixes the map and nothing else, so a run repeats exactly.

Measured per map:
  expansions  queue pops (D* Lite) or closed cells (A*). Split into the initial plan on an all-free map and
              the replans that follow sensing. The initial plan is shown separately because it is a one-off
              cost that does not grow with the number of replans.
  seconds     time spent inside each planner (perf_counter, pure Python, one core).
  mismatches  replans where D* Lite and A* disagree on the cost from the rover's cell. This must be 0.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .explorer import Explorer
from .world import World, generate

SIZES = (21, 41, 61)
DENSITIES = (0.10, 0.20, 0.30)
SEEDS = 5
RADIUS = 2


def run_map(size: int, density: float, seed: int, radius: int = RADIUS) -> dict:
    """Explore one map twice (D*-driven and A*-driven) and collect the numbers above.

    Expansions and time come from the D*-driven run, so both planners are measured on the same route.
    The A*-driven run only supplies its own route length, and its cross-check counts go into `mismatches`.
    """
    def explore(driver: str) -> Explorer:
        ex = Explorer(World(size, generate(size, density, seed)), radius=radius, driver=driver).run()
        assert ex.done, f"the rover did not reach the goal on {size}x{size} d={density} seed={seed}"
        return ex

    d = explore("dstar")
    a = explore("astar")
    row = {"size": size, "density": density, "seed": seed,
           "steps_dstar": d.steps, "steps_astar": a.steps, "replans": d.replans,
           "mismatches": d.mismatches + a.mismatches}
    for name in ("dstar", "astar"):
        row[f"init_{name}"] = d.init_expansions[name]
        row[f"replan_{name}"] = d.expansions(name) - d.init_expansions[name]
        row[f"total_{name}"] = d.expansions(name)
        row[f"ms_{name}"] = round(d.seconds[name] * 1000, 1)
    return row


def summarise(rows: list[dict]) -> dict:
    """Averages over the maps of one (size, density) cell, plus the ratios the table reports."""
    def mean(key):
        return statistics.fmean(r[key] for r in rows)
    out = {k: round(mean(k), 1) for k in (
        "steps_dstar", "steps_astar", "replans", "init_dstar", "replan_dstar", "total_dstar",
        "init_astar", "replan_astar", "total_astar", "ms_dstar", "ms_astar")}
    out["maps"] = len(rows)
    out["mismatches"] = sum(r["mismatches"] for r in rows)
    out["ratio_total"] = round(out["total_dstar"] / out["total_astar"], 3)
    out["ratio_replans"] = round(out["replan_dstar"] / out["replan_astar"], 3) if out["replan_astar"] else None
    return out


def run(sizes=SIZES, densities=DENSITIES, seeds: int = SEEDS, radius: int = RADIUS) -> dict:
    """Run the whole matrix and return {"config": ..., "rows": per map, "summary": per (size, density)}."""
    rows, summary = [], []
    for size in sizes:
        for density in densities:
            cell = [run_map(size, density, seed, radius) for seed in range(seeds)]
            rows.extend(cell)
            summary.append({"size": size, "density": density, **summarise(cell)})
    return {
        "config": {"sizes": list(sizes), "densities": list(densities), "seeds": seeds, "radius": radius,
                   "connectivity": 4, "heuristic": "manhattan", "unknown_cells": "planned as free"},
        "rows": rows,
        "summary": summary,
    }


def markdown(result: dict) -> str:
    """The results table, as the README and results/rover_benchmark.md show it."""
    c = result["config"]
    lines = [
        f"Sensor radius {c['radius']} (square), {c['seeds']} seeded maps per row, 4-connected grid, "
        "Manhattan heuristic, unknown cells planned as free. Start top-left, goal bottom-right. "
        "Both runs drive the same routes with the same belief changes; expansions are queue pops (D* Lite) "
        "or closed cells (A*), averaged over the maps.",
        "",
        "| size | density | steps | replans | D* init exp | D* replan exp | A* init exp | A* replan exp | "
        "D* total / A* total | D* ms | A* ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in result["summary"]:
        lines.append(
            f"| {s['size']}x{s['size']} | {int(round(s['density'] * 100))}% | {s['steps_dstar']:.0f} | "
            f"{s['replans']:.1f} | {s['init_dstar']:.0f} | {s['replan_dstar']:.0f} | {s['init_astar']:.0f} | "
            f"{s['replan_astar']:.0f} | {s['total_dstar']:.0f} / {s['total_astar']:.0f} "
            f"({s['ratio_total']:.2f}x) | {s['ms_dstar']:.0f} | {s['ms_astar']:.0f} |")
    mism = sum(r["mismatches"] for r in result["rows"])
    replans = sum(r["replans"] for r in result["rows"])
    lines += ["", f"Cost check: {mism} replans where D* Lite and A* disagreed on the cost from the rover's cell "
              f"(out of {replans} replans across {len(result['rows'])} maps)."]
    return "\n".join(lines) + "\n"


def write(result: dict, out_dir: Path) -> None:
    """Write rover_benchmark.json (every map) and rover_benchmark.md (the table) into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rover_benchmark.json").write_text(json.dumps(result, indent=1) + "\n")
    (out_dir / "rover_benchmark.md").write_text(markdown(result))
