"""Benchmark: success rate, steps and time for each agent across board sizes.

    python -m queens benchmark              # the full run committed under results/
    python -m queens benchmark --quick      # a fast check with fewer trials and shorter caps

Every run has a time cap (and a step cap where the agent has one), so a run that cannot finish is
recorded as a failure with its reason, never as a hang. Trial i of every agent uses the seed
SEED_BASE + i, so the agents at a given size start from comparable random placements. Backtracking is
deterministic, so it runs once per size.

Steps mean different things per agent (see queens/agents.py), so the table also reports candidate
squares scored: the same unit of work for every agent.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .agents import Result, run
from .board import is_solution

SEED_BASE = 1000

# (agent, size, trials, per-run time cap in seconds, step cap or None). A None step cap means the
# agent's own default budget applies. Hill climbing on 10,000 queens is skipped: one step scores 10^8
# squares, so a single step alone would take about a minute in Python.
FULL_PLAN = [
    ("backtrack", 8, 1, 10.0, 2_000_000),
    ("backtrack", 16, 1, 10.0, 2_000_000),
    ("backtrack", 32, 1, 10.0, 2_000_000),
    ("backtrack", 128, 1, 10.0, 2_000_000),
    ("backtrack", 1000, 1, 10.0, 2_000_000),
    ("backtrack", 10000, 1, 10.0, 2_000_000),
    ("hill", 8, 200, 5.0, None),
    ("hill", 16, 100, 5.0, None),
    ("hill", 32, 30, 10.0, None),
    ("hill", 128, 3, 15.0, None),
    ("hill", 1000, 2, 20.0, None),
    ("hill", 10000, 0, 0.0, None),  # listed so the skip below is recorded in the results
    ("anneal", 8, 200, 5.0, None),
    ("anneal", 16, 100, 5.0, None),
    ("anneal", 32, 30, 10.0, None),
    ("anneal", 128, 5, 20.0, None),
    ("anneal", 1000, 3, 30.0, None),
    ("anneal", 10000, 2, 60.0, None),
    ("minconf", 8, 200, 5.0, None),
    ("minconf", 16, 200, 5.0, None),
    ("minconf", 32, 100, 5.0, None),
    ("minconf", 128, 100, 5.0, None),
    ("minconf", 1000, 50, 10.0, None),
    ("minconf", 10000, 20, 20.0, None),
    ("minconf", 100000, 3, 60.0, None),
    ("minconf", 1000000, 1, 120.0, None),
]

SKIPPED = {
    ("hill", 10000): "not run: one step scores 10^8 squares (N^2), too slow to finish a single step in Python",
}

# Quick mode: a handful of trials, short caps, and the sizes that finish in seconds.
QUICK_PLAN = [
    (agent, n, min(trials, 5), min(cap, 3.0), steps)
    for agent, n, trials, cap, steps in FULL_PLAN
    if n <= 1000 or (agent == "minconf" and n <= 10000)
]


def _one(agent: str, n: int, seed: int, cap: float, steps: int | None) -> Result:
    kwargs: dict = {"time_limit": cap}
    if steps is not None:
        kwargs["max_steps"] = steps
    return run(agent, n, seed, **kwargs)


def summarize(results: list[Result]) -> dict:
    """Success rate, the median steps / work of the solved runs, and median time over all runs."""
    solved = [r for r in results if r.solved]
    reasons: dict[str, int] = {}
    for r in results:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    out = {
        "runs": len(results),
        "solved": len(solved),
        "success_rate": round(len(solved) / len(results), 4),
        "seconds_median": round(statistics.median(r.seconds for r in results), 4),
        "seconds_max": round(max(r.seconds for r in results), 3),
        "reasons": reasons,
    }
    if solved:
        out["steps_median_solved"] = statistics.median(r.steps for r in solved)
        out["evaluations_median_solved"] = statistics.median(r.evaluations for r in solved)
        out["restarts_median_solved"] = statistics.median(r.restarts for r in solved)
        out["seconds_median_solved"] = round(statistics.median(r.seconds for r in solved), 4)
    return out


def run_benchmark(plan=FULL_PLAN, progress=None) -> dict:
    """Run every case in the plan and return nested results: agents -> size -> summary."""
    agents: dict[str, dict[str, dict]] = {}
    skipped: dict[str, str] = {}
    for agent, n, trials, cap, steps in plan:
        if (agent, n) in SKIPPED:
            skipped[f"{agent} N={n}"] = SKIPPED[(agent, n)]
            continue
        results = []
        for i in range(trials):
            r = _one(agent, n, SEED_BASE + i, cap, steps)
            # A checked solution: every reported solve must be a valid placement, not just a count of 0.
            if r.solved and not is_solution(r.cols):
                raise AssertionError(f"{agent} reported a solution for n={n} that is not one")
            results.append(r)
            if agent == "backtrack":
                break  # deterministic: one run is the whole answer
        summary = summarize(results)
        agents.setdefault(agent, {})[str(n)] = summary
        if progress:
            progress(agent, n, summary)
    return {"seed_base": SEED_BASE, "agents": agents, "skipped": skipped}


def to_markdown(data: dict) -> str:
    """One row per (agent, size). Time is the median over all runs, so capped runs count at their cap."""
    lines = [
        "Seeds 1000 onward, the same seeds for every agent. Steps are each agent's own unit (see queens/agents.py);",
        "candidate squares scored is the work measure shared by all four. Time is the median over all runs.",
        "",
        "| agent | N | runs | solved | steps (median, solved) | squares scored (median, solved) "
        "| time (median) | how the other runs ended |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    order = ["backtrack", "hill", "anneal", "minconf"]
    for agent in order:
        for n, s in sorted(data["agents"].get(agent, {}).items(), key=lambda kv: int(kv[0])):
            steps = f"{s['steps_median_solved']:,.0f}" if "steps_median_solved" in s else "—"
            work = f"{s['evaluations_median_solved']:,.0f}" if "evaluations_median_solved" in s else "—"
            ends = ", ".join(f"{k} {v}" for k, v in s["reasons"].items() if k != "solved") or "—"
            lines.append(f"| {agent} | {int(n):,} | {s['runs']} | {s['solved']} | {steps} | {work} | "
                         f"{s['seconds_median']:.3g} s | {ends} |")
    for key, why in data.get("skipped", {}).items():
        lines.append(f"| {key} | | | | | | | {why} |")
    return "\n".join(lines) + "\n"


def write_results(data: dict, out_dir: str | Path) -> tuple[Path, Path]:
    """Write queens_benchmark.json and queens_benchmark.md into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "queens_benchmark.json"
    md_path = out / "queens_benchmark.md"
    json_path.write_text(json.dumps(data, indent=2) + "\n")
    md_path.write_text(to_markdown(data))
    return json_path, md_path
