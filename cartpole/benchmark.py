"""Benchmark every agent on the same seeded episodes, and write the committed results.

Every agent plays episodes whose start states come from seeds BENCH_SEED_BASE + i. Training used
seeds 0-4 and PD tuning used 9000-9099, so these episodes are unseen by every method. Learned
policies act greedily (argmax of the policy), as they do on the web page. Random agents draw
their actions from a seeded generator, one generator per episode.

The reported numbers per agent: mean steps balanced (a full episode is 500), standard deviation,
a 95% confidence interval for the mean, and the share of episodes that reached the 500-step cap.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .agents import AGENT_NAMES, DATA_DIR, LABELS, RandomAgent, load_agent
from .env import MAX_STEPS, run_episode

BENCH_SEED_BASE = 10_000
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def evaluate(name: str, episodes: int, seed_base: int = BENCH_SEED_BASE, data_dir: Path = DATA_DIR) -> list[int]:
    """Step counts for `episodes` seeded runs of one agent."""
    if name == "random":
        return [run_episode(RandomAgent(seed_base + i), seed_base + i) for i in range(episodes)]
    agent = load_agent(name, data_dir=data_dir)  # loaded once: the weights are read from disk a single time
    return [run_episode(agent, seed_base + i) for i in range(episodes)]


def summarise(name: str, steps: list[int]) -> dict:
    a = np.asarray(steps, dtype=float)
    n = len(a)
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    return {
        "label": LABELS[name],
        "episodes": n,
        "mean": round(float(a.mean()), 2),
        "std": round(std, 2),
        "ci95": round(1.96 * std / math.sqrt(n), 2) if n > 1 else 0.0,
        "min": int(a.min()),
        "max": int(a.max()),
        "reached_max": int((a >= MAX_STEPS).sum()),
        "pct_reached_max": round(100.0 * float((a >= MAX_STEPS).mean()), 1),
        "steps": [int(s) for s in steps],
    }


def run_benchmark(episodes: int, names=AGENT_NAMES, data_dir: Path = DATA_DIR, log=print) -> dict:
    rows = {}
    for name in names:
        log(f"benchmarking {name} on {episodes} episodes")
        rows[name] = summarise(name, evaluate(name, episodes, data_dir=data_dir))
    return {"episodes": episodes, "seed_base": BENCH_SEED_BASE, "max_steps": MAX_STEPS, "agents": rows}


def table(doc: dict) -> str:
    """The benchmark as a Markdown table, the same numbers the JSON holds."""
    lines = [
        "| Agent | Mean steps | 95% CI | Std | Min | Reached 500 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in doc["agents"].values():
        lines.append(f"| {r['label']} | {r['mean']:.1f} | ±{r['ci95']:.1f} | {r['std']:.1f} | "
                     f"{r['min']} | {r['pct_reached_max']:.1f}% |")
    return "\n".join(lines)


def write_results(doc: dict, data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR) -> None:
    """Save the full JSON next to the weights (the page reads it), and a Markdown summary under results/."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "benchmark.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    results_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "# CartPole benchmark\n\n"
        f"{doc['episodes']} seeded episodes per agent (seeds {doc['seed_base']}+i), 500-step cap. "
        "Learned policies act greedily.\n\n"
        f"{table(doc)}\n"
    )
    (results_dir / "cartpole_benchmark.md").write_text(body)
