"""Compare placement strategies on the same seeded games and write the results to results/.

Strategies:
  random       a uniformly random legal placement (the floor)
  hand         one-piece greedy with the hand-picked weights (HAND_WEIGHTS)
  ga           one-piece greedy with the GA-tuned weights from tetris/tuned.json
  ga_lookahead the GA weights plus one-piece lookahead with the preview piece (top_k candidates)

Every strategy plays the same seeds, drawn from a range the GA never trains on. Each game is played until
the stack tops out or reaches a piece cap. The benchmark reports the mean lines cleared with a 95%
confidence interval (Student's t over the games), and how many games reached the cap while still alive.
A game that hits the cap is a lower bound on its lines, so a high cap-hit count means the cap is too low to
rank the strategies.

Two settings are committed: the 10x20 game board with a 2,000-piece cap (results/tetris_benchmark), and a
harder setting with a narrow board, no lookahead and a high cap, where games end on their own
(results/tetris_benchmark_hard).
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from .board import H
from .features import HAND_WEIGHTS
from .game import greedy_policy, lookahead_policy, play, random_policy
from .tuned import load_tuned

BENCH_SEED_BASE = 50_000
STRATEGIES = ("random", "hand", "ga", "ga_lookahead")
LOOKAHEAD_TOP_K = 6

# Two-sided 95% Student's t critical values for 1..30 degrees of freedom; beyond 30 the normal value is used.
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
        2.045, 2.042]


def t95(df: int) -> float:
    return _T95[df - 1] if 1 <= df <= len(_T95) else 1.96


def _policy_for(name: str, seed: int, ga_weights):
    if name == "random":
        return random_policy(seed)
    if name == "hand":
        return greedy_policy(HAND_WEIGHTS)
    if name == "ga":
        return greedy_policy(ga_weights)
    if name == "ga_lookahead":
        return lookahead_policy(ga_weights, top_k=LOOKAHEAD_TOP_K)
    raise ValueError(name)


def _run(task) -> dict:
    name, seed, cap, height, ga_weights = task
    t0 = time.time()
    r = play(_policy_for(name, seed, ga_weights), seed, cap, height=height)
    return {"strategy": name, "seed": seed, "lines": r.lines, "pieces": r.pieces,
            "topped_out": r.topped_out, "capped": r.capped, "seconds": time.time() - t0}


@dataclass(frozen=True)
class Summary:
    strategy: str
    games: int
    mean_lines: float
    stderr: float
    ci95: float  # half-width of the 95% confidence interval for the mean
    median_lines: float
    min_lines: int
    max_lines: int
    capped: int
    topped_out: int
    mean_pieces: float
    seconds_per_game: float


def summarise(rows: list[dict]) -> Summary:
    lines = [r["lines"] for r in rows]
    n = len(rows)
    sd = statistics.stdev(lines) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    return Summary(
        strategy=rows[0]["strategy"],
        games=n,
        mean_lines=statistics.fmean(lines),
        stderr=se,
        ci95=t95(n - 1) * se if n > 1 else 0.0,
        median_lines=statistics.median(lines),
        min_lines=min(lines),
        max_lines=max(lines),
        capped=sum(r["capped"] for r in rows),
        topped_out=sum(r["topped_out"] for r in rows),
        mean_pieces=statistics.fmean(r["pieces"] for r in rows),
        seconds_per_game=statistics.fmean(r["seconds"] for r in rows),
    )


def run_benchmark(games: int = 30, cap: int = 2000, workers: int = 1, strategies=STRATEGIES,
                  ga_weights=None, height: int = H) -> dict:
    """Play every strategy on the same `games` seeds and return the per-game rows and the summaries."""
    if ga_weights is None:
        ga_weights = load_tuned()["weights"]
    seeds = [BENCH_SEED_BASE + i for i in range(games)]
    tasks = [(name, s, cap, height, ga_weights) for name in strategies for s in seeds]
    with Pool(workers) as pool:
        rows = pool.map(_run, tasks, chunksize=1)
    by = {name: [r for r in rows if r["strategy"] == name] for name in strategies}
    summaries = [summarise(by[name]) for name in strategies]
    return {"games": games, "piece_cap": cap, "board_height": height, "seeds": seeds,
            "ga_weights": list(ga_weights), "lookahead_top_k": LOOKAHEAD_TOP_K,
            "summaries": [s.__dict__ for s in summaries], "rows": rows}


LABELS = {"random": "random placement", "hand": "hand-picked weights", "ga": "GA-tuned weights",
          "ga_lookahead": "GA-tuned + 1-piece lookahead"}


def to_markdown(result: dict) -> str:
    """The summary table for results/*.md and the README."""
    head = ("| Strategy | Games | Mean lines | 95% CI | Median | Min-max | Hit cap | Topped out | "
            "Mean pieces | s per game |")
    source = f" The 'ga' row uses the weights in {result['ga_source']}." if result.get("ga_source") else ""
    lines = [f"Board {result.get('board_height', H)} rows, piece cap {result['piece_cap']}.{source}", "", head,
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in result["summaries"]:
        lines.append(
            f"| {LABELS[s['strategy']]} | {s['games']} | {s['mean_lines']:.1f} | ± {s['ci95']:.1f} | "
            f"{s['median_lines']:.0f} | {s['min_lines']}-{s['max_lines']} | {s['capped']} / {s['games']} | "
            f"{s['topped_out']} | {s['mean_pieces']:.0f} | {s['seconds_per_game']:.2f} |")
    return "\n".join(lines) + "\n"


def write_results(result: dict, out_base: Path) -> None:
    """Write <out_base>.json (everything, per game) and <out_base>.md (the table)."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_base.with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n")
    out_base.with_suffix(".md").write_text(to_markdown(result))
