"""Benchmark: win rate and guesses per game for each agent on each classic size.

    python -m minesweeper benchmark                    # the full run committed under results/
    python -m minesweeper benchmark --games 50         # a quick check

Every agent plays the same seeded boards (BENCH_SEED_BASE + i for game i), so the comparison is
paired: an agent that loses a game lost it to the same mine layout the others faced. The first
click is free and is not counted as a guess. Win rates come with a 95% Wilson interval, because
with a few hundred games the difference between two agents can be within noise.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .agents import AGENT_NAMES, GameResult, play
from .board import PRESETS

BENCH_SEED_BASE = 10_000
# Games per size. Expert is the slow one: a probability move there costs a few milliseconds of counting.
GAMES = {"beginner": 1000, "intermediate": 500, "expert": 200}


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. It behaves better than the normal interval near 0 and 1."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(results: list[GameResult]) -> dict:
    """Win rate with its interval, guesses and reveals per game, and how many wins needed no guess."""
    n = len(results)
    wins = sum(r.won for r in results)
    lo, hi = wilson(wins, n)
    return {
        "games": n,
        "wins": wins,
        "win_rate": round(wins / n, 4),
        "win_ci95": [round(lo, 4), round(hi, 4)],
        "guesses_per_game": round(sum(r.guesses for r in results) / n, 2),
        "guesses_per_win": round(sum(r.guesses for r in results if r.won) / max(1, wins), 2),
        "wins_without_guessing": sum(1 for r in results if r.won and r.guesses == 0),
        "reveals_per_game": round(sum(r.reveals for r in results) / n, 1),
        "seconds_per_game": round(sum(r.seconds for r in results) / n, 4),
    }


def run_benchmark(games: dict[str, int] | None = None, agents: tuple[str, ...] = AGENT_NAMES,
                  presets: tuple[str, ...] | None = None) -> dict:
    """Play every agent on every size, and return the nested results."""
    games = games or GAMES
    presets = presets or tuple(PRESETS)
    result = {
        "seed_base": BENCH_SEED_BASE,
        "games": {p: games[p] for p in presets},
        "boards": {p: dict(zip(("rows", "cols", "mines"), PRESETS[p])) for p in presets},
        "agents": defaultdict(dict),
    }
    for preset in presets:
        rows, cols, mines = PRESETS[preset]
        seeds = [BENCH_SEED_BASE + i for i in range(games[preset])]
        for agent in agents:
            runs = [play(rows, cols, mines, seed, agent) for seed in seeds]
            result["agents"][agent][preset] = summarize(runs)
    result["agents"] = dict(result["agents"])
    return result


def to_markdown(result: dict) -> str:
    """Tables for the README and results/: one table per size, one row per agent."""
    lines = [
        f"Seeds {result['seed_base']} onward, the same boards for every agent. "
        "Win rate has a 95% Wilson interval.",
        "",
    ]
    for preset in result["games"]:
        b = result["boards"][preset]
        lines += [
            f"### {preset.capitalize()}: {b['cols']}x{b['rows']} with {b['mines']} mines "
            f"({result['games'][preset]} games)",
            "",
            "| agent | win rate (95% CI) | guesses per game | wins with no guess | reveals per game |",
            "|---|---|---|---|---|",
        ]
        for agent, by_preset in result["agents"].items():
            s = by_preset[preset]
            lo, hi = s["win_ci95"]
            lines.append(
                f"| {agent} | {s['win_rate']:.1%} ({lo:.1%}-{hi:.1%}) | {s['guesses_per_game']:.2f} | "
                f"{s['wins_without_guessing']} of {s['wins']} | {s['reveals_per_game']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_results(result: dict, out_dir: str | Path) -> tuple[Path, Path]:
    """Write minesweeper_benchmark.json and .md into `out_dir`. Returns the two paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "minesweeper_benchmark.json"
    md_path = out / "minesweeper_benchmark.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    md_path.write_text(to_markdown(result))
    return json_path, md_path
