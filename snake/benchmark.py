"""Benchmark: apples, deaths, and survival for each agent on the same seeded boards.

    python -m snake benchmark                 # 200 games per agent, printed as a table
    python -m snake benchmark --games 50      # a quick check
    python -m snake benchmark --write         # also writes results/snake_benchmark.{json,md}
    python -m snake actions --write           # left/straight/right shares of the net; results/snake_net_actions.json

Every agent plays the same boards: game i uses seed BENCH_SEED_BASE + i. The boards are different
from the training seeds, so the evolved agent is scored on positions it was never trained on. Each
game ends in one of four ways: a wall or body collision (a death), starvation (the step cap with no
apple), a full board (a win), or the step cap. The table reports them separately because a looping
snake and a dead one are different failures.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .agents import AGENT_NAMES, DESCRIPTIONS, make_policy
from .board import ACTION_NAMES, BOARD, Game
from .net import Net, load_champion

BENCH_SEED_BASE = 50_000
DEFAULT_GAMES = 200
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_agent(name: str, games: int, size: int = BOARD, net: Net | None = None) -> list[dict]:
    """Play `games` seeded games with one agent. Returns one record per game."""
    records = []
    for i in range(games):
        seed = BENCH_SEED_BASE + i
        game = Game(seed, size)
        policy = make_policy(name, seed=seed, net=net)
        game.play(policy)
        records.append({"seed": seed, "apples": game.apples, "steps": game.steps, "cause": game.cause})
    return records


def summarize(records: list[dict]) -> dict:
    """Mean, median and max apples, the death rate, and the other endings, as plain numbers."""
    apples = [r["apples"] for r in records]
    n = len(records)
    causes = [r["cause"] for r in records]
    return {
        "games": n,
        "mean_apples": round(statistics.fmean(apples), 2),
        "median_apples": statistics.median(apples),
        "max_apples": max(apples),
        "death_rate": round(sum(c in ("wall", "body") for c in causes) / n, 3),
        "starved": causes.count("starved"),
        "full": causes.count("full"),
        "mean_steps": round(statistics.fmean(r["steps"] for r in records), 1),
    }


def run_benchmark(games: int = DEFAULT_GAMES, agents=AGENT_NAMES, size: int = BOARD,
                  net: Net | None = None, log=None) -> dict:
    """Run every agent and return {"games", "board", "seed_base", "agents": {name: summary}}.

    The evolved agent uses the committed champion unless `net` is given.
    """
    if net is None and "evolved" in agents:
        net, _ = load_champion()
    out = {"games": games, "board": size, "seed_base": BENCH_SEED_BASE, "agents": {}}
    for name in agents:
        t0 = time.perf_counter()
        records = run_agent(name, games, size, net)
        summary = summarize(records)
        summary["seconds"] = round(time.perf_counter() - t0, 1)
        out["agents"][name] = summary
        if log is not None:
            log(name, summary)
    return out


def to_markdown(result: dict) -> str:
    """The results as a Markdown table, for the README and results/snake_benchmark.md."""
    size = result["board"]
    lines = [
        f"{size}x{size} board, {result['games']} games per agent, seeds {result['seed_base']} onward "
        "(the same boards for every agent).",
        "",
        "| agent | mean apples | median | max | death rate | starved | full | mean steps |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in result["agents"].items():
        lines.append(
            f"| {name} | {s['mean_apples']:.2f} | {s['median_apples']:g} | {s['max_apples']} | "
            f"{100 * s['death_rate']:.1f}% | {s['starved']} | {s['full']} | {s['mean_steps']:.1f} |"
        )
    lines += ["", "Agents:"] + [f"- {name}: {DESCRIPTIONS[name]}" for name in result["agents"]]
    return "\n".join(lines) + "\n"


def write_results(result: dict, out_dir: Path = RESULTS_DIR) -> None:
    """Write the JSON and Markdown copies under results/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snake_benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "snake_benchmark.md").write_text(to_markdown(result))


def action_counts(games: int = 40, size: int = BOARD, net: Net | None = None) -> dict:
    """How often the evolved net picks each turn, over the first `games` benchmark boards.

    Same seeds as the benchmark (BENCH_SEED_BASE onward), so the run is fixed and repeatable. The
    benchmark table cannot show this: a net that almost never turns right can still score well on
    average, so the per-turn shares are reported on their own.
    """
    if net is None:
        net, _ = load_champion()
    counts = dict.fromkeys(ACTION_NAMES, 0)
    for i in range(games):
        seed = BENCH_SEED_BASE + i
        game = Game(seed, size)
        policy = make_policy("evolved", seed=seed, net=net)
        # Step by hand instead of Game.play so each chosen action can be counted.
        while game.alive:
            action = policy(game)
            counts[ACTION_NAMES[action]] += 1
            game.step(action)
    moves = sum(counts.values())
    return {
        "games": games,
        "board": size,
        "seed_base": BENCH_SEED_BASE,
        "moves": moves,
        "actions": counts,
        "shares": {name: round(n / moves, 4) for name, n in counts.items()},
    }


def write_action_counts(result: dict, out_dir: Path = RESULTS_DIR) -> None:
    """Write the per-turn counts to results/snake_net_actions.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snake_net_actions.json").write_text(json.dumps(result, indent=2) + "\n")
