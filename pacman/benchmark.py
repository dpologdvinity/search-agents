"""Benchmark: win rate and average score for each agent, on fixed seeds.

    python -m pacman benchmark --games 200

Every agent plays `games` games on each maze, using seeds BENCH_SEED_BASE + i. The
trainer draws its own seeds and never uses these, so the numbers are not training
results. The same seeds play the same games for every agent, which makes the comparison
fair: an agent that avoids a ghost in one game faces the same ghost in the others.
"""

from __future__ import annotations

from collections import Counter

from .agents import AGENTS, DESCRIPTIONS, make_agent
from .episode import Episode, run_episode
from .mazes import get, names

BENCH_SEED_BASE = 10_000


def summarize(episodes: list[Episode]) -> dict:
    """Win rate, mean score and turns, and how each game ended."""
    n = len(episodes)
    outcomes = Counter(ep.final.status for ep in episodes)
    return {
        "games": n,
        "win_rate": round(sum(ep.won for ep in episodes) / n, 4),
        "mean_score": round(sum(ep.score for ep in episodes) / n, 1),
        "mean_turns": round(sum(ep.final.turn for ep in episodes) / n, 1),
        "outcomes": {k: outcomes.get(k, 0) for k in ("won", "lost", "timeout")},
    }


def run_benchmark(games: int, mazes: tuple[str, ...] | None = None, agents: tuple[str, ...] = AGENTS) -> dict:
    """Play every agent on every maze and return the nested results."""
    mazes = mazes or names()
    seeds = [BENCH_SEED_BASE + i for i in range(games)]
    result = {"games_per_maze": games, "seeds": [seeds[0], seeds[-1]], "agents": {}}
    for name in agents:
        agent = make_agent(name)
        per_maze, everything = {}, []
        for m in mazes:
            episodes = [run_episode(agent, get(m), seed, name) for seed in seeds]
            per_maze[m] = summarize(episodes)
            everything += episodes
        result["agents"][name] = {
            "description": DESCRIPTIONS[name],
            "overall": summarize(everything),
            "mazes": per_maze,
        }
    return result


def to_markdown(result: dict) -> str:
    """A table of overall results per agent, plus one row per agent and maze."""
    lines = [
        f"Seeds {result['seeds'][0]}-{result['seeds'][1]}, {result['games_per_maze']} games per maze.",
        "",
        "| agent | win rate | mean score | mean turns | won / lost / timeout |",
        "|---|---|---|---|---|",
    ]
    for name, data in result["agents"].items():
        o = data["overall"]
        lines.append(
            f"| {name} | {100 * o['win_rate']:.1f}% | {o['mean_score']:.1f} | {o['mean_turns']:.1f} | "
            f"{o['outcomes']['won']} / {o['outcomes']['lost']} / {o['outcomes']['timeout']} |"
        )
    lines += ["", "| agent | maze | win rate | mean score |", "|---|---|---|---|"]
    for name, data in result["agents"].items():
        for m, s in data["mazes"].items():
            lines.append(f"| {name} | {get(m).title} | {100 * s['win_rate']:.1f}% | {s['mean_score']:.1f} |")
    return "\n".join(lines) + "\n"
