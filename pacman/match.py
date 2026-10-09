"""Match: a Pac-Man agent against each ghost policy, on fixed seeds, with win rate and score.

    python -m pacman match --games 200

Every game uses the benchmark's seeds (BENCH_SEED_BASE + i), so the same seeds replay on each
maze and under each ghost policy. The agent plays the same games twice: once against the A*
ghosts ("ai") and once against the chance ghosts ("chance", fixed odds with no planning). The
difference between the two rows is what the A* route planning is worth to the ghosts.
"""

from __future__ import annotations

from .agents import AGENT_LABELS, DESCRIPTIONS, make_agent
from .benchmark import BENCH_SEED_BASE, summarize
from .episode import run_episode
from .ghosts import CHANCE_ODDS, GHOST_LABELS, GHOST_POLICIES
from .mazes import get, names


def run_match(games: int, agent_name: str = "q", mazes: tuple[str, ...] | None = None) -> dict:
    """Play the agent on every maze, `games` seeds each, against every ghost policy.

    Returns the nested results: one block per ghost policy, with an overall summary and one
    summary per maze. Nothing here is random beyond the seeds, so a rerun gives the same numbers.
    """
    mazes = mazes or names()
    seeds = [BENCH_SEED_BASE + i for i in range(games)]
    agent = make_agent(agent_name)
    result = {
        "agent": agent_name,
        "agent_label": AGENT_LABELS[agent_name],
        "agent_description": DESCRIPTIONS[agent_name],
        "games_per_maze": games,
        "seeds": [seeds[0], seeds[-1]],
        "chance_odds": CHANCE_ODDS,
        "ghosts": {},
    }
    for policy in GHOST_POLICIES:
        per_maze, everything = {}, []
        for m in mazes:
            episodes = [run_episode(agent, get(m), seed, agent_name, ghosts=policy) for seed in seeds]
            per_maze[m] = summarize(episodes)
            everything += episodes
        result["ghosts"][policy] = {
            "label": GHOST_LABELS[policy],
            "overall": summarize(everything),
            "mazes": per_maze,
        }
    return result


def match_markdown(result: dict) -> str:
    """The match as Markdown: which algorithm is on each side, then one row per ghost policy."""
    lines = [
        f"Pac-Man agent: {result['agent_label']} ({result['agent']}). {result['agent_description']}",
        "Ghosts: " + " vs ".join(GHOST_LABELS[p] for p in result["ghosts"]) + ".",
        f"Seeds {result['seeds'][0]}-{result['seeds'][1]}, {result['games_per_maze']} games per maze, "
        f"{len(result['ghosts']['ai']['mazes'])} mazes.",
        "",
        "| ghosts | win rate | mean score | mean turns | won / lost / timeout |",
        "|---|---|---|---|---|",
    ]
    for _policy, data in result["ghosts"].items():
        o = data["overall"]
        lines.append(
            f"| {data['label']} | {100 * o['win_rate']:.1f}% | {o['mean_score']:.1f} | {o['mean_turns']:.1f} | "
            f"{o['outcomes']['won']} / {o['outcomes']['lost']} / {o['outcomes']['timeout']} |"
        )
    odds = ", ".join(f"{k} {round(100 * v)}%" for k, v in result["chance_odds"].items())
    lines += ["", f"Chance odds, relative to the ghost's last move: {odds}."]
    return "\n".join(lines) + "\n"
