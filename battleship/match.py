"""Head-to-head race: the AI and chance each sink the other's fleet, taking turns, and the first to finish wins.

    python -m battleship match                                 # 100 races, probability AI vs chance, seed 1
    python -m battleship match --ai hunt --games 500 --seed 7   # the hunt baseline is much faster per race

Each race has two random legal fleets: the AI shoots at the chance player's fleet and chance shoots back at the
AI's. Shots alternate, so the side that sinks the last ship of the other fleet first wins and no race is drawn.
The first shooter alternates by game index so neither side keeps the first shot. Every race is seeded from the
game index, so a run repeats exactly.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from random import Random

from .agents import AGENTS, ChanceAgent, make_agent
from .board import Fleet, Knowledge, random_fleet


@dataclass(frozen=True)
class Race:
    """Outcome of one race. winner is "ai" or "chance"; the shot counts include the shot that finished the race."""

    winner: str
    ai_shots: int
    chance_shots: int


def race(ai_name: str, rng: Random, ai_first: bool = True) -> Race:
    """Play one race to the end. The AI's fleet is the one chance shoots at, and the reverse.

    Each shooter keeps its own Knowledge of the enemy fleet and sees only the answers to its own shots, the same
    way the single-player games work.
    """
    ai_fleet = Fleet(random_fleet(rng))  # chance fires at this
    chance_fleet = Fleet(random_fleet(rng))  # the AI fires at this
    ai = make_agent(ai_name, rng)
    chance = ChanceAgent(rng)
    ai_k = Knowledge()
    chance_k = Knowledge()
    turns = [("ai", ai, ai_k, chance_fleet), ("chance", chance, chance_k, ai_fleet)]
    if not ai_first:
        turns.reverse()
    shots = {"ai": 0, "chance": 0}
    while True:
        for side, agent, know, enemy in turns:
            cell = agent.choose(know)
            shot = enemy.fire(cell)
            know.observe(cell, shot.result, shot.ship)
            shots[side] += 1
            if enemy.all_sunk:
                return Race(side, shots["ai"], shots["chance"])


def match(ai_name: str, games: int, seed: int = 1) -> dict:
    """Play `games` races. Returns wins, the shot counts of each side, and the wall-clock seconds."""
    wins = {"ai": 0, "chance": 0}
    ai_shots: list[int] = []
    chance_shots: list[int] = []
    start = time.perf_counter()
    for g in range(games):
        result = race(ai_name, Random(seed * 1_000_003 + g), ai_first=g % 2 == 0)
        wins[result.winner] += 1
        ai_shots.append(result.ai_shots)
        chance_shots.append(result.chance_shots)
    return {"ai": ai_name, "games": games, "seed": seed, "wins": wins, "ai_shots": ai_shots,
            "chance_shots": chance_shots, "seconds": time.perf_counter() - start}


def report(result: dict) -> str:
    """Plain-text summary: the algorithms, the wins, and the mean shots to finish."""
    games = result["games"]
    ai, chance = result["wins"]["ai"], result["wins"]["chance"]
    ai_label = AGENTS[result["ai"]].label
    lines = [
        f"AI: {ai_label} ({result['ai']})  vs  chance: {ChanceAgent.label} ({ChanceAgent.name})",
        f"{games} races, seed {result['seed']}: AI won {ai} ({100 * ai / games:.1f}%), chance won {chance} "
        f"({100 * chance / games:.1f}%). Races cannot be drawn: the first fleet sunk decides.",
        f"mean shots fired when the race ended: AI {statistics.fmean(result['ai_shots']):.1f}, "
        f"chance {statistics.fmean(result['chance_shots']):.1f}",
        f"({result['seconds']:.2f} s, {1000 * result['seconds'] / games:.2f} ms per race)",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m battleship match", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ai", choices=sorted(set(AGENTS) - {"chance"}), default="probability",
                        help="the AI side (default probability)")
    parser.add_argument("--games", type=int, default=100, help="races to play (default 100)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    args = parser.parse_args(argv)
    print(report(match(args.ai, args.games, args.seed)))
    return 0
