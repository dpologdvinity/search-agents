"""Benchmark: RAVE-MCTS against plain UCT, random, and the shortest-path heuristic, plus search speed.

    python -m hexgame benchmark                   # the full run committed under results/
    python -m hexgame benchmark --games 10 --size 5   # a quick check

Each matchup plays `games` seeded games. The first player alternates colour from game to game, so
each agent plays both DOWN and ACROSS equally often. The seed of game g is the same for every matchup,
and the seed of each search is drawn from the game's RNG, so a run repeats exactly. Win rates carry a
95% Wilson interval. Speed is measured on an empty board: search simulations per second and the raw
rate of random fill playouts.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from .agents import player
from .board import ACROSS, DOWN, EMPTY, fill_playout, other, winner_of
from .mcts import search

MATCHUPS = [("rave", "uct"), ("rave", "shortest"), ("uct", "shortest"), ("rave", "random"), ("uct", "random")]


def play_game(n: int, first, second, seed: int) -> tuple[int, int]:
    """Play one game. `first` moves as DOWN, `second` as ACROSS. Returns (winner, moves played)."""
    rng = random.Random(seed)
    cells = [EMPTY] * (n * n)
    players = {DOWN: first, ACROSS: second}
    to_move, moves = DOWN, 0
    while True:
        w = winner_of(n, cells)
        if w is not None:
            return w, moves
        cell = players[to_move](n, cells, to_move, rng)
        cells[cell] = to_move
        moves += 1
        to_move = other(to_move)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def match(n: int, a: str, b: str, games: int, sims: int) -> dict:
    """Agent `a` against agent `b` over `games` games, alternating which colour `a` plays."""
    pa, pb = player(a, sims), player(b, sims)
    wins = {DOWN: [0, 0], ACROSS: [0, 0]}  # colour -> [games, wins for a]
    a_wins = total_moves = 0
    start = time.perf_counter()
    for g in range(games):
        a_colour = DOWN if g % 2 == 0 else ACROSS
        if a_colour == DOWN:
            winner, moves = play_game(n, pa, pb, seed=1000 + g)
        else:
            winner, moves = play_game(n, pb, pa, seed=1000 + g)
        won = winner == a_colour
        a_wins += won
        total_moves += moves
        wins[a_colour][0] += 1
        wins[a_colour][1] += won
    lo, hi = wilson(a_wins, games)
    return {
        "a": a, "b": b, "games": games,
        "a_wins": a_wins, "a_rate": round(a_wins / games, 4), "ci95": [lo, hi],
        "a_as_down": f"{wins[DOWN][1]}/{wins[DOWN][0]}", "a_as_across": f"{wins[ACROSS][1]}/{wins[ACROSS][0]}",
        "mean_moves": round(total_moves / games, 1),
        "seconds": round(time.perf_counter() - start, 1),
    }


def speed(n: int, sims: int, reps: int = 2000) -> dict:
    """Simulations per second of each MCTS variant on an empty board, plus raw random-fill playouts per second."""
    out = {}
    for name, rave in (("rave", True), ("uct", False)):
        res = search(n, [EMPTY] * (n * n), DOWN, sims=sims, rave=rave, seed=7)
        out[name] = {"sims": res["sims"], "seconds": round(res["seconds"], 3),
                     "sims_per_second": round(res["sims"] / res["seconds"]),
                     "nodes": res["nodes"]}
    rng = random.Random(7)
    start = time.perf_counter()
    for _ in range(reps):
        fill_playout(n, [EMPTY] * (n * n), DOWN, rng)
    dt = time.perf_counter() - start
    out["playouts_per_second"] = round(reps / dt)
    return out


def run(size: int = 7, sims: int = 400, games: int = 40, matchups=MATCHUPS) -> dict:
    """The whole benchmark as one JSON-friendly dict."""
    start = time.perf_counter()
    return {
        "size": size, "sims_per_move": sims, "games_per_matchup": games,
        "seed_base": 1000,
        "speed": speed(size, sims=max(sims, 2000)),
        "matchups": [match(size, a, b, games, sims) for a, b in matchups],
        "seconds": round(time.perf_counter() - start, 1),
    }


def to_markdown(result: dict) -> str:
    """Tables for the README and results/."""
    n, sims, g = result["size"], result["sims_per_move"], result["games_per_matchup"]
    sp = result["speed"]
    lines = [
        f"Hex {n}x{n}, {sims} simulations per move, {g} seeded games per matchup (colours alternate). "
        "Win rate is for the first agent, with a 95% Wilson interval.",
        "",
        "| matchup | first agent win rate (95% CI) | wins as DOWN | wins as ACROSS | mean moves |",
        "|---|---|---|---|---|",
    ]
    for m in result["matchups"]:
        lo, hi = m["ci95"]
        lines.append(f"| {m['a']} vs {m['b']} | {100 * m['a_rate']:.1f}% ({100 * lo:.1f}%-{100 * hi:.1f}%) "
                     f"| {m['a_as_down']} | {m['a_as_across']} | {m['mean_moves']} |")
    lines += [
        "",
        f"Speed on an empty {n}x{n} board: RAVE {sp['rave']['sims_per_second']} simulations/s, "
        f"UCT {sp['uct']['sims_per_second']} simulations/s, random fill playouts "
        f"{sp['playouts_per_second']}/s (single core, pure Python).",
    ]
    return "\n".join(lines) + "\n"


def write_results(result: dict, out_dir: str | Path) -> tuple[Path, Path]:
    """Write hexgame_benchmark.json and .md into `out_dir`. Returns the two paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "hexgame_benchmark.json"
    md_path = out / "hexgame_benchmark.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    md_path.write_text(to_markdown(result))
    return json_path, md_path
