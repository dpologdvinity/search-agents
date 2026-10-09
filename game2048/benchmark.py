"""Benchmark 2048 agents over full games.

    python -m game2048.benchmark greedy --games 2000 --seed 500000 --agents ntuple@results/game2048_ntuple_v1.npz
    python -m game2048.benchmark search --games 40 --workers 5 --budget 0.05 --seed 500000 \
        --agents expectimax expectimax@results/game2048_heuristic_weights_v1.json

An agent spec is a name, optionally followed by @PATH: a tuner weights file
for the expectimax agents, an n-tuple .npz for the n-tuple agents. Without
@PATH the shipped files are used. `greedy` plays the n-tuple policy without
search, vectorized over all games. `search` plays each expectimax agent game
by game in parallel processes; the per-move budget is wall-clock time, so
results depend on the machine.

Games use seeds `--seed + i`, so pick a seed range the tuner and the
trainer did not use to get held-out numbers. Each row reports the mean score
with a 95% normal interval and each reach rate with a 95% Wilson interval.
Results are written to results/game2048_<tag>.md and .json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .batch import max_exponent_batch, move_batch, new_games, spawn_batch
from .board import max_exponent, play_game
from .expectimax import Expectimax, evaluate, load_weights
from .ntuple import NTupleNetwork

RESULTS = Path("results")
Z = 1.96  # 95% two-sided

NAMES = {
    "expectimax_full": "Expectimax, tuned eval, full-width (no cache or cutoff)",
    "expectimax": "Expectimax, tuned eval + cache + probability cutoff",
    "ntuple_search": "Expectimax, learned n-tuple eval",
    "ntuple": "N-tuple network, greedy (no search)",
}


def split_spec(spec: str) -> tuple[str, str | None]:
    """'expectimax@path.json' -> ('expectimax', 'path.json'); a bare name has no path."""
    name, _, path = spec.partition("@")
    if name not in NAMES:
        raise SystemExit(f"unknown agent {name!r}; choose from {', '.join(NAMES)}")
    return name, path or None


def greedy_games(n: int, seed: int, net: NTupleNetwork):
    """Play n games with the greedy n-tuple policy; returns [(score, max exponent, moves)]."""
    rng = np.random.default_rng(seed)
    boards = new_games(n, rng)
    scores = np.zeros(n, dtype=np.int64)
    moves = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    while not done.all():
        q = np.full((4, n), -np.inf)
        afters, rewards = [], []
        for d in range(4):
            after, reward = move_batch(boards, d)
            legal = (after != boards) & ~done
            afters.append(after)
            rewards.append(reward)
            if legal.any():
                q[d, legal] = reward[legal] + net.value_batch(after[legal])
        done |= np.isneginf(q.max(axis=0))
        live = ~done
        best = q.argmax(axis=0)
        rows = np.arange(n)
        chosen = np.stack(afters)[best, rows]
        scores[live] += np.stack(rewards)[best, rows][live]
        moves[live] += 1
        boards[live] = spawn_batch(chosen[live], rng)
    return list(zip(scores.tolist(), max_exponent_batch(boards).tolist(), moves.tolist()))


def _search_game(spec):
    agent_spec, budget, seed = spec
    name, path = split_spec(agent_spec)
    if name == "ntuple_search":
        net = NTupleNetwork.load(path) if path else NTupleNetwork.load()
        searcher = Expectimax(budget=budget, evaluator=net.value, rewards=True)
    else:
        fast = name == "expectimax"
        weights = load_weights(path) if path else None
        evaluator = partial(evaluate, weights=weights) if weights else None
        searcher = Expectimax(budget=budget, cache=fast, prob_cutoff=1e-4 if fast else 0.0, evaluator=evaluator)
    depths = []

    def choose(board):
        info = searcher.search(board)
        depths.append(info.depth)
        return info.move

    board, score, moves = play_game(choose, random.Random(seed))
    return agent_spec, score, max_exponent(board), moves, statistics.mean(depths) if depths else 0


def mean_interval(values) -> tuple[float, float]:
    """(mean, half-width of the 95% normal interval)."""
    n = len(values)
    if n < 2:
        return float(values[0]) if n else 0.0, float("nan")
    return statistics.mean(values), Z * statistics.stdev(values) / math.sqrt(n)


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n, as fractions."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(label: str, games) -> str:
    """games: [(score, max exponent, moves, ...)] -> one Markdown row with 95% intervals."""
    scores = [g[0] for g in games]
    tiles = [g[1] for g in games]
    n = len(games)
    mean, hw = mean_interval(scores)

    def reach(k):
        hits = sum(t >= k for t in tiles)
        lo, hi = wilson(hits, n)
        return f"{100 * hits / n:.1f}% ({100 * lo:.1f}-{100 * hi:.1f})"

    return (f"| {label} | {n} | {mean:,.0f} ({mean - hw:,.0f} to {mean + hw:,.0f}) "
            f"| {statistics.median(scores):,.0f} | {reach(11)} | {reach(12)} | {reach(13)} | {1 << max(tiles)} |")


HEADER = ("| Agent | Games | Mean score (95% CI) | Median score | Reached 2048 (95% CI) "
          "| Reached 4096 (95% CI) | Reached 8192 (95% CI) | Best tile |\n"
          "|---|---|---|---|---|---|---|---|")


def label_for(spec: str) -> str:
    name, path = split_spec(spec)
    return NAMES[name] + (f" [{Path(path).name}]" if path else "")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["greedy", "search"])
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--budget", type=float, default=0.1)
    p.add_argument("--agents", nargs="*", help="agent specs; default: ntuple, or the three search agents")
    p.add_argument("--seed", type=int, default=0, help="game i is played from seed + i")
    p.add_argument("--tag", help="output name: results/game2048_<tag>.md (default: the mode)")
    args = p.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    tag = args.tag or args.mode
    default = ["ntuple"] if args.mode == "greedy" else ["expectimax_full", "expectimax", "ntuple_search"]
    args.agents = args.agents or default

    if args.mode == "greedy":
        start = time.time()
        rows, data = [], {}
        for spec in args.agents:
            name, path = split_spec(spec)
            if name != "ntuple":
                raise SystemExit("greedy mode plays the n-tuple policy only")
            net = NTupleNetwork.load(path) if path else NTupleNetwork.load()
            games = greedy_games(args.games, args.seed, net)
            rows.append(summarize(label_for(spec), games))
            data[spec] = games
        table = HEADER + "\n" + "\n".join(rows)
        print(table, f"\n({time.time() - start:.0f} s)")
        (RESULTS / f"game2048_{tag}.md").write_text(table + "\n")
        (RESULTS / f"game2048_{tag}.json").write_text(json.dumps(data))
        return

    specs = [(a, args.budget, args.seed + i) for a in args.agents for i in range(args.games)]
    by_agent = {a: [] for a in args.agents}
    with Pool(args.workers) as pool:
        for agent, score, tile, moves, depth in pool.imap_unordered(_search_game, specs):
            by_agent[agent].append((score, tile, moves, depth))
            print(f"{agent}: score {score} tile {1 << tile} moves {moves} mean depth {depth:.2f}", flush=True)
    rows = [summarize(f"{label_for(a)} ({args.budget * 1000:.0f} ms/move)", g) for a, g in by_agent.items()]
    table = HEADER + "\n" + "\n".join(rows)
    print(table)
    (RESULTS / f"game2048_{tag}.md").write_text(table + "\n")
    (RESULTS / f"game2048_{tag}.json").write_text(json.dumps(by_agent))


if __name__ == "__main__":
    main()
