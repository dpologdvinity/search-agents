"""Benchmark 2048 agents over full games.

    python -m game2048.benchmark greedy --games 1000
    python -m game2048.benchmark search --games 20 --workers 8 --budget 0.05

`greedy` plays the n-tuple policy without search, vectorized over all games.
`search` plays each expectimax agent game by game in parallel processes;
the per-move budget is wall-clock time, so results depend on the machine.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .batch import max_exponent_batch, move_batch, new_games, spawn_batch
from .board import max_exponent, play_game
from .expectimax import Expectimax
from .ntuple import NTupleNetwork

RESULTS = Path("results")


def greedy_games(n: int, seed: int):
    """Play n games with the greedy n-tuple policy; returns [(score, max exponent, moves)]."""
    net = NTupleNetwork.load()
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
    agent, budget, seed = spec
    if agent == "ntuple_search":
        net = NTupleNetwork.load()
        searcher = Expectimax(budget=budget, evaluator=net.value, rewards=True)
    else:
        fast = agent == "expectimax"
        searcher = Expectimax(budget=budget, cache=fast, prob_cutoff=1e-4 if fast else 0.0)
    depths = []

    def choose(board):
        info = searcher.search(board)
        depths.append(info.depth)
        return info.move

    board, score, moves = play_game(choose, random.Random(seed))
    return agent, score, max_exponent(board), moves, statistics.mean(depths) if depths else 0


def summarize(name, games):
    """games: [(score, max exponent, moves, ...)] -> one Markdown row."""
    scores = [g[0] for g in games]
    tiles = [g[1] for g in games]
    reach = lambda k: 100 * sum(t >= k for t in tiles) / len(tiles)  # noqa: E731
    return (f"| {name} | {len(games)} | {statistics.mean(scores):,.0f} | {statistics.median(scores):,.0f} "
            f"| {reach(11):.0f}% | {reach(12):.0f}% | {reach(13):.0f}% | {1 << max(tiles)} |")


HEADER = ("| Agent | Games | Mean score | Median score | Reached 2048 | Reached 4096 | Reached 8192 | Best tile |\n"
          "|---|---|---|---|---|---|---|---|")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["greedy", "search"])
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--budget", type=float, default=0.1)
    p.add_argument("--agents", nargs="*", default=["expectimax_full", "expectimax", "ntuple_search"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)

    if args.mode == "greedy":
        start = time.time()
        games = greedy_games(args.games, args.seed)
        row = summarize("N-tuple network, greedy (no search)", games)
        print(HEADER + "\n" + row, f"\n({time.time() - start:.0f} s)")
        (RESULTS / "game2048_greedy.md").write_text(HEADER + "\n" + row + "\n")
        (RESULTS / "game2048_greedy.json").write_text(json.dumps(games))
        return

    specs = [(a, args.budget, args.seed + i) for a in args.agents for i in range(args.games)]
    by_agent = {a: [] for a in args.agents}
    with Pool(args.workers) as pool:
        for agent, score, tile, moves, depth in pool.imap_unordered(_search_game, specs):
            by_agent[agent].append((score, tile, moves, depth))
            print(f"{agent}: score {score} tile {1 << tile} moves {moves} mean depth {depth:.2f}", flush=True)
    names = {"expectimax_full": "Expectimax, tuned eval, full-width (no cache or cutoff)",
             "expectimax": "Expectimax, tuned eval + cache + probability cutoff",
             "ntuple_search": "Expectimax, learned n-tuple eval"}
    rows = [summarize(f"{names[a]} ({args.budget * 1000:.0f} ms/move)", g) for a, g in by_agent.items()]
    table = HEADER + "\n" + "\n".join(rows)
    print(table)
    (RESULTS / "game2048_search.md").write_text(table + "\n")
    (RESULTS / "game2048_search.json").write_text(json.dumps(by_agent))


if __name__ == "__main__":
    main()
