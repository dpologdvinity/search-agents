"""Measure the served AlphaZero network against fixed-depth alpha-beta minimax.

    python -m connect4.evaluate                 # 50 games per opponent, writes results/connect4_eval.json
    python -m connect4.evaluate --games 20 --simulations 400

This evaluates the exported NumPy weights (connect4/data/az_connect4.npz), i.e.
exactly the model the web demo and the terminal game play, so it needs no torch.

Each match alternates who moves first, and every game starts with two random
moves: minimax is deterministic, so without them half the games would be
identical replays of each other.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .board import COLS, Board
from .mcts import MCTS
from .minimax import Minimax
from .net import load
from .puct import search

RESULTS = Path(__file__).resolve().parent.parent / "results" / "connect4_eval.json"
TRAIN_LOG = RESULTS.with_name("connect4_train_log.jsonl")


def play_game(first, second, rng: np.random.Generator) -> int:
    """Play one game between two move functions; returns +1 if `first` wins, -1 if `second` wins, 0 for a draw."""
    board = Board()
    for _ in range(2):  # random opening so games differ
        board = board.play(int(rng.integers(COLS)))
    players = (first, second)
    turn = 0
    while True:
        col = players[turn](board)
        if board.is_winning_move(col):
            return 1 if turn == 0 else -1
        board = board.play(col)
        if board.is_full():
            return 0
        turn ^= 1


def match(agent, opponent, games: int, rng: np.random.Generator) -> dict:
    """Win/draw/loss counts for `agent`, which moves first in the even-numbered games."""
    counts = {"win": 0, "draw": 0, "loss": 0}
    for i in range(games):
        if i % 2 == 0:
            r = play_game(agent, opponent, rng)
        else:
            r = -play_game(opponent, agent, rng)
        counts["win" if r > 0 else "loss" if r < 0 else "draw"] += 1
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m connect4.evaluate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=50, help="games per opponent (default 50)")
    parser.add_argument("--simulations", type=int, default=200, help="PUCT simulations per move (default 200)")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6], help="minimax depths to play")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)

    net = load()

    def alphazero(board: Board) -> int:
        return int(search(board, net.evaluate, simulations=args.simulations).root.visits.argmax())

    rng = np.random.default_rng(args.seed)
    results: dict = {"games_per_opponent": args.games, "simulations": args.simulations, "seed": args.seed}
    for depth in args.depths:
        mm = Minimax(max_depth=depth)
        start = time.perf_counter()
        counts = match(alphazero, lambda b, mm=mm: mm.search(b).move, args.games, rng)
        results[f"alphazero_vs_minimax_d{depth}"] = counts
        print(f"AlphaZero vs minimax depth {depth}: {counts} ({time.perf_counter() - start:.0f} s)", flush=True)

    # Baseline: search without a learned network (random rollouts), against the middle opponent.
    mid = args.depths[len(args.depths) // 2]
    mcts = MCTS(simulations=1000, seed=args.seed)
    mm = Minimax(max_depth=mid)
    counts = match(lambda b: mcts.search(b)["move"], lambda b: mm.search(b).move, args.games, rng)
    results[f"mcts1000_vs_minimax_d{mid}"] = counts
    print(f"MCTS (1,000 rollouts) vs minimax depth {mid}: {counts}", flush=True)

    # Record which training snapshot was measured (the last line of the training log).
    if TRAIN_LOG.exists():
        last = json.loads(TRAIN_LOG.read_text().splitlines()[-1])
        results["training"] = {"iteration": last["iteration"], "games": last["games"]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
