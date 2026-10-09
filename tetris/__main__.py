"""Command-line tools for Tetris: play, watch the agent, tune the weights, and benchmark.

    python -m tetris                       # play in the terminal (same as 'play')
    python -m tetris play --seed 7         # turn-based game, 'h' shows the agent's placement
    python -m tetris watch --pieces 300    # the agent plays a seeded game and prints every step
    python -m tetris evolve --gens 30      # run the GA and write results/tetris_ga_weights.json + the log
    python -m tetris benchmark --games 30  # the four-strategy comparison, written to results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .board import render
from .features import FEATURES, HAND_WEIGHTS
from .game import PieceBag, greedy_policy, lookahead_policy

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def cmd_play(args) -> int:
    from . import terminal

    return terminal.run(seed=args.seed)


def cmd_watch(args) -> int:
    """The agent plays one seeded game, printing the board after every placement."""
    from .board import empty_board
    from .tuned import load_tuned

    weights = load_tuned()["weights"]
    policy = lookahead_policy(weights, top_k=6) if args.lookahead else greedy_policy(weights)
    bag = PieceBag(args.seed)
    board = empty_board()
    lines = 0
    for n in range(1, args.pieces + 1):
        piece = bag.take()
        preview = bag.peek()
        move = policy(board, piece, preview)
        if move is None:
            print(f"top out after {n - 1} pieces with {lines} lines")
            return 0
        board = move.board
        lines += move.lines
        if args.delay:
            print("\033[2J\033[H", end="")  # clear the screen so the board animates in place
        print(f"piece {n}: {piece} into rotation {move.rot} at column {move.x + 1}, "
              f"clears {move.lines}, total {lines} lines")
        print(render(board))
        if args.delay:
            time.sleep(args.delay)
    print(f"capped at {args.pieces} pieces with {lines} lines, still alive")
    return 0


def cmd_evolve(args) -> int:
    from .evolve import GAConfig, evolve, save_tuned

    cfg = GAConfig(population=args.pop, generations=args.gens, train_games=args.games,
                   train_pieces=args.pieces, board_height=args.height, seed=args.seed)
    print(f"GA: population {cfg.population}, generations {cfg.generations}, {cfg.train_games} training games "
          f"per genome (fresh seeds each generation), cap {cfg.train_pieces} pieces, {cfg.board_height} rows, "
          f"{args.workers} workers")
    result = evolve(cfg, workers=args.workers, log_path=Path(args.log) if args.log else None)
    print("weights:")
    for name, w in zip(FEATURES, result["weights"]):
        print(f"  {name:18s} {w:+.4f}")
    print(f"training fitness (mean lines): {result['fitness']}")
    out = Path(args.out)
    save_tuned(result, out)
    print(f"wrote {out}")
    return 0


def cmd_cem(args) -> int:
    """Noisy cross-entropy tuning (v3). Writes the log and the final mean as JSON; checkpoints as it goes."""
    from .cem import CEMConfig, run_cem

    cfg = CEMConfig(iterations=args.iters, samples=args.samples, games=args.games, cap=args.cap,
                    height=args.height, seed=args.seed)
    print(f"CEM: {cfg.iterations} iterations, {cfg.samples} samples, {cfg.games} games per sample (fresh seeds), "
          f"cap {cfg.cap}, {cfg.height} rows, {args.workers} workers, start at the hand-picked weights")
    result = run_cem(cfg, workers=args.workers, checkpoint=Path(args.checkpoint), log_path=Path(args.log),
                     resume=args.resume)
    print("final mean:")
    for name, w in zip(FEATURES, result["weights"]):
        print(f"  {name:18s} {w:+.4f}")
    doc = {"features": list(FEATURES), "weights": result["weights"], "config": result["config"],
           "history": result["history"], "sigma": result["sigma"], "best_sample": result["best"]}
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


def cmd_benchmark(args) -> int:
    from .benchmark import run_benchmark, to_markdown, write_results

    t0 = time.time()
    strategies = ("random", "hand", "ga") if args.no_lookahead else ("random", "hand", "ga", "ga_lookahead")
    ga_weights = None
    if args.weights:
        ga_weights = json.loads(Path(args.weights).read_text())["weights"]
    result = run_benchmark(games=args.games, cap=args.cap, workers=args.workers, strategies=strategies,
                           ga_weights=ga_weights, height=args.height)
    result["ga_source"] = args.weights or "tetris/tuned.json"
    print(to_markdown(result))
    print(f"hand-picked weights: {list(HAND_WEIGHTS)}")
    print(f"done in {time.time() - t0:.0f} s")
    if args.out:
        write_results(result, Path(args.out))
        print(f"wrote {args.out}.json and {args.out}.md")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tetris", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("play", help="play in the terminal; 'h' shows the agent's placement")
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("watch", help="the agent plays one seeded game and prints each step")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pieces", type=int, default=200)
    p.add_argument("--delay", type=float, default=0.0, help="seconds between steps (0 = as fast as possible)")
    p.add_argument("--lookahead", action="store_true", help="use one-piece lookahead with the preview")
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("evolve", help="tune the weights with the genetic algorithm")
    p.add_argument("--pop", type=int, default=16)
    p.add_argument("--gens", type=int, default=16)
    p.add_argument("--games", type=int, default=6, help="training games per genome, fresh seeds each generation")
    p.add_argument("--pieces", type=int, default=2500, help="piece cap per training game")
    p.add_argument("--height", type=int, default=10, help="rows of the training board")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log", default=str(RESULTS / "tetris_train_log.jsonl"))
    p.add_argument("--out", default=str(RESULTS / "tetris_ga_weights.json"),
                   help="where the GA weights go (default results/tetris_ga_weights.json; tetris/tuned.json is kept)")
    p.set_defaults(fn=cmd_evolve)

    p = sub.add_parser("cem", help="tune the weights with the noisy cross-entropy method (v3)")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--games", type=int, default=3, help="fresh seeded games per sample per iteration")
    p.add_argument("--cap", type=int, default=100000, help="piece cap per game (matches the benchmark)")
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--checkpoint", default="/tmp/tetris_cem/checkpoint.json")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log", default=str(RESULTS / "tetris_cem_log.jsonl"))
    p.add_argument("--out", default=str(RESULTS / "tetris_cem_weights.json"))
    p.set_defaults(fn=cmd_cem)

    p = sub.add_parser("benchmark", help="compare random, hand, GA and GA + lookahead on seeded games")
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--cap", type=int, default=2000, help="piece cap per game")
    p.add_argument("--height", type=int, default=20, help="rows of the board (20 is the game board)")
    p.add_argument("--no-lookahead", action="store_true", help="leave out the lookahead strategy")
    p.add_argument("--weights", default=None, help="weights JSON for the 'ga' row (default tetris/tuned.json)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default=str(RESULTS / "tetris_benchmark"), help="writes .json and .md; '' to skip")
    p.set_defaults(fn=cmd_benchmark)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        argv = ["play"]
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
