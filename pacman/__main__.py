"""Pac-Man from the command line.

    python -m pacman                          play in the terminal (w/a/s/d + Enter each turn)
    python -m pacman --maze vault --seed 7    the same, on another maze or seed
    python -m pacman --watch q                watch an agent play: q, reflex, or random
    python -m pacman train --episodes 3000    learn the Q-agent's weights into pacman/data/weights.json
    python -m pacman benchmark --games 200    win rate and average score for each agent
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .agents import AGENTS, WEIGHTS_PATH, save_weights
from .benchmark import run_benchmark, to_markdown
from .mazes import names
from .qlearning import train
from .terminal import play_human, watch

RESULTS = Path(__file__).resolve().parent.parent / "results"


def cmd_train(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m pacman train", description="Learn Q-agent weights by self-play.")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alpha", type=float, nargs=2, default=(0.01, 0.001), metavar=("START", "END"),
                    help="learning rate, falling linearly from START to END")
    ap.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    ap.add_argument("--out", type=Path, default=WEIGHTS_PATH)
    ap.add_argument("--log", type=Path, default=RESULTS / "pacman_train_log.jsonl")
    args = ap.parse_args(argv)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w") as log:
        def on_log(record: dict) -> None:
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps({k: v for k, v in record.items() if k != "weights"}), flush=True)

        result = train(args.episodes, seed=args.seed, alpha=tuple(args.alpha), gamma=args.gamma, on_log=on_log)
    trained = {"episodes": args.episodes, "seed": args.seed, "alpha": list(args.alpha), "gamma": args.gamma,
               "mazes": ["lanes", "vault"], "final": result.history[-1]}
    save_weights(args.out, result.weights, trained)
    print(f"wrote {args.out}")
    return 0


def cmd_benchmark(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m pacman benchmark", description="Win rate and score per agent.")
    ap.add_argument("--games", type=int, default=200, help="games per agent per maze")
    ap.add_argument("--out", type=Path, default=RESULTS / "pacman_benchmark.json")
    args = ap.parse_args(argv)

    result = run_benchmark(args.games)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(to_markdown(result))
    print(to_markdown(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["train"]:
        return cmd_train(argv[1:])
    if argv[:1] == ["benchmark"]:
        return cmd_benchmark(argv[1:])

    ap = argparse.ArgumentParser(prog="python -m pacman", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maze", choices=names(), default="lanes", help="maze to play (default lanes)")
    ap.add_argument("--seed", type=int, default=None, help="game seed (default: random)")
    ap.add_argument("--watch", choices=AGENTS, help="watch an agent instead of playing")
    ap.add_argument("--delay", type=float, default=0.25, help="seconds between turns when watching")
    args = ap.parse_args(argv)

    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    try:
        if args.watch:
            watch(args.watch, args.maze, seed, delay=args.delay)
        else:
            status = play_human(args.maze, seed)
            print({"won": "You cleared the maze!", "lost": "Caught by a ghost.",
                   "timeout": "Out of turns.", "quit": "You quit."}[status])
    except FileNotFoundError as e:  # Q-agent weights not trained yet
        print(e, file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
