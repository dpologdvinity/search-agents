"""Play 2048 in the terminal, or watch an agent play.

    python -m game2048                     # you play: w/a/s/d then Enter
    python -m game2048 --watch ntuple      # watch an agent play a full game
    python -m game2048 benchmark ...       # see game2048.benchmark

During your turn: w = up, s = down, a = left, d = right, h = hint, q = quit.
"""

from __future__ import annotations

import argparse
import random
import sys

from .agents import DESCRIPTIONS, choose
from .board import DIRECTIONS, legal_moves, max_exponent, move, new_game, spawn, to_grid

KEYS = {"w": 0, "s": 1, "a": 2, "d": 3}  # same order as board.UP, DOWN, LEFT, RIGHT


def render(board: int, score: int) -> str:
    """The grid with right-aligned tiles, plus the score."""
    rows = ["+------" * 4 + "+"]
    for row in to_grid(board):
        rows.append("|" + "|".join(f"{v:^6}" if v else "      " for v in row) + "|")
        rows.append("+------" * 4 + "+")
    rows.append(f"score {score:,}   best tile {1 << max_exponent(board)}")
    return "\n".join(rows)


def ask_direction(board: int, read=input) -> int | None:
    """Prompt until the player enters a legal direction; None means quit."""
    legal = {d for d, _, _ in legal_moves(board)}
    while True:
        text = read("Move (w/a/s/d, h = hint, q = quit): ").strip().lower()
        if text == "q":
            return None
        if text == "h":
            hint = choose(board, "expectimax")
            print(f"  hint: {DIRECTIONS[hint['move']]} (searched {hint['depth']} moves ahead)")
            continue
        if text in KEYS and KEYS[text] in legal:
            return KEYS[text]
        print("  that direction does not move any tile")


def play(agent: str | None, seed: int | None, read=input, quiet=False) -> tuple[int, int]:
    """Play one game; agent None means the human plays. Returns (score, best tile)."""
    rng = random.Random(seed)
    board, score = new_game(rng), 0
    while legal_moves(board):
        if not quiet:
            print("\n" + render(board, score))
        if agent is None:
            direction = ask_direction(board, read)
            if direction is None:
                break
        else:
            direction = choose(board, agent)["move"]
            if not quiet:
                print(f"{agent} plays {DIRECTIONS[direction]}")
        board, gained = move(board, direction)
        score += gained
        board = spawn(board, rng)  # the game adds a 2 (90%) or 4 (10%) after every move
    print("\n" + render(board, score))
    return score, 1 << max_exponent(board)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["benchmark"]:
        from .benchmark import main as benchmark_main

        return benchmark_main(argv[1:]) or 0
    parser = argparse.ArgumentParser(prog="python -m game2048", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", choices=sorted(DESCRIPTIONS), help="let an agent play instead of you")
    parser.add_argument("--quiet", action="store_true", help="with --watch, print only the final board")
    parser.add_argument("--seed", type=int, help="random seed for tile spawns")
    args = parser.parse_args(argv)
    try:
        score, tile = play(args.watch, args.seed, quiet=args.quiet)
    except FileNotFoundError as e:  # n-tuple weights not trained yet
        print(e, file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
        return 0
    print(f"Game over: score {score:,}, best tile {tile}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
