"""Solve one board from the command line.

    python -m npuzzle astar 7,2,4,5,0,6,8,3,1
    python -m npuzzle idastar --heuristic pdb 12,4,3,9,15,8,1,5,7,0,10,14,6,13,11,2
"""

from __future__ import annotations

import argparse
import sys

from . import ALGORITHMS, HEURISTICS, INFORMED
from .board import format_board, size_of, validate
from .search import SearchLimits

ARROWS = {"Up": "↑", "Down": "↓", "Left": "←", "Right": "→"}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m npuzzle", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("algorithm", choices=sorted(ALGORITHMS))
    parser.add_argument("board", help="tiles row by row, comma-separated, 0 = blank")
    parser.add_argument("--heuristic", choices=sorted(HEURISTICS), default="manhattan")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)

    try:
        board = validate(int(t) for t in args.board.split(","))
    except ValueError as e:
        parser.error(str(e))
    n = size_of(board)
    limits = SearchLimits(max_nodes=args.max_nodes, max_seconds=args.max_seconds)
    kwargs = {"heuristic": HEURISTICS[args.heuristic]} if args.algorithm in INFORMED else {}
    result = ALGORITHMS[args.algorithm](board, limits=limits, **kwargs)

    print(format_board(board, n))
    print(f"\n{result.algorithm}: {result.status}")
    if result.solved:
        print(f"{result.cost} moves: {' '.join(ARROWS[a] for a in result.path)}")
    print(f"expanded {result.expanded:,}, generated {result.generated:,}, "
          f"max frontier {result.max_frontier:,}, {result.seconds:.3f} s")
    return 0 if result.solved else 1


if __name__ == "__main__":
    sys.exit(main())
