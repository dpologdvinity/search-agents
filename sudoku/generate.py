"""Generate Sudoku puzzles with a unique solution, rated by how hard they are to solve.

    python -m sudoku.generate --count 200 --seed 0

A puzzle starts as a random complete grid (a valid base pattern with its
digits, rows, columns, row bands, and column stacks shuffled). Cells are then
removed one at a time in random order, keeping each removal only if the puzzle
still has exactly one solution. Difficulty is the number of guesses the
propagation solver needs, so "easy" puzzles fall to propagation alone.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .solver import ALL, BIT, CELL_UNITS, PEERS, to_string

PUZZLES = Path(__file__).parent / "puzzles.txt"


def random_solution(rng: random.Random) -> list[int]:
    """A uniformly shuffled variant of a valid base grid."""
    def shuffled_groups():
        groups = [0, 1, 2]
        rng.shuffle(groups)
        return [3 * g + i for g in groups for i in rng.sample(range(3), 3)]

    rows, cols = shuffled_groups(), shuffled_groups()
    digits = list(range(1, 10))
    rng.shuffle(digits)
    # Base pattern: row r, column c holds (3*(r%3) + r//3 + c) % 9.
    return [digits[(3 * (r % 3) + r // 3 + c) % 9] for r in rows for c in cols]


def count_solutions(cells: list[int], limit: int = 2) -> int:
    """Number of solutions, stopping once `limit` are found."""
    domains = [BIT[v] if v else ALL for v in cells]

    def eliminate(dom, queue):
        while queue:
            i = queue.pop()
            if dom[i] & (dom[i] - 1):
                continue
            for j in PEERS[i]:
                if dom[j] & dom[i]:
                    dom[j] &= ~dom[i]
                    if not dom[j]:
                        return False
                    if not dom[j] & (dom[j] - 1):
                        queue.append(j)
            for unit in CELL_UNITS[i]:
                for d in range(1, 10):
                    places = [j for j in unit if dom[j] & BIT[d]]
                    if not places:
                        return False
                    if len(places) == 1 and dom[places[0]] != BIT[d]:
                        dom[places[0]] = BIT[d]
                        queue.append(places[0])
        return True

    found = 0

    def search(dom):
        nonlocal found
        best, best_n = None, 10
        for i in range(81):
            n = bin(dom[i]).count("1")
            if 1 < n < best_n:
                best, best_n = i, n
        if best is None:
            found += 1
            return
        for d in range(1, 10):
            if dom[best] & BIT[d] and found < limit:
                trial = dom[:]
                trial[best] = BIT[d]
                if eliminate(trial, [best]):
                    search(trial)

    if eliminate(domains, [i for i in range(81) if cells[i]]):
        search(domains)
    return found


def make_puzzle(rng: random.Random) -> tuple[str, str]:
    """(puzzle, solution) with a unique solution and no removable clue left."""
    solution = random_solution(rng)
    cells = solution[:]
    for i in rng.sample(range(81), 81):
        kept = cells[i]
        cells[i] = 0
        if count_solutions(cells) != 1:
            cells[i] = kept
    return to_string(cells), to_string(solution)


def difficulty(puzzle: str) -> int:
    """Guesses the propagation solver needs (0 = solvable by propagation alone)."""
    from .solver import solve_propagate

    return solve_propagate(puzzle).nodes


def load_puzzles(path: Path = PUZZLES) -> list[dict]:
    """Rows of the generated set: puzzle, solution, clues, guesses."""
    rows = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            puzzle, solution, guesses = line.split()
            rows.append({"puzzle": puzzle, "solution": solution, "guesses": int(guesses),
                         "clues": sum(ch != "0" for ch in puzzle)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=PUZZLES)
    args = p.parse_args(argv)
    rng = random.Random(args.seed)
    lines = [f"# {args.count} generated puzzles (seed {args.seed}): puzzle solution guesses"]
    for k in range(args.count):
        puzzle, solution = make_puzzle(rng)
        lines.append(f"{puzzle} {solution} {difficulty(puzzle)}")
        if (k + 1) % 20 == 0:
            print(f"{k + 1}/{args.count}", flush=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

