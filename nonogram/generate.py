"""Random pictures that have exactly one solution.

A uniformly random grid is a poor puzzle: most random pictures have many solutions, because nothing ties
neighbouring cells together. So the generator draws smooth random blobs instead:

  1. Pick random values on a coarse grid (one value per 3x3 block, plus one).
  2. Interpolate bilinearly up to the full size, so neighbouring cells share a value.
  3. Threshold at 0.5: cells above it are filled. This gives blobs with edges a line can read.
  4. Keep the picture only if its density is between 30% and 70%, so the clues are neither trivial
     nor a wall of filled cells.
  5. Count the solutions of its clues with the hybrid search, stopping at two. Keep the picture only if
     the count is exactly one. A search that exceeds its node budget counts as a rejection.

The seed fixes every draw, so a request with the same seed returns the same puzzle.
"""

from __future__ import annotations

import random

from .puzzle import Puzzle
from .solvers import count_solutions

THRESHOLD = 0.5  # cells whose interpolated noise is at least this are filled
DENSITY_MIN = 0.30  # keep pictures between these filled fractions
DENSITY_MAX = 0.70
CELL_SPAN = 3  # one noise value per this many cells, so blobs are about 3 cells across
ATTEMPTS = 40  # pictures drawn before giving up
NODE_BUDGET = 300  # hybrid guesses allowed per uniqueness check (a rejected picture costs at most about 1 s)
RANDOM_MIN, RANDOM_MAX = 5, 20  # sizes the random generator offers, per side; the server and the CLI both check this


class NoUniquePicture(RuntimeError):
    """No unique picture was found within the attempt limit."""


def random_picture(rows: int, cols: int, rng: random.Random) -> list[list[int]]:
    """A blobby 0/1 picture: bilinear interpolation of coarse random noise, thresholded."""
    # The coarse grid always has at least 2 x 2 points, so every cell has four neighbours to blend.
    gr = -(-rows // CELL_SPAN) + 1
    gc = -(-cols // CELL_SPAN) + 1
    coarse = [[rng.random() for _ in range(gc)] for _ in range(gr)]
    picture = []
    for r in range(rows):
        # The cell's position in coarse-grid units, split into a cell (y0) and a fraction (fy).
        y = r * (gr - 1) / (rows - 1) if rows > 1 else 0.0
        y0 = min(int(y), gr - 2)
        fy = y - y0
        line = []
        for c in range(cols):
            x = c * (gc - 1) / (cols - 1) if cols > 1 else 0.0
            x0 = min(int(x), gc - 2)
            fx = x - x0
            top = coarse[y0][x0] * (1 - fx) + coarse[y0][x0 + 1] * fx
            bottom = coarse[y0 + 1][x0] * (1 - fx) + coarse[y0 + 1][x0 + 1] * fx
            value = top * (1 - fy) + bottom * fy
            line.append(1 if value >= THRESHOLD else 0)
        picture.append(line)
    return picture


def random_puzzle(rows: int, cols: int, seed: int | None = None, attempts: int = ATTEMPTS,
                  node_budget: int | None = NODE_BUDGET) -> tuple[Puzzle, int, int]:
    """A puzzle with exactly one solution. Returns (puzzle, seed, attempts used).

    Raises NoUniquePicture when `attempts` pictures all fail the density or uniqueness test.
    """
    if seed is None:
        seed = random.randrange(2**31)
    rng = random.Random(seed)
    name = f"Random {rows}x{cols}"
    total = rows * cols
    for attempt in range(1, attempts + 1):
        picture = random_picture(rows, cols, rng)
        filled = sum(map(sum, picture))
        if not DENSITY_MIN <= filled / total <= DENSITY_MAX:
            continue
        puzzle = Puzzle.from_picture(name, picture)
        status, _, _ = count_solutions(puzzle, limit=2, max_guesses=node_budget)
        if status == "unique":
            return puzzle, seed, attempt
    raise NoUniquePicture(f"no unique {rows}x{cols} picture in {attempts} tries (seed {seed})")
