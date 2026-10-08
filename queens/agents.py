"""Four agents for N-Queens. Each returns a Result, so the benchmark, CLI and server treat them alike.

  backtrack  row-by-row depth-first search with bitmasks. Systematic: it proves when no solution exists.
  hill       steepest-ascent hill climbing on the conflict count, with random restarts.
  anneal     simulated annealing on the conflict count, one proposed queen move at a time.
  minconf    min-conflicts repair (Minton et al.) from a greedy start. Local search, but each repair
             scans every column of one row, so it stays cheap even for a million queens.

A "step" means a different thing for each agent, and the Result says which in its docstrings:
  backtrack  one change to the partial board: a queen placed, or a queen taken back
  hill       one move played; a restart counts as a step too
  anneal     one proposed move (accepted or rejected)
  minconf    one repair: a conflicted queen moved to a best column

"evaluations" counts candidate squares scored, the work measure that is comparable across agents.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from .board import Board

TRACE_POINTS = 400  # the conflict trace is thinned to about this many samples
FRAME_CAP = 600  # board frames (for the watch view) are thinned to about this many
GREEDY_TRIES = 4096  # random columns a row tries in the greedy start before settling for the best seen
REJECTION_TRIES = 4000  # random rows min-conflicts tries to find a conflicted one before scanning all rows
DEFAULT_MAX_STEPS = 10_000_000


@dataclass
class Result:
    """What one agent run produced. `cols[r]` is the column of the row-r queen, or -1 if the row is empty."""

    agent: str
    n: int
    solved: bool
    reason: str  # "solved", "step cap", "time cap", "restart cap", or "exhausted" (no solution exists)
    steps: int
    evaluations: int
    seconds: float
    conflicts: int
    cols: list[int]
    restarts: int = 0
    trace: list[int] = field(default_factory=list)  # conflicts (backtrack: rows still open) after each sampled step
    frames: list[list[int]] | None = None  # placements at sampled steps, when requested
    trace_steps: list[int] = field(default_factory=list)  # the step number of each trace sample
    frame_steps: list[int] | None = None  # the step number of each frame


class Recorder:
    """Keeps a trace and optional frames of bounded length, whatever the run length.

    Every step is kept until the list holds 2x its target. Then every second sample is dropped and the
    stride doubles, so the kept samples are always the steps that are multiples of the stride. A
    16-queens run of 60 steps keeps all 60; a million-step run keeps about TRACE_POINTS of them. Step 0
    (the starting placement) is always kept, and finish() adds the final state.
    """

    def __init__(self, frames: bool):
        self.trace_stride = 1
        self.trace: list[int] = []
        self.trace_steps: list[int] = []  # the step number of each trace sample
        self.frame_stride = 1
        self.frames: list[list[int]] | None = [] if frames else None
        self.frame_steps: list[int] = []

    def tick(self, step: int, value: int, cols: list[int]) -> None:
        if step % self.trace_stride == 0:
            self.trace.append(value)
            self.trace_steps.append(step)
            if len(self.trace) > 2 * TRACE_POINTS:
                self.trace = self.trace[::2]
                self.trace_steps = self.trace_steps[::2]
                self.trace_stride *= 2
        if self.frames is not None and step % self.frame_stride == 0:
            self.frames.append(list(cols))
            self.frame_steps.append(step)
            if len(self.frames) > 2 * FRAME_CAP:
                self.frames = self.frames[::2]
                self.frame_steps = self.frame_steps[::2]
                self.frame_stride *= 2

    def finish(self, step: int, value: int, cols: list[int]) -> None:
        """Add the final state unless it is already the last sample (compared by value, not by step,
        because thinning can drop the last step's sample)."""
        if not self.trace or self.trace[-1] != value:
            self.trace.append(value)
            self.trace_steps.append(step)
        if self.frames is not None and (not self.frames or self.frames[-1] != cols):
            self.frames.append(list(cols))
            self.frame_steps.append(step)


def random_cols(n: int, rng: random.Random) -> list[int]:
    """A uniformly random placement: each row gets an independent random column."""
    return [rng.randrange(n) for _ in range(n)]


def _deadline(time_limit: float | None) -> float:
    return time.perf_counter() + time_limit if time_limit is not None else math.inf


# ── backtracking ─────────────────────────────────────────────────────────────

def backtrack(n: int, *, max_steps: int | None = None, time_limit: float | None = None,
              record_frames: bool = False) -> Result:
    """Row-by-row depth-first search with column and diagonal bitmasks.

    For row r the search keeps three n-bit masks built from the queens above:
      cols  columns already taken
      ld    columns hit by the down-left diagonal from above; the mask shifts left one bit per row
      rd    columns hit by the down-right diagonal from above; the mask shifts right one bit per row
    so the open columns of row r are full & ~(cols | ld | rd). The lowest open bit is tried first.
    The search is iterative because a recursion of depth 10,000 would overflow Python's stack.

    It is exhaustive: returning "exhausted" proves no solution exists (n = 2 and 3). Its partial
    boards never contain a conflict, because a queen is only placed on an open square. So the trace
    counts rows still open, not conflicts. Plain row-order search is deterministic: a second run
    gives the same result, so the benchmark runs it once per size.
    """
    t0 = time.perf_counter()
    deadline = _deadline(time_limit)
    rec = Recorder(record_frames)
    full = (1 << n) - 1
    cols_at = [0] * (n + 1)  # per row: columns taken above it
    ld_at = [0] * (n + 1)
    rd_at = [0] * (n + 1)
    open_at = [0] * (n + 1)  # per row: the columns still to try there
    placed = [-1] * n  # the partial board; rows at or past the current depth are -1
    open_at[0] = full
    row = 0
    steps = 0  # placements plus retreats
    nodes = 0  # placements: each one examined a set of open columns
    rec.tick(0, n, placed)
    reason = "solved"
    while row < n:
        bits = open_at[row]
        if bits == 0:
            # Dead end: no open column in this row. Retreat and try the next column of the row above.
            if row == 0:
                reason = "exhausted"
                break
            row -= 1
            placed[row] = -1
            steps += 1
        else:
            bit = bits & -bits  # lowest open column
            open_at[row] = bits ^ bit
            placed[row] = bit.bit_length() - 1
            # Masks for the next row: this queen's column is taken, and its diagonals move one column
            # outward per row. The left mask is clipped to n bits; the right one drops bits off the end.
            c = cols_at[row] | bit
            ld = ((ld_at[row] | bit) << 1) & full
            rd = (rd_at[row] | bit) >> 1
            row += 1
            cols_at[row] = c
            ld_at[row] = ld
            rd_at[row] = rd
            open_at[row] = full & ~(c | ld | rd)
            steps += 1
            nodes += 1
        rec.tick(steps, n - row, placed)
        if max_steps is not None and steps >= max_steps:
            reason = "step cap"
            break
        if steps & 255 == 0 and time.perf_counter() > deadline:
            reason = "time cap"
            break
    solved = row == n and reason == "solved"
    rec.finish(steps, 0 if solved else n - row, placed)
    return Result("backtrack", n, solved, reason, steps, nodes, time.perf_counter() - t0, 0,
                  list(placed), 0, rec.trace, rec.frames,
                  rec.trace_steps, rec.frame_steps)


# ── steepest-ascent hill climbing ────────────────────────────────────────────

def hill_climb(n: int, rng: random.Random, *, max_steps: int | None = None, max_restarts: int | None = None,
               time_limit: float | None = None, record_frames: bool = False) -> Result:
    """Steepest-ascent hill climbing on the conflict count, with random restarts.

    A step scores every move (row, column) and plays the one that lowers the conflict count the most,
    ties broken at random. A move's cost change is attacks_if(new) - queen_attacks(old), so the scan
    is n * n candidate squares per step. When no move lowers the count, the search is at a local
    minimum or on a plateau. Sideways moves are not allowed, so it restarts from a fresh placement.
    Each restart is counted as a step and as a restart.
    """
    t0 = time.perf_counter()
    deadline = _deadline(time_limit)
    rec = Recorder(record_frames)
    board = Board(n, random_cols(n, rng))
    steps = evals = restarts = 0
    rec.tick(0, board.conflicts, board.cols)
    reason = "solved"
    while board.conflicts > 0:
        if max_steps is not None and steps >= max_steps:
            reason = "step cap"
            break
        if time.perf_counter() > deadline:
            reason = "time cap"
            break
        # Score every move. Inlined lookups: this loop is the hot path for hill climbing.
        col, d1, d2, cur_cols = board.col, board.d1, board.d2, board.cols
        best_d, best = 0, []
        for r in range(n):
            cur = cur_cols[r]
            base = col[cur] + d1[r + cur] + d2[r - cur + n - 1] - 3  # attacks on the row-r queen itself
            for c in range(n):
                if c == cur:
                    continue
                d = col[c] + d1[r + c] + d2[r - c + n - 1] - base
                if d < best_d:
                    best_d, best = d, [(r, c)]
                elif d == best_d and d < 0:
                    best.append((r, c))
        evals += n * n
        if best:
            r, c = best[rng.randrange(len(best))]
            board.move(r, c)
        else:
            # Local minimum or plateau: start again from a random placement.
            restarts += 1
            if max_restarts is not None and restarts > max_restarts:
                reason = "restart cap"
                break
            board = Board(n, random_cols(n, rng))
        steps += 1
        rec.tick(steps, board.conflicts, board.cols)
    if board.conflicts == 0:
        reason = "solved"
    rec.finish(steps, board.conflicts, board.cols)
    return Result("hill", n, board.conflicts == 0, reason, steps, evals, time.perf_counter() - t0,
                  board.conflicts, list(board.cols), restarts, rec.trace, rec.frames,
                  rec.trace_steps, rec.frame_steps)


# ── simulated annealing ──────────────────────────────────────────────────────

def anneal(n: int, rng: random.Random, *, max_steps: int = DEFAULT_MAX_STEPS, run_length: int | None = None,
           t0: float = 2.0, t1: float = 0.05, time_limit: float | None = None,
           record_frames: bool = False) -> Result:
    """Simulated annealing on the conflict count, one proposed move at a time.

    A proposal moves a random queen to a random other column. Its cost change d is
    attacks_if(new) - queen_attacks(old), an O(1) lookup. A move with d <= 0 is always taken, and an
    uphill move of size d is taken with probability exp(-d / T). The temperature falls geometrically
    from t0 to t1 over a run of run_length proposals. A run that ends unsolved starts again from a fresh
    random placement, and max_steps caps the total proposals.
    """
    t_start = time.perf_counter()
    deadline = _deadline(time_limit)
    rec = Recorder(record_frames)
    if run_length is None:
        run_length = max(200, 60 * n)
    board = Board(n, random_cols(n, rng))
    steps = evals = restarts = 0
    temp = t0
    cool = (t1 / t0) ** (1.0 / run_length)  # per-proposal multiplier that takes t0 to t1 over one run
    run_pos = 0
    rec.tick(0, board.conflicts, board.cols)
    reason = "solved"
    while board.conflicts > 0:
        if steps >= max_steps:
            reason = "step cap"
            break
        if steps & 1023 == 0 and time.perf_counter() > deadline:
            reason = "time cap"
            break
        if run_pos == run_length:
            restarts += 1
            board = Board(n, random_cols(n, rng))
            temp, run_pos = t0, 0
            rec.tick(steps, board.conflicts, board.cols)
            continue
        r = rng.randrange(n)
        cur = board.cols[r]
        c = (cur + 1 + rng.randrange(n - 1)) % n  # uniform over the other n-1 columns
        d = board.attacks_if(r, c) - board.queen_attacks(r)
        if d <= 0 or rng.random() < math.exp(-d / temp):
            board.move(r, c)
        evals += 1
        steps += 1
        run_pos += 1
        temp *= cool
        rec.tick(steps, board.conflicts, board.cols)
    if board.conflicts == 0:
        reason = "solved"
    rec.finish(steps, board.conflicts, board.cols)
    return Result("anneal", n, board.conflicts == 0, reason, steps, evals, time.perf_counter() - t_start,
                  board.conflicts, list(board.cols), restarts, rec.trace, rec.frames,
                  rec.trace_steps, rec.frame_steps)


# ── min-conflicts ────────────────────────────────────────────────────────────

class _Lines:
    """Queen counts per column and diagonal, kept twice: as lists for scalar work, and as numpy arrays
    so a whole row can be scored in one vectorised pass. Both copies are updated together in add/drop."""

    def __init__(self, n: int):
        self.n = n
        self.col = [0] * n
        self.d1 = [0] * (2 * n - 1)
        self.d2 = [0] * (2 * n - 1)
        self.col_np: np.ndarray | None = None  # made by sync(), once the greedy start is done
        self.d1_np: np.ndarray | None = None
        self.d2_np: np.ndarray | None = None

    def sync(self) -> None:
        """Make the numpy copies from the lists. Called once after the greedy start, when the lists are
        final: building the arrays in one go is far cheaper than updating them a million times."""
        self.col_np = np.array(self.col, dtype=np.int64)
        self.d1_np = np.array(self.d1, dtype=np.int64)
        self.d2_np = np.array(self.d2, dtype=np.int64)

    def add(self, r: int, c: int) -> int:
        """Place a queen; returns the conflicts it creates (the queens already on its three lines)."""
        n = self.n
        i1, i2 = r + c, r - c + n - 1
        created = self.col[c] + self.d1[i1] + self.d2[i2]
        self.col[c] += 1
        self.d1[i1] += 1
        self.d2[i2] += 1
        self.col_np[c] += 1
        self.d1_np[i1] += 1
        self.d2_np[i2] += 1
        return created

    def drop(self, r: int, c: int) -> int:
        """Take a queen off; returns the conflicts it removes (a negative number)."""
        n = self.n
        i1, i2 = r + c, r - c + n - 1
        self.col[c] -= 1
        self.d1[i1] -= 1
        self.d2[i2] -= 1
        self.col_np[c] -= 1
        self.d1_np[i1] -= 1
        self.d2_np[i2] -= 1
        return -(self.col[c] + self.d1[i1] + self.d2[i2])

    def queen_sums(self, cols_np: np.ndarray) -> np.ndarray:
        """For every row at once: its column count + its two diagonal counts. A queen is in conflict when
        this is above 3, since its own three lines contribute 3."""
        n = self.n
        idx = np.arange(n)
        return self.col_np[cols_np] + self.d1_np[idx + cols_np] + self.d2_np[idx - cols_np + n - 1]

    def scan(self, r: int) -> np.ndarray:
        """Attacks on every column of row r, as one array (index = column).

        Column c uses column count col[c], diagonal d1[r + c], and diagonal d2[r - c + n - 1]. For
        c = 0 .. n-1 the d1 indices are the contiguous slice r .. r+n-1. The d2 indices run the
        other way, so the slice is reversed. Both are views, so the scan costs one pass over n.
        """
        n = self.n
        return self.col_np + self.d1_np[r:r + n] + self.d2_np[r:r + n][::-1]


def _start(n: int, rng: random.Random) -> tuple[_Lines, list[int], int, int]:
    """The greedy start for min-conflicts: returns (line counts, columns, conflicts, candidate squares scored).

    Rows are filled top to bottom. Each row tries up to GREEDY_TRIES random unused columns and keeps the
    first one no queen attacks, or else the least attacked one tried. Near the end of the fill almost
    every column is used and a free diagonal is rare, so a row can miss; those misses are the conflicts
    that the repair phase has to fix. Measured at 4,096 tries: about 10 conflicts are left at every size
    from 1,000 to 1,000,000 queens, and the repairs take 26 to 87 steps.
    """
    lines = _Lines(n)
    cols = [-1] * n
    conflicts = 0
    evals = 0
    # Only an unused column (no queen in it yet) can be attack-free for the row, because any used column
    # already has a queen attacking along it. So the tries are drawn from the unused columns, kept in a
    # list with O(1) removal. Late in the fill this is what keeps the tries few.
    # The loop runs once per queen, so it uses local names and rng.random (about 3x cheaper than
    # randrange). Index = int(u * m) for u in [0, 1).
    col, d1, d2 = lines.col, lines.d1, lines.d2
    rnd = rng.random
    free = list(range(n))  # unused columns
    pos = list(range(n))  # pos[c] = where column c sits in free
    for r in range(n):
        best_c, best_a = -1, 1 << 60
        d2_base = r + n - 1  # d2 index for column c is d2_base - c
        m = len(free)
        tries = 0
        while tries < GREEDY_TRIES:
            c = free[int(rnd() * m)]
            tries += 1
            a = d1[r + c] + d2[d2_base - c]  # col[c] is 0 for an unused column
            if a < best_a:
                best_c, best_a = c, a
                if a == 0:
                    break
        evals += tries
        # The column is used from now on: take it off the free list (swap with the last entry, then pop).
        i = pos[best_c]
        last = free[-1]
        free[i] = last
        pos[last] = i
        free.pop()
        # Place the queen on the lists only (the same arithmetic as _Lines.add, without the numpy copies).
        conflicts += col[best_c] + d1[r + best_c] + d2[d2_base - best_c]
        col[best_c] += 1
        d1[r + best_c] += 1
        d2[d2_base - best_c] += 1
        cols[r] = best_c
    lines.sync()
    return lines, cols, conflicts, evals


def min_conflicts(n: int, rng: random.Random, *, max_steps: int | None = None, time_limit: float | None = None,
                  record_frames: bool = False) -> Result:
    """Min-conflicts repair (Minton, Johnston, Philips and Laird, 1992), started from a greedy placement.

    Start. The greedy placement described in _start.

    Restart. Min-conflicts can cycle on a plateau: at eight queens a single conflict can move between
    configurations forever without finding a free column. If the conflict count has not reached a new low
    for max(200, 20 n) steps, the run starts again from a fresh greedy placement, and the result counts
    the restarts.

    Repair. Pick a conflicted queen uniformly at random (see REJECTION_TRIES). Take it off, then score
    every column for its row with _Lines.scan and move it to a column with the fewest attacks. Ties are
    broken at random, and the current column is one of the candidates. A step costs a few passes over
    n numbers in C, so the number of steps is what decides the running time.
    """
    t0 = time.perf_counter()
    deadline = _deadline(time_limit)
    budget = max_steps if max_steps is not None else DEFAULT_MAX_STEPS
    rec = Recorder(record_frames)
    lines, cols, conflicts, evals = _start(n, rng)
    steps = 0
    restarts = 0
    best_seen = conflicts  # fewest conflicts so far in this start; a start that stalls is restarted
    stall = 0
    stall_limit = max(200, 20 * n)
    rec.tick(0, conflicts, cols)
    reason = "solved"
    cols_np = np.array(cols, dtype=np.int64)  # the same placement as an array, for the vectorised conflict test
    while conflicts > 0:
        if steps >= budget:
            reason = "step cap"
            break
        if steps & 127 == 0 and time.perf_counter() > deadline:
            reason = "time cap"
            break
        if stall > stall_limit:
            # Plateau trap: repairs cycle among configurations with the same conflict count. Min-conflicts
            # has no way out of such a cycle, so start again from a fresh greedy placement.
            restarts += 1
            lines, cols, conflicts, added = _start(n, rng)
            evals += added
            cols_np = np.array(cols, dtype=np.int64)
            best_seen, stall = conflicts, 0
            rec.tick(steps, conflicts, cols)
            continue
        # Pick a conflicted row uniformly. Random tries are cheap while conflicts are common. When they are
        # rare, a random row is almost never conflicted, so after REJECTION_TRIES misses one vectorised pass
        # lists every conflicted row instead. Both paths choose uniformly, so the rule is unchanged.
        r = -1
        for _ in range(REJECTION_TRIES):
            cand = rng.randrange(n)
            c = cols[cand]
            if lines.col[c] + lines.d1[cand + c] + lines.d2[cand - c + n - 1] > 3:
                r = cand
                break
        if r < 0:
            bad = np.flatnonzero(lines.queen_sums(cols_np) > 3)
            r = int(bad[rng.randrange(len(bad))])
        c = cols[r]
        conflicts += lines.drop(r, c)
        scores = lines.scan(r)
        best = scores.min()
        candidates = np.flatnonzero(scores == best)
        new_c = int(candidates[rng.randrange(len(candidates))])
        conflicts += lines.add(r, new_c)
        cols[r] = new_c
        cols_np[r] = new_c
        evals += n
        steps += 1
        if conflicts < best_seen:
            best_seen, stall = conflicts, 0
        else:
            stall += 1
        rec.tick(steps, conflicts, cols)
    rec.finish(steps, conflicts, cols)
    return Result("minconf", n, conflicts == 0, reason, steps, evals, time.perf_counter() - t0,
                  conflicts, cols, restarts, rec.trace, rec.frames,
                  rec.trace_steps, rec.frame_steps)


# ── dispatch ─────────────────────────────────────────────────────────────────

AGENT_NAMES = ("backtrack", "hill", "anneal", "minconf")


def run(agent: str, n: int, seed: int | None = None, **kwargs) -> Result:
    """Run one agent by name. Keyword arguments are passed to that agent (caps, frames, time limit)."""
    rng = random.Random(seed)
    if agent == "backtrack":
        return backtrack(n, **kwargs)
    if agent == "hill":
        return hill_climb(n, rng, **kwargs)
    if agent == "anneal":
        return anneal(n, rng, **kwargs)
    if agent == "minconf":
        return min_conflicts(n, rng, **kwargs)
    raise ValueError(f"unknown agent {agent!r}: choose from {', '.join(AGENT_NAMES)}")
