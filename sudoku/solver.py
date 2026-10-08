"""Sudoku as a constraint satisfaction problem: three backtracking solvers.

Boards are 81-character strings read row by row, with 0 or '.' for empty cells.

  plain      backtracking over cells in row order, trying values 1-9
  mrv_fc     minimum-remaining-values cell choice plus forward checking
             (undo an assignment that leaves any neighbor with no legal value)
  propagate  bitmask domains with constraint propagation run to a fixed
             point after every assignment (naked singles: a cell with one
             candidate takes it; hidden singles: a digit with one place in
             a unit goes there), then MRV branching

Every solver returns a SolveResult with statistics, and can record a trace
of assignments and backtracks for visualization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Each cell's 20 peers: the other cells in its row, column, and 3x3 box.
UNITS = ([[9 * r + c for c in range(9)] for r in range(9)]
         + [[9 * r + c for r in range(9)] for c in range(9)]
         + [[9 * (br + r) + bc + c for r in range(3) for c in range(3)]
            for br in (0, 3, 6) for bc in (0, 3, 6)])
CELL_UNITS = [[u for u in UNITS if i in u] for i in range(81)]
PEERS = [sorted({j for u in CELL_UNITS[i] for j in u} - {i}) for i in range(81)]
ALL = 0x3FE  # bits 1..9
BIT = [1 << d for d in range(10)]


def parse(text: str) -> list[int]:
    cells = [0 if ch in "0." else int(ch) for ch in text.strip() if ch in "0123456789."]
    if len(cells) != 81:
        raise ValueError(f"expected 81 cells, got {len(cells)}")
    return cells


def to_string(cells) -> str:
    return "".join(str(v) for v in cells)


def is_valid_solution(cells, puzzle=None) -> bool:
    if any(not 1 <= v <= 9 for v in cells):
        return False
    if puzzle is not None and any(p and p != v for p, v in zip(puzzle, cells)):
        return False
    return all(sorted(cells[i] for i in u) == list(range(1, 10)) for u in UNITS)


def givens_consistent(cells) -> bool:
    return all(not cells[i] or all(cells[j] != cells[i] for j in PEERS[i]) for i in range(81))


@dataclass
class SolveResult:
    solver: str
    status: str  # "solved", "unsolvable", or "limit"
    solution: str | None = None
    nodes: int = 0          # assignments tried
    backtracks: int = 0     # assignments undone
    seconds: float = 0.0
    # (kind, cell, value): kind is "assign", "undo", or "propagate"
    trace: list = field(default_factory=list)


class _Limit(Exception):
    pass


class _Run:
    def __init__(self, name, max_nodes, trace_limit):
        self.name = name
        self.max_nodes = max_nodes
        self.trace_limit = trace_limit
        self.nodes = self.backtracks = 0
        self.trace = []
        self.start = time.perf_counter()

    def assign(self, cell, value, kind="assign"):
        if kind == "assign":
            self.nodes += 1
            if self.max_nodes is not None and self.nodes > self.max_nodes:
                raise _Limit
        if len(self.trace) < self.trace_limit:
            self.trace.append((kind, cell, value))

    def undo(self, cell):
        self.backtracks += 1
        if len(self.trace) < self.trace_limit:
            self.trace.append(("undo", cell, 0))

    def result(self, status, cells=None):
        return SolveResult(self.name, status, to_string(cells) if cells else None,
                           self.nodes, self.backtracks, time.perf_counter() - self.start, self.trace)


def _legal(cells, i):
    used = 0
    for j in PEERS[i]:
        used |= BIT[cells[j]]
    return [d for d in range(1, 10) if not used & BIT[d]]


# ── plain backtracking ──────────────────────────────────────────────────

def solve_plain(puzzle: str, max_nodes=None, trace_limit=0) -> SolveResult:
    cells = parse(puzzle)
    run = _Run("plain", max_nodes, trace_limit)
    if not givens_consistent(cells):
        return run.result("unsolvable")
    empties = [i for i in range(81) if not cells[i]]

    def bt(k):
        if k == len(empties):
            return True
        i = empties[k]
        for d in _legal(cells, i):
            cells[i] = d
            run.assign(i, d)
            if bt(k + 1):
                return True
            cells[i] = 0
            run.undo(i)
        return False

    try:
        return run.result("solved", cells) if bt(0) else run.result("unsolvable")
    except _Limit:
        return run.result("limit")


# ── MRV + forward checking ──────────────────────────────────────────────

def solve_mrv_fc(puzzle: str, max_nodes=None, trace_limit=0) -> SolveResult:
    """Choose the empty cell with the fewest legal values; after each
    assignment, undo it if any empty peer is left with none."""
    cells = parse(puzzle)
    run = _Run("mrv_fc", max_nodes, trace_limit)
    if not givens_consistent(cells):
        return run.result("unsolvable")

    def select_mrv():
        best, best_n = None, 10
        for i in range(81):
            if not cells[i]:
                n = len(_legal(cells, i))
                if n < best_n:
                    best, best_n = i, n
                    if n <= 1:
                        break
        return best

    def forward_check(i):
        return all(cells[j] or _legal(cells, j) for j in PEERS[i])

    def bt():
        i = select_mrv()
        if i is None:
            return True
        for d in _legal(cells, i):
            cells[i] = d
            run.assign(i, d)
            if forward_check(i) and bt():
                return True
            cells[i] = 0
            run.undo(i)
        return False

    try:
        return run.result("solved", cells) if bt() else run.result("unsolvable")
    except _Limit:
        return run.result("limit")


# ── bitmask domains + propagation + MRV ─────────────────────────────────

def solve_propagate(puzzle: str, max_nodes=None, trace_limit=0) -> SolveResult:
    cells = parse(puzzle)
    run = _Run("propagate", max_nodes, trace_limit)
    if not givens_consistent(cells):
        return run.result("unsolvable")

    domains = [ALL] * 81
    for i, v in enumerate(cells):
        if v:
            domains[i] = BIT[v]

    def propagate(dom, changed):
        """Run naked and hidden singles to a fixed point. Returns False on a contradiction."""
        queue = list(changed)
        while queue:
            i = queue.pop()
            if dom[i] & (dom[i] - 1):
                continue  # not a single value
            bit = dom[i]
            for j in PEERS[i]:
                if dom[j] & bit:
                    dom[j] &= ~bit
                    if not dom[j]:
                        return False
                    if not dom[j] & (dom[j] - 1):
                        run.assign(j, dom[j].bit_length() - 1, "propagate")
                        queue.append(j)
            for unit in CELL_UNITS[i]:
                for d in range(1, 10):
                    places = [j for j in unit if dom[j] & BIT[d]]
                    if not places:
                        return False
                    if len(places) == 1 and dom[places[0]] != BIT[d]:
                        dom[places[0]] = BIT[d]
                        run.assign(places[0], d, "propagate")
                        queue.append(places[0])
        return True

    def search(dom):
        best, best_n = None, 10
        for i in range(81):
            n = bin(dom[i]).count("1")
            if 1 < n < best_n:
                best, best_n = i, n
        if best is None:
            return dom
        for d in range(1, 10):
            if dom[best] & BIT[d]:
                trial = dom[:]
                trial[best] = BIT[d]
                run.assign(best, d)
                if propagate(trial, [best]):
                    solved = search(trial)
                    if solved is not None:
                        return solved
                run.undo(best)
        return None

    try:
        if not propagate(domains, [i for i in range(81) if cells[i]]):
            return run.result("unsolvable")
        solved = search(domains)
    except _Limit:
        return run.result("limit")
    if solved is None:
        return run.result("unsolvable")
    return run.result("solved", [d.bit_length() - 1 for d in solved])


SOLVERS = {"plain": solve_plain, "mrv_fc": solve_mrv_fc, "propagate": solve_propagate}
