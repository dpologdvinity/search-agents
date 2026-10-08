"""The three solving methods, the uniqueness check, and the hint.

  line    Repeated line passes (lines.py) until nothing changes. Stops early when the grid is stuck.
  hybrid  Line passes, then a guess on the most constrained line's first unknown cell, and backtracking.
          The search counts solutions up to a limit, so the same code proves uniqueness: it stops at the
          second solution, or reports that the search space held none.
  sat     cnf.py turns the whole puzzle into CNF, and dpll.py solves it. The uniqueness check solves
          once, adds a blocking clause that forbids that picture, and solves again.

Every solution returned is checked against the clues before it is reported, so a bug shows up as a
failed check rather than as a wrong picture. Budgets (guesses, propagations) are counted, not timed,
so a result does not depend on machine speed. A run that hits a budget reports status "undecided".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .cnf import encode
from .dpll import DPLL, UNKNOWN, UNSAT, Trace
from .lines import analyse
from .puzzle import EMPTY, FILLED, Puzzle, clue_text, clues_of, empty_grid
from .puzzle import UNKNOWN as CELL_UNKNOWN

METHODS = ("line", "hybrid", "sat")
MAX_EVENTS = 4000  # cap on the replay log sent to the page

# Budgets. Counted in work, not seconds, so results do not depend on the machine.
SAT_PROPAGATIONS_SERVER = 600_000  # about 6 s of DPLL on this machine, the most a request may spend
SAT_PROPAGATIONS_BENCH = 5_000_000
GUESSES_SERVER = 2_000  # about 6 s of hybrid search (roughly 3 ms per guess)


@dataclass
class Stats:
    method: str
    decisions: int = 0  # guesses (hybrid) or decided literals (sat)
    conflicts: int = 0
    propagations: int = 0  # cells fixed by line passes (line, hybrid) or implied literals (sat)
    backtracks: int = 0  # guesses undone (hybrid) or decision levels undone (sat)
    sweeps: int = 0  # row sweeps and column sweeps of the line solver
    seconds: float = 0.0
    variables: int | None = None
    clauses: int | None = None
    pure_eliminated: int | None = None

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "decisions": self.decisions,
            "conflicts": self.conflicts,
            "propagations": self.propagations,
            "backtracks": self.backtracks,
            "sweeps": self.sweeps,
            "seconds": round(self.seconds, 4),
            "variables": self.variables,
            "clauses": self.clauses,
            "pure_eliminated": self.pure_eliminated,
        }


@dataclass
class Result:
    method: str
    status: str  # "unique", "multiple", "contradiction", or "undecided"
    rows: int
    cols: int
    solution: list | None = None
    second: list | None = None
    partial: list | None = None  # line method: the grid after line passes
    rounds: list | None = None  # line method, traced: the cells each line pass fixed
    events: list | None = None  # hybrid and sat, traced: ["d", r, c, v], ["c"], ["b"]
    events_truncated: bool = False
    stats: Stats = field(default_factory=lambda: Stats("line"))

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "rows": self.rows,
            "cols": self.cols,
            "status": self.status,
            "solution": self.solution,
            "second": self.second,
            "partial": self.partial,
            "rounds": self.rounds,
            "events": self.events,
            "events_truncated": self.events_truncated,
            "stats": self.stats.as_dict(),
        }


class _Events:
    """Collects replay events up to MAX_EVENTS. Disabled events cost nothing."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.items: list = []
        self.truncated = False

    def add(self, event) -> None:
        if not self.enabled:
            return
        if len(self.items) < MAX_EVENTS:
            self.items.append(event)
        else:
            self.truncated = True


def verify(puzzle: Puzzle, grid) -> bool:
    """True when `grid` is a 0/1 picture whose clues are exactly the puzzle's clues."""
    if any(v not in (0, 1) for row in grid for v in row):
        return False
    rows, cols = clues_of(grid)
    return rows == puzzle.row_clues and cols == puzzle.col_clues


# -- line passes ---------------------------------------------------------------------------------


def _line_pass(puzzle: Puzzle, grid, axis: str, stats: Stats):
    """Analyse every row (axis "rows") or column (axis "cols") once, in order, writing forced cells back
    into the grid as it goes. Returns (lines that changed, consistent?)."""
    rows, cols = puzzle.rows, puzzle.cols
    count = rows if axis == "rows" else cols
    clues = puzzle.row_clues if axis == "rows" else puzzle.col_clues
    changed = []
    for idx in range(count):
        known = list(grid[idx]) if axis == "rows" else [grid[r][idx] for r in range(rows)]
        info = analyse(clues[idx], known)
        if not info.consistent:
            return changed, False
        fixed = []
        for i, k in enumerate(known):
            if k == CELL_UNKNOWN:
                v = info.fixed_value(i)
                if v != CELL_UNKNOWN:
                    r, c = (idx, i) if axis == "rows" else (i, idx)
                    grid[r][c] = v
                    fixed.append([r, c, v])
        if fixed:
            stats.propagations += len(fixed)
            name = f"r{idx + 1}" if axis == "rows" else f"c{idx + 1}"
            changed.append({"line": name, "fixed": fixed})
    return changed, True


def propagate(puzzle: Puzzle, grid, rounds: list | None, stats: Stats) -> bool:
    """Alternate row and column passes until a full round changes nothing. False on contradiction.

    `rounds`, when given, collects one entry per pass that fixed anything; the page replays them.
    """
    while True:
        progressed = False
        for axis in ("rows", "cols"):
            stats.sweeps += 1
            changed, ok = _line_pass(puzzle, grid, axis, stats)
            if not ok:
                return False
            if changed:
                progressed = True
                if rounds is not None:
                    rounds.append({"axis": axis, "pass": stats.sweeps, "lines": changed})
        if not progressed:
            return True


# -- methods --------------------------------------------------------------------------------------


def _solve_line(puzzle: Puzzle, trace: bool, stats: Stats) -> Result:
    grid = empty_grid(puzzle.rows, puzzle.cols)
    rounds = [] if trace else None
    if not propagate(puzzle, grid, rounds, stats):
        status = "contradiction"
    elif all(v != CELL_UNKNOWN for row in grid for v in row):
        status = "unique"  # every cell is forced, so this picture satisfies the clues and is the only one
    else:
        status = "undecided"  # line solving alone is stuck: a search is needed
    solved = status == "unique"
    return Result("line", status, puzzle.rows, puzzle.cols,
                  solution=[row[:] for row in grid] if solved else None,
                  partial=[row[:] for row in grid], rounds=rounds, stats=stats)


def _search_hybrid(puzzle: Puzzle, limit: int, max_guesses: int | None, events: _Events, stats: Stats):
    """Depth-first search with line propagation at every node. Returns (solutions, budget_hit)."""
    solutions: list = []

    class OutOfBudget(Exception):
        pass

    def dfs(grid) -> None:
        if len(solutions) >= limit:
            return
        if max_guesses is not None and stats.decisions > max_guesses:
            raise OutOfBudget
        if not propagate(puzzle, grid, None, stats):
            stats.conflicts += 1
            events.add(["c"])
            return
        # Most constrained line: the row or column with the fewest unknown cells that still has one.
        # Guessing there collapses the most arrangements per guess.
        best = None
        for r in range(puzzle.rows):
            n_unknown = sum(1 for v in grid[r] if v == CELL_UNKNOWN)
            if n_unknown and (best is None or n_unknown < best[0]):
                best = (n_unknown, "rows", r)
        for c in range(puzzle.cols):
            n_unknown = sum(1 for r in range(puzzle.rows) if grid[r][c] == CELL_UNKNOWN)
            if n_unknown and (best is None or n_unknown < best[0]):
                best = (n_unknown, "cols", c)
        if best is None:
            solutions.append([row[:] for row in grid])  # no unknown cell left: every line is consistent
            return
        _, axis, idx = best
        cells = [(idx, i) for i in range(puzzle.cols) if grid[idx][i] == CELL_UNKNOWN] if axis == "rows" \
            else [(i, idx) for i in range(puzzle.rows) if grid[i][idx] == CELL_UNKNOWN]
        r, c = cells[0]  # the first unknown cell of that line
        for v in (FILLED, EMPTY):
            child = [row[:] for row in grid]
            child[r][c] = v
            stats.decisions += 1
            events.add(["d", r, c, v])
            before = len(solutions)
            dfs(child)
            if len(solutions) >= limit:
                return
            if len(solutions) == before:  # this guess led nowhere
                stats.backtracks += 1
                events.add(["b"])

    try:
        dfs(empty_grid(puzzle.rows, puzzle.cols))
    except OutOfBudget:
        return solutions, True
    return solutions, False


def _solve_hybrid(puzzle: Puzzle, trace: bool, max_guesses: int | None, stats: Stats) -> Result:
    events = _Events(trace)
    solutions, budget_hit = _search_hybrid(puzzle, 2, max_guesses, events, stats)
    if budget_hit:
        status = "undecided"
    elif not solutions:
        status = "contradiction"
    elif len(solutions) == 1:
        status = "unique"
    else:
        status = "multiple"
    return Result("hybrid", status, puzzle.rows, puzzle.cols,
                  solution=solutions[0] if solutions else None,
                  second=solutions[1] if len(solutions) > 1 else None,
                  events=events.items if trace else None, events_truncated=events.truncated, stats=stats)


def _solve_sat(puzzle: Puzzle, trace: bool, max_propagations: int | None, stats: Stats) -> Result:
    cnf = encode(puzzle)
    stats.variables = cnf.nvars
    stats.clauses = len(cnf.clauses)
    cells = cnf.cell_vars()
    events = _Events(trace)
    cols = puzzle.cols
    cell_count = puzzle.rows * cols

    def sink(event) -> None:
        # Decisions on auxiliary variables are counted in stats but not replayed; only cells are drawn.
        if event[0] == "d":
            var = event[1]
            if var <= cell_count:
                events.add(["d", (var - 1) // cols, (var - 1) % cols, FILLED if event[2] else EMPTY])
        else:
            events.add([event[0]])

    trace_sink = Trace(sink=sink)

    def read(solver: DPLL):
        model = solver.model()
        return [[1 if model[cell] else 0 for cell in row] for row in cnf.cell_var]

    def absorb(solver: DPLL) -> None:
        s = solver.stats
        stats.decisions += s.decisions
        stats.conflicts += s.conflicts
        stats.propagations += s.propagations
        stats.backtracks += s.backtracks
        if stats.pure_eliminated is None:
            stats.pure_eliminated = s.pure_eliminated

    first = DPLL(cnf.nvars, cnf.clauses, priority=cells, trace=trace_sink)
    status = first.solve(max_propagations)
    absorb(first)
    if status == UNKNOWN:
        return Result("sat", "undecided", puzzle.rows, puzzle.cols, events=events.items if trace else None,
                      events_truncated=events.truncated, stats=stats)
    if status == UNSAT:
        return Result("sat", "contradiction", puzzle.rows, puzzle.cols, events=events.items if trace else None,
                      events_truncated=events.truncated, stats=stats)
    solution = read(first)

    # Uniqueness: forbid this picture on the cell variables and ask again. UNSAT proves uniqueness.
    # The clause is false exactly on this picture: a filled cell must be empty, or an empty one filled.
    blocking = []
    for r in range(puzzle.rows):
        for c in range(cols):
            var = cnf.cell_var[r][c]
            blocking.append(-var if solution[r][c] else var)
    second_solver = DPLL(cnf.nvars, cnf.clauses + [blocking], priority=cells, trace=trace_sink)
    status2 = second_solver.solve(max_propagations)
    absorb(second_solver)
    if status2 == UNKNOWN:
        return Result("sat", "undecided", puzzle.rows, puzzle.cols, solution=solution,
                      events=events.items if trace else None, events_truncated=events.truncated, stats=stats)
    if status2 == UNSAT:
        return Result("sat", "unique", puzzle.rows, puzzle.cols, solution=solution,
                      events=events.items if trace else None, events_truncated=events.truncated, stats=stats)
    return Result("sat", "multiple", puzzle.rows, puzzle.cols, solution=solution, second=read(second_solver),
                  events=events.items if trace else None, events_truncated=events.truncated, stats=stats)


def solve(puzzle: Puzzle, method: str = "hybrid", *, trace: bool = False, max_guesses: int | None = None,
          max_propagations: int | None = None) -> Result:
    """Solve with one method. `trace` records the replay (rounds for line, events for hybrid and sat).

    Every solution is verified against the clues; a failed check raises, because it means a bug.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    stats = Stats(method)
    start = time.perf_counter()
    if method == "line":
        result = _solve_line(puzzle, trace, stats)
    elif method == "hybrid":
        result = _solve_hybrid(puzzle, trace, max_guesses, stats)
    else:
        result = _solve_sat(puzzle, trace, max_propagations, stats)
    stats.seconds = time.perf_counter() - start
    for grid in (result.solution, result.second):
        if grid is not None and not verify(puzzle, grid):
            raise RuntimeError(f"{method} returned a picture that does not satisfy the clues")
    return result


def count_solutions(puzzle: Puzzle, limit: int = 2, max_guesses: int | None = None):
    """Count solutions up to `limit` with the hybrid search. Returns (status, solutions found, guesses)."""
    stats = Stats("hybrid")
    solutions, budget_hit = _search_hybrid(puzzle, limit, max_guesses, _Events(False), stats)
    status = "undecided" if budget_hit and len(solutions) < limit else (
        "contradiction" if not solutions else "unique" if len(solutions) == 1 else "multiple")
    return status, solutions, stats.decisions


# -- hints ---------------------------------------------------------------------------------------


def hint(puzzle: Puzzle, grid) -> dict:
    """Find one cell that line solving forces, given the marks the player has made, and say why.

    Lines are checked rows first, then columns. A line that cannot fit the marks is reported as a
    contradiction, naming the line, so the player can look for the wrong mark there.
    """
    rows, cols = puzzle.rows, puzzle.cols
    infos = []
    for axis in ("rows", "cols"):
        count = rows if axis == "rows" else cols
        clues = puzzle.row_clues if axis == "rows" else puzzle.col_clues
        for idx in range(count):
            known = list(grid[idx]) if axis == "rows" else [grid[r][idx] for r in range(rows)]
            infos.append((axis, idx, known, clues[idx], analyse(clues[idx], known)))
    for axis, idx, _, clue, info in infos:
        if not info.consistent:
            name = _line_name(axis, idx)
            return _no_hint(contradiction=True,
                            message=f"{name} (clue {clue_text(clue)}) cannot fit the marks made so far")
    if all(v != CELL_UNKNOWN for row in grid for v in row):
        return _no_hint(message="every cell is marked and the marks satisfy the clues")
    for axis, idx, known, clue, info in infos:
        for i, k in enumerate(known):
            if k != CELL_UNKNOWN:
                continue
            v = info.fixed_value(i)
            if v == CELL_UNKNOWN:
                continue
            r, c = (idx, i) if axis == "rows" else (i, idx)
            reason, span = _reason(axis, idx, i, v, clue, info)
            state = "filled" if v == FILLED else "empty"
            return {
                "contradiction": False,
                "found": True,
                "line": _line_name(axis, idx),
                "row": r,
                "col": c,
                "value": v,
                "reason": reason,
                "span": list(span) if span else None,
                "message": f"row {r + 1}, column {c + 1} must be {state}",
            }
    return _no_hint(message="no single line forces a cell right now: try SOLVE, or guess and check")


def _line_name(axis: str, idx: int) -> str:
    return f"row {idx + 1}" if axis == "rows" else f"column {idx + 1}"


def _no_hint(contradiction: bool = False, message: str = "") -> dict:
    return {"contradiction": contradiction, "found": False, "line": None, "row": None, "col": None,
            "value": None, "reason": message, "span": None, "message": message}


def _reason(axis: str, idx: int, i: int, v: int, clue, info):
    """A sentence explaining why cell i of this line is forced to v, and the span used (or None)."""
    name = _line_name(axis, idx)
    along = "columns" if axis == "rows" else "rows"
    if v == FILLED:
        # Prefer the run whose overlap covers the cell: it is the most direct explanation.
        best = None
        for k, length in enumerate(clue):
            span = info.overlap(k, length)
            if span and span[0] <= i <= span[1] and (best is None or length > best[0]):
                best = (length, span)
        if best is not None:
            length, (a, b) = best
            return (f"{name} has clue {clue_text(clue)}. Its run of {length} covers {along} {a + 1} to {b + 1} "
                    "in every arrangement, so this cell is filled."), (a, b)
        return f"every arrangement of {name} (clue {clue_text(clue)}) fills this cell.", None
    return f"no arrangement of {name} (clue {clue_text(clue)}) puts a run on this cell, so it is empty.", None
