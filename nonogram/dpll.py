"""A DPLL SAT solver: unit propagation with watched literals, pure-literal elimination, VSIDS-lite branching,
phase saving, and chronological backtracking. No clause learning (this is DPLL, not CDCL).

Input is a CNF in DIMACS style: a clause is a list of non-zero ints, where +v is variable v and -v is its
negation. Internally a literal is an int: 2*v for +v and 2*v+1 for -v, so the negation of a literal is
lit ^ 1, and value[lit] is 1 (true), -1 (false) or 0 (unassigned).

How the search works:
  1. Pure literals (a variable that appears with one sign only) are set to that sign and every clause they
     satisfy is removed. This is sound for satisfiability and runs once, before the search.
  2. Unit propagation. Each clause of two or more literals watches two of them. When a watched literal
     becomes false, the solver looks for another non-false literal to watch. If there is none, the clause
     is unit (its other watch is forced) or empty (a conflict).
  3. Branching. VSIDS-lite: every variable has an activity that rises when it appears in a conflict
     clause, and activities decay by 0.95 per conflict, so recent conflicts weigh more. The unassigned
     variable with the highest activity is decided next, with the last polarity it had (phase saving).
     Cell variables start with activity 1.0 so the search decides the picture before the auxiliaries.
  4. Conflict. The most recent decision that has not been flipped yet is undone and its negation is
     forced. If both values of a decision have failed, the solver backtracks further. That is chronological
     backtracking, so no learned clause is kept.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

SAT = "SAT"
UNSAT = "UNSAT"
UNKNOWN = "UNKNOWN"  # the propagation budget or the deadline ran out before an answer

DECAY = 0.95  # activity decay per conflict, as in MiniSat's VSIDS
PRIOR = 1.0  # initial activity given to the priority (cell) variables


def _lit(d: int) -> int:
    """DIMACS integer to internal literal."""
    return 2 * d if d > 0 else -2 * d + 1


@dataclass
class Stats:
    decisions: int = 0
    conflicts: int = 0
    propagations: int = 0  # literals forced by unit propagation (not decided)
    backtracks: int = 0  # decision levels undone
    pure_eliminated: int = 0


@dataclass
class Trace:
    """Optional callback sink. Receives ('d', var, positive) for decisions, ('c',) for conflicts,
    and ('b',) for backtracks."""

    sink: object = None
    events: list = field(default_factory=list)


def pure_literal_eliminate(clauses: list[list[int]], nvars: int):
    """Repeatedly set pure literals and drop the clauses they satisfy.

    Returns (remaining clauses, {var: sign}, number of variables fixed). Setting a pure literal never
    makes the formula unsatisfiable, so the solver can ignore those variables afterwards.
    """
    fixed: dict[int, bool] = {}
    while True:
        pos = [False] * (nvars + 1)
        neg = [False] * (nvars + 1)
        for cl in clauses:
            for d in cl:
                (pos if d > 0 else neg)[abs(d)] = True
        # A variable seen with exactly one sign is pure: that sign satisfies all of its clauses.
        pure = {v: pos[v] for v in range(1, nvars + 1) if pos[v] != neg[v]}
        if not pure:
            return clauses, fixed
        fixed.update(pure)
        # Every clause that mentions a pure variable contains its pure literal, so all of them are satisfied.
        clauses = [cl for cl in clauses if not any(abs(d) in pure for d in cl)]


class DPLL:
    """One satisfiability problem. Build it, call solve(), then read model() or stats."""

    def __init__(self, nvars: int, clauses, priority=(), eliminate_pure: bool = True, trace: Trace | None = None):
        self.nvars = nvars
        self.stats = Stats()
        self.trace = trace
        self.ok = True
        self.timed_out = False  # set when solve() stops at its deadline rather than its budget
        self.heap_rebuilds = 0
        if eliminate_pure:
            clauses, fixed = pure_literal_eliminate([list(c) for c in clauses], nvars)
            self.stats.pure_eliminated = len(fixed)
        else:
            fixed = {}
        size = 2 * (nvars + 1)
        self.value = [0] * size
        self.watches: list[list[int]] = [[] for _ in range(size)]
        self.clauses: list[list[int]] = []
        self.trail: list[int] = []
        self.qhead = 0
        self.decisions: list[tuple[int, int, bool]] = []  # (trail length before, decided literal, flipped?)
        self.activity = [0.0] * (nvars + 1)
        self.inc = 1.0
        self.phase = [2 * v + 1 for v in range(nvars + 1)]  # default polarity: false
        for v in priority:
            self.activity[v] = PRIOR
        self.heap = [(-self.activity[v], v) for v in range(1, nvars + 1)]
        heapq.heapify(self.heap)

        # Pure literals are fixed at the root, before any clause is watched.
        for v, sign in fixed.items():
            self._assign(2 * v if sign else 2 * v + 1)
        for cl in clauses:
            lits = list({_lit(d) for d in cl})
            if not lits:
                self.ok = False  # an empty clause: no assignment can satisfy it
                return
            if len(lits) == 1:
                if self.value[lits[0]] == -1:
                    self.ok = False
                    return
                if self.value[lits[0]] == 0:
                    self._assign(lits[0])
                continue
            # Tautologies (x or not x) are always true and do not need watching.
            if any(lit ^ 1 in lits for lit in lits):
                continue
            ci = len(self.clauses)
            self.clauses.append(lits)
            self.watches[lits[0]].append(ci)
            self.watches[lits[1]].append(ci)

    # -- assignment -------------------------------------------------------------------------------

    def _assign(self, lit: int) -> None:
        self.value[lit] = 1
        self.value[lit ^ 1] = -1
        self.trail.append(lit)

    def _propagate(self):
        """Unit propagation over the trail. Returns a conflicting clause index, or None."""
        value, watches, clauses, trail = self.value, self.watches, self.clauses, self.trail
        while self.qhead < len(trail):
            lit = trail[self.qhead]
            self.qhead += 1
            false_lit = lit ^ 1  # the literal that just became false; its watchers must move
            ws = watches[false_lit]
            i = 0
            while i < len(ws):
                ci = ws[i]
                cl = clauses[ci]
                if cl[0] == false_lit:  # keep the false literal in slot 1
                    cl[0], cl[1] = cl[1], cl[0]
                first = cl[0]
                if value[first] == 1:  # the clause is satisfied by its other watch
                    i += 1
                    continue
                for k in range(2, len(cl)):
                    lk = cl[k]
                    if value[lk] != -1:  # found a literal that is not false: watch it instead
                        cl[1], cl[k] = lk, cl[1]
                        watches[lk].append(ci)
                        ws[i] = ws[-1]  # drop ci from this watch list (swap-remove, do not advance i)
                        ws.pop()
                        break
                else:
                    if value[first] == -1:
                        return ci  # every literal is false: conflict
                    self._assign(first)  # unit: the remaining watch is forced
                    self.stats.propagations += 1
                    i += 1
        return None

    def _undo(self, to: int) -> None:
        """Unassign every literal above trail position `to`, saving phases and re-queueing variables."""
        trail, value = self.trail, self.value
        while len(trail) > to:
            lit = trail.pop()
            value[lit] = 0
            value[lit ^ 1] = 0
            v = lit >> 1
            self.phase[v] = lit  # phase saving: the next decision on v reuses this polarity
            heapq.heappush(self.heap, (-self.activity[v], v))
        self.qhead = to
        if len(self.heap) > 4 * self.nvars + 1024:  # stale entries pile up; rebuild from live variables
            self.heap = [(-self.activity[v], v) for v in range(1, self.nvars + 1) if value[2 * v] == 0]
            heapq.heapify(self.heap)
            self.heap_rebuilds += 1

    def _bump(self, ci: int) -> None:
        """Raise the activity of every variable in conflict clause ci, then decay all activities."""
        for lit in self.clauses[ci]:
            v = lit >> 1
            self.activity[v] += self.inc
            if self.value[2 * v] == 0:
                heapq.heappush(self.heap, (-self.activity[v], v))
        self.inc /= DECAY
        if self.inc > 1e100:  # rescale to avoid float overflow; the order of activities is unchanged
            self.activity = [a * 1e-100 for a in self.activity]
            self.inc *= 1e-100
            self.heap = [(-self.activity[v], v) for v in range(1, self.nvars + 1) if self.value[2 * v] == 0]
            heapq.heapify(self.heap)

    def _pick(self) -> int:
        """The unassigned variable with the highest activity, or 0 when every variable is assigned."""
        heap = self.heap
        while heap:
            _, v = heapq.heappop(heap)
            if self.value[2 * v] == 0:  # entries for assigned variables are stale and skipped
                return v
        return 0

    # -- search -----------------------------------------------------------------------------------

    def solve(self, max_propagations: int | None = None, deadline: float | None = None) -> str:
        """Return SAT, UNSAT, or UNKNOWN when the propagation budget runs out or `deadline` passes.

        `deadline` is a time.perf_counter() value. It is checked once per loop turn, which is one
        propagation round plus one decision or conflict, so the overshoot is a single round.
        """
        if not self.ok:
            return UNSAT
        if self._propagate() is not None:
            return UNSAT
        while True:
            if deadline is not None and time.perf_counter() > deadline:
                self.timed_out = True
                return UNKNOWN
            conflict = self._propagate()
            if conflict is not None:
                self.stats.conflicts += 1
                self._emit(("c",))
                self._bump(conflict)
                # Undo flipped decisions first: both of their values have failed.
                while self.decisions and self.decisions[-1][2]:
                    start, _, _ = self.decisions.pop()
                    self._undo(start)
                    self.stats.backtracks += 1
                    self._emit(("b",))
                if not self.decisions:
                    return UNSAT
                start, lit, _ = self.decisions[-1]
                self._undo(start)
                self.stats.backtracks += 1
                self._emit(("b",))
                self.decisions[-1] = (start, lit ^ 1, True)
                self._assign(lit ^ 1)  # the forced negation of the failed decision
                continue
            if max_propagations is not None and self.stats.propagations > max_propagations:
                return UNKNOWN
            v = self._pick()
            if v == 0:
                return SAT
            lit = self.phase[v]
            self.decisions.append((len(self.trail), lit, False))
            self.stats.decisions += 1
            self._emit(("d", v, (lit & 1) == 0))
            self._assign(lit)

    def _emit(self, event) -> None:
        if self.trace is not None and self.trace.sink is not None:
            self.trace.sink(event)

    def model(self) -> list[bool]:
        """After SAT: the truth value of each variable (index 0 unused). Unassigned variables read False."""
        return [False] + [self.value[2 * v] == 1 for v in range(1, self.nvars + 1)]


def check_model(clauses, model) -> bool:
    """True when every clause has a true literal under `model` (as returned by DPLL.model())."""
    return all(any((model[abs(d)] if d > 0 else not model[abs(d)]) for d in cl) for cl in clauses)
