"""A nonogram clue as a finite automaton over the cell values 0 (empty) and 1 (filled).

A clue such as (3, 1) says: a run of three filled cells, then at least one empty cell, then a single
filled cell, with empty cells everywhere else. The set of lines that satisfy the clue is a regular
language, and a small deterministic automaton recognises it. Both the line solver (lines.py) and the
SAT encoding (cnf.py) read the same automaton, so the two methods agree on what a clue means.

States:
  A[k]      k runs are finished and no run is open. Reached at the start (k = 0) and after an empty cell.
            Reading 1 here opens run k (0-based), so it leads to B[k, 1] or D[k+1].
  B[k, j]   inside run k, with j of its cells filled so far (1 <= j < length of run k). Reading 0 is
            a dead end: runs cannot be cut short.
  D[k+1]    run k has just been completed, so the next cell must be empty. Reading 1 is a dead end,
            which is what stops two runs from touching.
A line is accepted when it ends in A[m] or D[m], where m is the number of runs. A[m] also covers the
case of an empty clue (m = 0), which accepts only all-empty lines.

The automaton is deterministic: each state has at most one successor per symbol, so a word has at
most one run through it. The state count is sum(runs) + m + 1, so it stays small (at most about 2n).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

DEAD = -1  # successor used when a symbol has no transition


@dataclass(frozen=True)
class Automaton:
    """States are integers 0..size-1. delta[s][v] is the successor of state s on symbol v, or DEAD."""

    runs: tuple[int, ...]
    size: int
    start: int
    accept: frozenset[int]
    delta: tuple[tuple[int, int], ...]
    # run_start[k] is the state A[k]: k runs are done and run k is about to open. Used to find where
    # run k can begin (see lines.py).
    run_start: tuple[int, ...]
    names: tuple[str, ...]  # readable state names, e.g. "A0", "B1,2", "D1", for debugging and tests

    def step(self, state: int, value: int) -> int:
        """Successor of `state` on the symbol `value` (0 or 1)."""
        return self.delta[state][value]

    def accepts(self, cells) -> bool:
        """True when the 0/1 sequence `cells` is a line that satisfies the clue."""
        s = self.start
        for v in cells:
            s = self.delta[s][v]
            if s == DEAD:
                return False
        return s in self.accept


@lru_cache(maxsize=4096)
def build(runs: tuple[int, ...]) -> Automaton:
    """Build the automaton for a clue. Cached, because a puzzle reuses the same clues many times."""
    if any(length < 1 for length in runs):
        raise ValueError(f"run lengths must be at least 1, got {runs}")
    m = len(runs)
    names: list[str] = []

    def new(name: str) -> int:
        names.append(name)
        return len(names) - 1

    # Allocate ids first so every transition can refer to any state.
    a_state = [new(f"A{k}") for k in range(m + 1)]
    d_state = [None] + [new(f"D{k}") for k in range(1, m + 1)]  # d_state[k]: k runs done, last cell filled
    b_state: dict[tuple[int, int], int] = {}
    for k, length in enumerate(runs):
        for j in range(1, length):
            b_state[(k, j)] = new(f"B{k},{j}")

    delta: list[list[int]] = [[DEAD, DEAD] for _ in names]
    for k in range(m + 1):
        # A[k] on empty stays in A[k]. On filled it opens run k, if there is one to open.
        delta[a_state[k]][0] = a_state[k]
        if k < m:
            delta[a_state[k]][1] = b_state[(k, 1)] if runs[k] > 1 else d_state[k + 1]
    for (k, j), s in b_state.items():
        # Inside run k: a filled cell extends it. Completing the run moves to D[k+1].
        delta[s][1] = d_state[k + 1] if j + 1 == runs[k] else b_state[(k, j + 1)]
    for k in range(1, m + 1):
        # After a completed run an empty cell is required: back to A[k]; a filled cell is a dead end.
        delta[d_state[k]][0] = a_state[k]

    accept = frozenset({a_state[m]} | ({d_state[m]} if m else set()))
    return Automaton(
        runs=tuple(runs),
        size=len(names),
        start=a_state[0],
        accept=accept,
        delta=tuple((row[0], row[1]) for row in delta),
        run_start=tuple(a_state),
        names=tuple(names),
    )


def reach_sets(aut: Automaton, length: int, known=None):
    """Forward and backward reachability over positions 0..length.

    forward[i] is the set of states the automaton can be in after reading the first i cells, using
    only the symbols allowed by `known` (a list of -1 unknown, 0 or 1 fixed). backward[i] is the set
    of states from which some suffix starting at cell i is accepted. Both sets are computed by one
    sweep each, so the cost is O(length * states).
    """
    if known is None:
        known = [-1] * length
    forward = [set() for _ in range(length + 1)]
    forward[0] = {aut.start}
    for i in range(length):
        allowed = (0, 1) if known[i] == -1 else (known[i],)
        nxt = forward[i + 1]
        for s in forward[i]:
            for v in allowed:
                t = aut.delta[s][v]
                if t != DEAD:
                    nxt.add(t)

    backward = [set() for _ in range(length + 1)]
    backward[length] = set(aut.accept)
    for i in range(length - 1, -1, -1):
        allowed = (0, 1) if known[i] == -1 else (known[i],)
        cur = backward[i]
        for s in range(aut.size):
            # s is useful at position i when some allowed symbol leads to a state that is useful at i+1.
            for v in allowed:
                t = aut.delta[s][v]
                if t != DEAD and t in backward[i + 1]:
                    cur.add(s)
                    break
    return forward, backward
