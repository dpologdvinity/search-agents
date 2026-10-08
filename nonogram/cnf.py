"""Encode a whole nonogram as a CNF formula for the DPLL solver.

Variables
  Cell variables: x(r, c) is true when cell (r, c) is filled. They are numbered 1..R*C in row-major order.
  Automaton variables: for each line, one variable per (position i, automaton state q). s[i][q] is true
  when the line's clue automaton is in state q after reading the first i cells of that line. Only states
  that are both reachable from the start and able to reach acceptance are encoded, which keeps the
  formula small. These are the "sequential/automaton" encoding: the clue is a regular language, so the
  line's state sequence is a witness that the clue holds.
  Transition variables: y[i][q][v] is true when the line is in state q at position i, cell i has value v,
  and the automaton moves to a state that is still encoded. They tie consecutive states together.

Clauses, for a line with cells x_0..x_{n-1} and automaton states q, with delta the transition function:
  start      s[0][start] is true, and the other states at position 0 are false.
  transition y <-> (s[i][q] and [x_i = v]) for each live transition; s[i+1][t] <-> OR of the y's into t.
             The two directions matter: the forward half propagates the state, and the backward half stops
             a state from being true without a path that reaches it, so the final state is the real one.
  dead end   where the transition is dead, (not s[i][q]) or (not [x_i = v]) is forced.
  accept     at position n, at least one accepting state is true.
Because the automaton is deterministic, exactly one state is true at each position once the cells are
fixed, so satisfying assignments are exactly the pictures whose lines all fit their clues.

The uniqueness check in solvers.py adds one blocking clause on the cell variables and asks again.
"""

from __future__ import annotations

from dataclasses import dataclass

from .automaton import DEAD, build, reach_sets
from .puzzle import Puzzle


@dataclass
class CNF:
    nvars: int
    clauses: list[list[int]]
    cell_var: list[list[int]]  # cell_var[r][c] is the DIMACS variable of cell (r, c)

    def cell_vars(self) -> list[int]:
        return [v for row in self.cell_var for v in row]


def encode(puzzle: Puzzle) -> CNF:
    """The CNF whose models are exactly the solutions of `puzzle`."""
    rows, cols = puzzle.rows, puzzle.cols
    cell_var = [[r * cols + c + 1 for c in range(cols)] for r in range(rows)]
    counter = [rows * cols]
    clauses: list[list[int]] = []

    def new_var() -> int:
        counter[0] += 1
        return counter[0]

    lines = []
    for r in range(rows):
        lines.append((puzzle.row_clues[r], [cell_var[r][c] for c in range(cols)]))
    for c in range(cols):
        lines.append((puzzle.col_clues[c], [cell_var[r][c] for r in range(rows)]))
    for clue, xs in lines:
        _encode_line(clue, xs, clauses, new_var)
    return CNF(counter[0], clauses, cell_var)


def _encode_line(clue, xs, clauses, new_var) -> None:
    """Append the clauses that force the cell variables `xs` to fit `clue`."""
    aut = build(tuple(clue))
    n = len(xs)
    forward, backward = reach_sets(aut, n)
    # Live states at position i: reachable from the start and able to finish. Others are never true.
    live = [forward[i] & backward[i] for i in range(n + 1)]
    if aut.start not in live[0]:  # no accepted line exists for this clue at this length
        clauses.append([])  # the empty clause makes the whole formula unsatisfiable
        return

    svar = [{q: new_var() for q in sorted(live[i])} for i in range(n + 1)]
    clauses.append([svar[0][aut.start]])
    for q in svar[0]:
        if q != aut.start:
            clauses.append([-svar[0][q]])  # no other state can be the start

    for i in range(n):
        x = xs[i]
        incoming: dict[int, list[int]] = {t: [] for t in svar[i + 1]}
        for q, s in svar[i].items():
            for v in (0, 1):
                t = aut.delta[q][v]
                # "x_i = v" as a literal: x itself when v is 1, -x when v is 0.
                cell_is_v = x if v == 1 else -x
                if t == DEAD or t not in svar[i + 1]:
                    clauses.append([-s, -cell_is_v])  # from state q, cell value v leads nowhere: forbid it
                    continue
                y = new_var()
                # y <-> (s and cell_is_v)
                clauses.append([-y, s])
                clauses.append([-y, cell_is_v])
                clauses.append([y, -s, -cell_is_v])
                # y forces the successor state.
                clauses.append([-y, svar[i + 1][t]])
                incoming[t].append(y)
        # The successor state is true only if some transition into it is taken.
        for t, ys in incoming.items():
            clauses.append([-svar[i + 1][t]] + ys)

    accepting = [svar[n][q] for q in svar[n] if q in aut.accept]
    clauses.append(accepting)  # an empty list here means the line can never be accepted
