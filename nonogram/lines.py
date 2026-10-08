"""Line solving: what one row or column forces, given its clue and the cells already known.

For a line of n cells, the clue's automaton (automaton.py) and two position sweeps give the exact
answer to "which values can cell i take in some arrangement that fits the clue and the known cells?"
without listing the arrangements, of which there can be exponentially many:

  forward[i]   states reachable after the first i cells (a prefix that fits the clue so far)
  backward[i]  states from which the rest of the line can still be completed to fit the clue

Cell i can be value v when some state s in forward[i] has a v-transition to a state t in backward[i+1].
That joins a fitting prefix, cell i = v, and a fitting suffix, which is exactly a full arrangement.
Taking the union over all such arrangements gives each cell's possible values; a cell with one
possible value is forced. The sweeps cost O(n * states), a few hundred operations for a 20-cell line.

Each run also gets its valid start positions, the starts that occur in some arrangement. Their overlap
(from the latest start to the earliest start plus length minus one) is covered by that run in every
arrangement, which is the explanation the hint gives.
"""

from __future__ import annotations

from dataclasses import dataclass

from .automaton import DEAD, build, reach_sets

# Bit flags for the values a cell can still take. A cell with mask EMPTY_OK | FILLED_OK is unknown.
EMPTY_OK = 1
FILLED_OK = 2
UNKNOWN = -1


@dataclass(frozen=True)
class LineInfo:
    """The result of analysing one line.

    possible[i]  bitmask of values cell i can take (EMPTY_OK, FILLED_OK, or both).
    consistent   False when no arrangement fits the clue and the known cells (the grid is contradictory).
    starts[k]    valid start positions of run k over all arrangements that fit.
    """

    possible: tuple[int, ...]
    consistent: bool
    starts: tuple[tuple[int, ...], ...]

    def fixed_value(self, i: int) -> int:
        """0 or 1 when cell i is forced, UNKNOWN when both values are possible."""
        mask = self.possible[i]
        if mask == FILLED_OK:
            return 1
        if mask == EMPTY_OK:
            return 0
        return UNKNOWN

    def overlap(self, k: int, length: int) -> tuple[int, int] | None:
        """Cells that run k (of this length) covers in every arrangement, as (first, last), or None."""
        starts = self.starts[k]
        if not starts:
            return None
        first, last = max(starts), min(starts) + length - 1
        return (first, last) if first <= last else None


def analyse(runs, known) -> LineInfo:
    """Analyse a line. `runs` is the clue, `known` is the line's cells (-1 unknown, 0 empty, 1 filled)."""
    runs = tuple(runs)
    n = len(known)
    aut = build(runs)
    forward, backward = reach_sets(aut, n, known)

    possible = []
    for i in range(n):
        mask = 0
        for s in forward[i]:
            for v in (0, 1):
                if known[i] != UNKNOWN and known[i] != v:
                    continue  # the symbol contradicts a known cell
                t = aut.delta[s][v]
                # Keep only transitions that can still be completed to an accepted line.
                if t != DEAD and t in backward[i + 1]:
                    mask |= 1 << v
        possible.append(mask)

    consistent = aut.start in backward[0]

    # Valid starts: run k opens at cell s when the automaton can be in A[k] before s, reads a 1 at s
    # (which must be allowed by the known cell), and the state after it can still be completed.
    starts = []
    for k in range(len(runs)):
        opened = aut.run_start[k]
        after = aut.delta[opened][1]
        valid = []
        if after != DEAD:
            for s in range(n):
                if known[s] == 0 or opened not in forward[s]:
                    continue
                if after in backward[s + 1]:
                    valid.append(s)
        starts.append(tuple(valid))
    return LineInfo(tuple(possible), consistent, tuple(starts))


def solve_line(runs, known) -> list[int] | None:
    """The line with every forced cell filled in (0 or 1) and the rest left at -1.

    Returns None when the line is contradictory. One call is one "line pass" in the solver.
    """
    info = analyse(runs, known)
    if not info.consistent:
        return None
    return [info.fixed_value(i) for i in range(len(known))]
