"""What the agent can prove from the revealed numbers, and how likely each covered cell is to be a mine.

Three layers, each one stronger than the last:

1. Single-cell rules. A revealed number n with k covered neighbours and m known mines among them
   needs n - m more mines. If that is 0, every covered neighbour is safe. If it equals the number of
   covered neighbours, every one is a mine. Each new fact can trigger more rules, so the pass repeats
   until nothing changes.

2. Constraint components. Each revealed number is a constraint "exactly `need` of these covered cells
   are mines". Constraints that share a covered cell form a component, and components are independent
   once the total mine count is fixed. For each component we count the layouts that satisfy all its
   constraints, grouped by how many mines they use (a backtracking search with memoization, below).
   The rest of the board (covered cells touching no number, the "interior") holds whatever mines are
   left over, and the number of ways to place k mines among n cells is C(n, k). Summing over the
   split of mines between components and the interior gives the exact weight of every full layout
   that fits the board. A cell is certainly safe if no weighted layout puts a mine on it, and
   certainly a mine if every weighted layout does.

3. Exact probabilities. The weight of the layouts where a cell is a mine, divided by the total
   weight, is its exact mine probability. Every weight is an integer, so probabilities are compared
   exactly (as numerator over the shared total) and only rounded for display.

Nothing here is sampled or approximated. A layout counted here is one that agrees with every number on
the board and with the mine count, so the deductions are proofs, not guesses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .board import UNKNOWN, View, label, neighbour_table

LEVEL_RULES = 1
LEVEL_CSP = 2
LEVEL_PROBABILITY = 3

# Cache of component results. Components that do not change between two moves are not recounted.
_CACHE: dict[tuple, ComponentStats] = {}
_CACHE_LIMIT = 4096


@dataclass(frozen=True)
class Constraint:
    """`need` mines lie among `cells` (covered neighbours of the revealed number at `source`)."""

    cells: tuple[int, ...]
    need: int
    source: int


@dataclass(frozen=True)
class Component:
    """One independent piece of the frontier: its covered cells, how many constraints tie them, and how
    many layouts fit them (ignoring the global mine count)."""

    cells: tuple[int, ...]
    constraints: int
    layouts: int


@dataclass
class Analysis:
    """What the agent knows about one view.

    `safe` and `mines` are proven facts, each with a short reason in `why`. At level 3 (and level 2)
    `numerator` holds, for every covered cell that is not a proven mine, the weight of layouts in
    which it is a mine, and `total` is the weight of all layouts. Probability = numerator / total.
    """

    view: View
    level: int
    safe: tuple[int, ...]
    mines: tuple[int, ...]
    why: dict[int, str]
    covered: tuple[int, ...]  # covered cells that are not proven mines (safe ones included)
    remaining: int  # mines not yet proven
    interior: tuple[int, ...] = ()
    components: tuple[Component, ...] = ()
    numerator: dict[int, int] = field(default_factory=dict)
    total: int = 0

    def probability(self, cell: int) -> float | None:
        """Exact mine probability of a covered cell, or None when this level did not compute it."""
        if not self.total or cell not in self.numerator:
            return None
        return self.numerator[cell] / self.total

    def unknown_neighbours(self, cell: int) -> int:
        """Covered neighbours of `cell` (a rough measure of how much an opening could reveal)."""
        table = neighbour_table(self.view.rows, self.view.cols)
        return sum(1 for n in table[cell] if self.view.cells[n] == UNKNOWN)


def _choose(n: int, k: int) -> int:
    """C(n, k), with 0 for impossible counts. Used to place leftover mines among the interior."""
    return math.comb(n, k) if 0 <= k <= n else 0


def _convolve(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Combine two layout-count tables: counts[k] = layouts with k mines, for independent pieces."""
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def build_constraints(view: View, known_mine: set[int], known_safe: set[int]) -> list[Constraint]:
    """One constraint per revealed number that still touches covered cells.

    Known mines are subtracted from the number, and known-safe cells are left out, since they are
    not candidates for a mine. A revealed zero has no covered neighbours, so it adds nothing.
    """
    table = neighbour_table(view.rows, view.cols)
    out = []
    for i, v in enumerate(view.cells):
        if v <= 0:
            continue
        hits = 0
        open_cells = []
        for n in table[i]:
            if n in known_mine:
                hits += 1
            elif view.cells[n] == UNKNOWN and n not in known_safe:
                open_cells.append(n)
        if open_cells:
            out.append(Constraint(tuple(open_cells), v - hits, i))
    return out


def rule_pass(view: View, known_mine: set[int], known_safe: set[int], why: dict[int, str]) -> None:
    """Level 1: apply the single-cell rules until no new fact appears. Facts are added to the two sets."""
    table = neighbour_table(view.rows, view.cols)
    cols = view.cols
    changed = True
    while changed:
        changed = False
        for i, v in enumerate(view.cells):
            if v <= 0:
                continue
            covered = [n for n in table[i] if view.cells[n] == UNKNOWN]
            open_cells = [n for n in covered if n not in known_mine and n not in known_safe]
            if not open_cells:
                continue
            hits = sum(1 for n in covered if n in known_mine)
            need = v - hits
            where = label(cols, i)
            if need == 0:  # all the mines this number wants are already known: the rest are safe
                for n in open_cells:
                    known_safe.add(n)
                    why.setdefault(n, f"{where} shows {v}, and its mines are already placed")
                changed = True
            elif need == len(open_cells):  # every covered neighbour has to be a mine
                for n in open_cells:
                    known_mine.add(n)
                    why.setdefault(n, f"{where} shows {v} with {need} covered neighbour(s): all are mines")
                changed = True


@dataclass
class ComponentStats:
    """Layout counts for one component: `total[k]` layouts use k mines, and `with_mine[c][k]` is the same
    count restricted to layouts where covered cell c is a mine."""

    order: tuple[int, ...]
    total: list[int]
    with_mine: dict[int, list[int]]


def _count_layouts(order: tuple[int, ...], constraints: Sequence[tuple[tuple[int, ...], int]],
                   fixed: tuple[int, int] | None = None) -> list[int]:
    """Count the layouts of `order` (cells in search order) that satisfy every constraint.

    Backtracking assigns cells one at a time, in `order`. After each assignment every constraint
    touching that cell must still be satisfiable: its residual need is at least 0, and at most the
    number of its cells not yet assigned. The search returns, for each mine total, how many complete
    layouts reach it.

    Memoization is what keeps this fast. The future of the search depends only on the cell index and
    the residual needs of the constraints still open at that index, not on the choices that led there.
    So each (index, residual needs) state is solved once, and its per-mine-count table is reused.
    `fixed=(cell, value)` forces one cell, which gives the layouts with that cell as a mine.
    """
    m = len(order)
    pos = {c: i for i, c in enumerate(order)}
    members = []
    needs = []
    for cells, need in constraints:
        members.append(sorted(pos[c] for c in cells))
        needs.append(need)
    # open_at[i]: constraints that have started before cell i and still have cells at or after i.
    # Only their residual needs carry information forward, so they are the memo key.
    open_at = [
        tuple(j for j, mem in enumerate(members) if mem[0] < i <= mem[-1]) for i in range(m + 1)
    ]
    # touch[i]: (constraint, members after i) for every constraint that contains cell i.
    touch: list[list[tuple[int, int]]] = [[] for _ in range(m)]
    for j, mem in enumerate(members):
        for k, i in enumerate(mem):
            touch[i].append((j, len(mem) - k - 1))
    fixed_idx = None
    if fixed is not None:
        fixed_idx = (pos[fixed[0]], fixed[1])

    memo: dict[tuple[int, tuple[int, ...]], list[int]] = {}

    def solve(i: int, state: tuple[int, ...]) -> list[int]:
        if i == m:
            return [1]  # every cell assigned, and every constraint was closed at zero (see pruning)
        key = (i, state)
        hit = memo.get(key)
        if hit is not None:
            return hit
        cur = dict(zip(open_at[i], state))
        values = (0, 1)
        if fixed_idx is not None and fixed_idx[0] == i:
            values = (fixed_idx[1],)
        out = [0] * (m - i + 1)
        for x in values:
            nxt = {}
            for j, after in touch[i]:
                r = cur.get(j, needs[j]) - x  # a constraint not yet open starts at its full need
                if r < 0 or r > after:  # too many mines, or too few cells left to supply the rest
                    break
                nxt[j] = r
            else:
                child = tuple(nxt[j] if j in nxt else cur[j] for j in open_at[i + 1])
                for k, v in enumerate(solve(i + 1, child)):
                    out[k + x] += v
        memo[key] = out
        return out

    return solve(0, ())


def component_stats(order: tuple[int, ...], constraints: tuple[tuple[tuple[int, ...], int], ...]) -> ComponentStats:
    """Layout counts for one component: the total, and the count with each covered cell forced to a mine.

    Results are cached on the exact component (cells and constraints), so a component that did not
    change since the last move costs nothing.
    """
    key = (order, constraints)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    total = _count_layouts(order, constraints)
    with_mine = {c: _count_layouts(order, constraints, fixed=(c, 1)) for c in order}
    stats = ComponentStats(order, total, with_mine)
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.clear()
    _CACHE[key] = stats
    return stats


def _components(constraints: list[Constraint]) -> list[list[Constraint]]:
    """Group constraints into components: two constraints are linked when they share a covered cell.

    Union-find over cells, then each constraint goes to the component of its cells.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for con in constraints:
        root = find(con.cells[0])
        for c in con.cells[1:]:
            parent[find(c)] = root
    groups: dict[int, list[Constraint]] = {}
    for con in constraints:
        groups.setdefault(find(con.cells[0]), []).append(con)
    return list(groups.values())


def _search_order(cells: list[int]) -> tuple[int, ...]:
    """Row-major order. Constraints span neighbouring cells, so this keeps the open constraints few."""
    return tuple(sorted(cells))


def analyse(view: View, level: int = LEVEL_PROBABILITY) -> Analysis:
    """Everything the agent can prove about `view`, at the requested level.

    Level 1 returns the single-cell rule deductions only. Levels 2 and 3 add the component analysis
    and the exact layout weights. Level 3 and level 2 share the same numbers: level 2 uses only the
    zero and full cases, level 3 uses the whole ranking when it guesses.
    """
    known_mine: set[int] = set()
    known_safe: set[int] = set()
    why: dict[int, str] = {}
    rule_pass(view, known_mine, known_safe, why)
    if level <= LEVEL_RULES:
        covered = tuple(i for i, v in enumerate(view.cells) if v == UNKNOWN and i not in known_mine)
        return Analysis(view, level, tuple(sorted(known_safe)), tuple(sorted(known_mine)), why, covered,
                        view.mines - len(known_mine))

    constraints = build_constraints(view, known_mine, known_safe)
    frontier = {c for con in constraints for c in con.cells}
    covered_all = [i for i, v in enumerate(view.cells) if v == UNKNOWN and i not in known_mine
                   and i not in known_safe]
    interior = tuple(i for i in covered_all if i not in frontier)
    remaining = view.mines - len(known_mine)

    comps = _components(constraints)
    stats = []
    components = []
    for group in comps:
        cells = sorted({c for con in group for c in con.cells})
        order = _search_order(cells)
        cons_key = tuple(sorted((tuple(sorted(con.cells)), con.need) for con in group))
        st = component_stats(order, cons_key)
        stats.append((order, st))
        components.append(Component(tuple(order), len(group), sum(st.total)))

    def weight(t: int) -> int:
        # Layouts of the interior with the leftover mines: `remaining - t` mines among len(interior) cells.
        return _choose(len(interior), remaining - t)

    # G: layouts of all components together, by total frontier mines. Z: weight of every valid layout.
    G = [1]
    for _, st in stats:
        G = _convolve(G, st.total)
    total = sum(G[t] * weight(t) for t in range(len(G)))
    if total == 0:
        raise ValueError("the revealed numbers and the mine count cannot all be true")

    numerator: dict[int, int] = {}
    for idx, (order, st) in enumerate(stats):
        # Everything except this component: its layout counts, combined.
        others = [1]
        for jdx, (_, other) in enumerate(stats):
            if jdx != idx:
                others = _convolve(others, other.total)
        # R[j]: weight of the full layouts when this component uses j mines.
        R = [sum(others[s] * weight(j + s) for s in range(len(others))) for j in range(len(st.total))]
        for c in order:
            numerator[c] = sum(st.with_mine[c][j] * R[j] for j in range(len(R)))
    if interior:
        # A given interior cell is a mine in C(n-1, k-1) of the layouts that put k mines among n cells.
        n = len(interior)
        per_cell = sum(G[t] * _choose(n - 1, remaining - t - 1) for t in range(len(G)))
        for c in interior:
            numerator[c] = per_cell

    safe = set(known_safe)
    mines = set(known_mine)
    for c, num in numerator.items():
        if num == 0:
            safe.add(c)
            if c in frontier:
                why.setdefault(c, "every layout that fits the numbers and the mine count leaves it clear")
            else:
                why.setdefault(c, "no layout that fits the mine count puts a mine here")
        elif num == total:
            mines.add(c)
            why.setdefault(c, "every layout that fits the numbers and the mine count has a mine here")
    covered = tuple(sorted(c for c in covered_all if c not in mines))
    return Analysis(
        view=view, level=level, safe=tuple(sorted(safe)), mines=tuple(sorted(mines)), why=why,
        covered=covered, remaining=remaining, interior=interior, components=tuple(components),
        numerator=numerator, total=total,
    )
