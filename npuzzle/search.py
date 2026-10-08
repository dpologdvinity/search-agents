"""Uninformed and informed graph search over N-Puzzle boards.

Every search takes a start board (any sequence of tiles) and returns a
SearchResult. Optional limits stop a search early. An optional
on_expand(node, frontier_size) callback sees every expanded node; the web
visualizer uses it to stream progress.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .board import INVERSE, Node, goal, is_solvable, neighbors, size_of, successors, validate
from .frontier import PriorityFrontier, QueueFrontier, StackFrontier
from .heuristics import manhattan

ExpandCallback = Callable[[Node, int], None]


@dataclass
class SearchLimits:
    """Stop conditions. None means no limit."""

    max_nodes: int | None = None
    max_seconds: float | None = None


@dataclass
class SearchResult:
    """Outcome and statistics of one search.

    status is "solved", "unsolvable", "exhausted" (nothing left to search,
    which only happens for bounded searches), or "limit".
    expanded counts nodes whose successors were generated; generated counts
    successors created; max_depth is the deepest expanded node.
    """

    algorithm: str
    status: str
    path: list[str] = field(default_factory=list)
    cost: int = 0
    expanded: int = 0
    generated: int = 0
    max_depth: int = 0
    max_frontier: int = 0
    seconds: float = 0.0

    @property
    def solved(self) -> bool:
        return self.status == "solved"


class _Run:
    """Statistics, limits, and the callback for one search."""

    def __init__(self, algorithm, limits, on_expand):
        self.algorithm = algorithm
        self.limits = limits or SearchLimits()
        self.on_expand = on_expand
        self.expanded = self.generated = self.max_depth = self.max_frontier = 0
        self.start = time.perf_counter()

    def expand(self, node, frontier_size):
        """Record an expansion and report it to the callback."""
        self.expanded += 1
        if node.depth > self.max_depth:
            self.max_depth = node.depth
        if frontier_size > self.max_frontier:
            self.max_frontier = frontier_size
        if self.on_expand is not None:
            self.on_expand(node, frontier_size)

    def over_limit(self):
        lim = self.limits
        if lim.max_nodes is not None and self.expanded >= lim.max_nodes:
            return True
        # Reading the clock on every expansion is slow; every 1024 is enough.
        if lim.max_seconds is not None and self.expanded % 1024 == 0:
            return time.perf_counter() - self.start >= lim.max_seconds
        return False

    def result(self, status, path=None):
        path = path or []
        return SearchResult(self.algorithm, status, path, len(path), self.expanded, self.generated,
                            self.max_depth, self.max_frontier, time.perf_counter() - self.start)


def _start(board, algorithm, limits, on_expand):
    board = validate(board)
    n = size_of(board)
    return board, n, _Run(algorithm, limits, on_expand)


def bfs(board, limits=None, on_expand=None):
    """Breadth-first search, testing for the goal when a node is generated.

    Optimal because every move costs 1. Testing at generation instead of at
    removal finds the goal one full layer of expansions earlier.
    """
    start, n, run = _start(board, "bfs", limits, on_expand)
    if not is_solvable(start, n):
        return run.result("unsolvable")
    target = goal(n)
    if start == target:
        return run.result("solved")
    frontier = QueueFrontier()
    frontier.add(Node(start))
    reached = {start}
    while len(frontier):
        if run.over_limit():
            return run.result("limit")
        node = frontier.pop()
        run.expand(node, len(frontier))
        for action, nxt in successors(node.board, n):
            if nxt in reached:
                continue
            run.generated += 1
            child = node.child(action, nxt)
            if nxt == target:
                return run.result("solved", child.path())
            reached.add(nxt)
            frontier.add(child)
    return run.result("exhausted")


def dfs(board, limits=None, on_expand=None):
    """Depth-first graph search. Finds some path quickly, rarely a short one.

    Successors are pushed in reverse so they are explored in action order.
    """
    start, n, run = _start(board, "dfs", limits, on_expand)
    if not is_solvable(start, n):
        return run.result("unsolvable")
    target = goal(n)
    frontier = StackFrontier()
    frontier.add(Node(start))
    done = set()
    while len(frontier):
        node = frontier.pop()
        if node.board == target:
            return run.result("solved", node.path())
        if run.over_limit():
            return run.result("limit")
        done.add(node.board)
        run.expand(node, len(frontier))
        for action, nxt in reversed(list(successors(node.board, n))):
            if nxt not in done and nxt not in frontier:
                run.generated += 1
                frontier.add(node.child(action, nxt))
    return run.result("exhausted")


def best_first(board, priority, algorithm, limits=None, on_expand=None):
    """Best-first graph search ordered by priority(node, n).

    With priority g + h for a consistent heuristic h this is A*, and the
    first goal removed from the frontier is optimal. A board already in the
    frontier is replaced when a cheaper path to it appears (decrease-key).
    """
    start, n, run = _start(board, algorithm, limits, on_expand)
    if not is_solvable(start, n):
        return run.result("unsolvable")
    target = goal(n)
    frontier = PriorityFrontier()
    root = Node(start)
    frontier.add(root, priority(root, n))
    done = set()
    while len(frontier):
        node = frontier.pop()
        if node.board == target:
            return run.result("solved", node.path())
        if run.over_limit():
            return run.result("limit")
        done.add(node.board)
        run.expand(node, len(frontier))
        for action, nxt in successors(node.board, n):
            if nxt in done:
                continue
            run.generated += 1
            child = node.child(action, nxt)
            p = priority(child, n)
            if nxt in frontier:
                frontier.decrease_key(child, p)
            else:
                frontier.add(child, p)
    return run.result("exhausted")


def a_star(board, heuristic=manhattan, limits=None, on_expand=None):
    """A*: f = g + h, ties broken toward the deeper node.

    Many boards share each f value; among them the deeper one has the
    smaller h and is likely nearer the goal, which cuts expansions sharply.
    """
    return best_first(board, lambda nd, n: (nd.depth + heuristic(nd.board, n), -nd.depth),
                      "astar", limits, on_expand)


def ucs(board, limits=None, on_expand=None):
    """Uniform-cost search: best-first on g alone. Optimal, no heuristic."""
    return best_first(board, lambda nd, n: nd.depth, "ucs", limits, on_expand)


def greedy(board, heuristic=manhattan, limits=None, on_expand=None):
    """Greedy best-first search on h alone. Fast, but paths are not optimal."""
    return best_first(board, lambda nd, n: heuristic(nd.board, n), "greedy", limits, on_expand)


def weighted_a_star(board, heuristic=manhattan, weight=2.0, limits=None, on_expand=None):
    """A* with f = g + w*h. With an admissible h, the path is at most w times optimal."""
    return best_first(board, lambda nd, n: (nd.depth + weight * heuristic(nd.board, n), -nd.depth),
                      "wastar", limits, on_expand)


def bidirectional_bfs(board, limits=None, on_expand=None):
    """Breadth-first search from the start and the goal at the same time.

    Each step expands one full layer of whichever side has the smaller
    frontier. When a layer reaches boards the other side has seen, the
    cheapest meeting point gives an optimal path. Each side searches about
    half the depth, so far fewer nodes are expanded than with plain BFS.
    """
    start, n, run = _start(board, "bibfs", limits, on_expand)
    if not is_solvable(start, n):
        return run.result("unsolvable")
    target = goal(n)
    if start == target:
        return run.result("solved")

    moves = neighbors(n)
    # parents[side][board] = (parent board, action, depth)
    parents = ({start: (None, None, 0)}, {target: (None, None, 0)})
    layers = ([start], [target])

    while layers[0] and layers[1]:
        side = 0 if len(layers[0]) <= len(layers[1]) else 1
        mine, other = parents[side], parents[1 - side]
        next_layer = []
        best = None  # (total cost, meeting board)
        for b in layers[side]:
            if run.over_limit():
                return run.result("limit")
            _, action, depth = mine[b]
            # Goal-side actions lead away from the goal; the prefix marks them.
            label = action if side == 0 or action is None else "goal:" + action
            run.expand(Node(b, None, label, depth), len(layers[0]) + len(layers[1]))
            blank = b.index(0)
            for t, act in moves[blank]:
                nxt = list(b)
                nxt[blank], nxt[t] = b[t], 0
                nxt = tuple(nxt)
                if nxt in mine:
                    continue
                run.generated += 1
                mine[nxt] = (b, act, depth + 1)
                next_layer.append(nxt)
                if nxt in other:
                    total = depth + 1 + other[nxt][2]
                    if best is None or total < best[0]:
                        best = (total, nxt)
        layers = (next_layer, layers[1]) if side == 0 else (layers[0], next_layer)
        if best is not None:
            return run.result("solved", _join(parents, best[1]))
    return run.result("exhausted")


def _join(parents, meet):
    forward, backward = parents
    path, b = [], meet
    while forward[b][0] is not None:
        b, action = forward[b][0], forward[b][1]
        path.append(action)
    path.reverse()
    b = meet
    while backward[b][0] is not None:  # walk toward the goal, reversing each move
        action = backward[b][1]
        path.append(INVERSE[action])
        b = backward[b][0]
    return path
