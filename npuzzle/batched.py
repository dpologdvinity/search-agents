"""Batch weighted A*: best-first search that evaluates the heuristic on many states at once.

Calling a neural network once per state costs tens of microseconds in
Python, mostly overhead. This search pops up to `batch_size` states per
iteration, expands them all, and scores every new child in one batched
heuristic call, which amortizes that overhead (DeepCubeA's BWAS).

Priority is f = g_weight * g + h. With g_weight < 1 the search leans toward
states the heuristic rates close to the goal: faster, but paths can be
longer than optimal. Paths are also not guaranteed optimal when h can
overestimate, as the learned heuristic can.
"""

from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from .batch import manhattan_batch, pdb_batch
from .board import Node, goal, is_solvable, neighbors
from .search import _start


def batch_weighted_a_star(board, batch_heuristic, batch_size=100, g_weight=1.0,
                          limits=None, on_expand=None, name="bwas"):
    """Search with batch_heuristic: (B, n*n) int8 array -> (B,) estimates."""
    start, n, run = _start(board, name, limits, on_expand)
    if not is_solvable(start, n):
        return run.result("unsolvable")

    moves = neighbors(n)
    target = goal(n)
    counter = itertools.count()
    best_g = {start: 0}
    parent = {start: (None, None)}
    closed = set()
    h0 = float(batch_heuristic(np.array([start], dtype=np.int8))[0])
    heap = [(h0, 0, next(counter), start)]

    while heap:
        batch = []
        while heap and len(batch) < batch_size:
            _, neg_g, _, config = heapq.heappop(heap)
            if config in closed or -neg_g != best_g[config]:
                continue  # stale entry: a cheaper path to config was found later
            if config == target:
                return _result(run, parent, config)
            closed.add(config)
            batch.append(config)
        if not batch:
            break
        if run.over_limit():
            return run.result("limit")

        new = []
        for config in batch:
            g = best_g[config]
            run.expand(Node(config, None, parent[config][1], g), len(heap))
            blank = config.index(0)
            for to, action in moves[blank]:
                child = list(config)
                child[blank], child[to] = child[to], 0
                child = tuple(child)
                if child in closed:
                    continue
                if g + 1 < best_g.get(child, math.inf):
                    run.generated += 1
                    best_g[child] = g + 1
                    parent[child] = (config, action)
                    new.append(child)
        if new:
            h = batch_heuristic(np.array(new, dtype=np.int8))
            for child, hc in zip(new, h.tolist()):
                g = best_g[child]
                heapq.heappush(heap, (g_weight * g + hc, -g, next(counter), child))
    return run.result("exhausted")


def _result(run, parent, config):
    path = []
    while parent[config][0] is not None:
        config, action = parent[config]
        path.append(action)
    path.reverse()
    return run.result("solved", path)


def batch_heuristic(name: str, n: int):
    """Batched version of a heuristic by name."""
    if name == "manhattan":
        return lambda boards: manhattan_batch(boards, n)
    if name == "pdb":
        if n != 4:
            raise ValueError("the pattern database covers the 4x4 puzzle only")
        return pdb_batch
    if name == "neural":
        from .neural import load

        if n != 4:
            raise ValueError("the neural heuristic covers the 4x4 puzzle only")
        return load()
    raise ValueError(f"no batched heuristic named {name!r}")
