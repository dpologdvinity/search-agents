"""DBSCAN: density-based clustering with an eps radius and a minPts threshold.

A point is a core point when at least minPts points (itself included) lie within eps. Core points within
eps of each other form one cluster, which grows like a flood fill: each cluster starts at the lowest-index
unvisited core point and absorbs every core point reachable through eps-neighbours. A non-core point within
eps of some core point is a border point; it joins the cluster of its lowest-index core neighbour (a fixed
rule, so the result does not depend on visiting order). Everything else is noise, labelled -1.

Cluster ids are numbered in order of their lowest core index, so the labels are canonical.
"""

from collections import deque

import numpy as np

NOISE = -1


def neighbourhoods(X: np.ndarray, eps: float) -> list[list[int]]:
    """For each point, the sorted indices of all points within eps (squared-distance test, self included)."""
    if not eps >= 0:  # also rejects NaN; eps is squared below, so a negative value would pass silently as |eps|
        raise ValueError(f"eps must be zero or more, got {eps}")
    X = np.asarray(X, dtype=np.float64)
    limit = eps * eps
    out: list[list[int]] = []
    for i in range(len(X)):
        d2 = ((X - X[i]) ** 2).sum(axis=1)
        out.append([int(j) for j in np.flatnonzero(d2 <= limit)])
    return out


def dbscan(X: np.ndarray, eps: float, min_pts: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, core) where labels[i] is a cluster id or NOISE, and core marks core points."""
    nbrs = neighbourhoods(X, eps)
    n = len(nbrs)
    core = np.array([len(nb) >= min_pts for nb in nbrs], dtype=bool)
    labels = np.full(n, NOISE, dtype=np.int64)
    cid = 0
    for i in range(n):
        if not core[i] or labels[i] != NOISE:
            continue
        labels[i] = cid
        queue: deque[int] = deque([i])
        while queue:  # flood fill over core points only
            p = queue.popleft()
            for q in nbrs[p]:
                if core[q] and labels[q] == NOISE:
                    labels[q] = cid
                    queue.append(q)
        cid += 1
    for i in range(n):  # border points take their lowest-index core neighbour's cluster
        if core[i]:
            continue
        for q in nbrs[i]:
            if core[q]:
                labels[i] = labels[q]
                break
    return labels, core
