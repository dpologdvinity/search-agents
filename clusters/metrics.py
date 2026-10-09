"""Silhouette score: how well each point sits in its own cluster compared with the nearest other cluster.

For point i, a(i) is its mean distance to the other points of its cluster and b(i) the smallest mean distance
to the points of any other cluster. s(i) = (b - a) / max(a, b) lies in [-1, 1]; a singleton scores 0. The
score is the mean of s(i). Values near 1 mean compact, well-separated clusters; near 0 means overlap.
Points labelled -1 (DBSCAN noise) are left out.
"""

import numpy as np


def silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Mean silhouette over all points, or None when fewer than two clusters exist."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    keep = labels >= 0  # DBSCAN noise (-1) is not a cluster, so it is left out of the score
    X, labels = X[keep], labels[keep]
    ids = np.unique(labels)
    if len(ids) < 2:
        return None
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    total = 0.0
    for i in range(len(X)):
        same = labels == labels[i]
        n_same = int(same.sum())
        if n_same <= 1:
            continue  # singleton: s = 0
        a = float(D[i, same].sum()) / (n_same - 1)
        b = min(float(D[i, labels == j].mean()) for j in ids if j != labels[i])
        m = max(a, b)
        total += (b - a) / m if m > 0 else 0.0
    return total / len(X)
