"""k-means (Lloyd's algorithm) with k-means++ or random seeding, plus the elbow sweep.

One iteration is two steps: assign every point to its nearest centre, then move each centre to the mean of
its points. `kmeans_steps` yields each step, so the page can animate them one at a time; `kmeans` runs them
to the end. Both are the same code path, so the animation and the CLI agree.

Ties go to the lower centre index in both the labelling and the seeding draws, so results are deterministic.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from clusters.rng import Mulberry32


def assign(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Index of the nearest centre for each point (squared Euclidean, ties to the lower index)."""
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return np.argmin(d, axis=1)


def update(X: np.ndarray, labels: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Move each centre to the mean of its points. A centre with no points stays where it was."""
    out = C.copy()
    for j in range(len(C)):
        mask = labels == j
        count = int(mask.sum())
        if count:
            out[j] = X[mask].sum(axis=0) / count
    return out


def sse(X: np.ndarray, labels: np.ndarray, C: np.ndarray) -> float:
    """Within-cluster sum of squared errors (the inertia)."""
    diff = X - C[labels]
    return float((diff * diff).sum())


def kmeans_pp(X: np.ndarray, k: int, rng: Mulberry32) -> np.ndarray:
    """k-means++ seeding: the first centre is a uniform point; each next one is drawn with probability
    proportional to the squared distance to the nearest centre chosen so far (D^2 sampling)."""
    n = len(X)
    first = rng.index(n)
    centres = [X[first]]
    d2 = ((X - X[first]) ** 2).sum(axis=1)
    for _ in range(1, k):
        cum = np.cumsum(d2)
        total = float(cum[-1])
        if total <= 0.0:
            j = rng.index(n)
        else:
            r = rng.random() * total
            j = int(np.searchsorted(cum, r, side="right"))
            j = min(j, n - 1)
        centres.append(X[j])
        d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(axis=1))
    return np.array(centres, dtype=np.float64)


def random_init(X: np.ndarray, k: int, rng: Mulberry32) -> np.ndarray:
    """k distinct data points chosen by a seeded Fisher-Yates shuffle."""
    order = list(range(len(X)))
    for i in range(len(order) - 1, 0, -1):
        j = rng.index(i + 1)
        order[i], order[j] = order[j], order[i]
    return X[np.array(order[:k], dtype=np.int64)].astype(np.float64)


@dataclass
class KMeansStep:
    """One animation frame. phase is 'init', 'assign' or 'update'; inertia is the SSE of that frame."""

    phase: str
    iteration: int
    labels: np.ndarray | None
    centers: np.ndarray
    inertia: float | None


@dataclass
class KMeansResult:
    """Final state of a run: labels, centres, SSE, iterations used, and the SSE after each assign step."""

    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)


def kmeans_steps(
    X: np.ndarray,
    k: int,
    seed: int = 0,
    init: str = "kmeans++",
    centers: np.ndarray | None = None,
    max_iter: int = 100,
) -> Iterator[KMeansStep]:
    """Yield the run one step at a time. Stops when an assign step leaves every label unchanged.

    `centers` overrides the seeding (the page passes the centroid the user dragged). The inertia after an
    assign step is never larger than after the previous update, so the sequence is non-increasing.
    """
    X = np.asarray(X, dtype=np.float64)
    rng = Mulberry32(seed)
    if centers is not None:
        C = np.array(centers, dtype=np.float64)
    elif init == "kmeans++":
        C = kmeans_pp(X, k, rng)
    elif init == "random":
        C = random_init(X, k, rng)
    else:
        raise ValueError(f"unknown init {init!r}")
    yield KMeansStep("init", 0, None, C.copy(), None)
    labels: np.ndarray | None = None
    for it in range(1, max_iter + 1):
        new = assign(X, C)
        yield KMeansStep("assign", it, new, C.copy(), sse(X, new, C))
        if labels is not None and np.array_equal(new, labels):
            return
        labels = new
        C = update(X, labels, C)
        yield KMeansStep("update", it, labels, C.copy(), sse(X, labels, C))


def kmeans(
    X: np.ndarray,
    k: int,
    seed: int = 0,
    init: str = "kmeans++",
    centers: np.ndarray | None = None,
    max_iter: int = 100,
) -> KMeansResult:
    """Run k-means to convergence (or max_iter) and return the final state.

    The generator ends right after an assign step when labels stopped changing (converged), and right
    after an update step when max_iter ran out. In the second case the labels are recomputed for the
    final centres so the result is always a valid labelling.
    """
    X = np.asarray(X, dtype=np.float64)
    last: KMeansStep | None = None
    history: list[float] = []
    for step in kmeans_steps(X, k, seed, init, centers, max_iter):
        if step.phase == "assign":
            history.append(float(step.inertia))  # type: ignore[arg-type]
        last = step
    assert last is not None and last.phase != "init"
    converged = last.phase == "assign"
    if converged:
        labels, final_centres = last.labels, last.centers
        inertia = history[-1]
    else:
        final_centres = last.centers
        labels = assign(X, final_centres)
        inertia = sse(X, labels, final_centres)
    assert labels is not None
    return KMeansResult(labels, final_centres, inertia, len(history), converged, history)


def elbow(X: np.ndarray, k_max: int = 10, seed: int = 0) -> list[tuple[int, float]]:
    """Inertia of k-means for k = 1..k_max, all with the same seed (so the sweep is reproducible)."""
    out = []
    for k in range(1, k_max + 1):
        out.append((k, kmeans(X, k, seed=seed).inertia))
    return out
