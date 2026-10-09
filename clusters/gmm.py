"""Gaussian mixture model with full covariances, fitted by expectation-maximisation (EM).

E step: responsibilities r[i, k] = P(component k | x_i) under the current parameters.
M step: weights, means and covariances re-estimated from the responsibilities.
The log-likelihood never decreases across iterations (EM's guarantee), which the tests check.

Start: k-means++ centres give one hard labelling, which seeds the first M step. `reg` is added to each
covariance diagonal to keep a component from collapsing onto one point; it is tiny by default.
"""

from dataclasses import dataclass, field

import numpy as np

from clusters.kmeans import assign, kmeans_pp, random_init
from clusters.rng import Mulberry32

LOG_2PI = float(np.log(2.0 * np.pi))


def _log_pdf(X: np.ndarray, mu: np.ndarray, S: np.ndarray) -> np.ndarray:
    """log N(x | mu, S) for every row of X, using the closed-form 2x2 inverse."""
    a, b, c, d = S[0, 0], S[0, 1], S[1, 0], S[1, 1]
    det = a * d - b * c
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    dx = X[:, 0] - mu[0]
    dy = X[:, 1] - mu[1]
    maha = dx * (ia * dx + ib * dy) + dy * (ic * dx + id_ * dy)
    return -0.5 * (2.0 * LOG_2PI + np.log(det) + maha)


@dataclass
class GMMResult:
    """Fitted mixture: weights (k,), means (k, 2), covs (k, 2, 2), responsibilities (n, k), hard labels,
    the average log-likelihood after each E step, and whether it converged before max_iter."""

    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    resp: np.ndarray
    labels: np.ndarray
    loglik: float
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)


def _m_step(X: np.ndarray, resp: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights, means and covariances from responsibilities."""
    n, k = resp.shape
    weights = np.zeros(k)
    means = np.zeros((k, 2))
    covs = np.zeros((k, 2, 2))
    for j in range(k):
        r = resp[:, j]
        nk = max(float(r.sum()), 1e-10)
        weights[j] = nk / n
        means[j] = (r[:, None] * X).sum(axis=0) / nk
        diff = X - means[j]
        covs[j] = (diff.T * r) @ diff / nk + reg * np.eye(2)
    return weights, means, covs


def gmm_em(
    X: np.ndarray,
    k: int,
    seed: int = 0,
    init: str = "kmeans++",
    max_iter: int = 200,
    tol: float = 1e-6,
    reg: float = 1e-6,
) -> GMMResult:
    """Fit a k-component mixture. Stops when the average log-likelihood gains less than tol."""
    X = np.asarray(X, dtype=np.float64)
    rng = Mulberry32(seed)
    centres = kmeans_pp(X, k, rng) if init == "kmeans++" else random_init(X, k, rng)
    hard = assign(X, centres)
    resp = np.zeros((len(X), k))
    resp[np.arange(len(X)), hard] = 1.0
    weights, means, covs = _m_step(X, resp, reg)
    history: list[float] = []
    prev = -np.inf
    converged = False
    iterations = 0
    for it in range(1, max_iter + 1):
        # E step under the current parameters
        logp = np.stack([np.log(weights[j]) + _log_pdf(X, means[j], covs[j]) for j in range(k)], axis=1)
        m = logp.max(axis=1, keepdims=True)
        lognorm = m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))
        resp = np.exp(logp - lognorm[:, None])
        ll = float(lognorm.mean())
        history.append(ll)
        iterations = it
        if ll - prev < tol and it > 1:
            converged = True
            break
        prev = ll
        # Out of iterations: stop before the M step, so the returned weights, means and covariances are
        # the ones that produced resp, labels and loglik. An M step here would leave them one update ahead.
        if it == max_iter:
            break
        weights, means, covs = _m_step(X, resp, reg)
    return GMMResult(weights, means, covs, resp, np.argmax(resp, axis=1), history[-1], iterations, converged, history)


def ellipse(cov: np.ndarray, sigma: float = 2.0) -> tuple[float, float, float]:
    """Axis lengths (semi-major, semi-minor) and angle in radians of the sigma-contour of a 2x2 covariance."""
    a, b, d = float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 1])
    tr, det = a + d, a * d - b * b
    disc = np.sqrt(max(tr * tr / 4.0 - det, 0.0))
    l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc
    angle = 0.5 * float(np.arctan2(2.0 * b, a - d))
    return sigma * float(np.sqrt(max(l1, 0.0))), sigma * float(np.sqrt(max(l2, 0.0))), angle
