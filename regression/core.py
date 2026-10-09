"""Linear and logistic regression from scratch: features, two least-squares solvers, gradient descent.

Everything here is written so that web/js/regression-core.js can follow it operation by operation.
The least-squares solvers are hand-written (Gaussian elimination with partial pivoting for the normal
equations, Householder QR for the augmented system), so the two methods can be compared directly
instead of being hidden behind np.linalg.

Conventions shared with the page:
- Column 0 of the design matrix is the intercept. Ridge (L2) penalises every weight except the intercept.
- The regression objective is J(w) = (1/n) * (sum_i (y_i - x_i.w)^2 + lam * sum_{k>=1} w_k^2).
- The logistic objective is J(w) = (1/n) * (sum_i log-loss_i + lam * sum_{k>=1} w_k^2).
- Gradient descent records the loss before each step, so losses[t] belongs to the weights used at step t.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def poly_features(x, degree: int) -> np.ndarray:
    """Design matrix for a polynomial in one variable: columns x^0, x^1, ..., x^degree.

    Column 0 is all ones, so the first fitted weight is the intercept.
    """
    x = np.asarray(x, dtype=float).ravel()
    return np.stack([x**k for k in range(degree + 1)], axis=1)


def poly_features_2d(xy, degree: int) -> np.ndarray:
    """Design matrix for a polynomial in two variables: every monomial x1^i * x2^j with i + j <= degree.

    Monomials are ordered by total degree t = 0..degree, and within each t by decreasing power of x1.
    For degree 1 the columns are [1, x1, x2]; for degree 2 they are [1, x1, x2, x1^2, x1*x2, x2^2].
    """
    xy = np.asarray(xy, dtype=float)
    x1, x2 = xy[:, 0], xy[:, 1]
    cols = []
    for t in range(degree + 1):
        for i in range(t, -1, -1):
            cols.append(x1**i * x2 ** (t - i))
    return np.stack(cols, axis=1)


def _penalty_vector(p: int) -> np.ndarray:
    """D = diag(0, 1, ..., 1): the ridge penalty skips the intercept."""
    d = np.ones(p)
    d[0] = 0.0
    return d


def _gauss_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A w = b by Gaussian elimination with partial pivoting, written out so the port can mirror it.

    Partial pivoting swaps in the row with the largest entry in the current column, which keeps the
    multipliers at most 1 in size and limits the growth of rounding error.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    p = len(b)
    for k in range(p):
        piv = k + int(np.argmax(np.abs(A[k:, k])))
        if A[piv, k] == 0.0:
            raise np.linalg.LinAlgError("singular normal equations: the design has no spread in some direction")
        if piv != k:
            A[[k, piv]] = A[[piv, k]]
            b[[k, piv]] = b[[piv, k]]
        for r in range(k + 1, p):
            f = A[r, k] / A[k, k]
            A[r, k:] -= f * A[k, k:]
            b[r] -= f * b[k]
    # Back substitution on the now upper-triangular system.
    w = np.zeros(p)
    for r in range(p - 1, -1, -1):
        w[r] = (b[r] - A[r, r + 1 :] @ w[r + 1 :]) / A[r, r]
    return w


def solve_normal(X, y, ridge: float = 0.0) -> np.ndarray:
    """Least squares through the normal equations: solve (X^T X + lam D) w = X^T y.

    Fast, but forming X^T X squares the condition number of the design, so high-degree polynomials
    lose precision here first. That is why solve_qr exists next to it.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = X.shape[1]
    A = X.T @ X + ridge * np.diag(_penalty_vector(p))
    return _gauss_solve(A, X.T @ y)


def _rank_tolerance(A: np.ndarray) -> float:
    """Norm below which a remaining column counts as zero: the usual rule max(m, p) * eps * (largest column norm)."""
    m, p = A.shape
    largest = float(np.sqrt((A * A).sum(axis=0)).max()) if p else 0.0
    return max(m, p) * float(np.finfo(float).eps) * largest


def _householder_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least squares by Householder QR on an (m x p) matrix with m >= p, then back substitution.

    Each Householder reflection H = I - 2 v v^T / (v^T v) zeroes the entries below the diagonal of one
    column, and it is applied to the right-hand side as well, so the residual norm is never changed.
    The reflection is chosen with the sign that avoids cancellation in v[0] = x[0] - alpha.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, p = A.shape
    tol = _rank_tolerance(A)
    for k in range(p):
        x = A[k:, k]
        norm = float(np.sqrt(x @ x))
        # After rounding, a column that depends on the earlier ones leaves a tiny norm, not an exact zero.
        # Dividing by it would give huge, meaningless weights, so it is refused.
        if norm <= tol:
            raise np.linalg.LinAlgError("rank deficient design: a column depends on the columns before it")
        alpha = -norm if x[0] >= 0.0 else norm
        v = x.copy()
        v[0] -= alpha
        vv = float(v @ v)
        if vv == 0.0:
            continue
        # H A = A - (2 / vv) v (v^T A), applied to the trailing block only.
        A[k:, k:] -= np.outer(2.0 * v / vv, v @ A[k:, k:])
        b[k:] -= (2.0 * (v @ b[k:]) / vv) * v
    R = A[:p, :p]
    c = b[:p]
    w = np.zeros(p)
    for r in range(p - 1, -1, -1):
        w[r] = (c[r] - R[r, r + 1 :] @ w[r + 1 :]) / R[r, r]
    return w


def solve_qr(X, y, ridge: float = 0.0) -> np.ndarray:
    """Least squares by QR. With ridge, the penalty becomes extra rows sqrt(lam) * e_k (k >= 1).

    Minimising ||[y; 0] - [X; sqrt(lam) E] w||^2 is the same as minimising the ridge objective, and QR
    never forms X^T X, so the conditioning stays that of X itself.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = X.shape[1]
    if ridge > 0.0:
        E = np.eye(p)[1:]  # rows e_1 .. e_{p-1}: the intercept gets no penalty row
        A = np.vstack([X, np.sqrt(ridge) * E])
        b = np.concatenate([y, np.zeros(p - 1)])
    else:
        A, b = X, y
    return _householder_solve(A, b)


def sigmoid(z) -> np.ndarray:
    """The logistic function 1 / (1 + exp(-z)). The argument is clipped so exp never overflows.

    Beyond |z| = 500 the result is 0 or 1 to machine precision, so the clip changes nothing visible.
    """
    z = np.clip(np.asarray(z, dtype=float), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def regression_loss(X, y, w, ridge: float = 0.0) -> float:
    """J(w) = mean squared error plus the ridge penalty, both scaled by 1/n as on the page."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(y)
    r = y - X @ w
    return float((r @ r + ridge * (w[1:] @ w[1:])) / n)


def logistic_loss(X, labels, w, ridge: float = 0.0) -> float:
    """Mean log-loss plus the ridge penalty, computed stably.

    log p = -logaddexp(0, -z) and log(1 - p) = -logaddexp(0, z) avoid taking log of a rounded probability,
    which is what makes the loss blow up or go to -inf for confident predictions.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(labels)
    z = X @ w
    log_p = -np.logaddexp(0.0, -z)
    log_1mp = -np.logaddexp(0.0, z)
    nll = -(labels * log_p + (1.0 - labels) * log_1mp)
    return float((nll.sum() + ridge * (w[1:] @ w[1:])) / n)


def regression_gradient(X, y, w, ridge: float = 0.0) -> np.ndarray:
    """Gradient of regression_loss: (2/n) (X^T (X w - y) + lam D w), with D = diag(0, 1, ..., 1)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(y)
    d = _penalty_vector(len(w))
    return (2.0 / n) * (X.T @ (X @ w - y) + ridge * d * w)


def logistic_gradient(X, labels, w, ridge: float = 0.0) -> np.ndarray:
    """Gradient of logistic_loss: (1/n) (X^T (sigmoid(X w) - y) + 2 lam D w).

    The sigmoid's derivative cancels against the log-loss's, which is why the residual p - y appears
    with no extra factor. The penalty's derivative is 2 lam w, hence the factor 2.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(labels)
    d = _penalty_vector(len(w))
    return (1.0 / n) * (X.T @ (sigmoid(X @ w) - labels) + 2.0 * ridge * d * w)


@dataclass
class GDResult:
    """Outcome of a gradient descent run.

    w is the final weight vector. losses[t] is the objective at the weights used by step t, and the last
    entry is the objective at the final w, so len(losses) = steps + 1 when the run finishes. diverged is
    True when a loss became non-finite or exceeded DIVERGE_LIMIT; the run then stops early.
    """

    w: np.ndarray
    losses: list[float] = field(default_factory=list)
    diverged: bool = False


DIVERGE_LIMIT = 1e12


def _healthy(loss: float) -> bool:
    """A loss is usable when it is finite and below the divergence limit."""
    return bool(np.isfinite(loss) and loss <= DIVERGE_LIMIT)


def _run_descent(loss_fn, grad_fn, p: int, lr: float, steps: int, w0) -> GDResult:
    """Shared loop for both models: record loss, stop if unhealthy, otherwise take one step."""
    w = np.zeros(p) if w0 is None else np.array(w0, dtype=float)
    losses: list[float] = []
    diverged = False
    for _ in range(steps):
        loss = loss_fn(w)
        losses.append(loss)
        if not _healthy(loss):
            diverged = True
            break
        w = w - lr * grad_fn(w)
    else:
        # The loop ran to completion: record the loss at the final weights.
        loss = loss_fn(w)
        losses.append(loss)
        diverged = not _healthy(loss)
    return GDResult(w=w, losses=losses, diverged=diverged)


def gradient_descent(X, y, lr: float, steps: int, ridge: float = 0.0, w0=None) -> GDResult:
    """Batch gradient descent on the regression objective.

    The gradient is (2/n) (X^T (X w - y) + lam D w). Watching it creep toward the closed-form answer is
    the point of the lab: a small learning rate is slow, a large one can overshoot and diverge.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    p = X.shape[1]
    return _run_descent(
        lambda w: regression_loss(X, y, w, ridge), lambda w: regression_gradient(X, y, w, ridge), p, lr, steps, w0
    )


def logistic_gd(X, labels, lr: float, steps: int, ridge: float = 0.0, w0=None) -> GDResult:
    """Batch gradient descent on the mean log-loss. The gradient is (1/n) (X^T (p - y) + 2 lam D w)."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=float)
    p = X.shape[1]
    return _run_descent(
        lambda w: logistic_loss(X, labels, w, ridge), lambda w: logistic_gradient(X, labels, w, ridge), p, lr, steps, w0
    )


def train_test_mask(n: int) -> np.ndarray:
    """Boolean mask, True for test points: every fourth point (index 3, 7, 11, ...) is held out.

    The split is deterministic and spread across the x range, so the test error does not depend on a
    lucky random draw.
    """
    return np.arange(n) % 4 == 3


def mse(X, y, w) -> float:
    """Mean squared error without any penalty term."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    r = y - X @ np.asarray(w, dtype=float)
    return float(np.mean(r * r))


def accuracy(X, labels, w) -> float:
    """Fraction of points where the predicted probability is on the same side of 0.5 as the label."""
    p = sigmoid(np.asarray(X, dtype=float) @ np.asarray(w, dtype=float))
    return float(np.mean((p >= 0.5) == (np.asarray(labels) > 0.5)))
