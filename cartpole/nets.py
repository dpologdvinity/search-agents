"""Small neural networks in pure NumPy: a forward pass, hand-written backprop, Adam, and .npz files.

Both the policy and the critic are one-hidden-layer MLPs with a tanh hidden layer:

    h = tanh(W1 @ x + b1)        x: normalised observation, shape (4,)
    z = W2 @ h + b2              policy: 2 logits (left, right); critic: 1 value

The policy turns its logits into probabilities with a softmax. The forward pass is written
for a batch of rows (T, 4) so training can process every step of a batch at once, and the
same code handles a single observation with T = 1.

Backprop is the chain rule written out by hand. Given dL/dz for each row, the gradients are

    dW2 = dZ^T H,   db2 = sum_rows dZ,   dH = dZ W2,
    dA  = dH * (1 - H^2)   (the derivative of tanh is 1 - tanh^2),
    dW1 = dA^T X,   db1 = sum_rows dA.

tests/test_cartpole.py compares these against finite differences.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import THETA_LIMIT, X_LIMIT

# Observations are scaled to roughly [-1, 1] before they reach a network. The scale is fixed
# (not learned), so the browser can apply the same scaling without any extra data.
OBS_SCALE = np.array([1.0 / X_LIMIT, 1.0 / 2.0, 1.0 / THETA_LIMIT, 1.0 / 2.0])


def normalise(obs) -> np.ndarray:
    """Scale one observation (4,) or a batch (T, 4) into network inputs."""
    return np.asarray(obs, dtype=np.float64) * OBS_SCALE


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Subtracting the row max keeps exp() from overflowing."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class MLP:
    """tanh hidden layer followed by a linear output. Parameters live in a dict so Adam and .npz can see them."""

    def __init__(self, params: dict[str, np.ndarray]):
        self.params = params

    @classmethod
    def init(cls, rng: np.random.Generator, n_in: int, n_hidden: int, n_out: int,
             out_scale: float = 1.0) -> MLP:
        """Random start. Hidden weights use a 1/sqrt(fan_in) scale; out_scale shrinks the output layer.

        A small out_scale (0.1 for the policy) makes the starting policy close to 50/50, so early
        episodes explore both directions instead of committing to one.
        """
        return cls({
            "W1": rng.normal(0.0, 1.0 / np.sqrt(n_in), (n_hidden, n_in)),
            "b1": np.zeros(n_hidden),
            "W2": rng.normal(0.0, out_scale / np.sqrt(n_hidden), (n_out, n_hidden)),
            "b2": np.zeros(n_out),
        })

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (H, Z): the hidden activations and the outputs, both with one row per input row."""
        p = self.params
        H = np.tanh(X @ p["W1"].T + p["b1"])
        Z = H @ p["W2"].T + p["b2"]
        return H, Z

    def backward(self, X: np.ndarray, H: np.ndarray, dZ: np.ndarray) -> dict[str, np.ndarray]:
        """Gradients of the loss with respect to every parameter, given dL/dZ for each row.

        The caller is responsible for the 1/T averaging and the sign, which depend on the loss.
        """
        W2 = self.params["W2"]
        dW2 = dZ.T @ H
        db2 = dZ.sum(axis=0)
        dH = dZ @ W2
        dA = dH * (1.0 - H * H)     # through the tanh: its derivative is 1 - tanh(a)^2
        dW1 = dA.T @ X
        db1 = dA.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


class Adam:
    """Adam (Kingma and Ba 2015). Keeps running averages of the gradient and its square, per parameter.

    The update is lr * m_hat / (sqrt(v_hat) + eps), where m_hat and v_hat are the averages
    corrected for starting at zero. Gradient ascent is handled by the caller flipping the sign.
    """

    def __init__(self, params: dict[str, np.ndarray], lr: float, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = params
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            self.params[k] -= self.lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write named arrays to .npz. float32 is plenty for these weights and keeps the files small."""
    np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k].astype(np.float64) for k in f.files}
