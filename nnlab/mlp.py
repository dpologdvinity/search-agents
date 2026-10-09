"""A multilayer perceptron for binary classification, with backprop written out by hand.

Network: input (F features) -> hidden layers (tanh, ReLU or sigmoid) -> one sigmoid output unit.
Loss: binary cross-entropy on the output, plus an optional L2 penalty (lambda / 2) * sum(W^2) on the
weights (biases are not penalised). Training uses mini-batch gradient descent with plain SGD or Adam.

Notation: layer l maps a_l (its input) to z_l = a_l W_l^T + b_l, and a_{l+1} = f(z_l). The last
layer uses the sigmoid, so a_L is the predicted probability p.

web/js/nnlab-core.js mirrors every function here, with the same operation order where it matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rng import Rng

ACTIVATIONS = ("tanh", "relu", "sigmoid")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def activation(name: str, z: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "sigmoid":
        return sigmoid(z)
    raise ValueError(f"unknown activation {name!r}")


def activation_grad(name: str, a: np.ndarray) -> np.ndarray:
    """f'(z) written in terms of the output a = f(z), which backprop already has in hand.

    tanh:    d/dz tanh = 1 - tanh^2
    sigmoid: d/dz sigma = sigma (1 - sigma)
    relu:    d/dz max(0, z) = 1 where z > 0. a > 0 exactly when z > 0, so a is enough.
    """
    if name == "tanh":
        return 1.0 - a * a
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "relu":
        return (a > 0).astype(np.float64)
    raise ValueError(f"unknown activation {name!r}")


@dataclass
class Net:
    """Weights W[l] with shape (out, in), biases b[l] with shape (out,), and the hidden activation."""

    W: list[np.ndarray]
    b: list[np.ndarray]
    act: str

    @property
    def sizes(self) -> list[int]:
        return [self.W[0].shape[1]] + [w.shape[0] for w in self.W]


def init_net(sizes: list[int], act: str, rng: Rng) -> Net:
    """Uniform weights with a scale that keeps signal size roughly constant from layer to layer.

    Glorot (Xavier) uniform for every layer: a = sqrt(6 / (fan_in + fan_out)). ReLU hidden layers use
    He's rule instead, a = sqrt(6 / fan_in), because ReLU zeroes half its inputs. Biases start at 0.
    """
    W, b = [], []
    L = len(sizes) - 1
    for li in range(L):
        fan_in, fan_out = sizes[li], sizes[li + 1]
        is_hidden = li < L - 1
        scale = math.sqrt(6.0 / fan_in) if (act == "relu" and is_hidden) else math.sqrt(6.0 / (fan_in + fan_out))
        w = np.zeros((fan_out, fan_in))
        for j in range(fan_out):
            for i in range(fan_in):
                w[j, i] = scale * (2.0 * rng.random() - 1.0)
        W.append(w)
        b.append(np.zeros(fan_out))
    return Net(W, b, act)


def forward(net: Net, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run the network. Returns (a, z): a[l] is the input to layer l (a[0] = X, a[-1] = p), z[l] the pre-activations."""
    a = [X]
    z = []
    L = len(net.W)
    for li in range(L):
        zl = a[-1] @ net.W[li].T + net.b[li]
        z.append(zl)
        a.append(sigmoid(zl) if li == L - 1 else activation(net.act, zl))
    return a, z


def predict(net: Net, X: np.ndarray) -> np.ndarray:
    """Predicted probability of the pink class, shape (n, 1)."""
    a, _ = forward(net, X)
    return a[-1]


def loss_and_grads(net: Net, X: np.ndarray, y: np.ndarray, l2: float = 0.0):
    """Mean cross-entropy plus the L2 term, and its gradient with respect to every W and b.

    Chain rule, one layer at a time from the output back. Let dz_l = dL/dz_l.
      1. Output: L = mean BCE(z_L, y) with p = sigmoid(z_L). The sigmoid and the log cancel, so
         dL/dz_L = (p - y) / B. The B divides because the loss is a mean over the batch.
      2. Weights: z_l = a_l W_l^T + b_l, so dL/dW_l = dz_l^T a_l and dL/db_l = sum over rows of dz_l.
         The L2 term adds lambda * W_l.
      3. Input to the layer: dL/da_l = dz_l W_l. This is the output of layer l-1.
      4. Through the activation: dz_{l-1} = dL/da_l * f'(z_{l-1}), an element-wise product.
    Steps 2 to 4 repeat down to the first layer.

    The BCE is computed from the logit z directly, max(z, 0) - z y + log(1 + exp(-|z|)), which does not
    overflow for large |z|. Returns (data_loss, L2_loss, grads_W, grads_b).
    """
    a, z = forward(net, X)
    B = X.shape[0]
    zo = z[-1]
    data_loss = float(np.mean(np.maximum(zo, 0.0) - zo * y + np.log1p(np.exp(-np.abs(zo)))))
    l2_loss = 0.5 * l2 * float(sum(np.sum(w * w) for w in net.W))

    L = len(net.W)
    gW = [None] * L
    gb = [None] * L
    dz = (sigmoid(zo) - y) / B  # step 1
    for li in reversed(range(L)):
        gW[li] = dz.T @ a[li] + l2 * net.W[li]  # step 2
        gb[li] = dz.sum(axis=0)
        if li > 0:
            da = dz @ net.W[li]  # step 3
            dz = da * activation_grad(net.act, a[li])  # step 4
    return data_loss, l2_loss, gW, gb


def evaluate(net: Net, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Data loss (no L2 term) and accuracy at the 0.5 threshold, i.e. z >= 0. An empty set gives (nan, nan)."""
    if X.shape[0] == 0:
        return float("nan"), float("nan")
    a, z = forward(net, X)
    zo = z[-1]
    loss = float(np.mean(np.maximum(zo, 0.0) - zo * y + np.log1p(np.exp(-np.abs(zo)))))
    acc = float(np.mean((zo >= 0.0) == (y >= 0.5)))
    return loss, acc


class Optimizer:
    """Plain gradient descent (kind 'sgd') or Adam (kind 'adam', Kingma and Ba 2015).

    Adam keeps a running mean m of the gradient and a running mean v of its square, both with bias
    correction, so the step is about lr in size whatever the gradient's scale:
      m = b1 m + (1 - b1) g,  v = b2 v + (1 - b2) g^2
      p -= lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)
    """

    def __init__(self, kind: str, net: Net, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        if kind not in ("sgd", "adam"):
            raise ValueError(f"unknown optimizer {kind!r}")
        self.kind, self.lr, self.b1, self.b2, self.eps = kind, lr, b1, b2, eps
        self.t = 0
        self.mW = [np.zeros_like(w) for w in net.W]
        self.vW = [np.zeros_like(w) for w in net.W]
        self.mb = [np.zeros_like(v) for v in net.b]
        self.vb = [np.zeros_like(v) for v in net.b]

    def step(self, net: Net, gW: list[np.ndarray], gb: list[np.ndarray]) -> None:
        self.t += 1
        for li in range(len(net.W)):
            net.W[li] = self._update(net.W[li], gW[li], self.mW, self.vW, li)
            net.b[li] = self._update(net.b[li], gb[li], self.mb, self.vb, li)

    def _update(self, p, g, m_list, v_list, li):
        if self.kind == "sgd":
            return p - self.lr * g
        m = self.b1 * m_list[li] + (1.0 - self.b1) * g
        v = self.b2 * v_list[li] + (1.0 - self.b2) * g * g
        m_list[li], v_list[li] = m, v
        mhat = m / (1.0 - self.b1**self.t)
        vhat = v / (1.0 - self.b2**self.t)
        return p - self.lr * mhat / (np.sqrt(vhat) + self.eps)


def shuffled_indices(n: int, rng: Rng) -> list[int]:
    """Fisher-Yates shuffle: walk from the end, swap each slot with a uniformly chosen earlier-or-equal slot."""
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def train_epoch(net: Net, opt: Optimizer, X: np.ndarray, y: np.ndarray, rng: Rng, batch: int, l2: float) -> None:
    """One pass over the training set in a fresh random order, one optimiser step per mini-batch.

    batch <= 0 or batch >= n means full-batch gradient descent. The last batch may be smaller.
    """
    n = X.shape[0]
    if n == 0:
        return
    order = np.array(shuffled_indices(n, rng), dtype=np.int64)
    bs = n if batch <= 0 or batch >= n else batch
    for start in range(0, n, bs):
        idx = order[start : start + bs]
        _, _, gW, gb = loss_and_grads(net, X[idx], y[idx], l2)
        opt.step(net, gW, gb)
