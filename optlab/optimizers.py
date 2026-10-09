"""Gradient-based optimizers written from scratch, one update rule per class.

Every optimizer has the same two-step interface, so the race can evaluate the gradient wherever the
optimizer asks for it:

    q = opt.query(x)          # where to take the gradient (Nesterov looks ahead)
    x = opt.update(x, g)      # g = gradient at q, plus any noise

The update rules are written in the order the JavaScript port uses, with explicit products rather than
pow(), so the two ports agree to the last bit where the arithmetic allows it.

Update rules (x is the position, g the gradient, lr the learning rate):

    sgd       x <- x - lr g
    momentum  v <- beta v - lr g(x);            x <- x + v
    nesterov  v <- beta v - lr g(x + beta v);   x <- x + v
    rmsprop   s <- rho s + (1 - rho) g^2;       x <- x - lr g / (sqrt(s) + eps)
    adagrad   G <- G + g^2;                     x <- x - lr g / (sqrt(G) + eps)
    adam      m <- b1 m + (1 - b1) g;  v <- b2 v + (1 - b2) g^2
              x <- x - lr (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

NAMES = ("sgd", "momentum", "nesterov", "rmsprop", "adam", "adagrad")
LABELS = {
    "sgd": "SGD",
    "momentum": "Momentum",
    "nesterov": "Nesterov",
    "rmsprop": "RMSProp",
    "adam": "Adam",
    "adagrad": "AdaGrad",
}


@dataclass(frozen=True)
class Hyper:
    """Shared hyperparameters. Each optimizer reads only the ones it uses."""

    momentum: float = 0.9  # beta, for momentum and Nesterov
    beta1: float = 0.9  # Adam first-moment decay
    beta2: float = 0.999  # Adam second-moment decay
    eps: float = 1e-8  # added under the square root (RMSProp, AdaGrad, Adam)
    rho: float = 0.9  # RMSProp squared-gradient decay


DEFAULT_HYPER = Hyper()


class Optimizer:
    """Base class: `lr` is the step size; subclasses override query() and update()."""

    name = ""

    def __init__(self, lr: float, hyper: Hyper = DEFAULT_HYPER):
        self.lr = float(lr)
        self.h = hyper

    def query(self, x: list[float]) -> list[float]:
        return [x[0], x[1]]

    def update(self, x: list[float], g: list[float]) -> list[float]:
        raise NotImplementedError


class SGD(Optimizer):
    name = "sgd"

    def update(self, x, g):
        return [x[0] - self.lr * g[0], x[1] - self.lr * g[1]]


class Momentum(Optimizer):
    """Heavy-ball momentum: the gradient is taken at x."""

    name = "momentum"

    def __init__(self, lr, hyper=DEFAULT_HYPER):
        super().__init__(lr, hyper)
        self.v = [0.0, 0.0]

    def update(self, x, g):
        b = self.h.momentum
        out = [0.0, 0.0]
        for i in range(2):
            self.v[i] = b * self.v[i] - self.lr * g[i]
            out[i] = x[i] + self.v[i]
        return out


class Nesterov(Momentum):
    """Nesterov momentum: the gradient is taken at the look-ahead point x + beta v."""

    name = "nesterov"

    def query(self, x):
        b = self.h.momentum
        return [x[0] + b * self.v[0], x[1] + b * self.v[1]]


class RMSProp(Optimizer):
    name = "rmsprop"

    def __init__(self, lr, hyper=DEFAULT_HYPER):
        super().__init__(lr, hyper)
        self.s = [0.0, 0.0]

    def update(self, x, g):
        rho, eps = self.h.rho, self.h.eps
        out = [0.0, 0.0]
        for i in range(2):
            self.s[i] = rho * self.s[i] + (1.0 - rho) * (g[i] * g[i])
            out[i] = x[i] - self.lr * g[i] / (math.sqrt(self.s[i]) + eps)
        return out


class AdaGrad(Optimizer):
    name = "adagrad"

    def __init__(self, lr, hyper=DEFAULT_HYPER):
        super().__init__(lr, hyper)
        self.G = [0.0, 0.0]

    def update(self, x, g):
        eps = self.h.eps
        out = [0.0, 0.0]
        for i in range(2):
            self.G[i] = self.G[i] + g[i] * g[i]
            out[i] = x[i] - self.lr * g[i] / (math.sqrt(self.G[i]) + eps)
        return out


class Adam(Optimizer):
    """Adam with bias correction. The powers beta^t are kept as running products, not pow(), for parity."""

    name = "adam"

    def __init__(self, lr, hyper=DEFAULT_HYPER):
        super().__init__(lr, hyper)
        self.m = [0.0, 0.0]
        self.v = [0.0, 0.0]
        self.b1t = 1.0
        self.b2t = 1.0

    def update(self, x, g):
        b1, b2, eps = self.h.beta1, self.h.beta2, self.h.eps
        self.b1t *= b1
        self.b2t *= b2
        out = [0.0, 0.0]
        for i in range(2):
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g[i]
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * (g[i] * g[i])
            mh = self.m[i] / (1.0 - self.b1t)
            vh = self.v[i] / (1.0 - self.b2t)
            out[i] = x[i] - self.lr * mh / (math.sqrt(vh) + eps)
        return out


CLASSES = {
    "sgd": SGD,
    "momentum": Momentum,
    "nesterov": Nesterov,
    "rmsprop": RMSProp,
    "adam": Adam,
    "adagrad": AdaGrad,
}


def make(name: str, lr: float, hyper: Hyper = DEFAULT_HYPER) -> Optimizer:
    """Build an optimizer by name."""
    try:
        return CLASSES[name](lr, hyper)
    except KeyError:
        raise KeyError(f"unknown optimizer {name!r}; choose from {', '.join(NAMES)}") from None
