"""Train a network on a preset dataset, one epoch at a time, and record the curves.

The seed fixes everything: the dataset points use Rng(seed), the initial weights Rng(seed + 1) and the
mini-batch order Rng(seed + 2). The browser uses the same three streams, so the same config gives the
same curves in both places (see tests/test_nnlab_parity.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEFAULTS
from .data import build_split, generate
from .mlp import Net, Optimizer, evaluate, init_net, train_epoch
from .rng import Rng


@dataclass
class TrainConfig:
    data: str = DEFAULTS["data"]
    n: int = DEFAULTS["n"]
    seed: int = DEFAULTS["seed"]
    noise: float = DEFAULTS["noise"]
    features: tuple[str, ...] = tuple(DEFAULTS["features"])
    test_frac: float = DEFAULTS["test_frac"]
    hidden: tuple[int, ...] = tuple(DEFAULTS["hidden"])
    act: str = DEFAULTS["act"]
    lr: float = DEFAULTS["lr"]
    batch: int = DEFAULTS["batch"]
    l2: float = DEFAULTS["l2"]
    optimizer: str = DEFAULTS["optimizer"]
    epochs: int = DEFAULTS["epochs"]


@dataclass
class History:
    """Per-epoch metrics. Entry 0 is the untrained network, so the curves start at the initial loss."""

    epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    test_acc: list[float] = field(default_factory=list)

    def record(self, epoch: int, net: Net, Xtr, ytr, Xte, yte) -> None:
        tl, ta = evaluate(net, Xtr, ytr)
        vl, va = evaluate(net, Xte, yte)
        self.epoch.append(epoch)
        self.train_loss.append(tl)
        self.test_loss.append(vl)
        self.train_acc.append(ta)
        self.test_acc.append(va)


@dataclass
class Run:
    config: TrainConfig
    net: Net
    points: list[dict]
    split: tuple
    history: History


def setup(cfg: TrainConfig):
    """Build the points, the split and the untrained network from a config."""
    points = generate(cfg.data, cfg.n, cfg.seed)
    Xtr, ytr, Xte, yte = build_split(points, cfg.features, cfg.noise, cfg.test_frac)
    sizes = [len(cfg.features), *cfg.hidden, 1]
    net = init_net(sizes, cfg.act, Rng(cfg.seed + 1))
    return points, (Xtr, ytr, Xte, yte), net


def train(cfg: TrainConfig, on_epoch=None) -> Run:
    """Run cfg.epochs epochs. on_epoch(epoch, history) is called after each one, for live output."""
    points, split, net = setup(cfg)
    Xtr, ytr, Xte, yte = split
    opt = Optimizer(cfg.optimizer, net, cfg.lr)
    shuffle_rng = Rng(cfg.seed + 2)
    history = History()
    history.record(0, net, Xtr, ytr, Xte, yte)
    for e in range(1, cfg.epochs + 1):
        train_epoch(net, opt, Xtr, ytr, shuffle_rng, cfg.batch, cfg.l2)
        history.record(e, net, Xtr, ytr, Xte, yte)
        if on_epoch is not None:
            on_epoch(e, history)
    return Run(cfg, net, points, split, history)
