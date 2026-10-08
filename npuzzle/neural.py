"""Learned 15-puzzle heuristic: NumPy inference for a network trained in npuzzle.train.

    h(s) = PDB(s) + softplus(net(one_hot(s)))

The network predicts only how far the pattern database underestimates the
true cost, so h never drops below the PDB. It is not admissible: it can
overestimate, so searches using it may return longer-than-optimal paths.

Inference uses NumPy only, so serving does not need torch.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .batch import pdb_batch

N = 4
CELLS = N * N
WEIGHTS = Path(__file__).parent / "data" / "neural_15.npz"


def one_hot(boards: np.ndarray) -> np.ndarray:
    """(B, 16) boards -> (B, 256) float32; feature 16*i + t is 1 if tile t is at index i."""
    batch = len(boards)
    x = np.zeros((batch, CELLS * CELLS), dtype=np.float32)
    x[np.arange(batch)[:, None], np.arange(CELLS) * CELLS + boards.astype(np.int64)] = 1.0
    return x


def softplus(x):
    return np.logaddexp(0.0, x)


class NeuralHeuristic:
    """MLP with residual blocks, loaded from exported weights.

    Layer layout must match npuzzle.train.ResidualMLP:
        input -> Linear -> ReLU -> [Linear -> ReLU -> Linear, + skip, ReLU] x k -> Linear
    """

    def __init__(self, path=WEIGHTS):
        with np.load(path) as w:
            self.w_in, self.b_in = w["in.weight"], w["in.bias"]
            k = int(w["blocks"])
            self.blocks = [
                (w[f"b{i}.0.weight"], w[f"b{i}.0.bias"], w[f"b{i}.1.weight"], w[f"b{i}.1.bias"])
                for i in range(k)
            ]
            self.w_out, self.b_out = w["out.weight"], w["out.bias"]

    def residual(self, boards: np.ndarray) -> np.ndarray:
        x = one_hot(boards)
        x = np.maximum(x @ self.w_in.T + self.b_in, 0.0)
        for w1, b1, w2, b2 in self.blocks:
            y = np.maximum(x @ w1.T + b1, 0.0)
            x = np.maximum(x + y @ w2.T + b2, 0.0)
        return (x @ self.w_out.T + self.b_out)[:, 0]

    def __call__(self, boards: np.ndarray) -> np.ndarray:
        """Batch heuristic: (B, 16) boards -> (B,) float estimates; 0 at the goal."""
        boards = np.asarray(boards)
        h = pdb_batch(boards) + softplus(self.residual(boards))
        goal = (boards == np.arange(CELLS)).all(axis=1)
        return np.where(goal, 0.0, h)


@lru_cache(maxsize=1)
def load() -> NeuralHeuristic:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"{WEIGHTS} missing; train it with: python -m npuzzle.train")
    return NeuralHeuristic()


def neural(config: tuple[int, ...], n: int) -> float:
    """Single-state heuristic. Slow per call; prefer batched search."""
    if n != N:
        raise ValueError("the neural heuristic covers the 4x4 puzzle only")
    return float(load()(np.array([config], dtype=np.int8))[0])
