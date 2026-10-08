"""Policy-value network inference in NumPy, plus the board encoding shared with training.

The network (trained in connect4.train_az) maps a position to
  policy: 7 logits, one per column
  value:  a score in [-1, 1] for the player to move
Batch norm is folded into the conv weights at export, so inference is just
convolutions, ReLUs, and matrix multiplies.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .board import COLS, H1, ROWS, Board

WEIGHTS = Path(__file__).parent / "data" / "az_connect4.npz"

# Bit index of each playable cell, laid out as (row, col) with row 0 at the bottom.
_CELL_BITS = np.array([[c * H1 + r for c in range(COLS)] for r in range(ROWS)])


def _bits(x: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(x.to_bytes(8, "little"), dtype=np.uint8), bitorder="little")


def encode(boards: list[Board]) -> np.ndarray:
    """(B, 2, 6, 7) float32: plane 0 = stones of the player to move, plane 1 = opponent's."""
    out = np.empty((len(boards), 2, ROWS, COLS), dtype=np.float32)
    for i, b in enumerate(boards):
        out[i, 0] = _bits(b.current)[_CELL_BITS]
        out[i, 1] = _bits(b.current ^ b.mask)[_CELL_BITS]
    return out


def legal_mask(boards: list[Board]) -> np.ndarray:
    mask = np.zeros((len(boards), COLS), dtype=bool)
    for i, b in enumerate(boards):
        for c in b.legal_moves():
            mask[i, c] = True
    return mask


def _conv3x3(x, w, b):
    """Same-padded 3x3 convolution: x (B, C, H, W), w (O, C, 3, 3) -> (B, O, H, W)."""
    batch, chans, h, wd = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    patches = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))  # B,C,H,W,3,3
    cols = patches.transpose(0, 2, 3, 1, 4, 5).reshape(batch * h * wd, chans * 9)
    out = cols @ w.reshape(w.shape[0], -1).T + b
    return out.reshape(batch, h, wd, -1).transpose(0, 3, 1, 2)


def _conv1x1(x, w, b):
    return np.einsum("bchw,oc->bohw", x, w[:, :, 0, 0]) + b[None, :, None, None]


relu = lambda x: np.maximum(x, 0.0)  # noqa: E731


class PolicyValueNet:
    def __init__(self, path=WEIGHTS):
        with np.load(path) as w:
            self.w = {k: w[k] for k in w.files}
        self.blocks = int(self.w["blocks"])

    def __call__(self, x: np.ndarray):
        """x (B, 2, 6, 7) -> (policy logits (B, 7), values (B,))."""
        w = self.w
        h = relu(_conv3x3(x, w["stem.w"], w["stem.b"]))
        for i in range(self.blocks):
            y = relu(_conv3x3(h, w[f"b{i}.1.w"], w[f"b{i}.1.b"]))
            y = _conv3x3(y, w[f"b{i}.2.w"], w[f"b{i}.2.b"])
            h = relu(h + y)
        p = relu(_conv1x1(h, w["pol.conv.w"], w["pol.conv.b"])).reshape(len(x), -1)
        logits = p @ w["pol.fc.w"].T + w["pol.fc.b"]
        v = relu(_conv1x1(h, w["val.conv.w"], w["val.conv.b"])).reshape(len(x), -1)
        v = relu(v @ w["val.fc1.w"].T + w["val.fc1.b"])
        value = np.tanh(v @ w["val.fc2.w"].T + w["val.fc2.b"])[:, 0]
        return logits, value

    def evaluate(self, boards: list[Board]):
        """Priors over legal columns (B, 7) and values (B,) for a batch of positions."""
        logits, values = self(encode(boards))
        return masked_softmax(logits, legal_mask(boards)), values


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, logits, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@lru_cache(maxsize=1)
def load() -> PolicyValueNet:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"{WEIGHTS} missing; train it with: python -m connect4.train_az")
    return PolicyValueNet()
