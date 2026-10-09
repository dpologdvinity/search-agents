"""The shipped Connect Four weights: which checkpoint they came from, and that the torch-free loader reads them."""

import numpy as np

from connect4.board import Board
from connect4.net import WEIGHTS, PolicyValueNet


def test_shipped_weights_record_their_checkpoint():
    # Exported from the iteration-96 checkpoint, after 34,944 self-play games (docs/connect4-training.md, phase 5).
    with np.load(WEIGHTS) as w:
        assert int(w["iteration"]) == 96
        assert int(w["games"]) == 34944


def test_loader_ignores_metadata_and_returns_legal_priors():
    priors, values = PolicyValueNet().evaluate([Board()])
    assert priors.shape == (1, 7) and np.isclose(priors.sum(), 1.0)
    assert values.shape == (1,) and -1.0 <= values[0] <= 1.0
