import numpy as np
import pytest

from connect4.board import COLS, Board
from connect4.net import PolicyValueNet, encode, legal_mask
from connect4.puct import Tree, run_simulations, search


def uniform(boards):
    mask = legal_mask(boards)
    return mask / mask.sum(axis=1, keepdims=True), np.zeros(len(boards))


def test_encode_planes():
    board = Board.from_moves("4455")  # X: d1 e1, O: d2 e2; X to move
    x = encode([board])[0]
    assert x.shape == (2, 6, 7)
    assert x[0, 0, 3] == x[0, 0, 4] == 1  # player to move (X) on the bottom row
    assert x[1, 1, 3] == x[1, 1, 4] == 1  # opponent one row up
    assert x.sum() == 4


def test_puct_takes_win_and_blocks_with_uninformed_evaluator():
    win = Board.from_moves("121212")
    assert int(search(win, uniform, simulations=200).root.visits.argmax()) == 0
    block = Board.from_moves("323272")
    assert int(search(block, uniform, simulations=800).root.visits.argmax()) == 1


def test_batched_trees_match_visit_totals():
    trees = [Tree(Board()) for _ in range(4)]
    run_simulations(trees, uniform, 50)
    for tree in trees:
        assert tree.root.total_visits() == 50
        assert np.isclose(tree.policy(1.0).sum(), 1.0)


torch = pytest.importorskip("torch")


def test_numpy_net_matches_torch_with_batchnorm(tmp_path):
    from connect4.train_az import AZNet, export

    torch.manual_seed(0)
    model = AZNet(channels=16, blocks=2)
    # Give batch norm non-trivial statistics so folding is actually tested.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
    path = tmp_path / "az.npz"
    export(model, path)
    rng = np.random.default_rng(0)
    boards = []
    for _ in range(32):
        b = Board()
        for _ in range(int(rng.integers(0, 20))):
            moves = [c for c in b.legal_moves() if not b.is_winning_move(c)]
            if not moves:
                break
            b = b.play(int(rng.choice(moves)))
        boards.append(b)
    x = encode(boards)
    with torch.no_grad():
        logits_t, v_t = model.eval()(torch.from_numpy(x))
    logits_n, v_n = PolicyValueNet(path)(x)
    np.testing.assert_allclose(logits_n, logits_t.numpy(), atol=1e-4)
    np.testing.assert_allclose(v_n, v_t.numpy(), atol=1e-5)


def test_self_play_rows_are_consistent():
    from connect4.train_az import AZNet, self_play

    torch.manual_seed(0)
    rows, results = self_play(AZNet(channels=8, blocks=1), games=3, simulations=8,
                              rng=np.random.default_rng(0))
    assert len(rows) % 2 == 0
    for board, pi, z in rows:
        assert z in (-1.0, 0.0, 1.0)
        assert np.isclose(pi.sum(), 1.0)
        assert all(pi[c] == 0 for c in range(COLS) if not board.can_play(c))
