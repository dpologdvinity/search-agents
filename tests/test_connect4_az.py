import json
from collections import deque

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


def test_resolve_device_and_cpu_evaluator():
    from connect4.train_az import AZNet, resolve_device, torch_evaluator

    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        assert resolve_device("auto").type == "cpu"
        with pytest.raises(RuntimeError):
            resolve_device("cuda")
    # amp is a no-op on CPU, so the evaluator still returns normalized priors.
    priors, values = torch_evaluator(AZNet(channels=8, blocks=1), amp=True)([Board(), Board.from_moves("4")])
    assert priors.shape == (2, COLS) and values.shape == (2,)
    assert np.allclose(priors.sum(axis=1), 1.0)


def test_checkpoint_round_trip_with_buffer(tmp_path):
    from connect4.net import PolicyValueNet
    from connect4.train_az import (
        AZNet,
        export,
        load_buffer,
        load_checkpoint,
        save_buffer,
        save_checkpoint,
        self_play,
        train_step,
    )

    model = AZNet(channels=8, blocks=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rows, _ = self_play(model, games=1, simulations=2, rng=np.random.default_rng(0))
    train_step(model, opt, rows[:8])  # gives the optimizer state to save

    ckpt, buf = tmp_path / "az.pt", tmp_path / "az.buffer.npz"
    save_checkpoint(ckpt, model, opt, iteration=7, games=14)
    save_buffer(buf, deque(rows, maxlen=len(rows)))

    model2 = AZNet(channels=8, blocks=1)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    assert load_checkpoint(ckpt, model2, opt2, "cpu") == (7, 14)
    for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
        assert torch.equal(a, b)
    assert torch.equal(opt.state_dict()["state"][0]["exp_avg"], opt2.state_dict()["state"][0]["exp_avg"])

    restored = load_buffer(buf, maxlen=len(rows))
    assert len(restored) == len(rows)
    for (board, pi, z), (board2, pi2, z2) in zip(rows, restored):
        assert board == board2 and board.moves == board2.moves
        assert np.array_equal(pi, pi2) and z == z2
    assert len(load_buffer(buf, maxlen=5)) == 5  # a smaller maxlen keeps the newest rows

    # export runs on a model that has gradients and optimizer state (detach keeps it off the graph).
    export(model, tmp_path / "sub" / "az.npz")
    assert PolicyValueNet(tmp_path / "sub" / "az.npz").blocks == 1


def test_cpu_short_run_resumes_without_losing_progress(tmp_path):
    from connect4.train_az import main

    ckpt, npz, log = tmp_path / "az.pt", tmp_path / "az.npz", tmp_path / "log.jsonl"
    common = ["--device", "cpu", "--threads", "1", "--games", "2", "--simulations", "4", "--steps", "1",
              "--batch", "16", "--channels", "8", "--blocks", "1", "--eval-every", "0", "--save-buffer",
              "--checkpoint", str(ckpt), "--weights-out", str(npz), "--log", str(log)]
    main(["--iterations", "1", *common])
    assert ckpt.exists() and (tmp_path / "az.buffer.npz").exists() and npz.exists()
    # Resume from the same checkpoint: counters continue and the buffer is reloaded, not reset.
    main(["--resume", "--amp", "--iterations", "1", *common])

    state = torch.load(ckpt, weights_only=False)
    assert state["iteration"] == 2 and state["games"] == 4
    first, second = (json.loads(line) for line in log.read_text().splitlines())
    assert (first["iteration"], second["iteration"]) == (1, 2)
    assert second["buffer"] > first["buffer"] and second["device"] == "cpu" and second["amp"] is False

    with pytest.raises(SystemExit):  # --resume with no checkpoint must not silently start over
        main(["--resume", "--iterations", "1", "--device", "cpu",
              "--checkpoint", str(tmp_path / "missing.pt"), "--log", str(log)])


def test_export_records_iteration_and_games(tmp_path):
    from connect4.net import PolicyValueNet
    from connect4.train_az import AZNet, export

    torch.manual_seed(0)
    path = tmp_path / "az.npz"
    export(AZNet(channels=8, blocks=1), path, iteration=7, games=896)
    with np.load(path) as w:
        assert int(w["iteration"]) == 7 and int(w["games"]) == 896
    assert PolicyValueNet(path).blocks == 1  # the serving loader ignores the metadata keys
