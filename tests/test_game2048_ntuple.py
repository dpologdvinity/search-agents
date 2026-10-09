import json
import random

import numpy as np
import pytest

from game2048.batch import move_batch, new_games, spawn_batch, transpose_batch
from game2048.board import empty_cells, from_grid, move, to_grid, transpose
from game2048.ntuple import PATTERNS, SYMMETRIES, NTupleNetwork, canonical


def random_boards(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        grid = [[rng.choice([0, 0, 2, 4, 8, 16, 32, 64, 128]) for _ in range(4)] for _ in range(4)]
        out.append(from_grid(grid))
    return out


def test_batch_moves_match_scalar():
    boards = random_boards(300, 0)
    arr = np.array(boards, dtype=np.uint64)
    for d in range(4):
        new, score = move_batch(arr, d)
        for b, nb, s in zip(boards, new.tolist(), score.tolist()):
            assert (nb, s) == move(b, d)
    assert transpose_batch(arr).tolist() == [transpose(b) for b in boards]


def test_spawn_batch_adds_one_tile():
    rng = np.random.default_rng(0)
    boards = np.array(random_boards(300, 1), dtype=np.uint64)
    spawned = spawn_batch(boards, rng)
    for before, after in zip(boards.tolist(), spawned.tolist()):
        if empty_cells(before):
            assert len(empty_cells(after)) == len(empty_cells(before)) - 1
            assert bin(before ^ after).count("1") in (1, 2)  # a 2 (0b0001) or 4 (0b0010)
        else:
            assert after == before
    assert all(len(empty_cells(b)) == 14 for b in new_games(50, rng).tolist())


def test_patterns_are_valid_cell_sets():
    for pattern in PATTERNS:
        assert len(pattern) in (5, 6)
        assert len(set(pattern)) == len(pattern)
        assert all(0 <= c < 16 for c in pattern)


def test_patterns_have_no_symmetric_duplicates():
    # Two patterns that are rotations or reflections of each other share one table entry per
    # board, so the second adds no information. Computed from the 8 symmetries, not listed.
    shapes = [canonical(p) for p in PATTERNS]
    assert len(set(shapes)) == len(PATTERNS), shapes


@pytest.fixture
def random_net():
    rng = np.random.default_rng(0)
    net = NTupleNetwork()
    for table in net.tables:
        table[:] = rng.normal(size=table.shape).astype(np.float32)
    return net


def test_scalar_and_batch_values_agree(random_net):
    boards = random_boards(50, 2)
    batch = random_net.value_batch(np.array(boards, dtype=np.uint64))
    for b, v in zip(boards, batch):
        assert random_net.value(b) == pytest.approx(v, rel=1e-5)


def test_value_is_invariant_under_symmetry(random_net):
    for b in random_boards(20, 3):
        grid = to_grid(b)
        rotated = from_grid([list(r) for r in zip(*grid[::-1])])   # 90 degrees clockwise
        mirrored = from_grid([row[::-1] for row in grid])
        v = random_net.value(b)
        assert random_net.value(rotated) == pytest.approx(v, rel=1e-5)
        assert random_net.value(mirrored) == pytest.approx(v, rel=1e-5)


def test_save_and_load_round_trip(tmp_path, random_net):
    path = tmp_path / "net.npz"
    random_net.save(path)
    loaded = NTupleNetwork.load(path)
    assert loaded.patterns == PATTERNS
    assert [t.dtype for t in loaded.tables] == [np.float16] * len(PATTERNS)
    for b in random_boards(10, 4):
        # float16 storage rounds each table entry; the sums stay close.
        assert loaded.value(b) == pytest.approx(random_net.value(b), rel=2e-3, abs=1.0)


def test_short_training_run_learns_something(tmp_path):
    from game2048 import train_td

    out = tmp_path / "w.npz"
    train_td.main(["--minutes", "0.05", "--games", "64", "--out", str(out),
                   "--log", str(tmp_path / "log.jsonl")])
    loaded = NTupleNetwork.load(out)
    assert any(t.any() for t in loaded.tables)
    # The run ends with a record of the games finished so far, so the log is never missing its last games.
    records = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert records and records[-1]["games"] > 0


def test_legacy_npz_layout_loads(tmp_path):
    """The first committed network stored one (patterns, 16 ** 5) array plus the pattern cells."""
    rng = np.random.default_rng(0)
    patterns = np.array(PATTERNS[:4], dtype=np.int8)  # four 5-cell patterns, as in the legacy layout
    weights = rng.normal(size=(len(patterns), 16 ** 5)).astype(np.float16)
    path = tmp_path / "legacy.npz"
    np.savez(path, patterns=patterns, weights=weights)
    loaded = NTupleNetwork.load(path)
    assert loaded.patterns == tuple(tuple(int(c) for c in row) for row in patterns)
    assert [t.dtype for t in loaded.tables] == [np.float16] * len(patterns)
    for got, want in zip(loaded.tables, weights):
        assert np.array_equal(got, want)


def test_batched_td_update_does_not_scale_with_batch_size():
    # Regression: summing updates for weights shared across a batch made
    # common weights take batch-size-times steps and diverge to NaN.
    from game2048.train_td import td_update

    board = from_grid([[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    single, batch = NTupleNetwork(), NTupleNetwork()
    td_update(single, np.array([board], dtype=np.uint64), np.array([10.0]), alpha=0.01)
    td_update(batch, np.array([board] * 500, dtype=np.uint64), np.full(500, 10.0), alpha=0.01)
    for a, b in zip(batch.tables, single.tables):
        np.testing.assert_allclose(a, b)
    assert batch.value(board) > 0


def test_symmetry_maps_are_a_group_of_eight():
    assert len(SYMMETRIES) == 8
    assert len({tuple(m) for m in SYMMETRIES}) == 8
