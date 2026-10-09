"""The evaluation-function agent: features on hand-made boards, the choice rule, determinism, committed weights."""

import json

import numpy as np
import pytest

from snake import benchmark, cem
from snake.agents import AGENT_NAMES, make_policy
from snake.board import LEFT, RIGHT, STRAIGHT, Game
from snake.evaluator import FEATURE_NAMES, N_FEATURES, WEIGHTS_PATH, choose, load_weights, move_features, move_table
from snake.evolve import TRAIN_HI, TRAIN_LO


def game_at(body, heading, food, size=12):
    """A game at a chosen position, for rule checks. Body is head first."""
    return Game.from_state(size, body, heading, food)


def test_open_board_straight_move_features():
    # Head (6,6) heading east, food at (0,0). Straight takes the head to (7,6); the tail (4,6) moves away.
    g = game_at([(6, 6), (5, 6), (4, 6)], 1, (0, 0))
    f = move_features(g, STRAIGHT)
    assert len(f) == N_FEATURES == len(FEATURE_NAMES) == 8
    assert f[0] == 0.0  # no apple
    assert f[1] == pytest.approx(1 - 13 / 24)  # Manhattan 7 + 6 = 13 on an open board
    assert f[2] == 1.0 and f[3] == pytest.approx(1.0) and f[4] == 1.0  # everything reachable, room for 3 cells
    assert f[5] == 1.0  # the tail is reachable
    assert f[6] == pytest.approx(3 / 4)  # north, east and south of (7,6) are empty; west is the body
    assert f[7] == pytest.approx(3 / 144)


def test_eating_move_features():
    # Food straight ahead at (7,6): the snake eats, grows to 4, and its tail (4,6) stays in place.
    g = game_at([(6, 6), (5, 6), (4, 6)], 1, (7, 6))
    f = move_features(g, STRAIGHT)
    assert f[0] == 1.0 and f[1] == 1.0 and f[2] == 1.0  # the apple, near food, food reachable
    assert f[3] == pytest.approx(1.0)  # 140 empty cells, all reachable from the head
    assert f[5] == 1.0  # the tail stays put but is a valid escape goal
    assert f[7] == pytest.approx(4 / 144)


def test_a_pocket_has_no_food_no_escape_and_little_room():
    # A 5x5 board. The head moves west from (1,0) onto (0,0). Its only empty neighbour is (0,1), which is walled
    # by the body on its other two sides, so the head's region is that one cell. The tail at (0,3) lies outside
    # the wall, so there is no escape route, and the food at (4,4) is out of reach.
    body = [(1, 0), (1, 1), (1, 2), (0, 2), (0, 3), (0, 4)]
    g = game_at(body, 3, (4, 4), size=5)
    f = move_features(g, STRAIGHT)
    assert f[0] == 0.0 and f[1] == 0.0 and f[2] == 0.0  # no apple, no food route
    assert f[3] == pytest.approx(1 / 19)  # one reachable cell over the 19 empty cells
    assert f[4] == pytest.approx(1 / 6)  # one cell of room over a length of six
    assert f[5] == 0.0  # the tail cannot be reached
    assert f[6] == pytest.approx(1 / 4)  # only (0,1) is an empty neighbour of (0,0)
    assert f[7] == pytest.approx(6 / 25)


def test_move_table_flags_fatal_moves_and_choose_matches_it():
    # Heading east in the bottom-right corner: straight and right both leave the board, so only left is safe.
    g = game_at([(11, 11), (10, 11), (9, 11)], 1, (0, 0))
    rows = {r["action"]: r for r in move_table(g, np.zeros(N_FEATURES))}
    assert rows[STRAIGHT]["dies"] and rows[STRAIGHT]["score"] is None
    assert rows[RIGHT]["dies"] and not rows[LEFT]["dies"]
    w = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    assert choose(g, w) == LEFT


def test_choose_takes_the_highest_weighted_safe_move():
    g = game_at([(6, 6), (5, 6), (4, 6)], 1, (6, 2))  # food straight north of the head
    w = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    assert choose(g, w) == LEFT  # north is a left turn from east, and it is nearer the food


def test_policy_is_a_function_of_the_position_only():
    w = np.array([1.0, 0.5, 0.2, 0.5, 0.3, 0.5, 0.1, 0.0])
    g1, g2 = Game(7), Game(7)
    for _ in range(50):
        assert choose(g1, w) == choose(g2, w)
        if not g1.alive:
            break
        g1.step(choose(g1, w))
        g2.step(choose(g2, w))
    assert g1.body == g2.body and g1.apples == g2.apples


def test_evaluate_weights_is_repeatable():
    w = np.array([1.0, 0.5, 0.2, 0.5, 0.3, 0.5, 0.1, 0.0])
    a = cem.evaluate_weights(w, [1, 2, 3])
    b = cem.evaluate_weights(w, [1, 2, 3])
    assert a == b


def test_cem_is_reproducible_and_worker_count_does_not_change_it(monkeypatch):
    monkeypatch.setattr(cem, "VALIDATION_SEEDS", [40_000, 40_001])  # a short validation keeps the test quick
    base = dict(pop=6, games=2, generations=2, elites=2, seed=4)
    w1, s1, h1, g1 = cem.cem(cem.CEMSettings(workers=1, **base))
    w2, s2, h2, g2 = cem.cem(cem.CEMSettings(workers=2, **base))
    strip = lambda h: [{k: v for k, v in r.items() if k != "elapsed_s"} for r in h]  # wall-clock time differs
    assert np.array_equal(w1, w2) and s1 == s2 and g1 == g2 and strip(h1) == strip(h2)
    assert [r["generation"] for r in h1] == [1, 2]


def test_seed_sets_are_disjoint():
    # Training, validation and the benchmark each use their own boards.
    assert not any(TRAIN_LO <= s < TRAIN_HI for s in cem.VALIDATION_SEEDS)
    assert max(cem.VALIDATION_SEEDS) < benchmark.BENCH_SEED_BASE


def test_committed_weights_load_and_record_their_training():
    assert WEIGHTS_PATH.exists()
    w, meta = load_weights()
    assert w.shape == (N_FEATURES,) and np.all(np.isfinite(w))
    data = json.loads(WEIGHTS_PATH.read_text())
    assert data["features"] == list(FEATURE_NAMES)
    assert len(meta["history"]) == meta["settings"]["generations"]
    assert meta["best_generation"] >= 1
    assert np.allclose(np.round(w, 4), w)  # committed at four decimals, the precision that was validated


def test_evolved_eval_is_a_registered_agent():
    assert "evolved-eval" in AGENT_NAMES
    g = Game(3)
    g.play(make_policy("evolved-eval", seed=3), max_steps=50)
    assert g.steps > 0


def test_bad_weight_file_is_rejected(tmp_path):
    bad = tmp_path / "w.json"
    bad.write_text(json.dumps({"weights": [1.0, 2.0]}))
    with pytest.raises(ValueError):
        load_weights(bad)
