import json

import numpy as np

from game2048.expectimax import FEATURES
from game2048.tune import greedy_scores, main


def test_greedy_scores_are_reproducible_and_weight_sensitive():
    w = np.array([1.0, 1.0, 0.5, 1.0, 0.5, 0.2])
    a = greedy_scores(w, games=20, seed=3)
    b = greedy_scores(w, games=20, seed=3)
    assert a.shape == (20,) and (a > 0).all()
    np.testing.assert_array_equal(a, b)
    # Rewarding a full board should play much worse than rewarding empty cells.
    bad = greedy_scores(np.array([-5.0, 0, 0, 0, 0, 0]), games=20, seed=3)
    assert bad.mean() < a.mean()


def test_tuner_keeps_weights_non_negative_and_records_settings(tmp_path):
    # Regression: unbounded sampling let the smoothness penalty get a negative weight, which rewards roughness.
    out = tmp_path / "weights.json"
    main(["--generations", "2", "--population", "6", "--elite", "2", "--games", "10", "--out", str(out)])
    data = json.loads(out.read_text())
    assert set(data["weights"]) == set(FEATURES)
    assert min(data["weights"].values()) >= 0
    assert data["settings"]["population"] == 6 and data["settings"]["elite"] == 2
    assert len(data["generations"]) == 2


def test_evaluate_scores_saved_weights_on_shared_games(tmp_path):
    # Held-out report: each file is scored on the same seed, and the recorded command is kept.
    shipped = tmp_path / "a.json"
    shipped.write_text(json.dumps({"weights": dict(zip(FEATURES, [1.0, 1.0, 0.0, 1.0, 0.5, 0.2])),
                                   "greedy_mean_score": 1.0}))
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({"weights": dict(zip(FEATURES, [-5.0, 0, 0, 0, 0, 0])), "greedy_mean_score": 2.0}))
    out = tmp_path / "report.json"
    main(["--evaluate", str(shipped), str(bad), "--games", "20", "--seed", "3", "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["command"].startswith("python -m game2048.tune --evaluate")
    good_row, bad_row = report["results"]
    assert good_row["mean"] > bad_row["mean"]
    assert good_row["ci95"][0] < good_row["mean"] < good_row["ci95"][1]
    assert good_row["mean"] == greedy_scores(np.array([1.0, 1.0, 0.0, 1.0, 0.5, 0.2]), 20, seed=3).mean()
    assert bad_row["recorded_greedy_mean_score"] == 2.0
    assert out.with_suffix(".md").read_text().startswith("Command:")
