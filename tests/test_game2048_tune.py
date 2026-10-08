import numpy as np

from game2048.tune import greedy_scores


def test_greedy_scores_are_reproducible_and_weight_sensitive():
    w = np.array([1.0, 1.0, 0.5, 1.0, 0.5, 0.2])
    a = greedy_scores(w, games=20, seed=3)
    b = greedy_scores(w, games=20, seed=3)
    assert a.shape == (20,) and (a > 0).all()
    np.testing.assert_array_equal(a, b)
    # Rewarding a full board should play much worse than rewarding empty cells.
    bad = greedy_scores(np.array([-5.0, 0, 0, 0, 0, 0]), games=20, seed=3)
    assert bad.mean() < a.mean()
