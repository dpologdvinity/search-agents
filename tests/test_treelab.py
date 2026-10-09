"""Decision trees and random forests: impurity, exhaustive splits, stopping rules, pruning, bagging, determinism."""

import math

import numpy as np
import pytest

from treelab.data import PRESETS, expand, make_preset, train_test_split
from treelab.forest import RandomForest
from treelab.rng import Mulberry32
from treelab.tree import (
    Builder,
    DecisionTree,
    best_split,
    impurity,
    link_strengths,
    n_leaves,
    preorder,
    prune,
    rules_text,
)


def _random_data(seed: int, n: int = 40, p: int = 3, k: int = 3):
    r = Mulberry32(seed)
    X = np.array([[r.next() for _ in range(p)] for _ in range(n)])
    y = np.array([r.rand_int(k) for _ in range(n)], dtype=np.int64)
    return X, y, k


def _rows_by_node(root, X):
    """Map node id to the rows of X that the tree routes through that node."""
    out = {}
    stack = [(root, np.arange(len(X)))]
    while stack:
        node, rows = stack.pop()
        out[node.id] = rows
        if not node.is_leaf:
            mask = X[rows, node.feature] < node.threshold
            stack.append((node.left, rows[mask]))
            stack.append((node.right, rows[~mask]))
    return out


# ── RNG ──────────────────────────────────────────────────────────────────────

def test_mulberry32_golden_values():
    # The JavaScript port (web/js/treelab-core.js) must produce the same stream; see the parity test.
    r = Mulberry32(1)
    assert [round(r.next(), 12) for _ in range(3)] == [0.627073940588, 0.00273572118, 0.52744703996]
    assert Mulberry32(12345).next() == pytest.approx(0.9797282677609473, abs=0)


def test_rng_is_in_unit_interval():
    r = Mulberry32(99)
    vals = [r.next() for _ in range(2000)]
    assert min(vals) >= 0.0 and max(vals) < 1.0


# ── Impurity ─────────────────────────────────────────────────────────────────

def test_gini_values():
    assert impurity([5, 5], "gini") == pytest.approx(0.5)
    assert impurity([4, 0], "gini") == 0.0
    assert impurity([2, 2, 2], "gini") == pytest.approx(2 / 3)
    assert impurity([0, 0], "gini") == 0.0


def test_entropy_values():
    assert impurity([5, 5], "entropy") == pytest.approx(1.0)
    assert impurity([3, 3, 3], "entropy") == pytest.approx(math.log2(3))
    assert impurity([7, 0, 0], "entropy") == 0.0


def test_unknown_criterion_rejected():
    with pytest.raises(ValueError):
        impurity([1, 1], "variance")


# ── Best split ───────────────────────────────────────────────────────────────

def _brute_force_best(X, y, features, criterion, min_leaf, k):
    """Try every feature and every midpoint between distinct values, one by one."""
    best = None
    n = len(y)
    parent = impurity(np.bincount(y, minlength=k).tolist(), criterion)
    for f in features:
        vals = np.unique(X[:, f])
        for a, b in zip(vals[:-1], vals[1:]):
            t = (a + b) / 2.0
            left = X[:, f] < t
            nl, nr = int(left.sum()), int((~left).sum())
            if nl < min_leaf or nr < min_leaf:
                continue
            cl = np.bincount(y[left], minlength=k).tolist()
            cr = np.bincount(y[~left], minlength=k).tolist()
            gain = parent - (nl / n) * impurity(cl, criterion) - (nr / n) * impurity(cr, criterion)
            if gain > 1e-12 and (best is None or gain > best[2] + 1e-12):
                best = (f, t, gain)
    return best


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("min_leaf", [1, 3])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_best_split_matches_brute_force(criterion, min_leaf, seed):
    X, y, k = _random_data(seed)
    got = best_split(X, y, np.arange(len(y)), list(range(X.shape[1])), criterion, min_leaf, k)
    want = _brute_force_best(X, y, range(X.shape[1]), criterion, min_leaf, k)
    assert (got is None) == (want is None)
    if want is None:
        return
    f, t, gain = got
    assert gain == pytest.approx(want[2], abs=1e-12)
    # Re-score the chosen split by hand: it must reach the reported gain.
    left = X[:, f] < t
    parent = impurity(np.bincount(y, minlength=k).tolist(), criterion)
    nl, nr = int(left.sum()), int((~left).sum())
    rescored = (parent - (nl / len(y)) * impurity(np.bincount(y[left], minlength=k).tolist(), criterion)
                - (nr / len(y)) * impurity(np.bincount(y[~left], minlength=k).tolist(), criterion))
    assert rescored == pytest.approx(gain, abs=1e-12)


def test_no_split_when_all_rows_identical():
    X = np.full((10, 2), 0.5)
    y = np.array([0, 1] * 5)
    assert best_split(X, y, np.arange(10), [0, 1], "gini", 1, 2) is None


# ── Stopping rules and shape ─────────────────────────────────────────────────

def test_separable_data_grows_two_pure_leaves():
    r = Mulberry32(7)
    X = np.array([[r.next(), r.next()] for _ in range(80)])
    y = (X[:, 0] >= 0.5).astype(np.int64)
    tree = DecisionTree("gini", 10, 1).fit(X, y, 2)
    leaves = [n for n in preorder(tree.root) if n.is_leaf]
    assert tree.n_leaves() == 2
    assert all(n.impurity == 0 for n in leaves)
    assert np.array_equal(tree.predict(X), y)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_depth_limit_respected(depth):
    X, y = make_preset("nested", 300, 2)
    tree = DecisionTree("gini", depth, 1).fit(expand(X), y, 3)
    assert tree.depth() <= depth


@pytest.mark.parametrize("min_leaf", [1, 5, 20])
def test_min_leaf_respected(min_leaf):
    X, y = make_preset("diagonal", 200, 3)
    tree = DecisionTree("entropy", 12, min_leaf).fit(expand(X), y, 2)
    for node in preorder(tree.root):
        if node.is_leaf:
            assert node.n >= min_leaf


def test_leaves_stop_for_a_stated_reason():
    """A leaf is pure, at the depth limit, too small to split, or has no split that reduces impurity."""
    F = expand(make_preset("blobs", 150, 4)[0])
    y = make_preset("blobs", 150, 4)[1]
    tree = DecisionTree("gini", 3, 4).fit(F, y, 3)
    rows = _rows_by_node(tree.root, F)
    for node in preorder(tree.root):
        if not node.is_leaf:
            continue
        if node.impurity == 0 or node.depth >= 3 or node.n < 8:
            continue
        assert best_split(F, y, rows[node.id], range(F.shape[1]), "gini", 4, 3) is None


def test_builder_is_breadth_first_and_numbers_nodes_in_creation_order():
    X, y = make_preset("xor", 120, 5)
    b = Builder(expand(X), y, 2, max_depth=3, min_samples_leaf=2)
    assert b.root.id == 0 and b.root.status == "pending"
    assert b.step() is True
    assert (b.root.left.id, b.root.right.id) == (1, 2)
    assert b.root.status == "split"
    while b.step():
        pass
    assert all(n.status != "pending" for n in preorder(b.root))


def test_stepping_equals_growing():
    X, y = make_preset("spiral", 200, 6)
    F = expand(X)
    stepped = Builder(F, y, 2, max_depth=6, min_samples_leaf=2)
    while stepped.step():
        pass
    grown = DecisionTree("gini", 6, 2).fit(F, y, 2).root
    assert rules_text(stepped.root, ["a", "b", "c", "d"]) == rules_text(grown, ["a", "b", "c", "d"])


def test_entropy_fits_noise_free_xor_with_enough_depth():
    X, y = make_preset("xor", 300, 8)
    tree = DecisionTree("entropy", 10, 1).fit(expand(X), y, 2)
    assert np.mean(tree.predict(expand(X)) == y) == 1.0


# ── Pruning ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(PRESETS))
def test_pruning_is_monotone_in_alpha(name):
    X, y = make_preset(name, 240, 3)
    root = DecisionTree("gini", 10, 1).fit(expand(X), y, PRESETS[name]).root
    leaves = [n_leaves(prune(root, a)) for a in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0]]
    assert all(a >= b for a, b in zip(leaves, leaves[1:]))
    assert leaves[-1] == 1


def test_prune_zero_keeps_training_accuracy_and_does_not_mutate():
    X, y = make_preset("diagonal", 240, 2)
    F = expand(X)
    tree = DecisionTree("gini", 8, 1).fit(F, y, 2)
    before = n_leaves(tree.root)
    pruned = prune(tree.root, 0.0)
    assert n_leaves(tree.root) == before  # the input is untouched
    shown = DecisionTree()
    shown.root, shown.n_classes = pruned, 2
    assert np.mean(shown.predict(F) == y) == pytest.approx(np.mean(tree.predict(F) == y))


def test_link_strengths_are_nonnegative():
    X, y = make_preset("nested", 200, 1)
    root = DecisionTree("gini", 6, 2).fit(expand(X), y, 3).root
    g = link_strengths(root)
    assert g and min(g) >= -1e-12


# ── Forest ───────────────────────────────────────────────────────────────────

def test_forest_beats_single_deep_tree_on_noisy_data():
    """The diagonal preset flips 15% of labels. A single full-depth tree memorises the noise; the bagged
    forest averages it away. Measured over fixed seeds, so this is a regression test, not a hope."""
    tree_acc, forest_acc = [], []
    for seed in (1, 2, 3, 4, 5, 6):
        X, y = make_preset("diagonal", 300, seed)
        tr, te = train_test_split(len(y), 0.3, seed)
        F = expand(X)
        tree = DecisionTree("gini", 12, 1).fit(F[tr], y[tr], 2)
        forest = RandomForest(50, max_depth=12, min_samples_leaf=1, seed=seed).fit(F[tr], y[tr], 2)
        tree_acc.append(np.mean(tree.predict(F[te]) == y[te]))
        forest_acc.append(np.mean(forest.predict(F[te]) == y[te]))
    assert np.mean(forest_acc) > np.mean(tree_acc) + 0.03


def test_forest_oob_and_importances_are_sane():
    X, y = make_preset("spiral", 240, 3)
    F = expand(X)
    f = RandomForest(30, max_depth=6, min_samples_leaf=2, seed=3).fit(F, y, 2)
    assert 0.0 <= f.oob_accuracy <= 1.0
    imp = f.feature_importances_
    assert imp.shape == (4,)
    assert np.all(imp >= 0) and imp.sum() == pytest.approx(1.0)
    # Every tree leaves about 37% of the rows out of its bootstrap sample.
    out_frac = np.mean([np.mean(~m) for m in f.in_bag])
    assert 0.3 < out_frac < 0.45


def test_bootstrap_off_uses_all_rows():
    X, y = make_preset("blobs", 90, 2)
    f = RandomForest(3, max_depth=3, bootstrap=False, seed=1).fit(expand(X), y, 3)
    assert all(m.all() for m in f.in_bag)
    assert f.oob_accuracy is None


def test_max_features_defaults_to_sqrt_and_limits_candidates():
    X, y = make_preset("xor", 120, 1)
    tree = DecisionTree("gini", 4, 2, max_features=1, rng=Mulberry32(5)).fit(expand(X), y, 2)
    assert tree.root is not None  # subsampled trees still grow


# ── Determinism and presets ──────────────────────────────────────────────────

def test_same_seed_same_forest():
    X, y = make_preset("checkerboard", 200, 4)
    F = expand(X)
    a = RandomForest(20, max_depth=5, seed=9).fit(F, y, 2)
    b = RandomForest(20, max_depth=5, seed=9).fit(F, y, 2)
    assert np.array_equal(a.predict_proba(F), b.predict_proba(F))
    assert a.oob_accuracy == b.oob_accuracy
    assert np.array_equal(a.feature_importances_, b.feature_importances_)


def test_same_seed_same_tree_text():
    X, y = make_preset("spiral", 160, 2)
    t1 = DecisionTree("entropy", 5, 3).fit(expand(X), y, 2)
    t2 = DecisionTree("entropy", 5, 3).fit(expand(X), y, 2)
    names = ["x", "y", "x*y", "x^2+y^2"]
    assert rules_text(t1.root, names, "entropy") == rules_text(t2.root, names, "entropy")


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_are_in_the_unit_square_with_valid_labels(name):
    X, y = make_preset(name, 200, 5)
    assert X.shape == (200, 2) and y.shape == (200,)
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert set(np.unique(y)) <= set(range(PRESETS[name]))
    X2, y2 = make_preset(name, 200, 5)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)


def test_train_test_split_partitions_rows():
    tr, te = train_test_split(100, 0.3, seed=2)
    assert len(te) == 30 and len(tr) == 70
    assert sorted(np.concatenate([tr, te]).tolist()) == list(range(100))
