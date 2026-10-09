"""Clustering lab: the Python reference (clusters/) checked against properties and brute-force references.

Covers the guarantees the page describes: k-means inertia never increases, k-means++ seeds with D^2
probabilities, DBSCAN matches a brute-force union-find reference, EM's log-likelihood never decreases,
runs are deterministic for a seed, silhouette matches a naive loop, and the CLI prints what it promises.
"""

import math

import numpy as np
import pytest

from clusters.__main__ import main
from clusters.data import PRESETS, make_dataset
from clusters.dbscan import NOISE, dbscan, neighbourhoods
from clusters.gmm import ellipse, gmm_em
from clusters.kmeans import assign, elbow, kmeans, kmeans_pp, kmeans_steps, sse
from clusters.metrics import silhouette
from clusters.rng import Mulberry32


def accuracy(labels, truth):
    """Best agreement between found labels and the generating components, maximised over matchings.

    Greedy matching by majority is enough for these presets (each found cluster maps to its best truth id)."""
    total = 0
    for cid in set(labels.tolist()) - {NOISE}:
        members = truth[labels == cid]
        total += np.bincount(members).max()
    return total / len(truth)


# ── rng and presets ──────────────────────────────────────────────────────


def test_mulberry32_reference_values():
    # Fixed outputs for seed 1: a change here means the JavaScript port no longer matches.
    rng = Mulberry32(1)
    first = [rng.random() for _ in range(3)]
    assert first == pytest.approx([0.6270739405881613, 0.002735721180215478, 0.5274470399599522], abs=1e-15)
    assert Mulberry32(1).normal() == pytest.approx(0.9659740590152261, abs=1e-12)


@pytest.mark.parametrize("name", PRESETS)
def test_presets_shape_range_and_determinism(name):
    X, truth = make_dataset(name, seed=3, n=200)
    assert X.shape == (200, 2) and truth.shape == (200,)
    assert X.min() >= 0.0 and X.max() <= 1.0
    X2, truth2 = make_dataset(name, seed=3, n=200)
    assert np.array_equal(X, X2) and np.array_equal(truth, truth2)
    if name != "uniform":
        other, _ = make_dataset(name, seed=4, n=200)
        assert not np.array_equal(X, other)


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError):
        make_dataset("spiral")


# ── k-means ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["blobs", "aniso", "uniform", "smiley"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kmeans_inertia_is_monotone_non_increasing(name, seed):
    X, _ = make_dataset(name, seed=seed, n=180)
    for k in (2, 4, 6):
        res = kmeans(X, k, seed=seed)
        hist = np.array(res.history)
        assert np.all(np.diff(hist) <= 1e-9), (name, seed, k, hist)
        assert res.inertia == pytest.approx(sse(X, res.labels, res.centers))


def test_kmeans_steps_alternate_and_end_converged():
    X, _ = make_dataset("blobs", seed=2, n=120)
    steps = list(kmeans_steps(X, 4, seed=2))
    assert steps[0].phase == "init"
    phases = [s.phase for s in steps[1:]]
    assert phases[0::2] == ["assign"] * len(phases[0::2])
    assert phases[1::2] == ["update"] * len(phases[1::2])
    assert steps[-1].phase == "assign"  # stops right after an assign that changed nothing
    assert np.array_equal(steps[-1].labels, assign(X, steps[-1].centers))


def test_kmeans_centres_override_starts_from_given_points():
    X, _ = make_dataset("blobs", seed=0, n=100)
    start = X[[0, 1, 2, 3]]
    first = next(kmeans_steps(X, 4, centers=start))
    assert np.array_equal(first.centers, start)


def test_kmeans_random_init_uses_data_points():
    X, _ = make_dataset("moons", seed=5, n=80)
    first = next(kmeans_steps(X, 3, seed=5, init="random"))
    for c in first.centers:
        assert np.any(np.all(X == c, axis=1))


def test_kmeans_pp_matches_d2_sampling_distribution():
    # Four points on a line; once the first centre is fixed, the second is drawn with P proportional to D^2.
    X = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    counts = {}
    trials = 6000
    for seed in range(trials):
        rng = Mulberry32(seed)
        C = kmeans_pp(X, 2, rng)
        first = int(np.flatnonzero(np.all(X == C[0], axis=1))[0])
        second = int(np.flatnonzero(np.all(X == C[1], axis=1))[0])
        counts.setdefault(first, [0, 0, 0, 0])[second] += 1
    # Condition on the first centre being the origin: D^2 to the others is 1, 9, 100.
    row = counts[0]
    total = sum(row)
    expected = {1: 1 / 110, 2: 9 / 110, 3: 100 / 110}
    assert row[0] == 0
    for j, p in expected.items():
        assert row[j] / total == pytest.approx(p, abs=0.03)


def test_kmeans_pp_gives_distinct_centres_on_blobs():
    X, _ = make_dataset("blobs", seed=7, n=200)
    C = kmeans_pp(X, 4, Mulberry32(7))
    assert len({tuple(np.round(c, 12)) for c in C}) == 4


def test_kmeans_is_deterministic_for_a_seed():
    X, _ = make_dataset("aniso", seed=1, n=150)
    a = kmeans(X, 3, seed=9)
    b = kmeans(X, 3, seed=9)
    assert np.array_equal(a.labels, b.labels) and np.array_equal(a.centers, b.centers)
    assert a.history == b.history


def test_elbow_is_decreasing_and_has_a_bend_on_blobs():
    X, _ = make_dataset("blobs", seed=2, n=300)
    curve = elbow(X, k_max=8, seed=2)
    inertias = [v for _, v in curve]
    assert [k for k, _ in curve] == list(range(1, 9))
    assert all(a >= b - 1e-9 for a, b in zip(inertias, inertias[1:], strict=False))
    # The big drops stop around the true count of four.
    drop_before = inertias[0] - inertias[3]
    drop_after = inertias[3] - inertias[4]
    assert drop_before > 5 * drop_after


# ── DBSCAN ───────────────────────────────────────────────────────────────


def brute_dbscan(X, eps, min_pts):
    """Reference DBSCAN: every neighbourhood by brute force, cores joined by union-find.

    Cluster ids follow the lowest core index of each component, and a border point takes the cluster of its
    lowest-index core neighbour, the same rules as clusters.dbscan."""
    n = len(X)
    nb = []
    for i in range(n):
        row = []
        for j in range(n):
            dx, dy = X[i][0] - X[j][0], X[i][1] - X[j][1]
            if dx * dx + dy * dy <= eps * eps:
                row.append(j)
        nb.append(row)
    core = [len(r) >= min_pts for r in nb]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        if core[i]:
            for j in nb[i]:
                if core[j]:
                    parent[find(i)] = find(j)
    labels = [NOISE] * n
    ids = {}
    for i in range(n):
        if core[i]:
            r = find(i)
            if r not in ids:
                ids[r] = len(ids)
            labels[i] = ids[r]
    for i in range(n):
        if not core[i]:
            cores = [j for j in nb[i] if core[j]]
            if cores:
                labels[i] = labels[min(cores)]
    return np.array(labels), np.array(core)


@pytest.mark.parametrize("name", ["blobs", "rings", "moons", "smiley", "uniform"])
@pytest.mark.parametrize("eps,min_pts", [(0.05, 4), (0.09, 5), (0.15, 3)])
def test_dbscan_matches_brute_force(name, eps, min_pts):
    X, _ = make_dataset(name, seed=4, n=90)
    labels, core = dbscan(X, eps, min_pts)
    ref_labels, ref_core = brute_dbscan(X, eps, min_pts)
    assert np.array_equal(core, ref_core)
    assert np.array_equal(labels, ref_labels)


def test_dbscan_random_small_sets_match_brute_force():
    for seed in range(12):
        rng = Mulberry32(100 + seed)
        X = np.array([[rng.random(), rng.random()] for _ in range(50)])
        labels, core = dbscan(X, 0.12, 3)
        ref_labels, ref_core = brute_dbscan(X, 0.12, 3)
        assert np.array_equal(labels, ref_labels) and np.array_equal(core, ref_core)


def test_dbscan_separates_rings_where_kmeans_cannot():
    X, truth = make_dataset("rings", seed=0, n=300)
    labels, _ = dbscan(X, 0.09, 5)  # 0.07 splits the outer ring into arcs: the eps slider matters
    assert len(set(labels.tolist()) - {NOISE}) == 2
    assert accuracy(labels, truth) > 0.95
    km = kmeans(X, 2, seed=0)
    assert accuracy(km.labels, truth) < 0.8


def test_dbscan_marks_isolated_points_as_noise():
    X = np.array([[0.1, 0.1], [0.11, 0.1], [0.1, 0.11], [0.9, 0.9]])
    labels, core = dbscan(X, 0.02, 3)
    assert labels.tolist() == [0, 0, 0, NOISE]
    assert core.tolist() == [True, True, True, False]


def test_dbscan_neighbourhoods_include_self():
    X = np.array([[0.0, 0.0], [0.5, 0.5]])
    assert neighbourhoods(X, 0.1) == [[0], [1]]


def test_dbscan_rejects_negative_eps():
    X = np.array([[0.0, 0.0], [0.5, 0.5]])
    with pytest.raises(ValueError):
        neighbourhoods(X, -0.1)
    with pytest.raises(ValueError):
        dbscan(X, float("nan"), 2)


# ── Gaussian mixture (EM) ───────────────────────────────────────────────


@pytest.mark.parametrize("name,k", [("blobs", 4), ("aniso", 3), ("moons", 2), ("smiley", 4)])
def test_em_loglikelihood_is_non_decreasing(name, k):
    X, _ = make_dataset(name, seed=2, n=200)
    res = gmm_em(X, k, seed=2)
    hist = np.array(res.history)
    assert len(hist) >= 2
    assert np.all(np.diff(hist) >= -1e-7), hist  # per-point average; differences are rounding only


def test_em_recovers_anisotropic_blobs():
    X, truth = make_dataset("aniso", seed=1, n=300)
    res = gmm_em(X, 3, seed=1)
    assert res.converged
    assert accuracy(res.labels, truth) > 0.9
    assert res.resp.shape == (300, 3)
    assert np.allclose(res.resp.sum(axis=1), 1.0)
    assert res.weights.sum() == pytest.approx(1.0)


def _posterior(X, res):
    """Responsibilities and average log-likelihood of X under the weights, means and covariances in res,
    written with np.linalg rather than the module's own helpers, so it is an independent check."""
    logp = []
    for w, mu, S in zip(res.weights, res.means, res.covs, strict=True):
        diff = X - mu
        maha = np.einsum("ni,ij,nj->n", diff, np.linalg.inv(S), diff)
        logp.append(np.log(w) - 0.5 * (2 * math.log(2 * math.pi) + math.log(np.linalg.det(S)) + maha))
    logp = np.stack(logp, axis=1)
    m = logp.max(axis=1, keepdims=True)
    lognorm = m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))
    return np.exp(logp - lognorm[:, None]), float(lognorm.mean())


@pytest.mark.parametrize("max_iter", [1, 2, 3])
def test_em_result_is_consistent_when_max_iter_runs_out(max_iter):
    # Cut short by max_iter, the returned parameters must be the ones that gave resp, labels and loglik.
    # An M step after the last E step used to leave them one update ahead.
    X, _ = make_dataset("aniso", seed=0, n=200)
    res = gmm_em(X, 3, seed=0, max_iter=max_iter)
    assert res.iterations == max_iter and not res.converged
    resp, loglik = _posterior(X, res)
    assert np.allclose(res.resp, resp, atol=1e-10)
    assert np.array_equal(res.labels, np.argmax(resp, axis=1))
    assert res.loglik == pytest.approx(loglik, abs=1e-10)
    assert res.history[-1] == pytest.approx(loglik, abs=1e-10)
    assert len(res.history) == max_iter


def test_em_is_deterministic_and_ellipses_are_sane():
    X, _ = make_dataset("blobs", seed=6, n=160)
    a = gmm_em(X, 4, seed=6)
    b = gmm_em(X, 4, seed=6)
    assert np.array_equal(a.labels, b.labels) and a.loglik == b.loglik
    major, minor, angle = ellipse(a.covs[0], sigma=2.0)
    assert major >= minor >= 0.0
    assert -math.pi / 2 <= angle <= math.pi / 2


# ── silhouette ──────────────────────────────────────────────────────────


def naive_silhouette(X, labels):
    """Loop-only silhouette, the textbook definition, to check the vectorised version."""
    ids = sorted(set(labels.tolist()) - {NOISE})
    keep = [i for i in range(len(X)) if labels[i] != NOISE]
    scores = []
    for i in keep:
        same = [j for j in keep if labels[j] == labels[i] and j != i]
        if not same:
            scores.append(0.0)
            continue
        a = sum(math.dist(X[i], X[j]) for j in same) / len(same)
        b = min(
            sum(math.dist(X[i], X[j]) for j in keep if labels[j] == c) / sum(1 for j in keep if labels[j] == c)
            for c in ids
            if c != labels[i]
        )
        scores.append((b - a) / max(a, b))
    return sum(scores) / len(keep)


def test_silhouette_matches_naive_definition():
    X, _ = make_dataset("blobs", seed=3, n=60)
    labels = kmeans(X, 3, seed=3).labels
    assert silhouette(X, labels) == pytest.approx(naive_silhouette(X, labels))
    noisy = labels.copy()
    noisy[:5] = NOISE
    assert silhouette(X, noisy) == pytest.approx(naive_silhouette(X, noisy))


def test_silhouette_needs_two_clusters():
    X, _ = make_dataset("blobs", seed=0, n=40)
    assert silhouette(X, np.zeros(len(X), dtype=int)) is None


# ── command line ────────────────────────────────────────────────────────


def test_cli_run_prints_summary_and_scatter(capsys):
    assert main(["run", "--algo", "kmeans", "--k", "4", "--data", "blobs", "--seed", "2", "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "iterations" in out and "inertia" in out and "silhouette" in out
    assert "\x1b[" not in out
    assert "0" in out and "3" in out  # four clusters drawn with digits


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--k", "0"],
        ["run", "--n", "0"],
        ["run", "--width", "0"],
        ["run", "--max-iter", "0"],
        ["run", "--min-pts", "0"],
        ["run", "--eps", "-0.1"],
        ["elbow", "--k-max", "0"],
    ],
)
def test_cli_rejects_sizes_that_cannot_run(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "must be" in capsys.readouterr().err


def test_cli_run_dbscan_and_gmm(capsys):
    assert main(["run", "--algo", "dbscan", "--eps", "0.06", "--min-pts", "5", "--data", "rings"]) == 0
    assert "noise" in capsys.readouterr().out
    assert main(["run", "--algo", "gmm", "--k", "3", "--data", "aniso", "--seed", "1"]) == 0
    assert "log-likelihood" in capsys.readouterr().out


def test_cli_elbow_prints_ten_rows(capsys):
    assert main(["elbow", "--data", "blobs", "--seed", "2", "--k-max", "10"]) == 0
    rows = [line for line in capsys.readouterr().out.splitlines() if line.strip()[:1].isdigit()]
    assert len(rows) == 10
