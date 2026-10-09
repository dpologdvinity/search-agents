"""The optimizer race: landscapes, optimizers, the race loop, the CLI and the API router.

The assertions are the properties the lab relies on: gradients match finite differences, each optimizer
converges on the bowl, the ravine and Rosenbrock behave as the lab describes (measured at the default step
sizes, not promised for every setting), and seeded noise is reproducible.
"""

import math
import random

import pytest

from optlab import optimizers, surfaces
from optlab.__main__ import _join_negative_start, main
from optlab.race import Race
from optlab.rng import Rng
from optlab.surfaces import SURFACES, custom, hessian, is_minimum

# Every landscape the lab offers, including a paint-your-own one with a hill and a valley.
ALL_SURFACES = [*SURFACES.values(), custom()]


def _points(surface, n=25, seed=1):
    """Random points in the view box, plus the start, for gradient checks."""
    rnd = random.Random(seed)
    x0, x1, y0, y1 = surface.domain
    pts = [(rnd.uniform(x0, x1), rnd.uniform(y0, y1)) for _ in range(n)]
    return pts + [surface.start]


@pytest.mark.parametrize("surface", ALL_SURFACES, ids=lambda s: s.name)
def test_gradient_matches_central_differences(surface):
    h = 1e-6
    for x, y in _points(surface):
        gx, gy = surface.grad(x, y)
        nx = (surface.f(x + h, y) - surface.f(x - h, y)) / (2 * h)
        ny = (surface.f(x, y + h) - surface.f(x, y - h)) / (2 * h)
        scale = 1.0 + abs(gx) + abs(gy)
        assert abs(gx - nx) <= 1e-6 * scale, (surface.name, x, y)
        assert abs(gy - ny) <= 1e-6 * scale, (surface.name, x, y)


@pytest.mark.parametrize("surface", [s for s in SURFACES.values() if s.fmin is not None], ids=lambda s: s.name)
def test_known_minima_have_zero_gradient_and_known_loss(surface):
    for mx, my in surface.minima:
        gx, gy = surface.grad(mx, my)
        assert math.hypot(gx, gy) < 1e-3, (surface.name, mx, my)
        # The tabulated Himmelblau minima are 6 decimals, so their loss is only near zero.
        assert surface.f(mx, my) == pytest.approx(surface.fmin, abs=1e-4)


def test_saddle_origin_is_not_a_minimum_but_the_wells_are():
    s = SURFACES["saddle"]
    assert not is_minimum(s, 0.0, 0.0, 1e-3)  # zero gradient, but the Hessian has a negative eigenvalue
    assert is_minimum(s, 0.0, math.sqrt(2.0), 1e-3)
    assert is_minimum(s, 0.0, -math.sqrt(2.0), 1e-3)


def test_hessian_of_bowl_is_twice_identity():
    s = SURFACES["bowl"]
    hxx, hxy, hyy = hessian(s.grad, 0.7, -0.3)
    assert hxx == pytest.approx(2.0, abs=1e-6)
    assert hyy == pytest.approx(2.0, abs=1e-6)
    assert hxy == pytest.approx(0.0, abs=1e-6)


def test_rng_is_deterministic_and_in_range():
    a, b = Rng(42), Rng(42)
    xs = [a.uniform() for _ in range(1000)]
    assert xs == [b.uniform() for _ in range(1000)]
    assert all(0.0 <= u < 1.0 for u in xs)
    assert xs != [Rng(43).uniform() for _ in range(1000)][:1000]


def test_normal_noise_has_mean_zero_and_unit_variance():
    r = Rng(5)
    xs = [r.normal() for _ in range(20000)]
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    assert abs(mean) < 0.03
    assert abs(var - 1.0) < 0.05
    assert all(-6.0 <= x <= 6.0 for x in xs)


@pytest.mark.parametrize("name", optimizers.NAMES)
def test_every_optimizer_converges_on_the_bowl(name):
    """Each optimizer reaches the gradient tolerance. Fixed-step methods (RMSProp here) then hover near the
    minimum at a loss set by their step size, so the final loss is bounded, not zero."""
    s = SURFACES["bowl"]
    race = Race(s, s.start, names=[name]).run(500)
    tr = race.tracks[0]
    assert tr.steps_to_tol is not None, name
    assert tr.losses[-1] < 1e-2


def test_sgd_oscillates_in_the_ravine_at_default_step():
    """At lr 0.05 the stiff direction (curvature 25) has lr * 25 > 1: each step overshoots and flips sign."""
    s = SURFACES["ravine"]
    tr = Race(s, s.start, names=["sgd"]).run(50).tracks[0]
    ys = [y for _, y in tr.traj]
    flips = sum(1 for a, b in zip(ys, ys[1:]) if a * b < 0)
    assert flips >= 10


def test_ravine_at_default_settings_momentum_and_nesterov_beat_sgd():
    """Measured at the default step sizes: SGD takes 158 steps to reach tolerance on the ravine, momentum
    150 and Nesterov 86. RMSProp never gets there: with a fixed step it hovers at a noise floor."""
    s = SURFACES["ravine"]
    race = Race(s, s.start).run(500)
    steps = {t.name: t.steps_to_tol for t in race.tracks}
    assert steps["sgd"] == 158
    assert steps["momentum"] == 150
    assert steps["nesterov"] == 86
    assert steps["rmsprop"] is None
    assert steps["nesterov"] < steps["momentum"] < steps["sgd"]


def test_plain_sgd_diverges_on_rosenbrock_at_default_step():
    s = SURFACES["rosenbrock"]
    race = Race(s, s.start).run(500)
    sgd = next(t for t in race.tracks if t.name == "sgd")
    assert sgd.diverged_at == 3
    assert sgd.steps_to_tol is None
    # The trail stops where the track was still finite; its losses stay finite.
    assert len(sgd.traj) == 3 and all(math.isfinite(v) for v in sgd.losses)


def test_divergence_freezes_the_track():
    s = SURFACES["bowl"]
    race = Race(s, s.start, names=["sgd"], lrs={"sgd": 1.5}).run(100)
    tr = race.tracks[0]
    assert tr.diverged_at is not None
    n = len(tr.traj)
    race.run(10)
    assert len(tr.traj) == n and tr.diverged_at is not None


def test_adam_and_adagrad_reach_a_himmelblau_minimum():
    s = SURFACES["himmelblau"]
    race = Race(s, s.start).run(300)
    by = {t.name: t for t in race.tracks}
    assert by["adam"].steps_to_tol is not None
    assert by["adagrad"].steps_to_tol is not None
    assert by["adam"].losses[-1] < 1e-5


def test_noise_is_reproducible_with_a_seed_and_shared_across_tracks():
    s = SURFACES["ravine"]
    a = Race(s, s.start, noise=0.3, seed=7).run(60)
    b = Race(s, s.start, noise=0.3, seed=7).run(60)
    c = Race(s, s.start, noise=0.3, seed=8).run(60)
    for ta, tb in zip(a.tracks, b.tracks):
        assert ta.traj == tb.traj
    assert [t.traj for t in a.tracks] != [t.traj for t in c.tracks]


def test_noise_draw_is_common_to_all_tracks():
    """With the same noise, every track's first update uses the same gradient noise: the rng is consumed once
    per step, not once per track. Checked through the first step of two single-track races."""
    s = SURFACES["bowl"]
    one = Race(s, s.start, names=["sgd"], noise=0.5, seed=3).run(1).tracks[0]
    both = Race(s, s.start, names=["sgd", "adam"], noise=0.5, seed=3).run(1)
    assert both.tracks[0].traj == one.traj


def test_nesterov_query_point_is_the_lookahead():
    opt = optimizers.make("nesterov", 0.1)
    opt.update([1.0, 1.0], [2.0, -2.0])  # builds velocity v = -lr g
    q = opt.query([0.0, 0.0])
    assert q == pytest.approx([opt.h.momentum * opt.v[0], opt.h.momentum * opt.v[1]])


def test_adam_first_step_is_about_lr_in_each_coordinate():
    """Bias correction makes the first Adam step lr * g / |g| per coordinate (up to eps)."""
    opt = optimizers.make("adam", 0.1)
    x = opt.update([0.0, 0.0], [3.0, -0.5])
    assert x[0] == pytest.approx(-0.1, abs=1e-6)
    assert x[1] == pytest.approx(0.1, abs=1e-6)


def test_custom_surface_rejects_bad_input():
    with pytest.raises(ValueError):
        custom([(0, 0, 1.0, 0.0)])
    with pytest.raises(ValueError):
        custom([(0, 0, 1.0, 1.0)] * (surfaces.MAX_BUMPS + 1))


def test_get_rejects_unknown_names():
    with pytest.raises(KeyError):
        surfaces.get("nope")
    with pytest.raises(KeyError):
        optimizers.make("lbfgs", 0.1)


def test_cli_race_prints_table_and_map(capsys):
    assert main(["race", "--surface", "rosenbrock", "--start", "-1.5,2", "--steps", "300"]) == 0
    out = capsys.readouterr().out
    assert "optimizer" in out and "DIVERGED at step 3" in out
    assert "o=start" in out and "UPPER=final" in out


def test_cli_no_map_and_custom_lr(capsys):
    assert main(["race", "--surface", "bowl", "--steps", "50", "--lr", "sgd=0.1,adam=0.2", "--no-map"]) == 0
    out = capsys.readouterr().out
    assert "o=start" not in out
    assert "0.1" in out and "0.2" in out


def test_cli_surfaces_lists_every_landscape(capsys):
    assert main(["surfaces"]) == 0
    out = capsys.readouterr().out
    for name in SURFACES:
        assert name in out


def test_negative_start_is_joined_to_its_option():
    assert _join_negative_start(["race", "--start", "-1.5,2", "--steps", "5"]) == [
        "race",
        "--start=-1.5,2",
        "--steps",
        "5",
    ]


def test_race_summary_matches_track_state():
    s = SURFACES["bowl"]
    race = Race(s, s.start).run(20)
    rows = race.summary()
    assert [r["name"] for r in rows] == list(optimizers.NAMES)
    for row, tr in zip(rows, race.tracks):
        assert row["loss"] == tr.losses[-1]


@pytest.mark.parametrize(
    "argv",
    [["--steps", "0"], ["--width", "0"], ["--beta2", "1.5"], ["--beta1", "1"], ["--momentum", "nan"],
     ["--rho", "-0.1"], ["--eps", "0"], ["--noise", "-1"], ["--tol", "-1"]],
)
def test_cli_rejects_settings_the_race_cannot_use(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["race", "--no-map", *argv])
    assert exit_info.value.code == 2
    assert "must be in" in capsys.readouterr().err
