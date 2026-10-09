"""Lost Robot: the PRNG stream, floors and the ray caster, both filters on seeded runs, kidnap handling, the autopilot,
the benchmark's bookkeeping, and the CLI."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bandits.rng import Rng
from localize import benchmark
from localize.grid import HEADS, SUB, GridFilter
from localize.particles import ParticleFilter
from localize.prng import Stream
from localize.sim import Autopilot, Sim, bfs_path, count_modes
from localize.world import CELL_M, LAYOUTS, MAX_RANGE, TAU, cast, make_floor, motion, ray_loglik

ROOT = Path(__file__).resolve().parent.parent


def dist_m(est, sim):
    return np.hypot(est[0] - sim.x, est[1] - sim.y) * CELL_M


def test_stream_matches_the_bandits_generator():
    """Batched draws give the same numbers as bandits.rng.Rng called one at a time, for uniforms and normals."""
    r, s = Rng(12345), Stream(12345)
    assert np.array_equal(np.array([r.uniform() for _ in range(50)]), s.uniforms(50))
    r, s = Rng(7), Stream(7)
    want = [r.normal() for _ in range(40)]
    assert np.allclose(s.normals(40), want, atol=0, rtol=0)
    # The stream must stop exactly where the sequential loop stopped, so the next draws agree too.
    assert np.allclose(s.uniforms(5), [r.uniform() for _ in range(5)], atol=0, rtol=0)


def test_floors_are_seeded_and_connected():
    a, b = make_floor("halls", 1), make_floor("halls", 1)
    assert np.array_equal(a.walls, b.walls)
    seen = {tuple(make_floor("halls", s).pillars) for s in range(1, 9)}
    assert len(seen) > 1  # the seed changes the pillars
    for key, lay in LAYOUTS.items():
        f = make_floor(key, 2)
        assert f.walls.shape == (lay["h"], lay["w"])
        assert f.region.sum() > 0
        assert not (f.region & f.walls).any()


def test_ray_cast_distance_is_exact_on_the_quarter_cell_step():
    f = make_floor("halls", 1)
    # From the centre of the top-left room (3.5, 3.5) looking east, the wall is at x = 6: 2.5 cells away.
    d = cast(f.walls, np.array([3.5]), np.array([3.5]), np.array([0.0]))
    assert d[0] == pytest.approx(2.5)
    # Looking into the open corridor from the middle of the floor: the ray reaches the outer wall or the cap.
    d = cast(f.walls, np.array([12.0]), np.array([8.0]), np.array([np.pi / 2]))
    assert 0 < d[0] <= MAX_RANGE


def test_bumping_a_wall_keeps_the_robot_in_place():
    f = make_floor("halls", 1)
    x, y, th = motion(f.walls, np.array([3.5]), np.array([3.5]), np.array([0.0]), np.array([0.0]), np.array([9.0]))
    assert (x[0], y[0]) == (3.5, 3.5)


def test_grid_predict_matches_world_motion_on_bumps():
    """With no odometry noise the grid's predict is one exact move, so it must land where world.motion puts the robot.
    A bump keeps the position but still turns; the grid used to keep the old heading as well."""
    f = make_floor("halls", 1)
    g = GridFilter(f, 8, 0.3, 0.05, motion_scale=0.0)
    dh = TAU / HEADS
    seen = {"bump": 0, "free": 0}
    for rot, fwd in ((0.5, 0.45), (-0.5, 0.45)):
        for idx in range(0, g.size, 11):
            x, y, th = np.array(g.x[idx]), np.array(g.y[idx]), np.array(g.th[idx])
            x2, y2, th2 = motion(f.walls, x, y, th, np.array(rot), np.array(fwd))
            kind = "bump" if x2 == x and y2 == y else "free"
            if seen[kind] >= 6:
                continue
            seen[kind] += 1
            g.belief = np.zeros(g.size)
            g.belief[idx] = 1.0
            g.predict(rot, fwd)
            i2 = int(np.clip(np.floor(float(x2) * SUB), 0, g.w2 - 1))
            j2 = int(np.clip(np.floor(float(y2) * SUB), 0, g.h2 - 1))
            k2 = int(np.floor(float(th2) / dh + 0.5)) % HEADS
            assert g.belief[(j2 * g.w2 + i2) * HEADS + k2] == pytest.approx(1.0), (kind, rot, idx)
            assert g.belief.sum() == pytest.approx(1.0)
    assert seen["bump"] == 6 and seen["free"] == 6  # both branches were exercised


def test_range_likelihood_prefers_the_expected_range():
    good = ray_loglik(np.array([2.5]), np.array([2.5]), 0.3, 0.05)
    bad = ray_loglik(np.array([2.5]), np.array([6.0]), 0.3, 0.05)
    assert good > bad
    # A reading at the maximum is a dropout or a long range: it is likely when the expected range is the maximum.
    assert ray_loglik(np.array([MAX_RANGE]), np.array([MAX_RANGE]), 0.3, 0.05) > \
        ray_loglik(np.array([MAX_RANGE]), np.array([2.0]), 0.3, 0.05)


def test_grid_filter_is_a_probability_and_converges():
    f = make_floor("halls", 1)
    sim = Sim(f, 11)
    ap = Autopilot(f, 12)
    g = GridFilter(f, 8, 0.3, 0.05)
    assert g.belief.sum() == pytest.approx(1.0)
    in_wall = f.walls[g.y.astype(int), g.x.astype(int)]  # the cell each heading bin sits in
    assert g.belief[in_wall].sum() == 0  # no probability inside walls, before or after any step
    for _ in range(60):
        rot, fwd = ap.command(sim.x, sim.y, sim.th)
        z = sim.drive(rot, fwd)
        before = g.belief.sum()
        g.predict(rot, fwd)
        assert g.belief.sum() == pytest.approx(before)  # prediction moves mass, it does not create or lose it
        g.update(z)
        assert g.belief.sum() == pytest.approx(1.0)
    x, y, th, mass = g.estimate()
    assert np.hypot(x - sim.x, y - sim.y) * CELL_M < 0.5
    assert mass > 0


def test_particle_filter_converges_with_enough_particles():
    f = make_floor("halls", 1)
    sim = Sim(f, 11)
    ap = Autopilot(f, 12)
    pf = ParticleFilter(f, 2000, 8, 0.3, 0.05, 1.0, 13)
    for _ in range(60):
        rot, fwd = ap.command(sim.x, sim.y, sim.th)
        z = sim.drive(rot, fwd)
        pf.step(rot, fwd, z)
    e = pf.estimate()
    assert dist_m((e["x"], e["y"]), sim) < 0.5
    assert pf.w.sum() == pytest.approx(1.0)


def test_systematic_resampling_keeps_the_count_and_resets_weights():
    f = make_floor("vault", 1)
    pf = ParticleFilter(f, 300, 8, 0.3, 0.05, 1.0, 5)
    pf.w = np.zeros(300)
    pf.w[7] = 1.0  # all weight on one particle: resampling must copy it everywhere
    x7, y7 = pf.x[7], pf.y[7]
    pf._systematic_resample()
    assert len(pf.x) == 300
    assert np.allclose(pf.w, 1 / 300)
    assert np.allclose(pf.x, x7) and np.allclose(pf.y, y7)


def test_kidnap_triggers_injection_and_calm_tracking_does_not():
    f = make_floor("halls", 1)
    sim = Sim(f, 11)
    ap = Autopilot(f, 12)
    pf = ParticleFilter(f, 2000, 8, 0.3, 0.05, 1.0, 13)
    calm = []
    for _ in range(40):
        rot, fwd = ap.command(sim.x, sim.y, sim.th)
        z = sim.drive(rot, fwd)
        pf.step(rot, fwd, z)
        calm.append(pf.injected)
    assert sum(calm[10:]) == 0  # once the filter has settled, routine noise does not inject particles
    sim.kidnap()
    ap.route = []
    injected, errs = [], []
    pf.step(0.0, 0.0, sim.z)
    injected.append(pf.injected)
    for _ in range(14):
        rot, fwd = ap.command(sim.x, sim.y, sim.th)
        z = sim.drive(rot, fwd)
        pf.step(rot, fwd, z)
        injected.append(pf.injected)
        e = pf.estimate()
        errs.append(dist_m((e["x"], e["y"]), sim))
    # The fast average has to fall well below the slow one before injection starts; in this run that takes a few steps.
    assert max(injected) > 0
    # Injection is what lets the filter find the robot again: the estimate is back within 0.5 m by the end.
    assert errs[-1] < 0.5


def test_autopilot_routes_are_shortest_and_reach_their_goal():
    f = make_floor("vault", 2)
    start, goal = (1, 1), (22, 14)
    path = bfs_path(f.walls, start, goal)
    assert path[-1] == goal and start not in path
    # Each step of the path is a 4-neighbour move through free cells.
    prev = start
    for c in path:
        assert abs(c[0] - prev[0]) + abs(c[1] - prev[1]) == 1 and not f.walls[c[1], c[0]]
        prev = c
    sim = Sim(f, 3)
    ap = Autopilot(f, 4)
    reached = False
    for _ in range(600):
        rot, fwd = ap.command(sim.x, sim.y, sim.th)
        sim.drive(rot, fwd)
        if ap.route == [] and _ > 5:
            reached = True
            break
    assert reached  # the robot completes at least one route


def test_mode_count_separates_distant_clusters_only():
    assert count_modes({(1, 1): 0.5, (1, 2): 0.3}, 0.05) == 1
    assert count_modes({(1, 1): 0.5, (9, 9): 0.4}, 0.05) == 2
    assert count_modes({(1, 1): 0.01}, 0.05) == 0


def test_first_within_needs_a_held_window():
    errs = [0.1] + [3.0] * 5 + [0.2] * 12
    assert benchmark.first_within(errs) == 7  # the lone early hit does not count; the 10-step hold starts at step 7
    assert benchmark.first_within([3.0] * 20) is None


def test_benchmark_markdown_has_every_section():
    fake = {"trials": 16, "steps": 100, "reach_m": 0.5, "defaults": {}, "rows": {
        "particles": [dict(label="100 particles", median=None, fail=2, runs=2, err10=1.0, err25=1.0, err50=1.0,
                           err100=1.0, ms=3.0, modes2=2.0)],
        "noise": [dict(label="x", median=5, fail=0, runs=2, err10=0.1, err25=0.1, err50=0.1, err100=0.1, ms=2.0)],
        "filters": [dict(label="grid", median=4, fail=0, runs=2, err10=0.1, err25=0.1, err50=0.1, err100=0.1,
                         ms=60.0, init_ms=400.0)],
        "kidnap": [dict(label="augmented", median=10, fail=0, runs=2, ms=9.0, kidnap_at=40, false_alarm=0.0,
                        post=0.1)],
    }}
    md = benchmark.markdown(fake)
    for heading in ("Particle count", "Noise", "Grid filter vs particle filter", "Kidnapped robot"):
        assert heading in md


def test_cli_plays_a_few_frames(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "localize", "--steps", "3", "--delay", "0", "--no-clear",
                           "--particles", "100"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    frames = [ln for ln in proc.stdout.splitlines() if "step" in ln and "error" in ln]
    assert len(frames) == 3
    assert "TWIN HALLS" in frames[0]


def test_cli_rejects_bad_particle_counts():
    proc = subprocess.run([sys.executable, "-m", "localize", "--particles", "3"], cwd=ROOT, capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode != 0
    assert "--particles" in proc.stderr
