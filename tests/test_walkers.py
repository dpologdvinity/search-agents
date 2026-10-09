"""Fast checks for the EVOLVING WALKERS physics and genetic algorithm (walkers/core.py).

Covers the numerics (sine, PRNG), the physics sanity cases (a passive body comes to rest, free springs conserve
momentum), determinism, the genetic operators keeping genomes valid, a known creature's reproducible distance,
the text encoding, the presets file, and the command line. The cross-language check lives in
tests/test_walkers_parity.py.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import walkers.core as core
from walkers import study
from walkers.cli import main
from walkers.core import (
    DEFAULT_WORLD,
    Evolver,
    Mulberry32,
    crossover,
    decode,
    encode,
    is_valid,
    mutate,
    random_genome,
    repair,
    simulate,
    simulate_one,
    wsin,
)

ROOT = Path(__file__).resolve().parent.parent
PRESETS = ROOT / "walkers" / "data" / "champions.json"
WEB_PRESETS = ROOT / "web" / "data" / "walkers-champions.json"


def passive_pair():
    """Two nodes 0.4 m apart joined by a passive spring, dropped from SPAWN_LIFT onto the floor."""
    return {
        "nodes": [{"x": -0.2, "y": 0.0}, {"x": 0.2, "y": 0.0}],
        "springs": [{"a": 0, "b": 1, "rest": 0.3, "k": 120.0, "c": 2.0, "muscle": 0, "amp": 0, "freq": 1, "phase": 0}],
    }


def test_wsin_matches_math_sin():
    xs = np.linspace(-20, 20, 40001)
    worst = max(abs(wsin(float(x)) - math.sin(float(x))) for x in xs)
    assert worst < 1e-10


def test_mulberry32_matches_the_javascript_values():
    # Values printed by node for mulberry32(12345); the JavaScript generator is the reference.
    want = [0.9797282677609473, 0.3067522644996643, 0.484205421525985, 0.817934412509203, 0.5094283693470061]
    rng = Mulberry32(12345)
    assert [rng.next() for _ in range(5)] == want
    rng = Mulberry32(7)
    assert rng.int(10) == 0


def test_passive_body_comes_to_rest():
    """With no muscles, a dropped spring pair loses its energy to friction and damping and stops."""
    g = passive_pair()
    r = simulate([g], DEFAULT_WORLD, 24)[0]
    frames = r["frames"]
    late = frames[-1]
    earlier = frames[-6]  # 0.5 s before the end
    assert np.abs(late - earlier).max() < 0.005
    assert not r["exploded"]


def test_free_springs_conserve_momentum_without_ground_or_drag(monkeypatch):
    """Internal spring forces cancel pairwise, so with gravity 0, no drag and no ground contact the centre of
    mass moves in a straight line: its second difference over the frames is zero."""
    monkeypatch.setattr(core, "AIR_DRAG", 0.0)
    g = {
        "nodes": [{"x": -0.3, "y": 0.0}, {"x": 0.3, "y": 0.1}, {"x": 0.0, "y": 0.35}],
        "springs": [
            {"a": 0, "b": 1, "rest": 0.5, "k": 150.0, "c": 1.0, "muscle": 1, "amp": 0.3, "freq": 2.0, "phase": 0.5},
            {"a": 1, "b": 2, "rest": 0.4, "k": 200.0, "c": 2.0, "muscle": 0, "amp": 0, "freq": 1, "phase": 0},
            {"a": 0, "b": 2, "rest": 0.3, "k": 100.0, "c": 1.0, "muscle": 0, "amp": 0, "freq": 1, "phase": 0},
        ],
    }
    world = {**DEFAULT_WORLD, "gravity": 0.0, "duration": 1.0}
    r = simulate([g], world, 1)[0]
    com = r["frames"].mean(axis=1)  # (frames, 2)
    second_diff = com[2:] - 2 * com[1:-1] + com[:-2]
    assert np.abs(second_diff).max() < 1e-9
    assert not r["fallen"]


def pair_energy(monkeypatch, x0, x1, rest, k, duration):
    """Kinetic plus spring energy, one value per step, of two gravity-free, undamped nodes joined by one spring.

    Velocities are finite differences of the sampled frames (one frame per step), which is accurate to a fraction
    of a percent here. Each node has unit mass, as in the integrator.
    """
    monkeypatch.setattr(core, "AIR_DRAG", 0.0)
    g = {
        "nodes": [{"x": x0, "y": 0.0}, {"x": x1, "y": 0.0}],
        "springs": [{"a": 0, "b": 1, "rest": rest, "k": k, "c": 0.0, "muscle": 0, "amp": 0, "freq": 1, "phase": 0}],
    }
    world = {**DEFAULT_WORLD, "gravity": 0.0, "duration": duration}
    frames = simulate([g], world, 1)[0]["frames"]  # (steps + 1, 2, 2)
    velocity = np.diff(frames, axis=0) / core.DT
    kinetic = 0.5 * (velocity**2).sum(axis=(1, 2))
    stretch = np.linalg.norm(frames[1:, 1] - frames[1:, 0], axis=1) - rest
    return kinetic + 0.5 * k * stretch**2


def test_crossing_nodes_gain_energy_without_contact(monkeypatch):
    """Documented limitation (walkers how-it-works and README): there is no node-to-node contact, so nodes pass
    through each other. When a spring's length passes through zero its force direction flips, and undamped that
    adds energy. Drag damps it but does not prevent it. This pins the review's example: a 1.0 m spring (k 200,
    rest 0.4) released at rest has 35.8 J and peaks at 59.6 J (+67%). If this fails, the physics changed: re-check
    the champions and the 300-generation results, and update the documentation with the new numbers."""
    energy = pair_energy(monkeypatch, -0.5, 0.5, rest=0.4, k=200.0, duration=20.0)
    assert energy[0] == pytest.approx(35.8, rel=0.01)
    assert energy.max() == pytest.approx(59.6, rel=0.01)


def test_springs_that_never_cross_keep_their_energy(monkeypatch):
    """Control for the test above: with rest 0.9 the spring stays between 0.8 m and 1.0 m, never near zero, and its
    energy stays within about 5% of the start (the discrete integrator's small oscillation)."""
    energy = pair_energy(monkeypatch, -0.5, 0.5, rest=0.9, k=200.0, duration=20.0)
    assert np.abs(energy - energy[0]).max() / energy[0] < 0.06


def test_simulation_is_deterministic():
    rng = Mulberry32(4)
    g = random_genome(rng)
    a = simulate_one(g, DEFAULT_WORLD, 8)
    b = simulate_one(g, DEFAULT_WORLD, 8)
    assert a["fitness"] == b["fitness"]
    assert np.array_equal(a["frames"], b["frames"])


def test_same_seed_gives_the_same_evolution():
    histories = []
    for _ in range(2):
        ev = Evolver(seed=11, pop_size=10)
        histories.append([(s["best"], s["species"], s["champion"]["code"]) for s in (ev.step() for _ in range(3))])
    assert histories[0] == histories[1]


def test_genetic_operators_keep_genomes_valid():
    rng = Mulberry32(21)
    parents = [random_genome(rng) for _ in range(20)]
    children = 0
    for i in range(200):
        a = parents[i % len(parents)]
        b = parents[(i * 7 + 3) % len(parents)]
        child = mutate(a, rng, 0.9)
        assert is_valid(child), child
        mixed = crossover(a, b, rng, float(i % 3), float((i + 1) % 3))
        assert is_valid(mixed), mixed
        children += 2
    assert children == 400


def test_repair_connects_a_broken_body():
    g = {
        "nodes": [{"x": 0, "y": 0}, {"x": 0.3, "y": 0}, {"x": 0.9, "y": 0.5}, {"x": 0.9, "y": 0.9}],
        "springs": [{"a": 0, "b": 1, "rest": 0.3, "k": 100, "c": 1, "muscle": 1, "amp": 0.1, "freq": 1, "phase": 0}],
    }
    fixed = repair(g)
    assert is_valid(fixed)
    assert len(fixed["springs"]) == 3


def test_text_encoding_round_trips():
    rng = Mulberry32(8)
    for _ in range(20):
        g = random_genome(rng)
        code = encode(g)
        assert code.startswith("w1|")
        assert encode(decode(code)) == code
        back = decode(code)
        assert back["nodes"] == g["nodes"]
        assert back["springs"] == g["springs"]


def test_decode_rejects_other_text():
    for bad in ("", "w2|0,0|0,1,0.3,100,1,1,0.1,1,0", "w1|0,0;1,0|0,1,0.3"):
        try:
            decode(bad)
        except ValueError:
            continue
        raise AssertionError(f"decode accepted {bad!r}")


def test_a_known_creature_has_a_reproducible_distance():
    """The runner preset's distance, recorded when the study ran, must come back from the simulator."""
    doc = json.loads(PRESETS.read_text())
    runner = next(p for p in doc["presets"] if p["id"] == "runner")
    r = simulate([decode(runner["code"])], {**DEFAULT_WORLD, "terrain": runner["terrain"]}, 0)[0]
    assert abs(r["distance"] - runner["distance"]) < 1e-9


def test_presets_are_valid_and_the_page_copy_matches():
    doc = json.loads(PRESETS.read_text())
    ids = [p["id"] for p in doc["presets"]]
    assert {"runner", "inchworm"} <= set(ids) and set(ids) <= {"runner", "hopper", "inchworm"}
    for p in doc["presets"]:
        assert is_valid(decode(p["code"]))
        assert study.encode_check(p["code"])
    assert json.loads(WEB_PRESETS.read_text()) == doc


def test_evolution_improves_on_random_creatures():
    """A short run already beats the first generation's best on flat ground (a sanity check, not a benchmark)."""
    ev = Evolver(seed=2, pop_size=40)
    first = ev.step()["best"]
    best = first
    for _ in range(12):
        best = max(best, ev.step()["best"])
    assert best > first


def test_cli_evolve_prints_one_line_per_generation(capsys):
    assert main(["evolve", "--generations", "2", "--pop", "10", "--seed", "3"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len([line for line in lines if line.startswith("gen ")]) == 2


def test_cli_replay_no_animate_prints_the_distance(capsys):
    code = encode(passive_pair())
    assert main(["replay", code, "--no-animate"]) == 0
    assert "distance=" in capsys.readouterr().out


def test_cli_replay_animates_when_piped(capsys):
    """Off a terminal the replay prints frames without clearing the screen, and ends with the result line."""
    code = encode(passive_pair())
    assert main(["replay", code]) == 0
    out = capsys.readouterr().out
    assert "\033[" not in out
    assert "-- frame 0" in out and out.rstrip().splitlines()[-1].startswith("distance=")


def test_cli_rejects_a_bad_creature(capsys):
    assert main(["replay", "nonsense", "--no-animate"]) == 2
    assert "error" in capsys.readouterr().err


@pytest.mark.parametrize("rate", ["2", "-0.1", "nan"])
def test_cli_rejects_a_mutation_rate_outside_0_to_1(rate, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["evolve", "--generations", "1", "--mutation", rate])
    assert exit_info.value.code == 2
    assert "between 0 and 1" in capsys.readouterr().err


def test_exploded_creatures_score_zero_and_do_not_spread():
    """A deliberately violent creature (stiff, large amplitude) is frozen with fitness 0 while a calm one in the
    same batch is unaffected. Rows never mix."""
    calm = passive_pair()
    wild = {
        "nodes": [{"x": -0.2, "y": 0.0}, {"x": 0.2, "y": 0.0}, {"x": 0.0, "y": 0.5}],
        "springs": [
            {"a": 0, "b": 1, "rest": 0.1, "k": 600, "c": 0.2, "muscle": 1, "amp": 0.4, "freq": 4.0, "phase": 0},
            {"a": 0, "b": 2, "rest": 0.08, "k": 600, "c": 0.2, "muscle": 1, "amp": 0.4, "freq": 4.0, "phase": 1},
            {"a": 1, "b": 2, "rest": 0.08, "k": 600, "c": 0.2, "muscle": 1, "amp": 0.4, "freq": 4.0, "phase": 2},
        ],
    }
    alone = simulate([calm], DEFAULT_WORLD, 0)[0]
    together = simulate([calm, wild], DEFAULT_WORLD, 0)
    assert together[0]["fitness"] == alone["fitness"]
    assert together[0]["exploded"] is False
    if together[1]["exploded"]:
        assert together[1]["fitness"] == 0.0
    assert np.isfinite(together[1]["fitness"])


def test_mutation_of_a_three_node_fan_never_crashes_or_returns_an_empty_body():
    """Removing the hub of a 3-node fan can leave no springs before repair; every seed must still give a valid body."""
    fan = {
        "nodes": [{"x": 0.0, "y": 0.0}, {"x": 0.3, "y": 0.0}, {"x": 0.6, "y": 0.0}],
        "springs": [
            {"a": 0, "b": 1, "rest": 0.3, "k": 100, "c": 1, "muscle": 1, "amp": 0.1, "freq": 1, "phase": 0},
            {"a": 1, "b": 2, "rest": 0.3, "k": 100, "c": 1, "muscle": 0, "amp": 0, "freq": 1, "phase": 0},
        ],
    }
    for seed in range(300):
        child = mutate(fan, Mulberry32(seed), 1.0)
        assert is_valid(child), seed
