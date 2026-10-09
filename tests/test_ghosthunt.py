"""Ghost Hunt: the forward algorithm and Viterbi against brute-force enumeration, the motion models, the rules,
determinism, and the particle filter's convergence to the exact belief."""

import math
import random

import pytest

from bandits.rng import Rng
from ghosthunt import agent, benchmark, motion, sonar
from ghosthunt.game import Game
from ghosthunt.hmm import ExactFilter, forward, viterbi
from ghosthunt.maze import Maze, generate, render
from ghosthunt.particles import ParticleFilter


def tiny_maze(seed: int = 3) -> Maze:
    """A 5x5 maze: four rooms and three passages, 7 open cells, 28 hidden states. Small enough to enumerate."""
    return Maze(5, generate(5, seed))


def random_episode(maze: Maze, model: str, steps: int, sigma: float, seed: int):
    """A player walk and the ghost's true path with its readings, drawn from the model (for the tests)."""
    rnd = random.Random(seed)
    rng = Rng(seed + 99)
    emis = sonar.emission_table(sigma, maze.diameter)
    players = [maze.start]
    for _ in range(steps):
        opts = [maze.step[players[-1]][d] for d in range(4) if maze.step[players[-1]][d] >= 0]
        players.append(rnd.choice(opts + [players[-1]]))
    s = rnd.randrange(4 * maze.K)
    states, readings = [s], []
    readings.append(sonar.sample_reading(emis[maze.dist[s >> 2][players[0]]], rng.uniform()))
    for t in range(1, steps + 1):
        s = rnd.choice([s2 for s2, _ in motion.transitions(maze, model, s, players[t])]) \
            if False else _draw(motion.transitions(maze, model, s, players[t]), rng.uniform())
        states.append(s)
        readings.append(sonar.sample_reading(emis[maze.dist[s >> 2][players[t]]], rng.uniform()))
    return players, states, readings, emis


def _draw(options, u):
    acc = 0.0
    for s2, q in options:
        acc += q
        if u < acc:
            return s2
    return options[-1][0]


def brute_enumerate(maze, model, prior, emis, players, readings):
    """Every state sequence s_0..s_T with positive probability, with its joint probability (unnormalised)."""
    dist = maze.dist
    paths = [((s,), prior[s] * emis[dist[s >> 2][players[0]]][readings[0]])
             for s in range(len(prior)) if prior[s] > 0.0]
    for t in range(1, len(readings)):
        nxt = []
        for path, w in paths:
            for s2, q in motion.transitions(maze, model, path[-1], players[t]):
                if q > 0.0:
                    nxt.append((path + (s2,), w * q * emis[dist[s2 >> 2][players[t]]][readings[t]]))
        paths = nxt
    return paths


def brute_marginals(maze, model, prior, emis, players, readings):
    """Filtering posterior P(state at t | readings 0..t) by enumeration, one prefix at a time.

    This is the quantity the forward algorithm computes. Using all readings instead would give the smoothed
    posterior, which is a different (and not comparable) number.
    """
    out = []
    for t in range(len(readings)):
        paths = brute_enumerate(maze, model, prior, emis, players[:t + 1], readings[:t + 1])
        z = sum(w for _, w in paths)
        m = [0.0] * len(prior)
        for path, w in paths:
            m[path[t]] += w / z
        out.append(m)
    return out


@pytest.mark.parametrize("model", motion.MODELS)
@pytest.mark.parametrize("seed", [1, 2])
def test_forward_matches_brute_force(model, seed):
    maze = tiny_maze(seed + 2)
    prior = [1.0 / (4 * maze.K)] * (4 * maze.K)
    players, _, readings, emis = random_episode(maze, model, steps=3, sigma=1.0, seed=seed)
    got = forward(maze, model, prior, emis, players, readings)
    want = brute_marginals(maze, model, prior, emis, players, readings)
    for t in range(len(readings)):
        assert sum(got[t]) == pytest.approx(1.0, abs=1e-12)
        for s in range(len(prior)):
            assert got[t][s] == pytest.approx(want[t][s], abs=1e-10), (t, s)


def test_forward_with_a_patrol_prior_and_low_noise_matches_brute_force():
    maze = tiny_maze(5)
    prior = [0.0] * (4 * maze.K)
    for s in range(4 * maze.K):
        if s % 4 == 1:  # only eastward headings start, so the prior has zeros the recursion must keep
            prior[s] = 1.0 / maze.K
    players, _, readings, emis = random_episode(maze, "patrol", steps=3, sigma=0.5, seed=9)
    got = forward(maze, "patrol", prior, emis, players, readings)
    want = brute_marginals(maze, "patrol", prior, emis, players, readings)
    for t in range(len(readings)):
        for s in range(len(prior)):
            assert got[t][s] == pytest.approx(want[t][s], abs=1e-10)


@pytest.mark.parametrize("model", motion.MODELS)
def test_viterbi_finds_the_most_likely_path(model):
    maze = tiny_maze(4)
    prior = [1.0 / (4 * maze.K)] * (4 * maze.K)
    players, _, readings, emis = random_episode(maze, model, steps=3, sigma=1.0, seed=5)
    path = viterbi(maze, model, prior, emis, players, readings)
    paths = brute_enumerate(maze, model, prior, emis, players, readings)
    z = sum(w for _, w in paths)
    best = max(w for _, w in paths) / z

    def joint(states):  # probability of one full state sequence, the quantity Viterbi maximises
        p = prior[states[0]] * emis[maze.dist[states[0] >> 2][players[0]]][readings[0]]
        for t in range(1, len(states)):
            q = dict(motion.transitions(maze, model, states[t - 1], players[t])).get(states[t], 0.0)
            p *= q * emis[maze.dist[states[t] >> 2][players[t]]][readings[t]]
        return p / z

    assert len(path) == len(readings)
    assert joint(path) == pytest.approx(best, rel=1e-9)


def test_beliefs_are_normalised_through_a_game():
    g = Game(size=11, ghosts=2, model="lurker", sigma=1.0, seed=6)
    for _ in range(25):
        for gh in g.ghosts:
            assert sum(gh.filter.belief) == pytest.approx(1.0, abs=1e-9)
            assert sum(g.marginal(gh.id)) == pytest.approx(1.0, abs=1e-9)
        if g.done:
            break
        g.act(agent.choose(g))


@pytest.mark.parametrize("model", motion.MODELS)
def test_motion_rows_are_distributions_and_stay_on_open_cells(model):
    maze = Maze(11, generate(11, 2))
    for player in (0, maze.K // 2, maze.K - 1):
        for s in range(4 * maze.K):
            opts = motion.transitions(maze, model, s, player)
            assert sum(q for _, q in opts) == pytest.approx(1.0, abs=1e-12)
            for s2, q in opts:
                assert q >= 0.0
                k2 = s2 >> 2
                assert 0 <= k2 < maze.K
                if k2 != s >> 2:  # a move must follow an open passage
                    assert k2 in [maze.step[s >> 2][d] for d in range(4)]


def test_patrol_follows_corridors_and_never_waits():
    maze = Maze(11, generate(11, 2))
    for s in range(4 * maze.K):
        assert all(s2 != s for s2, _ in motion.transitions(maze, "patrol", s, 0))


def test_lurker_drifts_toward_the_player():
    maze = Maze(15, generate(15, 7))
    far = max(range(maze.K), key=lambda k: maze.dist[k][maze.start])
    s = 4 * far
    closer = 0.0
    for s2, q in motion.transitions(maze, "lurker", s, maze.start):
        if maze.dist[s2 >> 2][maze.start] < maze.dist[far][maze.start]:
            closer += q
    assert closer > 0.5


def test_maze_is_a_perfect_maze_and_deterministic():
    for n in (5, 11, 21):
        walls = generate(n, 42)
        assert walls == generate(n, 42)
        m = Maze(n, walls)
        edges = sum(1 for k in range(m.K) for d in range(4) if m.step[k][d] > k and m.step[k][d] >= 0)
        assert edges == m.K - 1  # a tree: connected with no loops
        assert all(v >= 0 for v in m.dist[0])
    with pytest.raises(ValueError):
        generate(10, 1)


def test_sonar_rows_are_distributions_peaked_at_the_true_distance():
    table = sonar.emission_table(1.0, 20)
    for d, row in enumerate(table):
        assert sum(row) == pytest.approx(1.0, abs=1e-9)
        assert row.index(max(row)) == d
    assert sonar.sample_reading(table[5], 0.0) == 0
    assert sonar.sample_reading(table[5], 0.999999) == 20 or sonar.sample_reading(table[5], 0.999999) >= 0


def test_rules_bust_hit_miss_and_illegal_moves():
    g = Game(size=11, ghosts=1, model="random", seed=3)
    gh = g.ghosts[0]
    # Put the ghost on the player's own cell: the stay-bust must remove it.
    gh.x = 4 * g.p + 0
    gh.true[-1] = g.p
    res = g.act("b.")
    assert res["busted"] == [0] and g.done and gh.bust_turn == 0
    g2 = Game(size=11, ghosts=1, model="random", seed=3)
    with pytest.raises(ValueError):
        g2.act("bZ")
    wall_dir = next(c for c, d in zip("NESW", range(4)) if g2.maze.step[g2.p][d] < 0)
    with pytest.raises(ValueError):
        g2.act(wall_dir)
    # A miss: busting an open neighbour the ghost is not on wastes the turn but keeps the ghost alive.
    g3 = Game(size=11, ghosts=1, model="random", seed=3)
    dir_ch = next(c for c, d in zip("NESW", range(4)) if g3.maze.step[g3.p][d] >= 0)
    g3.ghosts[0].x = 4 * g3.p  # standing on the player's own cell, not on the neighbour
    res = g3.act("b" + dir_ch)
    assert res["busted"] == [] and g3.ghosts[0].live and g3.t == 1


def test_same_seed_gives_the_same_game():
    def play(filt):
        g = Game(size=15, ghosts=2, model="patrol", sigma=1.0, seed=8, filt=filt, particles=50)
        while not g.done:
            g.act(agent.choose(g))
        return [g.trace(i) for i in range(len(g.ghosts))], g.t

    assert play("exact") == play("exact")
    assert play("particles") == play("particles")


def test_particle_filter_converges_to_the_exact_belief():
    """On one recorded game the particle belief approaches the exact one as N grows (measured, see the results file)."""
    g = Game(size=15, ghosts=1, model="random", sigma=1.0, seed=2, max_turns=40)
    while not g.done:
        g.act(agent.choose(g))
    gh = g.ghosts[0]
    readings, players = gh.readings, g.players[:len(gh.readings)]

    def kl_for(n):
        ex = ExactFilter(g.maze, "random", g.prior, g.emis)
        pf = ParticleFilter(g.maze, "random", g.prior, g.emis, n, Rng(77 + n))
        kls = []
        for t, r in enumerate(readings):
            if t:
                ex.predict(players[t])
                pf.predict(players[t])
            ex.update(r, players[t])
            pf.update(r, players[t])
            kls.append(benchmark.kl(ex.marginal(), pf.marginal()))
        return sum(kls) / len(kls)

    small, large = kl_for(20), kl_for(4000)
    assert large < small
    assert large < 0.05


def test_the_autopilot_busts_every_ghost_on_fixed_seeds():
    for seed in range(1, 6):
        g = Game(size=15, ghosts=2, model="random", sigma=1.0, seed=seed)
        while not g.done:
            g.act(agent.choose(g))
        assert g.busted == 2, seed


def test_render_and_cli_smoke(capsys):
    from ghosthunt import __main__ as cli

    g = Game(size=11, ghosts=1, seed=4)
    text = render(g.maze, {g.p: "P"})
    assert "P" in text and "#" in text
    assert cli.parse_word("w") == "N" and cli.parse_word("bd") == "bE" and cli.parse_word("b.") == "b."
    assert cli.parse_word("zz") is None
    assert cli.main(["--autopilot", "--delay", "0", "--size", "11", "--ghosts", "1", "--seed", "2"]) == 0
    assert "ghost" in capsys.readouterr().out


def test_belief_stays_sane_at_high_noise_and_long_games():
    # Regression guard for the numerical floors: a long game at the highest noise must never produce NaN.
    g = Game(size=21, ghosts=3, model="lurker", sigma=2.0, seed=12, filt="particles", particles=60, max_turns=120)
    while not g.done:
        g.act(agent.choose(g))
    for gh in g.ghosts:
        assert all(math.isfinite(x) for x in g.marginal(gh.id))
