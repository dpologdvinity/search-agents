"""Grid MDP lab: the model, the two planners, the two learners, and the command line.

Hand-computed values pin the transition model; the planners must agree with each other and satisfy the
Bellman optimality equation at convergence; the learners must reach the optimal greedy policy on a small
deterministic grid, and be reproducible from their seeds.
"""

import math

import pytest

from mdplab import Grid, GridMDP, Params, PolicyIterator, QLearner, ValueIterator, preset
from mdplab.__main__ import main
from mdplab.grid import PRESETS
from mdplab.learn import optimal_reference
from mdplab.solve import backup, reachable


def model(rows, **kw):
    return GridMDP(Grid(tuple(rows)), Params(**kw))


def solved_vi(mdp, eps=1e-12):
    vi = ValueIterator(mdp)
    vi.solve(eps)
    return vi


# ── grids and presets ────────────────────────────────────────────────────

def test_every_preset_is_a_valid_connected_grid():
    for key, pr in PRESETS.items():
        mdp = GridMDP(pr.grid(), pr.params)
        assert mdp.check_reachable(), key
        assert mdp.terminal.count(True) >= 1, key


@pytest.mark.parametrize("rows", [
    [],
    ["#S", "#"],
    ["#Q#"],
    ["#...#"],
    ["S.S"],
])
def test_bad_grids_are_rejected(rows):
    with pytest.raises(ValueError):
        Grid(tuple(rows))


@pytest.mark.parametrize("kw", [{"gamma": 1.5}, {"slip": 0.6}, {"alpha": 0.0}, {"decay": 0.0}])
def test_bad_parameters_are_rejected(kw):
    with pytest.raises(ValueError):
        Params(**kw).validate()


def test_preset_override_and_unknown_key():
    pr = preset("windy", slip=0.0)
    assert pr.params.slip == 0.0 and PRESETS["windy"].params.slip == 0.3
    with pytest.raises(KeyError):
        preset("nope")


# ── the transition model, by hand ────────────────────────────────────────

def test_transitions_sum_to_one_and_respect_walls():
    mdp = GridMDP(preset("rooms").grid(), preset("rooms").params)
    for s in mdp.states:
        for a in range(4):
            probs = [p for p, _, _ in mdp.P[s][a]]
            assert math.isclose(sum(probs), 1.0, abs_tol=1e-12)
            for _, dest, _ in mdp.P[s][a]:
                assert mdp.kinds[dest] != "#"


def test_corridor_hand_computed_values():
    # S at x=1, one empty cell, then a +1 cell. Each step costs living = -0.1, and the goal is terminal.
    # V(middle) = -0.1 + 1 = 0.9.  V(start) = -0.1 + 0.9 * 0.9 = 0.71.
    mdp = model(["#####", "#S.+#", "#####"], gamma=0.9, slip=0.0, living=-0.1, reward=1.0)
    vi = solved_vi(mdp)
    assert vi.V[mdp.start] == pytest.approx(0.71, abs=1e-9)
    assert vi.V[mdp.start + 1] == pytest.approx(0.9, abs=1e-9)


def test_slip_closed_form():
    # One step from S to a +1 cell, walls above and below. With slip 0.5 the intended move succeeds half
    # the time; the slips hit walls and leave the agent in place. V = 0.5 * 1 + 0.5 * gamma * V,
    # so V = 0.5 / (1 - 0.5 * gamma) = 10/11 at gamma = 0.9.
    mdp = model(["####", "#S+#", "####"], gamma=0.9, slip=0.5, living=0.0, reward=1.0)
    vi = solved_vi(mdp)
    assert vi.V[mdp.start] == pytest.approx(0.5 / 0.55, abs=1e-9)


def test_direction_for_matches_the_transition_table():
    mdp = model(["#####", "#S..#", "#####"], gamma=0.9, slip=0.2)
    counts = {0: 0, 1: 0, 2: 0}  # 0 intended, 1 left slip, 2 right slip
    n = 10000
    for i in range(n):
        d = mdp.direction_for(1, (i + 0.5) / n)
        counts[0 if d == 1 else (1 if d == 0 else 2)] += 1
    assert counts[0] / n == pytest.approx(0.8, abs=1e-3)
    assert counts[1] / n == pytest.approx(0.1, abs=1e-3)
    assert counts[2] / n == pytest.approx(0.1, abs=1e-3)


# ── planning ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", list(PRESETS))
def test_value_iteration_satisfies_the_bellman_optimality_equation(key):
    pr = PRESETS[key]
    mdp = GridMDP(pr.grid(), pr.params)
    vi = solved_vi(mdp, 1e-11)
    for s in mdp.states:
        assert vi.V[s] == pytest.approx(max(backup(mdp, vi.V, s)), abs=1e-8)


@pytest.mark.parametrize("key", list(PRESETS))
def test_value_and_policy_iteration_agree(key):
    pr = PRESETS[key]
    mdp = GridMDP(pr.grid(), pr.params)
    vi = solved_vi(mdp, 1e-11)
    vi_pi = vi.policy()
    pi = PolicyIterator(mdp, seed=3)
    summary = pi.solve(1e-11)
    assert summary["stable"] and not summary["capped"]
    for s in mdp.states:
        assert pi.V[s] == pytest.approx(vi.V[s], abs=1e-7)
        if pi.pi[s] != vi_pi[s]:
            # Only equally good actions may differ.
            q = backup(mdp, vi.V, s)
            assert q[pi.pi[s]] == pytest.approx(max(q), abs=1e-7)


def test_value_iteration_residual_contracts_by_gamma():
    pr = preset("rooms")
    mdp = GridMDP(pr.grid(), pr.params)
    vi = ValueIterator(mdp)
    res = [vi.sweep() for _ in range(40)]
    for k in range(1, len(res) - 1):
        assert res[k + 1] <= pr.params.gamma * res[k] + 1e-12


def test_policy_iteration_counts_rounds_and_is_stable():
    mdp = GridMDP(preset("maze").grid(), preset("maze").params)
    pi = PolicyIterator(mdp, seed=1)
    summary = pi.solve(1e-9)
    assert summary["stable"]
    assert 1 <= summary["rounds"] <= 50
    assert pi.last_changed == 0


def test_maze_shortcut_through_the_pit_is_avoided():
    pr = preset("maze")
    mdp = GridMDP(pr.grid(), pr.params)
    vi = solved_vi(mdp)
    pi = vi.policy()
    # Cell (4, 5) sits on the shortcut, one step before the pit at (5, 5). The best move is back the way it came.
    assert pi[5 * 9 + 4] == 3  # left
    # And the start leads right, along the long safe route.
    assert pi[mdp.start] == 1


def test_reachable_set_stops_at_terminals():
    pr = preset("cliff")
    mdp = GridMDP(pr.grid(), pr.params)
    pi = ValueIterator(mdp)
    pi.solve(1e-9)
    states = reachable(mdp, pi.policy())
    assert mdp.start in states
    assert all(not mdp.terminal[s] for s in states)


# ── learning ─────────────────────────────────────────────────────────────

TINY = ["######", "#S..+#", "######"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_q_learning_reaches_the_optimal_policy_on_a_tiny_deterministic_grid(seed):
    mdp = model(TINY, gamma=0.9, slip=0.0, living=-0.04, reward=1.0)
    vi = solved_vi(mdp)
    opt_pi, opt_Q = optimal_reference(mdp, vi.V)
    agent = QLearner(mdp, algo="q", seed=seed, alpha=0.5, epsilon=0.3, decay=0.99, eps_min=0.01)
    agent.run(400)
    frac, _ = agent.agreement(opt_pi, opt_Q)
    assert frac == 1.0
    assert agent.greedy(mdp.start) == 1  # right, toward the reward


def test_sarsa_also_learns_the_tiny_grid():
    mdp = model(TINY, gamma=0.9, slip=0.0, living=-0.04, reward=1.0)
    vi = solved_vi(mdp)
    opt_pi, opt_Q = optimal_reference(mdp, vi.V)
    agent = QLearner(mdp, algo="sarsa", seed=5, alpha=0.5, epsilon=0.3, decay=0.99, eps_min=0.01)
    agent.run(600)
    assert agent.agreement(opt_pi, opt_Q)[0] == 1.0


def test_learning_is_deterministic_for_a_seed():
    mdp = GridMDP(preset("rooms").grid(), preset("rooms").params)
    a = QLearner(mdp, seed=11)
    b = QLearner(mdp, seed=11)
    c = QLearner(mdp, seed=12)
    ra, rb, rc = a.run(300), b.run(300), c.run(300)
    assert ra == rb
    assert a.Q == b.Q
    assert ra != rc


def test_cliff_sarsa_earns_more_online_than_q_learning():
    # The Sutton and Barto result: with exploration on, Q-learning learns the edge-hugging route and
    # falls often, while SARSA learns the safer route inland, so its online returns are higher.
    pr = preset("cliff")
    mdp = GridMDP(pr.grid(), pr.params)
    q = QLearner(mdp, algo="q", seed=1)
    s = QLearner(mdp, algo="sarsa", seed=1)
    rq = q.run(3000)
    rs = s.run(3000)
    assert sum(rs[-500:]) / 500 > sum(rq[-500:]) / 500


def test_q_learning_on_the_maze_matches_the_optimal_policy():
    pr = preset("maze")
    mdp = GridMDP(pr.grid(), pr.params)
    vi = solved_vi(mdp)
    opt_pi, opt_Q = optimal_reference(mdp, vi.V)
    agent = QLearner(mdp, seed=1)
    agent.run(6000)
    assert agent.agreement(opt_pi, opt_Q)[0] >= 0.95


# ── command line ─────────────────────────────────────────────────────────

def test_cli_solve_prints_grids_and_full_agreement(capsys):
    assert main(["solve", "--preset", "cliff"]) == 0
    out = capsys.readouterr().out
    assert "VALUE ITERATION" in out and "POLICY ITERATION" in out
    assert "agreement with value iteration: 100.0%" in out


def test_cli_learn_prints_a_row_per_block(capsys):
    assert main(["learn", "--preset", "rooms", "--episodes", "400", "--every", "200", "--seed", "2"]) == 0
    out = capsys.readouterr().out
    assert "final agreement with the optimal policy" in out
    assert out.count("%") >= 3


@pytest.mark.parametrize(
    "argv",
    [
        ["solve", "--gamma", "2"],
        ["solve", "--gamma", "nan"],
        ["solve", "--slip", "0.6"],
        ["solve", "--eps", "0"],
        ["learn", "--alpha", "0"],
        ["learn", "--decay", "0"],
        ["learn", "--epsilon", "-0.1"],
        ["learn", "--max-steps", "0"],
        ["learn", "--episodes", "0"],
        ["learn", "--every", "0"],
    ],
)
def test_cli_rejects_values_outside_the_lab_bounds(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "must be in" in capsys.readouterr().err
