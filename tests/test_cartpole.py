import math
import os
import pty
import subprocess
import sys
import tty
from pathlib import Path

import numpy as np
import pytest

from cartpole import __main__ as cli
from cartpole.agents import LEFT, RIGHT, LinearAgent, NetAgent, PDAgent, RandomAgent, load_agent, load_weights
from cartpole.benchmark import BENCH_SEED_BASE, evaluate, run_benchmark, summarise
from cartpole.env import MAX_STEPS, THETA_LIMIT, X_LIMIT, CartPole, run_episode
from cartpole.nets import MLP, Adam, normalise, softmax
from cartpole.render import render_frame
from cartpole.train import VALUE_SCALE, discounted_returns, learning_summary, train, train_and_save

# ── Physics ──────────────────────────────────────────────────────────────

def test_first_step_matches_the_equations():
    """From rest at the origin, one push right: check the accelerations by hand, then the Euler update."""
    env = CartPole(0)
    env.x = env.x_dot = env.theta = env.theta_dot = 0.0
    env.step(RIGHT)
    temp = 10.0 / 1.1
    theta_acc = -temp / (0.5 * (4.0 / 3.0 - 0.1 / 1.1))
    x_acc = temp - 0.05 * theta_acc / 1.1
    assert env.x == pytest.approx(0.0)                       # positions use the old (zero) velocities
    assert env.x_dot == pytest.approx(0.02 * x_acc)
    assert env.theta == pytest.approx(0.0)
    assert env.theta_dot == pytest.approx(0.02 * theta_acc)
    assert env.theta_dot < 0                                  # pushing right tips the pole to the left


def test_push_left_is_the_mirror_image():
    right, left = CartPole(0), CartPole(0)
    for e in (right, left):
        e.x = e.x_dot = e.theta = e.theta_dot = 0.0
    right.step(RIGHT)
    left.step(LEFT)
    assert left.x_dot == pytest.approx(-right.x_dot) and left.theta_dot == pytest.approx(-right.theta_dot)


def test_failure_at_the_angle_and_track_limits():
    env = CartPole(0)
    env.reset(0)
    env.theta = THETA_LIMIT * 0.99
    env.theta_dot = 2.0
    while not env.done:
        env.step(LEFT)
    assert env.terminated and not env.truncated
    assert abs(env.theta) > THETA_LIMIT

    env.reset(0)
    env.x, env.x_dot = X_LIMIT * 0.999, 3.0
    env.step(RIGHT)
    assert env.terminated


def test_reset_is_reproducible_and_small():
    a, b = CartPole(7), CartPole(7)
    assert a.obs() == b.obs()
    assert all(abs(v) <= 0.05 for v in a.obs())
    assert CartPole(8).obs() != a.obs()


def test_stepping_a_finished_episode_is_an_error():
    env = CartPole(0)
    env.reset(0)
    env.theta = 1.0
    env.step(RIGHT)
    with pytest.raises(RuntimeError):
        env.step(RIGHT)


def test_episode_is_truncated_at_the_cap_and_reward_is_one_per_step():
    env = CartPole(1)
    env.reset(1)
    rewards = 0.0
    controller = PDAgent()
    while not env.done:
        _, r, _, _ = env.step(controller.act(env.obs()))
        rewards += r
    assert env.truncated and env.steps == MAX_STEPS and rewards == MAX_STEPS


def test_run_episode_records_a_consistent_trajectory():
    n, traj = run_episode(PDAgent(), 3, record=True)
    assert len(traj) == n
    for a, b in zip(traj, traj[1:]):
        assert a["next"] == b["obs"]


# ── Manual backprop, checked against finite differences ──────────────────

def _policy_loss(params, X, A, adv):
    _, Z = MLP(params).forward(X)
    logp = np.log(softmax(Z))[np.arange(len(A)), A]
    return -(adv * logp).mean()


def _numeric_grad(f, params, eps=1e-6):
    grads = {}
    for k, v in params.items():
        g = np.zeros_like(v)
        it = np.nditer(v, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = v[idx]
            v[idx] = old + eps
            hi = f(params)
            v[idx] = old - eps
            lo = f(params)
            v[idx] = old
            g[idx] = (hi - lo) / (2 * eps)
        grads[k] = g
    return grads


def test_policy_gradient_backprop_matches_finite_differences():
    rng = np.random.default_rng(0)
    net = MLP.init(rng, 4, 5, 2, out_scale=1.0)
    X = rng.normal(size=(12, 4))
    A = rng.integers(0, 2, size=12)
    adv = rng.normal(size=12)
    H, Z = net.forward(X)
    P = softmax(Z)
    dZ = -(adv[:, None] / len(A)) * (np.eye(2)[A] - P)     # the formula the trainer uses
    analytic = net.backward(X, H, dZ)
    numeric = _numeric_grad(lambda p: _policy_loss(p, X, A, adv), net.params)
    for k in analytic:
        assert np.allclose(analytic[k], numeric[k], atol=1e-7), k


def test_critic_backprop_matches_finite_differences():
    rng = np.random.default_rng(1)
    critic = MLP.init(rng, 4, 5, 1, out_scale=1.0)
    X = rng.normal(size=(9, 4))
    y = rng.normal(size=9) * 20

    def loss(p):
        _, Z = MLP(p).forward(X)
        return 0.5 * np.mean((VALUE_SCALE * Z[:, 0] - y) ** 2)

    H, Z = critic.forward(X)
    # dL/dz = VALUE_SCALE * (V - y) / T, because V = VALUE_SCALE * z.
    dZ = (VALUE_SCALE * (VALUE_SCALE * Z[:, 0] - y) / len(y))[:, None]
    analytic = critic.backward(X, H, dZ)
    numeric = _numeric_grad(loss, critic.params)
    for k in analytic:
        assert np.allclose(analytic[k], numeric[k], rtol=1e-5, atol=1e-7), k


def test_adam_minimises_a_quadratic():
    params = {"w": np.array([5.0, -3.0])}
    opt = Adam(params, lr=0.1)
    for _ in range(400):
        opt.step({"w": 2 * params["w"]})            # gradient of |w|^2
    assert np.allclose(params["w"], 0.0, atol=1e-2)


def test_discounted_returns():
    assert np.allclose(discounted_returns(np.ones(3), gamma=0.5), [1.75, 1.5, 1.0])


def test_softmax_is_stable_for_large_logits():
    p = softmax(np.array([[1000.0, 1000.0]]))
    assert np.allclose(p, [[0.5, 0.5]])


# ── Agents ───────────────────────────────────────────────────────────────

def test_pd_controller_balances_the_full_episode():
    steps = [run_episode(PDAgent(), 10_000 + i) for i in range(10)]
    assert min(steps) == MAX_STEPS


def test_random_agent_falls_quickly():
    steps = [run_episode(RandomAgent(i), 10_000 + i) for i in range(20)]
    assert np.mean(steps) < 60


def test_committed_reinforce_and_actor_critic_policies_balance():
    for name in ("reinforce", "actor_critic"):
        agent = load_agent(name)
        steps = [run_episode(agent, BENCH_SEED_BASE + i) for i in range(5)]
        assert np.mean(steps) >= 475, name


def test_committed_cem_policy_balances():
    agent = load_agent("cem")
    steps = [run_episode(agent, BENCH_SEED_BASE + i) for i in range(5)]
    assert np.mean(steps) >= 475


def test_net_agent_probs_and_value_are_consistent():
    agent = NetAgent(load_weights("actor_critic"))
    obs = (0.0, 0.0, 0.05, 0.0)
    pl, pr = agent.probs(obs)
    assert pl + pr == pytest.approx(1.0)
    assert agent.value(obs) is not None
    assert NetAgent(load_weights("reinforce")).value(obs) is None
    assert agent.act(obs) == (RIGHT if pr > pl else LEFT)


def test_linear_agent_pushes_toward_the_lean():
    agent = LinearAgent([0.0, 0.0, 1.0, 0.0, 0.0])        # score = normalised theta
    assert agent.act((0.0, 0.0, 0.1, 0.0)) == RIGHT
    assert agent.act((0.0, 0.0, -0.1, 0.0)) == LEFT


def test_evaluate_and_summarise():
    steps = evaluate("pd", 4)
    assert len(steps) == 4 and set(steps) == {MAX_STEPS}
    s = summarise("pd", steps)
    assert s["pct_reached_max"] == 100.0 and s["mean"] == MAX_STEPS and s["std"] == 0.0


# ── Training ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("algo", ["reinforce", "actor_critic"])
def test_policy_gradient_training_is_reproducible(algo):
    w1, r1 = train(algo, 3, 16)
    w2, r2 = train(algo, 3, 16)
    assert r1 == r2 and len(r1) == 16
    assert all(np.array_equal(w1[k], w2[k]) for k in w1)


def test_cem_training_is_reproducible():
    w1, r1 = train("cem", 2, 80)
    w2, r2 = train("cem", 2, 80)
    assert r1 == r2 and np.array_equal(w1["theta"], w2["theta"])


def test_training_improves_on_the_random_start():
    # 200 REINFORCE episodes already beat the random policy's ~20 steps on the same seed.
    _, returns = train("reinforce", 0, 200)
    assert np.mean(returns[-50:]) > np.mean(returns[:50])


def test_learning_summary_and_train_and_save(tmp_path):
    summary = learning_summary([10] * 100 + [500] * 100)
    assert summary["first50"] == 10 and summary["last50"] == 500
    assert summary["episodes_to_475_trailing100"] is not None
    out = train_and_save("cem", [0], 40, data_dir=tmp_path, log=lambda *_: None)
    assert out["best_seed"] == 0
    assert (tmp_path / "cem.npz").exists() and (tmp_path / "curves.json").exists()


def test_normalise_scales_each_variable():
    v = normalise((X_LIMIT, 2.0, THETA_LIMIT, -2.0))
    assert np.allclose(v, [1.0, 1.0, 1.0, -1.0])


# ── Rendering and the terminal game ──────────────────────────────────────

def test_render_frame_shows_cart_pole_and_fallen_flag():
    upright = render_frame(0.0, 0.0, 10)
    assert "[" in upright and "]" in upright and "|" in upright
    assert "FALLEN" not in upright
    fallen = render_frame(0.0, 0.5, 10)
    assert "FALLEN" in fallen
    lean = render_frame(0.5, -0.2, 3, action=LEFT, probs=(0.9, 0.1), value=12.5)
    assert "P(left) 0.90" in lean and "V 12.5" in lean


def test_play_turns_scripted_keys_and_quit():
    keys = iter(["a d", "q"])
    out = []
    steps = cli.play_turns(5, read=lambda _: next(keys), out=out.append)
    assert steps == 2
    assert out[-1] == "stopped after 2 steps"


def test_play_turns_treats_end_of_input_as_quit():
    def read(_):
        raise EOFError
    out = []
    steps = cli.play_turns(5, read=read, out=out.append)
    assert steps == 0
    assert out[-1] == "stopped after 0 steps"


def test_play_turns_ends_when_the_pole_falls():
    out = []
    steps = cli.play_turns(5, read=lambda _: "a" * 50, out=out.append)
    assert 0 < steps < MAX_STEPS and "the pole fell" in out[-1]


def test_watch_runs_one_episode_and_reports_it():
    out = []
    steps = cli.watch("pd", 10_000, delay=0.0, out=out.append, sleep=lambda _: None)
    assert steps == MAX_STEPS
    assert "500-step cap" in out[-1]


def test_play_slow_without_a_terminal_exits_with_a_message():
    # Piped stdin is not a tty. The command must say so instead of failing with a raw termios error.
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run([sys.executable, "-m", "cartpole", "play", "--slow"], input="", capture_output=True,
                          text=True, timeout=60, cwd=root)
    assert proc.returncode == 1 and "needs a terminal" in proc.stderr
    assert "Traceback" not in proc.stderr and "termios" not in proc.stderr


def test_play_slow_on_q_prints_the_outcome(monkeypatch):
    # A pseudo-terminal stands in for the user: 'q' is waiting, so the loop ends at once and reports it.
    master, slave = pty.openpty()
    class Terminal:
        def fileno(self):
            return slave

        def isatty(self):
            return True

    # The pty must be in cbreak mode before the key is written, or the key waits for a newline. play_slow's own
    # setcbreak call uses TCSAFLUSH, which would drop the key, so that call is stubbed out after this point.
    tty.setcbreak(slave)
    monkeypatch.setattr(tty, "setcbreak", lambda fd: None)
    try:
        os.write(master, b"q")
        monkeypatch.setattr(sys, "stdin", Terminal())
        out = []
        steps = cli.play_slow(5, tick=0.5, out=out.append)
    finally:
        os.close(master)
        os.close(slave)
    assert steps == 0 and out[-1] == "stopped after 0 steps"


def test_cli_help_and_bad_command_exit_cleanly():
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0


def test_benchmark_document_shape():
    doc = run_benchmark(2, names=("pd",), log=lambda *_: None)
    assert doc["episodes"] == 2 and doc["agents"]["pd"]["mean"] == MAX_STEPS
    assert "PD CONTROLLER" in doc["agents"]["pd"]["label"]
    assert "PD CONTROLLER" in cli.table(doc)


def test_failure_angle_is_twelve_degrees():
    assert math.isclose(THETA_LIMIT, 12 * math.pi / 180)
