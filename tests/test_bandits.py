import math
import re
import statistics

import pytest

from bandits import __main__ as cli
from bandits.agents import AGENTS, EXP3, LINEUP, UCB1, SlidingWindowUCB, Thompson, argmax, make_agent
from bandits.benchmark import run_benchmark, to_markdown, write_results
from bandits.env import Environment, bernoulli_kl, lai_robbins_rate, make_machines
from bandits.rng import Rng
from bandits.sim import band, interval, play, run_agent, score_pulls


def test_rng_is_reproducible_and_in_range():
    a, b = Rng(42), Rng(42)
    xs = [a.uniform() for _ in range(1000)]
    assert xs == [b.uniform() for _ in range(1000)]
    assert all(0.0 <= x < 1.0 for x in xs)
    assert Rng(43).uniform() != xs[0]


def test_rng_moments_match_their_distributions():
    r = Rng(5)
    us = [r.uniform() for _ in range(20000)]
    assert statistics.fmean(us) == pytest.approx(0.5, abs=0.01)
    zs = [r.normal() for _ in range(20000)]
    assert statistics.fmean(zs) == pytest.approx(0.0, abs=0.03)
    assert statistics.pstdev(zs) == pytest.approx(1.0, abs=0.03)
    betas = [r.beta(3, 7) for _ in range(20000)]
    assert statistics.fmean(betas) == pytest.approx(0.3, abs=0.01)
    gammas = [r.gamma(2.5) for _ in range(20000)]
    assert statistics.fmean(gammas) == pytest.approx(2.5, abs=0.05)


def test_rng_index_covers_all_values():
    r = Rng(3)
    seen = {r.index(4) for _ in range(400)}
    assert seen == {0, 1, 2, 3}


def test_machines_are_seeded_and_have_a_clear_best_arm():
    m1 = make_machines("bernoulli", 5, 300, 11)
    m2 = make_machines("bernoulli", 5, 300, 11)
    assert m1 == m2 and m1.schedule != make_machines("bernoulli", 5, 300, 12).schedule
    for seg in m1.schedule:
        ordered = sorted(seg)
        assert ordered[-1] - ordered[-2] >= 0.05 - 1e-12
        assert all(0.1 <= p <= 0.9 for p in seg)


def test_drifting_machines_redraw_every_period():
    m = make_machines("drifting", 4, 1200, 2)
    assert m.period == 500 and len(m.schedule) == 3
    assert m.mean_at(0) == m.schedule[0] and m.mean_at(999) == m.schedule[1]
    assert m.mean_at(1199) == m.schedule[2]


def test_outcomes_are_fixed_for_a_seed():
    env = Environment("bernoulli", 3, 50, 4)
    assert [env.reward(t, a) for t in range(50) for a in range(3)] == \
        [Environment("bernoulli", 3, 50, 4).reward(t, a) for t in range(50) for a in range(3)]
    assert {env.reward(t, a) for t in range(50) for a in range(3)} <= {0.0, 1.0}


def test_gaussian_reward_is_mean_plus_noise():
    env = Environment("gaussian", 3, 20, 4)
    for t in range(20):
        for a in range(3):
            assert abs(env.reward(t, a) - env.mean(t)[a]) < 1.0  # noise is 0.2 sd: far inside 1.0


def test_lai_robbins_rate_matches_hand_calculation():
    # Two Bernoulli arms at 0.5 and 0.25: rate = gap / KL(0.25 || 0.5) with gap 0.25.
    kl = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
    assert bernoulli_kl(0.25, 0.5) == pytest.approx(kl)
    assert lai_robbins_rate((0.5, 0.25), "bernoulli") == pytest.approx(0.25 / kl)
    # Gaussian arms: each suboptimal arm contributes 2 sigma^2 / gap.
    assert lai_robbins_rate((0.5, 0.3), "gaussian", sigma=0.2) == pytest.approx(2 * 0.04 / 0.2)


def test_lai_robbins_rate_refuses_drifting_machines():
    with pytest.raises(ValueError):
        lai_robbins_rate((0.5, 0.2), "drifting")


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        make_machines("roulette", 3, 10, 0)
    with pytest.raises(ValueError):
        make_machines("bernoulli", 1, 10, 0)
    with pytest.raises(ValueError):
        make_agent("nope", 3, 10, Rng(0))


def test_every_agent_returns_valid_arms_and_forces_the_first_round():
    env = Environment("drifting", 4, 120, 6)
    for key in AGENTS:
        agent = make_agent(key, 4, 120, Rng(1), sigma=1.0)
        first = []
        for t in range(120):
            arm = agent.choose()
            assert 0 <= arm < 4
            if t < 4:
                first.append(arm)
            agent.update(arm, env.reward(t, arm))
        if key not in ("exp3",):
            assert sorted(first) == [0, 1, 2, 3], key


def test_ucb_and_thompson_beat_greedy_on_average():
    # Ten machine sets, 2000 pulls each: the bandit agents lose much less regret than greedy.
    def mean_regret(key):
        return statistics.fmean(play(key, Environment("bernoulli", 5, 2000, s)).total_regret for s in range(10))

    greedy = mean_regret("greedy")
    assert mean_regret("ucb1") < greedy
    assert mean_regret("thompson") < greedy


def test_thompson_posterior_counts_wins_and_losses():
    agent = Thompson(3, 100, Rng(0))
    agent.update(0, 1.0)
    agent.update(0, 0.0)
    agent.update(2, 1.0)
    assert agent.posterior() == [(2.0, 2.0), (1.0, 1.0), (2.0, 1.0)]


def test_thompson_gaussian_posterior_has_known_precision():
    agent = Thompson(2, 100, Rng(0), sigma=0.2, bernoulli=False)
    agent.update(1, 0.5)
    (m0, s0), (m1, s1) = agent.posterior()
    assert m0 == 0.0 and s0 == pytest.approx(1.0)
    prec = 1.0 + 1.0 / 0.04
    assert m1 == pytest.approx((0.5 / 0.04) / prec) and s1 == pytest.approx(1 / math.sqrt(prec))


def test_exp3_probabilities_are_a_distribution_with_a_floor():
    agent = EXP3(4, 500, Rng(0))
    for _ in range(200):
        arm = agent.choose()
        agent.update(arm, 1.0 if arm == 2 else 0.0)
    probs = agent.probabilities()
    assert sum(probs) == pytest.approx(1.0)
    assert min(probs) >= agent.gamma / 4 - 1e-12
    assert argmax(probs) == 2


def test_sliding_window_forgets_old_rewards():
    agent = SlidingWindowUCB(2, 1000, Rng(0), window=10)
    for _ in range(50):
        agent.update(0, 1.0)
    assert len(agent._recent) == 10 and agent.wcounts == [10, 0]
    assert agent.wsums == [10.0, 0.0]
    # Arm 1 has no pulls in the window, so it is the next choice.
    assert agent.choose() == 1


def test_run_regret_is_consistent_with_scoring_the_same_arms():
    env = Environment("gaussian", 4, 300, 8)
    run = play("ucb1", env)
    assert all(b >= a for a, b in zip(run.regret, run.regret[1:]))
    scored = score_pulls(env, run.arms)
    assert scored.regret == pytest.approx(run.regret)
    assert scored.rewards == pytest.approx(run.rewards)


def test_score_pulls_rejects_bad_input():
    env = Environment("bernoulli", 3, 10, 0)
    with pytest.raises(ValueError):
        score_pulls(env, [0] * 11)
    with pytest.raises(ValueError):
        score_pulls(env, [3])


def test_run_agent_calls_back_each_pull():
    env = Environment("bernoulli", 3, 25, 1)
    seen = []
    run_agent(make_agent("greedy", 3, 25, Rng(1)), env, on_pull=lambda t, arm, r, agent: seen.append(t))
    assert seen == list(range(25))


def test_band_and_interval_on_known_values():
    mean, lo, hi = band([[1.0, 2.0], [3.0, 2.0]])
    assert mean == [2.0, 2.0] and lo[1] == hi[1] == 2.0
    assert hi[0] - mean[0] == pytest.approx(1.96 * math.sqrt(2 / 2))  # sd sqrt(2), n 2
    assert interval([5.0]) == (5.0, 5.0, 5.0)


def test_lineups_only_name_registered_agents():
    for kind, keys in LINEUP.items():
        assert set(keys) <= set(AGENTS), kind
    assert UCB1.key in LINEUP["bernoulli"] and "sw_ucb" in LINEUP["drifting"]


def test_small_benchmark_has_the_expected_shape(tmp_path):
    result = run_benchmark(seeds=2, horizon=1100, kinds=("bernoulli", "drifting"), checkpoints=(100, 1000, 1100))
    assert result["table_at"] == [1000, 1100]
    bern = result["settings"]["bernoulli"]
    assert set(bern["agents"]) == set(LINEUP["bernoulli"])
    assert len(bern["lai_robbins"]["curve"]) == 3 and bern["lai_robbins"]["rate"] > 0
    curve = bern["agents"]["ucb1"]["curve"]
    assert len(curve["mean"]) == 3 and all(lo <= m <= hi for m, lo, hi in zip(curve["mean"], curve["lo"], curve["hi"]))
    assert "lai_robbins" not in result["settings"]["drifting"]
    md = to_markdown(result)
    assert "Regret at T=1,000" in md and "Lai-Robbins floor" in md
    json_path, md_path = write_results(result, tmp_path)
    assert json_path.name == "bandits_benchmark.json" and md_path.read_text() == md


def test_benchmark_rejects_short_horizons():
    with pytest.raises(ValueError):
        run_benchmark(seeds=1, horizon=1000)


def test_parse_pull_accepts_letters_and_numbers():
    assert cli.parse_pull("a", 5) == 0 and cli.parse_pull(" E ", 5) == 4
    assert cli.parse_pull("3", 5) == 2 and cli.parse_pull("6", 5) is None and cli.parse_pull("f", 5) is None


def test_play_human_scores_the_player_against_the_lineup():
    answers = iter(["a", "b", "c", "q"])
    out = []
    used = cli.play_human("bernoulli", 3, 10, 7, read=lambda prompt: next(answers), out=out.append)
    assert used == 3
    text = "\n".join(out)
    assert "The casino, revealed" in text and "You beat" in text
    assert text.count("regret") >= 1 + len(LINEUP["bernoulli"])


def test_play_human_handles_end_of_input():
    def eof(prompt):
        raise EOFError

    out = []
    assert cli.play_human("gaussian", 3, 5, 1, read=eof, out=out.append) == 0
    assert "No pulls" in out[-1]


def test_watch_prints_one_line_per_pull_with_a_reason():
    out = []
    run = cli.watch("ucb1", "bernoulli", 4, 12, 3, out=out.append, sleep=lambda s: None)
    pull_lines = [line for line in out if re.match(r"\s+\d+\s+machine [A-D]\s", line)]
    assert len(pull_lines) == 12
    assert all("highest optimistic bound" in line or "first round" in line for line in pull_lines)
    assert len(run.arms) == 12


def test_main_dispatches_subcommands(tmp_path, capsys):
    assert cli.main(["watch", "--agent", "thompson", "--pulls", "5", "--seed", "2", "-k", "3"]) == 0
    assert "Thompson sampling" in capsys.readouterr().out
    assert cli.main(["benchmark", "--seeds", "1", "--horizon", "1100", "--out", str(tmp_path), "--no-write"]) == 0
    assert not (tmp_path / "bandits_benchmark.json").exists()


def test_main_rejects_bad_arguments():
    with pytest.raises(SystemExit):
        cli.main(["play", "--pulls", "5000"])
    with pytest.raises(SystemExit):
        cli.main(["benchmark", "--horizon", "500"])
    with pytest.raises(SystemExit):
        cli.main(["watch", "--agent", "nope"])


def test_describe_segments_labels_each_schedule_block():
    env = Environment("drifting", 3, 1000, 5)
    lines = cli.describe_segments(env)
    assert len(lines) == 2 and "pulls 1-500" in lines[0] and "pulls 501-1000" in lines[1]
