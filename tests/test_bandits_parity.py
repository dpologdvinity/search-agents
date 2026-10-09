"""Parity: the page's casino and agents (web/js/bandits-core.js) must match the bandits package.

There are 17 agent and casino pairs in the lineups (5 for bernoulli, 5 for gaussian, 7 for drifting). For each
pair and seeds 0 to 4, Python plays the casino, and node plays the same casino through the JavaScript core
(tests/js/bandits_parity.mjs). Arms, optimal flags and reasons must agree exactly; rewards and cumulative regret
must agree to 1e-9, since the polar-method Gaussian draws go through log, which can differ in the last bit between
libm and V8. The horizon is 1,200 pulls, so the drifting casino changes schedule twice. The node test is skipped
when node is not installed.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

from bandits.agents import LINEUP
from bandits.env import Environment
from bandits.sim import make_for, run_agent

NODE = shutil.which("node")
SCRIPT = pathlib.Path(__file__).resolve().parent / "js" / "bandits_parity.mjs"
SEEDS = range(5)
HORIZON = 1200
K = 5


def _pairs() -> list[tuple[str, str]]:
    return [(kind, key) for kind, keys in LINEUP.items() for key in keys]


def _cases() -> list[tuple[str, str, int]]:
    return [(kind, key, seed) for kind, key in _pairs() for seed in SEEDS]


def _machines_json(env: Environment) -> dict:
    """The casino in the shape GET /api/bandits/machines returns, which the page hands to bandits-core.js."""
    m = env.machines
    return {"kind": m.kind, "k": m.k, "seed": m.seed, "pulls": m.horizon, "period": m.period,
            "schedule": [list(seg) for seg in m.schedule]}


def _python_play(kind: str, key: str, seed: int) -> dict:
    env = Environment(kind, K, HORIZON, seed)
    reasons: list[str] = []
    run = run_agent(make_for(key, env), env, on_pull=lambda t, arm, reward, agent: reasons.append(agent.reason))
    return {"arms": run.arms, "rewards": run.rewards, "regret": run.regret, "optimal": run.optimal,
            "reasons": reasons}


def test_lineups_are_17_agent_and_casino_pairs():
    # The README's parity claim counts these pairs; keep the two in step.
    assert len(_pairs()) == 17


def test_js_core_plays_the_same_casinos_as_python():
    if NODE is None:
        pytest.skip("node is not installed")
    request = {"runs": []}
    for kind, key, seed in _cases():
        env = Environment(kind, K, HORIZON, seed)
        request["runs"].append({"key": key, "machines": _machines_json(env)})
    proc = subprocess.run([NODE, str(SCRIPT)], input=json.dumps(request), capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr
    js_runs = json.loads(proc.stdout)["runs"]
    assert len(js_runs) == len(_cases())
    for case, js in zip(_cases(), js_runs):
        py = _python_play(*case)
        assert js["arms"] == py["arms"], case
        assert js["optimal"] == py["optimal"], case
        assert js["reasons"] == py["reasons"], case
        assert js["rewards"] == pytest.approx(py["rewards"], abs=1e-9), case
        assert js["regret"] == pytest.approx(py["regret"], abs=1e-9), case
