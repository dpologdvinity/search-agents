"""The benchmark: every agent on every kind of casino, over many seeds, against the Lai-Robbins floor.

    python -m bandits benchmark      # 100 seeds, horizon 10,000; writes results/bandits_benchmark.{json,md}

For each kind of casino and each seed, one machine set is drawn and every agent in that kind's lineup
plays it, so agents are compared on identical machines and identical outcomes. Each seed gives one
cumulative-regret curve per agent; the table and the committed curves are the mean over seeds with a
95% band (see sim.band).

Memory: runs are reduced to checkpoint values as they finish, so the whole benchmark holds one run's
lists at a time, not 7 x 100 full curves.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .agents import AGENTS, LINEUP
from .env import DRIFT_PERIOD, GAP, KINDS, SIGMA, Environment, lai_robbins_rate
from .sim import band, interval, play

K = 5
SEEDS = 100
HORIZON = 10_000
CHECKPOINTS = tuple(range(100, HORIZON + 1, 100))  # the committed curves are sampled every 100 pulls
TABLE_AT = (1000, HORIZON)  # the table reports regret at these pull counts
LAST = 1000  # "% best machine" is measured over the last LAST pulls, after the agents have settled


def run_benchmark(seeds: int = SEEDS, horizon: int = HORIZON, k: int = K, kinds: tuple[str, ...] = KINDS,
                  checkpoints: tuple[int, ...] | None = None, progress=None) -> dict:
    """Run every lineup on every kind for seeds 0..seeds-1. Returns the JSON-ready result dict."""
    # The lai-robbins floor is averaged over the seeds, so zero seeds would divide by zero.
    if seeds < 1:
        raise ValueError("the benchmark needs at least one seed")
    if horizon <= 1000:
        raise ValueError("the benchmark horizon must be more than 1000 pulls (the table reports T=1000 and T=horizon)")
    if checkpoints is None:
        checkpoints = tuple(t for t in CHECKPOINTS if t <= horizon)
    table_at = tuple(sorted({1000, horizon}))
    result = {
        "seeds": seeds,
        "horizon": horizon,
        "machines": k,
        "sigma": SIGMA,
        "drift_period": DRIFT_PERIOD,
        "min_gap": GAP,
        "checkpoints": list(checkpoints),
        "table_at": list(table_at),
        "last_pulls": LAST,
        "agents": {key: {"label": AGENTS[key].label, "blurb": AGENTS[key].blurb} for key in AGENTS},
        "settings": {},
    }
    for kind in kinds:
        # Per agent, per seed: checkpoint regrets, regret at the table pull counts, and share of best pulls.
        per_agent: dict[str, dict[str, list]] = {
            key: {"curve": [], "at": [], "share": []} for key in LINEUP[kind]
        }
        rates = []
        for seed in range(seeds):
            env = Environment(kind, k, horizon, seed)
            if progress:
                progress(kind, seed)
            if kind != "drifting":
                rates.append(lai_robbins_rate(env.mean(0), kind, env.sigma))
            for key in LINEUP[kind]:
                run = play(key, env)
                per_agent[key]["curve"].append([run.regret[t - 1] for t in checkpoints])
                per_agent[key]["at"].append([run.regret[t - 1] for t in table_at])
                per_agent[key]["share"].append(run.share_optimal(horizon - LAST))
        setting = {"agents": {}}
        if kind != "drifting":
            # The floor is averaged over the same machine sets; its curve is c * ln(t).
            c = sum(rates) / len(rates)
            setting["lai_robbins"] = {
                "rate": round(c, 3),
                "curve": [round(c * math.log(t), 3) for t in checkpoints],
            }
        for key in LINEUP[kind]:
            data = per_agent[key]
            mean, lo, hi = band(data["curve"])  # one band per checkpoint, over the seeds
            setting["agents"][key] = {
                "regret_at": {
                    str(t): _rounded(interval([row[j] for row in data["at"]]))
                    for j, t in enumerate(table_at)
                },
                "share_best_last": _rounded(interval(data["share"])),
                "curve": {
                    "mean": [round(x, 3) for x in mean],
                    "lo": [round(x, 3) for x in lo],
                    "hi": [round(x, 3) for x in hi],
                },
            }
        result["settings"][kind] = setting
    return result


def _rounded(triple: tuple[float, float, float]) -> dict:
    mean, lo, hi = triple
    return {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4)}


def to_markdown(result: dict) -> str:
    """Tables for the README and results/: one table per kind of casino, one row per agent."""
    t_lo, t_hi = result["table_at"]
    lines = [
        f"Cumulative regret (expected reward lost against the best machine), mean over {result['seeds']} seeds "
        f"with a 95% band, {result['machines']} machines. Lower is better.",
        "",
    ]
    names = {"bernoulli": "Bernoulli machines", "gaussian": "Gaussian machines (sigma 0.2)",
             "drifting": f"Drifting Bernoulli machines (redrawn every {result['drift_period']} pulls)"}
    for kind, setting in result["settings"].items():
        lines += [f"### {names[kind]}", "",
                  f"| Agent | Regret at T={t_lo:,} | Regret at T={t_hi:,} | % best machine, pulls "
                  f"{t_hi - result['last_pulls'] + 1:,}-{t_hi:,} |",
                  "| --- | ---: | ---: | ---: |"]
        for key, agent in setting["agents"].items():
            a = agent["regret_at"][str(t_lo)]
            b = agent["regret_at"][str(t_hi)]
            s = agent["share_best_last"]
            lines.append(f"| {result['agents'][key]['label']} | {_fmt(a)} | {_fmt(b)} | "
                         f"{100 * s['mean']:.1f}% ± {50 * (s['hi'] - s['lo']):.1f} |")
        if "lai_robbins" in setting:
            lr = setting["lai_robbins"]
            floor = lr["curve"][result["checkpoints"].index(t_hi)] if t_hi in result["checkpoints"] else None
            if floor is not None:
                lines += ["", f"Lai-Robbins floor: {lr['rate']} ln(t) (mean over machine sets), "
                          f"so {floor:.1f} at T={t_hi:,}."]
        lines.append("")
    return "\n".join(lines)


def _fmt(triple: dict) -> str:
    half = (triple["hi"] - triple["lo"]) / 2
    return f"{triple['mean']:.1f} ± {half:.1f}"


def write_results(result: dict, out_dir: str | Path) -> tuple[Path, Path]:
    """Write bandits_benchmark.json and .md into `out_dir`. Returns the two paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "bandits_benchmark.json"
    md_path = out / "bandits_benchmark.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    md_path.write_text(to_markdown(result))
    return json_path, md_path
