"""Seeded benchmark: how fast the autopilot busts ghosts, how sharp the beliefs are, and how close the particle
filters get to the exact belief.

Three experiments, all on fixed seeds (the same mazes and ghost walks every run):
1. play: the autopilot plays full games with the exact filter and with particle filters of several sizes.
   Reports turns to bust every ghost, how often all were busted, and the mean probability the belief puts
   on each ghost's true cell.
2. viterbi: for the exact-filter games, how often the Viterbi path (the most likely whole path, using every
   reading including the ones after a turn) matches the true cell at each turn, against the filtered MAP cell
   (the most likely cell from the readings so far only).
3. convergence: the same recorded pings replayed through the exact filter and through particle filters of
   growing N. Reports KL(exact || particles) over cells, and the mass each puts on the true cell.

KL uses floor 1e-6 on the particle marginal so an empty cell costs a finite amount. Times are single-core
Python wall-clock seconds per game and only indicate orders of magnitude.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from bandits.rng import Rng

from . import agent, motion
from .game import TURN_CAP, Game
from .hmm import ExactFilter
from .particles import ParticleFilter

ROOT = Path(__file__).resolve().parent.parent
MODELS = motion.MODELS
SIGMAS = {"low": 0.5, "med": 1.0, "high": 2.0}
PLAY_FILTERS = [("exact", 0), ("particles", 10), ("particles", 50), ("particles", 200)]
CONVERGENCE_N = [10, 30, 100, 300, 1000]
KL_FLOOR = 1e-6


def play(model: str, sigma: float, seed: int, filt: str, particles: int, size: int = 15, ghosts: int = 2):
    """One autopilot game. Also returns the belief checks: the probability on each true cell (summed and counted)
    and, per ghost, whether the filtered MAP cell was right at each turn it was live."""
    g = Game(size=size, ghosts=ghosts, model=model, sigma=sigma, seed=seed, filt=filt, particles=particles,
             max_turns=TURN_CAP)
    mass_sum, mass_n = 0.0, 0
    hits: dict[int, list[bool]] = {gh.id: [] for gh in g.ghosts}
    while not g.done:
        for gh in g.ghosts:
            if gh.live:
                marg = g.marginal(gh.id)
                mass_sum += marg[gh.k]
                mass_n += 1
                hits[gh.id].append(agent._argmax(marg) == gh.k)
        g.act(agent.choose(g))
    return g, mass_sum, mass_n, hits


def kl(exact: list[float], approx: list[float]) -> float:
    """KL(exact || approx) over cells, with the approximation floored at KL_FLOOR."""
    total = 0.0
    for e, q in zip(exact, approx):
        if e > 0.0:
            total += e * math.log(e / max(q, KL_FLOOR))
    return total


def play_block(seeds: int, size: int, ghosts: int) -> list[dict]:
    rows = []
    for model in MODELS:
        for name, sigma in SIGMAS.items():
            for filt, n in PLAY_FILTERS:
                turns, all_busted, mass_sum, mass_n, secs = [], 0, 0.0, 0, 0.0
                for seed in range(1, seeds + 1):
                    t0 = time.perf_counter()
                    g, ms, mn, _ = play(model, sigma, seed, filt, n, size, ghosts)
                    secs += time.perf_counter() - t0
                    turns.append(g.t)
                    all_busted += int(g.busted == ghosts)
                    mass_sum += ms
                    mass_n += mn
                rows.append({
                    "model": model, "noise": name, "sigma": sigma,
                    "filter": "exact" if filt == "exact" else f"particles-{n}",
                    "turns_mean": sum(turns) / len(turns), "turns_median": sorted(turns)[len(turns) // 2],
                    "all_busted": all_busted / seeds, "true_mass": mass_sum / max(1, mass_n),
                    "seconds_per_game": secs / seeds,
                })
    return rows


def viterbi_block(seeds: int, size: int, ghosts: int) -> list[dict]:
    """Per noise level and motion model: Viterbi path accuracy against the filtered MAP accuracy, on the turns
    of every ghost that was busted (its whole path is then observed). The filtered MAP cell at turn t uses
    readings 0..t only, so Viterbi, which also uses later readings, should not do worse."""
    rows = []
    for model in MODELS:
        for name, sigma in SIGMAS.items():
            v_hit = m_hit = n_turns = 0
            for seed in range(1, seeds + 1):
                g, _, _, hits = play(model, sigma, seed, "exact", 0, size, ghosts)
                for gh in g.ghosts:
                    if gh.bust_turn is None:
                        continue
                    path = g.viterbi_path(gh.id)
                    for t, cell in enumerate(path):
                        v_hit += int(cell == gh.true[t])
                        m_hit += int(hits[gh.id][t])
                        n_turns += 1
            rows.append({"model": model, "noise": name, "sigma": sigma,
                         "viterbi_acc": v_hit / max(1, n_turns), "filtered_map_acc": m_hit / max(1, n_turns),
                         "scored_turns": n_turns})
    return rows


def convergence_block(seeds: int, size: int, ghosts: int) -> list[dict]:
    """Replay recorded pings through the exact filter and particle filters of growing N (random model)."""
    out = []
    for name, sigma in SIGMAS.items():
        kls = {n: [] for n in CONVERGENCE_N}
        mass_pf = {n: [] for n in CONVERGENCE_N}
        mass_ex = []
        for seed in range(1, seeds + 1):
            g, _, _, _ = play("random", sigma, seed, "exact", 0, size, ghosts)
            for gh in g.ghosts:
                readings = gh.readings
                players = g.players[:len(readings)]
                exact = ExactFilter(g.maze, "random", g.prior, g.emis)
                pfs = {n: ParticleFilter(g.maze, "random", g.prior, g.emis, n,
                                         _rng(seed, gh.id, n)) for n in CONVERGENCE_N}
                for t, r in enumerate(readings):
                    if t > 0:
                        exact.predict(players[t])
                        for pf in pfs.values():
                            pf.predict(players[t])
                    exact.update(r, players[t])
                    for pf in pfs.values():
                        pf.update(r, players[t])
                    e = exact.marginal()
                    truth = gh.true[t]
                    mass_ex.append(e[truth])
                    for n, pf in pfs.items():
                        q = pf.marginal()
                        kls[n].append(kl(e, q))
                        mass_pf[n].append(q[truth])
        for n in CONVERGENCE_N:
            out.append({"noise": name, "sigma": sigma, "particles": n,
                        "kl_mean": sum(kls[n]) / len(kls[n]),
                        "true_mass_pf": sum(mass_pf[n]) / len(mass_pf[n]),
                        "true_mass_exact": sum(mass_ex) / len(mass_ex)})
    return out


def _rng(seed: int, gid: int, n: int):
    """A filter stream for one (seed, ghost, N), so each replayed particle filter gets its own draws."""
    return Rng((seed * 1009 + gid * 31 + n * 7 + 12345) & 0xFFFFFFFF)


def run(play_seeds: int = 12, conv_seeds: int = 6, size: int = 15, ghosts: int = 2) -> dict:
    """All three experiments with the committed seeds. Returns a JSON-ready dict."""
    return {
        "config": {"size": size, "ghosts": ghosts, "play_seeds": play_seeds, "convergence_seeds": conv_seeds,
                   "turn_cap": TURN_CAP, "stay_p": motion.STAY_P, "straight_p": motion.STRAIGHT_P,
                   "beta": motion.BETA, "sigmas": SIGMAS, "bust_threshold": agent.BUST_THRESHOLD},
        "play": play_block(play_seeds, size, ghosts),
        "viterbi": viterbi_block(play_seeds, size, ghosts),
        "convergence": convergence_block(conv_seeds, size, ghosts),
    }


def markdown(result: dict) -> str:
    """The benchmark as three Markdown tables, the same ones the CLI prints and the page quotes."""
    lines = ["| motion | noise (sigma) | filter | turns (mean) | turns (median) | all busted | true-cell mass "
             "| s/game |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in result["play"]:
        lines.append(f"| {r['model']} | {r['noise']} ({r['sigma']}) | {r['filter']} | {r['turns_mean']:.1f} | "
                     f"{r['turns_median']} | {r['all_busted']:.0%} | {r['true_mass']:.3f} | "
                     f"{r['seconds_per_game']:.3f} |")
    lines += ["", "| motion | noise (sigma) | Viterbi path accuracy | filtered MAP accuracy | scored turns |",
              "|---|---|---:|---:|---:|"]
    for r in result["viterbi"]:
        lines.append(f"| {r['model']} | {r['noise']} ({r['sigma']}) | {r['viterbi_acc']:.3f} | "
                     f"{r['filtered_map_acc']:.3f} | {r['scored_turns']} |")
    lines += ["", "| noise (sigma) | particles | KL(exact to particles) | true-cell mass (particles) | "
                  "true-cell mass (exact) |", "|---|---:|---:|---:|---:|"]
    for r in result["convergence"]:
        lines.append(f"| {r['noise']} ({r['sigma']}) | {r['particles']} | {r['kl_mean']:.4f} | "
                     f"{r['true_mass_pf']:.3f} | {r['true_mass_exact']:.3f} |")
    return "\n".join(lines)


def write(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ghosthunt_benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "ghosthunt_benchmark.md").write_text(markdown(result) + "\n")
