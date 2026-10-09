"""Noisy cross-entropy method (CEM) for the nine feature weights, after Szita and Lorincz (2006).

CEM keeps a Gaussian over weight vectors, with a mean mu and a per-weight variance sigma^2. Each iteration:
  1. sample `samples` weight vectors from N(mu, sigma^2), clipped to the weight limits;
  2. score each one by the mean lines it clears before game over over `games` fresh seeded games;
  3. keep the top `elite_frac` (10-20% is the usual choice) and refit mu and sigma^2 to them;
  4. add a decreasing extra variance z_t to sigma^2 (the "noisy" part), so the search does not
     collapse onto one early elite. z_t shrinks to a small floor over the run.

The mean starts at the hand-picked weights, so the first distribution is centred on a strong baseline.
Every iteration uses fresh seeds (CEM_SEED_BASE + iteration * games + k), shared by all samples in that
iteration, so samples are compared on equal games and no game is reused across iterations. Games are played to
game over with a piece cap; the benchmark uses the same cap, so training and evaluation match.

The state (mu, sigma, history, best sample so far) is written to a checkpoint every few iterations, and
`resume=True` continues from it.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

from .features import FEATURES, HAND_WEIGHTS, WEIGHT_LIMIT
from .game import greedy_policy, play

CEM_SEED_BASE = 200_000  # disjoint from the GA seeds (1000+) and the benchmark seeds (50000+)


@dataclass(frozen=True)
class CEMConfig:
    iterations: int = 30
    samples: int = 30
    elite_frac: float = 0.15  # top 15% of the samples refit the distribution
    games: int = 3  # fresh seeded games per sample per iteration
    cap: int = 100_000  # piece cap; matches the held-out benchmark
    height: int = 10  # training board height, matches the benchmark
    init_sigma: float = 1.0  # starting standard deviation of each weight
    noise0: float = 1.0  # extra variance added at iteration 0 ...
    noise_decay: float = 20.0  # ... shrinking linearly to noise_min over this many iterations
    noise_min: float = 0.05
    checkpoint_every: int = 3
    seed: int = 0


def noise(t: int, cfg: CEMConfig) -> float:
    """Extra variance z_t added to sigma^2 at iteration t: starts at noise0, falls to noise_min."""
    return max(cfg.noise_min, cfg.noise0 * (1.0 - t / cfg.noise_decay))


def iteration_seeds(it: int, games: int) -> list[int]:
    return [CEM_SEED_BASE + it * games + k for k in range(games)]


def _clip(x: float) -> float:
    return max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, x))


def _evaluate(task) -> float:
    weights, seeds, cap, height = task
    policy = greedy_policy(weights)
    return sum(play(policy, s, cap, height=height).lines for s in seeds) / len(seeds)


def run_cem(cfg: CEMConfig, workers: int = 2, checkpoint: Path | None = None, log_path: Path | None = None,
            resume: bool = False, out=print) -> dict:
    """Run CEM and return {'weights', 'fitness', 'history', 'config', 'sigma', 'best'}.

    weights is the final mean mu. best is the best sampled vector seen (by its training score on that
    iteration's seeds), which is optimistic and is reported only for reference.
    """
    rng = random.Random(cfg.seed)
    n_feat = len(FEATURES)
    mu = list(HAND_WEIGHTS)
    sigma = [cfg.init_sigma] * n_feat
    history: list[dict] = []
    best = {"fitness": float("-inf"), "weights": None, "iteration": None}
    start = 0
    if resume and checkpoint is not None and checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        mu, sigma, history, best = state["mu"], state["sigma"], state["history"], state["best"]
        start = state["iteration"] + 1
        out(f"resumed from iteration {state['iteration']}")
    elif log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")

    n_elite = max(2, math.ceil(cfg.elite_frac * cfg.samples))
    with Pool(workers) as pool:
        for it in range(start, cfg.iterations):
            t0 = time.time()
            seeds = iteration_seeds(it, cfg.games)
            samples = [[_clip(rng.gauss(m, s)) for m, s in zip(mu, sigma)] for _ in range(cfg.samples)]
            tasks = [(w, seeds, cfg.cap, cfg.height) for w in samples]
            fits = pool.map(_evaluate, tasks, chunksize=1)
            order = sorted(range(cfg.samples), key=lambda i: -fits[i])
            elites = [samples[i] for i in order[:n_elite]]
            if fits[order[0]] > best["fitness"]:
                best = {"fitness": round(fits[order[0]], 3), "weights": [round(w, 4) for w in samples[order[0]]],
                        "iteration": it}
            # Refit: mean of the elites, their variance, plus the decreasing extra noise z_t.
            z = noise(it, cfg)
            mu = [sum(e[j] for e in elites) / n_elite for j in range(n_feat)]
            var = [sum((e[j] - mu[j]) ** 2 for e in elites) / n_elite + z for j in range(n_feat)]
            sigma = [math.sqrt(v) for v in var]
            row = {
                "iteration": it,
                "best_sample": round(fits[order[0]], 3),
                "elite_mean": round(sum(fits[i] for i in order[:n_elite]) / n_elite, 3),
                "population_mean": round(sum(fits) / len(fits), 3),
                "mean_sigma": round(sum(sigma) / n_feat, 4),
                "noise": round(z, 4),
                "mu": [round(w, 4) for w in mu],
                "seeds": seeds,
                "seconds": round(time.time() - t0, 1),
            }
            history.append({k: row[k] for k in ("iteration", "best_sample", "elite_mean", "population_mean",
                                                 "mean_sigma")})
            out(f"iter {it:3d}  best {row['best_sample']:8.1f}  elite mean {row['elite_mean']:8.1f}  "
                f"sigma {row['mean_sigma']:.3f}  ({row['seconds']:.0f} s)")
            if log_path is not None:
                with log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            if checkpoint is not None and ((it + 1) % cfg.checkpoint_every == 0 or it == cfg.iterations - 1):
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(json.dumps({"iteration": it, "mu": mu, "sigma": sigma, "history": history,
                                                  "best": best}) + "\n")

    return {"weights": [round(w, 4) for w in mu], "sigma": [round(s, 4) for s in sigma],
            "fitness": history[-1]["elite_mean"] if history else None, "history": history,
            "best": best, "config": asdict(cfg)}
