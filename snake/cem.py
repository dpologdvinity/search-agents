"""Cross-entropy method: evolves the eight weights of the evaluation-function agent (snake/evaluator.py).

The population is drawn from a diagonal Gaussian, one vector per candidate, and each candidate plays
seeded games with the evaluator's `choose`. Per generation:

  1. Boards: draw fresh training seeds for this generation. Every candidate plays the same boards
     (common random numbers), so candidates are compared fairly; the next generation sees new boards.
  2. Fitness: the mean per-game score over those boards, the same score the neural net is trained on
     (`evolve.game_score`): apples, a small survival bonus, and starvation counted as a death.
  3. Elites: the best `elites` candidates. The Gaussian is refit to them: mean = elite mean,
     standard deviation = elite spread, floored at `min_sigma` so the search never collapses to a point.

The champion is chosen like the net's: the best-ever candidate and the last generation's top five are
re-scored on fixed validation boards that no training generation can draw (seeds 40 000 to 40 031,
just above the training range), and the best of them is kept. The weights are rounded to four decimals
before that scoring, so the committed file scores exactly what was validated.

Candidates are scored in a pool of worker processes. A candidate's score is a pure function of its
weights and seeds, so the worker count does not change the result.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .board import BOARD, Game
from .evaluator import FEATURE_NAMES, N_FEATURES, choose
from .evolve import TRAIN_HI, TRAIN_LO, Score, game_score

# Fixed boards for choosing the champion. They sit above the training range (TRAIN_HI) and below the
# benchmark seeds (50 000 onward), so training, selection and the benchmark never share a board.
VALIDATION_SEEDS = list(range(40_000, 40_032))


@dataclass
class CEMSettings:
    """Cross-entropy hyperparameters. The defaults are the committed run; `python -m snake cem --help` lists flags."""

    pop: int = 48
    games: int = 6
    generations: int = 30
    elites: int = 12
    init_sigma: float = 1.0
    min_sigma: float = 0.05
    seed: int = 1
    workers: int = 2
    board: int = BOARD


def evaluate_weights(weights, seeds: list[int], size: int = BOARD) -> Score:
    """Mean game score and plain statistics over the given seeded games, with the weights fixed."""
    w = np.asarray(weights, dtype=np.float64)
    total, apples, steps, deaths, starved = 0.0, 0, 0, 0, 0
    for seed in seeds:
        game = Game(seed, size)
        while game.alive:
            game.step(choose(game, w))
        total += game_score(game)
        apples += game.apples
        steps += game.steps
        deaths += game.cause in ("wall", "body")
        starved += game.cause == "starved"
    n = len(seeds)
    return Score(total / n, apples / n, steps / n, deaths, starved)


def cem(s: CEMSettings, on_generation=None) -> tuple[np.ndarray, Score, list[dict], int]:
    """Run the cross-entropy method.

    Returns (champion weights, validation score, one record per generation, the champion's generation).

    `on_generation(record)` is called after each generation, so the CLI can print progress.
    """
    rng = np.random.default_rng(s.seed)
    mean = np.zeros(N_FEATURES)
    sigma = np.full(N_FEATURES, s.init_sigma)
    best_genome, best_gen, best_fit = None, 0, -np.inf
    candidates: list[tuple[np.ndarray, int]] = []
    history: list[dict] = []
    t0 = time.perf_counter()
    pool = Pool(s.workers) if s.workers > 1 else None
    try:
        for gen in range(s.generations):
            pop = mean + sigma * rng.standard_normal((s.pop, N_FEATURES))
            seeds = [int(v) for v in rng.integers(TRAIN_LO, TRAIN_HI, size=s.games)]
            args = [(g, seeds, s.board) for g in pop]
            scores = pool.starmap(evaluate_weights, args) if pool else [evaluate_weights(*a) for a in args]
            fitness = np.array([sc.fitness for sc in scores])
            order = np.argsort(-fitness, kind="stable")
            top = order[0]
            if scores[top].fitness > best_fit:
                best_genome, best_fit, best_gen = pop[top].copy(), scores[top].fitness, gen + 1
            record = {
                "generation": gen + 1,
                "best_fitness": round(float(fitness[top]), 4),
                "mean_fitness": round(float(fitness.mean()), 4),
                "best_apples": round(scores[top].apples, 3),
                "mean_apples": round(float(np.mean([sc.apples for sc in scores])), 3),
                "best_steps": round(scores[top].steps, 1),
                "best_starved": scores[top].starved,
                "sigma_mean": round(float(sigma.mean()), 4),
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }
            history.append(record)
            if on_generation is not None:
                on_generation(record)
            if gen == s.generations - 1:
                candidates = [(pop[i].copy(), gen + 1) for i in order[:5]]
                break
            elites = pop[order[: s.elites]]
            mean = elites.mean(axis=0)
            sigma = np.maximum(elites.std(axis=0), s.min_sigma)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    candidates.append((best_genome, best_gen))
    scored = []
    for genome, gen in candidates:
        rounded = np.round(genome, 4)
        scored.append((evaluate_weights(rounded, VALIDATION_SEEDS, s.board), rounded, gen))
    score, weights, gen = max(scored, key=lambda t: t[0].fitness)
    return weights, score, history, gen


def settings_dict(s: CEMSettings) -> dict:
    """The settings as a plain dict, for the weights file and the log."""
    return asdict(s)


def write_weights(path: Path, weights, meta: dict) -> None:
    """Write the weight file: the named features and weights, then the training metadata."""
    data = {"features": list(FEATURE_NAMES), "weights": [float(v) for v in weights], **meta}
    path.write_text(json.dumps(data, indent=1) + "\n")

