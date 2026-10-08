"""Genetic algorithm that tunes the nine feature weights.

A genome is the weight vector. Its fitness is the mean number of lines cleared before game over over a set
of seeded training games on the training board (BOARD_HEIGHT rows, a piece cap per game). Each game is
played until the stack tops out or the cap is reached, so fitness measures survival as well as clearing.

Seeds are fresh for every generation: generation g plays games with seeds TRAIN_SEED_BASE + g * games + i.
All genomes in one generation share those seeds, so they are compared on the same games (common random
numbers), but no generation sees the games of an earlier one. That stops the GA from fitting a fixed set of
games. Benchmark seeds (tetris/benchmark.py) are a separate range that the GA never plays.

One generation:
  1. elitism: the best `elites` genomes are copied over unchanged;
  2. tournament selection: pick `tournament` random genomes, keep the fittest, twice per child;
  3. arithmetic crossover: child = a * parent1 + (1 - a) * parent2, with a fresh a for each gene;
  4. Gaussian mutation: each gene gets noise with probability `mutation_rate`, clipped to the limits.

Genome evaluation is independent, so the population is evaluated in a process pool. The result depends
only on the seeds, not on the number of workers.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

from .features import FEATURES, WEIGHT_LIMIT
from .game import greedy_policy, play

TRAIN_SEED_BASE = 1000  # generation g uses seeds TRAIN_SEED_BASE + g * games + i; benchmarks use 50000+


@dataclass(frozen=True)
class GAConfig:
    population: int = 16
    generations: int = 16
    elites: int = 2
    tournament: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2  # per gene
    sigma: float = 0.3  # Gaussian mutation step, in weight units
    train_games: int = 6  # games per genome per generation (fresh seeds each generation)
    train_pieces: int = 2500  # piece cap per training game; a cap keeps a generation to about a minute
    board_height: int = 10  # rows of the training board (the harder setting; see benchmark.py)
    seed: int = 0


def fitness(weights, seeds, max_pieces: int, height: int = 10) -> float:
    """Mean lines cleared by the greedy bot with these weights before game over (or the piece cap)."""
    policy = greedy_policy(weights)
    return sum(play(policy, s, max_pieces, height=height).lines for s in seeds) / len(seeds)


def generation_seeds(gen: int, games: int) -> list[int]:
    """The training seeds for one generation: never reused by another generation."""
    return [TRAIN_SEED_BASE + gen * games + i for i in range(games)]


def _evaluate(task) -> float:
    weights, seeds, max_pieces, height = task
    return fitness(weights, seeds, max_pieces, height)


def _clip(x: float) -> float:
    return max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, x))


def _tournament(pop, fits, rng: random.Random, k: int):
    picks = rng.sample(range(len(pop)), k)
    best = max(picks, key=lambda i: fits[i])
    return pop[best]


def _crossover(p1, p2, rng: random.Random, rate: float):
    if rng.random() >= rate:
        return list(p1)
    child = []
    for a, b in zip(p1, p2):
        t = rng.random()
        child.append(t * a + (1 - t) * b)
    return child


def _mutate(genome, rng: random.Random, rate: float, sigma: float):
    return [_clip(g + rng.gauss(0.0, sigma)) if rng.random() < rate else g for g in genome]


def evolve(cfg: GAConfig, workers: int = 2, log_path: Path | None = None, out=print) -> dict:
    """Run the GA and return {'weights', 'fitness', 'history', 'config'}.

    The fitness of the returned genome is its best training score, on the games of the generation in which
    it was found. Each generation is appended to log_path as one JSON line.
    """
    rng = random.Random(cfg.seed)
    pop = [[rng.uniform(-1.0, 1.0) for _ in FEATURES] for _ in range(cfg.population)]
    history = []
    best_w, best_f = None, float("-inf")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")  # a fresh run replaces the old log

    with Pool(workers) as pool:
        for gen in range(cfg.generations + 1):
            t0 = time.time()
            seeds = generation_seeds(gen, cfg.train_games)
            tasks = [(w, seeds, cfg.train_pieces, cfg.board_height) for w in pop]
            fits = pool.map(_evaluate, tasks, chunksize=1)
            order = sorted(range(len(pop)), key=lambda i: -fits[i])
            if fits[order[0]] > best_f:
                best_f, best_w = fits[order[0]], list(pop[order[0]])
            row = {
                "gen": gen,
                "best": round(fits[order[0]], 3),
                "mean": round(sum(fits) / len(fits), 3),
                "worst": round(fits[order[-1]], 3),
                "best_weights": [round(w, 4) for w in pop[order[0]]],
                "seeds": seeds,
                "seconds": round(time.time() - t0, 2),
            }
            history.append({k: row[k] for k in ("gen", "best", "mean", "worst")})
            out(f"gen {gen:3d}  best {row['best']:8.2f}  mean {row['mean']:8.2f}  worst {row['worst']:8.2f}  "
                f"({row['seconds']:.1f} s)")
            if log_path is not None:
                with log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
            if gen == cfg.generations:
                break

            # Next generation: elites first, then children from tournament parents.
            nxt = [list(pop[i]) for i in order[: cfg.elites]]
            while len(nxt) < cfg.population:
                p1 = _tournament(pop, fits, rng, cfg.tournament)
                p2 = _tournament(pop, fits, rng, cfg.tournament)
                child = _crossover(p1, p2, rng, cfg.crossover_rate)
                nxt.append(_mutate(child, rng, cfg.mutation_rate, cfg.sigma))
            pop = nxt

    return {"weights": [round(w, 4) for w in best_w], "fitness": round(best_f, 3), "history": history,
            "config": asdict(cfg)}


def save_tuned(result: dict, path: Path) -> None:
    """Write the tuned weights, the GA settings and the per-generation history as JSON."""
    doc = {
        "features": list(FEATURES),
        "weights": result["weights"],
        "train_fitness": result["fitness"],
        "config": result["config"],
        "history": result["history"],
    }
    path.write_text(json.dumps(doc, indent=1) + "\n")
