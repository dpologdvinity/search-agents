"""Neuroevolution: a genetic algorithm that evolves the weights of the snake's network.

No gradients. Each genome is the flat vector of all 339 weights. Per generation:

  1. Boards: draw fresh training seeds for this generation. Every genome in the generation plays
     the same boards (common random numbers), so genomes are compared fairly, but the next
     generation sees new boards, so no genome can memorise a fixed set.
  2. Fitness: play each genome on those boards and average the per-game score (see below).
  3. Elitism: the best `elites` genomes are copied unchanged, so the best result never gets lost.
  4. Tournament selection: pick `tournament` random genomes and keep the fittest, twice per child.
  5. Uniform crossover: each weight comes from one parent or the other with probability 1/2.
  6. Gaussian mutation: each weight moves by N(0, sigma) with probability `mutation_rate`.

Per-game score:
  * apples eaten, the main term;
  * a survival bonus of at most 0.25, for lasting up to 300 steps, so early learners that
    survive longer are ranked above ones that die at once;
  * a step cost of 0.0002 per move beyond 300, so a snake that merely drifts for thousands of
    steps is not paid for it;
  * a starvation penalty of 0.5 and no survival bonus when the game ends by starvation, so looping
    is worse than dying cleanly.

Genomes are scored in a pool of worker processes (`Settings.workers`, default 2). The result does
not depend on the number of workers: each genome's score is a pure function of its weights and the
boards.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool

import numpy as np

from .board import BOARD, Game
from .net import N_PARAMS, Net

# Training boards are drawn from [TRAIN_LO, TRAIN_HI). The benchmark's seeds start at 50 000,
# so the two never overlap.
TRAIN_LO, TRAIN_HI = 1_000, 40_000
SURVIVAL_STEPS = 300
SURVIVAL_BONUS = 0.25
STEP_COST = 0.0002  # per move beyond SURVIVAL_STEPS
STARVE_PENALTY = 0.5
# Fixed boards for choosing the champion: fresh per-generation boards make scores incomparable across generations.
# These seeds fall inside the training range above, so the champion's validation score is a same-distribution
# check, not a held-out one. The committed net was trained and selected this way; it is not retrained.
VALIDATION_SEEDS = list(range(20_000, 20_016))


@dataclass
class Settings:
    """Evolution hyperparameters. The defaults are the committed run; `python -m snake evolve --help` lists flags."""

    pop: int = 60
    games: int = 8
    generations: int = 80
    elites: int = 4
    tournament: int = 3
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.15
    init_sigma: float = 0.5
    seed: int = 1
    workers: int = 2
    starve_penalty: float = STARVE_PENALTY
    board: int = BOARD


@dataclass
class Score:
    """One genome's result over its games: fitness plus the plain numbers behind it."""

    fitness: float
    apples: float  # mean apples per game
    steps: float  # mean moves per game
    deaths: int  # games that ended in a wall or body collision
    starved: int  # games that ended by starvation


def play_net(net: Net, seed: int, size: int = BOARD) -> Game:
    """Play one full game with the network. The network sees only the position, never the RNG."""
    game = Game(seed, size)
    while game.alive:
        game.step(net.act(game))
    return game


def game_score(game: Game, starve_penalty: float = STARVE_PENALTY) -> float:
    """The per-game fitness described in the module docstring."""
    if game.cause == "starved":
        return game.apples - starve_penalty
    bonus = SURVIVAL_BONUS * min(game.steps, SURVIVAL_STEPS) / SURVIVAL_STEPS
    cost = STEP_COST * max(0, game.steps - SURVIVAL_STEPS)
    return game.apples + bonus - cost


def evaluate(genome: np.ndarray, seeds: list[int], size: int = BOARD,
             starve_penalty: float = STARVE_PENALTY) -> Score:
    """Average fitness over the given seeded games. Every genome sees exactly the same boards."""
    net = Net.from_genome(genome)
    total, apples, steps, deaths, starved = 0.0, 0, 0, 0, 0
    for seed in seeds:
        game = play_net(net, seed, size)
        total += game_score(game, starve_penalty)
        apples += game.apples
        steps += game.steps
        deaths += game.cause in ("wall", "body")
        starved += game.cause == "starved"
    n = len(seeds)
    return Score(total / n, apples / n, steps / n, deaths, starved)


def _tournament(rng: np.random.Generator, fitness: np.ndarray, k: int) -> int:
    """Index of the fittest of k random contestants (with replacement)."""
    contenders = rng.integers(0, len(fitness), size=k)
    return int(contenders[np.argmax(fitness[contenders])])


def _child(rng, pop, fitness, s: Settings) -> np.ndarray:
    """One offspring: tournament-selected parents, uniform crossover, then Gaussian mutation."""
    a = pop[_tournament(rng, fitness, s.tournament)]
    b = pop[_tournament(rng, fitness, s.tournament)]
    take_a = rng.random(N_PARAMS) < 0.5  # uniform crossover mask
    child = np.where(take_a, a, b)
    mutate = rng.random(N_PARAMS) < s.mutation_rate
    child[mutate] += rng.normal(0.0, s.mutation_sigma, size=int(mutate.sum()))
    return child


def evolve(s: Settings, on_generation=None) -> tuple[np.ndarray, Score, list[dict], int]:
    """Run the GA. Returns (champion genome, its validation score, one log record per generation, its generation).

    `on_generation(record, best_genome)` is called after each generation with the best-ever training genome,
    so the CLI can print progress and write checkpoints as it goes.
    Candidates for champion are the best-ever genome and the top five genomes of the last generation.
    They are re-scored on VALIDATION_SEEDS, a fixed board set, and the best of them is the champion.
    """
    rng = np.random.default_rng(s.seed)
    pop = rng.normal(0.0, s.init_sigma, size=(s.pop, N_PARAMS))
    best_genome, best_gen, best_fit = None, 0, -np.inf
    candidates: list[tuple[np.ndarray, int]] = []
    history: list[dict] = []
    t0 = time.perf_counter()
    pool = Pool(s.workers) if s.workers > 1 else None
    try:
        for gen in range(s.generations):
            seeds = [int(v) for v in rng.integers(TRAIN_LO, TRAIN_HI, size=s.games)]
            args = [(g, seeds, s.board, s.starve_penalty) for g in pop]
            scores = pool.starmap(evaluate, args) if pool else [evaluate(*a) for a in args]
            fitness = np.array([sc.fitness for sc in scores])
            order = np.argsort(-fitness, kind="stable")
            top = order[0]
            if best_genome is None or scores[top].fitness > best_fit:
                best_genome, best_fit, best_gen = pop[top].copy(), scores[top].fitness, gen + 1
            record = {
                "generation": gen + 1,
                "best_fitness": round(float(fitness[top]), 4),
                "mean_fitness": round(float(fitness.mean()), 4),
                "best_apples": round(scores[top].apples, 3),
                "mean_apples": round(float(np.mean([sc.apples for sc in scores])), 3),
                "best_steps": round(scores[top].steps, 1),
                "best_starved": scores[top].starved,
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }
            history.append(record)
            if on_generation is not None:
                on_generation(record, best_genome)
            if gen == s.generations - 1:
                candidates = [(pop[i].copy(), gen + 1) for i in order[:5]]
                break
            # Next generation: elites survive unchanged, the rest are children of tournament winners.
            nxt = [pop[i].copy() for i in order[: s.elites]]
            while len(nxt) < s.pop:
                nxt.append(_child(rng, pop, fitness, s))
            pop = np.array(nxt)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    # Champion: the best of the best-ever genome and the last generation's top five, on fixed boards.
    candidates.append((best_genome, best_gen))
    scored = []
    for genome, gen in candidates:
        scored.append((evaluate(genome, VALIDATION_SEEDS, s.board), genome, gen))
    score, genome, gen = max(scored, key=lambda t: t[0].fitness)
    return genome, score, history, gen


def settings_dict(s: Settings) -> dict:
    """The settings as a plain dict, for the champion file and the log."""
    return asdict(s)
