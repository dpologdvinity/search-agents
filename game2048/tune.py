"""Tune the hand-crafted 2048 evaluation weights with the cross-entropy method.

Each generation samples candidate weight vectors from a Gaussian, scores
each by the mean final score of greedy one-move-lookahead games (all games
played in lockstep with NumPy, same random seed for every candidate), then
refits the Gaussian to the best candidates.

    python -m game2048.tune --generations 15
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .batch import move_batch, new_games, spawn_batch
from .expectimax import FEATURES, WEIGHTS, features_batch


def greedy_scores(weights: np.ndarray, games: int, seed: int, max_moves: int = 5000) -> np.ndarray:
    """Final scores of greedy play: each move maximizes weights . features(afterstate)."""
    rng = np.random.default_rng(seed)
    boards = new_games(games, rng)
    scores = np.zeros(games, dtype=np.int64)
    done = np.zeros(games, dtype=bool)
    for _ in range(max_moves):
        q = np.full((4, games), -np.inf)
        afters, gains = [], []
        for d in range(4):
            after, gained = move_batch(boards, d)
            legal = (after != boards) & ~done
            afters.append(after)
            gains.append(gained)
            if legal.any():
                q[d, legal] = features_batch(after[legal]) @ weights
        done |= np.isneginf(q.max(axis=0))
        if done.all():
            break
        live = ~done
        best = q.argmax(axis=0)
        rows = np.arange(games)
        scores[live] += np.stack(gains)[best, rows][live]
        boards[live] = spawn_batch(np.stack(afters)[best, rows][live], rng)
    return scores


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--elite", type=int, default=4)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    mean = np.ones(len(FEATURES))
    std = np.full(len(FEATURES), 2.0)
    history = []
    best_w, best_score = mean.copy(), -np.inf
    for gen in range(args.generations):
        candidates = mean + std * rng.standard_normal((args.population, len(FEATURES)))
        fitness = np.array([greedy_scores(w, args.games, seed=1000 + gen).mean() for w in candidates])
        elite = candidates[np.argsort(fitness)[-args.elite:]]
        mean, std = elite.mean(axis=0), elite.std(axis=0) + 0.05
        top = int(fitness.argmax())
        if fitness[top] > best_score:
            best_score, best_w = float(fitness[top]), candidates[top].copy()
        history.append({"generation": gen, "best": float(fitness.max()), "mean": float(fitness.mean())})
        print(json.dumps(history[-1]), flush=True)

    # Re-score the best candidate and the final mean on fresh games, keep the better one.
    final = {"best_seen": best_w, "final_mean": mean}
    scored = {k: float(greedy_scores(w, args.games * 2, seed=99).mean()) for k, w in final.items()}
    pick = max(scored, key=scored.get)
    weights = final[pick] / np.abs(final[pick]).max()  # scale is irrelevant to move choice
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS.write_text(json.dumps({
        "weights": dict(zip(FEATURES, map(float, weights.round(4)))),
        "method": "cross-entropy method on greedy one-move-lookahead play",
        "greedy_mean_score": scored[pick], "generations": history,
    }, indent=1))
    print(f"wrote {WEIGHTS}: {scored}")


if __name__ == "__main__":
    main()
