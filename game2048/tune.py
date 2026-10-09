"""Tune the hand-crafted 2048 evaluation weights with the cross-entropy method.

Each generation samples candidate weight vectors from a Gaussian, scores
each by the mean final score of greedy one-move-lookahead games (all games
played in lockstep with NumPy, same random seed for every candidate), then
refits the Gaussian to the best candidates.

Every feature is oriented so that a larger value is better (penalties such
as smoothness are zero or negative), so every weight is kept at zero or
above. Without that bound the tuner could pick a negative weight on a
penalty, which rewards the disorder the feature measures.

    python -m game2048.tune --generations 20 --population 32 --elite 8 --games 300 --seed 0

To compare saved weight files on the same held-out games (writes a .json and a .md report):

    python -m game2048.tune --evaluate game2048/data/heuristic_weights.json \\
        results/game2048_heuristic_weights_v2.json --games 3000 --seed 500000 \\
        --out results/game2048_heuristic_greedy_heldout.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .batch import move_batch, new_games, spawn_batch
from .expectimax import FEATURES, WEIGHTS, features_batch, load_weights


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


def evaluate(paths, games: int, seed: int, out: Path, command: str):
    """Score saved weight files on the same greedy games and write a held-out report.

    Every file plays the same boards (one seed for the whole batch), so the gap
    between rows comes from the weights, not from which games were drawn. The
    95% interval is normal on the sample standard deviation.
    """
    rows = []
    for path in paths:
        recorded = json.loads(Path(path).read_text())
        scores = greedy_scores(np.array(load_weights(path)), games, seed=seed).astype(float)
        std = scores.std(ddof=1)
        half = 1.96 * std / np.sqrt(games)
        rows.append({"file": str(path), "smooth": recorded["weights"]["smooth"],
                     "mean": float(scores.mean()), "std": float(std),
                     "ci95": [float(scores.mean() - half), float(scores.mean() + half)],
                     "recorded_greedy_mean_score": recorded.get("greedy_mean_score")})
        print(json.dumps(rows[-1]), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"command": command, "games": games, "seed": seed, "results": rows}, indent=1))
    lines = [f"Command: `{command}`", "",
             f"Greedy one-move lookahead, {games} games, one batch seed {seed}; 95% normal interval.", "",
             "| Weights file | Smoothness weight | Mean score (95% CI) | Std | Recorded greedy_mean_score |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['file']}` | {r['smooth']:g} | "
                     f"{r['mean']:,.0f} ({r['ci95'][0]:,.0f} to {r['ci95'][1]:,.0f}) | "
                     f"{r['std']:,.0f} | {r['recorded_greedy_mean_score']:,.2f} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out} and {out.with_suffix('.md')}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--elite", type=int, default=4)
    p.add_argument("--games", type=int, default=200, help="games per candidate per generation")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the Gaussian sampler (with --evaluate: the shared game seed)")
    p.add_argument("--out", type=Path, default=None, help=f"weights file to write (default {WEIGHTS})")
    p.add_argument("--evaluate", nargs="+", type=Path, metavar="WEIGHTS_JSON",
                   help="instead of tuning, score these weight files on held-out greedy games "
                        "and write --out (.json and .md)")
    args = p.parse_args(argv)

    if args.evaluate:
        if args.out is None:
            p.error("--evaluate needs --out")
        command = "python -m game2048.tune " + " ".join(sys.argv[1:] if argv is None else argv)
        return evaluate(args.evaluate, args.games, args.seed, args.out, command)
    args.out = args.out or WEIGHTS

    rng = np.random.default_rng(args.seed)
    mean = np.ones(len(FEATURES))
    std = np.full(len(FEATURES), 2.0)
    history = []
    best_w, best_score = mean.copy(), -np.inf
    for gen in range(args.generations):
        # Clip at zero: weights of penalty features must not turn into rewards.
        candidates = np.maximum(mean + std * rng.standard_normal((args.population, len(FEATURES))), 0.0)
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
    weights = final[pick] / max(final[pick].max(), 1e-12)  # scale is irrelevant to move choice
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "weights": dict(zip(FEATURES, map(float, weights.round(4)))),
        "method": "cross-entropy method on greedy one-move-lookahead play, weights bounded at zero",
        "settings": {"generations": args.generations, "population": args.population, "elite": args.elite,
                     "games_per_candidate": args.games, "seed": args.seed,
                     "generation_seeds": "1000 + generation", "final_rescore_seed": 99,
                     "final_rescore_games": args.games * 2},
        "greedy_mean_score": scored[pick], "generations": history,
    }, indent=1))
    print(f"wrote {args.out}: {scored}")


if __name__ == "__main__":
    main()
