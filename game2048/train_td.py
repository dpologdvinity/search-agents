"""Train the 2048 n-tuple network by TD(0) on afterstates, over many games at once.

At each step every game picks the move maximizing reward + V(afterstate).
The previous afterstate's value then moves toward the observed one-step
return (Szubert and Jaskowski, 2014):

    V(s'_prev) += alpha * (r + V(s'_next) - V(s'_prev))

with target 0 when the game ends. All games advance in lockstep as NumPy
arrays; a finished game restarts immediately. Tables are trained in float32
and written to --out as float16 (see game2048.ntuple), so a resumed run starts
from the rounded values.

    python -m game2048.train_td --minutes 70 --games 1000 --alpha 0.1 --out /tmp/ntuple_2048.npz

The shipped tables are game2048/data/ntuple_2048.npz, the default --out, so write to another path
first and copy the result over only when it is meant to ship.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from .batch import max_exponent_batch, move_batch, new_games, spawn_batch
from .ntuple import PATTERNS, SYMMETRIES, WEIGHTS, NTupleNetwork


def td_update(net: NTupleNetwork, boards: np.ndarray, delta: np.ndarray, alpha: float):
    """Move each weight used by these boards by alpha times its mean TD error.

    Sequential TD adds alpha * delta per board. Across a batch, though, many
    boards share weights (early in a game most patterns are empty), and
    summing their updates multiplies the step for those weights by up to the
    batch size, which diverges. Averaging per weight keeps every weight's
    step at alpha * (mean error) regardless of batch size.
    """
    for table, idx in zip(net.tables, net.indices_batch(boards)):  # idx: (symmetries, B)
        per_feature = np.broadcast_to(delta, idx.shape).ravel()
        used, inverse = np.unique(idx.ravel(), return_inverse=True)
        total = np.bincount(inverse, weights=per_feature)
        count = np.bincount(inverse)
        table[used] += (alpha * total / count).astype(table.dtype)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--games", type=int, default=1000, help="games played in parallel")
    ap.add_argument("--alpha", type=float, default=0.1, help="total step size, split across the 48 lookups")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", type=Path, default=WEIGHTS, help="where to write the network")
    ap.add_argument("--log", type=Path, default=Path("results/game2048_train_log.jsonl"))
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    if args.resume and args.out.exists():
        net = NTupleNetwork.load(args.out, dtype=np.float32)
    else:
        net = NTupleNetwork()
    # Each board uses len(PATTERNS) * len(SYMMETRIES) lookups; split the step across them.
    alpha = args.alpha / (len(PATTERNS) * len(SYMMETRIES))

    n = args.games
    boards = new_games(n, rng)
    prev_after = np.zeros(n, dtype=np.uint64)
    has_prev = np.zeros(n, dtype=bool)
    scores = np.zeros(n, dtype=np.int64)
    recent = deque(maxlen=2000)  # (score, max exponent) of finished games
    finished = 0
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("a" if args.resume else "w")
    start = last_log = time.time()

    def log_record():
        """Append the statistics of the recent finished games to the log and print them."""
        s = np.array([x[0] for x in recent])
        m = np.array([x[1] for x in recent])
        record = {"minutes": round((time.time() - start) / 60, 2), "games": finished,
                  "mean_score": float(s.mean()),
                  **{f"reach_{1 << k}": float((m >= k).mean()) for k in (10, 11, 12, 13)}}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    while time.time() - start < args.minutes * 60:
        # Evaluate all four moves for every game.
        q = np.full((4, n), -np.inf)
        afters, rewards = [], []
        for d in range(4):
            after, reward = move_batch(boards, d)
            legal = after != boards
            afters.append(after)
            rewards.append(reward)
            if legal.any():
                q[d, legal] = reward[legal] + net.value_batch(after[legal])
        best = q.argmax(axis=0)
        terminal = np.isneginf(q.max(axis=0))
        rows = np.arange(n)
        chosen_after = np.stack(afters)[best, rows]
        chosen_reward = np.stack(rewards)[best, rows]

        # TD update of the previous afterstate toward the new one-step return.
        upd = has_prev
        if upd.any():
            target = np.where(terminal[upd], 0.0, q.max(axis=0)[upd])
            delta = target - net.value_batch(prev_after[upd])
            if not np.isfinite(delta).all():
                raise FloatingPointError("TD error is not finite; lower --alpha")
            td_update(net, prev_after[upd], delta, alpha)

        # Finished games: record and restart.
        if terminal.any():
            for s, m in zip(scores[terminal], max_exponent_batch(boards[terminal])):
                recent.append((int(s), int(m)))
            finished += int(terminal.sum())
            k = int(terminal.sum())
            boards[terminal] = new_games(k, rng)
            scores[terminal] = 0
            has_prev[terminal] = False

        live = ~terminal
        prev_after[live] = chosen_after[live]
        has_prev[live] = True
        scores[live] += chosen_reward[live]
        boards[live] = spawn_batch(chosen_after[live], rng)

        if time.time() - last_log > 60 and recent:
            last_log = time.time()
            log_record()
            net.save(args.out)
    # One last record, so the log also covers the time after the final full minute.
    if recent:
        log_record()
    log.close()
    net.save(args.out)
    print(f"wrote {args.out} after {finished} games")


if __name__ == "__main__":
    main()
