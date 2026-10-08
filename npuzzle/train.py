"""Train the learned 15-puzzle heuristic with approximate value iteration.

No solver labels are needed. Each round:
  1. Sample boards by random walks of 0..max_depth moves from the goal.
  2. Target y(s) = 0 at the goal, else 1 + min over children c of h_target(c),
     where h_target is a frozen copy of the network (the Bellman equation for
     unit-cost moves).
  3. Fit h(s) = PDB(s) + softplus(net(s)) to y with mean squared error.
  4. Copy the network into the target network.

Repeated rounds push accurate values outward from the goal, as in DeepCubeA
(Agostinelli et al., 2019). Starting from the pattern database instead of
from zero means the network only learns the PDB's underestimate.

    python -m npuzzle.train --minutes 60
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .batch import children, is_goal, pdb_batch, scramble
from .neural import WEIGHTS, one_hot

N = 4


class ResidualMLP(nn.Module):
    """input -> Linear -> ReLU -> [Linear -> ReLU -> Linear, + skip, ReLU] x blocks -> Linear."""

    def __init__(self, hidden=512, blocks=2):
        super().__init__()
        self.inp = nn.Linear(N**4, hidden)
        self.blocks = nn.ModuleList(
            nn.ModuleList([nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)])
            for _ in range(blocks)
        )
        self.out = nn.Linear(hidden, 1)
        # Start the residual near zero so h starts near the PDB.
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, -4.0)

    def forward(self, x):
        x = torch.relu(self.inp(x))
        for l1, l2 in self.blocks:
            x = torch.relu(x + l2(torch.relu(l1(x))))
        return self.out(x)[:, 0]


def heuristic(model, boards, pdb):
    """h = PDB + softplus(net) as a torch tensor; 0 at the goal."""
    x = torch.from_numpy(one_hot(boards))
    h = torch.from_numpy(pdb.astype(np.float32)) + nn.functional.softplus(model(x))
    goal = torch.from_numpy(is_goal(boards, N))
    return torch.where(goal, torch.zeros_like(h), h)


@torch.no_grad()
def bellman_targets(target, boards, chunk=50_000):
    kids, valid = children(boards, N)
    flat = kids.reshape(-1, N * N)
    h = np.concatenate(
        [heuristic(target, flat[i : i + chunk], pdb_batch(flat[i : i + chunk])).numpy()
         for i in range(0, len(flat), chunk)]
    ).reshape(len(boards), 4)
    h = np.where(valid, h, np.inf)
    y = 1.0 + h.min(axis=1)
    return np.where(is_goal(boards, N), 0.0, y).astype(np.float32)


def export(model, path=WEIGHTS):
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    arrays = {
        "in.weight": sd["inp.weight"], "in.bias": sd["inp.bias"],
        "out.weight": sd["out.weight"], "out.bias": sd["out.bias"],
        "blocks": np.array(len(model.blocks)),
    }
    for i in range(len(model.blocks)):
        for j in range(2):
            arrays[f"b{i}.{j}.weight"] = sd[f"blocks.{i}.{j}.weight"]
            arrays[f"b{i}.{j}.bias"] = sd[f"blocks.{i}.{j}.bias"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_eval(path):
    """Boards and optimal costs from a test set file, if one exists."""
    if not path.exists():
        return None
    rows = json.loads(path.read_text())["states"]
    if not rows:
        return None
    boards = np.array([r["board"] for r in rows], dtype=np.int8)
    return boards, np.array([r["optimal_cost"] for r in rows], dtype=np.float32)


@torch.no_grad()
def evaluate(model, data):
    boards, optimal = data
    h = heuristic(model, boards, pdb_batch(boards)).numpy()
    pdb = pdb_batch(boards)
    return {
        "mae": float(np.abs(h - optimal).mean()),
        "pdb_mae": float(np.abs(pdb - optimal).mean()),
        "overestimate_rate": float((h > optimal).mean()),
        "n": len(optimal),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("--batch", type=int, default=50_000, help="states sampled per round")
    p.add_argument("--minibatch", type=int, default=1000)
    p.add_argument("--max-depth", type=int, default=200)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", type=Path, default=WEIGHTS.parent / "testset_15_seed0.json")
    p.add_argument("--log", type=Path, default=Path("results/npuzzle_train_log.jsonl"))
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = ResidualMLP(args.hidden, args.blocks)
    target = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("w")

    start = time.time()
    round_ = 0
    while time.time() - start < args.minutes * 60:
        depths = rng.integers(0, args.max_depth + 1, size=args.batch)
        boards = scramble(args.batch, N, depths, rng)
        y = torch.from_numpy(bellman_targets(target, boards))
        pdb = pdb_batch(boards)

        model.train()
        order = rng.permutation(args.batch)
        losses = []
        for i in range(0, args.batch, args.minibatch):
            idx = order[i : i + args.minibatch]
            pred = heuristic(model, boards[idx], pdb[idx])
            loss = nn.functional.mse_loss(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        target.load_state_dict(model.state_dict())
        round_ += 1

        record = {"round": round_, "minutes": round((time.time() - start) / 60, 2),
                  "loss": float(np.mean(losses)), "mean_target": float(y.mean())}
        if round_ % 10 == 0:
            data = load_eval(args.eval)
            if data is not None:
                record["eval"] = evaluate(model.eval(), data)
            export(model)
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    export(model)
    print(f"wrote {WEIGHTS} after {round_} rounds")


if __name__ == "__main__":
    main()
