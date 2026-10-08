"""AlphaZero-style training for Connect Four: self-play, then fit the network to the results.

Each iteration:
  1. Play `games` self-play games at once. Every move runs PUCT search
     guided by the current network; all games share one batched network
     call per simulation round. Root priors get Dirichlet noise, and the
     first moves are sampled from visit counts for variety.
  2. Store (position, visit distribution pi, final result z) for every move,
     plus left-right mirror images, in a replay buffer.
  3. Train on minibatches from the buffer:
     loss = (v - z)^2 - pi . log p   (+ weight decay)
  4. Every few iterations, play the network (with search) against
     alpha-beta minimax at several depths and log the score.

    python -m connect4.train_az --hours 2
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .board import COLS, ROWS, Board
from .minimax import Minimax
from .net import WEIGHTS, encode, legal_mask, masked_softmax
from .puct import Tree, run_simulations

CHECKPOINT = Path("checkpoints/az_connect4.pt")


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = torch.relu(self.n1(self.c1(x)))
        return torch.relu(x + self.n2(self.c2(y)))


class AZNet(nn.Module):
    def __init__(self, channels=32, blocks=4):
        super().__init__()
        self.stem = nn.Conv2d(2, channels, 3, padding=1, bias=False)
        self.stem_n = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(ResBlock(channels) for _ in range(blocks))
        self.pol_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.pol_n = nn.BatchNorm2d(2)
        self.pol_fc = nn.Linear(2 * ROWS * COLS, COLS)
        self.val_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.val_n = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(ROWS * COLS, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        h = torch.relu(self.stem_n(self.stem(x)))
        for block in self.blocks:
            h = block(h)
        p = torch.relu(self.pol_n(self.pol_conv(h))).flatten(1)
        v = torch.relu(self.val_n(self.val_conv(h))).flatten(1)
        v = torch.tanh(self.val_fc2(torch.relu(self.val_fc1(v))))
        return self.pol_fc(p), v[:, 0]


def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Conv followed by eval-mode batch norm, as one conv with bias."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale[:, None, None, None]
    b = bn.bias - bn.running_mean * scale
    return w.detach().numpy().astype(np.float32), b.detach().numpy().astype(np.float32)


def export(model: AZNet, path=WEIGHTS):
    model.eval()
    arrays = {"blocks": np.array(len(model.blocks))}
    arrays["stem.w"], arrays["stem.b"] = _fold(model.stem, model.stem_n)
    for i, blk in enumerate(model.blocks):
        arrays[f"b{i}.1.w"], arrays[f"b{i}.1.b"] = _fold(blk.c1, blk.n1)
        arrays[f"b{i}.2.w"], arrays[f"b{i}.2.b"] = _fold(blk.c2, blk.n2)
    arrays["pol.conv.w"], arrays["pol.conv.b"] = _fold(model.pol_conv, model.pol_n)
    arrays["val.conv.w"], arrays["val.conv.b"] = _fold(model.val_conv, model.val_n)
    for name, layer in (("pol.fc", model.pol_fc), ("val.fc1", model.val_fc1), ("val.fc2", model.val_fc2)):
        arrays[f"{name}.w"] = layer.weight.detach().numpy()
        arrays[f"{name}.b"] = layer.bias.detach().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def torch_evaluator(model: AZNet):
    @torch.inference_mode()
    def evaluate(boards):
        model.eval()
        logits, values = model(torch.from_numpy(encode(boards)))
        return masked_softmax(logits.numpy(), legal_mask(boards)), values.numpy()
    return evaluate


def self_play(model, games, simulations, rng, temperature_moves=8):
    """Play `games` games at once; returns training rows (planes, pi, z)."""
    evaluate = torch_evaluator(model)
    trees = [Tree(Board(), noise_fraction=0.25, dirichlet_alpha=1.0, rng=rng) for _ in range(games)]
    history = [[] for _ in range(games)]  # (board, pi) per move
    results = [None] * games  # result for the player who moved first
    active = list(range(games))
    while active:
        run_simulations([trees[g] for g in active], evaluate, simulations)
        still = []
        for g in active:
            tree = trees[g]
            board = tree.root.board
            explore = board.moves < temperature_moves
            pi = tree.policy(1.0)
            action = int(rng.choice(COLS, p=pi)) if explore else int(pi.argmax())
            history[g].append((board, pi))
            nxt = board.play(action)
            if nxt.last_player_won():
                results[g] = 1.0 if board.to_move_is_first() else -1.0
            elif nxt.is_full():
                results[g] = 0.0
            else:
                tree.advance(action)
                still.append(g)
        active = still

    rows = []
    for g in range(games):
        for board, pi in history[g]:
            z = results[g] if board.to_move_is_first() else -results[g]
            rows.append((board, pi, z))
            rows.append((board.mirror(), pi[::-1].copy(), z))
    return rows, results


def play_match(model, opponent_depth, games, simulations, rng):
    """Network + PUCT vs minimax at a fixed depth, alternating colors. Returns score in [0, 1]."""
    evaluate = torch_evaluator(model)
    mm = Minimax(max_depth=opponent_depth)
    score = 0.0
    for i in range(games):
        board = Board()
        # Two random opening moves so games differ.
        for _ in range(2):
            board = board.play(int(rng.integers(COLS)))
        net_first = i % 2 == 0
        while True:
            net_to_move = board.to_move_is_first() == net_first
            if net_to_move:
                tree = Tree(board)
                run_simulations([tree], evaluate, simulations)
                col = int(tree.root.visits.argmax())
            else:
                col = mm.search(board).move
            if board.is_winning_move(col):
                score += 1.0 if net_to_move else 0.0
                break
            board = board.play(col)
            if board.is_full():
                score += 0.5
                break
    return score / games


def train_step(model, opt, batch, device="cpu"):
    boards, pis, zs = zip(*batch)
    x = torch.from_numpy(encode(list(boards)))
    pi = torch.from_numpy(np.stack(pis).astype(np.float32))
    z = torch.tensor(zs, dtype=torch.float32)
    model.train()
    logits, v = model(x)
    value_loss = nn.functional.mse_loss(v, z)
    policy_loss = -(pi * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss = value_loss + policy_loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    return value_loss.item(), policy_loss.item()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--games", type=int, default=128, help="self-play games per iteration")
    p.add_argument("--simulations", type=int, default=100)
    p.add_argument("--buffer", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=400, help="training steps per iteration")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log", type=Path, default=Path("results/connect4_train_log.jsonl"))
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    model = AZNet(args.channels, args.blocks)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer: deque = deque(maxlen=args.buffer)
    iteration, games_played = 0, 0
    if args.resume and CHECKPOINT.exists():
        state = torch.load(CHECKPOINT, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        iteration, games_played = state["iteration"], state["games"]
        print(f"resumed at iteration {iteration}")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("a" if args.resume else "w")
    start = time.time()
    while time.time() - start < args.hours * 3600:
        t0 = time.time()
        rows, results = self_play(model, args.games, args.simulations, rng)
        buffer.extend(rows)
        games_played += args.games
        t1 = time.time()

        losses = [train_step(model, opt, py_rng.sample(buffer, min(args.batch, len(buffer))))
                  for _ in range(args.steps)]
        iteration += 1
        record = {
            "iteration": iteration, "games": games_played, "buffer": len(buffer),
            "minutes": round((time.time() - start) / 60, 2),
            "selfplay_s": round(t1 - t0, 1), "train_s": round(time.time() - t1, 1),
            "value_loss": float(np.mean([v for v, _ in losses])),
            "policy_loss": float(np.mean([p for _, p in losses])),
            "first_player_score": float(np.mean([(r + 1) / 2 for r in results])),
            "mean_game_length": len(rows) / 2 / args.games,
        }
        if iteration % args.eval_every == 0:
            record["vs_minimax"] = {d: play_match(model, d, 20, 100, rng) for d in (2, 4, 6)}
        CHECKPOINT.parent.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "iteration": iteration, "games": games_played}, CHECKPOINT)
        export(model)
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
