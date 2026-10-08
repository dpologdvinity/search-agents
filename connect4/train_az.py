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

The network runs on CPU or CUDA (--device). On CUDA, --amp runs the forward
passes in fp16 with loss scaling. Checkpoints load onto whichever device is
in use, so a CUDA checkpoint resumes on a CPU laptop and the reverse.

    python -m connect4.train_az --hours 2
    python -m connect4.train_az --resume --device cuda --amp --games 512 --batch 512
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from collections.abc import Callable
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


def resolve_device(name: str) -> torch.device:
    """Map --device to a torch device. "auto" picks CUDA when torch can see a GPU, else CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")
    return torch.device(name)


def _autocast(device: torch.device, amp: bool):
    """fp16 autocast on CUDA. On CPU it is a no-op, so CPU results stay exact fp32."""
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda")


def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Conv followed by eval-mode batch norm, as one conv with bias."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale[:, None, None, None]
    b = bn.bias - bn.running_mean * scale
    return w.detach().cpu().numpy().astype(np.float32), b.detach().cpu().numpy().astype(np.float32)


def export(model: AZNet, path=WEIGHTS):
    """Write the NumPy inference weights. Works for a model on any device: tensors are copied to the CPU first."""
    model.eval()
    path = Path(path)
    arrays = {"blocks": np.array(len(model.blocks))}
    arrays["stem.w"], arrays["stem.b"] = _fold(model.stem, model.stem_n)
    for i, blk in enumerate(model.blocks):
        arrays[f"b{i}.1.w"], arrays[f"b{i}.1.b"] = _fold(blk.c1, blk.n1)
        arrays[f"b{i}.2.w"], arrays[f"b{i}.2.b"] = _fold(blk.c2, blk.n2)
    arrays["pol.conv.w"], arrays["pol.conv.b"] = _fold(model.pol_conv, model.pol_n)
    arrays["val.conv.w"], arrays["val.conv.b"] = _fold(model.val_conv, model.val_n)
    for name, layer in (("pol.fc", model.pol_fc), ("val.fc1", model.val_fc1), ("val.fc2", model.val_fc2)):
        arrays[f"{name}.w"] = layer.weight.detach().cpu().numpy()
        arrays[f"{name}.b"] = layer.bias.detach().cpu().numpy()
    _atomic_write(path, lambda f: np.savez_compressed(f, **arrays))


def torch_evaluator(model: AZNet, amp: bool = False):
    """Batched PUCT evaluator on the model's device.

    Each call encodes the whole batch into one NumPy array, copies it to the device once, and brings the
    logits and values back in one copy. Legal-move masking and the softmax run in NumPy on the host.
    """
    device = next(model.parameters()).device

    @torch.inference_mode()
    def evaluate(boards):
        model.eval()
        x = torch.from_numpy(encode(boards)).to(device)
        with _autocast(device, amp):
            logits, values = model(x)
        # Concatenate so the result crosses back to the host in a single transfer.
        out = torch.cat((logits.float(), values.float()[:, None]), dim=1).cpu().numpy()
        return masked_softmax(out[:, :COLS], legal_mask(boards)), out[:, COLS]
    return evaluate


def self_play(model, games, simulations, rng, temperature_moves=8, amp=False):
    """Play `games` games at once; returns training rows (planes, pi, z)."""
    evaluate = torch_evaluator(model, amp)
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


def play_match(model, opponent_depth, games, simulations, rng, amp=False):
    """Network + PUCT vs minimax at a fixed depth, alternating colors. Returns score in [0, 1]."""
    evaluate = torch_evaluator(model, amp)
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


def train_step(model, opt, batch, device="cpu", amp=False, scaler=None):
    """One minibatch update. Returns (value_loss, policy_loss) as device tensors.

    The losses stay on the device so the caller syncs once per iteration, not once per step.
    With amp, the forward pass is fp16 and the loss is scaled so small fp16 gradients do not underflow.
    """
    device = torch.device(device)
    boards, pis, zs = zip(*batch)
    x = torch.from_numpy(encode(list(boards))).to(device)
    pi = torch.from_numpy(np.stack(pis).astype(np.float32)).to(device)
    z = torch.tensor(zs, dtype=torch.float32, device=device)
    model.train()
    with _autocast(device, amp):
        logits, v = model(x)
    logits, v = logits.float(), v.float()  # losses in fp32
    value_loss = nn.functional.mse_loss(v, z)
    policy_loss = -(pi * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss = value_loss + policy_loss
    opt.zero_grad()
    if scaler is None:
        loss.backward()
        opt.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    return value_loss.detach(), policy_loss.detach()


def _atomic_write(path: Path, write: Callable) -> None:
    """Write to a temp file beside `path`, then rename it over `path`.

    A run killed mid-save (for example a Kaggle session hitting its time limit) keeps the previous file
    instead of leaving a truncated checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:  # a file object stops numpy from appending ".npz" to the temp name
        write(f)
    os.replace(tmp, path)


def save_checkpoint(path, model, opt, iteration, games, scaler=None):
    """Model, optimizer, and counters, so a later --resume continues the same run.

    Tensors are saved with their device; load_checkpoint maps them onto whatever device it is given.
    """
    state = {"model": model.state_dict(), "opt": opt.state_dict(), "iteration": iteration, "games": games}
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    _atomic_write(Path(path), lambda f: torch.save(state, f))


def load_checkpoint(path, model, opt, device="cpu", scaler=None):
    """Restore a checkpoint saved on any device. Returns (iteration, games)."""
    state = torch.load(path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])  # moves the optimizer state onto each parameter's device
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state["iteration"], state["games"]


def buffer_path(checkpoint) -> Path:
    """The replay buffer lives beside its checkpoint: az_connect4.pt -> az_connect4.buffer.npz."""
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(checkpoint.stem + ".buffer.npz")


def save_buffer(path, buffer):
    """Replay rows as a compressed npz. Boards are stored as their bitboards and move count, which is
    smaller and safer to load than pickled Board objects."""
    rows = list(buffer)
    arrays = {
        "current": np.array([b.current for b, _, _ in rows], dtype=np.int64),
        "mask": np.array([b.mask for b, _, _ in rows], dtype=np.int64),
        "moves": np.array([b.moves for b, _, _ in rows], dtype=np.int64),
        "pi": np.array([pi for _, pi, _ in rows], dtype=np.float64).reshape(-1, COLS),
        "z": np.array([z for _, _, z in rows], dtype=np.float64),
    }
    _atomic_write(Path(path), lambda f: np.savez_compressed(f, **arrays))


def load_buffer(path, maxlen) -> deque:
    """Read a buffer written by save_buffer. If it holds more rows than maxlen, the newest are kept."""
    buffer: deque = deque(maxlen=maxlen)
    with np.load(path) as d:
        for cur, msk, mv, pi, z in zip(d["current"], d["mask"], d["moves"], d["pi"], d["z"]):
            buffer.append((Board(int(cur), int(msk), int(mv)), pi.copy(), float(z)))
    return buffer


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--iterations", type=int, default=0,
                   help="stop after this many iterations instead of after --hours (0 = use --hours)")
    p.add_argument("--games", type=int, default=128,
                   help="self-play games per iteration (512-1024 are fine on a GPU)")
    p.add_argument("--simulations", type=int, default=100)
    p.add_argument("--buffer", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=400, help="training steps per iteration")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--threads", type=int, default=6, help="CPU threads for torch (match the cores you pinned)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="where the network runs: auto uses CUDA when available")
    p.add_argument("--amp", action="store_true",
                   help="fp16 mixed precision for self-play inference and training (CUDA only)")
    p.add_argument("--eval-every", type=int, default=5, help="play minimax matches every N iterations (0 = never)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from --checkpoint (must exist)")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT,
                   help="checkpoint to resume from; may sit in a read-only directory")
    p.add_argument("--checkpoint-out", type=Path, default=None,
                   help="where to write checkpoints (default: --checkpoint)")
    p.add_argument("--weights-out", type=Path, default=WEIGHTS, help="where to write the exported NumPy weights")
    p.add_argument("--save-buffer", action="store_true",
                   help="also write the replay buffer beside the checkpoint; --resume reloads it if present")
    p.add_argument("--log", type=Path, default=Path("results/connect4_train_log.jsonl"))
    args = p.parse_args(argv)

    if args.resume and not args.checkpoint.exists():
        p.error(f"--resume: no checkpoint at {args.checkpoint}")
    checkpoint_out = args.checkpoint_out or args.checkpoint
    device = resolve_device(args.device)
    amp = args.amp and device.type == "cuda"
    if args.amp and not amp:
        print("--amp ignored: no CUDA device", flush=True)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    model = AZNet(args.channels, args.blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    buffer: deque = deque(maxlen=args.buffer)
    iteration, games_played = 0, 0
    if args.resume:
        iteration, games_played = load_checkpoint(args.checkpoint, model, opt, device, scaler)
        if buffer_path(args.checkpoint).exists():
            buffer = load_buffer(buffer_path(args.checkpoint), args.buffer)
            print(f"restored replay buffer with {len(buffer)} rows", flush=True)
        print(f"resumed at iteration {iteration} on {device}", flush=True)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("a" if args.resume else "w")
    start = time.time()
    run_iterations = 0
    while True:
        if args.iterations:
            if run_iterations >= args.iterations:
                break
        elif time.time() - start >= args.hours * 3600:
            break
        t0 = time.time()
        rows, results = self_play(model, args.games, args.simulations, rng, amp=amp)
        buffer.extend(rows)
        games_played += args.games
        t1 = time.time()

        losses = [train_step(model, opt, py_rng.sample(buffer, min(args.batch, len(buffer))), device, amp, scaler)
                  for _ in range(args.steps)]
        if losses:
            value_loss, policy_loss = torch.stack([torch.stack(pair) for pair in losses]).mean(dim=0).tolist()
        else:
            value_loss = policy_loss = float("nan")
        iteration += 1
        run_iterations += 1
        record = {
            "iteration": iteration, "games": games_played, "buffer": len(buffer),
            "minutes": round((time.time() - start) / 60, 2),
            "device": device.type, "amp": amp,
            "selfplay_s": round(t1 - t0, 1), "train_s": round(time.time() - t1, 1),
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "first_player_score": float(np.mean([(r + 1) / 2 for r in results])),
            "mean_game_length": len(rows) / 2 / args.games,
        }
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            record["vs_minimax"] = {d: play_match(model, d, 20, 100, rng, amp=amp) for d in (2, 4, 6)}
        save_checkpoint(checkpoint_out, model, opt, iteration, games_played, scaler)
        if args.save_buffer:
            save_buffer(buffer_path(checkpoint_out), buffer)
        export(model, args.weights_out)
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)
    log.close()


if __name__ == "__main__":
    main()
