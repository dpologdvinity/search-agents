"""Run AlphaZero Connect Four training inside a Kaggle GPU kernel.

What it does, in order:
  1. Finds the search-agents code: next to this script, in an attached dataset, or by cloning the public
     repo into /tmp. Kaggle images already have torch and numpy, so nothing is installed.
  2. Finds the checkpoint in the attached dataset (/kaggle/input/**/az_connect4.pt) and resumes from it.
     The input directory is read-only, so checkpoints, buffer, weights, and log are written to /kaggle/working.
  3. Checks that CUDA can really run the network (some GPUs are too old for the installed torch build). If not,
     it falls back to the CPU and says so.
  4. Runs connect4.train_az for --hours with GPU settings, then leaves az_connect4.pt, az_connect4.buffer.npz,
     az_connect4.npz (exported NumPy weights), connect4_train_log.jsonl, and train_stdout.log in /kaggle/working.

    python train_connect4.py --hours 8
    python train_connect4.py --hours 8 -- --games 768 --simulations 200   # extra args go to train_az

The script is intended to run as a Kaggle "script" kernel with GPU and internet enabled (see kaggle/README.md).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/dpologdvinity/search-agents"
INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
RUN_DIR = Path("/tmp/search-agents")  # a writable copy of the code; the input mounts are read-only
CHECKPOINT_NAME = "az_connect4.pt"
LOG_NAME = "connect4_train_log.jsonl"


def has_trainer(root: Path) -> bool:
    return (root / "connect4" / "train_az.py").exists()


def find_repo(explicit: str | None) -> Path:
    """Return a writable directory containing connect4/train_az.py.

    Order: --repo, the packaged copy in an attached dataset, the copy next to this script, then a clone.
    Packaged copies are copied to RUN_DIR first, because /kaggle/input is read-only.
    """
    packaged = [Path(explicit)] if explicit else []
    packaged += [Path(p) for p in glob.glob(str(INPUT / "*" / "search-agents"))]
    packaged += [Path(p) for p in glob.glob(str(INPUT / "*"))]
    for root in packaged:
        if has_trainer(root):
            if root != RUN_DIR:
                shutil.rmtree(RUN_DIR, ignore_errors=True)
                shutil.copytree(root, RUN_DIR, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            return RUN_DIR
    if "__file__" in globals():
        here = Path(__file__).resolve().parent.parent
        if has_trainer(here):
            return here
    if not has_trainer(RUN_DIR):
        shutil.rmtree(RUN_DIR, ignore_errors=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(RUN_DIR)], check=True)
    return RUN_DIR


def find_checkpoint(explicit: str | None) -> Path | None:
    """The attached checkpoint, or None to start a new run. Prefers the newest copy if several are attached."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"--checkpoint {path} does not exist")
        return path
    hits = sorted(INPUT.rglob(CHECKPOINT_NAME), key=lambda p: p.stat().st_mtime) if INPUT.exists() else []
    return hits[-1] if hits else None


def check_cuda(repo: Path) -> str:
    """Return "cuda" if a small forward and backward pass with AMP runs on the GPU, else "cpu".

    The torch build may not ship kernels for this GPU (for example, very new builds dropping Pascal),
    which only shows up when a kernel is launched, so run one.
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    try:
        sys.path.insert(0, str(repo))
        from connect4.train_az import AZNet

        net = AZNet(channels=8, blocks=1).cuda()
        x = torch.zeros(4, 2, 6, 7, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits, values = net(x)
        (logits.float().sum() + values.float().sum()).backward()
        torch.cuda.synchronize()
        print(f"GPU check passed: {name}", flush=True)
        return "cuda"
    except Exception as exc:  # any kernel or driver failure means: use the CPU
        print(f"GPU check failed on {name} ({exc}); training on CPU instead", flush=True)
        return "cpu"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hours", type=float, default=8.0,
                   help="training time; stops after the iteration that passes it (Kaggle GPU sessions: about 12 h)")
    p.add_argument("--repo", help="directory containing connect4/ (default: search for it)")
    p.add_argument("--checkpoint", help="checkpoint to resume (default: newest az_connect4.pt under /kaggle/input)")
    p.add_argument("--fresh", action="store_true",
                   help="start from scratch when no checkpoint is attached (otherwise that is an error)")
    p.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision on the GPU")
    p.add_argument("--threads", type=int, default=4, help="Kaggle notebooks have 4 CPU cores")
    p.add_argument("--output", type=Path, default=WORKING, help="where results are written")
    args, extra = p.parse_known_args(argv)  # anything unknown goes to connect4.train_az, last one wins
    extra = [a for a in extra if a != "--"]  # the separator before forwarded args is not for train_az

    repo = find_repo(args.repo)
    print(f"code: {repo}", flush=True)
    if '"--device"' not in (repo / "connect4" / "train_az.py").read_text():
        # The public repo gets snapshots only, so an old clone can predate the GPU options.
        raise SystemExit(f"{repo} predates --device; publish the trainer or attach a packaged copy (see README)")
    ckpt = find_checkpoint(args.checkpoint)
    if ckpt is None and not args.fresh:
        raise SystemExit(f"no {CHECKPOINT_NAME} under {INPUT}; attach the checkpoint dataset or pass --fresh")
    device = check_cuda(repo)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    log = out / LOG_NAME
    cmd = [sys.executable, "-u", "-m", "connect4.train_az",
           "--hours", str(args.hours), "--device", device,
           # GPU settings: 512 parallel self-play games in one batched network call per simulation round,
           # 512-row minibatches, 400k-row replay buffer (about 0.5 GB of RAM), and a 4-thread CPU budget.
           "--games", "512", "--simulations", "100", "--batch", "512", "--steps", "400",
           "--buffer", "400000", "--threads", str(args.threads), "--eval-every", "5",
           "--checkpoint-out", str(out / CHECKPOINT_NAME), "--weights-out", str(out / "az_connect4.npz"),
           "--log", str(log), "--save-buffer"]
    if device == "cuda" and not args.no_amp:
        cmd.append("--amp")
    if ckpt is not None:
        # Continue the training log from the dataset, so the file covers every session of the run.
        prior_log = ckpt.with_name(LOG_NAME)
        if prior_log.exists() and not log.exists():
            shutil.copy(prior_log, log)
        cmd += ["--resume", "--checkpoint", str(ckpt)]
        print(f"resuming from {ckpt}", flush=True)
    else:
        print("no checkpoint attached: starting a fresh run", flush=True)
    cmd += extra

    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), PYTHONPATH=str(repo))
    print("running:", " ".join(cmd), flush=True)
    with (out / "train_stdout.log").open("a") as stdout_copy:
        proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:  # echo to the kernel log and keep a copy in the output
            sys.stdout.write(line)
            stdout_copy.write(line)
        code = proc.wait()
    print("outputs:", flush=True)
    for path in sorted(out.iterdir()):
        print(f"  {path.name}  {path.stat().st_size} bytes", flush=True)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
