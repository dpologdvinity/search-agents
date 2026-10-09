"""Balance the pole in the terminal, watch an agent do it, train the policies, or benchmark them.

    python -m cartpole watch --agent reinforce --seed 3   # ASCII animation of one episode
    python -m cartpole play                                # you push: type a (left) or d (right), then Enter
    python -m cartpole play --slow                         # real time: the last key you pressed keeps pushing
    python -m cartpole train --algo reinforce --seeds 0 1 2 3 4 --episodes 1500
    python -m cartpole benchmark --episodes 200

Training writes cartpole/data/*.npz and curves.json; benchmark writes data/benchmark.json and
results/cartpole_benchmark.md; train also rewrites results/cartpole_learning.md.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time

import numpy as np

from .agents import AGENT_NAMES, DATA_DIR, LABELS, load_agent
from .benchmark import RESULTS_DIR, run_benchmark, table, write_results
from .env import LEFT, RIGHT, CartPole
from .render import CLEAR, render_frame
from .train import ALGOS, GAMMA, train_and_save

KEYS = {"a": LEFT, "d": RIGHT}


def watch(agent_name: str, seed: int, delay: float, out=print, sleep=time.sleep) -> int:
    """Animate one episode of an agent. Learned agents also show their probabilities and value."""
    agent = load_agent(agent_name, seed=seed)
    env = CartPole(seed)
    obs = env.reset(seed)
    while not env.done:
        action = agent.act(obs)
        probs = agent.probs(obs) if hasattr(agent, "probs") else None
        value = agent.value(obs) if hasattr(agent, "value") else None
        out(CLEAR + f"{LABELS[agent_name]}   seed {seed}\n"
            + render_frame(obs[0], obs[2], env.steps, action, probs, value))
        obs, _, _, _ = env.step(action)
        sleep(delay)
    out(CLEAR + f"{LABELS[agent_name]}   seed {seed}\n"
        + render_frame(env.x, env.theta, env.steps)
        + ("\n\nreached the 500-step cap: balanced the whole episode" if env.truncated
           else f"\n\nthe pole fell after {env.steps} steps"))
    return env.steps


def play_turns(seed: int, read=input, out=print) -> int:
    """Turn-based play. Each letter typed is one push, so a line of 'ad' is two steps."""
    env = CartPole(seed)
    env.reset(seed)
    out(CLEAR + render_frame(env.x, env.theta, env.steps))
    while not env.done:
        try:
            line = read("push (a = left, d = right, q = quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break  # Ctrl-D or Ctrl-C ends the game like typing q, with the same outcome line
        if "q" in line:
            break
        for ch in line:
            if ch not in KEYS:
                continue
            env.step(KEYS[ch])
            out(CLEAR + render_frame(env.x, env.theta, env.steps, KEYS[ch]))
            if env.done:
                break
    out(_outcome(env))
    return env.steps


def play_slow(seed: int, tick: float = 0.12, out=print) -> int:
    """Real-time play: the sim advances every `tick` seconds. The latest key sets the push, and it keeps
    pushing until you press the other key. Needs a terminal; uses cbreak mode so keys arrive unbuffered."""
    import termios
    import tty

    # cbreak needs a real terminal; with a pipe or a redirect, tcgetattr fails with a raw termios error.
    if not sys.stdin.isatty():
        raise SystemExit("play --slow needs a terminal. Run it in one, or use `python -m cartpole play` instead.")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    env = CartPole(seed)
    env.reset(seed)
    action = RIGHT  # the first push goes right; the pole is nearly upright, so either direction works
    quit_now = False
    try:
        tty.setcbreak(fd)
        while not env.done:
            deadline = time.monotonic() + tick
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                    break
                ch = os.read(fd, 1).decode(errors="ignore").lower()
                if ch in KEYS:
                    action = KEYS[ch]
                elif ch == "q":
                    quit_now = True  # leave both loops, so the tty is restored and the outcome is printed
                    break
            if quit_now:
                break
            env.step(action)
            out(CLEAR + render_frame(env.x, env.theta, env.steps, action))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    out(_outcome(env))
    return env.steps


def _outcome(env: CartPole) -> str:
    if env.truncated:
        return "reached the 500-step cap: the pole was balanced for the whole episode"
    if env.terminated:
        return f"the pole fell after {env.steps} steps"
    return f"stopped after {env.steps} steps"


def write_learning(data_dir=DATA_DIR, results_dir=RESULTS_DIR) -> None:
    """Results table for every algorithm in curves.json: the first and last 50 episodes per seed."""
    doc = json.loads((data_dir / "curves.json").read_text())
    lines = ["# CartPole learning curves", "",
             f"Training episodes per seed, sampled (not greedy) returns. Discount {doc.get('gamma', GAMMA)}.",
             "The learned policy shipped is the seed with the best last 50 episodes.", ""]
    for algo in ALGOS:
        entry = doc["algorithms"].get(algo)
        if not entry:
            continue
        lines += [f"## {LABELS.get(algo, algo).title()}", "",
                  "| Seed | Episodes | First 50 mean | Last 50 mean | Trailing-100 mean reaches 475 at episode |",
                  "|---:|---:|---:|---:|---:|"]
        for seed, rets in entry["seeds"].items():
            r = np.asarray(rets, dtype=float)
            hit = "never" if len(r) < 100 else _first_hit(r)
            star = " (shipped)" if int(seed) == entry["best_seed"] else ""
            lines.append(f"| {seed}{star} | {len(r)} | {r[:50].mean():.1f} | {r[-50:].mean():.1f} | {hit} |")
        lines.append("")
    lines += NOTES
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "cartpole_learning.md").write_text("\n".join(lines))


# Earlier actor-critic configurations, kept as a record. Each row is the last-50 mean of seeds 0-4 at
# 1500 episodes, from runs made while tuning. The shipped configuration is the Monte Carlo target above.
NOTES = [
    "## Actor-critic: configurations tried",
    "",
    "All with 1500 episodes per seed. Last-50 mean per seed 0 to 4.",
    "",
    "| Configuration | Last-50 means | Seeds at 475+ |",
    "|---|---|---:|",
    "| TD(0), raw advantage, batch 8, critic lr 1e-2 | 500.0, 60.1, 25.9, 41.6, 144.4 | 1 |",
    "| TD(0), batch-standardised advantage, batch 8, critic lr 1e-2 | 15.5, 87.6, 14.9, 77.5, 118.8 | 0 |",
    "| TD(0), batch-standardised advantage, batch 8, critic lr 1e-3 | 10.2, 9.3, 19.2, 10.0, 49.1 | 0 |",
    "| TD(0), batch-standardised advantage, one episode per update, critic lr 1e-3 | 9.2, 101.3, 9.2, 9.4, 22.3 | 0 |",
    "| TD(0), fixed-scale advantage (delta / 50), batch 8, critic lr 1e-2 (seeds 0-2 pilot) | 9, 9, 9 | 0 |",
    "| Monte Carlo target, batch 8, critic lr 1e-2 (shipped) | see the table above | |",
    "",
    "The TD(0) advantage bootstraps from the critic's own estimate, so it is only as good as the critic.",
    "In these runs the critic did not track V(s) well enough, and the advantage was mostly noise: the policy",
    "drifted into pushing one way. Giving the critic 1500 updates instead of 190 (one episode per update)",
    "did not fix it, so the update count is not the whole story. The Monte Carlo target has no bootstrap, so",
    "the critic learns the same returns REINFORCE sees; this is the variant that works here.",
]


def _first_hit(r) -> str:
    """First episode at which the trailing 100-episode mean reaches 475, or 'never'."""
    trailing = np.convolve(r, np.ones(100) / 100, mode="valid")
    hit = np.flatnonzero(trailing >= 475)
    return str(int(hit[0] + 100)) if len(hit) else "never"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m cartpole", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("watch", help="animate one episode of an agent")
    w.add_argument("--agent", choices=AGENT_NAMES, default="reinforce")
    w.add_argument("--seed", type=int, default=10_000)
    w.add_argument("--delay", type=float, default=0.03, help="seconds between frames")

    pl = sub.add_parser("play", help="push the cart yourself")
    pl.add_argument("--seed", type=int, default=10_000)
    pl.add_argument("--slow", action="store_true", help="real time instead of turn-based")
    pl.add_argument("--tick", type=float, default=0.12, help="seconds per step in --slow mode")

    t = sub.add_parser("train", help="train the policy-gradient or cross-entropy agents")
    t.add_argument("--algo", choices=ALGOS, default="reinforce")
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    t.add_argument("--episodes", type=int, default=1500)

    b = sub.add_parser("benchmark", help="seeded benchmark of every agent")
    b.add_argument("--episodes", type=int, default=200)

    args = p.parse_args(argv)
    if args.cmd == "watch":
        watch(args.agent, args.seed, args.delay)
    elif args.cmd == "play":
        if args.slow:
            play_slow(args.seed, args.tick)
        else:
            play_turns(args.seed)
    elif args.cmd == "train":
        summary = train_and_save(args.algo, args.seeds, args.episodes)
        write_learning()
        for s, r in summary["summaries"].items():
            print(f"seed {s}: first50 {r['first50']:.1f}  last50 {r['last50']:.1f}  "
                  f"reaches 475 at {r.get('episodes_to_475_trailing100')}")
        print(f"shipped seed {summary['best_seed']}")
    elif args.cmd == "benchmark":
        doc = run_benchmark(args.episodes)
        write_results(doc)
        print(table(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
