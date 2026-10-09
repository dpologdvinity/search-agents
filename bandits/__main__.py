"""Neon casino: play the slot machines yourself, watch an agent learn, or run the benchmark.

    python -m bandits play --pulls 200 --seed 7       # you pull; then see how you rank against the agents
    python -m bandits play --kind drifting            # the payouts are redrawn every 500 pulls
    python -m bandits watch --agent thompson --pulls 60 --seed 7 --delay 0.2
    python -m bandits benchmark                       # 100 seeds, T = 10,000; writes results/bandits_benchmark.*

Machines are named A, B, C, ... Type a letter (or a number 1..K) to pull, q to stop. The probabilities
stay hidden until the end. The seed fixes the machines and the outcomes, so `--seed` replays a casino.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from .agents import AGENTS, LINEUP, machine_name
from .benchmark import HORIZON, SEEDS, K, run_benchmark, to_markdown, write_results
from .env import KINDS, Environment
from .sim import make_for, play, run_agent, score_pulls

DEFAULT_OUT = "results"


def parse_pull(text: str, k: int) -> int | None:
    """A machine from typed input: a letter (A..) or a number (1..k). None if it is not a machine."""
    text = text.strip().upper()
    if len(text) == 1 and "A" <= text <= chr(ord("A") + k - 1):
        return ord(text) - ord("A")
    if text.isdigit() and 1 <= int(text) <= k:
        return int(text) - 1
    return None


def describe_segments(env: Environment) -> list[str]:
    """Each segment's means, for the reveal. Stationary casinos have one segment."""
    lines = []
    for i, seg in enumerate(env.machines.schedule):
        start = i * env.machines.period + 1
        end = min(env.horizon, (i + 1) * env.machines.period)
        odds = "  ".join(f"{machine_name(j)} {p:.3f}" for j, p in enumerate(seg))
        label = f"pulls {start}-{end}" if len(env.machines.schedule) > 1 else "all pulls"
        lines.append(f"  {label:>16}: {odds}")
    return lines


def play_human(kind: str, k: int, pulls: int, seed: int, read=input, out=print) -> int:
    """Interactive game. Returns how many pulls were made; the reveal and the comparison follow the game.

    The player and the agents see the same outcome table, so the comparison is on identical luck.
    """
    env = Environment(kind, k, pulls, seed)
    out(f"Neon casino: {k} machines ({kind}), {pulls} pulls, seed {seed}. "
        "Type a machine (A..) to pull, q to stop. The payouts are hidden.")
    arms: list[int] = []
    total = 0.0
    while len(arms) < pulls:
        try:
            text = read(f"pull {len(arms) + 1}/{pulls} > ")
        except EOFError:
            break
        if text.strip().lower() in ("q", "quit", "exit"):
            break
        arm = parse_pull(text, k)
        if arm is None:
            out(f"'{text.strip()}' is not a machine here: use A..{machine_name(k - 1)} or 1..{k}, or q to stop")
            continue
        t = len(arms)
        reward = env.reward(t, arm)
        total += reward
        arms.append(arm)
        shown = f"{reward:.2f}" if kind == "gaussian" else ("paid" if reward else "nothing")
        out(f"  pull {t + 1}: machine {machine_name(arm)} -> {shown}   (total {total:.2f})")
    if not arms:
        out("No pulls, so nothing to compare.")
        return 0

    # Score the player on the pulls made, using a casino of exactly that length (same outcomes).
    used = len(arms)
    env = Environment(kind, k, used, seed)
    mine = score_pulls(env, arms)
    out("")
    out("The casino, revealed:")
    for line in describe_segments(env):
        out(line)
    out("")
    out(f"{'player':<26}  regret {mine.total_regret:7.2f}   best machine {100 * mine.share_optimal():5.1f}%   "
        f"reward {sum(mine.rewards):8.2f}")
    rows = []
    for key in LINEUP[kind]:
        run = play(key, env)
        rows.append((AGENTS[key].label, run))
    for label, run in rows:
        out(f"{label:<26}  regret {run.total_regret:7.2f}   best machine {100 * run.share_optimal():5.1f}%   "
            f"reward {sum(run.rewards):8.2f}")
    beaten = sum(1 for _, run in rows if run.total_regret > mine.total_regret)
    tied = sum(1 for _, run in rows if run.total_regret == mine.total_regret)
    out("")
    out(f"You beat {beaten} of {len(rows)} agents on this casino" + (f" and tied {tied}." if tied else "."))
    return used


def watch(key: str, kind: str, k: int, pulls: int, seed: int, delay: float = 0.0, out=print, sleep=time.sleep):
    """Print an agent's pulls and its reason for each one. Returns the Run."""
    env = Environment(kind, k, pulls, seed)
    agent = make_for(key, env)
    out(f"{AGENTS[key].label} on {k} machines ({kind}), seed {seed}, {pulls} pulls")
    out(AGENTS[key].blurb)
    out("")

    def show(t, arm, reward, agent):
        shown = f"{reward:.2f}" if kind == "gaussian" else ("paid" if reward else "nothing")
        out(f"{t + 1:>5}  machine {machine_name(arm)}  {shown:>8}   {agent.reason}")
        if delay:
            sleep(delay)

    run = run_agent(agent, env, on_pull=show)
    out("")
    out(f"regret {run.total_regret:.2f}, best machine {100 * run.share_optimal():.1f}% of pulls")
    return run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bandits",
                                     description="Neon casino: multi-armed bandits, exploration vs exploitation.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_casino(p):
        p.add_argument("--kind", choices=KINDS, default="bernoulli",
                       help="bernoulli (0/1 payouts), gaussian (noisy real payouts), or drifting (payouts redrawn)")
        p.add_argument("-k", "--machines", type=int, default=K, help=f"number of machines, 2..20 (default {K})")
        p.add_argument("--seed", type=int, help="seed for the machines and outcomes (default: a random one, printed)")

    p_play = sub.add_parser("play", help="pull the machines yourself, then compare with the agents")
    add_casino(p_play)
    p_play.add_argument("--pulls", type=int, default=100, help="pulls in the game (default 100, up to 1000)")

    p_watch = sub.add_parser("watch", help="watch one agent play, with its reason for each pull")
    add_casino(p_watch)
    p_watch.add_argument("--agent", choices=sorted(AGENTS), default="ucb1")
    p_watch.add_argument("--pulls", type=int, default=60)
    p_watch.add_argument("--delay", type=float, default=0.0, help="seconds to pause after each pull")

    p_bench = sub.add_parser("benchmark", help="every agent on every casino over many seeds")
    p_bench.add_argument("--seeds", type=int, default=SEEDS)
    p_bench.add_argument("--horizon", type=int, default=HORIZON)
    p_bench.add_argument("--out", default=DEFAULT_OUT, help="directory for bandits_benchmark.json and .md")
    p_bench.add_argument("--no-write", action="store_true", help="print the tables but do not write results")

    args = parser.parse_args(argv)
    if args.command == "benchmark":
        if args.seeds < 1:
            parser.error("--seeds must be at least 1")
        if args.horizon <= 1000:
            parser.error("--horizon must be more than 1000")
        def progress(kind, seed):
            if seed % 25 == 0:
                print(f"  {kind}: seed {seed} of {args.seeds}", file=sys.stderr)

        result = run_benchmark(seeds=args.seeds, horizon=args.horizon, progress=progress)
        print(to_markdown(result))
        if not args.no_write:
            json_path, md_path = write_results(result, args.out)
            print(f"wrote {json_path} and {md_path}")
        return 0

    # Without --seed, pick one at random; the header line of play and watch prints it for replay.
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 2**31)
    try:
        if args.command == "play":
            if not 1 <= args.pulls <= 1000:
                parser.error("--pulls must be between 1 and 1000")
            play_human(args.kind, args.machines, args.pulls, seed)
        else:
            watch(args.agent, args.kind, args.machines, args.pulls, seed, delay=args.delay)
    except ValueError as e:
        parser.error(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
