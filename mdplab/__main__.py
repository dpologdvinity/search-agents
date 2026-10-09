"""Command line for the grid MDP lab.

    python -m mdplab solve --preset cliff --gamma 0.95 --slip 0.1
        Exact solutions: value iteration and policy iteration. Prints the value grid and the arrow policy.

    python -m mdplab learn --preset cliff --episodes 2000 --algo q
        Q-learning (or --algo sarsa) from sampled episodes. Prints block-average returns and how often
        the learned greedy policy agrees with the optimal one.

Every run is seeded (--seed), so the printed numbers repeat exactly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .grid import ARROWS, CLIFF, PIT, PRESETS, REWARD, WALL, Params, preset
from .learn import QLearner, optimal_reference
from .mdp import GridMDP
from .solve import PolicyIterator, ValueIterator, q_table, reachable


def render_values(mdp: GridMDP, V: list[float]) -> str:
    """Values as a grid of two-decimal numbers. Terminals show their payout; walls show ##."""
    rows = []
    for y in range(mdp.height):
        cells = []
        for x in range(mdp.width):
            i = y * mdp.width + x
            k = mdp.kinds[i]
            if k == WALL:
                cells.append("   ##  ")
            elif mdp.terminal[i]:
                cells.append(f"{mdp.cell_pay[i]:+7.2f}")
            elif k == CLIFF:
                cells.append("   C   ")
            else:
                cells.append(f"{V[i]:7.2f}")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def render_policy(mdp: GridMDP, pi: list[int]) -> str:
    """The policy as arrows, one per state. Walls, terminals and cliffs show their symbol instead."""
    rows = []
    for y in range(mdp.height):
        cells = []
        for x in range(mdp.width):
            i = y * mdp.width + x
            k = mdp.kinds[i]
            if k == WALL:
                cells.append("#")
            elif k == REWARD:
                cells.append("+")
            elif k == PIT:
                cells.append("-")
            elif k == CLIFF:
                cells.append("C")
            else:
                cells.append(ARROWS[pi[i]])
        rows.append(" ".join(cells))
    return "\n".join(rows)


def _params_from(args: argparse.Namespace, base: Params) -> Params:
    """The preset's parameters, with any command-line overrides applied."""
    overrides = {}
    for name in ("gamma", "slip", "living", "reward", "pit", "cliff", "alpha", "epsilon", "decay", "eps_min"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    if getattr(args, "max_steps", None) is not None:
        overrides["max_steps"] = args.max_steps
    return replace(base, **overrides)


def _in_range(low: float, high: float, *, low_open: bool = False, integer: bool = False):
    """argparse type factory for a number in [low, high], or (low, high] when low_open. NaN is rejected.

    The solvers check these bounds too, but only after the lab has started, so a bad flag used to print a
    traceback. Here it gives argparse's usual message and exit code 2."""

    def parse(text: str):
        value = int(text) if integer else float(text)
        inside = (value > low if low_open else value >= low) and value <= high
        if not inside:
            left = "(" if low_open else "["
            raise argparse.ArgumentTypeError(f"must be in {left}{low:g}, {high:g}], got {text}")
        return value

    return parse


def _add_world_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default="rooms", choices=sorted(PRESETS), help="layout to solve (default: rooms)")
    p.add_argument("--gamma", type=_in_range(0, 1), help="discount, in [0, 1]")
    p.add_argument("--slip", type=_in_range(0, 0.5), help="probability of a sideways slip, in [0, 0.5]")
    p.add_argument("--living", type=float, help="reward on every step")
    p.add_argument("--reward", type=float, help="payout for a + cell")
    p.add_argument("--pit", type=float, help="payout for a - cell")
    p.add_argument("--cliff", type=float, help="payout for a C cell")
    p.add_argument("--seed", type=int, default=1, help="seed for the random starting policy and for learning")


def cmd_solve(args: argparse.Namespace) -> int:
    pr = preset(args.preset)
    params = _params_from(args, pr.params)
    mdp = GridMDP(pr.grid(), params)
    if not mdp.check_reachable():
        print("warning: some states cannot reach a terminal cell; values there are not finite", file=sys.stderr)

    vi = ValueIterator(mdp)
    vi.solve(args.eps)
    vi_pi = vi.policy()

    pi = PolicyIterator(mdp, seed=args.seed)
    summary = pi.solve(args.eps)
    # Agreement counts equally good actions as agreeing (a dead end where every move is worth the same).
    vi_Q = q_table(mdp, vi.V)
    agree = sum(1 for s in mdp.states
                if pi.pi[s] == vi_pi[s] or vi_Q[s][pi.pi[s]] >= max(vi_Q[s]) - 1e-6) / len(mdp.states)

    print(f"{pr.label}  {mdp.width}x{mdp.height}  states={len(mdp.states)}  "
          f"gamma={params.gamma:g} slip={params.slip:g} living={params.living:g}")
    print()
    print(f"VALUE ITERATION  {vi.sweeps} sweeps to residual < {args.eps:g} (last residual {vi.residual:.2e})")
    print(render_values(mdp, vi.V))
    print()
    print("policy:")
    print(render_policy(mdp, vi_pi))
    print()
    stable = "stable" if summary["stable"] else "NOT stable (round cap reached)"
    capped = "  evaluation hit its sweep cap" if summary["capped"] else ""
    print(f"POLICY ITERATION  {summary['rounds']} improvement rounds, {summary['eval_sweeps']} evaluation sweeps, "
          f"policy {stable}{capped}")
    print(f"agreement with value iteration: {agree:.1%}")
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    pr = preset(args.preset)
    params = _params_from(args, pr.params)
    mdp = GridMDP(pr.grid(), params)

    vi = ValueIterator(mdp)
    vi.solve(1e-9)
    opt_pi, opt_Q = optimal_reference(mdp, vi.V)

    agent = QLearner(mdp, algo=args.algo, seed=args.seed, alpha=args.alpha, epsilon=args.epsilon,
                     decay=args.decay, eps_min=args.eps_min, max_steps=args.max_steps)
    name = "Q-learning" if args.algo == "q" else "SARSA"
    print(f"{name} on {pr.label}  alpha={agent.alpha:g} epsilon={agent.epsilon0:g} decay={agent.decay:g} "
          f"gamma={params.gamma:g} slip={params.slip:g} seed={args.seed}")
    scope = len(reachable(mdp, opt_pi))
    print(f"agreement is scored on the {scope} states the optimal policy reaches from the start")
    print(f"{'episodes':>12}  {'mean return':>12}  {'falls':>6}  {'agree':>7}")
    block = max(1, args.every)
    falls = 0
    done = 0
    while done < args.episodes:
        n = min(block, args.episodes - done)
        rets = []
        for _ in range(n):
            ep = agent.episode()
            rets.append(ep.ret)
            falls += ep.fell
        done += n
        frac, _ = agent.agreement(opt_pi, opt_Q)
        mean = sum(rets) / len(rets)
        print(f"{done:>12}  {mean:>12.2f}  {falls:>6}  {frac:>6.1%}")
    frac, _ = agent.agreement(opt_pi, opt_Q)
    print()
    print("learned greedy policy:")
    print(render_policy(mdp, agent.policy()))
    print()
    print("optimal policy (value iteration):")
    print(render_policy(mdp, opt_pi))
    print()
    print(f"final agreement with the optimal policy: {frac:.1%}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mdplab", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve", help="value iteration and policy iteration on a grid")
    _add_world_args(s)
    s.add_argument("--eps", type=_in_range(0, float("inf"), low_open=True), default=1e-6,
                   help="convergence threshold on the residual, above 0")
    s.set_defaults(func=cmd_solve)

    lp = sub.add_parser("learn", help="Q-learning or SARSA from sampled episodes")
    _add_world_args(lp)
    lp.add_argument("--algo", choices=("q", "sarsa"), default="q")
    lp.add_argument("--episodes", type=_in_range(1, float("inf"), integer=True), default=2000)
    lp.add_argument("--every", type=_in_range(1, float("inf"), integer=True), default=200,
                    help="print one row per this many episodes")
    lp.add_argument("--alpha", type=_in_range(0, 1, low_open=True), help="step size, in (0, 1]")
    lp.add_argument("--epsilon", type=_in_range(0, 1), help="starting exploration rate, in [0, 1]")
    lp.add_argument("--decay", type=_in_range(0, 1, low_open=True), help="per-episode multiplier on epsilon, in (0, 1]")
    lp.add_argument("--eps-min", dest="eps_min", type=_in_range(0, 1), help="floor for epsilon, in [0, 1]")
    lp.add_argument("--max-steps", dest="max_steps", type=_in_range(1, float("inf"), integer=True),
                    help="episode length cap, at least 1")
    lp.set_defaults(func=cmd_learn)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
