"""Play Hex in the terminal, watch two agents play, or run the benchmark.

    python -m hexgame                          # 7x7, you play DOWN (X) against RAVE-MCTS
    python -m hexgame play -n 9 --you across --agent uct --swap
    python -m hexgame watch --red rave --blue uct --sims 1000 --seed 3
    python -m hexgame benchmark --write        # the full run committed under results/

Moves are typed as cells: "c4" is column c, row 4, counted from the top left. In play, "h" shows the
hint (the top moves with visits and win rates), "swap" takes the first stone when the swap rule allows
it, and "q" quits. DOWN (X) joins the top row to the bottom row; ACROSS (O) joins the left column to
the right column.
"""

from __future__ import annotations

import argparse
import random
import sys

from .agents import AGENTS, HINT_SIMS, LEVELS, choose, describe_move
from .benchmark import run, to_markdown, write_results
from .board import ACROSS, DOWN, MAX_N, MIN_N, NAME, SWAP, Board, chain_marks, label, parse_cell, render
from .mcts import search


def hint_lines(board: Board, sims: int = HINT_SIMS, seed: int | None = None) -> list[str]:
    """The top five moves by visits, with their win rates and RAVE rates, from a fresh search."""
    res = search(board.n, board.cells, board.to_move, sims=sims, rave=True, seed=seed)
    order = sorted(range(board.n * board.n), key=lambda c: -res["visits"][c])
    lines = [f"hint: {res['sims']} simulations, win chance for {NAME[board.to_move]} {round(100 * res['value'])}%"]
    for c in [c for c in order if res["visits"][c]][:5]:
        wr = res["win_rate"][c]
        rv = res["rave_rate"][c]
        rv_text = f", RAVE {round(100 * rv)}%" if rv is not None else ""
        lines.append(f"  {label(board.n, c):>4}  {res['visits'][c]:>6} visits   win {round(100 * wr)}%{rv_text}")
    lines.append("  principal variation: " + " ".join(label(board.n, c) for c in res["pv"][:8]))
    return lines


def play(n: int, you: int, agent: str, level: int, swap: bool, seed, read=input, out=print) -> tuple[Board, int | None]:
    """Interactive game. Returns (final board, winner or None if the player quit)."""
    board = Board(n, swap_rule=swap)
    rng = random.Random(seed)
    out(render(board))
    out(f"You are {NAME[you]} ({'X' if you == DOWN else 'O'}). Agent: {agent}, level {level}."
        + (" Swap rule on." if swap else ""))
    while board.winner() is None:
        if board.to_move == you:
            can_swap = swap and len(board.moves) == 1 and you == ACROSS
            try:
                text = read(f"{NAME[you]} > ").strip().lower()
            except EOFError:
                return board, None
            if text in ("q", "quit", "exit"):
                return board, None
            if text == "h":
                for line in hint_lines(board, seed=rng.randrange(2**32)):
                    out(line)
                continue
            cell = parse_cell(n, text)
            if cell is None or (cell == SWAP and not can_swap):
                hint = "swap, " if can_swap else ""
                out(f"'{text}' is not a legal move here; type {hint}a cell such as c4, h for a hint, or q")
                continue
            try:
                board = board.play(cell)
            except ValueError as e:
                out(str(e))
                continue
            out(render(board))
        else:
            move, analysis = choose(board, agent, level=level, seed=rng.randrange(2**32))
            board = board.play(move)
            who = agent.upper()
            out(f"{who} plays {describe_move(n, move, analysis)}")
            out(render(board))
    w = board.winner()
    out(f"{NAME[w]} wins after {len(board.moves)} moves.")
    out(render(board, marks=chain_marks(board)))
    return board, w


def watch(n: int, red: str, blue: str, sims: int, seed: int, swap: bool, out=print) -> int:
    """Two agents play one game, printing each move. DOWN is red, ACROSS is blue. Returns the winner."""
    board = Board(n, swap_rule=swap)
    rng = random.Random(seed)
    names = {DOWN: red, ACROSS: blue}
    out(render(board))
    while board.winner() is None:
        mover, agent = board.to_move, names[board.to_move]
        move, analysis = choose(board, agent, sims=sims, seed=rng.randrange(2**32))
        board = board.play(move)
        out(f"move {len(board.moves):>2}  {NAME[mover]:<7} {agent:<8} {describe_move(n, move, analysis)}")
    w = board.winner()
    out(f"{NAME[w]} ({names[w]}) wins after {len(board.moves)} moves.")
    out(render(board, marks=chain_marks(board)))
    return w


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv.insert(0, "play")
    parser = argparse.ArgumentParser(prog="python -m hexgame", description="Hex: MCTS with RAVE, against UCT.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("play", help="play against an agent")
    p.add_argument("-n", "--size", type=int, default=7, help=f"board size, {MIN_N}..{MAX_N} (default 7)")
    p.add_argument("--you", choices=["down", "across"], default="down", help="which side you play (default down)")
    p.add_argument("--agent", choices=["rave", "uct", "shortest", "random"], default="rave")
    p.add_argument("--level", type=int, choices=sorted(LEVELS), default=2, help="search budget (default 2)")
    p.add_argument("--swap", action="store_true", help="use the swap rule")
    p.add_argument("--seed", type=int, help="seed the agent for a repeatable game")

    w = sub.add_parser("watch", help="two agents play one game")
    w.add_argument("-n", "--size", type=int, default=7)
    w.add_argument("--red", choices=sorted(AGENTS), default="rave", help="the DOWN (X) agent")
    w.add_argument("--blue", choices=sorted(AGENTS), default="uct", help="the ACROSS (O) agent")
    w.add_argument("--sims", type=int, default=LEVELS[2]["sims"], help="simulations per move")
    w.add_argument("--swap", action="store_true", help="use the swap rule")
    w.add_argument("--seed", type=int, default=0)

    b = sub.add_parser("benchmark", help="head-to-head results and speed")
    b.add_argument("-n", "--size", type=int, default=7)
    b.add_argument("--sims", type=int, default=400, help="simulations per move in the matches")
    b.add_argument("--games", type=int, default=40, help="games per matchup, split across both colours")
    b.add_argument("--write", action="store_true", help="write results/hexgame_benchmark.json and .md")

    args = parser.parse_args(argv)
    if hasattr(args, "size") and not MIN_N <= args.size <= MAX_N:
        parser.error(f"size must be between {MIN_N} and {MAX_N}")

    if args.command == "play":
        you = DOWN if args.you == "down" else ACROSS
        play(args.size, you, args.agent, args.level, args.swap, args.seed)
        return 0
    if args.command == "watch":
        watch(args.size, args.red, args.blue, args.sims, args.seed, args.swap)
        return 0
    result = run(size=args.size, sims=args.sims, games=args.games)
    print(to_markdown(result))
    if args.write:
        paths = write_results(result, "results")
        print("wrote " + ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
