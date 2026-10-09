"""Play Minesweeper in the terminal, watch the agent play, or run the benchmark.

    python -m minesweeper                          # beginner 9x9, you play
    python -m minesweeper -p expert --seed 7       # 30 columns x 16 rows, 99 mines, repeatable
    python -m minesweeper watch --agent probability -p intermediate --seed 3 --delay 0.2
    python -m minesweeper benchmark --out results  # rewrites the committed results under results/

Cells are typed as a column name then a row number, both from 1: `c5` is column c, row 5. Columns
past z continue as aa, ab, ... During play:

    c5        reveal the cell
    f c5      flag or unflag it (flags are notes; they do not change the rules)
    h         hint: a proven safe cell if there is one, else the safest guess and its probability
    p         probability map: each covered cell as its mine chance in tenths (9 = 90% and up)
    q         quit

The probability map and the hint use the same exact counts as the web page.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from .agents import AGENT_NAMES, DESCRIPTIONS, Agent, best_guess, describe_move
from .benchmark import run_benchmark, to_markdown, write_results
from .board import DEFAULT_PRESET, PRESETS, UNKNOWN, Game, View, col_name, label, parse_cell
from .inference import LEVEL_PROBABILITY, analyse

COMMANDS = ("play", "watch", "benchmark")


def _header(cols: int, width: int) -> str:
    return " " * 4 + " ".join(col_name(c).rjust(width) for c in range(cols))


def render(game: Game, flags: set[int] | None = None) -> str:
    """The board as text. `#` covered, `F` flagged, `.` zero, digits, `*` a mine (only after the game ends)."""
    flags = game.flags if flags is None else flags
    width = max(len(col_name(c)) for c in range(game.cols))
    mines = game.mine_set() if game.over else frozenset()
    lines = [_header(game.cols, width)]
    for r in range(game.rows):
        cells = []
        for c in range(game.cols):
            i = r * game.cols + c
            v = game.revealed[i]
            if i in mines:
                s = "*"
            elif v == UNKNOWN:
                s = "F" if i in flags else "#"
            else:
                s = "." if v == 0 else str(min(v, 8))
            cells.append(s.rjust(width))
        lines.append(f"{r + 1:>3} " + " ".join(cells))
    return "\n".join(lines)


def render_probabilities(view: View, analysis) -> str:
    """Each covered cell as its mine chance in tenths, `X` for a proven mine, `_` for a proven safe cell."""
    width = max(len(col_name(c)) for c in range(view.cols))
    lines = [_header(view.cols, width)]
    for r in range(view.rows):
        cells = []
        for c in range(view.cols):
            i = r * view.cols + c
            if view.cells[i] != UNKNOWN:
                s = "."
            elif i in analysis.mines:
                s = "X"
            elif i in analysis.safe:
                s = "_"
            else:
                p = analysis.probability(i)
                s = "?" if p is None else str(min(9, int(p * 10)))
            cells.append(s.rjust(width))
        lines.append(f"{r + 1:>3} " + " ".join(cells))
    return "\n".join(lines)


def hint_text(view: View) -> str:
    """The hint: the first proven safe cell and why, or the safest guess with its exact probability."""
    analysis = analyse(view, LEVEL_PROBABILITY)
    if analysis.safe:
        cell = analysis.safe[0]
        return f"certain: {label(view.cols, cell)} is safe. {analysis.why.get(cell, '')}"
    cell = best_guess(analysis)
    p = analysis.probability(cell)
    text = f"no certain move. Safest guess: {label(view.cols, cell)} with a {p:.1%} mine chance"
    if analysis.mines:
        names = ", ".join(label(view.cols, m) for m in analysis.mines[:6])
        text += f". Certain mines: {names}"
    return text


def play_game(preset: str, seed: int | None, read=None, out=print) -> int:
    """Interactive game. Returns the number of reveals made. `read` and `out` make it testable."""
    read = read or input  # looked up at call time, so tests can replace input()
    rows, cols, mines = PRESETS[preset]
    rng = random.Random(seed)
    game = Game(rows, cols, mines, rng)
    out(f"Minesweeper {preset}: {cols} columns x {rows} rows, {mines} mines. "
        "Type c5 to reveal, f c5 to flag, h for a hint, p for probabilities, q to quit.")
    while True:
        out("\n" + render(game))
        if game.over:
            break
        try:
            text = read("> ").strip().lower()
        except EOFError:
            break
        if text in ("q", "quit", "exit"):
            break
        if text == "h":
            out(hint_text(game.view()))
            continue
        if text == "p":
            analysis = analyse(game.view(), LEVEL_PROBABILITY)
            out(render_probabilities(game.view(), analysis))
            out(f"{len(analysis.components)} frontier components, {len(analysis.interior)} interior covered cells, "
                f"{analysis.remaining} mines left to place.")
            continue
        flag = text.startswith("f ")
        target = text[2:] if flag else text
        cell = parse_cell(rows, cols, target)
        if cell is None:
            out(f"'{text}' is not a cell on this {cols}x{rows} board. Try c5, or f c5 to flag.")
            continue
        if flag:
            if not game.toggle_flag(cell):
                out(f"{label(cols, cell)} is already revealed.")
            continue
        if game.revealed[cell] != UNKNOWN:
            out(f"{label(cols, cell)} is already revealed.")
            continue
        if cell in game.flags:
            out(f"{label(cols, cell)} is flagged. Use f {label(cols, cell)} to unflag it first.")
            continue
        game.reveal(cell)
    if game.won:
        out(f"Cleared with {game.reveals} reveals.")
    elif game.lost_on is not None:
        out(f"BOOM at {label(cols, game.lost_on)}. The board is shown with its mines.")
    return game.reveals


def watch_game(preset: str, agent_kind: str, seed: int, delay: float, out=print, sleep=time.sleep) -> bool:
    """Play one game with `agent_kind` and print every move with the agent's reason. Returns True if it won."""
    rows, cols, mines = PRESETS[preset]
    rng = random.Random(seed)
    game = Game(rows, cols, mines, rng)
    game.reveal(rng.randrange(rows * cols))
    agent = Agent(agent_kind, random.Random(seed * 7919 + 1))
    out(f"{agent_kind} on {preset} ({cols}x{rows}, {mines} mines), seed {seed}. {DESCRIPTIONS[agent_kind]}.")
    out(render(game))
    steps = 0
    while not game.over and steps < 2 * rows * cols:
        view = game.view()
        choice = agent.choose(view)
        out(describe_move(view, choice))
        game.reveal(choice.cell)
        steps += 1
        sleep(delay)
        out(render(game))
    out("Cleared." if game.won else f"Lost after {game.reveals} reveals.")
    return game.won


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS + ("-h", "--help"):
        argv = ["play"] + argv  # bare options mean "play"
    parser = argparse.ArgumentParser(prog="python -m minesweeper",
                                     description="Minesweeper with a constraint and probability agent.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_play = sub.add_parser("play", help="play in the terminal")
    p_play.add_argument("-p", "--preset", choices=tuple(PRESETS), default=DEFAULT_PRESET)
    p_play.add_argument("--seed", type=int, help="seed the mines, for a repeatable board")
    p_watch = sub.add_parser("watch", help="watch an agent play one game")
    p_watch.add_argument("-p", "--preset", choices=tuple(PRESETS), default=DEFAULT_PRESET)
    p_watch.add_argument("--agent", choices=AGENT_NAMES, default="probability")
    p_watch.add_argument("--seed", type=int, default=1)
    p_watch.add_argument("--delay", type=float, default=0.2, help="seconds between moves")
    p_bench = sub.add_parser("benchmark", help="win rates on fixed seeds")
    p_bench.add_argument("--games", type=int, help="games per size (default: the committed counts)")
    # No default directory: a run from the repo root must not overwrite the committed results by accident.
    p_bench.add_argument("--out", help="directory for minesweeper_benchmark.json and .md (omit to only print)")
    args = parser.parse_args(argv)

    if args.command == "play":
        play_game(args.preset, args.seed)
        return 0
    if args.command == "watch":
        watch_game(args.preset, args.agent, args.seed, args.delay)
        return 0
    games = None
    if args.games is not None:
        games = {p: args.games for p in PRESETS}
    result = run_benchmark(games)
    print(to_markdown(result))
    if args.out is not None:
        json_path, md_path = write_results(result, args.out)
        print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
