"""Hunt invisible ghosts from the terminal, watch the autopilot, or run the seeded benchmark.

    python -m ghosthunt                        # you play: w/a/s/d or N/E/S/W to move, . to wait
    python -m ghosthunt --autopilot            # the autopilot plays and the belief is drawn each turn
    python -m ghosthunt --filter particles --particles 100 --autopilot
    python -m ghosthunt --motion lurker --noise high --seed 4 --ghosts 3
    python -m ghosthunt benchmark              # the seeded benchmark, written under results/

Playing: "bw", "ba", "bs", "bd" (or bN, bE, bS, bW) busts the cell next to you, "b." busts the cell you stand on.
A bust removes a ghost only if it is on that cell at that moment; a miss costs the turn. "auto" hands the rest
of the game to the autopilot, "map" toggles the belief picture, "cheat" shows the true ghost cells, "q" quits.

The belief picture shades each open cell by the probability that some live ghost is there, from ' ' (none)
to '@' (almost certain). '#' is a wall and 'P' is you. Ghosts stay hidden unless you cheat or bust them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import agent, sonar
from .game import ACTIONS, FILTERS, Game
from .motion import MODELS

SHADES = " .:-=+*%@"
WORD_MOVES = {"w": "N", "d": "E", "s": "S", "a": "W"}
CLEAR = "\033[H\033[2J"


def belief_picture(game: Game, cheat: bool = False) -> str:
    """The maze as text: walls, the player, and each open cell shaded by the highest live-ghost belief."""
    maze = game.maze
    live = [gh for gh in game.ghosts if gh.live]
    marg = [game.marginal(gh.id) for gh in live]
    rows = []
    for r in range(maze.n):
        line = []
        for c in range(maze.n):
            cell = r * maze.n + c
            if maze.walls[cell]:
                line.append("#")
                continue
            k = maze.k_of[cell]
            if k == game.p:
                line.append("P")
                continue
            if cheat and any(gh.k == k for gh in live):
                line.append("g")
                continue
            v = max((m[k] for m in marg), default=0.0)
            line.append(SHADES[min(len(SHADES) - 1, int(v * (len(SHADES) - 1) + 0.5))])
        rows.append("".join(line))
    return "\n".join(rows)


def status(game: Game) -> str:
    """One line under the picture: turn, busts, and the last reading from each live ghost."""
    readings = " ".join(f"g{gh.id}:{gh.readings[-1]}" for gh in game.ghosts if gh.live and gh.readings)
    if game.busted == len(game.ghosts):
        state = "ALL BUSTED"
    else:
        state = "OUT OF TURNS" if game.t >= game.max_turns else "hunting"
    return (f"turn {game.t:3d}  busted {game.busted}/{len(game.ghosts)}  {state:<12}"
            f"sonar {readings}  noise sigma {game.sigma}  {game.model}  {game.filt}")


def parse_word(word: str) -> str | None:
    """Turn what the player typed into an action string from game.ACTIONS, or None if it is not one."""
    w = word.strip()
    if not w:
        return None
    if w in ACTIONS:  # exact strings: ".", "N", "bW", "b." and so on
        return w
    if w in WORD_MOVES:  # lower-case WASD keys (checked before the upper-case compass letters)
        return WORD_MOVES[w]
    if len(w) == 2 and w[0] == "b":  # bust: the second character is a WASD key or a compass letter or "."
        ch = w[1]
        ch = WORD_MOVES.get(ch, ch.upper())
        if ("b" + ch) in ACTIONS:
            return "b" + ch
    return None


def play_interactive(game: Game, cheat: bool, show_map: bool, out=sys.stdout, inp=sys.stdin) -> None:
    """The human game loop: read one action per line, print the picture after each turn."""
    auto = False
    print(belief_picture(game, cheat) if show_map else "", file=out)
    print(status(game), file=out)
    while not game.done:
        if auto:
            action = agent.choose(game)
        else:
            out.write("action (w/a/s/d, . wait, bw/ba/bs/bd or b. bust, auto, map, cheat, q): ")
            out.flush()
            line = inp.readline()
            if not line:
                break
            word = line.strip().lower()
            if word == "q":
                break
            if word == "auto":
                auto = True
                continue
            if word == "map":
                show_map = not show_map
                print(belief_picture(game, cheat) if show_map else "(map hidden)", file=out)
                continue
            if word == "cheat":
                cheat = not cheat
                print(belief_picture(game, cheat), file=out)
                continue
            action = parse_word(word)
            if action is None or action not in game.legal_actions():
                print(f"not a legal action here: {word!r}", file=out)
                continue
        result = game.act(action)
        for gid in result["busted"]:
            print(f"BUST! ghost {gid} is gone on turn {game.t - 1}.", file=out)
        if show_map:
            print(belief_picture(game, cheat), file=out)
        print(status(game), file=out)
    finish(game, out)


def watch(game: Game, delay: float, clear: bool, out=sys.stdout) -> None:
    """The autopilot plays the whole game, redrawing the belief picture each turn."""
    tty = clear and out.isatty()
    while not game.done:
        if tty:
            out.write(CLEAR)
        out.write(belief_picture(game) + "\n" + status(game) + "\n")
        out.flush()
        result = game.act(agent.choose(game))
        for gid in result["busted"]:
            out.write(f"BUST! ghost {gid} on turn {game.t - 1}\n")
        if delay and tty:
            time.sleep(delay)
    if tty:
        out.write(CLEAR)
    out.write(belief_picture(game) + "\n" + status(game) + "\n")
    finish(game, out)


def finish(game: Game, out) -> None:
    """Summary at the end: how many turns, and what the Viterbi replay of each bust looks like."""
    if game.busted == len(game.ghosts):
        out.write(f"\nall {len(game.ghosts)} ghosts busted in {game.t} turns.\n")
    else:
        out.write(f"\n{game.busted} of {len(game.ghosts)} ghosts busted after {game.t} turns.\n")
    for gh in game.ghosts:
        if gh.bust_turn is None:
            out.write(f"ghost {gh.id}: still free, true cell {gh.k}\n")
            continue
        trace = game.trace(gh.id)
        agree = sum(int(a == b) for a, b in zip(trace["viterbi"], trace["true"])) / len(trace["true"])
        out.write(f"ghost {gh.id}: busted on turn {gh.bust_turn}; Viterbi path matched {agree:.0%} of its turns\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ghosthunt",
        description="Ghost Hunt: track invisible ghosts from noisy sonar with an HMM (forward algorithm, "
                    "particle filter, Viterbi).")
    parser.add_argument("command", nargs="?", default="play", choices=["play", "benchmark"],
                        help="play (default) runs a game; benchmark runs the seeded comparison")
    parser.add_argument("--size", type=int, default=15, choices=[11, 15, 21], help="maze side (default 15)")
    parser.add_argument("--ghosts", type=int, default=2, help="number of ghosts, 1..3 (default 2)")
    parser.add_argument("--motion", choices=MODELS, default="random", help="ghost motion model (default random)")
    parser.add_argument("--noise", choices=list(sonar.NOISE_LEVELS), default="med",
                        help="sonar noise: low, med or high (default med)")
    parser.add_argument("--seed", type=int, default=1, help="seed for the maze and the ghosts (default 1)")
    parser.add_argument("--filter", choices=FILTERS, default="exact", help="belief: exact forward or particles")
    parser.add_argument("--particles", type=int, default=200, help="particles per ghost (default 200)")
    parser.add_argument("--autopilot", action="store_true", help="let the autopilot play and animate it")
    parser.add_argument("--delay", type=float, default=0.15, help="seconds between frames on a terminal")
    parser.add_argument("--cheat", action="store_true", help="show the true ghost cells")
    parser.add_argument("--play-seeds", type=int, default=12, help="benchmark: games per configuration (default 12)")
    parser.add_argument("--conv-seeds", type=int, default=6, help="benchmark: games for the convergence table")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "results"),
                        help="benchmark: directory for the results files")
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        from . import benchmark

        result = benchmark.run(args.play_seeds, args.conv_seeds)
        benchmark.write(result, Path(args.out))
        print(benchmark.markdown(result))
        return 0

    if not 1 <= args.ghosts <= 3:
        parser.error("--ghosts must be between 1 and 3")
    if args.particles < 2:
        parser.error("--particles must be at least 2")
    game = Game(size=args.size, ghosts=args.ghosts, model=args.motion, sigma=sonar.NOISE_LEVELS[args.noise],
                seed=args.seed, filt=args.filter, particles=args.particles)
    if args.autopilot:
        watch(game, args.delay, clear=True)
    else:
        print(f"Ghost Hunt: {args.ghosts} ghost(s), {args.motion} motion, {args.noise} noise, seed {args.seed}. "
              f"Bust a ghost by standing next to it and firing. Type 'auto' to watch the autopilot finish.")
        play_interactive(game, cheat=args.cheat, show_map=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
