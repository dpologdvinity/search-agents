"""Play Battleship in the terminal against the Bayesian agent, or watch it sink a fleet.

    python -m battleship                        # you vs the probability agent, random fleets
    python -m battleship --agent hunt --heatmap # the hunt/target baseline, odds map on from the start
    python -m battleship --watch --delay 0.1    # the agent sinks a random fleet while you watch
    python -m battleship benchmark --games 20   # average shots to win, per agent
    python -m battleship match --games 100      # AI vs chance races: the AI and chance sink each other's fleet

On your turn type a cell such as c7 to fire, m to show or hide the odds map, or q to quit. The odds map
shows the agent's probability that each cell of your fleet holds a ship, as digits 0-9 (tenths).
Your fleet and the agent's are both placed at random.
"""

from __future__ import annotations

import argparse
import sys
import time
from random import Random

from . import benchmark, match
from .agents import AGENTS, make_agent
from .board import FLEET_NAMES, Fleet, Knowledge, cell_name, mask_of, parse_cell, random_fleet
from .probability import ship_probabilities


def grid_lines(symbol) -> list[str]:
    """A 10x10 grid as text lines. symbol(cell) gives the one-character mark for that cell."""
    lines = ["   " + " ".join("ABCDEFGHIJ")]
    for r in range(10):
        lines.append(f"{r + 1:>2} " + " ".join(symbol(r * 10 + c) for c in range(10)))
    return lines


def side_by_side(blocks: list[tuple[str, list[str]]]) -> str:
    """Put several titled grids next to each other."""
    cols = [[title] + lines for title, lines in blocks]
    height = max(len(c) for c in cols)
    widths = [max(len(line) for line in c) for c in cols]
    rows = []
    for i in range(height):
        cells = [(c[i] if i < len(c) else "").ljust(w) for c, w in zip(cols, widths)]
        rows.append("    ".join(cells).rstrip())
    return "\n".join(rows)


def own_symbol(fleet: Fleet, seen: Knowledge):
    """Your fleet as the agent sees it: S afloat, X hit, # sunk, o missed, . untouched water."""
    ship_cells = mask_of(c for s in fleet.ships for c in s)

    def symbol(c: int) -> str:
        if (seen.sunk >> c) & 1:
            return "#"
        if (seen.hit >> c) & 1:
            return "X"
        if (seen.miss >> c) & 1:
            return "o"
        return "S" if (ship_cells >> c) & 1 else "."

    return symbol


def enemy_symbol(enemy: Fleet, mine: Knowledge, reveal: bool):
    """The enemy waters as you have fired on them. The fleet appears only when the game is over."""
    ship_cells = mask_of(c for s in enemy.ships for c in s)

    def symbol(c: int) -> str:
        if (mine.sunk >> c) & 1:
            return "#"
        if (mine.hit >> c) & 1:
            return "X"
        if (mine.miss >> c) & 1:
            return "o"
        if reveal and (ship_cells >> c) & 1:
            return "S"
        return "."

    return symbol


def own_mark(seen: Knowledge, c: int) -> str:
    """Mark for a known cell of your fleet: # sunk, X hit, o missed."""
    if (seen.sunk >> c) & 1:
        return "#"
    if (seen.hit >> c) & 1:
        return "X"
    return "o"


def odds_symbol(seen: Knowledge, probs):
    """The agent's odds that each cell of your fleet holds a ship: digit = tenths, known cells keep their mark."""

    def symbol(c: int) -> str:
        if (seen.hit >> c) & 1 or (seen.miss >> c) & 1:
            return own_mark(seen, c)
        return str(min(9, int(probs[c] * 10)))

    return symbol


def _frame(mine: Fleet, seen: Knowledge, enemy: Fleet | None, mine_k: Knowledge | None, odds, reveal=False) -> str:
    """The boards for one turn: your fleet (with the odds map if on), and your shots at the enemy."""
    blocks = [("YOUR FLEET", grid_lines(own_symbol(mine, seen)))]
    if odds is not None:
        blocks.append(("AGENT'S ODDS OF YOUR SHIPS", grid_lines(odds_symbol(seen, odds))))
    if enemy is not None:
        blocks.append(("ENEMY WATERS", grid_lines(enemy_symbol(enemy, mine_k, reveal))))
    return side_by_side(blocks)


def _describe(shot, who: str, target: str) -> str:
    """One line about a shot, e.g. 'You fire at C7: sunk their Carrier!'."""
    you = who == "You"
    verb = "fire" if you else "fires"
    if shot.result == "sunk":
        owner = "their" if you else "your"
        return f"{who} {verb} at {target}: sunk {owner} {FLEET_NAMES[shot.ship_index]}!"
    return f"{who} {verb} at {target}: {shot.result}."


def play(agent_name: str, rng: Random, read=input, write=print, heatmap: bool = False) -> str:
    """One game, you vs the agent. Returns "win", "lose", or "quit". `read` and `write` are injectable for tests."""
    mine = Fleet(random_fleet(rng))  # the agent fires at this
    enemy = Fleet(random_fleet(rng))  # you fire at this
    agent = make_agent(agent_name, rng)
    seen = Knowledge()  # what the agent has learned about your fleet
    mine_k = Knowledge()  # what you have learned about the enemy's
    write(f"Battleship. Your fleet is placed; the agent ({agent_name}) has its own. Ships: "
          + ", ".join(f"{n} ({len(s)})" for n, s in zip(FLEET_NAMES, mine.ships)))
    message = ""
    shots = 0
    while True:
        odds = ship_probabilities(seen, rng).probs if heatmap else None
        write(_frame(mine, seen, enemy, mine_k, odds))
        if message:
            write(message)
        message = ""
        while True:
            text = read("Fire at (e.g. c7), m = odds map, q = quit: ").strip().lower()
            if text in ("q", "quit", "exit"):
                return "quit"
            if text == "m":
                heatmap = not heatmap
                odds = ship_probabilities(seen, rng).probs if heatmap else None
                write(_frame(mine, seen, enemy, mine_k, odds))
                continue
            try:
                cell = parse_cell(text)
            except ValueError as e:
                write(str(e))
                continue
            if cell in enemy.fired:
                write(f"You already fired at {cell_name(cell)}.")
                continue
            break
        shot = enemy.fire(cell)
        mine_k.observe(cell, shot.result, shot.ship)
        shots += 1
        message = _describe(shot, "You", cell_name(cell))
        if enemy.all_sunk:
            write(_frame(mine, seen, enemy, mine_k, None, reveal=True))
            write(f"{message}\nYou sank the whole fleet in {shots} shots. You win!")
            return "win"

        acell = agent.choose(seen)
        ashot = mine.fire(acell)
        seen.observe(acell, ashot.result, ashot.ship)
        message += "\n" + _describe(ashot, "The agent", cell_name(acell))
        if mine.all_sunk:
            write(_frame(mine, seen, enemy, mine_k, None, reveal=True))
            write(f"{message}\nThe agent sank your fleet in {shots} of your shots. You lose.")
            return "lose"


def watch(agent_name: str, rng: Random, write=print, delay: float = 0.0, heatmap: bool = False,
          sleep=time.sleep) -> int:
    """The agent sinks a random fleet while you watch. Returns the number of shots it needed."""
    fleet = Fleet(random_fleet(rng))
    agent = make_agent(agent_name, rng)
    seen = Knowledge()
    while not fleet.all_sunk:
        cell = agent.choose(seen)
        shot = fleet.fire(cell)
        seen.observe(cell, shot.result, shot.ship)
        odds = ship_probabilities(seen, rng).probs if heatmap else None
        line = _describe(shot, "The agent", cell_name(cell))
        write(_frame(fleet, seen, None, None, odds) + f"\nshot {len(fleet.fired)}: {line}")
        if delay:
            sleep(delay)
    write(f"The agent sank the whole fleet in {len(fleet.fired)} shots.")
    return len(fleet.fired)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "benchmark":
        return benchmark.main(argv[1:])
    if argv and argv[0] == "match":
        return match.main(argv[1:])
    parser = argparse.ArgumentParser(prog="python -m battleship", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=sorted(AGENTS), default="probability",
                        help="opponent or watched agent: probability, hunt, random or chance (default probability)")
    parser.add_argument("--heatmap", action="store_true", help="show the agent's odds map from the start")
    parser.add_argument("--watch", action="store_true", help="watch the agent sink a random fleet")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between shots when watching")
    parser.add_argument("--seed", type=int, default=None, help="random seed, to replay a game")
    args = parser.parse_args(argv)
    rng = Random(args.seed) if args.seed is not None else Random()
    try:
        if args.watch:
            watch(args.agent, rng, delay=args.delay, heatmap=args.heatmap)
            return 0
        outcome = play(args.agent, rng, heatmap=args.heatmap)
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
        return 0
    if outcome == "quit":
        print("You quit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
