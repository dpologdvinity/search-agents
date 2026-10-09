"""Play Connect Four in the terminal against any of the agents, or watch two agents play.

    python -m connect4                         # you (X) vs AlphaZero, you move first
    python -m connect4 --agent minimax --level 3 --ai-first
    python -m connect4 --watch alphazero minimax
    python -m connect4 --match 20 --agent alphazero --level 1   # record over 20 games vs chance

During your turn, type a column number (1-7), "h" for a hint, or "q" to quit.
The "chance" opponent picks columns from fixed odds with no search, as a baseline.
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np

from .agents import AGENTS, ALPHAZERO_SIMS, MCTS_SIMS, MINIMAX_SECONDS, chance_move
from .board import COLS, Board
from .evaluate import match

NAMES = {"alphazero": "AlphaZero", "minimax": "Minimax", "mcts": "MCTS", "chance": "Chance"}


def render(board: Board, first_symbol="X", second_symbol="O") -> str:
    """ASCII board, top row first. The player who moved first is always X."""
    # Board.grid() is relative to the player to move (1 = to move, -1 = other);
    # convert that back to fixed symbols for whoever moved first.
    to_move_is_first = board.to_move_is_first()
    rows = []
    for row in board.grid():
        cells = []
        for v in row:
            if v == 0:
                cells.append(".")
            elif (v == 1) == to_move_is_first:
                cells.append(first_symbol)
            else:
                cells.append(second_symbol)
        rows.append("| " + " ".join(cells) + " |")
    return "\n".join(rows + ["+" + "-" * (2 * COLS + 1) + "+", "  " + " ".join(str(c + 1) for c in range(COLS))])


def describe(result: dict) -> str:
    """One line explaining how an agent chose its move."""
    a = result["analysis"]
    if a["kind"] == "alphazero":
        return (f"{a['simulations']} simulations; estimates its chance of winning at "
                f"{100 * a['win_probability']:.0f}%")
    if a["kind"] == "minimax":
        best = a["best"]
        verdict = {"win": "a forced win", "loss": "a forced loss"}.get(best["result"], f"score {best.get('score')}")
        return f"searched {a['depth']} plies, {a['nodes']:,} positions; sees {verdict}"
    if a["kind"] == "chance":
        # No search: the move was a draw from fixed odds, so say how likely that column was.
        return f"drawn from fixed odds, {100 * a['odds'][result['move']]:.1f}% for that column"
    return f"{a['simulations']} random playouts"


def ask_column(board: Board, agent: str, level: int, read=input) -> int | None:
    """Prompt until the player enters a legal column; None means quit."""
    while True:
        text = read("Your move (1-7, h = hint, q = quit): ").strip().lower()
        if text == "q":
            return None
        if text == "h":
            hint = AGENTS[agent](board, level)
            print(f"  hint: column {hint['move'] + 1} ({describe(hint)})")
            continue
        if text.isdigit() and board.can_play(int(text) - 1):
            return int(text) - 1
        print("  that column is full or not a column number")


def run_match(agent: str, level: int, games: int, seed: int = 0) -> dict:
    """`games` games of `agent` against the chance opponent, colours alternating. Returns win/draw/loss counts.

    Each game opens with two random moves (see connect4.evaluate.play_game), so the games differ even
    though the search agents are deterministic. The seed fixes both the openings and the chance draws.
    """
    def agent_move(board: Board) -> int:
        return AGENTS[agent](board, level)["move"]

    chance_rng = random.Random(seed)
    return match(agent_move, lambda board: chance_move(board, 1, chance_rng)["move"], games,
                 np.random.default_rng(seed))


def match_table(agent: str, level: int, games: int, counts: dict) -> str:
    """A markdown table with W/D/L for the agent and for the chance opponent, naming each algorithm."""
    budget = {"alphazero": f"{ALPHAZERO_SIMS[level]} simulations per move",
              "minimax": f"{MINIMAX_SECONDS[level]} s per move",
              "mcts": f"{MCTS_SIMS[level]} simulations per move"}.get(agent, "no search")
    lines = [f"Connect Four: {NAMES[agent]} (level {level}, {budget}) vs Chance (fixed odds), "
             f"{games} games, colours alternating",
             "",
             "| Algorithm | Wins | Draws | Losses |",
             "|---|---|---|---|",
             f"| {NAMES[agent]} ({agent}) | {counts['win']} | {counts['draw']} | {counts['loss']} |",
             f"| Chance (chance, fixed odds) | {counts['loss']} | {counts['draw']} | {counts['win']} |"]
    return "\n".join(lines)


def play(players: dict, level: int, read=input) -> int:
    """Run one game. players maps 1 (first) and 2 (second) to "human" or an agent name.

    Returns 1 or 2 for the winner, 0 for a draw, or -1 if the human quit.
    """
    board = Board()
    while True:
        number = 1 if board.to_move_is_first() else 2
        player = players[number]
        print("\n" + render(board))
        if player == "human":
            col = ask_column(board, players[3 - number], level, read)
            if col is None:
                return -1
        else:
            result = AGENTS[player](board, level)
            col = result["move"]
            print(f"\n{NAMES[player]} ({'X' if number == 1 else 'O'}) plays column {col + 1}: {describe(result)}")
        if board.is_winning_move(col):
            print("\n" + render(board.play(col)))
            return number
        board = board.play(col)
        if board.is_full():
            print("\n" + render(board))
            return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m connect4", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=sorted(AGENTS), default="alphazero", help="opponent (default alphazero)")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=2, help="search budget (default 2)")
    parser.add_argument("--ai-first", action="store_true", help="let the agent move first")
    parser.add_argument("--watch", nargs=2, metavar=("FIRST", "SECOND"), choices=sorted(AGENTS),
                        help="watch two agents play each other")
    parser.add_argument("--match", type=int, metavar="GAMES",
                        help="play GAMES games of --agent against chance and print the W/D/L table")
    parser.add_argument("--seed", type=int, default=0, help="seed for --match (default 0)")
    args = parser.parse_args(argv)

    if args.match is not None:
        if args.agent == "chance":
            parser.error("--match needs a search agent to play against chance")
        counts = run_match(args.agent, args.level, args.match, args.seed)
        print(match_table(args.agent, args.level, args.match, counts))
        return 0

    if args.watch:
        players = {1: args.watch[0], 2: args.watch[1]}
    elif args.ai_first:
        players = {1: args.agent, 2: "human"}
    else:
        players = {1: "human", 2: args.agent}

    try:
        winner = play(players, args.level)
    except FileNotFoundError as e:  # AlphaZero weights not trained yet
        print(e, file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
        return 0
    if winner == -1:
        print("You quit.")
    elif winner == 0:
        print("Draw: the board is full.")
    else:
        who = "You win" if players[winner] == "human" else f"{NAMES[players[winner]]} wins"
        print(f"{who} ({'X' if winner == 1 else 'O'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
