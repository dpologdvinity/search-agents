"""Play checkers in the terminal against any agent, watch agents play, or run a match.

    python -m checkers                              # you (red, moving first) vs alphabeta
    python -m checkers --agent minimax --level 3 --white
    python -m checkers --red mcts --white alphabeta # two agents play each other
    python -m checkers --watch alphabeta minimax    # the same, FIRST plays red
    python -m checkers --size 10 --no-forced-capture
    python -m checkers match mcts alphabeta --games 10 --level 1

During your turn, type a move such as 22-18 or 23x14x5, its number from the list,
h for a hint, or q to quit.

Players: "--red" and "--white" each take an agent name or "human". A side left out
takes the other side's opponent: "--white" alone means you play white against
--agent, and "--red mcts" alone means you play white against mcts.
"""

from __future__ import annotations

import argparse
import random
import sys

from .agents import AGENTS, choose, matchup_text
from .board import RED, SIZES, WHITE, Move, geometry, legal_moves, notation, start_position

NAMES = {"alphabeta": "alpha-beta", "minimax": "minimax", "mcts": "MCTS", "greedy": "greedy", "random": "random",
         "chance": "chance"}
SIDES = {RED: "red", WHITE: "white"}
PIECES = {1: "r", 2: "R", -1: "w", -2: "W"}

# A draw is declared after this many plies (one move by either side) with no capture
# and no man moving. Without a limit two kings can shuffle forever, and watch mode
# would never end.
QUIET_PLIES = 80
# Matches start each pair of games from a random opening of this many plies (an even number, so
# red is to move again). Deterministic agents would otherwise play the same game every time.
MATCH_OPENING_PLIES = 4


def _silent(*args, **kwargs):
    """Stand-in for print() when a game runs quietly (the match runner)."""


def render(board) -> str:
    """ASCII board, row 0 (squares 1-4 on 8x8) at the top. Empty dark squares show their number.

    Light squares are blank. Pieces are r/R for red man/king and w/W for white.
    """
    geo = geometry(board)
    line = "+" + "----+" * geo.n
    rows = [line]
    for r in range(geo.n):
        cells = []
        for c in range(geo.n):
            i = geo.index(r, c)
            if i < 0:  # light square
                cells.append("    ")
            elif board[i]:
                cells.append(f"{PIECES[board[i]]:^4}")
            else:  # the number is what the player types in notation
                cells.append(f"{i + 1:^4}")
        rows.append("|" + "|".join(cells) + "|")
        rows.append(line)
    return "\n".join(rows)


def count(n: int, one: str, many: str) -> str:
    """"1 ply", "2 plies", "1,234 positions": the number with the noun in the right form."""
    return f"{n:,} {one if n == 1 else many}"


def verdict(best) -> str:
    """The search's conclusion in words; best comes from agents.describe()."""
    if best is None:
        return "not scored"
    if best["result"] == "rate":
        return f"win rate {best['percent']:g}%"
    if best["result"] == "eval":
        return f"evaluation {best['score']:+d}"
    word = "win" if best["result"] == "win" else "loss"
    return f"forced {word} in {count(best['plies'], 'ply', 'plies')}"


def summary(analysis) -> str:
    """Depth, positions examined, cutoffs or rollouts, and verdict for one search."""
    if analysis["agent"] == "random":
        return "random choice; " + verdict(analysis["best"])
    parts = [f"depth {analysis['depth']}", count(analysis["nodes"], "position", "positions")]
    if analysis["agent"] == "alphabeta":
        parts.append(count(analysis["cutoffs"], "cutoff", "cutoffs"))
    if analysis["agent"] == "mcts":
        parts.append(count(analysis["rollouts"], "rollout", "rollouts"))
    return ", ".join(parts) + "; " + verdict(analysis["best"])


def as_move(move_json: dict) -> Move:
    """Rebuild a Move from the plain-list form that choose() returns."""
    return Move(tuple(move_json["path"]), tuple(move_json["captured"]), tuple(move_json["result"]))


def parse_move(text: str, moves: list[Move]) -> Move | None:
    """The legal move the player typed: its number in the list, or its notation.

    Notation is compared by path, so "23x14x5" and "23-14-5" name the same
    squares. A path fixes its captures, so the path alone identifies a move.
    Returns None if the text names no legal move.
    """
    if text.isdigit():  # a number is an index into the printed list, never a square
        n = int(text)
        return moves[n - 1] if 1 <= n <= len(moves) else None
    try:
        path = tuple(int(part) - 1 for part in text.replace("x", "-").split("-"))
    except ValueError:
        return None
    for move in moves:
        if move.path == path:
            return move
    return None


def ask_move(board, turn: int, agent: str, level: int, moves: list[Move], read=input, forced=True) -> Move | None:
    """Prompt until the player enters a legal move; None means quit.

    `agent` is the opponent's algorithm, reused for hints so a hint shows what it would play.
    """
    print("Legal moves: " + "   ".join(f"{i}. {notation(m)}" for i, m in enumerate(moves, 1)))
    while True:
        text = read("Your move (number or notation, h = hint, q = quit): ").strip().lower()
        if text == "q":
            return None
        if text == "h":
            hint = choose(board, turn, agent, level, forced=forced)
            print(f"  hint: {hint['move']['notation']} ({summary(hint['analysis'])})")
            continue
        move = parse_move(text, moves)
        if move is not None:
            return move
        print("  not a legal move: type a number from the list, a move such as 22-18, h, or q")


def play(players: dict, level: int, read=input, board=None, *, size: int = 8, forced: bool = True,
         by_nodes: bool = False, verbose: bool = True) -> int | None:
    """Run one game. players maps RED and WHITE to "human" or an agent name.

    Returns the winning side (RED or WHITE), 0 for a draw, or None if the human quit.
    A side with no legal move loses, which is how captures and blockades end a game.
    verbose=False prints nothing, for matches; by_nodes is passed to the agents
    (see agents.make_agent).
    """
    say = print if verbose else _silent
    if board is None:
        board = start_position(size)
    turn, quiet = RED, 0  # quiet counts plies since the last capture or man move
    say(matchup_text(players, {RED: level, WHITE: level}, size, forced=forced, by_nodes=by_nodes))
    say("Pieces: r red man, R red king, w white man, W white king. Squares are numbered from 1.")
    while True:
        say("\n" + render(board))
        moves = legal_moves(board, turn, forced)
        if not moves:
            say(f"{SIDES[turn].capitalize()} has no legal moves.")
            return -turn
        if quiet >= QUIET_PLIES:
            return 0
        player = players[turn]
        if player == "human":
            move = ask_move(board, turn, players[-turn], level, moves, read, forced)
            if move is None:
                return None
        else:
            result = choose(board, turn, player, level, forced=forced, by_nodes=by_nodes)
            move = as_move(result["move"])
            say(f"  {SIDES[turn].capitalize()} ({NAMES[player]}) plays {result['move']['notation']}: "
                f"{summary(result['analysis'])}")
        # Captures and man moves reset the draw count; a king stepping without capturing does not.
        quiet = 0 if move.captured or abs(board[move.path[0]]) == 1 else quiet + 1
        board = move.result
        turn = -turn


def players_from(args) -> dict:
    """The side-to-player map from the command-line options.

    A side left out takes the opposite of the other side: if the other side is a
    human, the missing side is the opponent (--agent); otherwise it is a human.
    With neither side given, you play red against --agent.
    """
    if args.watch:
        return {RED: args.watch[0], WHITE: args.watch[1]}
    red, white = args.red, args.white
    if red is None and white is None:
        return {RED: "human", WHITE: args.agent}
    if red is None:
        red = args.agent if white == "human" else "human"
    if white is None:
        white = args.agent if red == "human" else "human"
    return {RED: red, WHITE: white}


def opening_position(size: int, pair: int, forced: bool = True) -> tuple[int, ...]:
    """The position after MATCH_OPENING_PLIES random legal moves, from a generator seeded with the pair number.

    Both games of a pair start here, so each agent plays both colours from the same position.
    """
    rng = random.Random(pair)
    board, side = start_position(size), RED
    for _ in range(MATCH_OPENING_PLIES):
        board, side = rng.choice(legal_moves(board, side, forced)).result, -side
    return board


def match(first: str, second: str, games: int, level: int, *, size: int = 8, forced: bool = True,
          out=print) -> dict:
    """Play `games` games between two agents, alternating colours, and tally the results.

    Games come in pairs that share a random opening (opening_position), and the colours
    swap between the two games of a pair, so neither agent gets a fixed colour or a fixed
    position. Alpha-beta uses node budgets here (by_nodes), so the same match replays
    exactly on any machine. Returns {position: {"wins", "draws", "losses"}}, where 0 is
    `first` and 1 is `second`, so the names may repeat.
    """
    names = (first, second)
    tally = [{"wins": 0, "draws": 0, "losses": 0} for _ in names]
    out(f"Match: {first} vs {second}, {games} games at level {level} on {size}x{size}, "
        f"{'forced' if forced else 'optional'} captures, paired from {MATCH_OPENING_PLIES}-ply random openings")
    out(matchup_text({RED: first, WHITE: second}, {RED: level, WHITE: level}, size, forced=forced, by_nodes=True))
    out(f"{'Game':>4}  {'Red':<11} {'White':<11} Result")
    for g in range(games):
        seats = (0, 1) if g % 2 == 0 else (1, 0)  # seats[0] is the red player's index in names
        players = {RED: names[seats[0]], WHITE: names[seats[1]]}
        board = opening_position(size, g // 2, forced)
        winner = play(players, level, board=board, size=size, forced=forced, by_nodes=True, verbose=False)
        if winner == 0:
            result = "draw"
            for i in range(2):
                tally[i]["draws"] += 1
        else:
            seat = seats[0] if winner == RED else seats[1]
            other = 1 - seat
            tally[seat]["wins"] += 1
            tally[other]["losses"] += 1
            result = f"{names[seat]} wins ({SIDES[winner]})"
        out(f"{g + 1:>4}  {players[RED]:<11} {players[WHITE]:<11} {result}")
    out("")
    out(f"{'Agent':<12} {'Wins':>5} {'Draws':>6} {'Losses':>7}")
    for i, name in enumerate(names):
        t = tally[i]
        out(f"{name:<12} {t['wins']:>5} {t['draws']:>6} {t['losses']:>7}")
    return {i: tally[i] for i in range(2)}


def match_main(argv) -> int:
    """`python -m checkers match FIRST SECOND`: parse the options and run the match."""
    parser = argparse.ArgumentParser(prog="python -m checkers match",
                                     description="Play a match between two agents and print a win/draw/loss table.")
    parser.add_argument("first", choices=AGENTS)
    parser.add_argument("second", choices=AGENTS)
    parser.add_argument("--games", type=int, default=10, help="number of games (default 10)")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1, help="search budget (default 1)")
    parser.add_argument("--size", type=int, choices=SIZES, default=8, help="board size (default 8)")
    parser.add_argument("--no-forced-capture", action="store_true", help="captures are optional")
    args = parser.parse_args(argv)
    if args.games < 1:
        parser.error("--games must be at least 1")
    match(args.first, args.second, args.games, args.level, size=args.size,
          forced=not args.no_forced_capture)
    return 0


def main(argv=None, read=input) -> int:
    """Parse the options, set up the players, and play one game. Returns the exit status."""
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["match"]:
        return match_main(argv[1:])
    parser = argparse.ArgumentParser(prog="python -m checkers", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    players_choices = [*AGENTS, "human"]
    parser.add_argument("--agent", choices=AGENTS, default="alphabeta",
                        help="opponent (default alphabeta)")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=2, help="search budget (default 2)")
    parser.add_argument("--size", type=int, choices=SIZES, default=8,
                        help="board size: 8 (default), 10 or 12")
    parser.add_argument("--no-forced-capture", action="store_true",
                        help="captures are optional; by default a capture must be played")
    parser.add_argument("--red", nargs="?", const="human", choices=players_choices,
                        help="who plays red: an agent name or human (see the players note above)")
    parser.add_argument("--white", nargs="?", const="human", choices=players_choices,
                        help="who plays white: an agent name, or human alone (you play white)")
    parser.add_argument("--watch", nargs=2, metavar=("FIRST", "SECOND"), choices=AGENTS,
                        help="watch two agents play each other; FIRST plays red")
    args = parser.parse_args(argv)

    players = players_from(args)
    forced = not args.no_forced_capture
    try:
        winner = play(players, args.level, read, size=args.size, forced=forced)
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
        return 0
    if winner is None:
        print("You quit.")
    elif winner == 0:
        print(f"Draw: {QUIET_PLIES} plies without a capture or a man move.")
    else:
        who = "You win" if players[winner] == "human" else f"{NAMES[players[winner]].capitalize()} wins"
        print(f"{who} ({SIDES[winner]}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
