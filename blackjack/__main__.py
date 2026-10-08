"""Play blackjack in the terminal, print the basic-strategy table, or measure basic strategy by simulation.

    python -m blackjack                          # play with 500 chips against the dealer
    python -m blackjack --bankroll 1000 --seed 7
    python -m blackjack strategy                 # the exact basic-strategy table and the house edge
    python -m blackjack simulate --hands 100000  # measured return of basic strategy through a 6-deck shoe

During your decision: t = take a card (hit), s = stand, d = double, p = split, h = hint. The hint names the
basic-strategy action and the expected chips each legal action is worth. Bets are whole chips.
"""

from __future__ import annotations

import argparse
import random
import sys

from .game import Hand, Round, Shoe, simulate
from .rules import hard_and_ace, rank_label, rank_value, total
from .solver import ACTION_LETTERS, UPCARDS, basic_action, best, get_solver, row_label, table_rows

DECKS = 6
ACTION_KEYS = {"t": "hit", "hit": "hit", "s": "stand", "stand": "stand", "d": "double", "double": "double",
               "p": "split", "split": "split"}
PROMPT_NAMES = {"stand": "[s]tand", "hit": "[t]ake a card", "double": "[d]ouble", "split": "s[p]lit"}


def describe_ranks(ranks: list[int]) -> str:
    """Best total of a set of ranks with a soft or bust tag, e.g. "16", "soft 18", "busted at 24"."""
    values = [rank_value(r) for r in ranks]
    hard, ace = hard_and_ace(values)
    best = total(hard, ace)
    if hard > 21:
        return f"busted at {hard}"
    if ace and best != hard:
        return f"soft {best}"
    return str(best)


def show(rnd: Round, out, reveal: bool) -> None:
    """Print the dealer's cards (the hole card stays hidden until the round settles) and each player hand."""
    if reveal:
        out(f"Dealer:  {' '.join(rank_label(r) for r in rnd.dealer)}   ({describe_ranks(rnd.dealer)})")
    else:
        out(f"Dealer:  {rank_label(rnd.dealer[0])} ?")
    for i, hand in enumerate(rnd.hands, start=1):
        label = "You" if len(rnd.hands) == 1 else f"Hand {i}"
        cards = " ".join(rank_label(r) for r in hand.ranks)
        out(f"{label}:  {cards}   ({describe_ranks(hand.ranks)})   bet {hand.bet:g}")


def hint(rnd: Round, hand: Hand, funds: float, out) -> None:
    """Basic strategy's action for this hand, and the expected chips of every action that is legal now."""
    values = tuple(hand.values)
    upcard = rnd.upcard_value
    best = basic_action(values, upcard, hand.from_split, hand.split_aces)
    evs = get_solver().hand_values(values, upcard, from_split=hand.from_split, split_aces=hand.split_aces)
    parts = [f"{name} {evs[name] * hand.bet:+.2f}" for name in rnd.legal_actions(funds)]
    out(f"  Basic strategy: {best}.  Expected chips: " + ", ".join(parts))


def ask_bet(bankroll: float, read, out) -> int | None:
    """Prompt for a whole-chip bet no larger than the bankroll. None means the player quit."""
    while True:
        text = read(f"Bet (1-{int(bankroll)}, q to quit): ").strip().lower()
        if text in ("q", "quit"):
            return None
        if text.isdigit() and 1 <= int(text) <= bankroll:
            return int(text)
        out("  enter a whole number of chips, or q")


def ask_action(rnd: Round, funds: float, read, out) -> str:
    """Prompt until the player picks an action the rules allow here. 'h' shows the hint and asks again."""
    legal = rnd.legal_actions(funds)
    names = ", ".join(PROMPT_NAMES[a] for a in ("hit", "stand", "double", "split") if a in legal)
    while True:
        text = read(f"Action: {names}, [h]int? ").strip().lower()
        if text == "h":
            hint(rnd, rnd.current, funds, out)
            continue
        action = ACTION_KEYS.get(text)
        if action in legal:
            return action
        out("  that action is not available here")


def play(bankroll: float, seed: int | None, read=input, out=print) -> float:
    """Run a session until the player quits or can no longer cover the minimum bet. Returns the final bankroll."""
    shoe = Shoe(decks=DECKS, rng=random.Random(seed))
    out(f"Blackjack with a {DECKS}-deck shoe. Dealer stands on all 17s. Blackjack pays 3:2. Bankroll {bankroll:g}.")
    while bankroll >= 1:
        shoe.begin_round()
        bet = ask_bet(bankroll, read, out)
        if bet is None:
            break
        rnd = Round(shoe, bet)
        show(rnd, out, reveal=False)
        while rnd.current is not None:
            funds = bankroll - rnd.wagered
            action = ask_action(rnd, funds, read, out)
            rnd.act(action, funds=funds)
            show(rnd, out, reveal=False)
        nets = rnd.settle()
        show(rnd, out, reveal=True)
        for i, net in enumerate(nets, start=1):
            label = "" if len(nets) == 1 else f"Hand {i}: "
            if net > 0:
                out(f"  {label}you win {net:g}")
            elif net < 0:
                out(f"  {label}you lose {-net:g}")
            else:
                out(f"  {label}push")
        bankroll += sum(nets)
        out(f"Bankroll: {bankroll:g}\n")
    return bankroll


def format_strategy() -> str:
    """The exact basic-strategy table as text, with the house edge."""
    solver = get_solver()
    heads = ["T" if u == 10 else ("A" if u == 1 else str(u)) for u in UPCARDS]
    lines = ["Basic strategy, exact (infinite-deck model): S stand, H hit, D double, P split.",
             "Dealer stands on all 17s and checks for blackjack. Double on any two cards, one split.",
             "", " " * 6 + "  " + " ".join(f"{h:>2}" for h in heads)]
    for kind, t in table_rows():
        cells = []
        for u in UPCARDS:
            cells.append(f"{ACTION_LETTERS[best(solver.cell_values(kind, t, u))]:>2}")
        lines.append(f"{row_label(kind, t):>6}  " + " ".join(cells))
    edge = solver.house_edge()
    lines += ["", f"House edge with perfect basic strategy: {100 * edge:.3f}% of each bet "
                  f"(the player's expected return is {-100 * edge:.3f}%)."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] == "strategy":
            print(format_strategy())
            return 0
        if argv and argv[0] == "simulate":
            parser = argparse.ArgumentParser(prog="python -m blackjack simulate",
                                             description="Play basic strategy through a shoe and measure the return.")
            parser.add_argument("--hands", type=int, default=20_000, help="rounds to play (default 20000)")
            parser.add_argument("--seed", type=int, default=1, help="shuffle seed (default 1)")
            parser.add_argument("--decks", type=int, default=DECKS, help="decks in the shoe (default 6)")
            args = parser.parse_args(argv[1:])
            if not 1 <= args.hands <= 1_000_000:
                parser.error("--hands must be between 1 and 1000000")
            result = simulate(args.hands, seed=args.seed, decks=args.decks)
            edge = get_solver().house_edge()
            print(f"Rounds: {result.rounds:,} ({args.decks}-deck shoe, seed {args.seed})")
            print(f"Measured return per unit bet: {100 * result.mean:+.3f}% "
                  f"(standard error {100 * result.stderr:.3f}%)")
            print(f"Exact infinite-deck return:   {-100 * edge:+.3f}%")
            return 0
        parser = argparse.ArgumentParser(prog="python -m blackjack", description=__doc__,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--bankroll", type=int, default=500, help="starting chips (default 500)")
        parser.add_argument("--seed", type=int, default=None, help="shuffle seed (default random)")
        args = parser.parse_args(argv)
        if args.bankroll < 1:
            parser.error("--bankroll must be at least 1")
        play(args.bankroll, args.seed)
    except (EOFError, KeyboardInterrupt):
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
