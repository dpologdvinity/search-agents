"""Heads-up Leduc hold'em against a CFR+ bot or the chance player, plus training, exploitability, and the benchmark.

    python -m poker                         # play Leduc against the CFR+ bot (seats alternate each hand)
    python -m poker --opponent chance       # play against chance: fixed action odds, ignores its card
    python -m poker match --opponent chance --hands 20000 --seed 1   # CFR+ against chance, no typing
    python -m poker train --save            # train CFR and CFR+ for Leduc, write the committed files
    python -m poker train --game kuhn       # train Kuhn and check the game value against -1/18
    python -m poker exploit                 # best responses and exploitability of the committed strategy
    python -m poker benchmark --deals 100000  # the bot against each baseline, written to results/

During your turn: k = check, b = bet, c = call, r = raise, f = fold (words work too), h = hint,
q = quit. The hint shows the equilibrium mix for your information set: the probability of each legal
action if you held this card in this spot. The bot draws its actions from the same mix, so a bet with a
weak card is a bluff the equilibrium asks for, not a mistake.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from . import strategy
from .betting import BET, CALL, CHECK, FOLD, RAISE
from .bots import OPPONENTS, CFRBot, deal, head_to_head, make_opponent
from .leduc import BET_SIZES, card_name
from .tree import expected_value, exploitability

# How each opponent is named in the session banner and the match report. The names say which algorithm runs.
OPPONENT_LABELS = {
    "cfr": "CFR+ average strategy",
    "chance": "Chance (fixed odds, ignores its card)",
}

ACTION_WORDS = {
    "k": CHECK, "check": CHECK, "b": BET, "bet": BET, "c": CALL, "call": CALL,
    "r": RAISE, "raise": RAISE, "f": FOLD, "fold": FOLD,
}
# The same actions as they read after "you" (base verb) and after "the bot" (third person).
YOU_WORDS = {CHECK: "check", BET: "bet", CALL: "call", RAISE: "raise", FOLD: "fold"}
BOT_WORDS = {CHECK: "checks", BET: "bets", CALL: "calls", RAISE: "raises", FOLD: "folds"}


class Quit(Exception):
    """The player typed q: end the session."""


def _public_shown(state) -> bool:
    return state.public is not None and "/" in state.hist


def _describe(state, action: str, words: dict[str, str]) -> str:
    """The action in the given wording, with the bet size for bets and raises: "check", or "bet 2" in
    round 1 (round 2 is 4). `words` is YOU_WORDS after "you" and BOT_WORDS after "the bot"."""
    if action in (BET, RAISE):
        size = BET_SIZES[1] if "/" in state.hist else BET_SIZES[0]
        return f"{words[action]} {size}"
    return words[action]


def _legal_text(actions) -> str:
    names = {CHECK: "k=check", BET: "b=bet", CALL: "c=call", RAISE: "r=raise", FOLD: "f=fold"}
    return ", ".join(names[a] for a in actions)


def play_hand(state, me: int, bot_table: dict, rng: random.Random, read=input, out=print, bot=None) -> float:
    """Play one hand with the human in seat `me`. Returns the human's net chips for the hand.

    `bot` is the opponent; it defaults to the CFR+ bot that plays `bot_table`. The hint always reads
    `bot_table`, because the equilibrium mix is the human's own advice whoever the opponent is.
    """
    opp = 1 - me
    if bot is None:
        bot = CFRBot(bot_table)
    shown_public = False
    while not state.is_terminal():
        if _public_shown(state) and not shown_public:
            out(f"  The public card is turned up: {card_name(state.public)}.")
            shown_public = True
        if state.player() == me:
            put0, put1 = state.contributions()
            mine = put0 if me == 0 else put1
            pot = put0 + put1
            pub = card_name(state.public) if _public_shown(state) else "hidden"
            out(f"  Your card: {card_name(state.cards[me])}   public: {pub}   pot: {pot}   "
                f"you have put in: {mine}")
            acts = state.actions()
            out(f"  Legal: {_legal_text(acts)}")
            while True:
                text = read("> ").strip().lower()
                if text == "q":
                    raise Quit
                if text == "h":
                    key = state.infoset(me)
                    out(f"  hint: {strategy.describe_row(bot_table[key])}  (your information set: {key})")
                    continue
                action = ACTION_WORDS.get(text)
                if action is None or action not in acts:
                    out(f"  '{text}' is not a legal choice here. Legal: {_legal_text(acts)}, h=hint, q=quit")
                    continue
                break
            out(f"  you {_describe(state, action, YOU_WORDS)}")
            state = state.apply(action)
        else:
            acts = state.actions()
            action = bot.act(state, rng)
            assert action in acts
            out(f"  the bot {_describe(state, action, BOT_WORDS)}")
            state = state.apply(action)
    payoff = state.payoff() if me == 0 else -state.payoff()
    bot_card = card_name(state.cards[opp])
    out(f"  the bot held {bot_card}; public {card_name(state.public)}. "
        + ("You win" if payoff > 0 else "You lose" if payoff < 0 else "Split pot")
        + f" {abs(payoff):g} chips.")
    return payoff


def session(seed: int | None, table: dict, read=input, out=print, hands: int | None = None,
            opponent: str = "cfr") -> None:
    """Keep dealing hands until q, EOF, or `hands` hands. Seats alternate, so each player gets both."""
    rng = random.Random(seed)
    bot = make_opponent(opponent, table)
    net = 0.0
    played = 0
    out("Leduc hold'em. Six cards (two each of J, Q, K), a public card after round 1, bets of "
        f"{BET_SIZES[0]} then {BET_SIZES[1]}, at most two bets or raises a round.")
    if opponent == "cfr":
        out("The bot plays the CFR+ average strategy. Type h at any decision to see its equilibrium mix.")
    else:
        out(f"The bot plays {OPPONENT_LABELS[opponent]}: its moves ignore its card. "
            "Type h at any decision to see the equilibrium mix for your card.")
    try:
        while hands is None or played < hands:
            state = deal(rng)
            me = played % 2
            out(f"\nHand {played + 1}: you are seat {me + 1} ({'first' if me == 0 else 'second'} to act)")
            net += play_hand(state, me, table, rng, read, out, bot=bot)
            played += 1
            out(f"  net for the session: {net:+g} chips over {played} hands")
    except Quit:
        out("bye")
    except EOFError:
        out("")
    if played:
        out(f"Played {played} hands, net {net:+g} chips ({1000 * net / played / BET_SIZES[0]:+.0f} mbb/hand).")


def cmd_train(args) -> int:
    from . import train as tr

    if args.game == "kuhn":
        iters = args.iterations or 10_000
        for algo in ("cfr", "cfr+"):
            r = tr.run("kuhn", algo, iters)
            print(f"Kuhn {r.algorithm.upper()} {iters} iterations: exploitability {r.points[-1][1]:.2e}, "
                  f"value for seat 0 {r.value:.8f} (exact -1/18 = {tr.KUHN_VALUE:.8f}, "
                  f"error {abs(r.value - tr.KUHN_VALUE):.1e})")
        return 0

    iters = args.iterations or 10_000
    algos = ["cfr", "cfr+"] if args.save else [args.algo]
    runs = []
    for algo in algos:
        print(f"Leduc {algo.upper()}, {iters} iterations")
        r = tr.run("leduc", algo, iters, progress=lambda t, e: print(f"  iter {t:>6}  exploitability {e:.3e}")
                   if t in (1, 10, 100, 1000) or t == iters else None)
        print(f"  value for seat 0 {r.value:.5f}, {r.seconds:.0f} s")
        runs.append(r)
    if args.save:
        plus = next(r for r in runs if r.algorithm == "cfr+")
        kuhn = tr.kuhn_check(10_000)
        print(f"Kuhn check: CFR value error {kuhn['cfr']['value_error']:.1e}, "
              f"CFR+ value error {kuhn['cfr+']['value_error']:.1e} (exact -1/18)")
        meta = tr.save_leduc(plus, strategy.LEDUC_STRATEGY, strategy.LEDUC_TRAINING,
                             tr.RESULTS / "poker_train_log.jsonl", runs, extra={"kuhn_check": kuhn})
        print(f"saved {strategy.LEDUC_STRATEGY.name}: exploitability of the rounded table "
              f"{meta['exploitability']:.2e} chips, value {meta['value_seat0']:.5f}")
    return 0


def cmd_exploit(args) -> int:
    from . import train as tr

    if args.game == "kuhn":
        r = tr.run("kuhn", "cfr+", args.iterations or 10_000)
        tree = tr.tree_for("kuhn")
        sigma = r.solver.average()
        label = f"Kuhn CFR+ after {r.iterations} iterations"
    else:
        data = strategy.load()
        tree = tr.tree_for("leduc")
        sigma = strategy.profile(tree, data["strategy"])
        label = f"Leduc committed strategy ({data['algorithm']}, {data['iterations']} iterations)"
    expl, br0, br1 = exploitability(tree, sigma)
    value = expected_value(tree, sigma)
    big_blind = BET_SIZES[0] if args.game == "leduc" else 1  # Kuhn's bet is 1 chip
    print(label)
    print(f"  game value for seat 0:       {value:+.6f} chips")
    print(f"  best response, seat 0:       {br0:+.6f}")
    print(f"  best response, seat 1:       {br1:+.6f}")
    print(f"  exploitability:              {expl:.3e} chips per hand "
          f"({1000 * expl / big_blind:.4f} milli-big-blinds)")
    if args.game == "kuhn":
        print(f"  exact value -1/18:           {tr.KUHN_VALUE:+.6f}")
    return 0


def cmd_match(args) -> int:
    """The CFR+ bot against the chosen opponent, seeded, with no typing. Every deal is played in both seats,
    so the result is a fair comparison of the two algorithms rather than of the seats."""
    table = strategy.load()["strategy"]
    ai = CFRBot(table)
    opp = make_opponent(args.opponent, table)
    deals = max(1, args.hands // 2)  # each deal is two hands, one in each seat
    start = time.time()
    r = head_to_head(ai, opp, deals, args.seed)
    seconds = time.time() - start
    lo, hi = r.ci95_mbb()
    print(f"Poker match, seed {args.seed}: {OPPONENT_LABELS['cfr']} (cfr) vs {OPPONENT_LABELS[args.opponent]} "
          f"({args.opponent})")
    print(f"  {r.hands:,} hands ({deals:,} deals, each played in both seats), {seconds:.1f} s")
    print(f"  CFR+ side: {r.mbb_per_hand:+.1f} mbb/hand, 95% CI [{lo:+.1f}, {hi:+.1f}]")
    print(f"  CFR+ side: {r.mean_chips:+.4f} chips/hand, {r.mean_chips * r.hands:+.1f} chips over the match")
    return 0


def cmd_benchmark(args) -> int:
    from .benchmark import markdown, run_benchmark, write_results

    result = run_benchmark(args.deals, args.seed)
    write_results(result)
    print(markdown(result))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m poker",
                                     description="Leduc hold'em against a CFR+ bot, and the CFR tools behind it.")
    parser.add_argument("--seed", type=int, help="seed the deals and the bot, for a repeatable session")
    parser.add_argument("--hands", type=int, help="stop after this many hands (default: until q)")
    parser.add_argument("--opponent", choices=OPPONENTS, default="cfr",
                        help="who you play: cfr (the CFR+ average strategy) or chance (fixed odds)")
    sub = parser.add_subparsers(dest="command")

    t = sub.add_parser("train", help="train CFR or CFR+ and measure exploitability")
    t.add_argument("--game", choices=("leduc", "kuhn"), default="leduc")
    t.add_argument("--algo", choices=("cfr", "cfr+"), default="cfr+")
    t.add_argument("--iterations", type=int, help="iterations (default 10000)")
    t.add_argument("--save", action="store_true", help="write the committed Leduc strategy, curves, and log")

    e = sub.add_parser("exploit", help="best responses and exploitability of a strategy")
    e.add_argument("--game", choices=("leduc", "kuhn"), default="leduc")
    e.add_argument("--iterations", type=int, help="Kuhn only: iterations to train first (default 10000)")

    b = sub.add_parser("benchmark", help="the bot against each baseline; writes results/poker_benchmark.*")
    b.add_argument("--deals", type=int, default=100_000, help="deals per matchup, each played in both seats")
    b.add_argument("--seed", type=int, default=1)

    m = sub.add_parser("match", help="CFR+ against an opponent over seeded hands, printed as a result")
    m.add_argument("--opponent", choices=OPPONENTS, default="chance", help="the CFR+ bot's opponent")
    m.add_argument("--hands", type=int, default=20_000, help="hands to play (rounded down to whole deals)")
    m.add_argument("--seed", type=int, default=1)

    args = parser.parse_args(argv)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "exploit":
        return cmd_exploit(args)
    if args.command == "benchmark":
        return cmd_benchmark(args)
    if args.command == "match":
        return cmd_match(args)
    session(args.seed, strategy.load()["strategy"], hands=args.hands, opponent=args.opponent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
