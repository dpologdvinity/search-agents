"""Play Wordle in the terminal, watch the solver work, or benchmark the strategies.

    python -m wordle                         # you guess, the game scores it (h = hint)
    python -m wordle play --seed 7           # the same, with a repeatable secret word
    python -m wordle solve                   # the solver guesses; you type the feedback it gets
    python -m wordle watch --answer crane    # the solver plays a secret word and explains each guess
    python -m wordle benchmark               # every answer, every strategy; writes results/wordle_benchmark.*

Feedback is typed as g (green: right letter, right place), y (yellow: right letter, wrong place) and . (gray),
so "gy..g" means green, yellow, gray, gray, green. Pressing Enter at the word prompt in solve mode plays the
solver's suggestion.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time

import numpy as np

from . import bench, feedback, solver
from .feedback import ALL_GREEN, WORD_LENGTH
from .lexicon import Lexicon, get_lexicon

MAX_GUESSES = bench.MAX_GUESSES
LIST_CANDIDATES_UP_TO = 12  # print the remaining words by name when there are this few
HINT_OPTIONS = 8


def render_row(word: str, pattern: int) -> str:
    """Two lines: the guess, and the feedback under each letter (g, y or .)."""
    return "    " + " ".join(word.upper()) + "\n    " + " ".join(feedback.to_text(pattern))


def render_options(options: list[dict]) -> str:
    """Table of guesses with expected bits, worst case and expected words left. `could-win` marks candidates."""
    lines = ["    word   bits  worst  expect-left  could-win"]
    for o in options:
        lines.append(f"    {o['word']}  {o['bits']:5.2f}  {o['worst']:5d}  {o['expected_left']:11.1f}  "
                     f"{'yes' if o['candidate'] else ''}")
    return "\n".join(lines)


def gained_text(before, after, expected: float) -> str:
    """The bits this feedback really gave, next to the bits the guess was expected to give."""
    if len(after) == 0:  # the typed feedback fits no answer; the caller reports it
        return "no candidates left"
    real = math.log2(len(before) / len(after))
    left = "1 candidate" if len(after) == 1 else f"{len(after)} candidates"
    return f"gained {real:.2f} bits (expected {expected:.2f}); {left} left"


def _list_remaining(lex: Lexicon, cand, out) -> None:
    if 0 < len(cand) <= LIST_CANDIDATES_UP_TO:
        out("    still possible: " + ", ".join(lex.answers[i] for i in cand))


def _ask_word(lex: Lexicon, read, out, default: str | None = None, hint: bool = False) -> str | None:
    """Read a word from the allowed list. None means quit (q or end of input). Blank gives `default`.
    With hint=True, typing h returns "h" instead of a word."""
    while True:
        text = read("word> ").strip().lower()  # EOFError propagates to the caller, which ends the game
        if text in ("q", "quit", "exit"):
            return None
        if hint and text == "h":
            return "h"
        if not text and default is not None:
            return default
        if len(text) != WORD_LENGTH or text not in lex.guess_index:
            out(f"    '{text}' is not one of the {len(lex.guesses)} allowed words")
            continue
        return text


def _ask_feedback(read, out) -> int | None:
    """Read feedback until it parses. None means quit."""
    while True:
        text = read("feedback> ").strip().lower()
        if text in ("q", "quit", "exit"):
            return None
        try:
            return feedback.from_text(text)
        except ValueError as e:
            out(f"    {e}")


def play(lex: Lexicon, answer: str | None = None, seed: int | None = None, read=input, out=print) -> int:
    """You guess and the game scores it. Returns the guesses used, 0 if you quit, or -1 if you ran out.

    The secret is `answer` if given, else a random answer (repeatable with `seed`). Typing h shows the
    solver's best guesses from the candidates still possible, with the bits each would give. Hints cost nothing."""
    rng = random.Random(seed)
    secret_i = lex.answer_index[answer] if answer else rng.randrange(len(lex.answers))
    cand = np.arange(len(lex.answers), dtype=np.intp)
    out(f"Wordle: {WORD_LENGTH} letters, {MAX_GUESSES} guesses. Type a word; h = hint, q = quit.")
    turn = 0
    while turn < MAX_GUESSES:
        text = _ask_word(lex, read, out, hint=True)
        if text is None:
            return 0
        if text == "h":
            out(f"    {len(cand)} answers still possible. The solver's best guesses:")
            out(render_options(solver.rank(lex, cand, "entropy", top=HINT_OPTIONS)))
            continue
        g = lex.guess_index[text]
        pattern = int(lex.table[g, secret_i])
        expected = solver.guess_bits(lex, cand, g)
        after = solver.narrow(lex, cand, g, pattern)
        turn += 1
        out(render_row(text, pattern))
        out(f"    guess {turn}: {gained_text(cand, after, expected)}")
        cand = after
        if pattern == ALL_GREEN:
            out(f"Solved in {turn} guesses.")
            return turn
        _list_remaining(lex, cand, out)
    out(f"Out of guesses. The answer was {lex.answers[secret_i].upper()}.")
    return -1


def solve_game(lex: Lexicon, strategy: str = "entropy", read=input, out=print) -> int:
    """The solver guesses; you play its word (or another one) and type the feedback. Returns the guess count
    when solved, 0 if you quit, or -1 if the feedback left no candidates or the guesses ran out."""
    cand = np.arange(len(lex.answers), dtype=np.intp)
    out(f"Solver ({strategy}). Type the word you played (Enter = the suggestion), then the feedback.")
    for turn in range(1, MAX_GUESSES + 1):
        suggestion = lex.guesses[solver.choose(lex, cand, strategy)]
        g_sugg = lex.guess_index[suggestion]
        out(f"\nguess {turn}: {suggestion.upper()}  ({len(cand)} candidates, "
            f"{solver.guess_bits(lex, cand, g_sugg):.2f} expected bits)")
        if strategy != "random" and len(cand) > 1:  # the random baseline has no table to show
            out(render_options(solver.rank(lex, cand, strategy, top=3)))
        word = _ask_word(lex, read, out, default=suggestion)
        if word is None:
            return 0
        pattern = _ask_feedback(read, out)
        if pattern is None:
            return 0
        if pattern == ALL_GREEN:
            out(f"Solved on guess {turn}: {word.upper()}.")
            return turn
        g = lex.guess_index[word]
        expected = solver.guess_bits(lex, cand, g)
        after = solver.narrow(lex, cand, g, pattern)
        out(f"    {gained_text(cand, after, expected)}")
        cand = after
        if len(cand) == 0:
            out("No answer fits that feedback. Check the feedback you typed against the word you played.")
            return -1
        _list_remaining(lex, cand, out)
    out("Out of guesses.")
    return -1


def watch(lex: Lexicon, answer: str | None = None, seed: int | None = None, strategy: str = "entropy",
          delay: float = 0.0, out=print, sleep=time.sleep) -> int:
    """The solver plays one secret word and explains each step. Returns the guess count, or -1 if it failed."""
    rng = random.Random(seed)
    secret_i = lex.answer_index[answer] if answer else rng.randrange(len(lex.answers))
    cand = np.arange(len(lex.answers), dtype=np.intp)
    out(f"Watching the {strategy} solver. The secret word stays hidden until the end.")
    for turn in range(1, MAX_GUESSES + 1):
        g = solver.choose(lex, cand, strategy, rng)
        word = lex.guesses[g]
        out(f"\nguess {turn}: {word.upper()}  ({len(cand)} candidates)")
        if strategy != "random" and len(cand) > 1:
            out(render_options(solver.rank(lex, cand, strategy, top=3)))
        pattern = int(lex.table[g, secret_i])
        expected = solver.guess_bits(lex, cand, g)
        after = solver.narrow(lex, cand, g, pattern)
        out(render_row(word, pattern))
        out(f"    {gained_text(cand, after, expected)}")
        cand = after
        if pattern == ALL_GREEN:
            out(f"\nSolved in {turn} guesses.")
            return turn
        sleep(delay)
    out(f"\nOut of guesses. The answer was {lex.answers[secret_i].upper()}.")
    return -1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Play Wordle, watch the solver, or benchmark the strategies.")
    sub = parser.add_subparsers(dest="cmd")
    p_play = sub.add_parser("play", help="you guess; the game scores it (the default)")
    p_play.add_argument("--answer", help="the secret word (must be one of the answers)")
    p_play.add_argument("--seed", type=int, help="seed the random secret, for a repeatable game")
    p_solve = sub.add_parser("solve", help="the solver guesses; you type the feedback")
    p_solve.add_argument("--strategy", choices=solver.STRATEGIES, default="entropy")
    p_watch = sub.add_parser("watch", help="the solver plays a secret word and explains each guess")
    p_watch.add_argument("--answer", help="the secret word (must be one of the answers)")
    p_watch.add_argument("--seed", type=int, help="seed the random secret")
    p_watch.add_argument("--strategy", choices=solver.STRATEGIES, default="entropy")
    p_watch.add_argument("--delay", type=float, default=0.6, help="seconds between guesses (default 0.6)")
    p_bench = sub.add_parser("benchmark", help="play every answer with each strategy")
    p_bench.add_argument("--strategies", default=",".join(solver.STRATEGIES),
                         help="comma-separated, from: " + ", ".join(solver.STRATEGIES))
    p_bench.add_argument("--no-write", action="store_true", help="print only; do not write results/")
    args = parser.parse_args(argv)

    try:
        if args.cmd in (None, "play"):
            lex = get_lexicon()
            answer = getattr(args, "answer", None)
            if answer is not None and answer not in lex.answer_index:
                parser.error(f"{answer!r} is not one of the answers")
            play(lex, answer=answer, seed=getattr(args, "seed", None))
        elif args.cmd == "solve":
            solve_game(get_lexicon(), args.strategy)
        elif args.cmd == "watch":
            lex = get_lexicon()
            if args.answer is not None and args.answer not in lex.answer_index:
                parser.error(f"{args.answer!r} is not one of the answers")
            watch(lex, answer=args.answer, seed=args.seed, strategy=args.strategy, delay=args.delay)
        elif args.cmd == "benchmark":
            names = [s.strip() for s in args.strategies.split(",") if s.strip()]
            bad = [s for s in names if s not in solver.STRATEGIES]
            if bad:
                parser.error(f"unknown strategy: {', '.join(bad)}")
            return bench.main_benchmark(names, None if args.no_write else bench.RESULTS_DIR)
    except (EOFError, KeyboardInterrupt):
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
