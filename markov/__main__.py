"""Markov chain text from the command line.

    python -m markov generate --corpus alice --order 3 --words 60 --seed 1
    python -m markov generate --corpus sonnets --level char --order 4 --temperature 0.7
    python -m markov perplexity --corpus alice --order 1..5

`generate` prints the text and its copy statistics. `perplexity` prints train and held-out perplexity for each
order: the held-out tail (the last 10% of tokens) is never counted, so the gap shows overfitting as the order
grows. Corpora are bundled (see `CORPORA`); any text file path works too.
"""

from __future__ import annotations

import argparse
import sys

from bandits.rng import Rng

from .corpus import CORPORA, load_corpus
from .model import (
    COPY_RUN,
    DEFAULT_ALPHA,
    NGramModel,
    copy_report,
    detokenize,
    generate,
    ngram_sets,
    perplexity,
    split_index,
    tokenize,
)


def _at_least_one(text: str) -> int:
    """argparse type for --words: at least one token (zero printed only the prompt)."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _positive_float(text: str) -> float:
    """argparse type for --alpha: smoothing must be above 0, or unseen words get probability 0 and log(0) fails."""
    value = float(text)
    if not value > 0:  # also rejects NaN
        raise argparse.ArgumentTypeError(f"must be above 0, got {text}")
    return value


def parse_orders(text: str) -> list[int]:
    """Orders from '3', '1..5', '1-5' or '1,3,5'. Each must be 1 to 5. Repeats are dropped, so '5,5' is one row."""
    text = text.strip()
    try:
        if ".." in text or "-" in text:
            sep = ".." if ".." in text else "-"
            lo, hi = (int(x) for x in text.split(sep))
            orders = list(range(lo, hi + 1))
        else:
            orders = [int(x) for x in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad order {text!r}: use 3, 1..5 or 1,3,5") from exc
    if not orders or any(o < 1 or o > 5 for o in orders):
        raise argparse.ArgumentTypeError("orders must be between 1 and 5")
    return list(dict.fromkeys(orders))  # keep the order given, without repeats


def cmd_generate(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    tokens = tokenize(corpus.body, args.level)
    model = NGramModel(tokens, args.order)
    rng = Rng(args.seed)
    gen = generate(model, rng, args.words, args.temperature)
    print(detokenize(gen.tokens, args.level))
    min_run = COPY_RUN[args.level]
    report = copy_report(gen, ngram_sets(tokens), min_run)
    print(
        f"\n[{corpus.name} | {args.level} | order {args.order} | T {args.temperature:g} | seed {args.seed}] "
        f"vocab {model.vocab_size}, contexts {model.context_count}, "
        f"copied {report.copied_pct:.1f}% of {len(report.copied)} picks in verbatim runs of {min_run}+ tokens, "
        f"longest verbatim run {report.longest_run}, restarts {gen.restarts}",
        file=sys.stderr,
    )
    return 0


def cmd_perplexity(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    tokens = tokenize(corpus.body, args.level)
    cut = split_index(tokens)
    train = tokens[:cut]
    print(f"{corpus.name} ({args.level}): {len(train)} train tokens, {len(tokens) - cut} held out, "
          f"add-alpha smoothing with alpha {args.alpha:g}")
    print(f"{'order':>5}  {'contexts':>8}  {'n-grams':>7}  {'train ppl':>10}  {'held-out ppl':>13}")
    for order in args.order:
        model = NGramModel(train, order)
        train_ppl = perplexity(model, train, 0, alpha=args.alpha)  # the train span, scored by its own counts
        held_ppl = perplexity(model, tokens, cut, alpha=args.alpha)
        print(f"{order:>5}  {model.context_count:>8}  {model.ngram_count:>7}  {train_ppl:>10.2f}  {held_ppl:>13.2f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m markov", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="sample text from the chain")
    g.add_argument("--corpus", default="alice", help=f"one of {', '.join(CORPORA)}, or a text file")
    g.add_argument("--level", choices=("word", "char"), default="word")
    g.add_argument("--order", type=int, default=3, help="1 to 5: an order-n chain reads the last n-1 tokens")
    g.add_argument("--words", type=_at_least_one, default=60, help="tokens to generate, at least 1")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--temperature", type=float, default=1.0, help="counts become count**(1/T); 1 is unchanged")
    g.set_defaults(func=cmd_generate)

    pp = sub.add_parser("perplexity", help="train vs held-out perplexity per order")
    pp.add_argument("--corpus", default="alice", help=f"one of {', '.join(CORPORA)}, or a text file")
    pp.add_argument("--level", choices=("word", "char"), default="word")
    pp.add_argument("--order", type=parse_orders, default=parse_orders("1..5"), help="e.g. 1..5 or 2,3")
    pp.add_argument("--alpha", type=_positive_float, default=DEFAULT_ALPHA,
                    help="add-alpha smoothing constant, above 0")
    pp.set_defaults(func=cmd_perplexity)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate" and not 1 <= args.order <= 5:
        raise SystemExit("--order must be between 1 and 5")
    if args.command == "generate" and args.temperature <= 0:
        raise SystemExit("--temperature must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
