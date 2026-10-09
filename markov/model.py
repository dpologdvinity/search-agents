"""Word- or character-level n-gram Markov chain: counts, next-token distributions, sampling, perplexity.

An order-n model remembers, for every (n-1)-token context, how often each token followed it in the corpus.
Those counts are the whole model. Generating text means repeatedly looking up the last n-1 tokens and drawing
the next token in proportion to its count. The Markov assumption is the cut-off: the next token depends only
on the last n-1 tokens, not on anything earlier.

Temperature T reweights the counts to c**(1/T): T < 1 sharpens the distribution toward the most common
continuation, T > 1 flattens it. T = 1 uses the raw counts.

Every step draws exactly one uniform from the seeded generator (bandits/rng.py, mulberry32), so the page's
JavaScript port (web/js/markov-core.js) reproduces the same token sequence for the same corpus and seed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from bandits.rng import Rng

# A word is a run of letters with optional inner apostrophes, straight or curly (don't, men’s); every other
# non-space character is its own token, so punctuation is modelled too. The JavaScript regex is the same.
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|[^\sA-Za-z]")

# Longest verbatim run that the copy check looks for. Windows longer than this are not measured.
MAX_RUN = 20

# Punctuation that hugs the word before it when the text is rebuilt for display.
NO_SPACE_BEFORE = frozenset(".,;:!?)")
NO_SPACE_AFTER = frozenset("(")


def tokenize(text: str, level: str = "word") -> list[str]:
    """Split text into tokens. Whitespace runs collapse to one space first, so newlines do not matter."""
    body = " ".join(text.split())
    if level == "char":
        return list(body)
    if level == "word":
        return WORD_RE.findall(body)
    raise ValueError(f"unknown level {level!r}: use 'word' or 'char'")


def detokenize(tokens: list[str], level: str = "word") -> str:
    """Rebuild readable text: no space before closing punctuation or after an opening bracket."""
    if level == "char":
        return "".join(tokens)
    out: list[str] = []
    for tok in tokens:
        if out and tok not in NO_SPACE_BEFORE and out[-1] not in NO_SPACE_AFTER:
            out.append(" ")
        out.append(tok)
    return "".join(out)


class NGramModel:
    """Counts of every token that follows each (order-1)-token context in the training tokens.

    successors[ctx] is a dict token -> count. Dicts keep insertion order (first occurrence in the corpus),
    and the JavaScript Map does too, so sampling walks the candidates in the same order in both languages.
    """

    def __init__(self, tokens: list[str], order: int):
        if order < 1:
            raise ValueError("order must be at least 1")
        if len(tokens) < order:
            raise ValueError(f"need at least {order} tokens for an order-{order} model, got {len(tokens)}")
        self.order = order
        self.tokens = list(tokens)
        self.successors: dict[tuple[str, ...], dict[str, int]] = {}
        k = order - 1
        # Every window of `order` consecutive tokens is one observed n-gram: its first k tokens are the
        # context and its last token is the successor.
        for i in range(len(tokens) - k):
            ctx = tuple(tokens[i : i + k])
            succ = self.successors.setdefault(ctx, {})
            w = tokens[i + k]
            succ[w] = succ.get(w, 0) + 1

    @property
    def vocab_size(self) -> int:
        return len(set(self.tokens))

    @property
    def context_count(self) -> int:
        """Number of distinct (order-1)-token contexts seen in the corpus."""
        return len(self.successors)

    @property
    def ngram_count(self) -> int:
        """Number of distinct n-grams seen in the corpus."""
        return sum(len(d) for d in self.successors.values())

    def distribution(self, ctx, temperature: float = 1.0) -> list[tuple[str, float]] | None:
        """Next-token probabilities after ctx at this temperature, or None if ctx never precedes anything.

        Each count c becomes c**(1/T), and the weights are divided by their sum, so the result sums to 1.
        """
        d = self.successors.get(tuple(ctx))
        if not d:
            return None
        items = list(d.items())
        weights = [c ** (1.0 / temperature) for _, c in items]
        total = sum(weights)
        return [(w, x / total) for (w, _), x in zip(items, weights)]


def _pick(weights: list[float], rng: Rng) -> int:
    """Index drawn with probability proportional to weights, by inverse CDF on one uniform."""
    total = sum(weights)
    u = rng.uniform() * total
    cum = 0.0
    for idx, w in enumerate(weights):
        cum += w
        if u < cum:
            return idx
    return len(weights) - 1  # only reachable through rounding at the very top of the range


@dataclass(frozen=True)
class Pick:
    index: int         # position of the generated token in Generation.tokens
    token: str
    prob: float        # probability the sampled token had at this temperature


@dataclass(frozen=True)
class Generation:
    tokens: list[str]  # prompt, then everything generated (and any restart jumps)
    prompt_len: int    # the first prompt_len tokens are the starting corpus window
    picks: list[Pick]
    restarts: int      # times the chain hit a context with no successor and jumped to a corpus window


def generate(model: NGramModel, rng: Rng, count: int, temperature: float = 1.0) -> Generation:
    """Sample `count` tokens. The start is a seeded corpus window of order-1 tokens.

    The chain can only get stuck at the end of the corpus, where a context no longer has a successor. Then it
    jumps to another seeded corpus window and goes on, and the jump is recorded in `restarts`.
    """
    k = model.order - 1
    n_tokens = len(model.tokens)
    i = rng.index(n_tokens - k)  # windows with a successor start at 0 .. n-k-1
    out = model.tokens[i : i + k]
    prompt_len = len(out)
    picks: list[Pick] = []
    restarts = 0
    while len(picks) < count:
        ctx = tuple(out[len(out) - k :])  # the last k tokens; the empty tuple when k == 0
        succ = model.successors.get(ctx)
        if not succ:
            j = rng.index(n_tokens - k)
            out.extend(model.tokens[j : j + k])
            restarts += 1
            continue
        items = list(succ.items())
        weights = [c ** (1.0 / temperature) for _, c in items]
        idx = _pick(weights, rng)
        total = sum(weights)
        picks.append(Pick(len(out), items[idx][0], weights[idx] / total))
        out.append(items[idx][0])
    return Generation(out, prompt_len, picks, restarts)


def ngram_sets(tokens: list[str], max_len: int = MAX_RUN) -> list[set[tuple[str, ...]]]:
    """Index l-1 holds every window of length l that occurs in the corpus, for l = 1 .. max_len."""
    sets = []
    for length in range(1, max_len + 1):
        sets.append({tuple(tokens[i : i + length]) for i in range(len(tokens) - length + 1)})
    return sets


@dataclass(frozen=True)
class CopyReport:
    copied: list[bool]     # per pick: does the order-n window ending here occur verbatim in the corpus?
    copied_pct: float      # share of picks whose order-n window is copied
    longest_run: int       # longest verbatim run ending at a pick, capped at MAX_RUN


def copy_report(gen: Generation, sets: list[set[tuple[str, ...]]], order: int) -> CopyReport:
    """How much of the generated text is copied verbatim from the corpus.

    An order-n window that occurs in the corpus is a copy. The longest run ending at each pick is the longest
    suffix that occurs in the corpus. The check can stop at the first miss because a window can only occur if
    every shorter window ending at the same place occurs too.
    """
    out = gen.tokens
    copied: list[bool] = []
    longest = 0
    for pick in gen.picks:
        p = pick.index
        # Order-n window ending at this pick. Every pick has at least n-1 tokens before it.
        copied.append(tuple(out[p - order + 1 : p + 1]) in sets[order - 1])
        length = 1
        while length <= min(MAX_RUN, p + 1) and tuple(out[p - length + 1 : p + 1]) in sets[length - 1]:
            length += 1
        longest = max(longest, length - 1)
    pct = 100.0 * sum(copied) / len(copied) if copied else 0.0
    return CopyReport(copied, pct, longest)


def split_index(tokens: list[str], held_out_frac: float = 0.1) -> int:
    """Index where the held-out tail starts: the last held_out_frac of the tokens are held out."""
    return max(1, len(tokens) - int(round(len(tokens) * held_out_frac)))


DEFAULT_ALPHA = 0.01


def perplexity(model: NGramModel, tokens: list[str], start: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Perplexity of tokens[start:] under the model's counts with add-alpha smoothing.

    P(w | ctx) = (c(ctx, w) + alpha) / (C(ctx) + alpha * (V + 1)). The +1 is an unknown-token bucket, so a
    held-out word the corpus never had still gets a probability. A context the model never saw gives a uniform
    distribution over the vocabulary. Perplexity is exp of the mean negative log probability per token.

    alpha is small on purpose. Laplace smoothing (alpha = 1) spreads so much mass over the vocabulary that
    train perplexity rises again at order 3, which hides the overfitting this measure is meant to show.
    """
    k = model.order - 1
    V = model.vocab_size
    totals: dict[tuple[str, ...], int] = {}
    log_sum = 0.0
    count = 0
    for t in range(max(start, k), len(tokens)):
        ctx = tuple(tokens[t - k : t])
        succ = model.successors.get(ctx, {})
        if ctx not in totals:
            totals[ctx] = sum(succ.values())
        p = (succ.get(tokens[t], 0) + alpha) / (totals[ctx] + alpha * (V + 1))
        log_sum += math.log(p)
        count += 1
    if count == 0:
        raise ValueError("no tokens to score")
    return math.exp(-log_sum / count)
