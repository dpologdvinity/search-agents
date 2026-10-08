"""Choosing the next guess: three strategies over the same scoring pass.

The candidates are the answers still consistent with every feedback so far, and every guess in the allowed
list is a possible next move. For a guess g and candidates C, the guess splits C into buckets: one bucket per
feedback pattern, holding the candidates that would give that pattern. With the candidates equally likely
(the uniform prior), the feedback is a random variable and

    expected information  H(g) = -sum_p P(p) log2 P(p),   P(p) = bucket_p / |C|

is the number of bits the answer to "which feedback do I get?" tells us, on average. That is the entropy
strategy: maximise H. A large H means the buckets are many and even, so the candidate set shrinks fast
whatever the answer is.

The other two strategies rank the same buckets differently:
  minimax  minimise the largest bucket, the worst case after this guess; ties go to the smaller expected bucket
  random   pick any remaining candidate at random (the baseline: it uses the feedback, but no lookahead)

All strategies break exact ties in favour of a guess that could be the answer (it can win this turn), then by
the word list order, so results are repeatable.

Speed: one scoring pass takes a gather of guesses x candidates from the table and one bincount. The first
guess is the same for every game (all answers are candidates), so its ranking is computed once and cached.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from .feedback import N_PATTERNS
from .lexicon import Lexicon

STRATEGIES = ("entropy", "minimax", "random")
CHUNK = 1024  # guess rows scored per pass: bounds the temporary (rows x candidates) array


@dataclass(frozen=True)
class Scores:
    """Per-guess numbers for one candidate set, indexed like Lexicon.guesses."""

    bits: np.ndarray  # expected information of the feedback, in bits
    worst: np.ndarray  # size of the largest bucket (worst case after this guess)
    expected: np.ndarray  # expected number of candidates left: sum(bucket^2) / |C|
    is_candidate: np.ndarray  # True where the guess is one of the candidates (it could win now)


def score_guesses(lex: Lexicon, cand) -> Scores:
    """Score every allowed guess against the candidate set `cand` (answer indices)."""
    cand = np.asarray(cand, dtype=np.intp)
    n_cand = cand.size
    if n_cand == 0:
        raise ValueError("no candidates are left: the feedback does not match any answer")
    n_guesses = len(lex.guesses)
    bits = np.empty(n_guesses)
    worst = np.empty(n_guesses, dtype=np.int64)
    expected = np.empty(n_guesses)
    log_c = math.log2(n_cand)
    offsets = (np.arange(CHUNK, dtype=np.int32) * N_PATTERNS)[:, None]
    for lo in range(0, n_guesses, CHUNK):
        hi = min(n_guesses, lo + CHUNK)
        rows = hi - lo
        # pats[r, j] = pattern of guess lo+r against candidate j. Shift each row into its own 243 bins so
        # one bincount counts every (guess, pattern) pair at once.
        pats = lex.table[lo:hi][:, cand].astype(np.int32)
        flat = (pats + offsets[:rows]).ravel()
        counts = np.bincount(flat, minlength=rows * N_PATTERNS).reshape(rows, N_PATTERNS)
        n = counts.astype(np.float64)
        # H = log2|C| - (1/|C|) sum n log2 n. Empty buckets contribute 0 (n = 0 multiplies log2(1) = 0).
        bits[lo:hi] = log_c - (n * np.log2(np.maximum(n, 1.0))).sum(axis=1) / n_cand
        worst[lo:hi] = counts.max(axis=1)
        expected[lo:hi] = (n * n).sum(axis=1) / n_cand
    is_candidate = np.zeros(n_guesses, dtype=bool)
    is_candidate[lex.guess_of_answer[cand]] = True
    return Scores(bits=bits, worst=worst, expected=expected, is_candidate=is_candidate)


def guess_bits(lex: Lexicon, cand, guess_index: int) -> float:
    """Expected information of one guess against `cand`. Used to report how a played guess did."""
    cand = np.asarray(cand, dtype=np.intp)
    counts = np.bincount(lex.table[guess_index, cand], minlength=N_PATTERNS).astype(np.float64)
    return math.log2(cand.size) - float((counts * np.log2(np.maximum(counts, 1.0))).sum()) / cand.size


def narrow(lex: Lexicon, cand, guess_index: int, pattern: int) -> np.ndarray:
    """The candidates that would have produced `pattern` for this guess. This is the whole update step."""
    cand = np.asarray(cand, dtype=np.intp)
    return cand[lex.table[guess_index, cand] == pattern]


def candidates_after(lex: Lexicon, history) -> np.ndarray:
    """Replay (guess word, pattern id) pairs from the start and return the candidates that remain."""
    cand = np.arange(len(lex.answers), dtype=np.intp)
    for word, pattern in history:
        cand = narrow(lex, cand, lex.guess_index[word], pattern)
    return cand


def _order(lex: Lexicon, strategy: str, scores: Scores) -> np.ndarray:
    """Guess indices, best first, under a strategy. np.lexsort takes its keys last-to-first, so the primary
    key comes last. Ties fall to candidates first, then to the list order."""
    idx = np.arange(len(lex.guesses))
    prefer_candidate = (~scores.is_candidate).astype(np.int8)  # 0 sorts first
    if strategy == "entropy":
        return np.lexsort((idx, prefer_candidate, -np.round(scores.bits, 9)))
    if strategy == "minimax":
        return np.lexsort((idx, prefer_candidate, np.round(scores.expected, 9), scores.worst))
    raise ValueError(f"no ranking for strategy {strategy!r}")


def _check_strategy(strategy: str) -> None:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}, not {strategy!r}")


def _is_opening(lex: Lexicon, cand: np.ndarray) -> bool:
    """True when every answer is still a candidate, i.e. before any feedback."""
    return cand.size == len(lex.answers)


def _cached(lex: Lexicon, key: str, make):
    """Compute once per lexicon and keep the result in lex.cache. Only the opening is worth caching."""
    if key not in lex.cache:
        lex.cache[key] = make()
    return lex.cache[key]


def _scores(lex: Lexicon, cand: np.ndarray) -> Scores:
    if _is_opening(lex, cand):
        return _cached(lex, "opening_scores", lambda: score_guesses(lex, cand))
    return score_guesses(lex, cand)


def _ranking(lex: Lexicon, cand: np.ndarray, strategy: str, scores: Scores) -> np.ndarray:
    if _is_opening(lex, cand):
        return _cached(lex, f"opening_order:{strategy}", lambda: _order(lex, strategy, scores))
    return _order(lex, strategy, scores)


def choose(lex: Lexicon, cand, strategy: str = "entropy", rng: random.Random | None = None) -> int:
    """Index (into lex.guesses) of the guess this strategy plays next."""
    _check_strategy(strategy)
    cand = np.asarray(cand, dtype=np.intp)
    if cand.size == 1:  # one candidate left: guess it
        return int(lex.guess_of_answer[cand[0]])
    if strategy == "random":
        rng = rng or random.Random()
        return int(lex.guess_of_answer[rng.choice(cand.tolist())])
    return int(_ranking(lex, cand, strategy, _scores(lex, cand))[0])


def rank(lex: Lexicon, cand, strategy: str = "entropy", top: int = 8, rng: random.Random | None = None) -> list[dict]:
    """The best `top` guesses under a strategy, each with its numbers, for the hint displays."""
    _check_strategy(strategy)
    cand = np.asarray(cand, dtype=np.intp)
    scores = _scores(lex, cand)
    if strategy == "random":
        # The baseline has no ranking of its own, so the list shows candidates in a random order.
        rng = rng or random.Random()
        picks = rng.sample(cand.tolist(), min(top, cand.size))
        order = [int(lex.guess_of_answer[a]) for a in picks]
    else:
        order = _ranking(lex, cand, strategy, scores)[:top].tolist()
    return [describe(lex, g, scores) for g in order]


def describe(lex: Lexicon, g: int, scores: Scores) -> dict:
    """One guess's numbers as JSON-friendly values."""
    return {
        "word": lex.guesses[g],
        "bits": round(float(scores.bits[g]), 3),
        "worst": int(scores.worst[g]),
        "expected_left": round(float(scores.expected[g]), 2),
        "candidate": bool(scores.is_candidate[g]),
    }


def opening_table(lex: Lexicon, strategy: str = "entropy", top: int = 10) -> list[dict]:
    """The best first guesses when all answers are still possible. The first row is the strategy's opener."""
    return rank(lex, np.arange(len(lex.answers)), strategy, top=top)
