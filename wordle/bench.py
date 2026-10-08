"""Benchmark: play every answer with each strategy and count the guesses.

For each answer the solver starts from all answers as candidates, then loops: choose a guess, score it
against the hidden answer (table lookup), keep the consistent candidates. A game is solved on the guess that
equals the answer. A game that is not solved within MAX_GUESSES counts as a failure, recorded as 7.

Random seeds: the random strategy draws from random.Random(seed), seeded per answer, so the numbers repeat.
The entropy and minimax strategies are deterministic (ties go to the word list order).

    python -m wordle benchmark                                 # all three strategies; writes results/wordle_benchmark.*
    python -m wordle benchmark --strategies entropy --no-write # one strategy, printed only (no table rewrite)
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import solver
from .lexicon import Lexicon, get_lexicon

MAX_GUESSES = 6
FAILED = MAX_GUESSES + 1  # how a failed game is counted in the distribution
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def play(lex: Lexicon, answer_i: int, strategy: str, seed: int = 0) -> tuple[int, list[str]]:
    """Solve one answer. Returns (guesses used, or FAILED, and the words guessed)."""
    rng = random.Random(seed * 100_003 + answer_i)
    cand = np.arange(len(lex.answers), dtype=np.intp)
    target = int(lex.guess_of_answer[answer_i])
    words = []
    for turn in range(1, MAX_GUESSES + 1):
        g = solver.choose(lex, cand, strategy, rng)
        words.append(lex.guesses[g])
        if g == target:
            return turn, words
        cand = solver.narrow(lex, cand, g, int(lex.table[g, answer_i]))
    return FAILED, words


def run(lex: Lexicon, strategy: str, seed: int = 0) -> dict:
    """Play every answer with one strategy. Returns the summary plus the per-answer guess counts."""
    t0 = time.perf_counter()
    counts = np.empty(len(lex.answers), dtype=np.int64)
    worst_word = ("", 0)
    for i in range(len(lex.answers)):
        n, words = play(lex, i, strategy, seed)
        counts[i] = n
        if n > worst_word[1]:
            worst_word = (lex.answers[i], n)
    elapsed = time.perf_counter() - t0
    dist = Counter(int(n) for n in counts)
    solved = counts[counts <= MAX_GUESSES]
    return {
        "strategy": strategy,
        "answers": int(counts.size),
        "mean_guesses": round(float(counts.mean()), 4),
        "mean_guesses_solved": round(float(solved.mean()), 4) if solved.size else None,
        "distribution": {str(k): int(dist.get(k, 0)) for k in range(1, FAILED + 1)},
        "failures": int(dist.get(FAILED, 0)),
        "failure_rate": round(float((counts == FAILED).mean()), 6),
        "hardest_answer": {"word": worst_word[0], "guesses": int(worst_word[1])},
        "seconds": round(elapsed, 1),
        "per_answer": {lex.answers[i]: int(counts[i]) for i in range(counts.size)},
    }


def summarise(results: list[dict]) -> str:
    """A Markdown table of the strategies: mean guesses, the 1..6 distribution, failure rate."""
    header = "| strategy | mean guesses | 1 | 2 | 3 | 4 | 5 | 6 | fail (>6) | failure rate | seconds |"
    rule = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = []
    for r in results:
        d = r["distribution"]
        rows.append(
            f"| {r['strategy']} | {r['mean_guesses']:.3f} | "
            + " | ".join(str(d[str(k)]) for k in range(1, MAX_GUESSES + 1))
            + f" | {r['failures']} | {r['failure_rate'] * 100:.2f}% | {r['seconds']} |"
        )
    return "\n".join([header, rule, *rows])


def write_results(lex: Lexicon, results: list[dict], openers: dict, out_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    """Write results/wordle_benchmark.json (everything, incl. per-answer counts) and .md (the summary)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "answers": len(lex.answers),
        "guesses": len(lex.guesses),
        "max_guesses": MAX_GUESSES,
        "openers": openers,
        "strategies": results,
    }
    json_path = out_dir / "wordle_benchmark.json"
    json_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    lines = [
        "# Wordle benchmark",
        "",
        f"All {len(lex.answers)} answers, every allowed guess ({len(lex.guesses)} words) available to each strategy.",
        "Guess counts include the winning guess; 7 marks a game not solved in 6.",
        "",
        summarise(results),
        "",
        "## Best opening word (all answers still possible)",
        "",
        "| rank | word | expected bits | worst bucket | expected left | in answers |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for k, row in enumerate(openers["entropy"], 1):
        lines.append(f"| {k} | {row['word']} | {row['bits']:.3f} | {row['worst']} | {row['expected_left']:.1f} | "
                     f"{'yes' if row['candidate'] else 'no'} |")
    md_path = out_dir / "wordle_benchmark.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main_benchmark(strategies: list[str], out_dir: Path | None) -> int:
    lex = get_lexicon()
    openers = {"entropy": solver.opening_table(lex, "entropy", top=10),
               "minimax": solver.opening_table(lex, "minimax", top=5)}
    print(f"best opening (entropy): {openers['entropy'][0]['word']} ({openers['entropy'][0]['bits']} bits)")
    results = []
    for s in strategies:
        r = run(lex, s)
        results.append(r)
        print(f"{s:>8}: mean {r['mean_guesses']:.3f}, failures {r['failures']}, {r['seconds']} s")
    print()
    print(summarise(results))
    if out_dir is not None:
        json_path, md_path = write_results(lex, results, openers, out_dir)
        print(f"\nwrote {json_path} and {md_path}")
    return 0
