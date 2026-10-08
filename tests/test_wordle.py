import math
import random
import re
from collections import Counter

import numpy as np
import pytest

from wordle import bench, feedback, solver
from wordle.__main__ import main, play, solve_game, watch
from wordle.build_lists import five_letter
from wordle.feedback import ALL_GREEN, GRAY, GREEN, N_PATTERNS, YELLOW
from wordle.lexicon import Lexicon, encode, get_lexicon


@pytest.fixture(scope="module")
def lex():
    return get_lexicon()


def small_lexicon():
    """A hand-sized lexicon, so the expected numbers can be worked out by hand."""
    return Lexicon.from_words(["crane", "slate", "trace", "abide", "apple", "allee", "llama", "hello"],
                              ["tares", "lares"])


# ── Feedback rule ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "guess, answer, expected",
    [
        ("speed", "abide", "..y.y"),  # the second E has no unused E left: gray
        ("allee", "apple", "gy..g"),  # the first L is yellow; the second L and the first E find nothing unused
        ("llama", "hello", "yy..."),  # two L's in the guess, two in the answer: both yellow
        ("crane", "crane", "ggggg"),
        ("crane", "xxxxx", "....."),
    ],
)
def test_score_hand_checked_cases(guess, answer, expected):
    assert feedback.to_text(feedback.score(guess, answer)) == expected


def test_pattern_ids_round_trip():
    assert feedback.score("crane", "crane") == ALL_GREEN == 242
    assert feedback.score("xxxxx", "yyyyy") == GRAY == 0
    for p in range(N_PATTERNS):
        assert feedback.pattern_id(feedback.tiles(p)) == p
        assert feedback.from_text(feedback.to_text(p)) == p


def test_feedback_text_parsing():
    assert feedback.from_text("gy..g") == feedback.pattern_id([GREEN, YELLOW, GRAY, GRAY, GREEN])
    assert feedback.from_text("G Y x b .") == feedback.from_text("gy...")
    for bad in ("gy.g", "gyy..g", "gy.z.", ""):
        with pytest.raises(ValueError):
            feedback.from_text(bad)


def test_table_matches_the_scalar_rule(lex):
    """The vectorised table must agree with the plain rule on random pairs (all pairs take too long)."""
    rng = random.Random(3)
    for _ in range(3000):
        g = rng.randrange(len(lex.guesses))
        a = rng.randrange(len(lex.answers))
        assert int(lex.table[g, a]) == feedback.score(lex.guesses[g], lex.answers[a])


def test_table_handles_duplicates_in_a_small_lexicon():
    small = small_lexicon()
    for w in small.guesses:
        for i, a in enumerate(small.answers):
            assert int(small.table[small.guess_index[w], i]) == feedback.score(w, a)


# ── Word lists ─────────────────────────────────────────────────────────────

def test_bundled_lists_are_five_letter_lowercase(lex):
    assert len(lex.answers) > 1000 and len(lex.guesses) >= len(lex.answers)
    for w in lex.guesses:
        assert len(w) == 5 and w.isalpha() and w.islower() and w.isascii()
    assert set(lex.answers) <= set(lex.guesses)
    assert lex.table.shape == (len(lex.guesses), len(lex.answers)) and lex.table.dtype == np.uint8


def test_build_filter_keeps_only_plain_five_letter_words():
    words = ["crane", "Crane", "cranes", "can't", "abcd", "abcde", "café", "slate", "slate", "tare"]
    assert five_letter(words) == ["abcde", "crane", "slate"]  # abcde is five plain letters, so it is kept


def test_encode_maps_letters_to_codes():
    assert encode(["abcde", "zzzzz"]).tolist() == [[0, 1, 2, 3, 4], [25] * 5]


# ── Solver ─────────────────────────────────────────────────────────────────

def test_scores_match_a_direct_entropy_computation(lex):
    rng = random.Random(5)
    cand = np.array(sorted(rng.sample(range(len(lex.answers)), 150)), dtype=np.intp)
    scores = solver.score_guesses(lex, cand)
    for g in rng.sample(range(len(lex.guesses)), 25):
        buckets = Counter(int(lex.table[g, a]) for a in cand)
        n = len(cand)
        h = -sum((c / n) * math.log2(c / n) for c in buckets.values())
        assert scores.bits[g] == pytest.approx(h, abs=1e-9)
        assert scores.worst[g] == max(buckets.values())
        assert scores.expected[g] == pytest.approx(sum(c * c for c in buckets.values()) / n)


def test_opening_is_a_high_entropy_word_and_is_cached(lex):
    top = solver.opening_table(lex, "entropy", top=3)
    assert top[0]["bits"] > 6.0 and top[0]["bits"] >= top[1]["bits"] >= top[2]["bits"]
    assert solver.choose(lex, np.arange(len(lex.answers)), "entropy") == lex.guess_index[top[0]["word"]]
    assert ("opening_order:entropy" in lex.cache) and ("opening_scores" in lex.cache)


def test_one_candidate_is_guessed_and_two_candidates_prefer_a_candidate(lex):
    g = solver.choose(lex, [lex.answer_index["crane"]], "minimax")
    assert lex.guesses[g] == "crane"
    g = solver.choose(lex, [lex.answer_index["crane"], lex.answer_index["slate"]], "entropy")
    assert lex.guesses[g] in ("crane", "slate")


def test_narrow_keeps_the_answer_and_matches_the_feedback(lex):
    secret = lex.answer_index["abide"]
    cand = np.arange(len(lex.answers), dtype=np.intp)
    g = lex.guess_index["tares"]
    after = solver.narrow(lex, cand, g, int(lex.table[g, secret]))
    assert secret in after.tolist()
    assert all(int(lex.table[g, a]) == int(lex.table[g, secret]) for a in after)


def test_guess_bits_agrees_with_the_scores(lex):
    cand = np.arange(0, len(lex.answers), 7)
    s = solver.score_guesses(lex, cand)
    g = lex.guess_index["slate"]
    assert solver.guess_bits(lex, cand, g) == pytest.approx(s.bits[g], abs=1e-9)


def test_random_strategy_is_repeatable_with_a_seed(lex):
    cand = np.arange(len(lex.answers), dtype=np.intp)
    a = [solver.choose(lex, cand, "random", random.Random(9)) for _ in range(3)]
    b = [solver.choose(lex, cand, "random", random.Random(9)) for _ in range(3)]
    assert a == b and lex.guesses[a[0]] in lex.guesses


def test_unknown_strategy_is_rejected(lex):
    with pytest.raises(ValueError):
        solver.choose(lex, [0, 1], "vibes")


def test_entropy_solves_a_sample_within_six(lex):
    for i in range(0, len(lex.answers), 80):
        n, words = bench.play(lex, i, "entropy")
        assert n <= bench.MAX_GUESSES, (lex.answers[i], words)
        assert words[-1] == lex.answers[i]


def test_benchmark_run_counts_every_answer():
    small = small_lexicon()
    r = bench.run(small, "entropy")
    assert r["answers"] == len(small.answers) == 8
    assert sum(r["distribution"].values()) == 8 and r["failures"] == 0
    assert set(r["per_answer"]) == set(small.answers)
    assert r["mean_guesses"] == pytest.approx(np.mean(list(r["per_answer"].values())), abs=1e-3)


def test_benchmark_summary_table_text():
    result = {"strategy": "x", "mean_guesses": 3.5, "failures": 2, "failure_rate": 0.2, "seconds": 0.1,
              "distribution": {str(k): (2 if k == 1 else 0) for k in range(1, 8)}}
    text = bench.summarise([result])
    assert "| x | 3.500 |" in text and "20.00%" in text


# ── Terminal game ──────────────────────────────────────────────────────────

def scripted(lines):
    it = iter(lines)

    def read(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return read


def test_play_scores_a_win_and_reports_bits(lex):
    out = []
    n = play(lex, answer="abide", read=scripted(["tares", "abide"]), out=out.append)
    text = "\n".join(out)
    assert n == 2 and "Solved in 2 guesses" in text and "expected" in text


def test_play_hint_then_quit(lex):
    out = []
    n = play(lex, answer="abide", read=scripted(["h", "q"]), out=out.append)
    assert n == 0 and any("word" in line and "bits" in line for line in out)


def test_play_rejects_words_outside_the_list(lex):
    out = []
    play(lex, answer="abide", read=scripted(["zzzzz", "q"]), out=out.append)
    assert any("not one of the" in line for line in out)


def test_solve_mode_plays_the_suggestion_and_scores_it_by_hand(lex):
    """Play the suggestion (Enter at the word prompt) and type the true feedback for it, as a person would
    against a real Wordle board. The reader looks up the suggestion the solver just printed."""
    secret = "abide"
    out = []
    suggestion = re.compile(r"guess \d+: ([A-Z]{5})")

    def read(prompt):
        if prompt.startswith("word"):
            return ""
        word = suggestion.findall("\n".join(out))[-1].lower()
        return feedback.to_text(feedback.score(word, secret))

    n = solve_game(lex, "entropy", read=read, out=out.append)
    text = "\n".join(out)
    assert 1 <= n <= 6 and f"Solved on guess {n}" in text
    assert n == len(re.findall(r"^guess \d+: ", text, flags=re.M))  # one suggestion line per turn


def test_solve_mode_rejects_bad_feedback_then_quits(lex):
    out = []
    n = solve_game(lex, "entropy", read=scripted(["", "gy.g", "q"]), out=out.append)
    assert n == 0 and any("needs 5" in line for line in out)


def test_watch_solves_the_answer(lex):
    out = []
    n = watch(lex, answer="slate", strategy="minimax", out=out.append, sleep=lambda s: None)
    assert 1 <= n <= 6 and any("Solved in" in line for line in out)


def test_cli_benchmark_rejects_an_unknown_strategy():
    with pytest.raises(SystemExit):
        main(["benchmark", "--strategies", "vibes", "--no-write"])


def test_cli_play_rejects_an_unknown_answer():
    with pytest.raises(SystemExit):
        main(["play", "--answer", "zzzzz"])


def test_cli_watch_runs(capsys):
    assert main(["watch", "--answer", "abide", "--delay", "0"]) == 0
    assert "guess 1" in capsys.readouterr().out

