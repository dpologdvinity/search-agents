"""Markov chain text: counts on tiny texts, probabilities, temperature, seeded sampling, copy stats, perplexity."""

import argparse
import math
import re
from pathlib import Path

import pytest

from bandits.rng import Rng
from markov import (
    COPY_RUN,
    CORPORA,
    NGramModel,
    copy_report,
    detokenize,
    generate,
    load_corpus,
    ngram_sets,
    perplexity,
    split_index,
    tokenize,
)
from markov.__main__ import main, parse_orders
from markov.corpus import DATA_DIR
from markov.model import Generation, Pick

ROOT = Path(__file__).resolve().parent.parent


def test_word_tokens_keep_punctuation_and_apostrophes():
    assert tokenize("Hi, there!  don't\n stop.") == ["Hi", ",", "there", "!", "don't", "stop", "."]


def test_char_tokens_collapse_whitespace():
    assert tokenize("a \n b", "char") == ["a", " ", "b"]


def test_unknown_level_is_an_error():
    with pytest.raises(ValueError):
        tokenize("x", "syllable")


def test_detokenize_hugs_punctuation():
    toks = tokenize("Well, (said she) it is done.")
    assert detokenize(toks) == "Well, (said she) it is done."


def test_counts_on_a_tiny_text():
    toks = "a b a b a c".split()
    m1 = NGramModel(toks, 1)
    assert m1.successors == {(): {"a": 3, "b": 2, "c": 1}}
    m2 = NGramModel(toks, 2)
    assert m2.successors == {("a",): {"b": 2, "c": 1}, ("b",): {"a": 2}}
    assert m2.context_count == 2
    assert m2.ngram_count == 3
    assert m2.vocab_size == 3


def test_order_longer_than_text_is_rejected():
    with pytest.raises(ValueError):
        NGramModel(["a", "b"], 3)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_probabilities_sum_to_one_for_every_context(temperature):
    toks = tokenize(load_corpus("pride").body)
    model = NGramModel(toks, 2)
    for ctx in list(model.successors)[:50]:
        dist = model.distribution(ctx, temperature)
        assert math.isclose(sum(p for _, p in dist), 1.0, rel_tol=1e-12)


def test_unseen_context_has_no_distribution():
    model = NGramModel("a b c".split(), 2)
    assert model.distribution(("zzz",)) is None


def test_temperature_sharpens_and_flattens():
    model = NGramModel("a b a b a b a c".split(), 1)
    p_cold = dict(model.distribution((), 0.2))
    p_warm = dict(model.distribution((), 1.0))
    p_hot = dict(model.distribution((), 3.0))
    assert p_cold["a"] > p_warm["a"] > p_hot["a"]
    assert p_warm["a"] == pytest.approx(4 / 8)  # raw counts at T = 1: a 4 times, b 3, c 1


def test_sampling_is_deterministic_for_a_seed():
    toks = tokenize(load_corpus("alice").body)
    model = NGramModel(toks, 3)
    first = generate(model, Rng(7), 80).tokens
    again = generate(model, Rng(7), 80).tokens
    other = generate(model, Rng(8), 80).tokens
    assert first == again
    assert first != other


def test_generation_emits_the_requested_count_and_stays_in_vocabulary():
    toks = tokenize(load_corpus("sonnets").body)
    model = NGramModel(toks, 2)
    gen = generate(model, Rng(3), 50, 1.0)
    assert len(gen.picks) == 50
    assert set(p.token for p in gen.picks) <= set(toks)
    assert all(0 < p.prob <= 1 for p in gen.picks)


def test_copy_share_rises_with_order():
    """Alice, words, seed 1, 120 picks: order 1 copies no pick and order 5 copies every pick. The seed fixes the
    run, so these shares are exact, not approximate."""
    toks = tokenize(load_corpus("alice").body)
    sets = ngram_sets(toks)
    shares = {}
    for order in (1, 5):
        gen = generate(NGramModel(toks, order), Rng(1), 120)
        shares[order] = copy_report(gen, sets, COPY_RUN["word"]).copied_pct
    assert shares[1] == 0.0
    assert shares[5] == 100.0


def test_copy_report_flags_novel_text():
    toks = "a b c d e".split()
    # With min_run 2 the corpus windows are "a b", "b c", "c d" and "d e". The prompt "a b" is a copied window, but
    # the picks are not: "b zzz" and "zzz c" are novel, and "c" alone is not a window with its neighbour.
    gen = Generation(["a", "b", "zzz", "c"], 2, [Pick(2, "zzz", 1.0), Pick(3, "c", 1.0)], 0)
    report = copy_report(gen, ngram_sets(toks), 2)
    assert report.copied == [False, False]
    assert report.copied_pct == 0.0
    assert report.longest_run == 1  # "c" alone is in the corpus; "zzz c" is not


def test_copy_report_matches_a_brute_force_check_on_sampled_text():
    """Each pick is copied exactly when some window of COPY_RUN tokens that covers it occurs in the corpus. This
    checks every window that covers the token, a different route from the marking loop in copy_report."""
    toks = tokenize(load_corpus("sonnets").body)
    sets = ngram_sets(toks)
    min_run = COPY_RUN["word"]
    corpus_windows = {tuple(toks[i : i + min_run]) for i in range(len(toks) - min_run + 1)}
    for order in (1, 3):
        gen = generate(NGramModel(toks, order), Rng(5), 80)
        out = gen.tokens
        covered = [
            any(
                tuple(out[s : s + min_run]) in corpus_windows
                for s in range(max(0, t - min_run + 1), min(t, len(out) - min_run) + 1)
            )
            for t in range(len(out))
        ]
        report = copy_report(gen, sets, min_run)
        assert report.copied == [covered[p.index] for p in gen.picks], order


@pytest.mark.parametrize("name", CORPORA)
def test_train_perplexity_falls_with_order(name):
    """Train perplexity drops steeply from order 1 to 3, then plateaus. Add-alpha smoothing can nudge the
    plateau up by a fraction of a percent, so the later orders are allowed a 2% wobble."""
    toks = tokenize(load_corpus(name).body)
    cut = split_index(toks)
    train = toks[:cut]
    values = [perplexity(NGramModel(train, o), train, 0) for o in range(1, 6)]
    assert values[0] > values[1] > values[2], values
    assert all(b <= a * 1.02 for a, b in zip(values[2:], values[3:])), values


@pytest.mark.parametrize("name", CORPORA)
def test_held_out_perplexity_is_finite_and_above_one(name):
    toks = tokenize(load_corpus(name).body)
    cut = split_index(toks)
    model = NGramModel(toks[:cut], 3)
    value = perplexity(model, toks, cut)
    assert 1.0 < value < float("inf")


def test_split_holds_out_the_tail():
    toks = list("abcdefghij")
    assert split_index(toks, 0.2) == 8


def test_parse_orders_accepts_ranges_and_lists():
    assert parse_orders("1..5") == [1, 2, 3, 4, 5]
    assert parse_orders("2-3") == [2, 3]
    assert parse_orders("1,4") == [1, 4]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_orders("0..3")
    assert parse_orders("5,5") == [5]  # a repeated order would print the same row twice
    assert parse_orders("3,1,3") == [3, 1]


def test_cli_rejects_zero_words_and_non_positive_alpha(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["generate", "--words", "0"])
    assert exit_info.value.code == 2
    for alpha in ("0", "-0.1", "nan"):
        with pytest.raises(SystemExit) as exit_info:
            main(["perplexity", "--alpha", alpha])
        assert exit_info.value.code == 2
    assert "must be above 0" in capsys.readouterr().err


def test_corpora_have_a_source_header_and_stay_small():
    for name in CORPORA:
        raw = (DATA_DIR / f"{name}.txt").read_text(encoding="utf-8")
        head = raw.partition("\n---\n")[0]
        assert head.startswith("Source:"), name
        assert "public domain in the US" in head or "US government" in head, name
        assert "gutenberg.org/ebooks/" in head or "archives.gov" in head, name
        assert len(raw.encode()) <= 30_000, name


def test_corpora_bodies_avoid_the_banned_words():
    banned = re.compile("cou" "rse|orig" "inal|assign" "ment|home" "work", re.IGNORECASE)
    for name in CORPORA:
        assert not banned.search(load_corpus(name).body), name


def test_curly_apostrophes_stay_inside_words():
    assert tokenize("don’t stop, men’s") == ["don’t", "stop", ",", "men’s"]


def test_web_copies_match_the_package_corpora():
    for name in CORPORA:
        pkg = (DATA_DIR / f"{name}.txt").read_bytes()
        web = (ROOT / "web" / "data" / "markov" / f"{name}.txt").read_bytes()
        assert pkg == web, name


def test_cli_generate_and_perplexity_run(capsys):
    assert main(["generate", "--corpus", "constitution", "--order", "2", "--words", "12", "--seed", "4"]) == 0
    out = capsys.readouterr()
    assert out.out.strip()
    assert "copied" in out.err
    assert main(["perplexity", "--corpus", "constitution", "--order", "1..3"]) == 0
    table = capsys.readouterr().out
    assert "held-out ppl" in table
    assert len(table.strip().splitlines()) == 2 + 3
