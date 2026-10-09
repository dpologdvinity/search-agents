import itertools
import random
import time

import pytest

from minesweeper.__main__ import main, play_game, render, watch_game
from minesweeper.agents import AGENT_NAMES, Agent, best_guess, play
from minesweeper.benchmark import run_benchmark, to_markdown, wilson
from minesweeper.board import UNKNOWN, Game, View, col_name, label, neighbour_table, parse_cell
from minesweeper.inference import LEVEL_CSP, LEVEL_PROBABILITY, LEVEL_RULES, analyse


def game_with_mines(rows, cols, mines_at):
    """A game whose mine layout is fixed, so the rules can be checked by hand."""
    g = Game(rows, cols, len(mines_at), random.Random(0))
    g._mine = set(mines_at)
    return g


def brute_probabilities(view: View):
    """Exact mine probabilities by enumerating every layout that fits the numbers and the mine count."""
    table = neighbour_table(view.rows, view.cols)
    covered = [i for i, v in enumerate(view.cells) if v == UNKNOWN]
    count = 0
    per = {c: 0 for c in covered}
    for combo in itertools.combinations(covered, view.mines):
        s = set(combo)
        if all(v < 0 or sum(1 for n in table[i] if n in s) == v for i, v in enumerate(view.cells)):
            count += 1
            for c in combo:
                per[c] += 1
    return count, per


# ── Rules ────────────────────────────────────────────────────────────────


def test_column_names_and_cell_labels():
    assert [col_name(c) for c in (0, 1, 25, 26, 27, 29)] == ["a", "b", "z", "aa", "ab", "ad"]
    assert label(30, 30 * 15 + 29) == "ad16"
    assert parse_cell(16, 30, "ad16") == 30 * 15 + 29
    assert parse_cell(9, 9, " C5 ") == 4 * 9 + 2
    assert parse_cell(9, 9, "j1") is None  # column past the board
    assert parse_cell(9, 9, "c10") is None  # row past the board
    assert parse_cell(9, 9, "5c") is None


def test_neighbour_counts():
    t = neighbour_table(5, 5)
    assert len(t[0]) == 3 and len(t[2]) == 5 and len(t[12]) == 8


def test_first_click_is_safe_and_opens():
    for seed in range(20):
        rng = random.Random(seed)
        g = Game(9, 9, 10, rng)
        first = rng.randrange(81)
        g.reveal(first)
        assert g.lost_on is None
        assert g.revealed[first] != UNKNOWN
        assert first not in g.mine_set()
        assert not (set(neighbour_table(9, 9)[first]) & g.mine_set())
        assert len(g.mine_set()) == 10


def test_flood_fill_opens_zeros_and_stops_at_numbers():
    # Mines in the top-right corner only. Revealing the bottom-left corner opens the whole rest.
    g = game_with_mines(4, 4, [3, 7])
    opened = g.reveal(0)
    assert g.revealed[0] == 0 and len(opened) == 16 - 2 - 0
    # Cell 2 touches both mines (3 and 7) and cell 11 touches one (7): they show numbers and stop the spread.
    assert g.revealed[2] == 2 and g.revealed[6] == 2 and g.revealed[11] == 1
    assert g.revealed[3] == UNKNOWN and g.revealed[7] == UNKNOWN


def test_win_and_loss():
    g = game_with_mines(2, 2, [3])
    g.reveal(0)
    g.reveal(1)
    g.reveal(2)
    assert g.won and g.over
    g = game_with_mines(2, 2, [3])
    g.reveal(3)
    assert g.lost_on == 3 and g.over and not g.won


def test_flags_block_reveals_until_removed():
    g = game_with_mines(3, 3, [8])
    assert g.toggle_flag(0)
    assert g.reveal(0) == []  # a flagged cell cannot be revealed
    assert g.toggle_flag(0)  # unflag it, then it reveals
    assert g.reveal(0)
    assert not g.toggle_flag(0)  # a revealed cell cannot be flagged


def test_view_shows_numbers_and_hides_the_layout():
    g = game_with_mines(3, 3, [4])  # the centre is the only mine
    g.reveal(0)  # a corner next to the centre mine shows 1 and opens nothing
    v = g.view()
    assert v.cells[0] == 1 and v.cells[4] == UNKNOWN
    assert sum(1 for x in v.cells if x != UNKNOWN) == 1
    assert v.mines == 1


# ── Inference ────────────────────────────────────────────────────────────


def test_single_cell_rules_on_a_hand_board():
    # 1x3 with one mine: the 1 at the left has one covered neighbour, so that neighbour is the mine.
    view = View(1, 3, 1, (1, UNKNOWN, UNKNOWN))
    a = analyse(view, LEVEL_RULES)
    assert a.mines == (1,) and a.safe == ()
    # The rules cannot see the mine count. Level 2 can: the mine is already placed, so cell 2 is clear.
    a = analyse(view, LEVEL_CSP)
    assert a.mines == (1,) and a.safe == (2,)


def test_rules_propagate_to_a_fixpoint():
    # 1x5 with two mines: the 1 at cell 0 forces cell 1; then the 2 at cell 2 needs one more among cell 3 alone.
    view = View(1, 5, 2, (1, UNKNOWN, 2, UNKNOWN, UNKNOWN))
    a = analyse(view, LEVEL_RULES)
    assert a.mines == (1, 3)
    assert 4 in a.covered  # nothing touches cell 4, so the rules say nothing about it


def test_components_split_independent_frontiers():
    # 1x7, two mines. The 1 at cell 1 covers {0, 2}; the 1 at cell 5 covers {4, 6}. Neither pair is
    # settled by the rules, and the two pairs share no cell, so they are two components. Cell 3 is
    # touched by no number and the mine count is already used up by the pairs, so it is safe.
    view = View(1, 7, 2, (UNKNOWN, 1, UNKNOWN, UNKNOWN, UNKNOWN, 1, UNKNOWN))
    a = analyse(view, LEVEL_CSP)
    assert sorted(c.cells for c in a.components) == [(0, 2), (4, 6)]
    assert all(c.layouts == 2 for c in a.components)
    assert 3 in a.safe
    assert a.probability(0) == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(60))
def test_counts_match_brute_force(seed):
    rng = random.Random(seed)
    rows, cols = rng.choice([(3, 4), (4, 4), (4, 5), (5, 5)])
    mines = rng.randint(2, max(2, rows * cols // 4))
    g = Game(rows, cols, mines, random.Random(rng.random()))
    g.reveal(rng.randrange(rows * cols))
    # Reveal at random until few enough cells are covered for brute force to enumerate every layout.
    while not g.over and sum(1 for v in g.view().cells if v == UNKNOWN) > 14:
        cov = [i for i, v in enumerate(g.view().cells) if v == UNKNOWN]
        g.reveal(rng.choice(cov))
    view = g.view()
    if g.over:
        pytest.skip("the game ended before the board narrowed")
    count, per = brute_probabilities(view)
    a = analyse(view, LEVEL_PROBABILITY)
    for c in range(rows * cols):
        if view.cells[c] != UNKNOWN:
            continue
        exact = per[c] / count
        if c in a.mines:
            assert exact == 1.0
        elif c in a.safe:
            assert exact == 0.0
        else:
            assert a.probability(c) == pytest.approx(exact, abs=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_proofs_are_sound_on_real_games(seed):
    rows, cols, mines = 9, 9, 10
    rng = random.Random(seed)
    g = Game(rows, cols, mines, rng)
    g.reveal(rng.randrange(81))
    truth = g.mine_set()
    while not g.over:
        view = g.view()
        for level in (LEVEL_RULES, LEVEL_CSP, LEVEL_PROBABILITY):
            a = analyse(view, level)
            assert set(a.safe).isdisjoint(truth), (seed, level)
            assert set(a.mines) <= truth, (seed, level)
        a = analyse(view, LEVEL_PROBABILITY)
        if a.safe:
            g.reveal(a.safe[0])
        else:
            g.reveal(best_guess(a))


def test_probabilities_sum_to_the_remaining_mines():
    # Every layout has exactly `remaining` mines among the unproven cells, so the weights must add up to that.
    rng = random.Random(3)
    g = Game(9, 9, 10, rng)
    g.reveal(rng.randrange(81))
    for _ in range(12):
        if g.over:
            break
        a = analyse(g.view(), LEVEL_PROBABILITY)
        if a.safe:
            g.reveal(a.safe[0])
        else:
            g.reveal(best_guess(a))
    a = analyse(g.view(), LEVEL_PROBABILITY)
    assert a.total > 0
    assert sum(a.numerator.values()) == a.total * a.remaining


def test_inconsistent_view_is_rejected():
    view = View(2, 2, 1, (UNKNOWN, 2, UNKNOWN, UNKNOWN))  # a 2 with only three covered cells and one mine
    with pytest.raises(ValueError):
        analyse(view, LEVEL_PROBABILITY)


# ── Agents and games ─────────────────────────────────────────────────────


def test_first_click_is_not_a_guess():
    r = play(9, 9, 10, seed=5, kind="random")
    assert r.guesses <= r.reveals


def test_games_are_repeatable_per_seed():
    a = play(9, 9, 10, seed=42, kind="probability")
    b = play(9, 9, 10, seed=42, kind="probability")
    assert (a.won, a.guesses, a.reveals) == (b.won, b.guesses, b.reveals)


def test_probability_agent_beats_random_on_beginner():
    wins_random = sum(play(9, 9, 10, s, "random").won for s in range(30))
    wins_prob = sum(play(9, 9, 10, s, "probability").won for s in range(30))
    assert wins_prob > wins_random and wins_prob >= 20


def test_best_guess_breaks_ties_toward_more_covered_neighbours():
    # Nothing is known and every cell is 1/3 likely. The middle cell has two covered neighbours,
    # so a zero there would open the most board: that is the guess.
    view = View(1, 3, 1, (UNKNOWN, UNKNOWN, UNKNOWN))
    a = analyse(view, LEVEL_PROBABILITY)
    assert all(a.probability(c) == pytest.approx(1 / 3) for c in range(3))
    assert best_guess(a) == 1


def test_agent_names_are_checked():
    with pytest.raises(ValueError):
        Agent("oracle")
    assert set(AGENT_NAMES) == {"random", "rules", "csp", "probability"}


# ── Benchmark ────────────────────────────────────────────────────────────


def test_wilson_interval_brackets_the_rate():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and lo == pytest.approx(0.404, abs=0.002)
    assert wilson(0, 10)[0] == 0.0
    assert wilson(0, 0) == (0.0, 0.0)


def test_small_benchmark_is_paired_and_renders():
    result = run_benchmark({"beginner": 6}, agents=("random", "probability"), presets=("beginner",))
    assert set(result["agents"]) == {"random", "probability"}
    md = to_markdown(result)
    assert "Beginner: 9x9 with 10 mines (6 games)" in md and "| probability |" in md


# ── CLI ──────────────────────────────────────────────────────────────────


def test_play_game_scripted_inputs():
    lines = iter(["zz", "f a1", "h", "p", "q"])
    out = []
    reveals = play_game("beginner", seed=3, read=lambda prompt: next(lines), out=out.append)
    text = "\n".join(out)
    assert "is not a cell" in text
    assert "hint" in text or "certain" in text or "no certain move" in text
    assert "frontier components" in text
    assert reveals == 0


def test_play_game_reveal_and_end():
    out = []
    seq = iter(["e5"] * 1 + ["q"])
    play_game("beginner", seed=3, read=lambda prompt: next(seq), out=out.append)
    assert any("#" in line for line in out)  # the board is shown


def test_render_marks_covered_flagged_and_mines():
    g = game_with_mines(2, 2, [0])
    g.toggle_flag(1)
    text = render(g)
    assert "#" in text and "F" in text and "*" not in text
    g.reveal(0)
    assert "*" in render(g)


def test_watch_plays_to_the_end_without_sleeping():
    out = []
    won = watch_game("beginner", "probability", seed=11, delay=0.0, out=out.append, sleep=lambda s: None)
    assert isinstance(won, bool)
    assert out[-1] in ("Cleared.",) or out[-1].startswith("Lost after")


def test_cli_benchmark_writes_results(tmp_path, capsys):
    main(["benchmark", "--games", "2", "--out", str(tmp_path)])
    assert (tmp_path / "minesweeper_benchmark.json").exists()
    assert (tmp_path / "minesweeper_benchmark.md").exists()
    assert "Expert" in capsys.readouterr().out


def test_cli_bare_options_mean_play(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "q")
    assert main(["-p", "beginner", "--seed", "1"]) == 0
    assert "Minesweeper beginner" in capsys.readouterr().out


def test_cli_benchmark_writes_only_when_asked(tmp_path, monkeypatch, capsys):
    # Without --out the run only prints: a run from the repo root must not rewrite the committed results.
    monkeypatch.chdir(tmp_path)
    assert main(["benchmark", "--games", "1"]) == 0
    assert list(tmp_path.iterdir()) == []
    assert "Expert" in capsys.readouterr().out


def test_a_centre_first_click_on_a_tiny_board_still_places_mines():
    # On 3x3 the click's neighbourhood is the whole board, so only the click itself stays clear.
    g = Game(3, 3, 2, random.Random(0))
    g.reveal(4)
    assert len(g.mine_set()) == 2 and 4 not in g.mine_set()


def crafted_board(rows: int, cols: int, mines: int, shows, seed: int) -> View:
    """A consistent board: random mines, and only the covered-or-revealed pattern `shows(r, c)` picks.

    Patterns such as every other row leave long chains of covered cells between numbers, which is what
    makes the exact count expensive. Every number agrees with the layout, so the board is a real one.
    """
    rng = random.Random(seed)
    layout = set(rng.sample(range(rows * cols), mines))
    table = neighbour_table(rows, cols)
    cells = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        if i not in layout and shows(r, c):
            cells.append(sum(1 for n in table[i] if n in layout))
        else:
            cells.append(UNKNOWN)
    return View(rows, cols, mines, tuple(cells))


# Two wide-frontier patterns. Before the work budget these ran for over a minute on the 16x30 board.
WIDE_PATTERNS = {
    "every other row": lambda r, c: r % 2 == 0,
    "checkerboard": lambda r, c: (r + c) % 2 == 0,
}


@pytest.mark.parametrize("name", list(WIDE_PATTERNS))
def test_wide_frontier_boards_stop_at_the_work_budget(name):
    view = crafted_board(16, 30, 99, WIDE_PATTERNS[name], seed=1)
    start = time.perf_counter()
    a = analyse(view, LEVEL_PROBABILITY)
    # The budget keeps this near a second; the bound is loose so a slow machine does not fail the test.
    assert time.perf_counter() - start < 15
    assert a.exact is False
    assert all(a.probability(c) is not None and 0 <= a.probability(c) <= 1 for c in a.covered)


def test_within_budget_the_count_is_exact_and_matches_an_unlimited_one():
    g = Game(16, 16, 40, random.Random(3))
    g.reveal(136)
    for _ in range(40):  # a few real-play positions, each with its own frontier
        view = g.view()
        a = analyse(view, LEVEL_PROBABILITY)
        b = analyse(view, LEVEL_PROBABILITY, budget=10**9)
        assert a.exact and b.exact
        assert (a.numerator, a.total, a.safe, a.mines) == (b.numerator, b.total, b.safe, b.mines)
        covered = [i for i, v in enumerate(view.cells) if v == UNKNOWN]
        if a.safe:
            g.reveal(a.safe[0])
        elif covered:
            g.reveal(best_guess(a))
        if g.over:
            break


def test_over_budget_keeps_the_proofs_and_gives_estimates_instead():
    view = crafted_board(9, 9, 10, WIDE_PATTERNS["checkerboard"], seed=2)
    exact = analyse(view, LEVEL_PROBABILITY, budget=10**9)
    fallback = analyse(view, LEVEL_PROBABILITY, budget=1)
    assert exact.exact and not fallback.exact
    assert set(fallback.safe) <= set(exact.safe) and set(fallback.mines) <= set(exact.mines)
    assert fallback.numerator == {}
    assert all(fallback.probability(c) is not None and 0 <= fallback.probability(c) <= 1
               for c in fallback.covered)
    guess = best_guess(fallback)
    assert fallback.estimate[guess] == min(fallback.estimate[c] for c in fallback.covered)
