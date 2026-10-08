import random

import pytest

from hexgame.__main__ import main, play, watch
from hexgame.agents import choose, describe_move, player
from hexgame.benchmark import play_game, run, to_markdown, wilson
from hexgame.board import (
    ACROSS,
    DOWN,
    EMPTY,
    SWAP,
    Board,
    connect,
    fill_playout,
    label,
    neighbours,
    parse_cell,
    render,
    winner_of,
)
from hexgame.heuristic import distance, shortest_path_move
from hexgame.mcts import search


def cells_from(n, pairs):
    """A board list from {cell: player} pairs."""
    cells = [EMPTY] * (n * n)
    for cell, p in pairs:
        cells[cell] = p
    return cells


def test_neighbours_are_the_six_hexagons_around_a_cell():
    nb = neighbours(5)
    centre = 2 * 5 + 2
    assert sorted(nb[centre]) == sorted([centre - 1, centre + 1, centre - 5, centre + 5, centre - 4, centre + 4])
    assert len(nb[0]) == 2  # acute corner (0, 0): right and below
    assert sorted(nb[4]) == [3, 8, 9]  # obtuse corner (0, 4): left, below, and the diagonal (1, 3)
    # Adjacency is symmetric.
    for a, others in enumerate(nb):
        for b in others:
            assert a in nb[b]


def test_connect_returns_a_shortest_chain_from_edge_to_edge():
    n = 5
    column = [0, 5, 10, 15, 20]  # column 0, top to bottom
    cells = cells_from(n, [(c, DOWN) for c in column])
    assert connect(n, cells, DOWN) == column
    assert connect(n, cells, ACROSS) is None
    # With two routes of equal length, the returned chain must still be a real chain of adjacent stones.
    cells = cells_from(n, [(c, DOWN) for c in (0, 6, 11, 16, 20, 5, 10, 15)])
    chain = connect(n, cells, DOWN)
    assert len(chain) == 5 and chain[0] < n and chain[-1] >= (n - 1) * n
    assert all(b in neighbours(n)[a] for a, b in zip(chain, chain[1:]))


def test_every_completed_board_has_exactly_one_winner():
    rng = random.Random(11)
    for n in (5, 7, 9, 11):
        for _ in range(60):
            cells = [EMPTY] * (n * n)
            w, order = fill_playout(n, cells, DOWN, rng)
            down = connect(n, cells, DOWN) is not None
            across = connect(n, cells, ACROSS) is not None
            assert down != across  # Hex has no draws
            assert w == (DOWN if down else ACROSS)
            assert sorted(order) == [i for i in range(n * n)]


def test_fill_playout_fills_in_place_and_returns_the_play_order():
    rng = random.Random(3)
    cells = cells_from(5, [(0, DOWN), (24, ACROSS)])
    w, order = fill_playout(5, cells, ACROSS, rng)
    assert len(order) == 23 and set(order) == set(range(25)) - {0, 24}
    assert winner_of(5, cells) == w
    assert all(v != EMPTY for v in cells)


def test_labels_round_trip():
    for n in (5, 11):
        for cell in range(n * n):
            assert parse_cell(n, label(n, cell)) == cell
    assert label(7, 2 * 7 + 3) == "d3"
    assert parse_cell(7, "h1") is None  # column h is off a 7x7 board
    assert parse_cell(7, "swap") == SWAP
    assert parse_cell(7, "c9") is None


def test_board_rules_and_turn_order():
    b = Board(5)
    assert b.to_move == DOWN
    b = b.play(12)
    assert b.to_move == ACROSS and b.cells[12] == DOWN
    with pytest.raises(ValueError):
        b.play(12)  # taken
    with pytest.raises(ValueError):
        b.play(25)  # off the board
    with pytest.raises(ValueError):
        b.play(SWAP)  # no swap rule
    assert SWAP not in b.legal()


def test_swap_gives_the_first_stone_to_the_second_player():
    b = Board(5, swap_rule=True).play(12)
    assert SWAP in b.legal()
    s = b.play(SWAP)
    assert s.cells[12] == ACROSS  # the second player now owns the stone
    assert s.to_move == DOWN       # and the first player moves next
    assert len(s.moves) == 2
    with pytest.raises(ValueError):
        s.play(SWAP)  # swap is only the reply to the first move


def test_game_over_stops_further_moves():
    n = 5
    # DOWN takes column 0 (cells 0, 5, 10, 15, 20); ACROSS answers in column 4, which connects nothing.
    b = Board.from_moves(n, [0, 4, 5, 9, 10, 14, 15, 19, 20])
    assert b.winner() == DOWN and b.legal() == []
    with pytest.raises(ValueError):
        b.play(24)
    assert [label(n, c) for c in b.chain()] == ["a1", "a2", "a3", "a4", "a5"]


def test_render_marks_rows_and_shows_stones():
    b = Board(5).play(12)  # c3, the centre
    text = render(b)
    assert text.splitlines()[0].strip() == "a b c d e"
    assert text.count("X") == 1 and text.count(".") == 24


def test_search_blocks_an_immediate_threat():
    # DOWN has column 0 down to row 3, so cell 20 (a5) connects it. ACROSS must block there.
    n = 5
    cells = cells_from(n, [(c, DOWN) for c in (0, 5, 10, 15)] + [(12, ACROSS)])
    res = search(n, cells, ACROSS, sims=1500, seed=4)
    assert res["move"] == 20
    assert res["visits"][20] == max(res["visits"])


def test_search_takes_an_immediate_win():
    n = 5
    cells = cells_from(n, [(c, DOWN) for c in (0, 5, 10, 15)] + [(c, ACROSS) for c in (3, 8, 13)])
    res = search(n, cells, DOWN, sims=800, seed=2)
    assert res["move"] == 20
    assert res["value"] > 0.95


def test_search_statistics_are_consistent_and_do_not_touch_the_input():
    n = 7
    cells = cells_from(n, [(24, DOWN), (0, ACROSS)])
    snapshot = list(cells)
    for rave in (True, False):
        res = search(n, cells, ACROSS, sims=400, rave=rave, seed=9)
        assert res["sims"] == 400
        assert sum(res["visits"]) == 400  # every simulation passes through exactly one root child
        assert 0 < res["playouts"] <= res["sims"]
        assert all(res["visits"][c] == 0 for c in (0, 24))  # occupied cells are never children
        assert res["move"] in [c for c in range(n * n) if res["visits"][c] > 0]
        assert res["pv"][0] == res["move"] and len(res["pv"]) >= 1
        assert 0.0 <= res["value"] <= 1.0
        assert res["rave"] is rave
    assert cells == snapshot


def test_rave_uses_amaf_statistics_once_the_tree_has_them():
    # After a search, the root's AMAF counts must be filled for moves that were played in playouts.
    n = 5
    res = search(n, [EMPTY] * 25, DOWN, sims=300, rave=True, seed=5)
    assert sum(1 for r in res["rave_rate"] if r is not None) >= 20


def test_distance_and_shortest_path_player():
    n = 5
    assert distance(n, [EMPTY] * 25, DOWN) == 5  # one empty cell per row
    cells = cells_from(n, [(0, DOWN)])
    assert distance(n, cells, DOWN) == 4
    # DOWN's column walls off ACROSS's rows 0-3, so ACROSS must go through a5 and then cross five columns.
    cells = cells_from(n, [(c, DOWN) for c in (0, 5, 10, 15)])
    assert distance(n, cells, ACROSS) == 5
    # The race heuristic blocks the one-move threat as well.
    cells = cells_from(n, [(c, DOWN) for c in (0, 5, 10, 15)] + [(12, ACROSS)])
    move = shortest_path_move(n, cells, ACROSS, random.Random(1))
    assert move == 20


def test_wilson_interval_brackets_the_rate():
    lo, hi = wilson(30, 40)
    assert lo < 0.75 < hi
    assert wilson(0, 0) == (0.0, 0.0)
    assert wilson(40, 40)[1] == 1.0


def test_players_finish_a_game_and_describe_their_moves():
    n = 5
    winner, moves = play_game(n, player("random"), player("shortest"), seed=21)
    assert winner in (DOWN, ACROSS) and 5 <= moves <= 25
    move, analysis = choose(Board(n), "uct", sims=200, seconds=None, seed=1)
    assert 0 <= move < 25
    assert describe_move(n, move, analysis).startswith(label(n, move))


def test_swap_decision_returns_a_legal_move():
    b = Board(5, swap_rule=True).play(12)
    move, analysis = choose(b, "rave", sims=300, seconds=None, seed=8)
    assert move in b.legal()
    assert analysis["swap"]["chose"] in ("swap", "play")
    assert (move == SWAP) == (analysis["swap"]["chose"] == "swap")


def test_small_benchmark_has_the_right_shape():
    result = run(size=5, sims=60, games=4, matchups=[("rave", "random"), ("uct", "shortest")])
    assert [m["games"] for m in result["matchups"]] == [4, 4]
    assert all(0 <= m["a_wins"] <= 4 for m in result["matchups"])
    assert result["speed"]["rave"]["sims_per_second"] > 0
    md = to_markdown(result)
    assert "| matchup |" in md and "rave vs random" in md


def test_play_quits_and_reports_bad_input():
    lines = []
    script = iter(["zz", "h", "q"])
    board, winner = play(5, DOWN, "random", 1, False, seed=1,
                         read=lambda prompt: next(script), out=lines.append)
    assert winner is None and board.moves == []
    assert any("not a legal move" in line for line in lines)
    assert any("hint:" in line for line in lines)


def test_play_runs_the_agent_replies_until_the_player_quits():
    lines = []
    answers = iter(["a1", "b1", "c1"])  # the player's moves; anything after that quits
    board, winner = play(5, DOWN, "shortest", 1, False, seed=2,
                         read=lambda prompt: next(answers, "q"), out=lines.append)
    assert winner is None
    # A player move that lands on an agent stone is refused and asks again, so the exact count depends on
    # the agent's replies. What must hold: moves come in pairs (player, then agent), and every agent move is printed.
    assert len(board.moves) % 2 == 0 and len(board.moves) >= 2
    assert sum("SHORTEST plays" in line for line in lines) == len(board.moves) // 2


def test_watch_plays_one_game_to_the_end():
    lines = []
    w = watch(5, "rave", "shortest", sims=60, seed=3, swap=False, out=lines.append)
    assert w in (DOWN, ACROSS)
    assert any("wins after" in line for line in lines)


def test_main_rejects_bad_size_and_runs_benchmark_help(capsys):
    with pytest.raises(SystemExit):
        main(["play", "-n", "4"])
    with pytest.raises(SystemExit):
        main(["--help"])
    capsys.readouterr()
