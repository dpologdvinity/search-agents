"""Checkers board sizes, the forced-capture rule, and the agents on every size.

The 8x8 forced-capture behaviour is covered by test_checkers.py. This file covers what
the bigger boards and the optional-capture rule add.
"""

import random

import pytest

from checkers.__main__ import main, players_from
from checkers.agents import AGENTS, DESCRIPTIONS, choose, describe
from checkers.board import RED, WHITE, geometry, index, legal_moves, parse_board, perft, start_position
from checkers.mcts import MCTS
from checkers.search import WIN, AlphaBeta, Greedy, Minimax, RandomMove, evaluate

SIZES = (8, 10, 12)


def board_with(pieces, n=8):
    """Board from {(row, col): piece} on an n x n board."""
    b = [0] * (n * n // 2)
    for (r, c), p in pieces.items():
        b[index(r, c, n)] = p
    return tuple(b)


def midgame(n, plies, seed):
    """A position a few random legal plies into the game, with the side to move."""
    rng = random.Random(seed)
    b, side = start_position(n), RED
    for _ in range(plies):
        moves = legal_moves(b, side)
        if not moves:
            break
        b, side = rng.choice(moves).result, -side
    return b, side


# ── Geometry and start positions ─────────────────────────────────────────

def test_square_counts_and_start_rows():
    assert [geometry(start_position(n)).cells for n in SIZES] == [32, 50, 72]
    assert [start_position(n).count(RED) for n in SIZES] == [12, 20, 30]
    assert [start_position(n).count(WHITE) for n in SIZES] == [12, 20, 30]


def test_centre_squares_on_8x8_are_unchanged():
    # The centre set must stay as it was, so 8x8 evaluations do not move.
    assert geometry(start_position(8)).CENTER == frozenset({9, 10, 13, 14, 17, 18, 21, 22})


@pytest.mark.parametrize("n", SIZES)
def test_start_position_is_symmetric_and_evaluates_to_zero(n):
    start = start_position(n)
    assert evaluate(start, RED) == 0 == evaluate(start, WHITE)


def test_parse_board_checks_the_size():
    assert len(parse_board([0] * 50)) == 50  # size inferred from the length
    assert len(parse_board([0] * 50, 10)) == 50
    with pytest.raises(ValueError):
        parse_board([0] * 50, 8)  # a 10x10 board sent as 8x8
    with pytest.raises(ValueError):
        parse_board([0] * 40)  # not a supported size
    with pytest.raises(ValueError):
        parse_board([5] + [0] * 49)  # values must be from -2 to 2


# ── Move generation ──────────────────────────────────────────────────────

# Perft pinned from this implementation. Depth 1 is checked by hand: on 10x10 only the front
# red row (row 6) can move, with 5 men giving 2+2+2+2+1 moves = 9; on 12x12 the front row
# (row 7) has 1+2+2+2+2+2 = 11. Depth 2 is 9 x 9 and 11 x 11 because every reply leaves
# white's front row with the same number of moves.
@pytest.mark.parametrize("n, counts", [(10, [9, 81, 658]), (12, [11, 121, 1222])])
def test_perft_of_the_bigger_start_positions(n, counts):
    assert [perft(start_position(n), RED, d) for d in range(1, 4)] == counts


def test_perft_with_optional_captures_is_pinned_on_8x8():
    # Depth 1 and 2 are the same as with forced captures (no captures are possible yet);
    # depth 3 is where a capture can first be made optionally.
    assert [perft(start_position(8), RED, d, forced=False) for d in range(1, 4)] == [7, 49, 379]


def test_optional_captures_add_simple_moves_when_a_capture_exists():
    # Red can capture white at (5,2) from (6,1), and also has two free men.
    b = board_with({(6, 1): 1, (5, 2): -1, (6, 5): 1, (0, 7): -1})
    forced = legal_moves(b, RED, forced=True)
    assert len(forced) == 1 and forced[0].captured  # only the capture is legal
    optional = legal_moves(b, RED, forced=False)
    assert sum(1 for m in optional if m.captured) == 1
    assert sorted(m.path for m in optional if not m.captured) == sorted([
        (index(6, 1), index(5, 0)), (index(6, 5), index(5, 4)), (index(6, 5), index(5, 6))])


def test_optional_captures_still_finish_a_chain():
    # The double jump (6,1) -> (4,3) -> (2,5) must be completed; it cannot stop after one jump.
    b = board_with({(6, 1): 1, (5, 2): -1, (3, 4): -1, (0, 7): -1})
    captures = [m for m in legal_moves(b, RED, forced=False) if m.captured]
    assert len(captures) == 1
    assert captures[0].path == (index(6, 1), index(4, 3), index(2, 5))


def test_forced_and_optional_agree_when_no_capture_exists():
    b = start_position(10)
    assert legal_moves(b, RED, forced=True) == legal_moves(b, RED, forced=False)


@pytest.mark.parametrize("n", SIZES)
def test_kings_move_one_step_on_every_size(n):
    # American rules: no flying kings. A king in the middle of the board has four
    # neighbours and no captures, so it has exactly four one-step moves.
    b = board_with({(n // 2, n // 2 - 1): 2, (0, n - 1): -1}, n)
    moves = legal_moves(b, RED)
    assert len(moves) == 4
    assert all(len(m.path) == 2 and not m.captured for m in moves)


# ── Agents on every size, both capture rules ─────────────────────────────

@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("forced", [True, False])
@pytest.mark.parametrize("n", SIZES)
def test_every_agent_returns_a_legal_move(agent, forced, n):
    b, side = midgame(n, 12, seed=n)
    level = 1
    result = choose(b, side, agent, level, forced=forced, by_nodes=True)
    legal = {tuple(m.path) for m in legal_moves(b, side, forced)}
    assert tuple(result["move"]["path"]) in legal
    assert tuple(result["move"]["result"]) in {m.result for m in legal_moves(b, side, forced)}
    assert result["analysis"]["agent"] == agent


def test_alpha_beta_matches_minimax_with_optional_captures():
    # Stand-pat quiescence must score the same in both searches when captures are optional.
    for seed in range(3):
        b, side = midgame(8, 10, seed)
        mm = Minimax(3, forced=False).search(b, side)
        ab = AlphaBeta(max_depth=3, max_seconds=60, forced=False).search(b, side)
        assert ab.score == mm.score


def test_greedy_takes_the_winning_capture():
    b = board_with({(5, 2): 1, (4, 3): -1})
    info = Greedy().search(b, RED)
    assert info.move.captured == (index(4, 3),)
    assert info.score == WIN - 1


def test_random_move_is_seeded():
    b, side = midgame(10, 10, seed=2)
    first = RandomMove(seed=7).search(b, side).move
    assert RandomMove(seed=7).search(b, side).move == first
    assert first in legal_moves(b, side)


def test_mcts_is_deterministic_with_a_seed():
    b, side = midgame(8, 8, seed=4)
    a = MCTS(iterations=200, seed=3).search(b, side)
    c = MCTS(iterations=200, seed=3).search(b, side)
    assert a.move == c.move
    assert a.score == c.score
    assert [s for _, s in a.root] == [s for _, s in c.root]
    assert a.rollouts == 200


def test_mcts_reports_a_win_rate_for_its_move():
    b, side = midgame(8, 8, seed=4)
    info = MCTS(iterations=300, seed=1).search(b, side)
    assert 0.0 <= info.score <= 1.0
    assert describe(info.score)["result"] == "rate"


def test_mcts_finds_the_winning_capture():
    b = board_with({(5, 2): 1, (4, 3): -1})
    info = MCTS(iterations=50, seed=0).search(b, RED)
    assert info.move.captured == (index(4, 3),)


def test_mcts_rejects_bad_settings():
    with pytest.raises(ValueError):
        MCTS(iterations=0)
    with pytest.raises(ValueError):
        MCTS(policy="flying")


def test_describe_handles_every_kind_of_score():
    assert describe(None) is None
    assert describe(0.25) == {"result": "rate", "percent": 25.0}
    assert describe(WIN - 3) == {"result": "win", "plies": 3}
    assert describe(-(WIN - 3)) == {"result": "loss", "plies": 3}
    assert describe(12) == {"result": "eval", "score": 12}


def test_every_agent_has_a_description():
    assert set(AGENTS) == set(DESCRIPTIONS)


# ── Command line ─────────────────────────────────────────────────────────

def test_players_from_covers_the_flag_combinations():
    class Args:
        agent, red, white, watch = "minimax", None, None, None

    def players(**kw):
        a = Args()
        for k, v in kw.items():
            setattr(a, k, v)
        return players_from(a)

    assert players() == {RED: "human", WHITE: "minimax"}
    assert players(white="human") == {RED: "minimax", WHITE: "human"}  # old --white
    assert players(white="greedy") == {RED: "human", WHITE: "greedy"}
    assert players(red="mcts", white="alphabeta") == {RED: "mcts", WHITE: "alphabeta"}
    assert players(red="mcts") == {RED: "mcts", WHITE: "human"}
    assert players(watch=["random", "greedy"]) == {RED: "random", WHITE: "greedy"}


def test_cli_match_plays_two_games_and_prints_a_table(capsys):
    assert main(["match", "greedy", "random", "--games", "2", "--size", "8"]) == 0
    out = capsys.readouterr().out
    assert "Match: greedy vs random, 2 games" in out
    assert "Agent" in out and "Wins" in out
    assert out.count("\n") >= 6  # header, two games, a blank line, the table


def test_cli_match_is_reproducible(capsys):
    main(["match", "greedy", "random", "--games", "2"])
    first = capsys.readouterr().out
    main(["match", "greedy", "random", "--games", "2"])
    assert capsys.readouterr().out == first


def test_match_pairs_share_an_opening_and_pairs_differ(monkeypatch):
    # Deterministic agents repeat the same game for the same seats and start, so each game
    # must begin from its pair's opening: games 1 and 2 share one, games 3 and 4 another.
    import checkers.__main__ as cli

    starts = []
    real_play = cli.play

    def spy(players, level, read=input, board=None, **kwargs):
        starts.append(board)
        return real_play(players, level, read, board=board, **kwargs)

    monkeypatch.setattr(cli, "play", spy)
    cli.match("greedy", "random", 4, 1, out=lambda *a, **k: None)
    assert starts == [cli.opening_position(8, 0), cli.opening_position(8, 0),
                      cli.opening_position(8, 1), cli.opening_position(8, 1)]
    assert cli.opening_position(8, 0) != cli.opening_position(8, 1)
    assert cli.opening_position(8, 0) != start_position(8)


def test_cli_rejects_a_zero_game_match():
    with pytest.raises(SystemExit):
        main(["match", "greedy", "random", "--games", "0"])


def test_render_numbers_every_square_of_a_12x12_board():
    from checkers.__main__ import render

    text = render((0,) * 72)  # empty, so the square numbers show instead of pieces
    assert " 72 " in text  # the last square is numbered
    assert text.count("\n") == 2 * 12  # a rule line above each of the 12 rows, plus one below the last


# ── Chance agent and the matchup line ────────────────────────────────────

def test_chance_weight_by_move_type():
    from checkers.chance import CHANCE_WEIGHTS, move_kind

    # Red can capture white at (5,2) from (6,1), can step (6,5) into the centre (5,4), and has (6,5)->(5,6)
    # and (6,1)->(5,0), which are other man moves.
    b = board_with({(6, 1): 1, (5, 2): -1, (6, 5): 1, (0, 7): -1})
    kinds = {m.path: move_kind(b, m) for m in legal_moves(b, RED, forced=False)}
    assert sorted(kinds.values()) == ["capture", "centre_man", "man", "man"]
    assert kinds[(index(6, 5), index(5, 4))] == "centre_man"
    assert CHANCE_WEIGHTS == {"capture": 5, "centre_man": 3, "man": 2, "king": 1}


def test_chance_samples_in_proportion_to_its_weights():
    # Optional captures: the weights are capture 5, centre man 3, two other man moves 2 each, total 12.
    # Over many seeds the capture should be chosen about 5/12 of the time.
    from checkers.chance import Chance

    b = board_with({(6, 1): 1, (5, 2): -1, (6, 5): 1, (0, 7): -1})
    trials = 3000
    captures = sum(Chance(seed=s, forced=False).search(b, RED).move.captured != () for s in range(trials))
    assert abs(captures / trials - 5 / 12) < 0.03


def test_chance_is_seeded():
    from checkers.chance import Chance

    b, side = midgame(10, 10, seed=3)
    assert Chance(seed=9).search(b, side).move == Chance(seed=9).search(b, side).move
    assert choose(b, side, "chance", 1)["move"] == choose(b, side, "chance", 1)["move"]


def test_chance_is_in_the_registry_and_described():
    from checkers.agents import make_agent

    assert "chance" in AGENTS and "chance" in DESCRIPTIONS
    assert "5×" in DESCRIPTIONS["chance"] and "2×" in DESCRIPTIONS["chance"]
    assert make_agent("chance", 1).search(start_position(8), RED).move in legal_moves(start_position(8), RED)


def test_agent_labels_name_the_algorithm_and_budget():
    from checkers.agents import agent_label, matchup_text

    assert agent_label("mcts", 2, 8) == "MCTS (1,000 playouts)"
    assert agent_label("alphabeta", 2) == "alpha-beta search, iterative deepening, 0.8 s"
    assert agent_label("alphabeta", 1, by_nodes=True) == "alpha-beta search, iterative deepening, 3,000 nodes"
    assert agent_label("minimax", 2, 8) == "minimax, depth 4"
    assert agent_label("chance", 1) == "chance (fixed odds)"
    assert matchup_text({RED: "mcts", WHITE: "alphabeta"}, {RED: 2, WHITE: 2}) == (
        "Red: MCTS (1,000 playouts) vs White: alpha-beta search, iterative deepening, 0.8 s")
    assert matchup_text({RED: "human", WHITE: "minimax"}, {RED: 2, WHITE: 2}) == "White: minimax, depth 4"


def test_cli_prints_the_matchup_line_at_the_start_of_a_game(capsys):
    from checkers.__main__ import play

    play({RED: "human", WHITE: "minimax"}, 1, read=lambda prompt="": "q")
    assert "White: minimax, depth 2" in capsys.readouterr().out


def test_match_runs_alphabeta_against_chance(capsys):
    assert main(["match", "alphabeta", "chance", "--games", "2", "--level", "1"]) == 0
    out = capsys.readouterr().out
    assert "Red: alpha-beta search, iterative deepening, 3,000 nodes vs White: chance (fixed odds)" in out
    assert "chance" in out.split("Agent")[1]
