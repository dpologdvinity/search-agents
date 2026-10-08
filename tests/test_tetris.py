import io
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from tetris import __main__ as cli
from tetris.benchmark import run_benchmark
from tetris.board import FULL, H, W, drop_row, empty_board, lock, render
from tetris.evolve import GAConfig, evolve
from tetris.features import FEATURES, HAND_WEIGHTS, ROW_TRANSITIONS, board_features
from tetris.game import PieceBag, greedy_policy, lookahead_policy, play, random_policy
from tetris.pieces import PIECES, ROTATIONS
from tetris.search import best_move, legal_moves
from tetris.terminal import Game, run
from tetris.tuned import load_tuned

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "web" / "js" / "tetris_engine.js"


def rows_from(*strings):
    """Board from picture rows, top first: '#' filled, '.' empty. Padded to H rows with empty rows."""
    rows = [0] * H
    for i, s in enumerate(reversed(strings)):
        rows[i] = sum(1 << c for c, ch in enumerate(s) if ch == "#")
    return tuple(rows)


# ── pieces ────────────────────────────────────────────────────────────────

def test_seven_pieces_with_the_expected_rotation_counts():
    assert PIECES == "IOTSZJL"
    counts = {p: len(ROTATIONS[p]) for p in PIECES}
    assert counts == {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4}


def test_every_rotation_has_four_cells_and_matching_masks():
    for p in PIECES:
        for rot in ROTATIONS[p]:
            assert len(rot.cells) == 4
            assert sum(m.bit_count() for m in rot.masks) == 4
            assert len(rot.masks) == rot.height and max(m.bit_length() for m in rot.masks) == rot.width


def test_spawn_shapes_fit_in_the_top_two_rows():
    assert all(ROTATIONS[p][0].height <= 2 for p in PIECES)


# ── board ─────────────────────────────────────────────────────────────────

def test_drop_puts_an_o_on_the_floor_and_on_top_of_a_stack():
    board = empty_board()
    o = ROTATIONS["O"][0]
    assert drop_row(board, o, 0) == 0
    board = rows_from("..........", "..........", "##........")
    assert drop_row(board, o, 0) == 1 and drop_row(board, o, 2) == 0


def test_a_blocked_spawn_position_is_not_a_placement():
    tall = tuple([FULL] * (H - 1) + [0])  # everything full up to row H-2: the O cannot spawn at the top
    assert drop_row(tall, ROTATIONS["O"][0], 0) is None


def test_a_full_row_clears_and_erodes_the_piece_cells():
    # Bottom row full except columns 0-1. An O dropped into columns 0-1 completes that row.
    rows = [0] * H
    rows[0] = FULL & ~0b11
    o = ROTATIONS["O"][0]
    py = drop_row(tuple(rows), o, 0)
    assert py == 0
    new, lines, eroded, topped = lock(tuple(rows), o, 0, py)
    assert lines == 1 and not topped
    assert eroded == 2  # 1 line times this O's 2 cells in the cleared row
    assert new[0] == 0b11  # the O's upper row drops into the bottom row
    assert new[1] == 0


def test_render_shows_the_board_top_first():
    text = render(rows_from("#.........", "..........")).splitlines()
    assert text[-2] == "|..........|" and text[-3] == "|#.........|" and len(text) == H + 1


# ── features ──────────────────────────────────────────────────────────────

def test_row_transition_table_counts_walls_as_filled():
    assert ROW_TRANSITIONS[0] == 2  # an empty row: the left wall and the right wall are the only changes
    assert ROW_TRANSITIONS[FULL] == 0
    assert ROW_TRANSITIONS[0b1] == 2  # one block at the left edge: a change to its right, and to the right wall
    assert len(ROW_TRANSITIONS) == FULL + 1


def test_features_of_a_single_block_in_the_corner():
    board = rows_from("#.........")  # column 0 filled, one row high
    row_t, col_t, holes, wells, agg, bump = board_features(board)
    assert (row_t, col_t, holes, wells, agg, bump) == (2, 10, 0, 0, 1, 1)


def test_holes_and_wells():
    hole = rows_from("#.........", ".#........")  # column 0: the bottom cell is empty under a block
    assert board_features(hole)[2] == 1
    well = rows_from("#.#.......")  # column 1 is an empty cell with blocks on both sides
    assert board_features(well)[3] == 1
    deep = rows_from("#.#.......", "#.#.......")  # two stacked well cells: 1 + 2
    assert board_features(deep)[3] == 3


def test_feature_names_match_the_weights():
    assert len(FEATURES) == len(HAND_WEIGHTS) == 9


# ── search and game ───────────────────────────────────────────────────────

def test_the_search_finds_the_line_clearing_placement():
    rows = [0] * H
    rows[0] = FULL & ~0b11  # the bottom row has a two-wide gap at columns 0-1, which an O fills exactly
    move, moves = best_move(tuple(rows), "O", HAND_WEIGHTS)
    assert move.lines == 1 and move.x == 0 and move.y == 0
    assert move.score == max(m.score for m in moves)  # the choice is the best candidate, by construction


def test_every_legal_placement_is_reachable_on_an_empty_board():
    moves = legal_moves(empty_board(), "T", HAND_WEIGHTS)
    assert len(moves) == 34  # T has two 3-wide and two 2-wide states: 8 + 9 + 8 + 9 columns to choose from
    assert {(m.rot, m.x) for m in moves} == {(ri, px) for ri, r in enumerate(ROTATIONS["T"])
                                             for px in range(W - r.width + 1)}


def test_lookahead_with_one_candidate_is_the_greedy_choice():
    rng = random.Random(3)
    board = empty_board()
    for _ in range(40):
        p = rng.choice(PIECES)
        greedy, _ = best_move(board, p, HAND_WEIGHTS)
        if greedy is None:
            break
        look, _ = best_move(board, p, HAND_WEIGHTS, preview=rng.choice(PIECES), top_k=1)
        assert (look.rot, look.x, look.y) == (greedy.rot, greedy.x, greedy.y)
        board = greedy.board


def test_a_seeded_game_is_deterministic_and_the_bag_is_a_permutation():
    bag = PieceBag(5)
    first_seven = [bag.take() for _ in range(7)]
    assert sorted(first_seven) == sorted(PIECES)
    a = play(greedy_policy(HAND_WEIGHTS), 9, max_pieces=150)
    b = play(greedy_policy(HAND_WEIGHTS), 9, max_pieces=150)
    assert a == b
    assert a.capped and a.lines > 20


def test_random_placement_tops_out_quickly():
    r = play(random_policy(1), 1, max_pieces=2000)
    assert r.topped_out and r.pieces < 200 and r.lines < 20


def test_lookahead_game_runs_to_the_cap():
    r = play(lookahead_policy(HAND_WEIGHTS, top_k=4), 2, max_pieces=60)
    assert r.pieces == 60 and r.capped


def test_benchmark_smoke():
    result = run_benchmark(games=2, cap=40, workers=1, strategies=("random", "hand"), ga_weights=HAND_WEIGHTS)
    names = [s["strategy"] for s in result["summaries"]]
    assert names == ["random", "hand"] and all(s["games"] == 2 for s in result["summaries"])


# ── GA and committed weights ──────────────────────────────────────────────

def test_ga_is_reproducible_and_returns_nine_weights(tmp_path):
    cfg = GAConfig(population=6, generations=2, train_games=1, train_pieces=40, seed=4)
    a = evolve(cfg, workers=1, log_path=tmp_path / "log.jsonl", out=lambda *_: None)
    b = evolve(cfg, workers=1, out=lambda *_: None)
    assert a["weights"] == b["weights"] and len(a["weights"]) == 9
    assert [h["gen"] for h in a["history"]] == [0, 1, 2]
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 3


def test_committed_tuned_weights_match_the_feature_set():
    doc = load_tuned()
    assert doc["features"] == list(FEATURES)
    assert len(doc["weights"]) == 9
    assert len(doc["history"]) == doc["config"]["generations"] + 1
    assert doc["config"]["board_height"] == 10  # tuned on the narrow, harder board


def test_games_run_on_a_shorter_board():
    assert len(empty_board(10)) == 10
    r = play(greedy_policy(HAND_WEIGHTS), 3, max_pieces=5000, height=10)
    assert r.topped_out or r.capped
    assert r.lines > 0


def test_ga_generations_use_fresh_seeds():
    from tetris.evolve import generation_seeds
    g0, g1 = generation_seeds(0, 6), generation_seeds(1, 6)
    assert len(set(g0) | set(g1)) == 12  # no game is shared between generations


# ── terminal game ─────────────────────────────────────────────────────────

def test_terminal_hint_and_hard_drop():
    out = io.StringIO()
    code = run(seed=3, weights=HAND_WEIGHTS, stdin=io.StringIO("h\n \n q\n"), stdout=out)
    text = out.getvalue()
    assert code == 0
    assert "agent:" in text and "pieces 2" in text and "quit after" in text


def test_terminal_movement_is_bounded():
    g = Game(seed=1, weights=HAND_WEIGHTS)
    for _ in range(20):
        g.move(-1)
    assert g.x == 0
    for _ in range(20):
        g.move(1)
    assert g.x + ROTATIONS[g.piece][g.rot].width == W


def test_cli_watch_runs_and_reports_the_cap(capsys):
    assert cli.main(["watch", "--pieces", "15", "--seed", "2"]) == 0
    assert "capped at 15 pieces" in capsys.readouterr().out


def test_cli_rejects_unknown_commands():
    with pytest.raises(SystemExit):
        cli.main(["nonsense"])


# ── JavaScript engine parity ──────────────────────────────────────────────

def _python_cases():
    """Boards from real games (several stages each) with the tuned weights and a few random ones."""
    rng = random.Random(11)
    weights = [load_tuned()["weights"], HAND_WEIGHTS, [rng.uniform(-3, 3) for _ in FEATURES]]
    cases = []
    for seed in (21, 22):
        bag = PieceBag(seed)
        board = empty_board()
        policy = greedy_policy(HAND_WEIGHTS)
        for n in range(40):
            piece = bag.take()
            if n % 13 == 0:
                for w in weights:
                    cases.append({"board": list(board), "piece": piece, "weights": list(w)})
            move = policy(board, piece, bag.peek())
            if move is None:
                break
            board = move.board
    return cases


def test_js_engine_matches_python_placements_and_features(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    (tmp_path / "tetris_engine.mjs").write_text(ENGINE.read_text())
    runner = tmp_path / "runner.mjs"
    runner.write_text(
        "import { readFileSync } from 'node:fs';\n"
        "import { legalMoves } from './tetris_engine.mjs';\n"
        "const cases = JSON.parse(readFileSync(0, 'utf8'));\n"
        "const out = cases.map((c) => legalMoves(c.board, c.piece, c.weights).map((m) => ({\n"
        "  rot: m.rot, x: m.x, y: m.y, lines: m.lines, features: m.features, score: m.score })));\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    cases = _python_cases()
    proc = subprocess.run([node, str(runner)], input=json.dumps(cases), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)
    assert len(js) == len(cases)
    for case, js_moves in zip(cases, js):
        py_moves = legal_moves(tuple(case["board"]), case["piece"], case["weights"])
        py_keys = [(m.rot, m.x, m.y, m.lines) for m in py_moves]
        assert py_keys == [(j["rot"], j["x"], j["y"], j["lines"]) for j in js_moves]
        for m, j in zip(py_moves, js_moves):
            assert j["features"] == pytest.approx(list(m.features), abs=1e-9)
            assert j["score"] == pytest.approx(m.score, abs=1e-9)
