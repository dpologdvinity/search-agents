import itertools
import random

import pytest

from sokoban.__main__ import main
from sokoban.analysis import describe
from sokoban.benchmark import CONFIGS, run_benchmark, summarize
from sokoban.board import apply_move, is_solved, parse_level, reachable, render, walk_path
from sokoban.deadlock import frozen_boxes, is_deadlock
from sokoban.heuristics import h_matching, h_simple
from sokoban.levels import LEVEL_TEXT, OPTIMAL_PUSHES, all_levels, get_level
from sokoban.matching import min_cost_matching
from sokoban.play import Game, play
from sokoban.search import ASTAR, BFS, GREEDY, solve
from sokoban.tables import tables

# A small corridor: the box can be pushed right to the goal, one push.
CORRIDOR = """
#####
#@$.#
#####
"""

# A box in a corner with the goal elsewhere: no push can ever move it, so the level is unsolvable.
CORNER_TRAP = """
#####
#$ .#
#@  #
#####
"""


def replay(level, moves):
    """Apply a move string to the level's start. Returns the final (player, boxes) or raises on a bad move."""
    player, boxes = level.start_player, set(level.start_boxes)
    for ch in moves:
        out = apply_move(level, player, boxes, ch)
        assert out is not None, f"illegal move {ch!r}"
        player, boxes, _ = out
    return player, boxes


def test_parse_reads_symbols_and_pads_with_walls():
    lv = parse_level(CORRIDOR, "corridor")
    assert len(lv.goals) == 1 and len(lv.start_boxes) == 1
    assert lv.floor[lv.start_player] == 1
    # The padding ring is wall: floor outside the text is not floor.
    assert lv.floor[0] == 0 and lv.floor[lv.width * lv.height - 1] == 0


@pytest.mark.parametrize("text, message", [
    ("#####\n#$ .#\n#####", "exactly one player"),
    ("#####\n#@$$#\n#####", "one goal per box"),
    ("#####\n#@ .#\n#####", "at least one box"),
    ("#####\n#@x.#\n#####", "unexpected character"),
])
def test_parse_rejects_broken_levels(text, message):
    with pytest.raises(ValueError, match=message):
        parse_level(text)


def test_render_round_trips_every_built_in_level():
    for lv, (_, _, text) in zip(all_levels(), LEVEL_TEXT):
        expected = [line.rstrip() for line in text.strip("\n").split("\n")]
        assert render(lv, lv.start_player, lv.start_boxes) == expected


def test_walking_stays_out_of_boxes_and_walls():
    lv = parse_level(CORRIDOR)
    path = walk_path(lv, lv.start_player, lv.start_player, set(lv.start_boxes))
    assert path == ""
    # The box blocks the way right, so the goal cell is unreachable by walking.
    assert walk_path(lv, lv.start_player, lv.goals[0], set(lv.start_boxes)) is None


def test_push_moves_the_box_and_is_refused_by_walls_and_boxes():
    lv = parse_level(CORRIDOR)
    player, boxes, pushed = apply_move(lv, lv.start_player, set(lv.start_boxes), "d")
    assert pushed and boxes == {lv.goals[0]}
    # Walking into the wall is refused.
    assert apply_move(lv, lv.start_player, set(lv.start_boxes), "w") is None
    # A box behind a box cannot be pushed.
    two = parse_level("#######\n#@$$..#\n#######\n")
    assert apply_move(two, two.start_player, set(two.start_boxes), "d") is None


def test_dead_squares_are_corners_and_dead_ends_but_not_goals():
    lv = parse_level(LEVEL_TEXT[5][2].strip("\n"))  # Crossroads
    t = tables(lv)
    for g in lv.goals:
        assert t.dead[g] == 0
    # Floor cells in the room corners cannot take a box to a goal.
    corner = lv.at(2, 2)  # the top-left interior cell (padded coordinates add one ring of walls)
    assert t.dead[corner] == 1


def test_push_distance_along_a_row():
    # Goal at column 1; a box at column 4 needs three pushes, at column 3 two.
    lv = parse_level("#######\n#.  $ #\n#  @  #\n#######\n")
    t = tables(lv)
    assert t.nearest[lv.at(2, 5)] == 3  # the box's cell (padded coordinates: text row 1, column 4)
    assert t.nearest[lv.at(2, 4)] == 2
    # Walls carry -1 in the display table.
    assert t.nearest[0] == -1


def test_frozen_box_in_a_two_by_two_block_is_a_deadlock():
    # Four boxes in a 2x2 block with the goals on the outside: none can ever move, so it is a deadlock.
    lv = parse_level("#######\n#@  ..#\n#  $$ #\n#  $$ #\n#.   .#\n#######\n")
    occ = {lv.at(3, 4), lv.at(3, 5), lv.at(4, 4), lv.at(4, 5)}
    assert frozen_boxes(lv, occ) == occ
    assert is_deadlock(lv, tables(lv).dead, occ)


def test_a_box_against_a_wall_with_free_sides_is_not_frozen():
    lv = parse_level(CORRIDOR)
    assert frozen_boxes(lv, set(lv.start_boxes)) == set()


def test_matching_is_exact_against_brute_force():
    rng = random.Random(7)
    for n in range(1, 6):
        for _ in range(40):
            cost = [[rng.randint(0, 9) for _ in range(n)] for _ in range(n)]
            best = min(sum(cost[r][p[r]] for r in range(n)) for p in itertools.permutations(range(n)))
            total, match = min_cost_matching(cost)
            assert total == best
            assert sorted(match) == list(range(n))
            assert sum(cost[r][match[r]] for r in range(n)) == best


def test_built_in_levels_solve_with_the_optimal_push_count():
    for n, (name, pushes, _) in enumerate(LEVEL_TEXT, 1):
        lv = get_level(n)
        r = solve(lv, ASTAR, "matching", True, node_budget=20_000, time_limit=20)
        assert r.solved, name
        assert r.pushes == pushes == OPTIMAL_PUSHES[name], name
        player, boxes = replay(lv, r.moves)
        assert is_solved(lv, boxes), name


def test_pruning_does_not_change_the_optimal_push_count():
    for lv in all_levels()[:8]:
        on = solve(lv, ASTAR, "matching", True, node_budget=20_000, time_limit=20)
        off = solve(lv, ASTAR, "matching", False, node_budget=20_000, time_limit=20)
        assert on.solved and off.solved
        assert on.pushes == off.pushes


def test_bfs_and_simple_astar_agree_with_matching_on_small_levels():
    for lv in all_levels()[:6]:
        pushes = {
            (algo, heur): solve(lv, algo, heur, True, node_budget=20_000, time_limit=20).pushes
            for algo, heur in ((BFS, "none"), (ASTAR, "simple"), (ASTAR, "matching"))
        }
        assert len(set(pushes.values())) == 1, pushes


def test_greedy_returns_a_valid_plan_that_may_be_longer():
    lv = get_level(11)  # Aisle: greedy takes a longer route than the optimal 18 pushes
    r = solve(lv, GREEDY, "matching", True, node_budget=20_000, time_limit=20)
    assert r.solved and r.pushes >= OPTIMAL_PUSHES[lv.name]
    assert is_solved(lv, replay(lv, r.moves)[1])


def test_heuristics_never_exceed_the_remaining_pushes_on_an_optimal_plan():
    for n in (1, 2, 4, 6, 7):
        lv = get_level(n)
        r = solve(lv, ASTAR, "matching", True, node_budget=20_000, time_limit=20)
        t = tables(lv)
        player, boxes = lv.start_player, set(lv.start_boxes)
        total = r.pushes
        done = 0
        assert h_matching(t, sorted(boxes)) <= total and h_simple(t, sorted(boxes)) <= total
        for ch in r.moves:
            out = apply_move(lv, player, boxes, ch)
            player, boxes, pushed = out
            if pushed:
                done += 1
                remaining = total - done
                assert h_simple(t, sorted(boxes)) <= remaining
                assert h_matching(t, sorted(boxes)) <= remaining


def test_exhaustive_search_confirms_deadlocks_never_reach_a_goal():
    """On tiny levels, enumerate every reachable position and check the deadlock test is sound."""
    for n in (1, 2, 3, 4):
        lv = get_level(n)
        t = tables(lv)
        start = (lv.start_player, frozenset(lv.start_boxes))
        seen = {start}
        frontier = [start]
        while frontier:
            new = []
            for player, boxes in frontier:
                region = reachable(lv, player, set(boxes))
                for p in region:
                    for off in lv.offsets:
                        b = p + off
                        if b not in boxes:
                            continue
                        nb = b + off
                        if not lv.floor[nb] or nb in boxes:
                            continue
                        new_boxes = frozenset((boxes - {b}) | {nb})
                        state = (b, new_boxes)
                        if state not in seen:
                            seen.add(state)
                            new.append(state)
            frontier = new
        for player, boxes in seen:
            if is_deadlock(lv, t.dead, set(boxes)):
                # A deadlocked position must have no solved position among those reachable from it.
                sub = {(player, boxes)}
                stack = [(player, boxes)]
                while stack:
                    pl, bx = stack.pop()
                    assert not is_solved(lv, bx), f"level {n}: deadlock flagged on a solvable position"
                    for p in reachable(lv, pl, set(bx)):
                        for off in lv.offsets:
                            b = p + off
                            if b not in bx or not lv.floor[b + off] or (b + off) in bx:
                                continue
                            nxt = (b, frozenset((bx - {b}) | {b + off}))
                            if nxt not in sub:
                                sub.add(nxt)
                                stack.append(nxt)


def test_unsolvable_level_is_reported_as_exhausted_or_deadlocked():
    lv = parse_level(CORNER_TRAP)
    assert solve(lv, ASTAR, "matching", True).status == "exhausted"
    assert solve(lv, ASTAR, "matching", False).status == "exhausted"


def test_node_budget_stops_the_search_and_reports_it():
    r = solve(get_level(12), BFS, "none", False, node_budget=50, time_limit=20)
    assert r.status == "budget" and not r.solved and r.moves is None
    assert r.expanded == 50


def test_time_limit_stops_the_search():
    r = solve(get_level(12), BFS, "none", False, node_budget=10**9, time_limit=0.05)
    assert r.status == "time"


def test_solving_from_mid_game_position():
    lv = get_level(1)
    mid = lv.with_state(lv.start_player, set(lv.goals))  # boxes already on goals
    r = solve(mid, ASTAR, "matching", True)
    assert r.solved and r.moves == ""


def test_describe_flags_dead_squares_and_frozen_boxes():
    lv = parse_level(CORNER_TRAP)
    pos = describe(lv, sorted(lv.start_boxes))
    assert pos.deadlock == "dead square"
    ok = describe(get_level(1), sorted(get_level(1).start_boxes))
    assert ok.deadlock is None and ok.h_matching >= 1


def test_game_undo_and_restart_and_blocked_moves():
    lv = get_level(1)
    game = Game.start(lv)
    assert game.move("w") != ""  # blocked by a wall
    assert game.moves == ""
    assert game.move("d") == "" and game.pushes == 1 and game.solved
    assert game.undo() and not game.solved and game.moves == ""
    game.restart()
    assert game.player == lv.start_player and game.moves == ""


def test_terminal_play_solves_warm_up_from_scripted_input():
    script = iter(["d"])
    lines = []
    solved = play(1, read=lambda prompt="": next(script), out=lines.append, delay=0)
    assert solved == 1
    assert any("Solved 'Warm-up' in 1 pushes" in line for line in lines)


def test_terminal_play_hint_and_solve_do_not_change_the_board_unless_asked():
    script = iter(["h", "x"])
    lines = []
    play(1, read=lambda prompt="": next(script), out=lines.append, delay=0)
    assert any(line.startswith("hint: push D") or "push D" in line for line in lines)
    assert any("solution: D" in line for line in lines)


def test_cli_levels_and_solve_exit_codes(capsys):
    assert main(["levels"]) == 0
    out = capsys.readouterr().out
    assert "Warm-up" in out and "Dock" in out
    assert main(["solve", "--level", "3", "--algorithm", "bfs", "--no-prune"]) == 0
    assert "pushes 3" in capsys.readouterr().out
    assert main(["solve", "--level", "12", "--algorithm", "bfs", "--no-prune", "--budget", "30"]) == 1


def test_benchmark_rows_cover_every_configuration():
    runs = run_benchmark(node_budget=3000, time_limit=5, levels=all_levels()[:2])
    rows = summarize(runs)
    assert len(rows) == 2 * len(CONFIGS)
    assert all(r["levels"] == 2 for r in rows)
    matching_on = next(r for r in rows if r["config"] == "A* (matching)" and r["prune"])
    assert matching_on["solved"] == 2 and matching_on["optimal"] == 2
