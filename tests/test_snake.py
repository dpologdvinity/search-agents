import json

import numpy as np
import pytest

from snake import agents, benchmark, evolve
from snake.__main__ import main as snake_main
from snake.board import DIRS, LEFT, RIGHT, STRAIGHT, Game, action_toward, turn
from snake.net import CHAMPION_PATH, FLOOD_CAP, INPUT_NAMES, N_IN, N_PARAMS, Net, features, load_champion, room


def game_at(body, heading, food, size=12):
    """A game at a chosen position, for rule checks. Body is head first."""
    return Game.from_state(size, body, heading, food)


def test_start_position_and_food():
    g = Game(seed=3)
    assert g.body == [(6, 6), (5, 6), (4, 6)] and g.heading == 1  # three segments, heading east
    assert g.food not in g.body and g.food is not None
    assert g.alive and g.cause is None and g.apples == 0


def test_turns_map_to_headings():
    assert turn(1, LEFT) == 0 and turn(1, RIGHT) == 2 and turn(0, LEFT) == 3
    # Pointing at a direction from a heading: straight, right, left, and no action for a reversal.
    assert action_toward(1, 1) == STRAIGHT and action_toward(1, 2) == RIGHT
    assert action_toward(1, 0) == LEFT and action_toward(1, 3) is None


def test_moves_and_turns():
    g = game_at([(6, 6), (5, 6), (4, 6)], 1, (0, 0))
    assert g.step(LEFT) and g.heading == 0 and g.body[0] == (6, 5)  # turn north
    assert g.step(RIGHT) and g.heading == 1 and g.body[0] == (7, 5)  # back east
    assert g.length == 3 and g.steps == 2


def test_wall_ends_the_game():
    g = game_at([(11, 6), (10, 6), (9, 6)], 1, (0, 0))
    assert g.step(STRAIGHT) is False
    assert g.cause == "wall" and not g.alive
    assert g.step(STRAIGHT) is False and g.steps == 1  # a finished game ignores further steps


def test_body_collision_ends_the_game():
    # Head at (3,3) heading north. Turning right heads east into (4,3), a body segment that is not the tail.
    g = game_at([(3, 3), (3, 4), (4, 4), (4, 3), (4, 2)], 0, (0, 0))
    g.step(RIGHT)
    assert g.cause == "body"


def test_moving_into_the_leaving_tail_is_legal():
    # A 2x2 ring: the head moves into the cell the tail is leaving, which is allowed.
    g = game_at([(3, 3), (3, 4), (4, 4), (4, 3)], 0, (0, 0))  # head at (3,3) heading north
    g.step(RIGHT)  # heading east: (4,3) is the tail, which moves away this step
    assert g.alive and g.body[0] == (4, 3) and g.length == 4


def test_eating_grows_and_places_new_food():
    g = game_at([(3, 3), (2, 3), (1, 3)], 1, (4, 3))
    g.step(STRAIGHT)
    assert g.apples == 1 and g.length == 4 and g.food not in g.body and g.idle == 0


def test_starvation_cap_is_two_laps():
    g = game_at([(6, 6), (5, 6), (4, 6)], 1, (0, 0))  # the food is far away and never eaten
    assert g.idle_cap == 2 * 12 * 12
    g.idle = g.idle_cap - 1
    g.step(STRAIGHT)
    assert g.cause == "starved" and not g.alive


def test_full_board_is_a_win():
    # A 3x3 board with 8 segments in a ring and the food at the centre: eating fills the board.
    ring = [(0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
    g = game_at(ring, 1, (1, 1), size=3)
    g.step(STRAIGHT)
    assert g.apples == 1 and g.cause == "full" and g.food is None


def test_features_shape_and_a_known_position():
    g = Game(seed=0)
    x = features(g)
    assert x.shape == (N_IN,) and len(INPUT_NAMES) == N_IN
    # Start: heading east at (6,6). Ahead has 5 free cells; left (north) has 6; right (south) has 5.
    assert x[0] == 0 and x[1] == 0 and x[2] == 0  # nothing adjacent is blocked
    assert x[3] == pytest.approx(5 / 11) and x[4] == pytest.approx(6 / 11) and x[5] == pytest.approx(5 / 11)
    assert list(x[10:14]) == [0, 1, 0, 0]  # heading east
    # Food to the north-east, a tail behind: offsets are in the snake's frame, clipped to [-1, 1].
    g2 = game_at([(6, 6), (5, 6), (4, 6)], 1, (9, 4))
    x2 = features(g2)
    assert x2[6] == pytest.approx(3 / 11) and x2[7] == pytest.approx(-2 / 11)  # 3 ahead, 2 to the left
    assert x2[8] == pytest.approx(-2 / 11) and x2[9] == pytest.approx(0)  # tail 2 behind


def test_danger_inputs_see_walls_and_body():
    # Heading north along the top edge: ahead is a wall.
    g = game_at([(5, 0), (5, 1), (5, 2)], 0, (0, 11))
    x = features(g)
    assert x[0] == 1.0 and x[3] == 0.0
    # Body directly to the right of the head: danger right.
    g2 = game_at([(5, 5), (4, 5), (4, 6), (5, 6), (6, 6), (6, 5)], 0, (0, 0))  # a connected loop
    assert features(g2)[2] == 1.0


def test_genome_round_trip_and_param_count():
    assert N_PARAMS == 17 * 16 + 16 + 16 * 3 + 3 == 339
    rng = np.random.default_rng(0)
    genome = rng.normal(size=N_PARAMS)
    net = Net.from_genome(genome)
    assert np.allclose(net.genome(), genome)
    with pytest.raises(ValueError):
        Net.from_genome(np.zeros(N_PARAMS - 1))


def test_champion_loads_with_the_right_shapes():
    assert CHAMPION_PATH.exists()
    net, meta = load_champion()
    assert net.w1.shape == (17, 16) and net.w2.shape == (16, 3)
    assert meta["layers"] == [17, 16, 3] and meta["best_generation"] >= 1
    assert len(meta["history"]) == meta["settings"]["generations"]


def test_planner_survives_and_eats_on_seeded_games():
    for seed in (1, 2):
        g = Game(seed)
        g.play(agents.planner, max_steps=400)
        assert g.apples >= 3


def test_planner_route_is_a_path_to_food():
    g = Game(4)
    path = agents.planner_path(g)
    assert path and path[-1] == g.food
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip([g.body[0], *path], path))


def test_greedy_heads_for_food_when_it_can():
    g = game_at([(6, 6), (5, 6), (4, 6)], 1, (6, 2))  # food straight north of the head
    assert agents.greedy(g) == LEFT


def test_evolution_is_reproducible_and_logs_every_generation():
    s = evolve.Settings(pop=8, games=2, generations=3, elites=1, seed=5, workers=1)
    g1, sc1, h1, gen1 = evolve.evolve(s)
    g2, sc2, h2, gen2 = evolve.evolve(s)
    assert np.array_equal(g1, g2) and sc1 == sc2 and gen1 == gen2
    assert [r["generation"] for r in h1] == [1, 2, 3]
    assert {"best_fitness", "mean_fitness", "best_apples", "mean_apples", "best_steps"} <= set(h1[0])


def test_worker_count_does_not_change_the_result():
    base = dict(pop=6, games=2, generations=2, elites=1, seed=9)
    g1, sc1, h1, _ = evolve.evolve(evolve.Settings(workers=1, **base))
    g2, sc2, h2, _ = evolve.evolve(evolve.Settings(workers=2, **base))
    strip = lambda h: [{k: v for k, v in r.items() if k != "elapsed_s"} for r in h]  # wall-clock time differs
    assert np.array_equal(g1, g2) and sc1 == sc2 and strip(h1) == strip(h2)


def test_room_counts_the_reachable_pocket():
    # A wall down column 1 splits a 3x3 board: from (0, 0) the reachable pocket is column 0 only.
    wall = {(1, 0), (1, 1), (1, 2)}
    assert room(wall, 3, (0, 0)) == 3
    assert room(wall, 3, (1, 1)) == 0  # blocked start
    assert room(set(), 12, (0, 0)) == FLOOD_CAP


def test_benchmark_summary_and_table():
    result = benchmark.run_benchmark(games=2, agents=("random", "greedy"))
    assert set(result["agents"]) == {"random", "greedy"}
    s = result["agents"]["random"]
    assert s["games"] == 2 and 0 <= s["death_rate"] <= 1
    table = benchmark.to_markdown(result)
    assert "| agent | mean apples" in table and "| greedy |" in table


def test_cli_watch_and_turn_play(capsys):
    assert snake_main(["watch", "--agent", "greedy", "--seed", "2", "--delay", "0", "--steps", "20"]) == 0
    assert "apples" in capsys.readouterr().out


def test_turn_play_takes_scripted_keys():
    from snake.__main__ import play_turns

    lines = iter(["", "w", "x", "h", "q"])
    out = []
    apples = play_turns(seed=1, read=lambda _: next(lines), out=out.append)
    text = "\n".join(out)
    assert apples >= 0
    assert "is not a key" in text and "hint:" in text


def test_champion_json_is_small_and_plain():
    data = json.loads(CHAMPION_PATH.read_text())
    assert len(json.dumps(data)) < 60_000
    assert np.array(data["w1"]).shape == (17, 16)


def test_right_hand_is_the_next_heading():
    # Each heading's right-hand unit vector is the next heading clockwise (north's right is east).
    for h in range(4):
        x, y = DIRS[h]
        assert DIRS[(h + 1) % 4] == (-y, x)
        assert turn(h, RIGHT) == (h + 1) % 4
    # Training and benchmark boards never share a seed.
    assert evolve.TRAIN_HI <= benchmark.BENCH_SEED_BASE
