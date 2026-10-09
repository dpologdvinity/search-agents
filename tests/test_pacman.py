"""Pac-Man tests: layouts, A*, the turn rules, ghost AI, features, the Q-update math, and agents.

Most rule tests use tiny one-corridor layouts registered with the `tiny` helper, so each
expected score and collision can be worked out by hand from the layout.
"""

import json
import random

import pytest

from pacman import mazes
from pacman.__main__ import main as cli_main
from pacman.agents import QAgent, RandomAgent, load_weights, make_agent
from pacman.benchmark import BENCH_SEED_BASE
from pacman.engine import (
    DEATH_PENALTY,
    GHOST_POINTS,
    LOST,
    PELLET_POINTS,
    POWER_POINTS,
    SCARED_TURNS,
    STEP_COST,
    WIN_POINTS,
    WON,
    Game,
    initial_state,
    learning_reward,
    legal_actions,
)
from pacman.episode import run_episode
from pacman.features import FEATURE_NAMES, state_features, successor
from pacman.ghosts import CHANCE_ODDS, NO_HEADING, Ghost, chance_move, choose_move, target_for
from pacman.lookahead import SURVIVAL_HORIZON, survival_depths
from pacman.match import match_markdown, run_match
from pacman.mazes import ACTIONS, LAYOUTS, parse
from pacman.qlearning import td_update, train
from pacman.search import astar, distances
from pacman.terminal import play_human, render, watch

N, E, S, W = (ACTIONS.index(letter) for letter in "NESW")


def tiny(monkeypatch, name, rows):
    """Register a hand-made layout for one test and return it parsed."""
    monkeypatch.setitem(LAYOUTS, name, (name.title(), tuple(rows)))
    return parse(name)


# ── layouts ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", mazes.names())
def test_layouts_parse_with_valid_starts(name):
    m = mazes.get(name)
    assert m.width == 15 and m.height == 11
    assert m.pac_start not in m.pellets
    assert all(g not in m.pellets for g in m.ghost_starts)
    assert 2 <= len(m.ghost_starts) <= 4
    assert len(m.powers) == 4 and m.powers <= m.pellets


def test_parse_rejects_unreachable_cells(monkeypatch):
    monkeypatch.setitem(LAYOUTS, "cut", ("Cut", ("#####", "#P#G#", "#####")))
    with pytest.raises(ValueError, match="cannot be reached"):
        parse("cut")


def test_parse_rejects_ragged_rows_and_missing_markers(monkeypatch):
    monkeypatch.setitem(LAYOUTS, "ragged", ("R", ("###", "#P.G#", "###")))
    with pytest.raises(ValueError, match="same length"):
        parse("ragged")
    monkeypatch.setitem(LAYOUTS, "nopac", ("N", ("#####", "#..G#", "#####")))
    with pytest.raises(ValueError, match="exactly one P"):
        parse("nopac")


# ── A* and BFS ───────────────────────────────────────────────────────────

def test_astar_finds_the_shortest_route_on_a_small_maze(monkeypatch):
    m = tiny(monkeypatch, "route", [
        "#######",
        "#P....#",
        "#.###.#",
        "#...#G#",
        "#######",
    ])
    start, goal = m.pac_start, m.ghost_starts[0]
    path = astar(m, start, goal)
    assert path[0] == start and path[-1] == goal
    assert len(path) - 1 == 6  # along the top row, then down the right-hand column
    for a, b in zip(path, path[1:], strict=False):
        assert b in m.nbr[a]


def test_astar_length_matches_bfs_on_real_maze():
    m = mazes.get("lanes")
    rng = random.Random(0)
    open_cells = [c for c in range(m.cells) if m.is_open[c]]
    for _ in range(25):
        s, g = rng.choice(open_cells), rng.choice(open_cells)
        assert len(astar(m, s, g)) - 1 == distances(m, s)[g]


def test_astar_to_self_is_a_single_cell():
    m = mazes.get("vault")
    assert astar(m, m.pac_start, m.pac_start) == [m.pac_start]


# ── turn rules ───────────────────────────────────────────────────────────

def test_walls_are_not_legal_and_refused_by_step():
    g = Game("lanes", seed=0)
    legal = set(legal_actions(g.state))
    walls = [a for a in range(4) if a not in legal]
    assert walls  # the start cell is not surrounded on all four sides
    for a in walls:
        with pytest.raises(ValueError):
            g.step(a)


def test_pellet_scores_and_chaser_steps_toward_pac(monkeypatch):
    m = tiny(monkeypatch, "lane", ["#########", "#P.o.G..#", "#########"])
    g = Game(m, seed=0)
    turn = g.step(E)
    assert turn.points == PELLET_POINTS
    assert turn.events == ("pellet",)
    assert turn.eaten == m.pac_start + 1
    assert m.pac_start + 1 not in g.state.pellets
    # The ghost is a chaser, so its target is Pac-Man and its route runs west toward him.
    assert turn.plans[0].target == g.state.pac
    assert g.state.ghosts[0].pos == m.ghost_starts[0] - 1


def test_power_pellet_scares_ghosts_and_eating_a_scared_ghost_scores(monkeypatch):
    # Cells: 0 wall, 1 Pac-Man, 2 pellet, 3 power pellet, 4 pellet, 5 ghost, 6 pellet, 7 wall.
    m = tiny(monkeypatch, "power", ["########", "#P.o.G.#", "########"])
    g = Game(m, seed=0)
    g.step(E)  # eat the pellet at col 2; the ghost walks in to col 4
    turn = g.step(E)  # eat the power pellet at col 3
    assert "power" in turn.events and turn.points == POWER_POINTS
    assert g.state.ghosts[0].scared == SCARED_TURNS - 1  # the timer ticks down at the end of the turn
    g.step(E)  # pellet at col 4; the scared ghost flees east to col 5, then col 6
    g.step(E)  # col 5 is empty; the ghost is cornered at col 6 and steps back onto Pac-Man
    assert g.state.points == PELLET_POINTS + POWER_POINTS + PELLET_POINTS + GHOST_POINTS
    assert g.state.ghosts[0].pos == m.ghost_starts[0]  # the eaten ghost returns home
    assert g.state.ghosts[0].scared == 0


def test_normal_ghost_on_destination_kills(monkeypatch):
    m = tiny(monkeypatch, "death", ["#######", "#P..G##", "#######"])
    g = Game(m, seed=0)
    g.step(E)
    turn = g.step(E)  # the chaser steps onto Pac-Man's cell; the last pellet is eaten first
    assert g.state.status == LOST
    assert "death" in turn.events and "win" not in turn.events
    assert learning_reward(turn) == PELLET_POINTS - STEP_COST - DEATH_PENALTY


def test_eating_the_last_pellet_wins(monkeypatch):
    m = tiny(monkeypatch, "win", ["#####", "#P.G#", "#####"])
    g = Game(m, seed=0)
    turn = g.step(E)
    assert g.state.status == WON
    assert turn.points == PELLET_POINTS + WIN_POINTS
    assert turn.events == ("pellet", "win")


def test_learning_reward_charges_a_step_cost(monkeypatch):
    m = tiny(monkeypatch, "rew", ["######", "#P..G#", "######"])
    turn = Game(m, seed=0).step(E)
    assert turn.points == PELLET_POINTS
    assert learning_reward(turn) == PELLET_POINTS - STEP_COST


def test_game_times_out_after_max_turns():
    g = Game("lanes", seed=5, max_turns=3)
    rng = random.Random(5)
    while not g.state.over:
        g.step(rng.choice(legal_actions(g.state)))
    assert g.state.turn <= 3
    assert g.state.status in {"timeout", "lost", "won"}


def test_same_seed_and_moves_replay_exactly():
    def play(seed):
        g = Game("vault", seed=seed)
        rng = random.Random(1)
        while not g.state.over:
            g.step(rng.choice(legal_actions(g.state)))
        return g.state

    assert play(42) == play(42)


def test_states_are_immutable_snapshots(monkeypatch):
    m = tiny(monkeypatch, "snap", ["#####", "#P.G#", "#####"])
    g = Game(m, seed=0)
    before = g.state
    g.step(E)
    assert before.turn == 0 and before.points == 0 and before.pac == m.pac_start
    assert g.state.turn == 1


# ── ghost AI ─────────────────────────────────────────────────────────────

def test_ambusher_aims_four_steps_ahead_and_stops_at_walls(monkeypatch):
    m = tiny(monkeypatch, "ambush", ["##########", "#P......G#", "##########"])
    ghost = Ghost(pos=m.ghost_starts[0], personality="ambusher", home=m.ghost_starts[0])
    pac = m.pac_start + 1  # column 2
    assert m.col[target_for(m, ghost, 1, pac, E)] == 6  # 4 steps ahead of column 2
    near_wall = m.pac_start + 6  # column 7: only column 8 is open before the wall
    assert m.col[target_for(m, ghost, 1, near_wall, E)] == 8


def test_scatter_chases_from_afar_and_retreats_when_close():
    m = mazes.get("lanes")
    far = Ghost(pos=m.ghost_starts[0], personality="scatter", home=m.ghost_starts[0])
    assert target_for(m, far, 2, m.pac_start, -1) == m.pac_start  # Manhattan 7 > 4: chase
    close_cell = m.pac_start + 1
    near = Ghost(pos=close_cell, personality="scatter", home=close_cell)
    assert target_for(m, near, 2, m.pac_start, -1) == m.corners[2]  # within 4: go to corner 2


def test_scared_ghost_flees_to_the_farthest_neighbour():
    m = mazes.get("lanes")
    pac = m.pac_start
    ghost_cell = m.nbr[pac][E]
    pac_dist = distances(m, pac)
    ghost = Ghost(pos=ghost_cell, personality="chaser", home=ghost_cell, scared=5)
    nxt, target, route = choose_move(m, ghost, 0, pac, -1, random.Random(0), pac_dist)
    assert target is None and route == (ghost_cell, nxt)
    assert pac_dist[nxt] == max(pac_dist[nb] for nb in m.nbr[ghost_cell] if nb != -1)


# ── features ─────────────────────────────────────────────────────────────

def test_feature_vector_has_one_entry_per_name_and_sane_ranges():
    state = initial_state(mazes.get("lanes"))
    feats = state_features(state)
    assert set(feats) == set(legal_actions(state))
    for f in feats.values():
        assert len(f) == len(FEATURE_NAMES)
        assert f[0] == 1.0
        assert all(x >= 0 for x in f)
        assert 0.25 <= f[FEATURE_NAMES.index("openness")] <= 1.0
        assert 0.0 <= f[FEATURE_NAMES.index("survival")] <= 1.0


def test_caught_and_adjacent_features_see_the_ghost(monkeypatch):
    m = tiny(monkeypatch, "feat", ["#####", "#PG.#", "#####"])
    f = state_features(initial_state(m))[E]  # Pac-Man would step onto the ghost
    assert f[FEATURE_NAMES.index("caught")] == 1.0
    assert f[FEATURE_NAMES.index("survival")] == 0.0
    m2 = tiny(monkeypatch, "feat2", ["######", "#P.G##", "######"])
    f2 = state_features(initial_state(m2))[E]  # Pac-Man would stand one step from the ghost
    assert f2[FEATURE_NAMES.index("ghost_adjacent")] == 1.0
    assert f2[FEATURE_NAMES.index("caught")] == 0.0


def _brute_survival(game, state, plies):
    """Reference survival without the memo: try every line of play, return the best survived plies."""
    if plies == 0 or state.over:
        return plies
    best = 0
    for action in legal_actions(state):
        game.state = state
        turn = game.step(action)
        if "death" not in turn.events:
            best = max(best, 1 + _brute_survival(game, game.state, plies - 1))
    return best


def _random_midgame_state(turns):
    """A living position after `turns` random moves, from the first seed whose game is still going."""
    for seed in range(100):
        rng = random.Random(seed)
        game = Game(mazes.get("vault"), seed=seed)
        for _ in range(turns):
            game.step(rng.choice(game.legal_actions()))
        if not game.state.over:
            return game.state
    raise AssertionError("no random game survives that long")


@pytest.mark.parametrize("turns", [0, 2, 4])
def test_survival_search_matches_brute_force_on_real_positions(turns):
    state = _random_midgame_state(turns)
    reference = Game(state.maze, seed=0)
    expected = {}
    for a in legal_actions(state):
        reference.state = state
        turn = reference.step(a)
        expected[a] = 0 if "death" in turn.events else 1 + _brute_survival(reference, reference.state, 3)
    assert survival_depths(state, horizon=4) == expected


def test_survival_depths_cover_every_legal_move_within_the_horizon():
    state = initial_state(mazes.get("lanes"))
    depths = survival_depths(state)
    assert set(depths) == set(legal_actions(state))
    assert all(0 <= d <= SURVIVAL_HORIZON for d in depths.values())


def test_successor_reports_pellet_eating_and_food_distance():
    state = initial_state(mazes.get("lanes"))
    action = legal_actions(state)[0]
    nb = state.maze.nbr[state.pac][action]
    s = successor(state, action)
    assert s.eats_pellet == (nb in state.pellets and nb not in state.maze.powers)
    assert s.food is not None and s.food >= 0


# ── Q-learning math ──────────────────────────────────────────────────────

def test_td_update_terminal_case_moves_weight_toward_reward():
    w = [0.0, 0.0]
    error = td_update(w, [1.0, 0.0], reward=10.0, next_feats=[], alpha=0.1, gamma=0.9)
    assert error == pytest.approx(10.0)
    assert w == pytest.approx([1.0, 0.0])  # 0.1 * error 10 * feature 1


def test_td_update_bootstraps_from_the_best_next_action():
    w = [1.0, 0.0]
    # Q(s, a) = 1. The best next value is max(1, 0) = 1, so the target is 2 + 0.5 * 1 = 2.5.
    error = td_update(w, [1.0, 0.0], reward=2.0, next_feats=[[1.0, 0.0], [0.0, 1.0]], alpha=0.1, gamma=0.5)
    assert error == pytest.approx(1.5)
    assert w[0] == pytest.approx(1.0 + 0.1 * 1.5)
    assert w[1] == 0.0  # its feature was zero this turn, so the weight does not move


# ── agents and training ──────────────────────────────────────────────────

def test_trained_agent_beats_random_on_fixed_seeds():
    q = QAgent(load_weights().weights)
    seeds = range(20_000, 20_010)
    q_score = sum(run_episode(q, mazes.get("vault"), s, "q").score for s in seeds)
    r_score = sum(run_episode(RandomAgent(), mazes.get("vault"), s, "random").score for s in seeds)
    assert q_score > 5 * r_score


def test_training_is_deterministic_for_a_seed():
    a = train(6, seed=3, log_every=3)
    b = train(6, seed=3, log_every=3)
    assert a.weights == b.weights
    assert len(a.history) == 2


def test_make_agent_and_terminal_output():
    assert make_agent("random").name == "random"
    assert make_agent("reflex").name == "reflex"
    with pytest.raises(KeyError):
        make_agent("ghost")
    text = render(Game("lanes", seed=0).state)
    assert "P" in text and "#" in text
    assert text.splitlines()[-1].startswith("Neon Lanes")


def test_human_play_refuses_bad_keys_and_quits():
    inputs = iter(["x", "q"])
    out = []
    assert play_human("lanes", 0, read=lambda _: next(inputs), out=out.append) == "quit"
    assert any("use w" in line for line in out)


# ── chance ghosts ────────────────────────────────────────────────────────

# A 5x5 cross: the ghost stands at the centre with all four arms open, Pac-Man at the north arm.
CROSS = ("#####", "##P##", "##.##", "#.G.#", "##.##")
# The same cross with the south arm walled off, so a ghost heading north has no way back.
TEE = ("#####", "##P##", "##.##", "#.G.#", "#####")


def _draws(maze, ghost, n, seed=0):
    """Count how often each direction index comes up in n chance draws from the same ghost."""
    rng = random.Random(seed)
    counts = {}
    for _ in range(n):
        _, d = chance_move(maze, ghost, rng)
        counts[d] = counts.get(d, 0) + 1
    return {d: c / n for d, c in counts.items()}


def test_chance_odds_are_a_probability_table():
    assert set(CHANCE_ODDS) == {"straight", "left", "right", "back"}
    assert all(p > 0 for p in CHANCE_ODDS.values())
    assert sum(CHANCE_ODDS.values()) == pytest.approx(1.0)
    assert CHANCE_ODDS == {"straight": 0.60, "left": 0.15, "right": 0.15, "back": 0.10}


def test_chance_draws_match_the_table_when_all_four_ways_are_open(monkeypatch):
    m = tiny(monkeypatch, "cross", CROSS)
    centre = m.ghost_starts[0]
    freq = _draws(m, Ghost(centre, "chaser", centre, heading=N), 20_000)
    # Heading north: straight is N, right is E, left is W, back is S.
    expected = {N: 0.60, E: 0.15, W: 0.15, S: 0.10}
    assert set(freq) == set(expected)
    for d, p in expected.items():
        assert freq[d] == pytest.approx(p, abs=0.015)


def test_chance_renormalises_over_the_open_directions(monkeypatch):
    m = tiny(monkeypatch, "tee", TEE)
    centre = m.ghost_starts[0]
    freq = _draws(m, Ghost(centre, "chaser", centre, heading=N), 20_000)
    # South is a wall, so the 10% for going back is shared out: 0.6, 0.15, 0.15 over 0.9.
    assert set(freq) == {N, E, W}
    assert freq[N] == pytest.approx(0.6 / 0.9, abs=0.015)
    assert freq[E] == pytest.approx(0.15 / 0.9, abs=0.015)
    assert freq[W] == pytest.approx(0.15 / 0.9, abs=0.015)


def test_chance_turns_are_relative_to_the_heading(monkeypatch):
    m = tiny(monkeypatch, "cross", CROSS)
    centre = m.ghost_starts[0]
    # Heading east: straight is E, a right turn is S (clockwise), a left turn is N, back is W.
    freq = _draws(m, Ghost(centre, "chaser", centre, heading=E), 20_000)
    assert freq[E] == pytest.approx(0.60, abs=0.015)
    assert freq[S] == pytest.approx(0.15, abs=0.015)
    assert freq[N] == pytest.approx(0.15, abs=0.015)
    assert freq[W] == pytest.approx(0.10, abs=0.015)


def test_chance_without_a_heading_is_uniform_over_open_directions(monkeypatch):
    m = tiny(monkeypatch, "cross", CROSS)
    centre = m.ghost_starts[0]
    freq = _draws(m, Ghost(centre, "chaser", centre), 20_000)
    for d in (N, E, S, W):
        assert freq[d] == pytest.approx(0.25, abs=0.015)


def test_chance_only_plays_legal_moves_and_seeds_replay(monkeypatch):
    m = mazes.get("lanes")
    for seed in range(5):
        a = Game(m, seed=seed, ghosts="chance")
        b = Game(m, seed=seed, ghosts="chance")
        while not a.state.over:
            action = a.legal_actions()[0]
            before = [g.pos for g in a.state.ghosts]
            ta, tb = a.step(action), b.step(action)
            assert ta.after == tb.after  # same seed and moves, same game
            for old, g in zip(before, a.state.ghosts, strict=True):
                # Each ghost moves to a neighbour of its old cell, or home if it was eaten. A ghost stays
                # put on a turn where Pac-Man walks into it first, because that collision ends the turn.
                assert g.pos == old or g.pos in m.nbr[old] or g.pos == g.home
                assert m.is_open[g.pos]
    assert a.state.turn > 0


def test_chance_records_each_ghosts_heading_and_has_no_route(monkeypatch):
    m = tiny(monkeypatch, "cross", CROSS)
    g = Game(m, seed=3, ghosts="chance")
    turn = g.step(S)  # Pac-Man steps south from the north arm; the north side is a wall
    moved = 0
    for before, after in zip(turn.before.ghosts, turn.after.ghosts, strict=True):
        if after.pos != before.pos:
            moved += 1
            assert after.heading == m.nbr[before.pos].index(after.pos)
    assert moved >= 1
    assert all(p.target is None and p.path == () for p in turn.plans)


def test_chance_ghost_policy_is_recorded_and_unknown_policies_refused():
    assert Game("lanes", seed=0).ghost_policy == "ai"
    assert Game("lanes", seed=0, ghosts="chance").ghost_policy == "chance"
    with pytest.raises(ValueError, match="unknown ghost policy"):
        Game("lanes", seed=0, ghosts="nope")


def test_ai_ghosts_never_record_a_heading():
    # The A* ghosts ignore headings, and leaving them unset keeps their lookahead memo keys unchanged.
    g = Game("lanes", seed=2)
    while not g.state.over and g.legal_actions():
        g.step(g.legal_actions()[0])
    assert all(gh.heading == NO_HEADING for gh in g.state.ghosts)


def test_chance_episode_is_recorded_and_watchable(monkeypatch):
    lines = []
    status = watch("random", "lanes", 4, delay=0, out=lines.append, sleep=lambda _: None,
                   clear=False, ghosts="chance")
    assert status in ("won", "lost", "timeout")
    assert lines[-1].startswith("random vs Chance (fixed odds): ")
    ep = run_episode(make_agent("random"), mazes.get("lanes"), 4, "random", ghosts="chance")
    assert ep.ghosts == "chance" and ep.final.status == status


def test_chance_human_play_uses_the_chance_ghosts():
    inputs = iter(["d", "q"])
    out = []
    assert play_human("lanes", 0, read=lambda _: next(inputs), out=out.append, ghosts="chance") == "quit"


def test_run_match_reports_both_ghost_policies_on_the_same_seeds():
    result = run_match(2, "random", mazes=("lanes",))
    assert list(result["ghosts"]) == ["ai", "chance"]
    assert result["seeds"] == [BENCH_SEED_BASE, BENCH_SEED_BASE + 1]
    for block in result["ghosts"].values():
        assert block["overall"]["games"] == 2
        assert set(block["mazes"]) == {"lanes"}
    text = match_markdown(result)
    assert "AI ghosts (A* routes)" in text and "Chance (fixed odds)" in text
    assert "straight 60%, left 15%, right 15%, back 10%" in text


def test_match_command_writes_results_quickly(tmp_path):
    out = tmp_path / "match.json"
    assert cli_main(["match", "--games", "1", "--agent", "random", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert set(doc["ghosts"]) == {"ai", "chance"}
    assert out.with_suffix(".md").read_text().startswith("Pac-Man agent: Random")
