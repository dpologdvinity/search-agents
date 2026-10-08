"""Pac-Man tests: layouts, A*, the turn rules, ghost AI, features, the Q-update math, and agents.

Most rule tests use tiny one-corridor layouts registered with the `tiny` helper, so each
expected score and collision can be worked out by hand from the layout.
"""

import random

import pytest

from pacman import mazes
from pacman.agents import QAgent, RandomAgent, load_weights, make_agent
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
from pacman.features import FEATURE_NAMES, features, successor
from pacman.ghosts import Ghost, choose_move, target_for
from pacman.mazes import ACTIONS, LAYOUTS, parse
from pacman.qlearning import td_update, train
from pacman.search import astar, distances
from pacman.terminal import play_human, render

E = ACTIONS.index("E")


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
    for a in legal_actions(state):
        f = features(state, a)
        assert len(f) == len(FEATURE_NAMES)
        assert f[0] == 1.0
        assert all(x >= 0 for x in f)
        assert 0.25 <= f[FEATURE_NAMES.index("openness")] <= 1.0


def test_caught_and_adjacent_features_see_the_ghost(monkeypatch):
    m = tiny(monkeypatch, "feat", ["#####", "#PG.#", "#####"])
    f = features(initial_state(m), E)  # Pac-Man would step onto the ghost
    assert f[FEATURE_NAMES.index("caught")] == 1.0
    m2 = tiny(monkeypatch, "feat2", ["######", "#P.G##", "######"])
    f2 = features(initial_state(m2), E)  # Pac-Man would stand one step from the ghost
    assert f2[FEATURE_NAMES.index("ghost_adjacent")] == 1.0
    assert f2[FEATURE_NAMES.index("caught")] == 0.0


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
