"""Tests for the warehouse package: single-robot search, collision detection, baselines, CBS, and limits.

The brute-force check computes the true minimum sum of costs on tiny maps by a layered search over
joint positions, so CBS optimality is tested against an independent method, not against itself.
"""

import itertools
import random

import pytest

from warehouse import Grid, Problem, cbs, independent, layout_grid, prioritized, random_instance
from warehouse.conflicts import find_conflicts, makespan, sum_of_costs
from warehouse.space_time import EdgeConstraint, Reservations, VertexConstraint, space_time_astar

SWAP_CORRIDOR = ["##.##", "....."]  # a one-wide corridor with a single side pocket at x=2


def swap_problem():
    """Robot 0 goes left to right, robot 1 right to left. Only the pocket lets them pass."""
    g = Grid(SWAP_CORRIDOR)
    return Problem.from_xy(g, [(0, 1), (4, 1)], [(4, 1), (0, 1)])


def brute_min_soc(problem, H=9):
    """Minimum sum of arrival times over all collision-free plans of at most H steps.

    A robot's cost is the last time it is off its goal, plus one. Robots may pass their goal early
    and leave again, as in the CBS model. The state is (joint positions, last non-goal time per robot),
    so the search is exact for small maps. Returns None if no plan finishes within H steps.
    """
    g, n, goals = problem.grid, problem.robots, problem.goals
    start = tuple(problem.starts)
    last0 = tuple(0 if s != gl else -1 for s, gl in zip(problem.starts, goals))
    states = {(start, last0)}
    for t in range(1, H + 1):
        nxt_states = set()
        for pos, last in states:
            options = [[pos[i], *g.neighbors[pos[i]]] for i in range(n)]
            for nxt in itertools.product(*options):
                if len(set(nxt)) < n:
                    continue  # two robots on one cell
                if any(nxt[i] == pos[j] and nxt[j] == pos[i] for i in range(n) for j in range(i + 1, n)):
                    continue  # swap
                new_last = tuple(t if nxt[i] != goals[i] else last[i] for i in range(n))
                nxt_states.add((nxt, new_last))
        states = nxt_states
    costs = [sum(v + 1 for v in last) for pos, last in states if all(pos[i] == goals[i] for i in range(n))]
    return min(costs) if costs else None


# Single-robot space-time search

def test_vertex_constraint_forces_a_wait_and_the_arrival_is_minimal():
    # On a 1-row corridor 0..5, blocking cell 2 at time 2 forces one wait at cell 1.
    # The fastest legal walk is 0,1,1,2,3,4,5: arrival at time 6.
    corridor = Grid(["......"])
    path, _ = space_time_astar(corridor, 0, 5, Reservations([VertexConstraint(0, 2, 2)]))
    assert path == (0, 1, 1, 2, 3, 4, 5)
    assert len(path) - 1 == 6


def test_edge_constraint_forbids_one_move_but_not_the_cell():
    # Forbidding the move 1 -> 2 that arrives at time 2: the robot waits on cell 1 and moves at time 3.
    corridor = Grid(["..."])
    path, _ = space_time_astar(corridor, 0, 2, Reservations([EdgeConstraint(0, 1, 2, 2)]))
    assert path == (0, 1, 1, 2)
    # The reverse move is not forbidden, so a robot on cell 2 may still walk back to cell 1.
    back, _ = space_time_astar(corridor, 2, 1, Reservations([EdgeConstraint(0, 1, 2, 2)]))
    assert back == (2, 1)


def test_no_path_within_horizon_returns_none():
    # Cell 1 is blocked for every time from 1 to 20, and the horizon stops at 10: there is no way through.
    corridor = Grid(["..."])
    res = Reservations([VertexConstraint(0, 1, t) for t in range(1, 21)])
    path, _ = space_time_astar(corridor, 0, 2, res, horizon=10)
    assert path is None


def test_parked_robot_blocks_its_cell_for_good():
    # A finished robot parked on cell 1 blocks it at every later time, so a path can't pass through.
    corridor = Grid(["..."])
    res = Reservations()
    res.add_path((1,))  # the other robot is already at its goal, cell 1, from time 0
    path, _ = space_time_astar(corridor, 0, 2, res)
    assert path is None


# Collision detection

def test_find_conflicts_vertex_and_edge_and_none():
    vertex = find_conflicts([(0, 1, 2), (3, 4, 2)])  # both robots stand on cell 2 at time 2
    assert len(vertex) == 1
    assert (vertex[0].kind, vertex[0].t, vertex[0].robots, vertex[0].cells) == ("vertex", 2, (0, 1), (2,))

    swap = find_conflicts([(0, 1), (1, 0)])
    assert len(swap) == 1
    assert (swap[0].kind, swap[0].t, swap[0].robots, swap[0].cells) == ("edge", 1, (0, 1), (0, 1))

    assert find_conflicts([(0, 1), (2, 3)]) == []


def test_sum_of_costs_and_makespan_clamp_finished_robots():
    paths = [(0, 1), (5, 6, 7, 8)]
    assert sum_of_costs(paths) == 1 + 3
    assert makespan(paths) == 3


# Baselines

def test_independent_astar_swap_on_a_two_cell_corridor_is_an_edge_conflict():
    g = Grid([".."])
    p = Problem.from_xy(g, [(0, 0), (1, 0)], [(1, 0), (0, 0)])
    conflicts = independent(p).conflicts
    assert any(c.kind == "edge" for c in conflicts)


def test_independent_astar_collides_at_the_doorway():
    # Robot 1 starts in the pocket and parks on the doorway cell (2, 1). Robot 0 walks through the
    # doorway at time 2 while robot 1 is still there, so the independent plan collides.
    g = Grid(SWAP_CORRIDOR)
    p = Problem.from_xy(g, [(0, 1), (2, 0)], [(4, 1), (2, 1)])
    result = independent(p)
    assert result.status == "collisions"
    assert [(c.kind, c.t, c.cells) for c in result.conflicts] == [("vertex", 2, (g.cell(2, 1),))]


def test_prioritized_fails_on_the_swap_corridor():
    result = prioritized(swap_problem())
    assert result.status == "failed" and result.paths is None


# CBS

def test_cbs_solves_swap_corridor_with_a_wait():
    result = cbs(swap_problem())
    assert result.status == "solved"
    assert result.sum_of_costs == 11  # robot 0 arrives at 5, robot 1 at 6
    assert find_conflicts(result.paths) == []
    # Robot 0 waits once, on cell (1, 1), so its path has a repeated cell.
    waits = sum(a == b for a, b in zip(result.paths[0], result.paths[0][1:]))
    assert waits >= 1
    assert result.conflicts_resolved >= 1 and result.high_nodes > 1


def test_cbs_names_the_tree_node_it_returns():
    # The returned node is a leaf: it was never expanded, but it was created as a child of an expanded node.
    # The page marks it by this id; a different leaf with the same cost must not be mistaken for it.
    result = cbs(swap_problem())
    children = {c["id"]: c for e in result.trace for c in e["children"] if "id" in c}
    assert result.solution in children
    assert children[result.solution]["cost"] == result.sum_of_costs
    assert all(e["id"] != result.solution for e in result.trace)
    assert cbs(swap_problem(), max_nodes=1).solution is None


@pytest.mark.parametrize("seed", range(5))
def test_cbs_is_collision_free_and_no_worse_than_prioritized(seed):
    problem = random_instance("bottleneck", 4, seed)
    exact = cbs(problem)
    greedy = prioritized(problem)
    assert exact.status == "solved"
    assert find_conflicts(exact.paths) == []
    # Prioritized planning may fail outright on some draws; when it does, there is nothing to compare.
    if greedy.status == "solved":
        assert find_conflicts(greedy.paths) == []
        assert exact.sum_of_costs <= greedy.sum_of_costs
    else:
        assert greedy.status == "failed" and greedy.paths is None


def test_cbs_matches_brute_force_optimum_on_tiny_maps():
    rng = random.Random(7)
    checked = 0
    while checked < 20:
        rows = ["".join("#" if rng.random() < 0.2 else "." for _ in range(4)) for _ in range(3)]
        grid = Grid(rows)
        free = grid.floor_cells()
        if len(free) < 4:
            continue
        cells = rng.sample(free, 4)
        try:
            problem = Problem(grid, tuple(cells[:2]), tuple(cells[2:]))
        except ValueError:
            continue  # goal unreachable from start
        # Small budget: draws that are infeasible (swap-locked) would otherwise grow the tree until the limit.
        result = cbs(problem, max_nodes=300, max_seconds=1.0)
        if result.status != "solved" or result.makespan > 8:
            continue  # unsolved within budget, or longer than the brute force looks ahead
        best = brute_min_soc(problem, H=9)
        assert result.sum_of_costs == best, rows
        checked += 1


# Limits and status

def test_node_limit_returns_status_without_a_plan():
    result = cbs(swap_problem(), max_nodes=1)
    assert result.status == "node_limit"
    assert result.paths is None and result.sum_of_costs is None


def test_time_limit_returns_status_without_a_plan():
    result = cbs(swap_problem(), max_seconds=0.0)
    assert result.status == "time_limit"
    assert result.paths is None


def test_unsolvable_instance_never_returns_a_plan():
    # Two robots that must swap through a one-cell corridor have no plan. The tree never empties,
    # because every constraint has a finite time and a waiting robot can always satisfy it, so the
    # search runs until a budget stops it. The test pins that behaviour: no plan, and a limit status.
    g = Grid([".."])
    p = Problem.from_xy(g, [(0, 0), (1, 0)], [(1, 0), (0, 0)])
    result = cbs(p, max_nodes=200, max_seconds=2.0)
    assert result.status in ("node_limit", "time_limit")
    assert result.paths is None


# Problems and instances

def test_problem_rejects_duplicate_starts_and_unreachable_goals():
    g = Grid(["...", "..."])
    with pytest.raises(ValueError, match="same cell"):
        Problem.from_xy(g, [(0, 0), (0, 0)], [(1, 0), (2, 0)])
    walled = Grid(["..#.."])
    with pytest.raises(ValueError, match="cannot be reached"):
        Problem.from_xy(walled, [(0, 0)], [(4, 0)])


def test_random_instance_is_deterministic_and_solved():
    a = random_instance("crossroads", 3, 5)
    b = random_instance("crossroads", 3, 5)
    assert (a.starts, a.goals) == (b.starts, b.goals)
    assert cbs(a).status == "solved"


def test_builtin_layouts_are_connected_floor():
    for name in ("aisles", "crossroads", "bottleneck"):
        grid = layout_grid(name)
        assert len(grid.largest_component()) == len(grid.floor_cells()), name
