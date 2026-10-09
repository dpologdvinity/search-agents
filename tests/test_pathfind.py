"""Tests for the pathfinding package: optimality against a brute-force reference, the weighted-A*
bound, perfect mazes, determinism, and the CLI.

The brute-force reference is Bellman-Ford over the same edges (Grid.neighbors and step_cost), with no
heap, no heuristic and no early exit, so it shares nothing with the searches except the movement rules.
"""

from __future__ import annotations

import random

import pytest

from pathfind.cli import main
from pathfind.grid import MUD, OPEN, SWAMP, WALL, Grid, heuristic
from pathfind.mazes import make_maze
from pathfind.race import race, render
from pathfind.rng import Rng
from pathfind.search import ALGOS, bfs, bidirectional_bfs, run, uniform_cost, weighted_astar


def brute_force(grid: Grid, start: int, goal: int, diagonal: bool) -> float | None:
    """Least path cost by Bellman-Ford relaxation over every edge; None when unreachable."""
    inf = float("inf")
    dist = [inf] * len(grid)
    dist[start] = 0.0
    edges = [
        (a, b, grid.step_cost(a, b))
        for a in range(len(grid))
        if grid.cells[a] != WALL
        for b, _mult in grid.neighbors(a, diagonal)
    ]
    for _ in range(len(grid)):
        changed = False
        for a, b, c in edges:
            if dist[a] + c < dist[b]:
                dist[b] = dist[a] + c
                changed = True
        if not changed:
            break
    return None if dist[goal] == inf else dist[goal]


def random_grid(rng: random.Random, n: int, wall_pct: int, terrain: bool) -> Grid:
    cells = []
    for _ in range(n * n):
        r = rng.randrange(100)
        if r < wall_pct:
            cells.append(WALL)
        elif terrain and r < wall_pct + 12:
            cells.append(SWAMP)
        elif terrain and r < wall_pct + 18:
            cells.append(MUD)
        else:
            cells.append(OPEN)
    return Grid(n, n, cells)


def random_cases(count: int, seed: int):
    rng = random.Random(seed)
    for k in range(count):
        n = rng.randrange(3, 8)
        grid = random_grid(rng, n, wall_pct=rng.choice([0, 15, 30]), terrain=k % 2 == 0)
        open_cells = [i for i, c in enumerate(grid.cells) if c != WALL]
        if len(open_cells) < 2:
            continue
        start, goal = rng.sample(open_cells, 2)
        yield grid, start, goal


CASES = list(random_cases(120, seed=2026))


def assert_valid_path(grid: Grid, path: list[int], start: int, goal: int, diagonal: bool) -> None:
    assert path[0] == start and path[-1] == goal
    for a, b in zip(path, path[1:], strict=False):
        assert b in [v for v, _m in grid.neighbors(a, diagonal)]


@pytest.mark.parametrize("diagonal", [False, True])
def test_uniform_cost_and_astar_match_brute_force(diagonal):
    for grid, start, goal in CASES:
        ref = brute_force(grid, start, goal, diagonal)
        ucs = uniform_cost(grid, start, goal, diagonal)
        assert (ucs.cost is None) == (ref is None)
        if ref is None:
            continue
        assert ucs.cost == pytest.approx(ref)
        assert_valid_path(grid, ucs.path, start, goal, diagonal)
        # A* with an admissible heuristic must match. Manhattan is only admissible without diagonals.
        names = ["manhattan", "euclidean", "octile"] if not diagonal else ["euclidean", "octile"]
        for heur in names:
            res = run("astar", grid, start, goal, diagonal=diagonal, heur=heur)
            assert res.found
            assert res.cost == pytest.approx(ref), (heur, diagonal)
            assert_valid_path(grid, res.path, start, goal, diagonal)


def test_bfs_is_optimal_on_unit_cost_grids():
    rng = random.Random(7)
    for _ in range(80):
        n = rng.randrange(3, 9)
        grid = random_grid(rng, n, wall_pct=25, terrain=False)
        open_cells = [i for i, c in enumerate(grid.cells) if c != WALL]
        if len(open_cells) < 2:
            continue
        start, goal = rng.sample(open_cells, 2)
        # Without diagonals every step costs 1, so the fewest steps is also the cheapest path.
        ref = brute_force(grid, start, goal, False)
        res = bfs(grid, start, goal, False)
        assert (res.cost is None) == (ref is None)
        if ref is not None:
            assert res.cost == pytest.approx(ref)


@pytest.mark.parametrize("weight", [1.0, 1.5, 2.0, 3.0])
def test_weighted_astar_within_bound(weight):
    for grid, start, goal in CASES:
        for diagonal in (False, True):
            ref = brute_force(grid, start, goal, diagonal)
            res = weighted_astar(grid, start, goal, diagonal, "octile", weight)
            if ref is None:
                assert not res.found
                continue
            assert res.found
            assert res.cost <= weight * ref + 1e-9


def test_every_algorithm_agrees_on_reachability():
    for grid, start, goal in CASES[:60]:
        ref = brute_force(grid, start, goal, False)
        for algo in ALGOS:
            res = run(algo, grid, start, goal)
            assert res.found == (ref is not None), algo
            if res.found:
                assert_valid_path(grid, res.path, start, goal, False)
                assert res.cost is not None
            assert res.expanded == len(res.steps)
            # A cell is expanded at most once, so the expansion order has no repeats.
            order = res.expansion_order
            assert len(order) == len(set(order)), algo


def test_bidirectional_path_is_connected():
    for grid, start, goal in CASES[:60]:
        res = bidirectional_bfs(grid, start, goal, True)
        if res.found:
            assert_valid_path(grid, res.path, start, goal, True)


@pytest.mark.parametrize("diagonal", [False, True])
def test_bidirectional_is_step_optimal_like_bfs(diagonal):
    # Both count steps, so bidirectional BFS must find a path with as few steps as BFS. Stopping at the
    # first meeting broke this on diagonal grids; the random grids here cover both movement rules.
    rng = random.Random(11)
    cases = list(CASES)
    for _ in range(120):
        n = rng.randrange(3, 10)
        grid = random_grid(rng, n, wall_pct=rng.choice([0, 20, 35]), terrain=False)
        open_cells = [i for i, c in enumerate(grid.cells) if c != WALL]
        if len(open_cells) >= 2:
            start, goal = rng.sample(open_cells, 2)
            cases.append((grid, start, goal))
    for grid, start, goal in cases:
        ref = bfs(grid, start, goal, diagonal)
        res = bidirectional_bfs(grid, start, goal, diagonal)
        assert res.found == ref.found, (start, goal)
        if ref.found:
            assert_valid_path(grid, res.path, start, goal, diagonal)
            assert len(res.path) == len(ref.path), (start, goal)


def test_bidirectional_finishes_the_level_before_stopping():
    # Open 4x3 grid with diagonals, start (0, 2) to goal (3, 0). The shortest path is 3 steps (two
    # diagonals and one straight step), so 4 cells. Stopping at the first meeting gave 4 steps.
    grid = Grid(4, 3)
    start, goal = grid.index(0, 2), grid.index(3, 0)
    assert len(bfs(grid, start, goal, True).path) == 4
    assert len(bidirectional_bfs(grid, start, goal, True).path) == 4


def test_wall_corners_block_diagonal_moves():
    # Two walls touching at a corner: the diagonal between them is not allowed.
    grid = Grid(2, 2, [OPEN, WALL, WALL, OPEN])
    assert [v for v, _m in grid.neighbors(0, diagonal=True)] == []
    assert bfs(grid, 0, 3, diagonal=True).found is False


def test_step_cost_uses_terrain_and_diagonal_length():
    # Centre cell (index 4) is swamp. Cost is the destination's terrain times the step length.
    grid = Grid(3, 3, [OPEN] * 4 + [SWAMP] + [OPEN] * 4)
    assert grid.step_cost(0, 1) == 1  # orthogonal onto open ground
    assert grid.step_cost(1, 4) == 5  # orthogonal onto swamp
    assert grid.step_cost(0, 4) == pytest.approx(5 * 1.4142135623730951)  # diagonal onto swamp
    assert grid.step_cost(4, 0) == pytest.approx(1.4142135623730951)  # leaving swamp costs the destination only


@pytest.mark.parametrize("kind", ["recursive", "prim"])
@pytest.mark.parametrize("size,seed", [(11, 1), (21, 2), (31, 3), (41, 9)])
def test_lattice_mazes_are_perfect(kind, size, seed):
    maze = make_maze(kind, size, size, seed)
    grid = maze.grid
    # Every lattice cell is open and every passage is a tree edge: V - 1 passages, all reachable.
    lattice = [grid.index(x, y) for y in range(1, size - 1, 2) for x in range(1, size - 1, 2)]
    assert all(grid.cells[i] != WALL for i in lattice)
    passages = 0
    for y in range(1, size - 1, 2):
        for x in range(1, size - 1, 2):
            if x + 2 < size - 1 and grid.cells[grid.index(x + 1, y)] != WALL:
                passages += 1
            if y + 2 < size - 1 and grid.cells[grid.index(x, y + 1)] != WALL:
                passages += 1
    assert passages == len(lattice) - 1
    # Every open cell is reachable from the start (one connected tree).
    open_cells = [i for i, c in enumerate(grid.cells) if c != WALL]
    seen = {maze.start}
    stack = [maze.start]
    while stack:
        u = stack.pop()
        for v, _m in grid.neighbors(u):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    assert seen == set(open_cells)
    # Border stays solid.
    assert all(grid.cells[grid.index(x, 0)] == WALL for x in range(size))


@pytest.mark.parametrize("kind", ["recursive", "prim", "scatter", "rooms", "open"])
def test_mazes_are_deterministic(kind):
    a = make_maze(kind, 25, 25, 11, swamp=10)
    b = make_maze(kind, 25, 25, 11, swamp=10)
    c = make_maze(kind, 25, 25, 12, swamp=10)
    assert a.grid.cells == b.grid.cells and (a.start, a.goal) == (b.start, b.goal)
    if kind != "open":
        assert a.grid.cells != c.grid.cells


def test_rooms_maze_is_connected_from_start_to_goal():
    for seed in range(1, 8):
        maze = make_maze("rooms", 41, 41, seed)
        assert uniform_cost(maze.grid, maze.start, maze.goal).found


def test_search_is_deterministic_for_same_input():
    maze = make_maze("prim", 31, 31, 4, swamp=15)
    for algo in ALGOS:
        a = run(algo, maze.grid, maze.start, maze.goal, diagonal=True)
        b = run(algo, maze.grid, maze.start, maze.goal, diagonal=True)
        assert a.steps == b.steps and a.path == b.path


def test_cost_optimal_algorithms_are_marked_optimal_on_swamp_map():
    maze = make_maze("prim", 41, 41, 3, swamp=20)
    optimal, rows = race(maze)
    by = {r.algo: r for r in rows}
    assert optimal is not None
    assert by["ucs"].is_optimal
    assert by["astar"].is_optimal
    assert by["wastar"].result.cost <= 2.0 * optimal + 1e-9


def test_frontier_sizes_match_step_count():
    maze = make_maze("rooms", 31, 31, 5)
    res = run("astar", maze.grid, maze.start, maze.goal, heur="euclidean")
    sizes = res.frontier_sizes()
    assert len(sizes) == res.expanded
    assert all(s >= 0 for s in sizes)


def test_heuristic_values():
    assert heuristic("manhattan", 0, 0, 3, 4) == 7
    assert heuristic("euclidean", 0, 0, 3, 4) == 5
    assert heuristic("octile", 0, 0, 3, 4) == pytest.approx(4 + (2**0.5 - 1) * 3)
    with pytest.raises(ValueError):
        heuristic("chebyshev", 0, 0, 1, 1)


def test_rng_is_deterministic_and_in_range():
    a, b = Rng(42), Rng(42)
    vals = [a.u32() for _ in range(50)]
    assert vals == [b.u32() for _ in range(50)]
    assert all(0 <= v < 2**32 for v in vals)
    assert len(set(vals)) == 50
    assert all(0 <= Rng(s).int(7) < 7 for s in range(100))


def test_cli_race_and_show_run(capsys):
    assert main(["race", "--size", "21", "--maze", "prim", "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "ALGO" in out and "optimal cost" in out
    assert out.count("*") > 0
    assert main(["show", "--algo", "astar", "--size", "21", "--seed", "3", "--trace", "3"]) == 0
    out = capsys.readouterr().out
    assert "g=" in out and "expanded=" in out
    assert main(["race", "--size", "21", "--algos", "nope"]) == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["race", "--size", "0"],
        ["race", "--size", "4"],
        ["show", "--size", "4"],
        ["race", "--weight", "0.5"],
        ["race", "--swamp", "150"],
        ["race", "--density", "-1"],
        ["show", "--trace", "-1"],
    ],
)
def test_cli_rejects_values_the_maps_and_searches_cannot_use(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "error" in capsys.readouterr().err


def test_render_marks_start_goal_and_path():
    grid = Grid(3, 1, [OPEN, SWAMP, OPEN])
    assert render(grid, 0, 2, [0, 1, 2]) == "S*G"
    assert render(Grid(3, 1, [OPEN, SWAMP, WALL]), 0, 1, []) == "SG#"
