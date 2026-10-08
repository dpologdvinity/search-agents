"""Rover: the PRNG and maps, both planners against a breadth-first reference, the explorer loop, and the CLI."""

import io
import random
from collections import deque

import pytest

from rover import __main__ as cli
from rover import benchmark
from rover.astar import AStar
from rover.dstar import INF, DStarLite
from rover.explorer import Explorer
from rover.world import Mulberry32, World, generate, neighbours, reachable, render, sensed_cells

# Fixed scenarios shared with the JavaScript port (tests/test_rover_parity.py runs web/js/rover_core.js on them).
# Each scenario edits the true map every 6 ticks at cell (t * 37 + 11) % N, unless that cell is the rover or the goal.
PARITY_SCENARIOS = [
    # (size, density, seed, radius) and the golden numbers the Python reference produces
    ((21, 0.25, 7, 2), dict(ticks=58, steps=58, replans=29, init_dstar=441, init_astar=41,
                            expansions_dstar=538, expansions_astar=715, walls=111)),
    ((15, 0.30, 3, 1), dict(ticks=50, steps=50, replans=22, init_dstar=225, init_astar=29,
                            expansions_dstar=303, expansions_astar=401, walls=72)),
]


def bfs_belief(n: int, walls, goal: int) -> list[float]:
    """Reference distances to the goal over `walls` (unknown counts as free). Independent of both planners."""
    dist = [INF] * (n * n)
    dist[goal] = 0
    queue = deque([goal])
    while queue:
        u = queue.popleft()
        for v in neighbours(n, u):
            if not walls[v] and dist[v] == INF:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def run_with_edits(n: int, density: float, seed: int, radius: int, driver: str = "dstar", every: int = 6):
    """The parity scenario: a fresh explorer, the true map edited on a fixed schedule. Returns (explorer, ticks)."""
    world = World(n, generate(n, density, seed))
    ex = Explorer(world, radius=radius, driver=driver)
    total = n * n
    t = 0
    while not ex.done and t < 20 * total:
        if t % every == 2:
            cell = (t * 37 + 11) % total
            ex.edit(cell, not world.walls[cell])
        ex.tick()
        t += 1
    return ex, t


# ── PRNG and maps ────────────────────────────────────────────────────────

def test_mulberry32_reference_values():
    # The first outputs for seed 1; the JavaScript port must print the same numbers.
    rng = Mulberry32(1)
    assert rng.next() == 0.6270739405881613
    assert rng.next() == 0.002735721180215478


def test_generate_is_seeded_and_keeps_corners_open():
    a = generate(21, 0.2, 5)
    assert a == generate(21, 0.2, 5)
    assert a != generate(21, 0.2, 6)
    assert a[0] == 0 and a[-1] == 0
    assert reachable(21, a, 0, 21 * 21 - 1)


@pytest.mark.parametrize("density", [0.1, 0.2, 0.3, 0.4])
def test_generated_goal_is_always_reachable(density):
    for seed in range(20):
        walls = generate(15, density, seed)
        assert reachable(15, walls, 0, 15 * 15 - 1)


def test_sensor_is_a_square_clipped_to_the_map():
    assert sorted(sensed_cells(5, 0, 1)) == [0, 1, 5, 6]          # corner: only three neighbours plus itself
    assert len(list(sensed_cells(9, 40, 2))) == 25                  # centre: 5x5 square
    assert set(sensed_cells(9, 40, 1)) == {30, 31, 32, 39, 40, 41, 48, 49, 50}


# ── planners against the breadth-first reference ─────────────────────────

def random_belief_walk(seed: int, n: int = 12, changes: int = 40):
    """Random wall flips fed to both planners, with the rover walking along D* Lite's own route."""
    rng = random.Random(seed)
    walls = bytearray(n * n)
    start, goal = 0, n * n - 1
    dstar = DStarLite(n, start, goal)
    astar = AStar(n, start, goal)
    pos = start
    for _ in range(changes):
        batch = []
        for _ in range(rng.randint(1, 3)):
            cell = rng.randrange(n * n)
            if cell in (pos, goal):
                continue
            walls[cell] ^= 1
            batch.append((cell, walls[cell]))
        ref = None
        if batch:
            dstar.on_change(batch, pos)
            astar.on_change(batch, pos)   # A*'s stored path starts at pos only right after a replan
            ref = bfs_belief(n, walls, goal)
            assert astar.cost_to_goal(pos) == ref[pos]
        else:
            ref = bfs_belief(n, walls, goal)
        assert dstar.cost_to_goal(pos) == ref[pos]
        nxt = dstar.next_cell(pos)
        if nxt is None:
            assert ref[pos] == INF
            continue
        assert not walls[nxt] and ref[nxt] == ref[pos] - 1   # the step really is one closer to the goal
        pos = nxt
        if pos == goal:
            break


@pytest.mark.parametrize("seed", range(8))
def test_planners_match_breadth_first_costs_under_random_changes(seed):
    random_belief_walk(seed)


def test_dstar_path_is_a_shortest_path_on_a_known_map():
    walls = generate(21, 0.3, 4)
    n = 21
    d = DStarLite(n, 0, n * n - 1)
    d.on_change([(c, 1) for c in range(n * n) if walls[c]], 0)
    path = d.path_from(0)
    assert path[0] == 0 and path[-1] == n * n - 1
    assert len(path) - 1 == bfs_belief(n, walls, n * n - 1)[0]
    assert all(not walls[c] for c in path)


def test_unreachable_goal_reports_no_route_and_no_crash():
    n = 6
    walls = bytearray(n * n)
    for r in range(n):
        walls[r * n + 3] = 1          # a full wall splits the grid
    d, a = DStarLite(n, 0, n * n - 1), AStar(n, 0, n * n - 1)
    changes = [(c, 1) for c in range(n * n) if walls[c]]
    d.on_change(changes, 0)
    a.on_change(changes, 0)
    assert d.cost_to_goal(0) == INF and d.next_cell(0) is None and d.path_from(0) == []
    assert a.cost_to_goal(0) == INF and a.next_cell(0) is None


# ── the explorer ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("driver", ["dstar", "astar"])
def test_explorer_reaches_goal_without_entering_walls(seed, driver):
    world = World(15, generate(15, 0.3, seed))
    ex = Explorer(world, radius=2, driver=driver).run()
    assert ex.done and not ex.stuck
    assert ex.mismatches == 0
    # The rover's route is never shorter than the true shortest path.
    assert ex.steps >= bfs_belief(15, world.walls, 15 * 15 - 1)[0]


def test_dstar_cost_matches_reference_at_every_tick_with_edits():
    for seed in range(4):
        world = World(13, generate(13, 0.25, seed))
        ex = Explorer(world, radius=1)
        goal = world.goal
        for t in range(20 * 13 * 13):
            if ex.done:
                break
            if t % 5 == 1:
                cell = (t * 29 + seed) % (13 * 13)
                replans = ex.replans
                ex.edit(cell, not world.walls[cell])
                if ex.replans != replans:   # A*'s stored path starts at the rover right after it replans
                    assert ex.planners["astar"].cost_to_goal(ex.pos) == bfs_belief(13, ex.belief, goal)[ex.pos]
            # A replan inside tick() happens before the move, so A*'s path starts at the cell the rover was on.
            here, replans = ex.pos, ex.replans
            ex.tick()
            if ex.replans != replans:
                ref = bfs_belief(13, ex.belief, goal)[here]
                assert ex.planners["astar"].cost_to_goal(here) == ref
                assert ex.planners["dstar"].cost_to_goal(here) == ref
            if ex.pos != goal:
                assert ex.planners["dstar"].cost_to_goal(ex.pos) == bfs_belief(13, ex.belief, goal)[ex.pos]


@pytest.mark.parametrize("case,golden", PARITY_SCENARIOS)
def test_parity_scenarios_match_the_golden_numbers(case, golden):
    n, density, seed, radius = case
    ex, ticks = run_with_edits(n, density, seed, radius)
    got = dict(ticks=ticks, steps=ex.steps, replans=ex.replans,
               init_dstar=ex.init_expansions["dstar"], init_astar=ex.init_expansions["astar"],
               expansions_dstar=ex.expansions("dstar"), expansions_astar=ex.expansions("astar"),
               walls=sum(ex.world.walls))
    assert ex.done and ex.mismatches == 0
    assert got == golden


def test_edit_refuses_the_rover_cell_and_the_goal():
    world = World(5, generate(5, 0.1, 1))
    ex = Explorer(world, radius=1)
    assert ex.edit(ex.pos, True) is False
    assert ex.edit(world.goal, True) is False
    assert world.walls[world.goal] == 0


def test_a_wall_dropped_in_view_triggers_a_replan_at_once():
    world = World(9, bytearray(81))
    ex = Explorer(world, radius=2)
    ex.refresh()                     # sense the start neighbourhood
    before = ex.replans
    cell = 2                         # two cells east of the start: inside the sensor square
    assert ex.edit(cell, True)
    assert ex.replans == before + 1 and ex.belief[cell] == 1
    assert ex.last_changes == [(cell, 1)]


def test_invalid_settings_are_rejected():
    world = World(5, generate(5, 0.1, 1))
    with pytest.raises(ValueError):
        Explorer(world, radius=0)
    with pytest.raises(ValueError):
        Explorer(world, radius=1, driver="bfs")


def test_render_marks_rover_goal_and_unseen_cells():
    world = World(3, bytearray(9))
    seen = bytearray(9)
    seen[0] = seen[1] = 1
    text = render(3, world.walls, seen, 0, 8)
    assert text.splitlines() == ["R. ", "   ", "  G"]   # rover, one seen free cell, unseen cells blank, goal


# ── CLI and benchmark ────────────────────────────────────────────────────

def test_cli_play_prints_a_frame_per_step_and_reaches_the_goal(capsys):
    buf = io.StringIO()
    ex = cli.play(11, 0.2, 3, 2, "dstar", 0.0, clear=False, out=buf)
    text = buf.getvalue()
    assert ex.done
    assert "reached the goal" in text and "D* Lite" in text
    assert "GOAL" in text


def test_cli_main_rejects_out_of_range_arguments():
    with pytest.raises(SystemExit):
        cli.main(["--size", "200"])
    with pytest.raises(SystemExit):
        cli.main(["--radius", "0"])


def test_benchmark_quick_run_writes_json_and_table(tmp_path):
    result = benchmark.run(sizes=(11,), densities=(0.2,), seeds=2, radius=2)
    benchmark.write(result, tmp_path)
    assert (tmp_path / "rover_benchmark.json").exists()
    table = (tmp_path / "rover_benchmark.md").read_text()
    assert "| 11x11 | 20% |" in table and "Cost check: 0 replans" in table
    assert sum(r["mismatches"] for r in result["rows"]) == 0
