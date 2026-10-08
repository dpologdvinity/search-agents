import itertools
import random

import pytest

from battleship import probability as P
from battleship.agents import HuntAgent, ProbabilityAgent, RandomAgent
from battleship.benchmark import run as run_benchmark
from battleship.board import (
    CELLS,
    FLEET,
    PLACEMENT_MASKS,
    PLACEMENTS,
    SIZE,
    Fleet,
    Knowledge,
    bits,
    cell_name,
    check_fleet,
    is_ship,
    mask_of,
    parse_cell,
    random_fleet,
)
from battleship.game import play_out

# A fixed legal fleet: Carrier on row 1, Battleship on row 3, Cruiser on row 5, Submarine on row 7, and
# Destroyer on row 9 at the left edge.
FIXED = [
    tuple(range(0, 5)),
    tuple(range(20, 24)),
    tuple(range(40, 43)),
    tuple(range(60, 63)),
    tuple(range(80, 82)),
]


def sink_ship(fleet: Fleet, k: Knowledge, index: int) -> None:
    """Fire at every cell of one ship, so the shooter learns it sank."""
    for c in FIXED[index]:
        s = fleet.fire(c)
        k.observe(c, s.result, s.ship)


def brute_force_probs(k: Knowledge) -> list[float]:
    """P(ship) per cell by listing every ordered combination of placements. Only for tiny remaining fleets."""
    blocked = k.miss | k.sunk
    options = [[m for m in PLACEMENT_MASKS[n] if not m & blocked] for n in k.remaining_lengths()]
    counts = [0] * CELLS
    total = 0
    for combo in itertools.product(*options):
        used = 0
        for m in combo:
            if used & m:
                break
            used |= m
        else:
            if k.open_hits & ~used:
                continue
            total += 1
            for c in bits(used):
                counts[c] += 1
    return [n / total for n in counts]


def state_with_three_ships_sunk() -> Knowledge:
    """Carrier, Battleship and Cruiser sunk; Submarine and Destroyer (3 and 2) remain."""
    fleet = Fleet(FIXED)
    k = Knowledge()
    for i in range(3):
        sink_ship(fleet, k, i)
    return k


def test_cell_names_round_trip():
    assert cell_name(62) == "C7"
    assert parse_cell("c7") == 62 == parse_cell("C7")
    assert parse_cell("J10") == 99 and parse_cell("a1") == 0
    for bad in ("", "k1", "a0", "a11", "7", "a", "aa1"):
        with pytest.raises(ValueError):
            parse_cell(bad)


def test_placements_are_straight_runs_on_the_board():
    for n, runs in PLACEMENTS.items():
        assert len(runs) == 2 * SIZE * (SIZE - n + 1)
        assert all(is_ship(run, n) for run in runs)


def test_is_ship_rejects_bent_wrapping_and_gapped_cells():
    assert is_ship([3, 4, 5], 3)
    assert not is_ship([8, 9, 10], 3)  # wraps from the end of row 0 into row 1
    assert not is_ship([0, 1, 11], 3)  # bent
    assert not is_ship([0, 2, 4], 3)  # gaps
    assert not is_ship([0, 10, 30], 3)  # not consecutive
    assert not is_ship([0, 1], 3)  # wrong length


def test_check_fleet_rules():
    assert check_fleet(FIXED)
    overlapping = list(FIXED)
    overlapping[1] = (4, 5, 6, 7)  # shares cell 4 with the Carrier
    assert not check_fleet(overlapping)
    assert not check_fleet(FIXED[:4])  # wrong fleet size
    assert not check_fleet([(0, 1, 2, 3, 4)] * 5)


@pytest.mark.parametrize("seed", range(50))
def test_random_fleets_are_legal(seed):
    ships = random_fleet(random.Random(seed))
    assert check_fleet(ships)
    assert sorted(len(s) for s in ships) == sorted(FLEET)


def test_fleet_answers_miss_hit_and_sunk():
    fleet = Fleet(FIXED)
    assert fleet.fire(55).result == "miss"
    assert fleet.fire(0).result == "hit"
    for c in (1, 2, 3):
        assert fleet.fire(c).result == "hit"
    s = fleet.fire(4)
    assert s.result == "sunk" and s.ship == FIXED[0] and s.ship_index == 0
    assert fleet.afloat() == 4 and not fleet.all_sunk
    with pytest.raises(ValueError):
        fleet.fire(4)  # no firing at the same cell twice


def test_knowledge_tracks_sunk_ships_and_remaining_lengths():
    k = state_with_three_ships_sunk()
    assert k.remaining_lengths() == [3, 2]
    assert k.open_hits == 0
    assert bin(k.sunk).count("1") == 5 + 4 + 3


def test_knowledge_rejects_inconsistent_reports():
    k = Knowledge()
    k.observe(0, "hit")
    with pytest.raises(ValueError):
        k.observe(0, "miss")  # fired at twice
    with pytest.raises(ValueError):
        k.observe(1, "sunk", (1, 2))  # a sunk ship of length 2 whose other cell was never hit
    with pytest.raises(ValueError):
        k.observe(5, "sunk", (5, 6, 7))  # 3 cells, but 6 was not hit
    with pytest.raises(ValueError):
        k.observe(9, "boom")
    with pytest.raises(ValueError):
        k.observe(200, "miss")


def test_exact_probabilities_match_brute_force():
    k = state_with_three_ships_sunk()
    k.observe(55, "miss")
    k.observe(61, "hit")  # open hit on the Submarine, in row 7
    belief = P.ship_probabilities(k, budget=10**9)
    assert belief.method == "exact"
    expected = brute_force_probs(k)
    for c in k.unknown_cells():
        assert belief.probs[c] == pytest.approx(expected[c], abs=1e-12)


def test_hit_neighbours_rise_above_far_cells():
    k = Knowledge()
    k.observe(44, "hit")  # open hit in the middle of the board, no ship sunk
    rng = random.Random(0)
    belief = P.ship_probabilities(k, rng)
    far = 0  # a corner far from the hit
    for neighbour in (34, 54, 43, 45):
        assert belief.probs[neighbour] > belief.probs[far]


def test_probabilities_sum_to_the_ship_cells_still_to_find():
    k = state_with_three_ships_sunk()
    k.observe(55, "miss")
    k.observe(61, "hit")
    expected_total = sum(k.remaining_lengths()) - bin(k.open_hits).count("1")
    for budget in (10**9, 0):  # exact, then forced sampling
        belief = P.ship_probabilities(k, random.Random(1), budget=budget)
        unknown_total = sum(belief.probs[c] for c in k.unknown_cells())
        assert unknown_total == pytest.approx(expected_total, abs=1e-6)
    assert P.ship_probabilities(k, random.Random(1), budget=0).method == "sampled"


def test_misses_are_zero_and_hits_are_one():
    k = state_with_three_ships_sunk()
    k.observe(55, "miss")
    k.observe(61, "hit")
    belief = P.ship_probabilities(k, random.Random(2))
    assert belief.probs[55] == 0.0
    assert belief.probs[61] == 1.0
    assert all(belief.probs[c] == 1.0 for c in FIXED[0] + FIXED[1] + FIXED[2])


def test_sampling_agrees_with_exact_counts():
    k = state_with_three_ships_sunk()
    k.observe(55, "miss")
    k.observe(61, "hit")
    exact = P.ship_probabilities(k, budget=10**9).probs
    lengths = k.remaining_lengths()
    free = P._free_placements(k.miss | k.sunk)
    estimate = P._sample(lengths, free, k.open_hits, random.Random(3), 20000)
    for c in k.unknown_cells():
        assert abs(estimate[c] - exact[c]) < 0.03


def test_no_consistent_placement_gives_none_and_uniform_choice():
    k = Knowledge()
    for c in range(CELLS):  # every cell missed: no ship can fit anywhere
        if c != 0:
            k.observe(c, "miss")
    belief = P.ship_probabilities(k, random.Random(0))
    assert belief.method == "none"
    cell = P.choose_cell(belief, k.unknown_cells(), random.Random(0))
    assert cell == 0


def test_hunt_fires_next_to_a_hit_then_along_the_line():
    k = Knowledge()
    k.observe(44, "hit")
    agent = HuntAgent(random.Random(4))
    for _ in range(20):
        assert agent.choose(k) in {34, 54, 43, 45}
    k.observe(45, "hit")  # two hits in a row: the next shot should extend the line
    for _ in range(20):
        assert agent.choose(k) in {43, 46}


def test_hunt_uses_parity_when_nothing_is_wounded():
    agent = HuntAgent(random.Random(5))
    k = Knowledge()
    for _ in range(10):
        c = agent.choose(k)
        assert (c // SIZE + c % SIZE) % 2 == 0
        k.observe(c, "miss")


def test_agents_only_fire_at_unknown_cells():
    rng = random.Random(6)
    for agent in (RandomAgent(rng), HuntAgent(rng), ProbabilityAgent(rng, samples=300)):
        fleet = Fleet(random_fleet(random.Random(9)))
        k = Knowledge()
        for _ in range(30):
            c = agent.choose(k)
            assert not (k.fired >> c) & 1
            s = fleet.fire(c)
            k.observe(c, s.result, s.ship)


def test_probability_agent_beats_random_on_fixed_fleets():
    fleets = [random_fleet(random.Random(seed)) for seed in range(3)]
    prob = [play_out(ProbabilityAgent(random.Random(seed)), f) for seed, f in enumerate(fleets)]
    rand = [play_out(RandomAgent(random.Random(seed)), f) for seed, f in enumerate(fleets)]
    assert sum(prob) / 3 < sum(rand) / 3 - 20
    assert all(p <= CELLS for p in prob)


def test_benchmark_table_has_every_agent():
    results = run_benchmark(games=1, seed=2)
    assert set(results) == {"random", "hunt", "probability"}
    assert all(len(r["shots"]) == 1 for r in results.values())


def test_words_match_masks():
    m = mask_of([0, 63, 64, 99])
    lo, hi = P._words(m)
    assert int(lo) == (1 << 0) | (1 << 63)
    assert int(hi) == (1 << 0) | (1 << 35)


def test_terminal_game_handles_bad_input_and_quits():
    from battleship.__main__ import play, watch

    answers = iter(["zz", "a1", "a1", "q"])  # not a cell, a shot, the same cell again, then quit
    lines = []
    outcome = play("probability", random.Random(3), read=lambda prompt="": next(answers), write=lines.append)
    assert outcome == "quit"
    text = "\n".join(lines)
    assert "not a cell" in text and "already fired" in text

    shots = watch("hunt", random.Random(4), write=lambda s: None)
    assert 17 <= shots <= CELLS
