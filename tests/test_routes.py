import itertools
import random

import pytest

from routes.tsp import (
    SOLVERS,
    distance_matrix,
    held_karp,
    order_crossover,
    reverse_segment,
    tour_length,
    two_opt_delta,
)


def random_cities(n, seed):
    rng = random.Random(seed)
    return [(rng.random(), rng.random()) for _ in range(n)]


def is_tour(tour, n):
    return sorted(tour) == list(range(n))


@pytest.mark.parametrize("seed", range(5))
def test_two_opt_delta_matches_recomputing(seed):
    rng = random.Random(seed)
    cities = random_cities(15, seed)
    dist = distance_matrix(cities)
    tour = list(range(15))
    rng.shuffle(tour)
    for _ in range(30):
        i, j = sorted(rng.sample(range(1, 15), 2))
        before = tour_length(tour, dist)
        delta = two_opt_delta(tour, i, j, dist)
        reverse_segment(tour, i, j)
        assert tour_length(tour, dist) == pytest.approx(before + delta)


@pytest.mark.parametrize("seed", range(5))
def test_order_crossover_keeps_a_valid_tour(seed):
    rng = random.Random(seed)
    a, b = list(range(20)), list(range(20))
    rng.shuffle(a)
    rng.shuffle(b)
    child = order_crossover(a, b, rng)
    assert is_tour(child, 20)


def test_order_crossover_example_from_docstring():
    class FixedSlice(random.Random):
        def sample(self, population, k):
            return [2, 4]  # keep positions 2..4 from parent A

    child = order_crossover([1, 2, 3, 4, 5, 6, 7], [5, 7, 1, 6, 2, 4, 3], FixedSlice())
    assert child == [6, 2, 3, 4, 5, 7, 1]


@pytest.mark.parametrize("n", [3, 5, 8])
def test_held_karp_matches_brute_force(n):
    cities = random_cities(n, n)
    dist = distance_matrix(cities)
    brute = min(tour_length([0, *p], dist) for p in itertools.permutations(range(1, n)))
    result = held_karp(cities)
    assert is_tour(result.tour, n)
    assert result.length == pytest.approx(brute)
    assert tour_length(result.tour, dist) == pytest.approx(brute)


def test_held_karp_refuses_large_instances():
    with pytest.raises(ValueError):
        held_karp(random_cities(13, 0))


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_heuristics_reach_the_optimum_on_small_maps(name):
    cities = random_cities(10, 42)
    optimum = held_karp(cities).length
    kwargs = {} if name == "nearest_neighbor_2opt" else {"seed": 0}
    result = SOLVERS[name](cities, **kwargs)
    assert is_tour(result.tour, 10)
    assert result.length == pytest.approx(tour_length(result.tour, distance_matrix(cities)))
    assert result.length <= optimum * 1.05
    assert 0 < len(result.frames) <= 160
    assert result.frames[-1]["best_length"] >= result.length - 1e-6


def test_annealing_records_a_cooling_schedule():
    result = SOLVERS["simulated_annealing"](random_cities(20, 1), iterations=5000, seed=1)
    temps = [f["temperature"] for f in result.frames]
    assert temps == sorted(temps, reverse=True)
    assert temps[-1] < temps[0] * 0.01


def test_annealing_result_is_two_opt_optimal():
    # The finishing pass means no single segment reversal can shorten the result.
    cities = random_cities(25, 3)
    dist = distance_matrix(cities)
    result = SOLVERS["simulated_annealing"](cities, iterations=3000, seed=2)
    n = len(result.tour)
    assert all(two_opt_delta(result.tour, i, j, dist) >= -1e-9 for i in range(1, n - 1) for j in range(i + 1, n))


def test_annealing_rejects_settings_that_break_the_schedule():
    cities = random_cities(8, 0)
    with pytest.raises(ValueError, match="at least one iteration"):
        SOLVERS["simulated_annealing"](cities, iterations=0)
    for cooling in (0.0, -0.5, 1.5):  # zero would divide by zero; above 1 would heat instead of cool
        with pytest.raises(ValueError, match="cooling"):
            SOLVERS["simulated_annealing"](cities, iterations=10, cooling=cooling)


def test_genetic_rejects_an_empty_population_or_no_generations():
    cities = random_cities(8, 0)
    with pytest.raises(ValueError, match="one tour and one generation"):
        SOLVERS["genetic_algorithm"](cities, population_size=0)
    with pytest.raises(ValueError, match="one tour and one generation"):
        SOLVERS["genetic_algorithm"](cities, generations=0)  # used to return a tour of infinite length


def test_genetic_needs_more_tours_than_the_elite():
    cities = random_cities(8, 0)
    with pytest.raises(ValueError, match="more tours than the 4 elite"):
        SOLVERS["genetic_algorithm"](cities, population_size=4, elite=4)  # every tour would be copied, none bred
    SOLVERS["genetic_algorithm"](cities, population_size=5, generations=2, elite=4)
