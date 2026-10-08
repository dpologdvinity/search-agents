"""Traveling salesman solvers: find a short closed tour through every city.

A tour is a list of city indices visiting each city exactly once and
returning to the start. Its length is the sum of straight-line distances
between consecutive cities, including the edge from the last city back to
the first.

Four solvers, from simplest to cleverest:

  nearest_neighbor_2opt  Greedy construction, then repeated 2-opt fixes until
                         no single fix helps (a local optimum). Fast baseline.
  simulated_annealing    Random 2-opt changes, sometimes accepting worse tours
                         so the search can climb out of local optima. The
                         chance of accepting a worse tour shrinks as the
                         "temperature" cools. Finishes with a 2-opt descent.
  genetic_algorithm      Evolves a population of tours: pick good parents,
                         combine them with order crossover, mutate, repeat.
  held_karp              Exact dynamic programming, O(n^2 * 2^n). Only for
                         small instances; used to measure how far the
                         heuristics are from the true optimum.

Each heuristic records "frames": snapshots of its progress that the web page
animates. Cities are (x, y) points; the page uses coordinates in [0, 1].
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

MAX_FRAMES = 150  # snapshots kept per run, enough for a smooth animation


@dataclass
class Result:
    """A finished run: the best tour found and how the search got there."""

    algorithm: str
    tour: list[int]
    length: float
    steps: int  # iterations (annealing), generations (genetic), or improvements (2-opt)
    frames: list[dict] = field(default_factory=list)


# ── Basics ────────────────────────────────────────────────────────────────


def distance_matrix(cities: list[tuple[float, float]]) -> list[list[float]]:
    """dist[i][j] = straight-line distance between cities i and j.

    Computed once up front so every later length calculation is a lookup.
    """
    return [[math.dist(a, b) for b in cities] for a in cities]


def tour_length(tour: list[int], dist) -> float:
    """Total length of the closed tour, including the edge back to the start."""
    return sum(dist[tour[k - 1]][tour[k]] for k in range(len(tour)))


def two_opt_delta(tour: list[int], i: int, j: int, dist) -> float:
    """Change in length if the segment tour[i..j] is reversed (a "2-opt move").

    Reversing a segment removes two edges, (a, b) and (c, d), and adds two
    new ones, (a, c) and (b, d). Every edge inside the segment is kept, just
    walked backwards, so only these four edges matter. That makes the change
    O(1) to compute instead of re-measuring the whole tour.

        before:  a -> b ... c -> d
        after:   a -> c ... b -> d
    """
    n = len(tour)
    a, b = tour[i - 1], tour[i]
    c, d = tour[j], tour[(j + 1) % n]
    return dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]


def reverse_segment(tour: list[int], i: int, j: int) -> None:
    """Apply a 2-opt move in place by reversing tour[i..j]."""
    tour[i:j + 1] = reversed(tour[i:j + 1])


class _Recorder:
    """Keeps at most MAX_FRAMES evenly spaced snapshots of a run."""

    def __init__(self, total_steps: int):
        self.every = max(1, total_steps // MAX_FRAMES)
        self.frames: list[dict] = []

    def record(self, step: int, best_tour, best_len: float, **extra) -> None:
        if step % self.every == 0:
            self.frames.append({"step": step, "best_length": round(best_len, 5), "tour": list(best_tour), **extra})


# ── Baseline: nearest neighbour + 2-opt ──────────────────────────────────


def nearest_neighbor(dist, start: int = 0) -> list[int]:
    """Greedy tour: from the current city, always go to the closest unvisited one.

    Quick and usually within about 25% of optimal, but it tends to leave a
    few long edges at the end when the remaining cities are far apart.
    """
    n = len(dist)
    tour, unvisited = [start], set(range(n)) - {start}
    while unvisited:
        here = tour[-1]
        nearest = min(unvisited, key=lambda city: dist[here][city])
        tour.append(nearest)
        unvisited.remove(nearest)
    return tour


def two_opt_descent(tour: list[int], dist, max_passes: int = 50, on_improve=None) -> int:
    """Apply improving 2-opt moves in place until none is left; return how many were made.

    2-opt removes crossings: whenever two edges cross, reversing the segment
    between them makes the tour shorter. Stopping when no move helps leaves
    a local optimum: a tour no single 2-opt move can improve, though not
    necessarily the best tour overall. `on_improve(count, tour, length)` is
    called after each improvement, for recording progress.
    """
    n = len(tour)
    length = tour_length(tour, dist)
    improvements = 0
    for _ in range(max_passes):
        improved = False
        # Try every pair of edges; take any reversal that shortens the tour.
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                delta = two_opt_delta(tour, i, j, dist)
                if delta < -1e-12:
                    reverse_segment(tour, i, j)
                    length += delta
                    improvements += 1
                    improved = True
                    if on_improve:
                        on_improve(improvements, tour, length)
        if not improved:
            break  # local optimum reached
    return improvements


def nearest_neighbor_2opt(cities, max_passes: int = 50) -> Result:
    """Nearest-neighbour tour, then 2-opt until no reversal shortens it (see two_opt_descent)."""
    dist = distance_matrix(cities)
    tour = nearest_neighbor(dist)
    length = tour_length(tour, dist)
    frames = [{"step": 0, "best_length": round(length, 5), "tour": list(tour)}]
    improvements = two_opt_descent(tour, dist, max_passes, on_improve=lambda count, t, ln: frames.append(
        {"step": count, "best_length": round(ln, 5), "tour": list(t)}))
    # Thin the frames so long runs stay a reasonable size.
    stride = max(1, len(frames) // MAX_FRAMES)
    frames = frames[::stride] + ([frames[-1]] if (len(frames) - 1) % stride else [])
    return Result("nearest_neighbor_2opt", tour, tour_length(tour, dist), improvements, frames)


# ── Simulated annealing ─────────────────────────────────────────────────


def initial_temperature(tour, dist, rng: random.Random, samples: int = 200, accept: float = 0.8) -> float:
    """Pick a starting temperature that accepts about 80% of uphill moves.

    A worse move with cost increase `delta` is accepted with probability
    exp(-delta / T). Solving exp(-avg_delta / T) = accept for T, using the
    average uphill delta of some random moves, gives a starting point that
    adapts to the map's scale instead of a magic constant.
    """
    n = len(tour)
    uphill = []
    for _ in range(samples):
        i, j = sorted(rng.sample(range(1, n), 2))
        delta = two_opt_delta(tour, i, j, dist)
        if delta > 0:
            uphill.append(delta)
    avg = sum(uphill) / len(uphill) if uphill else 1.0
    return -avg / math.log(accept)


def simulated_annealing(cities, iterations: int = 60_000, cooling: float | None = None,
                        start_temp: float | None = None, seed: int | None = None) -> Result:
    """Simulated annealing over 2-opt moves, starting from a random tour.

    Each iteration proposes reversing a random segment:
      - if it shortens the tour, always accept it;
      - if it lengthens the tour by delta, accept it with probability
        exp(-delta / T).
    Early on T is high, so the search wanders freely and can escape local
    optima. T shrinks geometrically each step (T <- T * cooling), so the
    search gradually becomes greedy and settles into a good tour. The best
    tour ever seen is kept, since the current tour may have wandered off.

    `cooling` defaults to a rate that takes T from its start to about 0.1%
    of it over the run.
    """
    rng = random.Random(seed)
    dist = distance_matrix(cities)
    n = len(cities)
    tour = list(range(n))
    rng.shuffle(tour)  # start random so the animation shows the untangling
    length = tour_length(tour, dist)
    best, best_len = list(tour), length

    temp = start_temp if start_temp is not None else initial_temperature(tour, dist, rng)
    rate = cooling if cooling is not None else 0.001 ** (1 / iterations)  # reach 0.1% of the start temperature
    rec = _Recorder(iterations)
    accepted_uphill = 0

    for step in range(iterations):
        i, j = sorted(rng.sample(range(1, n), 2)) if n > 3 else (1, n - 1)
        delta = two_opt_delta(tour, i, j, dist)
        # The Metropolis rule: always go downhill, sometimes go uphill.
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            reverse_segment(tour, i, j)
            length += delta
            if delta > 0:
                accepted_uphill += 1
            if length < best_len - 1e-12:
                best, best_len = list(tour), length
        rec.record(step, best, best_len, current_length=round(length, 5), temperature=temp,
                   uphill_accepted=accepted_uphill)
        temp *= rate

    # Finishing pass: random sampling can miss the last few improving moves as
    # the search freezes, so polish the best tour with a 2-opt descent. This
    # guarantees the result has no crossings.
    if two_opt_descent(best, dist):
        best_len = tour_length(best, dist)
        last = rec.frames[-1]
        rec.frames.append({**last, "step": iterations, "best_length": round(best_len, 5), "tour": list(best)})

    return Result("simulated_annealing", best, tour_length(best, dist), iterations, rec.frames)


# ── Genetic algorithm ───────────────────────────────────────────────────


def order_crossover(parent_a: list[int], parent_b: list[int], rng: random.Random) -> list[int]:
    """Order crossover (OX1): build a child tour from two parents.

    A plain crossover would duplicate or drop cities, so OX1 keeps the child
    a valid permutation:
      1. copy a random slice from parent A, in place;
      2. fill the remaining positions with the missing cities in the order
         they appear in parent B, starting after the slice and wrapping.

        A:  1 2 |3 4 5| 6 7        slice 3 4 5 kept from A
        B:  5 7 1 6 2 4 3          remaining order from B: 7 1 6 2
        child: 6 2 |3 4 5| 7 1     filled starting after the slice
    """
    n = len(parent_a)
    i, j = sorted(rng.sample(range(n), 2))
    child = [-1] * n  # -1 marks a position not filled yet
    child[i:j + 1] = parent_a[i:j + 1]
    kept = set(child[i:j + 1])
    fill = [city for k in range(n) if (city := parent_b[(j + 1 + k) % n]) not in kept]
    for k, city in enumerate(fill):
        child[(j + 1 + k) % n] = city
    return child


def inversion_mutation(tour: list[int], rng: random.Random) -> None:
    """Reverse a random segment in place (the same move 2-opt uses)."""
    i, j = sorted(rng.sample(range(len(tour)), 2))
    reverse_segment(tour, i, j)


def tournament(population, lengths, rng: random.Random, size: int = 3) -> list[int]:
    """Pick `size` random tours and return the shortest.

    Selection pressure comes from the tournament size: bigger tournaments
    favour the best tours more strongly, smaller ones keep more diversity.
    """
    contenders = rng.sample(range(len(population)), size)
    return population[min(contenders, key=lambda k: lengths[k])]


def genetic_algorithm(cities, population_size: int = 120, generations: int = 300,
                      mutation_rate: float = 0.3, elite: int = 4, seed: int | None = None) -> Result:
    """Evolve tours: selection, order crossover, inversion mutation, elitism.

    Each generation:
      1. the `elite` shortest tours are copied unchanged, so the best tour
         found is never lost;
      2. the rest of the next generation is bred: two parents chosen by
         tournament selection, combined with order crossover, and mutated
         with probability `mutation_rate`.
    Crossover passes on good sub-routes from both parents; mutation adds the
    variety needed to discover new ones.
    """
    rng = random.Random(seed)
    dist = distance_matrix(cities)
    n = len(cities)
    population = []
    for _ in range(population_size):
        tour = list(range(n))
        rng.shuffle(tour)
        population.append(tour)

    rec = _Recorder(generations)
    best: list[int] = list(population[0])
    best_len = math.inf
    for gen in range(generations):
        lengths = [tour_length(t, dist) for t in population]
        ranked = sorted(range(population_size), key=lambda k: lengths[k])
        if lengths[ranked[0]] < best_len:
            best, best_len = list(population[ranked[0]]), lengths[ranked[0]]
        rec.record(gen, best, best_len, mean_length=round(sum(lengths) / population_size, 5))

        next_gen = [list(population[k]) for k in ranked[:elite]]
        while len(next_gen) < population_size:
            child = order_crossover(tournament(population, lengths, rng), tournament(population, lengths, rng), rng)
            if rng.random() < mutation_rate:
                inversion_mutation(child, rng)
            next_gen.append(child)
        population = next_gen

    return Result("genetic_algorithm", best, best_len, generations, rec.frames)


# ── Exact solution for small instances ──────────────────────────────────

HELD_KARP_LIMIT = 12


def held_karp(cities) -> Result:
    """Exact shortest tour by dynamic programming over subsets (Held-Karp).

    cost[(S, j)] = length of the shortest path that starts at city 0, visits
    every city in the set S exactly once, and ends at city j. Paths are built
    up from smaller sets:

        cost[(S, j)] = min over k in S - {j} of cost[(S - {j}, k)] + dist[k][j]

    The best tour closes the path back to city 0. Sets are stored as
    bitmasks. Time O(n^2 * 2^n), memory O(n * 2^n): far better than trying
    all (n-1)! orders, but still only practical for a dozen or so cities.
    """
    n = len(cities)
    if n > HELD_KARP_LIMIT:
        raise ValueError(f"held_karp is limited to {HELD_KARP_LIMIT} cities")
    dist = distance_matrix(cities)
    if n <= 2:
        tour = list(range(n))
        return Result("held_karp", tour, tour_length(tour, dist), 0)

    # Bit (k - 1) of a mask stands for city k; city 0 is the fixed start.
    cost = {(1 << (j - 1), j): (dist[0][j], 0) for j in range(1, n)}
    for size in range(2, n):
        for mask in range(1, 1 << (n - 1)):
            if mask.bit_count() != size:
                continue
            for j in range(1, n):
                bit = 1 << (j - 1)
                if not mask & bit:
                    continue
                rest = mask ^ bit
                cost[(mask, j)] = min(
                    (cost[(rest, k)][0] + dist[k][j], k)
                    for k in range(1, n) if rest & (1 << (k - 1))
                )

    full = (1 << (n - 1)) - 1
    length, last = min((cost[(full, j)][0] + dist[j][0], j) for j in range(1, n))
    # Walk the stored predecessors back from the last city to rebuild the tour.
    tour, mask, j = [], full, last
    while j:
        tour.append(j)
        mask, j = mask ^ (1 << (j - 1)), cost[(mask, j)][1]
    tour.append(0)
    tour.reverse()
    return Result("held_karp", tour, length, 0)


SOLVERS = {
    "nearest_neighbor_2opt": nearest_neighbor_2opt,
    "simulated_annealing": simulated_annealing,
    "genetic_algorithm": genetic_algorithm,
}
