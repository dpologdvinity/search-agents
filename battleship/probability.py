"""Bayesian probability targeting: how likely each cell is to hold a ship, given what has been seen.

A *configuration* is one way to place the ships still afloat that agrees with everything observed:
  - no ship covers a miss or a cell of a sunk ship (those are known to be empty or taken),
  - every open hit (a hit on a ship not yet sunk) is covered by some remaining ship,
  - ships do not overlap.
With a uniform prior over fleet positions, every consistent configuration is equally likely, so

    P(cell holds a ship | observations) = (configurations using the cell) / (all configurations).

The agent fires at the unknown cell with the highest probability. Open hits make their neighbours rise
automatically, because a ship through a hit must cover one of the hit's neighbours. The probabilities of
all unknown cells sum to the number of ship cells still afloat, minus the open hits (which are certain).

Counting is exact where it is cheap. A depth-first search places the remaining ships one at a time, skipping
any branch where the open hits left uncovered need more cells than the ships still to be placed can supply.
Early in a game there are billions of configurations, so the search gives up after EXACT_BUDGET candidate
checks and the probabilities are estimated by importance sampling instead:

  1. Place ships one at a time, each drawn uniformly from the placements that fit. Multiply a running weight
     by the number of options at each step.
  2. Discard samples that leave an open hit uncovered (weight 0).
  3. Each sample's probability of being drawn is 1 / (product of option counts), so the weight makes the
     weighted samples an unbiased stand-in for a uniform draw over consistent configurations.
  4. P(cell) is the weighted fraction of samples that cover it.

All samples are drawn together with numpy. A 100-cell set is two 64-bit words, so "does this placement overlap
what is already placed" is a few bitwise operations over a samples-by-placements array.
"""

from __future__ import annotations

import random
from typing import NamedTuple

import numpy as np

from .board import CELLS, FLEET, PLACEMENT_MASKS, Knowledge, bits

# Samples drawn when exact counting is too large. The cost is samples x placements per ship, a few tens of
# milliseconds per shot at this size; the error on a cell's probability is about 0.02-0.03.
SAMPLES = 1000
# Work the exact search may do, in candidate checks. Exact counting is only attempted when the placements the
# ships could take, ignoring overlaps, fit in half this budget, so an attempt never runs out part-way.
EXACT_BUDGET = 60_000


class Belief(NamedTuple):
    """Ship probabilities for all 100 cells, and how they were computed.

    probs:  P(ship) per cell, row by row. Known hits and sunk cells are 1.0, misses 0.0.
    method: "exact" (counted every configuration), "sampled" (importance sampling), "done" (no ships
            afloat), or "none" (no consistent placement was found; only possible with inconsistent input).
    count:  configurations counted (exact), or samples drawn (sampled).
    """

    probs: tuple[float, ...]
    method: str
    count: int


class _OverBudget(Exception):
    """Raised inside the exact search when it has done more work than the budget allows."""


_WORD = (1 << 64) - 1


def _words(mask: int) -> tuple[np.uint64, np.uint64]:
    """A 100-cell bitmask as two 64-bit words (cells 0-63, then 64-99), for vectorised bit tests."""
    return np.uint64(mask & _WORD), np.uint64(mask >> 64)


def _membership(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Turn rows of two-word bitmasks into (rows, 100) 0/1 matrices: entry [i, c] is 1 if cell c is in row i.

    The words are unpacked little-endian, so bit 0 of the low word comes first.
    """
    lo_bits = np.unpackbits(lo.astype("<u8").view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")
    hi_bits = np.unpackbits(hi.astype("<u8").view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")
    return np.concatenate([lo_bits, hi_bits[:, :CELLS - 64]], axis=1)


# Per ship length: the low and high words of every placement, as arrays indexed by placement number.
_LO = {n: np.array([m & _WORD for m in masks], dtype=np.uint64) for n, masks in PLACEMENT_MASKS.items()}
_HI = {n: np.array([m >> 64 for m in masks], dtype=np.uint64) for n, masks in PLACEMENT_MASKS.items()}


def _free_placements(blocked: int) -> dict[int, np.ndarray]:
    """For each ship length, the indices of placements that avoid the blocked cells (misses, sunk ships)."""
    lo, hi = _words(blocked)
    return {n: np.flatnonzero(((_LO[n] & lo) | (_HI[n] & hi)) == 0) for n in set(FLEET)}


def _count_exact(lengths, free: dict[int, np.ndarray], hits: int, budget: int) -> dict[int, int]:
    """Count consistent configurations exactly.

    Returns {union bitmask: number of placements with exactly that union}. Two ships of the same length can
    be placed in either order, so each set of ships is counted twice; that factor is the same for every
    configuration, so probabilities are unaffected.
    """
    options_of = {n: [PLACEMENT_MASKS[n][p] for p in free[n]] for n in set(lengths)}
    # tail[i]: total length of ships i.. still to place, the most cells they can cover.
    tail = [sum(lengths[i:]) for i in range(len(lengths) + 1)]
    counts: dict[int, int] = {}
    work = 0

    def place(i: int, used: int) -> None:
        nonlocal work
        if i == len(lengths):
            if not hits & ~used:  # every open hit is covered
                counts[used] = counts.get(used, 0) + 1
            return
        # Prune: each open hit still uncovered needs a ship cell, and the ships left supply tail[i] cells.
        if (hits & ~used).bit_count() > tail[i]:
            return
        options = options_of[lengths[i]]
        work += len(options)
        if work > budget:
            raise _OverBudget
        for m in options:
            if not m & used:
                place(i + 1, used | m)

    place(0, 0)
    return counts


def _sample(lengths, free: dict[int, np.ndarray], hits: int, rng: random.Random,
            samples: int) -> np.ndarray | None:
    """Estimate the per-cell probabilities by importance sampling (see the module docstring).

    Returns a length-100 array of probabilities, or None if no sample satisfied the open hits.
    """
    gen = np.random.default_rng(rng.getrandbits(64))  # numpy generator, seeded from the agent's generator
    used_lo = np.zeros(samples, dtype=np.uint64)  # cells covered by the ships placed so far, two words
    used_hi = np.zeros(samples, dtype=np.uint64)
    weight = np.ones(samples)
    for n in lengths:
        lo, hi = _LO[n][free[n]], _HI[n][free[n]]  # placements of this length that avoid misses and sunk ships
        overlap = ((used_lo[:, None] & lo[None, :]) | (used_hi[:, None] & hi[None, :])) != 0
        valid = ~overlap  # (samples, placements)
        count = valid.sum(axis=1)
        weight *= count  # 0 when no placement fits: that sample is inconsistent
        # Uniform choice among the valid placements: give each valid one a random key and take the largest.
        # Invalid placements get -1, so they never win; a sample with no valid placement picks garbage, but its
        # weight is already 0.
        keys = np.where(valid, gen.random((samples, len(lo)), dtype=np.float32), np.float32(-1.0))
        pick = keys.argmax(axis=1)
        used_lo |= lo[pick]
        used_hi |= hi[pick]
    keep = weight > 0
    if hits:
        hl, hh = _words(hits)
        keep &= ((used_lo & hl) == hl) & ((used_hi & hh) == hh)  # every open hit covered
    w = weight * keep
    total = w.sum()
    if total == 0.0:
        return None
    # Each sample's weight goes to every cell that sample covers.
    return (w @ _membership(used_lo, used_hi).astype(np.float64)) / total


def ship_probabilities(k: Knowledge, rng: random.Random | None = None, samples: int = SAMPLES,
                       budget: int = EXACT_BUDGET) -> Belief:
    """P(ship) for every cell, given what the shooter knows. See the module docstring for the model."""
    probs = [0.0] * CELLS
    for c in bits(k.hit):  # hits (sunk ships included) are certainly ship cells
        probs[c] = 1.0
    lengths = k.remaining_lengths()
    if not lengths:
        return Belief(tuple(probs), "done", 0)

    free = _free_placements(k.miss | k.sunk)
    if any(len(free[n]) == 0 for n in lengths):
        return Belief(tuple(probs), "none", 0)
    hits = k.open_hits

    # Ships ignoring overlaps: the most configurations there could be. The exact search does at most about
    # twice this much work, so when it fits in half the budget it is guaranteed to finish.
    bound = 1
    for n in lengths:
        bound *= len(free[n])
    counts = None
    if bound <= budget // 2:
        try:
            counts = _count_exact(lengths, free, hits, budget)
        except _OverBudget:  # the bound is loose, so this can still happen; then sample
            counts = None

    if counts is not None:
        total = sum(counts.values())
        if total == 0:
            return Belief(tuple(probs), "none", 0)
        unions = list(counts)
        lo = np.array([u & _WORD for u in unions], dtype=np.uint64)
        hi = np.array([u >> 64 for u in unions], dtype=np.uint64)
        n = np.array([counts[u] for u in unions], dtype=np.float64)
        weight_on = (n @ _membership(lo, hi).astype(np.float64)) / total
        for c in range(CELLS):
            if weight_on[c]:
                probs[c] = float(weight_on[c])
        return Belief(tuple(probs), "exact", total)

    estimate = _sample(lengths, free, hits, rng or random.Random(), samples)
    if estimate is None:
        return Belief(tuple(probs), "none", 0)
    for c in range(CELLS):
        if not (k.hit >> c) & 1:  # hits keep their certain value of 1.0
            probs[c] = float(estimate[c])
    return Belief(tuple(probs), "sampled", samples)


def choose_cell(belief: Belief, unknown: list[int], rng: random.Random) -> int:
    """The unknown cell with the highest probability. Ties (common in exact counts) are broken at random.

    If the belief is "none" there is nothing to go on, so the choice is uniform over the unknown cells.
    """
    if belief.method == "none" or not unknown:
        return rng.choice(unknown)
    best = max(belief.probs[c] for c in unknown)
    ties = [c for c in unknown if belief.probs[c] >= best - 1e-12]
    return rng.choice(ties)
