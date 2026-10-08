"""Battleship rules: placements, the fleet, shots, and what a shooter has learned.

Cells are numbered 0-99, row by row: cell = row * 10 + col. Players write them as a column letter and a
row number, so cell 62 is "C7" (column C, row 7). Sets of cells are stored as Python ints used as bitmasks
(bit i set means cell i is in the set): intersections and unions are then single operations, which keeps the
probability model fast.

Rules: the fleet is one ship of each length in FLEET (5, 4, 3, 3, 2). Ships are straight lines of consecutive
cells, horizontal or vertical, fully on the board, and may not overlap. Ships may touch. A shot is a miss or a
hit; the shot that hits a ship's last open cell sinks it, and the shooter then learns every cell of that ship.
The game is won by sinking the whole fleet.
"""

from __future__ import annotations

import random
from functools import reduce
from typing import NamedTuple

SIZE = 10
CELLS = SIZE * SIZE
FLEET = (5, 4, 3, 3, 2)
FLEET_NAMES = ("Carrier", "Battleship", "Cruiser", "Submarine", "Destroyer")
COLUMNS = "ABCDEFGHIJ"


def mask_of(cells) -> int:
    """Bitmask with one bit per cell."""
    return reduce(lambda m, c: m | (1 << c), cells, 0)


def bits(mask: int):
    """Yield the cell numbers set in a bitmask, lowest first."""
    while mask:
        low = mask & -mask  # lowest set bit
        yield low.bit_length() - 1
        mask ^= low


def cell_name(cell: int) -> str:
    """Cell 62 -> 'C7'."""
    return f"{COLUMNS[cell % SIZE]}{cell // SIZE + 1}"


def parse_cell(text: str) -> int:
    """'c7' or 'C7' -> 62. Raises ValueError for anything that is not a cell on the board."""
    text = str(text).strip().upper()
    if len(text) < 2 or len(text) > 3 or text[0] not in COLUMNS or not text[1:].isdigit():
        raise ValueError(f"not a cell: {text!r} (use a column letter A-J and a row 1-10, e.g. C7)")
    row = int(text[1:])
    if not 1 <= row <= SIZE:
        raise ValueError(f"row must be 1-{SIZE}: {text!r}")
    return (row - 1) * SIZE + COLUMNS.index(text[0])


def _placements(length: int) -> tuple[tuple[int, ...], ...]:
    """Every straight run of `length` cells on the board, horizontal first, as tuples of cell numbers."""
    runs = []
    for r in range(SIZE):
        for c in range(SIZE - length + 1):
            runs.append(tuple(r * SIZE + c + k for k in range(length)))
    for c in range(SIZE):
        for r in range(SIZE - length + 1):
            runs.append(tuple((r + k) * SIZE + c for k in range(length)))
    return tuple(runs)


# Precomputed once: all placements per ship length, as cell tuples and as bitmasks.
PLACEMENTS = {n: _placements(n) for n in set(FLEET)}
PLACEMENT_MASKS = {n: tuple(mask_of(run) for run in runs) for n, runs in PLACEMENTS.items()}


def is_ship(cells, length: int) -> bool:
    """True if cells are `length` consecutive cells on one row or one column of the board."""
    cells = list(cells)
    if len(cells) != length or any(not 0 <= c < CELLS for c in cells) or len(set(cells)) != length:
        return False
    cells.sort()
    same_row = all(c // SIZE == cells[0] // SIZE for c in cells)
    step = 1 if same_row else SIZE
    same_line = same_row or all(c % SIZE == cells[0] % SIZE for c in cells)
    return same_line and all(b - a == step for a, b in zip(cells, cells[1:]))


def check_fleet(ships) -> bool:
    """True if ships are a legal fleet: one straight ship per length in FLEET, no overlaps."""
    ships = [tuple(s) for s in ships]
    if sorted(len(s) for s in ships) != sorted(FLEET):
        return False
    if not all(is_ship(s, len(s)) for s in ships):
        return False
    # Overlapping ships would share cells and leave fewer distinct cells than the fleet has.
    return mask_of(c for s in ships for c in s).bit_count() == sum(FLEET)


def random_fleet(rng: random.Random) -> list[tuple[int, ...]]:
    """A uniformly random legal fleet, in FLEET order.

    Each ship is drawn uniformly from all its placements, and the whole draw is repeated until no two ships
    overlap. Each legal fleet is then equally likely, which is exactly the prior the probability model assumes.
    About a third of draws are legal, so this takes about three tries.
    """
    while True:
        ships = [rng.choice(PLACEMENTS[n]) for n in FLEET]
        if mask_of(c for s in ships for c in s).bit_count() == sum(FLEET):
            return ships


class Shot(NamedTuple):
    """Result of one shot. ship_index and ship are set only when the shot sinks a ship."""

    cell: int
    result: str  # "miss", "hit", or "sunk"
    ship_index: int = -1
    ship: tuple[int, ...] = ()


class Fleet:
    """A hidden fleet that answers shots. Used for both sides; neither side sees the other's fleet."""

    def __init__(self, ships):
        if not check_fleet(ships):
            raise ValueError("not a legal fleet")
        self.ships = [tuple(s) for s in ships]
        self._owner = {c: k for k, s in enumerate(self.ships) for c in s}
        self._open = [len(s) for s in self.ships]  # cells of each ship not yet hit
        self.fired: set[int] = set()
        self.sunk_count = 0

    def fire(self, cell: int) -> Shot:
        """Answer a shot at cell. Firing at the same cell twice is an error."""
        if not 0 <= cell < CELLS:
            raise ValueError(f"cell {cell} is off the board")
        if cell in self.fired:
            raise ValueError(f"{cell_name(cell)} has already been fired at")
        self.fired.add(cell)
        k = self._owner.get(cell)
        if k is None:
            return Shot(cell, "miss")
        self._open[k] -= 1
        if self._open[k] == 0:
            self.sunk_count += 1
            return Shot(cell, "sunk", k, self.ships[k])
        return Shot(cell, "hit")

    @property
    def all_sunk(self) -> bool:
        return self.sunk_count == len(self.ships)

    def afloat(self) -> int:
        """Number of ships not yet sunk."""
        return len(self.ships) - self.sunk_count


class Knowledge:
    """Everything a shooter has learned about the enemy fleet.

    That is its misses, its hits, and for each sunk ship its full cell list. This is the only input the agents
    and the probability model see, so the same class serves the agents, the server, and the terminal game.
    """

    def __init__(self):
        self.fired = 0  # bitmask of every cell fired at
        self.miss = 0
        self.hit = 0  # every hit, including the cells of sunk ships
        self.sunk = 0  # cells of sunk ships
        self.sunk_lengths: list[int] = []

    def observe(self, cell: int, result: str, ship=()) -> None:
        """Record one shot and its answer. Reports must come in the order they happened.

        A "sunk" report must name the sunk ship's cells, and every one of them must already be a hit (the
        earlier shots on that ship), so the rules can be checked on untrusted input too.
        """
        # Validate everything before changing any state, so a rejected report leaves the knowledge untouched.
        if not 0 <= cell < CELLS:
            raise ValueError(f"cell {cell} is off the board")
        bit = 1 << cell
        if self.fired & bit:
            raise ValueError(f"{cell_name(cell)} was fired at twice")
        if result not in ("miss", "hit", "sunk"):
            raise ValueError(f"unknown result {result!r}")
        if result != "sunk":
            if result == "miss":
                self.miss |= bit
            else:
                self.hit |= bit
            self.fired |= bit
            return
        ship = tuple(ship)
        if not is_ship(ship, len(ship)) or cell not in ship or len(ship) not in FLEET:
            raise ValueError(f"{cell_name(cell)} is not inside a valid ship of a fleet length")
        ship_mask = mask_of(ship)
        if ship_mask & ~(self.hit | bit):
            raise ValueError("a sunk ship has cells that were never hit")
        if self.sunk & ship_mask:
            raise ValueError("ships overlap")
        if self.sunk_lengths.count(len(ship)) >= FLEET.count(len(ship)):
            raise ValueError(f"too many sunk ships of length {len(ship)}")
        self.fired |= bit
        self.hit |= bit | ship_mask
        self.sunk |= ship_mask
        self.sunk_lengths.append(len(ship))

    @property
    def open_hits(self) -> int:
        """Hits on ships not yet sunk. Each one is certainly part of a ship still afloat."""
        return self.hit & ~self.sunk

    def remaining_lengths(self) -> list[int]:
        """Lengths of the ships still afloat, longest first."""
        rest = list(FLEET)
        for n in self.sunk_lengths:
            rest.remove(n)
        return sorted(rest, reverse=True)

    def unknown_cells(self) -> list[int]:
        """Cells not fired at yet."""
        return [c for c in range(CELLS) if not (self.fired >> c) & 1]
