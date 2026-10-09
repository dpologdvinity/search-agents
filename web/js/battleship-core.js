// Battleship core: ship placement and the random fleet, ported from battleship/board.py.
//
// This file has no DOM code, so the node tests can run it. The rules match the server: a ship is a straight run
// of consecutive cells, horizontal or vertical, fully on the board. Ships may touch but may not overlap.

export const SIZE = 10;

/** Cells of a straight ship of `length` cells, starting at `start` and running right (or down if vertical). Null if it runs off the board. */
export function shipCells(start, length, vertical, size = SIZE) {
  const r0 = Math.floor(start / size), c0 = start % size;
  const cells = [];
  for (let k = 0; k < length; k++) {
    const r = r0 + (vertical ? k : 0), c = c0 + (vertical ? 0 : k);
    if (r >= size || c >= size) return null;
    cells.push(r * size + c);
  }
  return cells;
}

/**
 * A uniformly random legal fleet: one ship per entry of `lengths`, in that order, each a list of cell numbers.
 *
 * Each ship draws a start cell and an orientation uniformly. A draw that runs off the board is redrawn, so each
 * ship ends up uniform over its legal placements. If two ships overlap, the whole fleet is redrawn. Every legal
 * fleet then has the same probability, which is the uniform prior the probability model assumes. The server does
 * the same in random_fleet(). `random` returns floats in [0, 1); tests pass a seeded generator.
 */
export function randomFleet(lengths, random = Math.random, size = SIZE) {
  // A fleet that cannot fit would make the loops below redraw forever, so refuse it up front.
  if (lengths.some((n) => n < 1 || n > size) || lengths.reduce((a, b) => a + b, 0) > size * size) {
    throw new RangeError(`ship lengths must fit the ${size}x${size} board`);
  }
  for (;;) {
    const ships = lengths.map((length) => {
      for (;;) {
        const cells = shipCells(Math.floor(random() * size * size), length, random() < 0.5, size);
        if (cells) return cells;
      }
    });
    const all = ships.flat();
    if (new Set(all).size === all.length) return ships;
  }
}
