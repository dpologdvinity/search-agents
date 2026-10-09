// The battleship fleet generator (web/js/battleship-core.js): legality, and uniformity over legal fleets.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { randomFleet, shipCells } from '../../web/js/battleship-core.js';

// Seeded generator (mulberry32), so every run draws the same fleets.
function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Every straight ship of `length` on a size x size board, as cell lists (horizontal and vertical).
function placements(length, size) {
  const out = [];
  for (let start = 0; start < size * size; start++) {
    for (const vertical of [false, true]) {
      const cells = shipCells(start, length, vertical, size);
      if (cells) out.push(cells);
    }
  }
  return out;
}

// Upper critical value of a chi-square distribution (Wilson-Hilferty approximation), at z for p = 0.001.
function chiSquareCritical(df) {
  const z = 3.09;
  return df * (1 - 2 / (9 * df) + z * Math.sqrt(2 / (9 * df))) ** 3;
}

function isStraight(cells, size) {
  const sorted = [...cells].sort((a, b) => a - b);
  const sameRow = sorted.every((c) => Math.floor(c / size) === Math.floor(sorted[0] / size));
  const step = sameRow ? 1 : size;
  const sameCol = sameRow || sorted.every((c) => c % size === sorted[0] % size);
  return sameCol && sorted.every((c, i) => i === 0 || c - sorted[i - 1] === step);
}

test('a ship is a straight run that stays on the board', () => {
  assert.deepEqual(shipCells(0, 3, false, 10), [0, 1, 2]);
  assert.deepEqual(shipCells(0, 3, true, 10), [0, 10, 20]);
  assert.equal(shipCells(8, 2, false, 3), null, 'runs past the right edge');
  assert.equal(shipCells(6, 2, true, 3), null, 'runs past the bottom edge');
});

test('random fleets are legal: one straight ship per length, on the board, no overlaps', () => {
  const rng = mulberry32(7);
  const lengths = [5, 4, 3, 3, 2];
  for (let trial = 0; trial < 2000; trial++) {
    const fleet = randomFleet(lengths, rng, 10);
    assert.equal(fleet.length, lengths.length);
    const all = fleet.flat();
    assert.equal(new Set(all).size, all.length, 'ships overlap');
    fleet.forEach((cells, i) => {
      assert.equal(cells.length, lengths[i], 'ship length');
      assert.ok(cells.every((c) => c >= 0 && c < 100), 'on the board');
      assert.ok(isStraight(cells, 10), `ship ${i} is a straight run`);
    });
  }
});

test('fleets that cannot fit are refused instead of looping', () => {
  const rng = mulberry32(1);
  assert.throws(() => randomFleet([11], rng, 10), RangeError);
  assert.throws(() => randomFleet([2, 2, 2, 2, 2], rng, 3), RangeError, 'ten cells on a nine-cell board');
});

test('the 3x3 fleet with ships 2 and 2 is uniform over the legal fleets', () => {
  const size = 3;
  const twos = placements(2, size);
  // Every ordered pair of non-overlapping placements, one per ship. Ships may touch.
  const legal = new Set();
  let touching = 0;
  const adjacent = (c, d) => (Math.abs(c - d) === 1 && Math.floor(c / size) === Math.floor(d / size)) || Math.abs(c - d) === size;
  for (const a of twos) {
    for (const b of twos) {
      const all = [...a, ...b];
      if (new Set(all).size === all.length) {
        legal.add(`${a}|${b}`);
        if (a.some((c) => b.some((d) => adjacent(c, d)))) touching += 1;
      }
    }
  }
  assert.ok(touching > 0, 'the legal set includes fleets where ships touch');

  const rng = mulberry32(2026);
  const draws = 120000;
  const count = new Map();
  for (let i = 0; i < draws; i++) {
    const [a, b] = randomFleet([2, 2], rng, size);
    const key = `${a}|${b}`;
    count.set(key, (count.get(key) || 0) + 1);
  }

  for (const key of count.keys()) assert.ok(legal.has(key), `illegal fleet drawn: ${key}`);
  assert.equal(count.size, legal.size, 'every legal fleet is drawn');

  // Chi-square goodness of fit against the uniform distribution over the legal fleets.
  const expected = draws / legal.size;
  let chi2 = 0;
  for (const key of legal) {
    const observed = count.get(key) || 0;
    chi2 += (observed - expected) ** 2 / expected;
  }
  const df = legal.size - 1;
  assert.ok(chi2 < chiSquareCritical(df), `chi-square ${chi2.toFixed(1)} over ${df} df`);
});
