// Chance opponent for the endgame page: a line-for-line port of endgame/chance.py.
//
// Pure JavaScript with no DOM access, so the page and node (tests/test_endgame_parity.py) can both load it.
// A move is an object from /api/endgame/analyze: {mover, from, to, check, capture}. The position's white
// piece square is passed separately, because a capture is the lone king landing on that square.
//
// The odds are the same fixed table as CHANCE_WEIGHTS in endgame/chance.py: a move's category decides its
// weight, and a move is drawn with probability weight / (sum of the weights over the legal moves). Nothing
// here looks at what a move leads to; the tablebase is not consulted. pickAt takes a uniform number, so the
// page (seeded mulberry32) and the Python reference can be fed the same numbers and must choose the same move.

// The fixed odds. Why these numbers: see the docstring of CHANCE_WEIGHTS in endgame/chance.py.
export const CHANCE_WEIGHTS = { capture: 4, check: 2, king_centre: 2, other: 1 };

// How far a square is from the middle of the board, doubled so it stays an integer (1 at d4, e4, d5, e5).
// Squares are 8 * rank + file. Chebyshev distance: the king's own metric.
export function centreDistance(sq) {
  const f = sq & 7, r = sq >> 3;
  return Math.max(Math.abs(2 * f - 7), Math.abs(2 * r - 7));
}

// capture, check, king_centre or other: the first category that fits, in that order.
export function moveCategory(mv, wp) {
  if (mv.mover === 'k' && mv.to === wp) return 'capture';
  if (mv.check) return 'check';
  if ((mv.mover === 'K' || mv.mover === 'k') && centreDistance(mv.to) < centreDistance(mv.from)) return 'king_centre';
  return 'other';
}

// The table weight of each legal move, in the same order as moves.
export function weights(moves, wp) {
  return moves.map((mv) => CHANCE_WEIGHTS[moveCategory(mv, wp)]);
}

// The chance of each legal move: its weight over the total of the legal moves' weights. Sums to 1.
export function probabilities(moves, wp) {
  const ws = weights(moves, wp);
  const total = ws.reduce((a, b) => a + b, 0);
  return ws.map((w) => w / total);
}

// The move chosen by the uniform number u in [0, 1), by walking the cumulative weights.
export function pickAt(moves, wp, u) {
  if (!moves.length) return null;
  const ws = weights(moves, wp);
  const threshold = u * ws.reduce((a, b) => a + b, 0);
  let acc = 0;
  for (let i = 0; i < moves.length; i++) {
    acc += ws[i];
    if (threshold < acc) return moves[i];
  }
  return moves[moves.length - 1]; // only reachable through rounding when u is extremely close to 1
}

// A seeded uniform generator (mulberry32, the same one clusters-core.js uses). The same seed gives the same
// sequence, so a watched game can be repeated.
export function seededRandom(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
    t = (t ^ ((t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0)) >>> 0)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
