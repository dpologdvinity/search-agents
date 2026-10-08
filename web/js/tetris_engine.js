// Tetris engine for the page: the same rules, board features and placement search as tetris/ in Python,
// so the browser can play the AI with the committed weights and no server calls.
//
// Everything here is a pure function of its inputs (no DOM). A board is an array of H row bitmasks, row 0
// at the bottom, bit x = column x. The Python tests check that legalMoves() returns the same placements,
// lines and feature values as tetris/search.py for the same boards, so the two cannot drift apart.

export const W = 10;
export const H = 20;
export const FULL = (1 << W) - 1;
export const PIECES = 'IOTSZJL';
export const FEATURES = [
  'landing_height', 'eroded_cells', 'row_transitions', 'column_transitions',
  'holes', 'wells', 'aggregate_height', 'bumpiness', 'lines',
];
export const HAND_WEIGHTS = [-1, 1, -1, -1, -4, -1, 0, 0, 0];
export const GAME_OVER_SCORE = -1e9; // a candidate after which the preview piece has no placement

const SHAPES = {
  I: ['####'], O: ['##', '##'], T: [' # ', '###'], S: [' ##', '## '],
  Z: ['## ', ' ##'], J: ['#  ', '###'], L: ['  #', '###'],
};

// Popcount for non-negative 32-bit values (the masks here are at most 10 bits).
export function pc(n) {
  n = n - ((n >>> 1) & 0x55555555);
  n = (n & 0x33333333) + ((n >>> 2) & 0x33333333);
  return (((n + (n >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

// Cells are [x, y] pairs with y counting up from the bottom, as in Python.
const normalise = (cells) => {
  const mx = Math.min(...cells.map((c) => c[0]));
  const my = Math.min(...cells.map((c) => c[1]));
  return cells.map(([x, y]) => [x - mx, y - my]);
};
const rotateCW = (cells) => normalise(cells.map(([x, y]) => [y, -x]));
const cellKey = (cells) => cells.map(([x, y]) => `${x},${y}`).sort().join(';');

function makeRotation(cells) {
  const width = Math.max(...cells.map((c) => c[0])) + 1;
  const height = Math.max(...cells.map((c) => c[1])) + 1;
  const masks = [];
  for (let row = 0; row < height; row++) {
    let m = 0;
    for (const [x, y] of cells) if (y === row) m |= 1 << x;
    masks.push(m);
  }
  return { cells, width, height, masks };
}

// Rotation states per piece, built the same way as tetris/pieces.py: spawn shape first, duplicates dropped.
export const ROTATIONS = {};
for (const name of PIECES) {
  const rows = SHAPES[name];
  const cells = [];
  rows.forEach((row, i) => [...row].forEach((ch, x) => { if (ch === '#') cells.push([x, rows.length - 1 - i]); }));
  let cur = normalise(cells);
  const seen = [];
  const states = [];
  for (let k = 0; k < 4; k++) {
    const key = cellKey(cur);
    if (!seen.includes(key)) { seen.push(key); states.push(makeRotation(cur)); }
    cur = rotateCW(cur);
  }
  ROTATIONS[name] = states;
}

// Row transitions for every row pattern, with a filled wall on each side (see tetris/features.py).
function rowTransitions(mask) {
  const e = (mask << 1) | 1 | (1 << (W + 1));
  return pc((e ^ (e >> 1)) & ((1 << (W + 1)) - 1));
}
const ROW_TRANS = Array.from({ length: FULL + 1 }, (_, m) => rowTransitions(m));

export function fits(board, rot, px, py) {
  if (px < 0 || px + rot.width > W || py < 0) return false;
  for (let i = 0; i < rot.masks.length; i++) {
    const r = py + i;
    if (r >= H) continue; // above the ceiling
    if (board[r] & (rot.masks[i] << px)) return false;
  }
  return true;
}

// Bottom row where a piece dropped down column px comes to rest; null if the spawn spot is blocked.
export function dropRow(board, rot, px) {
  let y = H - rot.height;
  if (!fits(board, rot, px, y)) return null;
  while (y > 0 && fits(board, rot, px, y - 1)) y--;
  return y;
}

// Write a piece into the board and clear full rows: {board, lines, eroded, topped}.
export function lock(board, rot, px, py) {
  const rows = board.slice();
  let topped = false;
  rot.masks.forEach((m, i) => {
    const r = py + i;
    if (r >= H) { if (m) topped = true; return; }
    rows[r] |= m << px;
  });
  const kept = [];
  let lines = 0;
  for (const r of rows) { if (r === FULL) lines++; else kept.push(r); }
  let cleared = 0;
  if (lines) {
    rot.masks.forEach((m, i) => { if (py + i < H && rows[py + i] === FULL) cleared += pc(m); });
  }
  for (let i = 0; i < lines; i++) kept.push(0);
  return { board: kept, lines, eroded: lines * cleared, topped };
}

// The six board features from tetris/features.py: row transitions, column transitions, holes, wells,
// aggregate height, bumpiness. Only rows up to the highest filled cell are read.
export function boardFeatures(rows) {
  let top = 0;
  for (let r = H - 1; r >= 0; r--) if (rows[r]) { top = r + 1; break; }

  let rowT = 0;
  for (let r = 0; r < top; r++) rowT += ROW_TRANS[rows[r]];

  let colT = 0;
  let prev = FULL;
  for (let r = 0; r < top; r++) { colT += pc(prev ^ rows[r]); prev = rows[r]; }
  colT += pc(prev);

  let seen = 0;
  let holes = 0;
  let agg = 0;
  const heights = new Array(W).fill(0);
  for (let r = top - 1; r >= 0; r--) {
    const row = rows[r];
    holes += pc(seen & ~row & FULL);
    let fresh = row & ~seen;
    while (fresh) {
      const low = fresh & -fresh;
      heights[31 - Math.clz32(low)] = r + 1;
      fresh ^= low;
    }
    seen |= row;
    agg += pc(seen);
  }

  let bump = 0;
  for (let c = 0; c < W - 1; c++) bump += Math.abs(heights[c] - heights[c + 1]);

  let wells = 0;
  const run = new Array(W).fill(0);
  for (let r = 0; r < top; r++) {
    const row = rows[r];
    const left = ((row << 1) | 1) & FULL;
    const right = (row >> 1) | (1 << (W - 1));
    const well = ~row & left & right & FULL;
    for (let c = 0; c < W; c++) {
      if ((well >> c) & 1) { run[c] += 1; wells += run[c]; } else run[c] = 0;
    }
  }
  return [rowT, colT, holes, wells, agg, bump];
}

// The nine features in FEATURES order for a locked placement.
export function placementFeatures(rot, py, lines, eroded, newBoard) {
  const [rowT, colT, holes, wells, agg, bump] = boardFeatures(newBoard);
  return [py + (rot.height - 1) / 2, eroded, rowT, colT, holes, wells, agg, bump, lines];
}

export function dot(weights, feats) {
  let s = 0;
  for (let i = 0; i < weights.length; i++) s += weights[i] * feats[i];
  return s;
}

// Every resting placement of `piece`, in the same order as tetris/search.py (rotation, then column).
export function legalMoves(board, piece, weights) {
  const out = [];
  ROTATIONS[piece].forEach((rot, ri) => {
    for (let px = 0; px <= W - rot.width; px++) {
      const py = dropRow(board, rot, px);
      if (py === null) continue;
      const res = lock(board, rot, px, py);
      const feats = placementFeatures(rot, py, res.lines, res.eroded, res.board);
      const s = dot(weights, feats);
      out.push({ piece, rot: ri, x: px, y: py, lines: res.lines, features: feats, score: s, board: res.board, total: s });
    }
  });
  return out;
}

// The choice: {move, moves}. moves are the one-piece candidates; move is the best one, with its lookahead
// total when a preview is given (best one-piece score among the topK, plus the preview's best score).
export function bestMove(board, piece, weights, preview = null, topK = 6) {
  const moves = legalMoves(board, piece, weights);
  if (!moves.length) return { move: null, moves };
  if (preview === null) {
    let best = moves[0];
    for (const m of moves) if (m.score > best.score) best = m;
    return { move: best, moves };
  }
  const ranked = moves.slice().sort((a, b) => b.score - a.score).slice(0, topK);
  let best = null;
  for (const m of ranked) {
    const nxt = legalMoves(m.board, preview, weights);
    const bonus = nxt.length ? Math.max(...nxt.map((n) => n.score)) : GAME_OVER_SCORE;
    const total = m.score + bonus;
    if (best === null || total > best.total) best = { ...m, total };
  }
  return { move: best, moves };
}

// Seeded random numbers (mulberry32). The browser's games use this; they do not need to match Python's.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The 7-bag piece sequence, as in tetris/game.py.
export class Bag {
  constructor(seed) { this.rand = rng(seed); this.queue = []; }
  fill(n) {
    while (this.queue.length < n) {
      const bag = PIECES.split('');
      for (let i = bag.length - 1; i > 0; i--) {
        const j = Math.floor(this.rand() * (i + 1));
        [bag[i], bag[j]] = [bag[j], bag[i]];
      }
      this.queue.push(...bag);
    }
  }
  take() { this.fill(1); return this.queue.shift(); }
  peek() { this.fill(1); return this.queue[0]; }
}

// The game state for the page: falling piece, board, score. The AI does not use this for movement;
// it asks bestMove() for a target and the page animates the slide and drop.
export class Game {
  constructor(seed) {
    this.bag = new Bag(seed);
    this.board = new Array(H).fill(0);
    this.lines = 0;
    this.pieces = 0;
    this.over = false;
    this.spawn();
  }
  spawn() {
    this.piece = this.bag.take();
    this.rot = 0;
    this.pieces++;
    const r = this.shape();
    this.x = Math.floor((W - r.width) / 2);
    this.y = H - r.height;
    if (!fits(this.board, r, this.x, this.y)) this.over = true;
  }
  shape(k = this.rot) { return ROTATIONS[this.piece][k]; }
  move(dx) { if (fits(this.board, this.shape(), this.x + dx, this.y)) { this.x += dx; return true; } return false; }
  rotate() {
    const nxt = (this.rot + 1) % ROTATIONS[this.piece].length;
    for (const k of [0, -1, 1, -2, 2]) {
      if (fits(this.board, this.shape(nxt), this.x + k, this.y)) { this.rot = nxt; this.x += k; return true; }
    }
    return false;
  }
  softDrop() { if (fits(this.board, this.shape(), this.x, this.y - 1)) { this.y--; return true; } return false; }
  // Lock the falling piece where it is; returns the lock result and spawns the next piece.
  place(rot, x, y) {
    const res = lock(this.board, this.shape(rot), x, y);
    this.board = res.board;
    this.lines += res.lines;
    if (res.topped) this.over = true; else if (!this.over) this.spawn();
    return res;
  }
  hardDrop() {
    while (this.softDrop());
    return this.place(this.rot, this.x, this.y);
  }
}
