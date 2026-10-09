// Snake core: the rules, the senses, and the network's forward pass, with no DOM.
//
// This file mirrors snake/board.py, snake/net.py, snake/evaluator.py, and the baselines in snake/agents.py.
// The evolved net is fetched as JSON and run here with the same arithmetic as NumPy: the features, one
// tanh layer, then the argmax over three outputs. The evaluation function (the eight weights from
// snake/cem.py) is the same kind of arithmetic over move features. Keeping it DOM-free lets the Node
// parity tests (tests/test_snake_eval_parity.py and the notes) compare it with Python step for step.

// Headings clockwise from north. Turning right adds one; turning left subtracts one.
export const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
export const LEFT = 0, STRAIGHT = 1, RIGHT = 2;
export const ACTION_NAMES = ['left', 'straight', 'right'];
export const HEADING_NAMES = ['north', 'east', 'south', 'west'];
export const INPUT_NAMES = [
  'danger ahead', 'danger left', 'danger right',
  'free ahead', 'free left', 'free right',
  'food forward', 'food right',
  'tail forward', 'tail right',
  'heading N', 'heading E', 'heading S', 'heading W',
  'room ahead', 'room left', 'room right',
];
// Room counts stop at this many cells, as in snake/net.py (FLOOD_CAP).
export const FLOOD_CAP = 30;

// A seeded RNG (mulberry32) so a game can be replayed from its seed in tests and the page.
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

const turn = (heading, action) => (heading + (action === LEFT ? -1 : action === RIGHT ? 1 : 0) + 4) % 4;

function placeFood(game) {
  const taken = new Set(game.body.map(([x, y]) => x + ',' + y));
  const free = [];
  for (let y = 0; y < game.size; y++) {
    for (let x = 0; x < game.size; x++) if (!taken.has(x + ',' + y)) free.push([x, y]);
  }
  return free.length ? free[Math.floor(game.rand() * free.length)] : null;
}

// A new game, the same start as snake/board.py: three segments heading east in the middle.
export function newGame(size, seed = Date.now()) {
  const cx = Math.floor(size / 2), cy = Math.floor(size / 2);
  const game = {
    size, rand: rng(seed), body: [[cx, cy], [cx - 1, cy], [cx - 2, cy]], heading: 1,
    apples: 0, steps: 0, idle: 0, idleCap: 2 * size * size, cause: null, food: null,
  };
  game.food = placeFood(game);
  return game;
}

// Build a game from a position (used when the page restores a state; no RNG is needed).
export function gameFrom(size, body, heading, food) {
  return { size, rand: rng(1), body: body.map((c) => c.slice()), heading, apples: 0, steps: 0,
    idle: 0, idleCap: 2 * size * size, cause: null, food: food ? food.slice() : null };
}

export const alive = (game) => game.cause === null;

// The cell the head would enter with `action`. Does not change the game.
export function nextCell(game, action) {
  const [dx, dy] = DIRS[turn(game.heading, action)];
  const [hx, hy] = game.body[0];
  return [hx + dx, hy + dy];
}

const inBounds = (game, [x, y]) => x >= 0 && x < game.size && y >= 0 && y < game.size;

// Same rule as Game._blocks: the tail is free when the snake does not eat this step.
function blocks(game, cell, eats) {
  const segs = eats ? game.body : game.body.slice(0, -1);
  return segs.some(([x, y]) => x === cell[0] && y === cell[1]);
}

// Advance one step. Returns true while the game goes on. Mirrors Game.step in snake/board.py.
export function step(game, action) {
  if (!alive(game)) return false;
  const cell = nextCell(game, action);
  game.heading = turn(game.heading, action);
  game.steps += 1;
  if (!inBounds(game, cell)) { game.cause = 'wall'; return false; }
  const eats = game.food !== null && cell[0] === game.food[0] && cell[1] === game.food[1];
  if (blocks(game, cell, eats)) { game.cause = 'body'; return false; }
  game.body.unshift(cell);
  if (eats) {
    game.apples += 1;
    game.idle = 0;
    game.food = placeFood(game);
    if (game.food === null) { game.cause = 'full'; return false; }
  } else {
    game.body.pop();
    game.idle += 1;
    if (game.idle >= game.idleCap) { game.cause = 'starved'; return false; }
  }
  return true;
}

// The safe-move test the baselines use: would `action` end the game?
export function wouldDie(game, action) {
  const cell = nextCell(game, action);
  if (!inBounds(game, cell)) return true;
  return blocks(game, cell, game.food !== null && cell[0] === game.food[0] && cell[1] === game.food[1]);
}

// Empty cells reachable from `start` (counting it), stopping at FLOOD_CAP. Zero if start is blocked.
// Mirrors snake/net.py.room: breadth-first over empty cells, the body is the wall.
export function room(game, occupied, start) {
  const size = game.size;
  const key = (x, y) => x + ',' + y;
  if (!(start[0] >= 0 && start[0] < size && start[1] >= 0 && start[1] < size) || occupied.has(key(start[0], start[1]))) return 0;
  const seen = new Set([key(start[0], start[1])]);
  const queue = [start];
  let qi = 0;
  while (qi < queue.length && seen.size < FLOOD_CAP) {
    const [cx, cy] = queue[qi++];
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx, ny = cy + dy, k = key(nx, ny);
      if (seen.has(k) || occupied.has(k) || !(nx >= 0 && nx < size && ny >= 0 && ny < size)) continue;
      seen.add(k);
      queue.push([nx, ny]);
      if (seen.size >= FLOOD_CAP) break;
    }
  }
  return seen.size;
}

// The 17 inputs of snake/net.py.features, in the same order. Works in the snake's own frame.
export function features(game) {
  const size = game.size;
  const occupied = new Set(game.body.map(([x, y]) => x + ',' + y));
  const [hx, hy] = game.body[0];
  const heading = game.heading;
  const ahead = DIRS[heading], right = DIRS[(heading + 1) % 4], left = DIRS[(heading + 3) % 4];
  const scale = size - 1;
  const blocked = (x, y) => !(x >= 0 && x < size && y >= 0 && y < size) || occupied.has(x + ',' + y);
  const clip = (v) => Math.max(-1, Math.min(1, v));
  const x = new Float64Array(17);
  [ahead, left, right].forEach(([dx, dy], k) => {
    x[k] = blocked(hx + dx, hy + dy) ? 1 : 0;
    let run = 0, cx = hx, cy = hy;
    for (;;) {
      cx += dx; cy += dy;
      if (blocked(cx, cy)) break;
      run += 1;
    }
    x[3 + k] = run / scale;
    x[14 + k] = room(game, occupied, [hx + dx, hy + dy]) / FLOOD_CAP;
  });
  const offset = ([cx, cy]) => {
    const ox = cx - hx, oy = cy - hy;
    return [clip((ox * ahead[0] + oy * ahead[1]) / scale), clip((ox * right[0] + oy * right[1]) / scale)];
  };
  if (game.food !== null) [x[6], x[7]] = offset(game.food);
  [x[8], x[9]] = offset(game.body[game.body.length - 1]);
  x[10 + heading] = 1;
  return x;
}

// The network as the champion JSON gives it: w1 [14][16], b1 [16], w2 [16][3], b2 [3].
export function forward(net, x) {
  const H = net.b1.length, O = net.b2.length, N = x.length;
  const h = new Float64Array(H);
  for (let j = 0; j < H; j++) {
    let s = net.b1[j];
    for (let i = 0; i < N; i++) s += x[i] * net.w1[i][j];
    h[j] = Math.tanh(s);
  }
  const logits = new Float64Array(O);
  for (let k = 0; k < O; k++) {
    let s = net.b2[k];
    for (let j = 0; j < H; j++) s += h[j] * net.w2[j][k];
    logits[k] = s;
  }
  return { h, logits };
}

export function softmax(logits) {
  const m = Math.max(...logits);
  const e = Array.from(logits, (v) => Math.exp(v - m));
  const total = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / total);
}

// The action the network takes: the first largest logit, as np.argmax does.
export function argmax(values) {
  let best = 0;
  for (let k = 1; k < values.length; k++) if (values[k] > values[best]) best = k;
  return best;
}

// Which inputs drove the chosen output: input i's share is x_i times the sum over hidden units of
// w1[i][j] * (1 - h_j^2) * w2[j][k], the gradient of that logit with respect to the input times the
// input. Positive pushes the choice up, negative pushes it down. The page uses these for the explanation.
export function attribution(net, x, h, k) {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    let s = 0;
    for (let j = 0; j < h.length; j++) s += net.w1[i][j] * (1 - h[j] * h[j]) * net.w2[j][k];
    out[i] = x[i] * s;
  }
  return out;
}

// One full decision for the current position: features, activations, probabilities, and the action.
export function decide(net, game) {
  const x = features(game);
  const { h, logits } = forward(net, x);
  const action = argmax(logits);
  return { x, h, logits, probs: softmax(logits), action, attr: attribution(net, x, h, action) };
}

// ── The evaluation-function agent (snake/evaluator.py) ────────────────────────────────────────
// Eight features of the position each safe move leads to, weighted and summed. The move with the
// largest sum is taken; ties keep the earlier move in ACTION_ORDER (straight, then left, then right).
export const EVAL_FEATURES = [
  'apple', 'food near', 'food reachable', 'area', 'room for body', 'tail reachable', 'exits', 'length',
];
const ACTION_ORDER = [STRAIGHT, LEFT, RIGHT];

// The features of the position after `action`, as snake/evaluator.py move_features. The body after the
// move is the new head, then the old body minus the tail unless the snake ate; the tail is an escape goal.
export function moveFeatures(game, action) {
  const size = game.size, cells = size * size;
  const head = nextCell(game, action);
  const eats = game.food !== null && head[0] === game.food[0] && head[1] === game.food[1];
  const body2 = [head, ...(eats ? game.body : game.body.slice(0, -1))];
  const key = (x, y) => x + ',' + y;
  const occupied = new Set(body2.map(([x, y]) => key(x, y)));
  const tail = body2[body2.length - 1];
  const tailKey = key(tail[0], tail[1]);
  const blocked = new Set(body2.slice(0, -1).map(([x, y]) => key(x, y)));
  const nEmpty = cells - body2.length;
  // One breadth-first search from the head over empty cells: food distance, reachable area, tail reach.
  const dist = new Map([[key(head[0], head[1]), 0]]);
  const queue = [head];
  let qi = 0, tailOk = false;
  while (qi < queue.length) {
    const [cx, cy] = queue[qi++];
    const d = dist.get(key(cx, cy));
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx, ny = cy + dy, k = key(nx, ny);
      if (dist.has(k) || !(nx >= 0 && nx < size && ny >= 0 && ny < size)) continue;
      if (k === tailKey) { tailOk = true; continue; }
      if (blocked.has(k)) continue;
      dist.set(k, d + 1);
      queue.push([nx, ny]);
    }
  }
  const reach = dist.size - 1;
  let foodNear = 0, foodReach = 0;
  if (eats) { foodNear = 1; foodReach = 1; }
  else if (game.food !== null && dist.has(key(game.food[0], game.food[1]))) {
    foodNear = Math.max(0, 1 - dist.get(key(game.food[0], game.food[1])) / (2 * size));
    foodReach = 1;
  }
  const [hx, hy] = head;
  let exits = 0;
  for (const [dx, dy] of DIRS) {
    const nx = hx + dx, ny = hy + dy;
    if (nx >= 0 && nx < size && ny >= 0 && ny < size && !occupied.has(key(nx, ny))) exits++;
  }
  const length = body2.length;
  return [
    eats ? 1 : 0,
    foodNear,
    foodReach,
    nEmpty > 0 ? reach / nEmpty : 0,
    Math.min(1, reach / length),
    tailOk ? 1 : 0,
    exits / 4,
    length / cells,
  ];
}

// Every move with its features, score, and whether it ends the game (score is null for those).
export function moveTable(game, weights) {
  return ACTION_ORDER.map((action) => {
    const dies = wouldDie(game, action);
    const features = moveFeatures(game, action);
    let score = null;
    if (!dies) {
      score = 0;
      for (let i = 0; i < features.length; i++) score += weights[i] * features[i];
    }
    return { action, features, score, dies };
  });
}

// The chosen move: the safe move with the largest score, or straight when every move ends the game.
export function chooseEval(weights, game) {
  let best = STRAIGHT, bestScore = -Infinity;
  for (const row of moveTable(game, weights)) {
    if (row.dies) continue;
    if (row.score > bestScore) { best = row.action; bestScore = row.score; }
  }
  return best;
}
