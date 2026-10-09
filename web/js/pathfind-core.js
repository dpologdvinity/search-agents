// Pathfinding core for the lab: maps, movement rules, heuristics and the seven searches. No DOM.
//
// This file is a line-for-line port of the Python package pathfind/. A seed gives the same map, the
// same expansion order and the same path here as there. tests/test_pathfind_parity.py checks that
// with a node run on fixed seeds, so keep the two in step. Ties are broken by insertion order
// (a heap keyed by priority, then by push number), and neighbours are visited in a fixed order.

export const WALL = 0;
export const OPEN = 1;
export const MUD = 3;
export const SWAMP = 5;
export const SQRT2 = 1.4142135623730951; // the same double as Math.SQRT2 and Python's math.sqrt(2)
export const DEFAULT_WEIGHT = 2;
export const ALGOS = ['bfs', 'dfs', 'ucs', 'greedy', 'astar', 'wastar', 'bidi'];
export const HEURISTICS = ['manhattan', 'euclidean', 'octile'];
export const LABELS = {
  bfs: 'Breadth-first',
  dfs: 'Depth-first',
  ucs: 'Uniform cost (Dijkstra)',
  greedy: 'Greedy best-first',
  astar: 'A*',
  wastar: 'Weighted A*',
  bidi: 'Bidirectional BFS',
};
export const MAZE_KINDS = ['recursive', 'prim', 'scatter', 'rooms', 'open'];

const DIRS4 = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const DIAGS = [[1, -1], [1, 1], [-1, 1], [-1, -1]];
const LATTICE_DIRS = [[0, -2], [2, 0], [0, 2], [-2, 0]];
const NOT_SEEN = -2;
const ROOT = -1;

// ── Seeded random numbers (mulberry32, same bits as pathfind/rng.py) ──────

export class Rng {
  constructor(seed) {
    this.a = seed >>> 0;
  }

  // Next unsigned 32-bit value. Math.imul keeps the low 32 bits of each product, like Python's mask.
  u32() {
    this.a = (this.a + 0x6d2b79f5) >>> 0;
    const a = this.a;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  }

  int(n) {
    return this.u32() % n;
  }

  percent(p) {
    return this.u32() % 100 < p;
  }
}

// ── Grid and heuristics ───────────────────────────────────────────────────

export class Grid {
  constructor(width, height, cells = null) {
    if (width < 1 || height < 1) throw new Error('grid must be at least 1x1');
    this.width = width;
    this.height = height;
    this.cells = cells ? cells.slice() : new Array(width * height).fill(OPEN);
    if (this.cells.length !== width * height) throw new Error('cells length must be width * height');
  }

  get length() {
    return this.width * this.height;
  }

  xy(i) {
    return [i % this.width, Math.floor(i / this.width)];
  }

  index(x, y) {
    return y * this.width + x;
  }

  inside(x, y) {
    return x >= 0 && x < this.width && y >= 0 && y < this.height;
  }

  passable(x, y) {
    return this.inside(x, y) && this.cells[y * this.width + x] !== WALL;
  }

  // [neighbour, step length] pairs in the fixed order. A diagonal is skipped when either orthogonal
  // neighbour is a wall, so it never squeezes between two walls that touch at a corner.
  neighbors(i, diagonal = false) {
    const [x, y] = this.xy(i);
    const out = [];
    for (const [dx, dy] of DIRS4) {
      if (this.passable(x + dx, y + dy)) out.push([this.index(x + dx, y + dy), 1]);
    }
    if (!diagonal) return out;
    for (const [dx, dy] of DIAGS) {
      const nx = x + dx, ny = y + dy;
      if (this.passable(nx, ny) && this.passable(x + dx, y) && this.passable(x, y + dy)) {
        out.push([this.index(nx, ny), SQRT2]);
      }
    }
    return out;
  }

  // Cost of moving from a to the adjacent cell b: terrain of b times the step length.
  stepCost(a, b) {
    const [ax, ay] = this.xy(a);
    const [bx, by] = this.xy(b);
    const step = ax !== bx && ay !== by ? SQRT2 : 1;
    return step * this.cells[b];
  }

  pathCost(path) {
    if (!path.length) return null;
    let total = 0;
    for (let k = 1; k < path.length; k++) total += this.stepCost(path[k - 1], path[k]);
    return total;
  }
}

export function heuristic(name, x, y, gx, gy) {
  const dx = Math.abs(x - gx);
  const dy = Math.abs(y - gy);
  if (name === 'manhattan') return dx + dy;
  if (name === 'euclidean') return Math.sqrt(dx * dx + dy * dy);
  if (name === 'octile') {
    const hi = dx > dy ? dx : dy;
    const lo = dx > dy ? dy : dx;
    return hi + (SQRT2 - 1) * lo;
  }
  throw new Error(`unknown heuristic ${name}`);
}

// ── Mazes (port of pathfind/mazes.py) ─────────────────────────────────────

// Start at (1, 1) and goal at the last odd interior coordinates (lattice cells).
function latticeEnds(width, height) {
  const gx = (width - 2) % 2 === 1 ? width - 2 : width - 3;
  const gy = (height - 2) % 2 === 1 ? height - 2 : height - 3;
  return [width + 1, gy * width + gx];
}

function recursive(width, height, rng) {
  const cells = new Array(width * height).fill(WALL);
  const stack = [[1, 1]];
  cells[width + 1] = OPEN;
  while (stack.length) {
    const [x, y] = stack[stack.length - 1];
    const options = [];
    for (const [dx, dy] of LATTICE_DIRS) {
      const nx = x + dx, ny = y + dy;
      if (nx > 0 && nx < width - 1 && ny > 0 && ny < height - 1 && cells[ny * width + nx] === WALL) {
        options.push([dx, dy]);
      }
    }
    if (!options.length) {
      stack.pop();
      continue;
    }
    const [dx, dy] = options[rng.int(options.length)];
    const nx = x + dx, ny = y + dy;
    cells[(y + dy / 2) * width + (x + dx / 2)] = OPEN;
    cells[ny * width + nx] = OPEN;
    stack.push([nx, ny]);
  }
  return cells;
}

function prim(width, height, rng) {
  const cells = new Array(width * height).fill(WALL);
  cells[width + 1] = OPEN;
  const frontier = [];
  const addNeighbours = (x, y) => {
    for (const [dx, dy] of LATTICE_DIRS) {
      const nx = x + dx, ny = y + dy;
      if (nx > 0 && nx < width - 1 && ny > 0 && ny < height - 1 && cells[ny * width + nx] === WALL) {
        frontier.push([nx, ny, x, y]);
      }
    }
  };
  addNeighbours(1, 1);
  while (frontier.length) {
    const i = rng.int(frontier.length);
    const last = frontier.length - 1;
    [frontier[i], frontier[last]] = [frontier[last], frontier[i]];
    const [nx, ny, px, py] = frontier.pop();
    if (cells[ny * width + nx] !== WALL) continue;
    cells[((py + ny) / 2) * width + (px + nx) / 2] = OPEN;
    cells[ny * width + nx] = OPEN;
    addNeighbours(nx, ny);
  }
  return cells;
}

function scatter(width, height, rng, density) {
  const cells = new Array(width * height).fill(WALL);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      if (!rng.percent(density)) cells[y * width + x] = OPEN;
    }
  }
  return cells;
}

function rooms(width, height, rng) {
  const cells = new Array(width * height).fill(WALL);
  const target = Math.max(2, Math.floor((width * height) / 150));
  const centres = [];
  const rects = [];
  for (let attempt = 0; attempt < target * 6; attempt++) {
    if (centres.length >= target) break;
    const rw = 3 + rng.int(5);
    const rh = 3 + rng.int(4);
    const x0 = 1 + rng.int(Math.max(1, width - rw - 2));
    const y0 = 1 + rng.int(Math.max(1, height - rh - 2));
    const x1 = x0 + rw - 1, y1 = y0 + rh - 1;
    if (x1 > width - 2 || y1 > height - 2) continue;
    // Reject a room that touches another (one cell of margin), so rooms stay distinct.
    const clash = rects.some(([bx0, by0, bx1, by1]) =>
      x0 - 1 <= bx1 && bx0 <= x1 + 1 && y0 - 1 <= by1 && by0 <= y1 + 1);
    if (clash) continue;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) cells[y * width + x] = OPEN;
    }
    rects.push([x0, y0, x1, y1]);
    centres.push([Math.floor((x0 + x1) / 2), Math.floor((y0 + y1) / 2)]);
  }
  if (centres.length < 2) throw new Error('map too small for two rooms');

  // L-shaped corridor: horizontal along a's row, then vertical along b's column.
  const link = (a, b) => {
    const [ax, ay] = a, [bx, by] = b;
    for (let x = Math.min(ax, bx); x <= Math.max(ax, bx); x++) cells[ay * width + x] = OPEN;
    for (let y = Math.min(ay, by); y <= Math.max(ay, by); y++) cells[y * width + bx] = OPEN;
  };
  for (let i = 1; i < centres.length; i++) link(centres[i - 1], centres[i]);
  for (let i = 0; i < centres.length; i++) {
    const j = rng.int(centres.length);
    if (j !== i) link(centres[i], centres[j]);
  }
  const first = centres[0], last = centres[centres.length - 1];
  return { cells, ends: [first[1] * width + first[0], last[1] * width + last[0]] };
}

/** Build a seeded map: {grid, start, goal}. Same arguments as make_maze() in pathfind/mazes.py. */
export function makeMaze(kind, width, height, seed, { density = 28, swamp = 0 } = {}) {
  if (!MAZE_KINDS.includes(kind)) throw new Error(`unknown maze kind ${kind}`);
  if (width < 5 || height < 5) throw new Error('maze needs at least 5x5 cells');
  const rng = new Rng(seed);
  let cells, start, goal;
  if (kind === 'recursive') {
    cells = recursive(width, height, rng);
    [start, goal] = latticeEnds(width, height);
  } else if (kind === 'prim') {
    cells = prim(width, height, rng);
    [start, goal] = latticeEnds(width, height);
  } else if (kind === 'scatter') {
    cells = scatter(width, height, rng, density);
    [start, goal] = latticeEnds(width, height);
    cells[start] = OPEN;
    cells[goal] = OPEN;
  } else if (kind === 'rooms') {
    const r = rooms(width, height, rng);
    cells = r.cells;
    [start, goal] = r.ends;
  } else {
    cells = new Array(width * height).fill(OPEN);
    start = 0;
    goal = width * height - 1;
  }
  const grid = new Grid(width, height, cells);
  if (swamp) {
    for (let i = 0; i < grid.cells.length; i++) {
      if (grid.cells[i] === OPEN && i !== start && i !== goal && rng.percent(swamp)) grid.cells[i] = SWAMP;
    }
  }
  return { grid, start, goal };
}

// ── Heap ordered by (priority, push number): equal priorities pop in push order ──

class Heap {
  constructor() {
    this.items = [];
  }

  get size() {
    return this.items.length;
  }

  less(a, b) {
    return a.p < b.p || (a.p === b.p && a.s < b.s);
  }

  push(item) {
    const h = this.items;
    h.push(item);
    let i = h.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!this.less(h[i], h[parent])) break;
      [h[i], h[parent]] = [h[parent], h[i]];
      i = parent;
    }
  }

  pop() {
    const h = this.items;
    const top = h[0];
    const last = h.pop();
    if (h.length) {
      h[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < h.length && this.less(h[l], h[m])) m = l;
        if (r < h.length && this.less(h[r], h[m])) m = r;
        if (m === i) break;
        [h[i], h[m]] = [h[m], h[i]];
        i = m;
      }
    }
    return top;
  }
}

// ── Searches (port of pathfind/search.py) ─────────────────────────────────
// Each returns {algo, found, path, cost, steps, generated}. steps holds one [cell, side, discovered]
// per expansion, where discovered is a list of [cell, g] pairs.

function chain(parent, v) {
  const out = [v];
  while (parent[v] !== ROOT) {
    v = parent[v];
    out.push(v);
  }
  return out.reverse();
}

function finish(grid, algo, steps, goal, path, generated) {
  const found = steps.length > 0 && steps[steps.length - 1][0] === goal;
  if (!found) return { algo, found: false, path: [], cost: null, steps, generated };
  return { algo, found: true, path, cost: grid.pathCost(path), steps, generated };
}

export function bfs(grid, start, goal, diagonal = false) {
  const n = grid.length;
  const parent = new Array(n).fill(NOT_SEEN);
  const g = new Array(n).fill(0);
  parent[start] = ROOT;
  const queue = [start];
  let head = 0;
  const steps = [];
  let generated = 0;
  while (head < queue.length) {
    const u = queue[head++];
    const fresh = [];
    steps.push([u, 0, fresh]);
    if (u === goal) break;
    for (const [v, mult] of grid.neighbors(u, diagonal)) {
      if (parent[v] === NOT_SEEN) {
        parent[v] = u;
        g[v] = g[u] + mult * grid.cells[v];
        queue.push(v);
        fresh.push([v, g[v]]);
        generated++;
      }
    }
  }
  const path = steps.length && steps[steps.length - 1][0] === goal ? chain(parent, goal) : [];
  return finish(grid, 'bfs', steps, goal, path, generated);
}

export function dfs(grid, start, goal, diagonal = false) {
  const n = grid.length;
  const closed = new Array(n).fill(false);
  const parent = new Array(n).fill(ROOT);
  const stack = [[start, ROOT, 0]];
  const steps = [];
  let generated = 0;
  while (stack.length) {
    const [u, p, gu] = stack.pop();
    if (closed[u]) continue;
    closed[u] = true;
    parent[u] = p;
    const fresh = [];
    steps.push([u, 0, fresh]);
    if (u === goal) break;
    for (const [v, mult] of grid.neighbors(u, diagonal)) {
      if (!closed[v]) {
        const gv = gu + mult * grid.cells[v];
        stack.push([v, u, gv]);
        fresh.push([v, gv]);
        generated++;
      }
    }
  }
  const path = steps.length && steps[steps.length - 1][0] === goal ? chain(parent, goal) : [];
  return finish(grid, 'dfs', steps, goal, path, generated);
}

// Shared loop for uniform cost (priority g), greedy (priority h) and A* (priority g + w*h).
function bestFirst(grid, algo, start, goal, { diagonal, heur, weight }) {
  const n = grid.length;
  const [gx, gy] = grid.xy(goal);
  const g = new Array(n).fill(Infinity);
  const parent = new Array(n).fill(ROOT);
  const closed = new Array(n).fill(false);
  const priority = (v, gv) => {
    if (algo === 'ucs') return gv;
    const [x, y] = grid.xy(v);
    const h = heuristic(heur, x, y, gx, gy);
    if (algo === 'greedy') return h;
    return gv + weight * h;
  };
  g[start] = 0;
  const heap = new Heap();
  let seq = 0;
  heap.push({ p: priority(start, 0), s: seq, v: start });
  const steps = [];
  let generated = 0;
  while (heap.size) {
    const { v: u } = heap.pop();
    if (closed[u]) continue;
    closed[u] = true;
    const fresh = [];
    steps.push([u, 0, fresh]);
    if (u === goal) break;
    for (const [v, mult] of grid.neighbors(u, diagonal)) {
      if (closed[v]) continue;
      const ng = g[u] + mult * grid.cells[v];
      if (ng < g[v]) {
        g[v] = ng;
        parent[v] = u;
        seq++;
        heap.push({ p: priority(v, ng), s: seq, v });
        fresh.push([v, ng]);
        generated++;
      }
    }
  }
  const path = steps.length && steps[steps.length - 1][0] === goal ? chain(parent, goal) : [];
  return finish(grid, algo, steps, goal, path, generated);
}

export function uniformCost(grid, start, goal, diagonal = false) {
  return bestFirst(grid, 'ucs', start, goal, { diagonal, heur: 'manhattan', weight: 0 });
}

export function greedy(grid, start, goal, diagonal = false, heur = 'octile') {
  return bestFirst(grid, 'greedy', start, goal, { diagonal, heur, weight: 1 });
}

export function astar(grid, start, goal, diagonal = false, heur = 'octile') {
  return bestFirst(grid, 'astar', start, goal, { diagonal, heur, weight: 1 });
}

export function weightedAstar(grid, start, goal, diagonal = false, heur = 'octile', weight = DEFAULT_WEIGHT) {
  if (weight < 1) throw new Error('weight must be at least 1');
  return bestFirst(grid, 'wastar', start, goal, { diagonal, heur, weight });
}

// Breadth-first from both ends, one whole level at a time, until the two searches have met. It searches
// in steps, like BFS, so its path is as short in steps as BFS's. The first meeting found is not always the
// shortest, so every meeting is scored by its total steps and the search stops only when the best one uses
// no more steps than the two completed levels reach (any shorter path has a cell both searches already found).
export function bidirectionalBfs(grid, start, goal, diagonal = false) {
  const n = grid.length;
  if (start === goal) return { algo: 'bidi', found: true, path: [start], cost: 0, steps: [[start, 0, []]], generated: 0 };
  const parents = [new Array(n).fill(NOT_SEEN), new Array(n).fill(NOT_SEEN)];
  const depth = [new Array(n).fill(0), new Array(n).fill(0)]; // steps from start (side 0) or to goal (side 1)
  const g = [new Array(n).fill(0), new Array(n).fill(0)];
  parents[0][start] = ROOT;
  parents[1][goal] = ROOT;
  const queues = [[start], [goal]];
  const heads = [0, 0];
  const done = [0, 0]; // whole levels each side has expanded
  const steps = [];
  let generated = 0;
  let best = -1; // steps in the shortest meeting path found so far
  let meet = -1;
  while (heads[0] < queues[0].length || heads[1] < queues[1].length) {
    if (best >= 0 && best <= done[0] + done[1]) break;
    // Expand the side with the smaller next level; ties go to the forward search.
    const waiting0 = queues[0].length - heads[0];
    const waiting1 = queues[1].length - heads[1];
    const side = waiting0 > 0 && (waiting1 === 0 || waiting0 <= waiting1) ? 0 : 1;
    // The queue holds exactly one level at this point, so expand all of it before switching sides.
    const levelEnd = queues[side].length;
    while (heads[side] < levelEnd) {
      const u = queues[side][heads[side]++];
      const fresh = [];
      steps.push([u, side, fresh]);
      for (const [v, mult] of grid.neighbors(u, diagonal)) {
        if (parents[side][v] !== NOT_SEEN) continue;
        parents[side][v] = u;
        depth[side][v] = depth[side][u] + 1;
        // The step is entered from u going forward, or from v going backward, so the cost is of the cell entered.
        const entered = side === 0 ? v : u;
        g[side][v] = g[side][u] + mult * grid.cells[entered];
        queues[side].push(v);
        fresh.push([v, g[side][v]]);
        generated++;
        if (parents[1 - side][v] !== NOT_SEEN) {
          const total = depth[side][v] + depth[1 - side][v];
          if (best < 0 || total < best) {
            best = total;
            meet = v;
          }
        }
      }
    }
    done[side]++;
  }
  if (best < 0) return { algo: 'bidi', found: false, path: [], cost: null, steps, generated };
  const forward = chain(parents[0], meet);
  const backward = chain(parents[1], meet).reverse(); // meet ... goal
  const path = forward.concat(backward.slice(1));
  return { algo: 'bidi', found: true, path, cost: grid.pathCost(path), steps, generated };
}

export function run(algo, grid, start, goal, { diagonal = false, heur = 'octile', weight = DEFAULT_WEIGHT } = {}) {
  switch (algo) {
    case 'bfs': return bfs(grid, start, goal, diagonal);
    case 'dfs': return dfs(grid, start, goal, diagonal);
    case 'ucs': return uniformCost(grid, start, goal, diagonal);
    case 'greedy': return greedy(grid, start, goal, diagonal, heur);
    case 'astar': return astar(grid, start, goal, diagonal, heur);
    case 'wastar': return weightedAstar(grid, start, goal, diagonal, heur, weight);
    case 'bidi': return bidirectionalBfs(grid, start, goal, diagonal);
    default: throw new Error(`unknown algorithm ${algo}`);
  }
}

// Frontier size after each expansion: cells discovered so far that are not yet expanded. Kept as a
// running count (seen minus expanded) so a long search costs one pass.
export function frontierSizes(result) {
  const seen = new Set();
  const expanded = new Set();
  let open = 0;
  const sizes = [];
  for (const [cell, , fresh] of result.steps) {
    if (!expanded.has(cell)) {
      expanded.add(cell);
      if (seen.has(cell)) open--;
    }
    for (const [v] of fresh) {
      if (!seen.has(v)) {
        seen.add(v);
        if (!expanded.has(v)) open++;
      }
    }
    sizes.push(open);
  }
  return sizes;
}
