// Rover core: a line-for-line port of rover/world.py, rover/dstar.py, rover/astar.py and rover/explorer.py.
//
// Pure JavaScript with no DOM access, so the page and node (tests/test_rover_parity.py) can both load it.
// The port keeps the Python tie-breaks exactly (neighbour order N, E, S, W; queue entries compared as
// (k1, k2, cell, version); A* compares (f, h, cell)), so both versions expand the same cells in the same
// order. Python is the reference: when the two disagree, the Python code wins and this file is fixed.
//
// Cells are numbered row-major, cell c = row * n + col. Walls and belief are Uint8Array (1 = wall).

export const INF = Infinity;
const M32 = 0xffffffff;

// ── seeded PRNG and maps ────────────────────────────────────────────────

// mulberry32. Math.imul keeps the low 32 bits of a product, like Python's `& 0xFFFFFFFF`, so both
// languages produce the same stream for the same seed.
export class Mulberry32 {
  constructor(seed) {
    this.a = seed >>> 0;
  }
  next() {
    this.a = (this.a + 0x6d2b79f5) >>> 0;
    let t = this.a;
    t = Math.imul(t ^ (t >>> 15), 1 | t) >>> 0;
    t = ((((t + (Math.imul(t ^ (t >>> 7), 61 | t) >>> 0)) >>> 0) ^ t) >>> 0);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

// The up to four orthogonal neighbours of a cell, in the order N, E, S, W.
export function neighbours(n, cell) {
  const r = Math.floor(cell / n), c = cell % n;
  const out = [];
  if (r > 0) out.push(cell - n);
  if (c < n - 1) out.push(cell + 1);
  if (r < n - 1) out.push(cell + n);
  if (c > 0) out.push(cell - 1);
  return out;
}

// True if a breadth-first search finds a wall-free path from start to goal.
export function reachable(n, walls, start, goal) {
  const seen = new Uint8Array(n * n);
  seen[start] = 1;
  const queue = [start];
  for (let qi = 0; qi < queue.length; qi++) {
    const u = queue[qi];
    if (u === goal) return true;
    for (const v of neighbours(n, u)) {
      if (!seen[v] && !walls[v]) {
        seen[v] = 1;
        queue.push(v);
      }
    }
  }
  return false;
}

// Seeded map: each cell is a wall with probability density; corners stay open; redraw until reachable.
export function generate(n, density, seed) {
  const start = 0, goal = n * n - 1;
  const rng = new Mulberry32(seed);
  for (;;) {
    const walls = new Uint8Array(n * n);
    for (let c = 0; c < n * n; c++) {
      if (c !== start && c !== goal && rng.next() < density) walls[c] = 1;
    }
    if (reachable(n, walls, start, goal)) return walls;
  }
}

// Cells inside the sensor square of half-width radius around cell, clipped to the map.
export function sensedCells(n, cell, radius) {
  const r0 = Math.floor(cell / n), c0 = cell % n;
  const out = [];
  for (let r = Math.max(0, r0 - radius); r <= Math.min(n - 1, r0 + radius); r++) {
    for (let c = Math.max(0, c0 - radius); c <= Math.min(n - 1, c0 + radius); c++) out.push(r * n + c);
  }
  return out;
}

// ── a binary min-heap ordered by a comparison function ──────────────────

class MinHeap {
  constructor(less) {
    this.items = [];
    this.less = less;
  }
  get size() { return this.items.length; }
  top() { return this.items[0]; }
  push(x) {
    const a = this.items;
    a.push(x);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(a[i], a[p])) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop() {
    const a = this.items;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && this.less(a[l], a[m])) m = l;
        if (r < a.length && this.less(a[r], a[m])) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]];
        i = m;
      }
    }
    return top;
  }
}

// Lexicographic comparison of two arrays of numbers.
function lexLess(a, b) {
  for (let i = 0; i < a.length; i++) {
    if (a[i] < b[i]) return true;
    if (a[i] > b[i]) return false;
  }
  return false;
}

// ── D* Lite ─────────────────────────────────────────────────────────────

// Same algorithm and bookkeeping as rover/dstar.py. See that file for the reasoning.
export class DStarLite {
  constructor(n, start, goal) {
    this.n = n;
    this.goal = goal;
    this.start = start;
    this.sLast = start;
    this.km = 0;
    const N = n * n;
    this.wall = new Uint8Array(N);
    this.g = new Array(N).fill(INF);
    this.rhs = new Array(N).fill(INF);
    // Queue entries are [k1, k2, cell, version]; ver[cell] is the live version, 0 when not queued.
    this.heap = new MinHeap(lexLess);
    this.ver = new Array(N).fill(0);
    this.version = 0;
    this.expansions = 0;
    this.lastExpanded = [];
    this.rhs[goal] = 0;
    this._insert(goal, this._key(goal));
    this._compute();
  }

  _h(a, b) {
    const n = this.n;
    return Math.abs(Math.floor(a / n) - Math.floor(b / n)) + Math.abs((a % n) - (b % n));
  }

  _cost(u, v) {
    return (this.wall[u] || this.wall[v]) ? INF : 1;
  }

  _key(u) {
    const m = Math.min(this.g[u], this.rhs[u]);
    return [m + this._h(this.start, u) + this.km, m];
  }

  _insert(u, key) {
    this.version += 1;
    this.ver[u] = this.version;
    this.heap.push([key[0], key[1], u, this.version]);
  }

  // Key of the cheapest live entry; stale entries are dropped first.
  _top() {
    const h = this.heap;
    while (h.size && this.ver[h.top()[2]] !== h.top()[3]) h.pop();
    return h.size ? [h.top()[0], h.top()[1]] : [INF, INF];
  }

  _pop() {
    for (;;) {
      const [k1, k2, u, v] = this.heap.pop();
      if (this.ver[u] === v) {
        this.ver[u] = 0;
        return [[k1, k2], u];
      }
    }
  }

  _update(u) {
    if (u !== this.goal) {
      let best = INF;
      for (const s of neighbours(this.n, u)) {
        const c = this._cost(u, s) + this.g[s];
        if (c < best) best = c;
      }
      this.rhs[u] = best;
    }
    if (this.ver[u]) this.ver[u] = 0;
    if (this.g[u] !== this.rhs[u]) this._insert(u, this._key(u));
  }

  _compute() {
    this.lastExpanded = [];
    for (;;) {
      const top = this._top();
      if (!this.heap.size) break;
      const ks = this._key(this.start);
      if (!(lexLess(top, ks) || this.rhs[this.start] !== this.g[this.start])) break;
      const [kOld, u] = this._pop();
      this.expansions += 1;
      this.lastExpanded.push(u);
      const kNew = this._key(u);
      if (lexLess(kOld, kNew)) {
        this._insert(u, kNew);
      } else if (this.g[u] > this.rhs[u]) {
        this.g[u] = this.rhs[u];
        for (const s of neighbours(this.n, u)) this._update(s);
      } else {
        this.g[u] = INF;
        for (const s of neighbours(this.n, u)) this._update(s);
        this._update(u);
      }
    }
  }

  // changes: array of [cell, isWall]. Repairs the plan for a rover now at pos.
  onChange(changes, pos) {
    this.km += this._h(this.sLast, pos);
    this.sLast = pos;
    this.start = pos;
    for (const [cell, wall] of changes) this.wall[cell] = wall;
    for (const [cell] of changes) {
      this._update(cell);
      for (const s of neighbours(this.n, cell)) this._update(s);
    }
    this._compute();
  }

  costToGoal(pos) {
    return this.g[pos];
  }

  // Neighbour with the lowest c(pos, s) + g(s); null if there is no route. Pure: changes nothing.
  nextCell(pos) {
    let best = null, bestV = INF;
    for (const s of neighbours(this.n, pos)) {
      const v = this._cost(pos, s) + this.g[s];
      if (v < bestV) { best = s; bestV = v; }
    }
    return bestV < INF ? best : null;
  }

  pathFrom(pos) {
    if (this.g[pos] === INF) return [];
    const path = [pos];
    let cur = pos;
    const limit = this.n * this.n;
    while (cur !== this.goal && path.length <= limit) {
      cur = this.nextCell(cur);
      if (cur === null) return [];
      path.push(cur);
    }
    return path;
  }
}

// ── A* baseline ─────────────────────────────────────────────────────────

// Same search as rover/astar.py: entries (f, h, cell), closed cells are skipped, neighbour order N, E, S, W.
export class AStar {
  constructor(n, start, goal) {
    this.n = n;
    this.goal = goal;
    this.wall = new Uint8Array(n * n);
    this.path = [];
    this.expansions = 0;
    this.lastExpanded = [];
    this._plan(start);
  }

  _h(a, b) {
    const n = this.n;
    return Math.abs(Math.floor(a / n) - Math.floor(b / n)) + Math.abs((a % n) - (b % n));
  }

  _plan(pos) {
    const n = this.n, goal = this.goal, wall = this.wall;
    const N = n * n;
    const g = new Array(N).fill(INF);
    const parent = new Array(N).fill(-1);
    const closed = new Uint8Array(N);
    g[pos] = 0;
    const h0 = this._h(pos, goal);
    const heap = new MinHeap(lexLess);
    heap.push([h0, h0, pos]);
    const order = [];
    while (heap.size) {
      const [, , u] = heap.pop();
      if (closed[u]) continue;
      closed[u] = 1;
      this.expansions += 1;
      order.push(u);
      if (u === goal) break;
      for (const v of neighbours(n, u)) {
        if (wall[v]) continue;
        const ng = g[u] + 1;
        if (ng < g[v]) {
          g[v] = ng;
          parent[v] = u;
          const hv = this._h(v, goal);
          heap.push([ng + hv, hv, v]);
        }
      }
    }
    this.lastExpanded = order;
    if (!closed[goal]) {
      this.path = [];
      return;
    }
    const path = [goal];
    let cur = goal;
    while (cur !== pos) {
      cur = parent[cur];
      path.push(cur);
    }
    path.reverse();
    this.path = path;
  }

  onChange(changes, pos) {
    for (const [cell, wall] of changes) this.wall[cell] = wall;
    this._plan(pos);
  }

  costToGoal(pos) {
    return this.path.length && this.path[0] === pos ? this.path.length - 1 : INF;
  }

  nextCell(pos) {
    while (this.path.length && this.path[0] !== pos) this.path.shift();
    if (this.path.length < 2) return null;
    return this.path[1];
  }
}

// ── the rover loop ──────────────────────────────────────────────────────

// One rover on one world. The world is {n, walls: Uint8Array, start, goal}. Mirrors rover/explorer.py.
export class Explorer {
  constructor(world, radius = 2, driver = 'dstar') {
    if (driver !== 'dstar' && driver !== 'astar') throw new Error('driver must be dstar or astar');
    if (radius < 1) throw new Error('the sensor radius must be at least 1');
    this.world = world;
    this.n = world.n;
    this.radius = radius;
    this.driver = driver;
    const N = this.n * this.n;
    this.belief = new Uint8Array(N);
    this.seen = new Uint8Array(N);
    this.pos = world.start;
    this.steps = 0;
    this.replans = 0;
    this.done = false;
    this.stuck = false;
    this.mismatches = 0;
    this.lastChanges = [];
    this.planners = {
      dstar: new DStarLite(this.n, world.start, world.goal),
      astar: new AStar(this.n, world.start, world.goal),
    };
    this.initExpansions = { dstar: this.planners.dstar.expansions, astar: this.planners.astar.expansions };
  }

  expansionsOf(name) {
    return this.planners[name].expansions;
  }

  // Route the driver would take from the rover's cell.
  path(name = this.driver) {
    if (name === 'dstar') return this.planners.dstar.pathFrom(this.pos);
    const p = this.planners.astar;
    return p.path.length && p.path[0] === this.pos ? p.path.slice() : [];
  }

  sense() {
    const changes = [];
    const walls = this.world.walls;
    for (const c of sensedCells(this.n, this.pos, this.radius)) {
      this.seen[c] = 1;
      if (walls[c] !== this.belief[c]) {
        this.belief[c] = walls[c];
        changes.push([c, walls[c]]);
      }
    }
    return changes;
  }

  refresh() {
    const changes = this.sense();
    if (!changes.length) return 0;
    this.replans += 1;
    this.lastChanges = changes;
    this.planners.dstar.onChange(changes, this.pos);
    this.planners.astar.onChange(changes, this.pos);
    const a = this.planners.dstar.costToGoal(this.pos);
    const b = this.planners.astar.costToGoal(this.pos);
    if (a !== b) this.mismatches += 1;
    return changes.length;
  }

  tick() {
    if (this.done) return;
    this.refresh();
    if (this.pos === this.world.goal) {
      this.done = true;
      return;
    }
    const nxt = this.planners[this.driver].nextCell(this.pos);
    if (nxt === null) {
      this.stuck = true;
      return;
    }
    if (this.world.walls[nxt]) throw new Error('the rover stepped into a wall');
    this.stuck = false;
    this.pos = nxt;
    this.steps += 1;
    if (this.pos === this.world.goal) this.done = true;
  }

  // The user changes the true map. Senses and replans at once if the cell is in view.
  edit(cell, wall) {
    if (cell === this.pos || cell === this.world.goal) return false;
    this.world.walls[cell] = wall ? 1 : 0;
    this.refresh();
    return true;
  }

  run(maxTicks) {
    const limit = maxTicks ?? 20 * this.n * this.n;
    for (let i = 0; i < limit && !this.done; i++) this.tick();
    return this;
  }
}
