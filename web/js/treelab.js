// Tree lab page. Paint points on a plane, grow a CART decision tree one split at a time (each split fires a laser
// across the region it cuts), and train a random forest on bootstrap samples. The math lives in treelab-core.js,
// a port of the Python package in treelab/. This file only handles state, drawing and input.

import {
  FEATURE_NAMES, PRESETS, fitForest, forestProba, leafOf, leafProba, linkStrengths, makeBuilder,
  makePreset, nLeaves, nodeLabel, preorder, prune, trainTestSplit, treeDepth,
} from './treelab-core.js';
import { burst } from './fx.js';

const G = 96;                 // plane resolution: G x G cells
const NCELL = G * G;
const MD = 16;                // path slots per cell (depth limit is 12)
const MAX_POINTS = 600;
const COLS = ['#00f5ff', '#ff00a0', '#00ff88', '#ffe600'];
const RGB = COLS.map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
const BG = [2, 6, 16];
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const LASER_MS = REDUCED ? 0 : 700;
const LINGER_MS = REDUCED ? 0 : 900;
const NODE_W = 150;
const NODE_H = 74;
const COL = 172;
const ROW = 118;
const PADX = 24;
const PADY = 24;

const $ = (id) => document.getElementById(id);
const pct = (v) => (v == null ? '—' : `${(100 * v).toFixed(1)}%`);

const S = {
  mode: 'tree',
  k: 2,
  cls: 0,
  erase: false,
  pts: [],                     // {x, y, c} in the unit square
  params: { criterion: 'gini', depth: 5, minLeaf: 3, extras: true, testFrac: 0.3 },
  seed: 1,
  nTrees: 25,
  maxF: 'sqrt',
  boot: true,
  showN: 8,
  train: [],                   // indices into pts
  test: [],
  builder: null,               // the full tree, growing
  full: null,                  // builder.root
  shown: null,                 // the view: full tree, or its pruned copy
  nodeById: new Map(),
  parentOf: new Map(),
  alpha: 0,
  splitIds: new Set(),
  seen: new Set(),
  playing: false,
  playTimer: 0,
  lasers: [],
  wallCache: new Map(),
  litSeen: new Set(),
  cellFeat: null,
  cellExtras: null,
  fullPath: null,
  shownLeaf: null,
  shownProb: null,
  shownConf: null,
  regionCv: null,
  hover: -1,
  hoverKey: null,
  hoverCv: null,
  hoverCell: -1,
  painting: false,
  paintDirty: false,
  cursor: null,
  forest: null,
  forestProb: null,
  forestConf: null,
  treeLabels: null,
  boundaries: [],
  forestTimer: 0,
  curve: null,
  acc: { train: null, test: null },
  forestAcc: { train: null, test: null, oob: null, single: null },
  size: 300,
  dpr: 1,
  zoom: 1,
  dirty: true,
  sweep: null,
};

// ── Features and data ───────────────────────────────────────────────────────

/** The feature row for a point; the same formula as expand() in treelab-core.js. */
function feat(x, y) {
  return S.params.extras ? [x, y, x * y, x * x + y * y] : [x, y];
}

function featNames() {
  return S.params.extras ? FEATURE_NAMES : FEATURE_NAMES.slice(0, 2);
}

function hasTwoClasses() {
  const seen = new Set(S.pts.map((p) => p.c));
  return seen.size >= 2 && S.train.length >= 2;
}

function splitData() {
  const { train, test } = trainTestSplit(S.pts.length, S.params.testFrac, 1);
  S.train = train;
  S.test = test;
}

function accuracyOn(root, idxs) {
  if (!root || !idxs.length) return null;
  let ok = 0;
  for (const i of idxs) {
    const p = S.pts[i];
    if (nodeLabel(leafOf(root, feat(p.x, p.y))) === p.c) ok++;
  }
  return ok / idxs.length;
}

function ensureCellFeats() {
  if (S.cellFeat && S.cellExtras === S.params.extras) return;
  S.cellExtras = S.params.extras;
  S.cellFeat = new Array(NCELL);
  for (let j = 0; j < G; j++) {
    for (let i = 0; i < G; i++) S.cellFeat[j * G + i] = feat((i + 0.5) / G, (j + 0.5) / G);
  }
}

// ── Tree state ──────────────────────────────────────────────────────────────

/** Rebuild the builder from the current data and parameters. The tree starts ungrown. */
function resetTree() {
  stopPlay();
  S.lasers = [];
  S.wallCache = new Map();
  S.seen = new Set();
  S.litSeen = new Set();
  S.splitIds = new Set();
  S.sweep = null;
  if (!hasTwoClasses()) {
    S.builder = null;
    S.full = null;
    S.shown = null;
    refreshTree();
    return;
  }
  const X = S.train.map((i) => feat(S.pts[i].x, S.pts[i].y));
  const y = S.train.map((i) => S.pts[i].c);
  const { criterion, depth, minLeaf } = S.params;
  S.builder = makeBuilder({ X, y, K: S.k, criterion, maxDepth: depth, minLeaf });
  S.full = S.builder.root;
  refreshTree();
  computeCurve();
}

/** Recompute the displayed tree (pruned view), the node indexes, and every cell's region and prediction. */
function refreshTree() {
  if (!S.full) {
    S.shown = null;
    S.nodeById = new Map();
    S.parentOf = new Map();
    S.fullPath = null;
    S.shownLeaf = null;
    S.shownLabel = null;
    S.regionCv = null;
    S.hoverCv = null;
    S.hover = -1;
    S.dirty = true;
    return;
  }
  S.shown = S.alpha > 0 ? prune(S.full, S.alpha) : S.full;
  S.nodeById = new Map();
  for (const nd of preorder(S.full)) S.nodeById.set(nd.id, nd);
  S.parentOf = new Map();
  for (const nd of preorder(S.shown)) {
    if (nd.left) {
      S.parentOf.set(nd.left.id, nd.id);
      S.parentOf.set(nd.right.id, nd.id);
    }
  }
  ensureCellFeats();
  computeFullPaths();
  computeShownCells();
  S.regionCv = null; // rebuilt when drawing
  S.hoverCv = null;
  S.hoverKey = null;
  S.dirty = true;
}

function computeFullPaths() {
  S.fullPath = new Int32Array(NCELL * MD).fill(-1);
  for (let c = 0; c < NCELL; c++) {
    let node = S.full;
    let d = 0;
    S.fullPath[c * MD + d++] = node.id;
    while (node.left) {
      node = S.cellFeat[c][node.feature] < node.threshold ? node.left : node.right;
      S.fullPath[c * MD + d++] = node.id;
    }
  }
}

function computeShownCells() {
  const K = S.k;
  S.shownLeaf = new Int32Array(NCELL);
  S.shownProb = new Float32Array(NCELL * K);
  S.shownConf = new Float32Array(NCELL);
  S.shownLabel = new Uint8Array(NCELL);
  for (let c = 0; c < NCELL; c++) {
    const leaf = leafOf(S.shown, S.cellFeat[c]);
    const pr = leafProba(leaf);
    S.shownLeaf[c] = leaf.id;
    S.shownLabel[c] = nodeLabel(leaf);
    S.shownConf[c] = pr[S.shownLabel[c]];
    for (let k = 0; k < K; k++) S.shownProb[c * K + k] = pr[k];
  }
}

function pathHas(c, id) {
  const base = c * MD;
  for (let d = 0; d < MD; d++) {
    const v = S.fullPath[base + d];
    if (v === id) return true;
    if (v < 0) return false;
  }
  return false;
}

function regionCount(id) {
  let n = 0;
  for (let c = 0; c < NCELL; c++) if (pathHas(c, id)) n++;
  return n;
}

/** Marching squares for f(x, y) = t of node id, clipped to the cells that reach that node. Flat x1,y1,x2,y2. */
function contourFor(id) {
  if (S.wallCache.has(id)) return S.wallCache.get(id);
  const nd = S.nodeById.get(id);
  const out = [];
  if (nd && nd.left) {
    const f = nd.feature;
    const t = nd.threshold;
    const val = (x, y) => {
      if (f === 0) return x;
      if (f === 1) return y;
      if (f === 2) return x * y;
      return x * x + y * y;
    };
    const pts = [];
    const cross = (xa, ya, ga, xb, yb, gb) => {
      if ((ga < 0) !== (gb < 0)) {
        const s = ga / (ga - gb);
        pts.push(xa + s * (xb - xa), ya + s * (yb - ya));
      }
    };
    for (let j = 0; j < G; j++) {
      for (let i = 0; i < G; i++) {
        const c = j * G + i;
        if (!pathHas(c, id)) continue;
        const x0 = i / G, x1 = (i + 1) / G, y0 = j / G, y1 = (j + 1) / G;
        const ga = val(x0, y0) - t, gb = val(x1, y0) - t, gc = val(x1, y1) - t, gd = val(x0, y1) - t;
        pts.length = 0;
        cross(x0, y0, ga, x1, y0, gb);
        cross(x1, y0, gb, x1, y1, gc);
        cross(x1, y1, gc, x0, y1, gd);
        cross(x0, y1, gd, x0, y0, ga);
        if (pts.length === 4) out.push(pts[0], pts[1], pts[2], pts[3]);
        else if (pts.length === 8) out.push(pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7]);
      }
    }
  }
  const arr = Float32Array.from(out);
  S.wallCache.set(id, arr);
  return arr;
}

// ── Forest ──────────────────────────────────────────────────────────────────

function maxFeaturesValue(p) {
  if (S.maxF === 'all') return p;
  if (S.maxF === '1') return 1;
  return null; // the default: floor(sqrt(p))
}

function trainForest() {
  clearTimeout(S.forestTimer);
  if (!hasTwoClasses()) {
    S.forest = null;
    S.dirty = true;
    refreshAll();
    return;
  }
  const { criterion, depth, minLeaf, extras } = S.params;
  const X = S.train.map((i) => feat(S.pts[i].x, S.pts[i].y));
  const y = S.train.map((i) => S.pts[i].c);
  const p = extras ? 4 : 2;
  S.forest = fitForest(X, y, S.k, {
    nTrees: S.nTrees, criterion, maxDepth: depth, minLeaf, maxFeatures: maxFeaturesValue(p),
    bootstrap: S.boot, seed: S.seed,
  });
  ensureCellFeats();
  const K = S.k;
  const T = S.forest.trees.length;
  S.forestProb = new Float32Array(NCELL * K);
  S.forestConf = new Float32Array(NCELL);
  S.treeLabels = new Uint8Array(T * NCELL);
  for (let c = 0; c < NCELL; c++) {
    const f = S.cellFeat[c];
    const avg = forestProba(S.forest, f);
    let best = 0;
    for (let k = 1; k < K; k++) if (avg[k] > avg[best]) best = k;
    S.forestConf[c] = avg[best];
    for (let k = 0; k < K; k++) S.forestProb[c * K + k] = avg[k];
    for (let t = 0; t < T; t++) S.treeLabels[t * NCELL + c] = nodeLabel(leafOf(S.forest.trees[t].root, f));
  }
  S.boundaries = [];
  for (let t = 0; t < T; t++) S.boundaries.push(boundaryOf(S.treeLabels.subarray(t * NCELL, (t + 1) * NCELL)));
  // Scores: out-of-bag from the fit, test accuracy, and a single deep tree (depth 12, leaves of one) for contrast.
  const deep = makeBuilder({ X, y, K, criterion, maxDepth: 12, minLeaf: 1 }).grow();
  S.forestAcc = {
    train: forestAccuracy(S.train),
    test: forestAccuracy(S.test),
    oob: S.forest.oob,
    single: accuracyOn(deep, S.test),
  };
  S.dirty = true;
  refreshAll();
}

function forestAccuracy(idxs) {
  if (!S.forest || !idxs.length) return null;
  let ok = 0;
  for (const i of idxs) {
    const p = S.pts[i];
    const avg = forestProba(S.forest, feat(p.x, p.y));
    let best = 0;
    for (let k = 1; k < avg.length; k++) if (avg[k] > avg[best]) best = k;
    if (best === p.c) ok++;
  }
  return ok / idxs.length;
}

/** Cell-edge boundary of one tree's labels, as flat plane-coordinate segments (jagged by construction). */
function boundaryOf(lab) {
  const out = [];
  for (let j = 0; j < G; j++) {
    for (let i = 0; i < G; i++) {
      const c = j * G + i;
      if (i + 1 < G && lab[c + 1] !== lab[c]) out.push((i + 1) / G, j / G, (i + 1) / G, (j + 1) / G);
      if (j + 1 < G && lab[c + G] !== lab[c]) out.push(i / G, (j + 1) / G, (i + 1) / G, (j + 1) / G);
    }
  }
  return Float32Array.from(out);
}

// ── Accuracy against depth ──────────────────────────────────────────────────

function computeCurve() {
  if (!hasTwoClasses()) {
    S.curve = null;
    return;
  }
  const X = S.train.map((i) => feat(S.pts[i].x, S.pts[i].y));
  const y = S.train.map((i) => S.pts[i].c);
  const { criterion, minLeaf } = S.params;
  const depths = [];
  for (let d = 1; d <= 12; d++) {
    const root = makeBuilder({ X, y, K: S.k, criterion, maxDepth: d, minLeaf }).grow();
    depths.push({ d, train: accuracyOn(root, S.train), test: accuracyOn(root, S.test) });
  }
  S.curve = depths;
}

// ── Stepping and playback ───────────────────────────────────────────────────

function newSplitNodes() {
  const out = [];
  for (const nd of preorder(S.full)) if (nd.status === 'split' && !S.splitIds.has(nd.id)) out.push(nd);
  return out;
}

function stepTree() {
  if (!S.builder) return false;
  const moved = S.builder.step();
  for (const nd of newSplitNodes()) {
    S.splitIds.add(nd.id);
    S.lasers.push({ id: nd.id, t0: performance.now() });
  }
  refreshTree();
  refreshAll();
  pulsePure();
  return moved;
}

/** A leaf that has just been evaluated and is pure lights up: a short burst from its box. Once per leaf. */
function pulsePure() {
  if (!S.full) return;
  let budget = 6;
  for (const nd of preorder(S.full)) {
    if (budget <= 0) break;
    if (nd.left || nd.status !== 'leaf' || nd.impurity !== 0 || S.litSeen.has(nd.id)) continue;
    S.litSeen.add(nd.id);
    budget--;
    const el = document.querySelector(`#tl-svg [data-id="${nd.id}"]`);
    if (el && !REDUCED) {
      const r = el.getBoundingClientRect();
      burst([r.left + r.width / 2, r.top + r.height / 2], { count: 26, speed: 4 });
    }
  }
}

function growFully() {
  stopPlay();
  if (!S.builder) return;
  S.builder.grow();
  for (const nd of preorder(S.full)) if (nd.status === 'split') S.splitIds.add(nd.id);
  refreshTree();
  refreshAll();
  pulsePure();
}

function stopPlay() {
  S.playing = false;
  clearTimeout(S.playTimer);
  const b = $('tl-play');
  b.setAttribute('aria-pressed', 'false');
  b.querySelector('.btn-txt').textContent = '▶ PLAY';
}

function togglePlay() {
  if (S.playing) {
    stopPlay();
    refreshAll();
    return;
  }
  if (!S.builder || !S.builder.queue.length) return;
  S.playing = true;
  const b = $('tl-play');
  b.setAttribute('aria-pressed', 'true');
  b.querySelector('.btn-txt').textContent = '❚❚ PAUSE';
  const tick = () => {
    if (!S.playing) return;
    if (!S.builder || !S.builder.queue.length) {
      stopPlay();
      refreshAll();
      return;
    }
    stepTree();
    S.playTimer = setTimeout(tick, REDUCED ? 120 : LASER_MS + 320);
  };
  tick();
}

// ── Sweep alpha ─────────────────────────────────────────────────────────────

function setAlpha(a) {
  S.alpha = a;
  $('tl-alpha').value = String(a);
  refreshTree();
  refreshAll();
}

/** Animate α from 0 to just past the largest link strength, so the whole tree shrinks to one leaf. */
function sweepAlpha() {
  if (!S.full) return;
  const gs = linkStrengths(S.full);
  const top = Math.max(0.005, gs.length ? Math.max(...gs) * 1.05 : 0.005);
  $('tl-alpha').max = String(Math.max(0.1, Math.ceil(top * 2000) / 2000));
  S.sweep = { t0: performance.now(), dur: REDUCED ? 0 : 3500, from: 0, to: top };
  S.dirty = true;
}

function stepSweep(now) {
  if (!S.sweep) return;
  const { t0, dur, from, to } = S.sweep;
  const p = dur ? Math.min(1, (now - t0) / dur) : 1;
  S.alpha = Math.min(Number($('tl-alpha').max), from + (to - from) * p);
  $('tl-alpha').value = String(S.alpha);
  refreshTree();
  refreshAll();
  if (p >= 1) S.sweep = null;
}

// ── Plane drawing ───────────────────────────────────────────────────────────

function sizeCanvases() {
  const cv = $('tl-plane');
  const rect = cv.getBoundingClientRect();
  const size = Math.max(200, Math.round(rect.width));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.round(size * dpr);
  cv.height = Math.round(size * dpr);
  S.size = size;
  S.dpr = dpr;
  const cc = $('tl-curve');
  const r2 = cc.getBoundingClientRect();
  cc.width = Math.round(r2.width * dpr);
  cc.height = Math.round(r2.height * dpr);
  S.dirty = true;
  drawCurve();
}

function classRGB(k) {
  return RGB[k] || RGB[0];
}

/** Region shading: class colour, brighter the more confident the prediction. Rows are flipped so y grows up. */
function buildRegion() {
  const K = S.k;
  const img = new ImageData(G, G);
  const d = img.data;
  const forest = S.mode === 'forest';
  if (forest ? !S.forest : !S.shown) return null;
  for (let j = 0; j < G; j++) {
    for (let i = 0; i < G; i++) {
      const c = j * G + i;
      const label = forest ? argmaxAt(c, K) : S.shownLabel[c];
      const conf = forest ? S.forestConf[c] : S.shownConf[c];
      const b = 0.16 + 0.7 * conf;
      const col = classRGB(label);
      const o = ((G - 1 - j) * G + i) * 4;
      d[o] = BG[0] + (col[0] - BG[0]) * b;
      d[o + 1] = BG[1] + (col[1] - BG[1]) * b;
      d[o + 2] = BG[2] + (col[2] - BG[2]) * b;
      d[o + 3] = 255;
    }
  }
  const cv = document.createElement('canvas');
  cv.width = G;
  cv.height = G;
  cv.getContext('2d').putImageData(img, 0, 0);
  return cv;
}

function argmaxAt(c, K) {
  let best = 0;
  for (let k = 1; k < K; k++) if (S.forestProb[c * K + k] > S.forestProb[c * K + best]) best = k;
  return best;
}

/** Dims every cell outside the hovered node's region. */
function buildHoverOverlay(id) {
  const img = new ImageData(G, G);
  const d = img.data;
  for (let j = 0; j < G; j++) {
    for (let i = 0; i < G; i++) {
      const c = j * G + i;
      const o = ((G - 1 - j) * G + i) * 4;
      if (pathHas(c, id)) continue;
      d[o] = BG[0]; d[o + 1] = BG[1]; d[o + 2] = BG[2]; d[o + 3] = 170;
    }
  }
  const cv = document.createElement('canvas');
  cv.width = G;
  cv.height = G;
  cv.getContext('2d').putImageData(img, 0, 0);
  return cv;
}

function drawPlane(now) {
  const cv = $('tl-plane');
  const ctx = cv.getContext('2d');
  const size = S.size;
  const dpr = S.dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#02050d';
  ctx.fillRect(0, 0, size, size);

  if (!S.regionCv) S.regionCv = buildRegion();
  if (S.regionCv) {
    ctx.imageSmoothingEnabled = S.mode === 'forest';
    ctx.drawImage(S.regionCv, 0, 0, size, size);
  }

  if (S.mode === 'forest' && S.forest) {
    // Glow: the averaged probabilities again, added on top and smoothed.
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = 0.14;
    ctx.drawImage(S.regionCv, 0, 0, size, size);
    ctx.restore();
    // Jagged boundaries of the first few trees.
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(255,255,255,0.22)';
    ctx.beginPath();
    const shown = Math.min(S.showN, S.boundaries.length);
    for (let t = 0; t < shown; t++) {
      const seg = S.boundaries[t];
      for (let q = 0; q < seg.length; q += 4) {
        ctx.moveTo(seg[q] * size, (1 - seg[q + 1]) * size);
        ctx.lineTo(seg[q + 2] * size, (1 - seg[q + 3]) * size);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  if (S.mode === 'tree' && S.hover >= 0) {
    if (S.hoverKey !== S.hover || !S.hoverCv) {
      S.hoverCv = buildHoverOverlay(S.hover);
      S.hoverKey = S.hover;
    }
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(S.hoverCv, 0, 0, size, size);
  }

  drawGrid(ctx, size);

  if (S.mode === 'tree' && S.shown) {
    // Walls: every split in the view, faint, so the partition is visible.
    ctx.save();
    ctx.strokeStyle = 'rgba(0,245,255,0.42)';
    ctx.lineWidth = 1.2;
    const stack = [S.shown];
    while (stack.length) {
      const nd = stack.pop();
      if (!nd.left) continue;
      strokeSegs(ctx, contourFor(nd.id), size);
      stack.push(nd.left, nd.right);
    }
    ctx.restore();
  }

  // Lasers: each split sweeps its line across the region it cut, then fades.
  for (const L of S.lasers) {
    const el = now - L.t0;
    const p = LASER_MS ? Math.min(1, el / LASER_MS) : 1;
    const fade = el > LASER_MS ? Math.max(0, 1 - (el - LASER_MS) / Math.max(1, LINGER_MS)) : 1;
    if (fade <= 0) continue;
    const segs = contourFor(L.id);
    const upto = Math.floor(p * (segs.length / 4));
    ctx.save();
    ctx.globalAlpha = fade;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgba(255,0,160,0.8)';
    ctx.shadowColor = '#ff00a0';
    ctx.shadowBlur = 18;
    ctx.lineWidth = 9;
    strokeSegs(ctx, segs, size, upto);
    ctx.strokeStyle = '#ffffff';
    ctx.shadowColor = '#00f5ff';
    ctx.shadowBlur = 10;
    ctx.lineWidth = 2.4;
    strokeSegs(ctx, segs, size, upto);
    if (upto > 0 && upto < segs.length / 4) { // the leading edge of the sweep
      const q = (upto - 1) * 4;
      ctx.beginPath();
      ctx.arc(segs[q + 2] * size, (1 - segs[q + 3]) * size, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }
    ctx.restore();
  }

  // Points: a soft halo and a solid dot for every row (cheap: no shadow blur per point); test rows get a ring.
  for (const p of S.pts) {
    const px = p.x * size;
    const py = (1 - p.y) * size;
    ctx.globalAlpha = 0.22;
    ctx.beginPath();
    ctx.arc(px, py, 8, 0, Math.PI * 2);
    ctx.fillStyle = COLS[p.c];
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.arc(px, py, 4.4, 0, Math.PI * 2);
    ctx.fillStyle = COLS[p.c];
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#02050d';
    ctx.stroke();
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.85)';
  ctx.lineWidth = 1.2;
  for (const i of S.test) {
    const p = S.pts[i];
    ctx.beginPath();
    ctx.arc(p.x * size, (1 - p.y) * size, 7.5, 0, Math.PI * 2);
    ctx.stroke();
  }

  if (S.cursor && S.painting === false && S.mode === 'tree') {
    ctx.beginPath();
    ctx.arc(S.cursor.x * size, (1 - S.cursor.y) * size, brushRadius() * size, 0, Math.PI * 2);
    ctx.strokeStyle = S.erase ? 'rgba(255,59,59,0.8)' : 'rgba(255,230,0,0.6)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function drawGrid(ctx, size) {
  ctx.save();
  ctx.strokeStyle = 'rgba(0,245,255,0.07)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let g = 0.25; g < 1; g += 0.25) {
    ctx.moveTo(g * size, 0); ctx.lineTo(g * size, size);
    ctx.moveTo(0, (1 - g) * size); ctx.lineTo(size, (1 - g) * size);
  }
  ctx.stroke();
  ctx.fillStyle = 'rgba(74,122,155,0.9)';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillText('x', size - 12, size - 6);
  ctx.fillText('y', 6, 12);
  ctx.restore();
}

function strokeSegs(ctx, segs, size, upto = Infinity) {
  const n = Math.min(segs.length / 4, upto);
  if (!n) return;
  ctx.beginPath();
  for (let q = 0; q < n * 4; q += 4) {
    ctx.moveTo(segs[q] * size, (1 - segs[q + 1]) * size);
    ctx.lineTo(segs[q + 2] * size, (1 - segs[q + 3]) * size);
  }
  ctx.stroke();
}

// ── Tree diagram (SVG) ──────────────────────────────────────────────────────

function renderTree() {
  const svg = $('tl-svg');
  const wrap = $('tl-tree-wrap');
  if (!S.shown) {
    svg.innerHTML = '';
    wrap.classList.add('empty');
    return;
  }
  wrap.classList.remove('empty');
  const xOf = new Map();
  let leaves = 0;
  let maxD = 0;
  const place = (nd) => {
    if (nd.depth > maxD) maxD = nd.depth;
    const x = nd.left ? (place(nd.left) + place(nd.right)) / 2 : leaves++;
    xOf.set(nd.id, x);
    return x;
  };
  place(S.shown);
  const cx = (nd) => PADX + COL / 2 + xOf.get(nd.id) * COL;
  const top = (nd) => PADY + nd.depth * ROW;
  const W = PADX * 2 + Math.max(1, leaves) * COL;
  const H = PADY * 2 + maxD * ROW + NODE_H;
  const names = featNames();
  const parts = [];
  const nodes = preorder(S.shown);
  for (const nd of nodes) {
    if (!nd.left) continue;
    for (const ch of [nd.left, nd.right]) {
      const x1 = cx(nd), y1 = top(nd) + NODE_H;
      const x2 = cx(ch), y2 = top(ch);
      const my = (y1 + y2) / 2;
      parts.push(`<path class="tl-edge" data-child="${ch.id}" d="M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}"/>`);
      const lx = (x1 + x2) / 2 + (ch === nd.left ? -10 : 10);
      parts.push(`<text class="tl-edge-lbl" x="${lx}" y="${my - 3}" text-anchor="middle">${ch === nd.left ? 'yes' : 'no'}</text>`);
    }
  }
  for (const nd of nodes) {
    const x0 = cx(nd) - NODE_W / 2;
    const y0 = top(nd);
    const isLeaf = !nd.left;
    const pending = isLeaf && nd.status === 'pending';
    const n = nd.counts.reduce((a, b) => a + b, 0);
    const label = nodeLabel(nd);
    const cls = ['tl-node', pending ? 'pending' : (isLeaf ? 'leaf' : 'split')];
    if (isLeaf && !pending && nd.impurity === 0 && n > 0) cls.push('lit');
    if (!S.seen.has(nd.id)) cls.push('fresh');
    const colour = isLeaf && !pending ? COLS[label] : COLS[0];
    let rule;
    if (pending) rule = '… waiting';
    else if (isLeaf) rule = `CLASS ${'ABCD'[label]}`;
    else rule = `${names[nd.feature]} < ${nd.threshold.toFixed(2)}`;
    const meta = `${S.params.criterion} ${nd.impurity.toFixed(3)} · n ${n}`;
    let bar = '';
    if (!pending && n > 0) {
      // Coordinates are local to the node (the outer group carries its position).
      let bx = 10;
      const bw = NODE_W - 20;
      for (let k = 0; k < S.k; k++) {
        const w = (bw * nd.counts[k]) / n;
        if (w > 0) bar += `<rect x="${bx}" y="46" width="${w}" height="7" fill="${COLS[k]}" opacity="0.9"/>`;
        bx += w;
      }
      bar = `<rect class="tl-bar-bg" x="10" y="46" width="${bw}" height="7"/>${bar}`;
    }
    const counts = pending ? '' : nd.counts.map((v, k) => `${'ABCD'[k]} ${v}`).join(' · ');
    // The position lives on an outer group: a CSS transform on the inner node (used by the grow animation)
    // would replace a transform attribute on the same element.
    parts.push(
      `<g transform="translate(${x0},${y0})"><g class="${cls.join(' ')}" data-id="${nd.id}" tabindex="0" ` +
      `style="--c:${colour}">` +
      `<rect class="tl-box" width="${NODE_W}" height="${NODE_H}" rx="4"/>` +
      `<text class="tl-rule" x="10" y="19">${rule}</text>` +
      `<text class="tl-meta" x="10" y="34">${meta}</text>` +
      bar +
      `<text class="tl-meta" x="10" y="66" font-size="9">${counts}</text>` +
      `</g></g>`,
    );
    S.seen.add(nd.id);
  }
  svg.innerHTML = parts.join('');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.dataset.w = String(W);
  svg.dataset.h = String(H);
  setZoom(S.zoom);
  applyHoverClasses();
}

function setZoom(z) {
  const svg = $('tl-svg');
  S.zoom = Math.max(0.45, Math.min(1.6, z));
  const W = Number(svg.dataset.w) || 300;
  const H = Number(svg.dataset.h) || 200;
  svg.setAttribute('width', String(Math.round(W * S.zoom)));
  svg.setAttribute('height', String(Math.round(H * S.zoom)));
  $('tl-zoom-val').textContent = `${Math.round(S.zoom * 100)}%`;
}

function fitZoom() {
  const svg = $('tl-svg');
  const W = Number(svg.dataset.w) || 300;
  const avail = $('tl-tree-wrap').clientWidth - 8;
  setZoom(Math.min(1, avail / W));
}

/** Highlight the ancestors of the hovered node in the diagram. */
function applyHoverClasses() {
  const path = new Set();
  let id = S.hover;
  while (id >= 0) {
    path.add(id);
    id = S.parentOf.has(id) ? S.parentOf.get(id) : -1;
  }
  const svg = $('tl-svg');
  svg.querySelectorAll('.tl-node').forEach((g) => g.classList.toggle('hot', path.has(Number(g.dataset.id))));
  svg.querySelectorAll('.tl-edge').forEach((e) => e.classList.toggle('hot', path.has(Number(e.dataset.child))));
}

function setHover(id) {
  if (id === S.hover) return;
  S.hover = id;
  applyHoverClasses();
  S.dirty = true;
  if (id >= 0 && S.nodeById.has(id)) {
    const nd = S.nodeById.get(id);
    const n = nd.counts.reduce((a, b) => a + b, 0);
    const share = regionCount(id) / NCELL;
    const txt = nd.left
      ? `node #${id}: ${featNames()[nd.feature]} < ${nd.threshold.toFixed(3)} · ${n} rows · region covers ${pct(share)} of the plane`
      : `leaf #${id}: class ${'ABCD'[nodeLabel(nd)]} · ${n} rows · region covers ${pct(share)} of the plane`;
    $('tl-hint').textContent = txt;
  } else if (S.mode === 'tree') {
    updateHint();
  }
}

// ── Charts and readouts ─────────────────────────────────────────────────────

function drawCurve() {
  const cv = $('tl-curve');
  const ctx = cv.getContext('2d');
  const dpr = S.dpr;
  const W = cv.width / dpr;
  const H = cv.height / dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 36, r: 12, t: 12, b: 24 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const data = S.curve || [];
  let lo = 0.5;
  for (const r of data) for (const v of [r.train, r.test]) if (v != null && v < lo) lo = Math.floor(v * 10) / 10;
  const xOf = (d) => pad.l + ((d - 1) / 11) * pw;
  const yOf = (v) => pad.t + (1 - (v - lo) / (1 - lo)) * ph;
  ctx.strokeStyle = 'rgba(0,245,255,0.12)';
  ctx.lineWidth = 1;
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = 'rgba(74,122,155,1)';
  ctx.textAlign = 'right';
  for (let v = lo; v <= 1.0001; v += (1 - lo) / 4) {
    ctx.beginPath();
    ctx.moveTo(pad.l, yOf(v));
    ctx.lineTo(W - pad.r, yOf(v));
    ctx.stroke();
    ctx.fillText(`${Math.round(v * 100)}%`, pad.l - 5, yOf(v) + 3);
  }
  ctx.textAlign = 'center';
  for (const d of [1, 3, 6, 9, 12]) ctx.fillText(String(d), xOf(d), H - 8);
  ctx.fillText('max depth', pad.l + pw / 2, H - 1);
  if (!data.length) return;
  const drawLine = (key, colour) => {
    ctx.strokeStyle = colour;
    ctx.shadowColor = colour;
    ctx.shadowBlur = 8;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (const r of data) {
      if (r[key] == null) continue;
      if (!started) { ctx.moveTo(xOf(r.d), yOf(r[key])); started = true; } else ctx.lineTo(xOf(r.d), yOf(r[key]));
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    for (const r of data) {
      if (r[key] == null) continue;
      ctx.beginPath();
      ctx.arc(xOf(r.d), yOf(r[key]), 2.6, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
    }
  };
  drawLine('train', COLS[0]);
  drawLine('test', COLS[1]);
  const dm = S.params.depth;
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = 'rgba(255,230,0,0.8)';
  ctx.beginPath();
  ctx.moveTo(xOf(dm), pad.t);
  ctx.lineTo(xOf(dm), pad.t + ph);
  ctx.stroke();
  ctx.restore();
  const now = data[dm - 1];
  const note = $('tl-curve-note');
  if (now && now.test != null && now.train != null) {
    const gap = now.train - now.test;
    note.textContent = `At depth ${dm}: train ${pct(now.train)}, test ${pct(now.test)}. ` +
      (gap > 0.12 ? 'The gap is overfitting: the tree is memorising its training rows.'
        : 'Train and test agree, so the tree is not memorising yet.');
  }
}

function updateHint() {
  if (S.mode === 'tree') {
    if (!hasTwoClasses()) {
      $('tl-hint').textContent = 'Paint at least two classes on the plane, or load a preset.';
    } else if (S.hoverCell >= 0) {
      $('tl-hint').textContent = cellText(S.hoverCell);
    } else {
      $('tl-hint').textContent = 'Hover the plane to read a cell, or a tree node to see its region.';
    }
  } else if (S.hoverCell >= 0 && S.forest) {
    $('tl-hint').textContent = cellText(S.hoverCell);
  } else {
    $('tl-hint').textContent = S.forest
      ? 'Hover the plane: each tree votes, and the forest averages the votes.'
      : 'Train the forest to see its boundary.';
  }
}

function cellText(c) {
  const i = c % G, j = Math.floor(c / G);
  const x = ((i + 0.5) / G).toFixed(2), y = ((j + 0.5) / G).toFixed(2);
  if (S.mode === 'tree' && S.shownLabel) {
    const leaf = S.nodeById.get(S.shownLeaf[c]);
    const n = leaf ? leaf.counts.reduce((a, b) => a + b, 0) : 0;
    return `x ${x} · y ${y} → class ${'ABCD'[S.shownLabel[c]]} at ${pct(S.shownConf[c])} · leaf #${S.shownLeaf[c]} (${n} rows)`;
  }
  if (S.forest && S.treeLabels) {
    const T = S.forest.trees.length;
    const votes = new Array(S.k).fill(0);
    for (let t = 0; t < T; t++) votes[S.treeLabels[t * NCELL + c]]++;
    const best = argmaxAt(c, S.k);
    return `x ${x} · y ${y} → class ${'ABCD'[best]} · ${votes.map((v, k) => `${'ABCD'[k]} ${v}`).join(' · ')} of ${T} trees · ${pct(S.forestConf[c])} avg`;
  }
  return `x ${x} · y ${y}`;
}

function updateChips() {
  const chip = (id, v) => { $(id).textContent = v; };
  chip('chip-mode', S.mode === 'tree' ? 'TREE' : 'FOREST');
  if (S.mode === 'tree') {
    chip('chip-leaves', S.shown ? String(nLeaves(S.shown)) : '0');
    chip('chip-depth', S.shown ? String(treeDepth(S.shown)) : '0');
    chip('chip-train', pct(S.acc.train));
    chip('chip-test', pct(S.acc.test));
  } else {
    chip('chip-leaves', S.forest ? String(S.forest.trees.length) : '0');
    chip('chip-depth', String(S.params.depth));
    chip('chip-train', pct(S.forestAcc.train));
    chip('chip-test', pct(S.forestAcc.test));
  }
  let status = 'READY';
  if (!hasTwoClasses()) status = 'PAINT 2+ CLASSES';
  else if (S.mode === 'tree' && S.playing) status = 'GROWING';
  else if (S.mode === 'tree' && S.builder && S.builder.queue.length === 0) status = 'DONE';
  else if (S.mode === 'forest' && !S.forest) status = 'UNTRAINED';
  chip('chip-status', status);
}

function row(label, value, cls = '') {
  return `<div class="row"><span>${label}</span><b class="${cls}">${value}</b></div>`;
}

function updateReadout() {
  const box = $('tl-readout');
  if (!hasTwoClasses()) {
    box.innerHTML = '<div class="row"><span>needs two classes</span></div>';
    return;
  }
  const lines = [row('rows', `${S.pts.length} · train ${S.train.length} · test ${S.test.length}`)];
  if (S.mode === 'tree') {
    const splits = S.full ? preorder(S.full).filter((nd) => nd.status === 'split').length : 0;
    const pure = S.shown ? preorder(S.shown).filter((nd) => !nd.left && nd.impurity === 0).length : 0;
    lines.push(row('splits made', String(splits)));
    lines.push(row('leaves (pure)', `${S.shown ? nLeaves(S.shown) : 0} (${pure})`));
    lines.push(row('train accuracy', pct(S.acc.train)));
    lines.push(row('test accuracy', pct(S.acc.test)));
    if (S.acc.train != null && S.acc.test != null) {
      const gap = S.acc.train - S.acc.test;
      lines.push(gap > 0.12
        ? `<div class="warn">Overfitting: train beats test by ${(100 * gap).toFixed(0)} points. Try less depth, a bigger minimum leaf, or raise α.</div>`
        : '<div class="ok">Train and test are close: the tree generalises to rows it has not seen.</div>');
    }
  } else if (S.forest) {
    const a = S.forestAcc;
    lines.push(row('trees', `${S.forest.trees.length} · ${S.boot ? 'bootstrap' : 'no bootstrap'}`));
    lines.push(row('out-of-bag', pct(a.oob)));
    lines.push(row('test accuracy', pct(a.test)));
    lines.push(row('single deep tree', `${pct(a.single)} (depth 12)`));
    if (a.test != null && a.single != null) {
      const d = a.test - a.single;
      lines.push(d >= 0.02
        ? `<div class="ok">The forest beats the single deep tree by ${(100 * d).toFixed(0)} points.</div>`
        : `<div class="warn">On this data the single deep tree keeps up: ${(100 * Math.abs(d)).toFixed(0)} points ${d >= 0 ? 'ahead' : 'behind'}.</div>`);
    }
  } else {
    lines.push(row('forest', 'not trained yet'));
  }
  box.innerHTML = lines.join('');
}

function updateForestPanel() {
  if (!S.forest) {
    $('tl-oob').textContent = '—';
    $('tl-ftest').textContent = '—';
    $('tl-single').textContent = 'Single deep tree: —';
    $('tl-imp').innerHTML = '<div class="field-hint">Train the forest to see which features it uses.</div>';
    return;
  }
  $('tl-oob').textContent = pct(S.forestAcc.oob);
  $('tl-ftest').textContent = pct(S.forestAcc.test);
  $('tl-single').textContent = `Single deep tree (depth 12): ${pct(S.forestAcc.single)} test`;
  const names = featNames();
  const imp = S.forest.importances;
  const top = Math.max(...imp, 1e-9);
  $('tl-imp').innerHTML = imp.map((v, f) =>
    `<div class="tl-imp-row"><span>${names[f]}</span><div class="tl-imp-track"><div class="tl-imp-bar" style="width:${((100 * v) / top).toFixed(1)}%"></div></div><span>${(100 * v).toFixed(1)}%</span></div>`,
  ).join('');
}

function updateTreePanel() {
  $('tl-depth-val').textContent = String(S.params.depth);
  $('tl-minleaf-val').textContent = String(S.params.minLeaf);
  $('tl-test-val').textContent = `${Math.round(S.params.testFrac * 100)}%`;
  $('tl-alpha-val').textContent = S.alpha > 0 ? `α = ${S.alpha.toFixed(4)}` : 'off';
  const note = $('tl-prune-note');
  if (S.full && S.shown) {
    const full = nLeaves(S.full);
    const now = nLeaves(S.shown);
    note.textContent = S.alpha > 0 ? `Pruned: ${now} of ${full} leaves survive α = ${S.alpha.toFixed(4)}.`
      : 'Cost-complexity pruning: raise α to collapse the weakest splits first.';
  }
  $('tl-trees-val').textContent = String(S.nTrees);
  $('tl-show-val').textContent = `${Math.min(S.showN, S.nTrees)} of ${S.nTrees}`;
}

function refreshAll() {
  updateTreePanel();
  S.acc = S.shown
    ? { train: accuracyOn(S.shown, S.train), test: accuracyOn(S.shown, S.test) }
    : { train: null, test: null };
  renderTree();
  $('tl-tree-empty').textContent = hasTwoClasses() ? '' : 'Paint at least two classes to grow a tree.';
  $('tl-tree-wrap').classList.toggle('empty', !S.shown);
  S.regionCv = null;
  S.dirty = true;
  updateChips();
  updateForestPanel();
  updateReadout();
  drawCurve();
  updateHint();
}

// ── Mode, data and parameter changes ────────────────────────────────────────

function setMode(mode) {
  S.mode = mode;
  const treeTab = $('tab-tree');
  const forestTab = $('tab-forest');
  treeTab.classList.toggle('act', mode === 'tree');
  forestTab.classList.toggle('act', mode === 'forest');
  treeTab.setAttribute('aria-selected', String(mode === 'tree'));
  forestTab.setAttribute('aria-selected', String(mode === 'forest'));
  $('tl-tree-panel').hidden = mode !== 'tree';
  $('tl-forest-panel').hidden = mode !== 'forest';
  $('tl-diagram').hidden = mode !== 'tree';
  if (mode === 'forest' && !S.forest) trainForest();
  S.regionCv = null;
  S.dirty = true;
  refreshAll();
}

/** Points or split settings changed: resplit, rebuild the tree (ungrown), and retrain the forest if it is on screen. */
function onDataChanged() {
  splitData();
  resetTree();
  forestChanged();
}

/** The forest is stale after any data or settings change. Retrain now if it is on screen, else on next view. */
function forestChanged() {
  clearTimeout(S.forestTimer);
  S.forest = null;
  S.forestAcc = { train: null, test: null, oob: null, single: null };
  if (S.mode === 'forest') S.forestTimer = setTimeout(trainForest, 120);
  else refreshAll();
}

function loadPreset() {
  const name = $('tl-preset').value;
  const seed = Math.max(0, Math.min(2147483647, Math.floor(Number($('tl-seed').value) || 0)));
  S.seed = seed;
  const { X, y } = makePreset(name, 240, seed);
  S.k = PRESETS[name];
  if (S.cls >= S.k) selectClass(0);
  S.pts = X.map((p, i) => ({ x: p[0], y: p[1], c: y[i] }));
  syncSwatches();
  onDataChanged();
}

function selectClass(c) {
  S.cls = c;
  document.querySelectorAll('.tl-swatch').forEach((b) => {
    b.setAttribute('aria-checked', String(Number(b.dataset.c) === c));
  });
}

function syncSwatches() {
  document.querySelectorAll('.tl-swatch').forEach((b) => {
    b.hidden = Number(b.dataset.c) >= S.k;
  });
}

function brushRadius() {
  return Number($('tl-brush').value) * 0.01;
}

/** Add points under the brush (or remove them, when erasing). Random scatter inside the brush disc. */
function applyBrush(x, y) {
  const r = brushRadius();
  if (S.erase) {
    const before = S.pts.length;
    S.pts = S.pts.filter((p) => (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) > r * r);
    if (S.pts.length !== before) {
      // Re-split now: the train and test index lists point into S.pts, and the plane draws the test rows
      // on every frame of the stroke. Left stale, the indices of removed points read undefined.
      S.paintDirty = true;
      splitData();
    }
    return;
  }
  for (let q = 0; q < 4 && S.pts.length < MAX_POINTS; q++) {
    const a = Math.random() * Math.PI * 2;
    const d = Math.sqrt(Math.random()) * r;
    const px = Math.min(1, Math.max(0, x + d * Math.cos(a)));
    const py = Math.min(1, Math.max(0, y + d * Math.sin(a)));
    S.pts.push({ x: px, y: py, c: S.cls });
    S.paintDirty = true;
  }
}

function planeXY(e) {
  const r = $('tl-plane').getBoundingClientRect();
  return { x: (e.clientX - r.left) / r.width, y: 1 - (e.clientY - r.top) / r.height };
}

function cellAt(x, y) {
  const i = Math.min(G - 1, Math.max(0, Math.floor(x * G)));
  const j = Math.min(G - 1, Math.max(0, Math.floor(y * G)));
  return j * G + i;
}

function onPlaneHover(xy) {
  S.cursor = xy;
  if (xy.x < 0 || xy.x > 1 || xy.y < 0 || xy.y > 1) {
    S.cursor = null;
    S.hoverCell = -1;
    setHover(-1);
    updateHint();
    S.dirty = true;
    return;
  }
  const c = cellAt(xy.x, xy.y);
  S.hoverCell = c;
  if (S.mode === 'tree' && S.shownLeaf) setHover(S.shownLeaf[c]);
  updateHint();
  S.dirty = true;
}

// ── Setup ───────────────────────────────────────────────────────────────────

function bindEvents() {
  const cv = $('tl-plane');
  cv.addEventListener('pointerdown', (e) => {
    cv.setPointerCapture(e.pointerId);
    S.painting = true;
    const xy = planeXY(e);
    applyBrush(xy.x, xy.y);
    S.cursor = xy;
    S.dirty = true;
  });
  cv.addEventListener('pointermove', (e) => {
    const xy = planeXY(e);
    if (S.painting) {
      applyBrush(xy.x, xy.y);
      S.cursor = xy;
      S.dirty = true;
      return;
    }
    onPlaneHover(xy);
  });
  const endPaint = () => {
    if (!S.painting) return;
    S.painting = false;
    if (S.paintDirty) {
      S.paintDirty = false;
      onDataChanged();
    }
  };
  cv.addEventListener('pointerup', endPaint);
  cv.addEventListener('pointercancel', endPaint);
  cv.addEventListener('pointerleave', () => {
    if (!S.painting) onPlaneHover({ x: -1, y: -1 });
  });

  document.querySelectorAll('.tl-swatch').forEach((b) => {
    b.addEventListener('click', () => selectClass(Number(b.dataset.c)));
  });
  $('tl-erase').addEventListener('click', (e) => {
    S.erase = !S.erase;
    e.currentTarget.setAttribute('aria-pressed', String(S.erase));
  });
  $('tl-clear').addEventListener('click', () => {
    S.pts = [];
    onDataChanged();
  });
  $('tl-load').addEventListener('click', loadPreset);
  $('tl-random-seed').addEventListener('click', () => {
    $('tl-seed').value = String(Math.floor(Math.random() * 2147483647));
    loadPreset();
  });

  $('tab-tree').addEventListener('click', () => setMode('tree'));
  $('tab-forest').addEventListener('click', () => setMode('forest'));

  $('tl-criterion').addEventListener('change', (e) => {
    S.params.criterion = e.target.value;
    onDataChanged();
  });
  $('tl-depth').addEventListener('input', (e) => {
    S.params.depth = Number(e.target.value);
    onDataChanged();
  });
  $('tl-minleaf').addEventListener('input', (e) => {
    S.params.minLeaf = Number(e.target.value);
    onDataChanged();
  });
  $('tl-extras').addEventListener('change', (e) => {
    S.params.extras = e.target.checked;
    S.cellFeat = null;
    onDataChanged();
  });
  $('tl-test').addEventListener('input', (e) => {
    S.params.testFrac = Number(e.target.value);
    onDataChanged();
  });

  $('tl-step').addEventListener('click', () => {
    stopPlay();
    stepTree();
  });
  $('tl-play').addEventListener('click', togglePlay);
  $('tl-grow').addEventListener('click', growFully);
  $('tl-reset').addEventListener('click', () => {
    resetTree();
    refreshAll();
  });
  $('tl-alpha').addEventListener('input', (e) => setAlpha(Number(e.target.value)));
  $('tl-sweep').addEventListener('click', sweepAlpha);

  $('tl-trees').addEventListener('input', (e) => {
    S.nTrees = Number(e.target.value);
    $('tl-show').max = String(S.nTrees);
    if (S.showN > S.nTrees) S.showN = S.nTrees;
    $('tl-show').value = String(S.showN);
    forestChanged();
  });
  $('tl-maxf').addEventListener('change', (e) => {
    S.maxF = e.target.value;
    forestChanged();
  });
  $('tl-boot').addEventListener('change', (e) => {
    S.boot = e.target.checked;
    forestChanged();
  });
  $('tl-show').addEventListener('input', (e) => {
    S.showN = Number(e.target.value);
    S.dirty = true;
    updateTreePanel();
  });
  $('tl-train-forest').addEventListener('click', trainForest);

  $('tl-zoom-in').addEventListener('click', () => setZoom(S.zoom * 1.2));
  $('tl-zoom-out').addEventListener('click', () => setZoom(S.zoom / 1.2));
  $('tl-zoom-fit').addEventListener('click', fitZoom);

  // The diagram: hovering a node highlights its region on the plane; focus does the same for keyboard users.
  const svg = $('tl-svg');
  const nodeFrom = (e) => {
    const g = e.target.closest ? e.target.closest('.tl-node') : null;
    return g ? Number(g.dataset.id) : -1;
  };
  svg.addEventListener('pointerover', (e) => {
    const id = nodeFrom(e);
    if (id >= 0) setHover(id);
  });
  svg.addEventListener('focusin', (e) => {
    const id = nodeFrom(e);
    if (id >= 0) setHover(id);
  });
  $('tl-tree-wrap').addEventListener('pointerleave', () => setHover(-1));

  window.addEventListener('resize', () => {
    sizeCanvases();
    refreshAll();
  });
}

function frame(now) {
  requestAnimationFrame(frame);
  if (S.sweep) stepSweep(now);
  S.lasers = S.lasers.filter((L) => now - L.t0 < LASER_MS + LINGER_MS);
  const animating = S.lasers.length > 0 || S.painting || S.cursor;
  if (!S.dirty && !animating) return;
  if (!S.regionCv) S.regionCv = buildRegion();
  drawPlane(now);
  S.dirty = false;
}

function init() {
  sizeCanvases();
  bindEvents();
  selectClass(0);
  syncSwatches();
  $('tl-depth').value = String(S.params.depth);
  $('tl-minleaf').value = String(S.params.minLeaf);
  $('tl-test').value = String(S.params.testFrac);
  $('tl-trees').value = String(S.nTrees);
  $('tl-show').value = String(S.showN);
  loadPreset();
  requestAnimationFrame(frame);
}

init();
