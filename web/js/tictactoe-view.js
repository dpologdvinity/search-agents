// Drawing for tic-tac-toe: the clickable board and the agent's search tree, shared by the tic-tac-toe page
// (tictactoe.js) and the 404 page (notfound.js). The rules and the search live in tictactoe-core.js; this
// module only draws their results. The tree is drawn one recorded node at a time in search order, so the
// drawing is the search itself.

import { expand } from './sfx.js';

const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const STEP_MS = 30;          // delay between drawn tree nodes; the whole tree takes a second or two
const SVG = 'http://www.w3.org/2000/svg';
// Tree layout: nine move columns (one per square, in square order), the agent's moves in a row, and the
// replies to each move stacked underneath. Fits the viewBox 0 0 522 236 on the <svg>.
const COL = 58, ROOT_X = 4.5 * COL;
const Y_ROOT = 22, Y_MOVE = 40, MOVE_H = 22, Y_REPLY = 88, REPLY_GAP = 18, REPLY_H = 15;

const fmt = (n) => Number(n).toLocaleString();

/**
 * Builds the nine squares inside boardEl. onPick(i) runs when square i is clicked. Returns render(board, opts),
 * which updates the marks: opts.last pops the square just played, opts.locked disables every square, and
 * opts.human ('X' or 'O') names the marks in the screen-reader labels.
 */
export function createBoard(boardEl, onPick) {
  for (let i = 0; i < 9; i++) {
    const cell = document.createElement('button');
    cell.type = 'button';
    cell.className = 'ttt-cell';
    cell.addEventListener('click', () => onPick(i));
    boardEl.appendChild(cell);
  }
  return function render(board, { last = -1, locked = false, human = 'X' } = {}) {
    [...boardEl.children].forEach((cell, i) => {
      const mark = board[i];
      cell.textContent = mark === '.' ? '' : mark;
      cell.className = 'ttt-cell' + (mark === 'X' ? ' is-x' : mark === 'O' ? ' is-o' : '') + (i === last ? ' just' : '');
      cell.disabled = mark !== '.' || locked;
      cell.setAttribute('aria-label', `square ${i + 1}, ${mark === '.' ? 'empty' : mark === human ? 'you' : 'agent'}`);
    });
  };
}

const svg = (tag, attrs = {}) => {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

/**
 * The search-tree panel: treeEl is the <svg>, searchedEl and prunedEl the counters. Returns {clear, draw}.
 * draw(result, agent, done) animates a search() result for the agent's mark and calls done() when finished;
 * clear() empties the panel and cancels a drawing in progress (its done() never runs).
 */
export function createTree(treeEl, searchedEl, prunedEl) {
  let run = 0;   // bumps on every draw and clear, so a stale drawing stops at its next step

  // Records hold X's value. Shown from the agent's side, so O negates it, and a bound flips with it: an X-side
  // "at most" is an O-side "at least".
  const agentValue = (rec, agent) => (agent === 'X' ? rec.v : -rec.v);
  const bound = (rec, agent) => {
    if (!rec.b) return '';
    const atMost = (rec.b === 'hi') === (agent === 'X');
    return atMost ? '≤' : '≥';
  };
  const valueText = (rec, agent) => {
    const a = agentValue(rec, agent);
    return bound(rec, agent) + (a > 0 ? '+1' : a < 0 ? '-1' : '0');
  };
  const classOf = (rec, agent) => {
    if (rec.b) return 'refuted';
    const a = agentValue(rec, agent);
    return a > 0 ? 'win' : a < 0 ? 'loss' : 'draw';
  };

  function drawMove(rec, agent) {
    const x = rec.m * COL + COL / 2;
    treeEl.appendChild(svg('line', { x1: ROOT_X, y1: Y_ROOT + 4, x2: x, y2: Y_MOVE, class: 't-edge' }));
    const g = svg('g', { class: `t-node move ${classOf(rec, agent)}`, 'data-move': rec.m });
    g.appendChild(svg('rect', { x: x - 23, y: Y_MOVE, width: 46, height: MOVE_H, rx: 3 }));
    const t = svg('text', { x, y: Y_MOVE + 14.5, 'text-anchor': 'middle' });
    t.textContent = `${rec.m + 1}  ${valueText(rec, agent)}`;
    g.appendChild(t);
    treeEl.appendChild(g);
  }

  // Replies sit under their move, in the order the search tried them. Pruned replies are struck through.
  function drawReply(move, rec, k, agent) {
    const x = move * COL + COL / 2;
    const y = Y_REPLY + k * REPLY_GAP;
    treeEl.appendChild(svg('path', { d: `M${x} ${Y_MOVE + MOVE_H} V${y + REPLY_H / 2}`, class: 't-edge' }));
    const g = svg('g', { class: `t-node reply ${rec.pruned ? 'pruned' : classOf(rec, agent)}` });
    g.appendChild(svg('rect', { x: x - 23, y, width: 46, height: REPLY_H, rx: 2 }));
    const t = svg('text', { x, y: y + 11, 'text-anchor': 'middle' });
    t.textContent = rec.pruned ? `${rec.m + 1}` : `${rec.m + 1}  ${valueText(rec, agent)}`;
    g.appendChild(t);
    if (rec.pruned) g.appendChild(svg('line', { x1: x - 20, y1: y + REPLY_H / 2, x2: x + 20, y2: y + REPLY_H / 2, class: 't-strike' }));
    treeEl.appendChild(g);
  }

  function clear() {
    run++;
    treeEl.replaceChildren();
    searchedEl.textContent = '0';
    prunedEl.textContent = '0';
  }

  // The readout lands on the exact totals (records only cover the first two plies), the chosen move is
  // marked, and done() runs after a short pause.
  function finish(result, done, id) {
    searchedEl.textContent = fmt(result.searched);
    prunedEl.textContent = fmt(result.pruned);
    treeEl.querySelectorAll('.t-node.move').forEach((g) => {
      if (Number(g.dataset.move) === result.move) g.classList.add('chosen');
    });
    setTimeout(() => { if (id === run) done(); }, REDUCED ? 0 : 260);
  }

  function draw(result, agent, done) {
    clear();
    const id = run;
    treeEl.appendChild(svg('text', { x: ROOT_X, y: Y_ROOT, class: 't-root', 'text-anchor': 'middle' })).textContent = `AGENT (${agent}) TO MOVE`;
    const steps = [];
    for (const rec of result.tree) {
      steps.push({ rec, move: null, k: 0 });
      rec.kids.forEach((kid, k) => steps.push({ rec: kid, move: rec.m, k }));
    }
    const total = steps.length || 1;
    // Each step draws one node and moves the counters to that record's running totals.
    const show = (step, n) => {
      if (step.move === null) drawMove(step.rec, agent);
      else drawReply(step.move, step.rec, step.k, agent);
      searchedEl.textContent = fmt(step.rec.s);
      prunedEl.textContent = fmt(step.rec.p);
      if (!REDUCED) expand(n / total);
    };
    if (REDUCED) {
      steps.forEach(show);
      finish(result, done, id);
      return;
    }
    let n = 0;
    const next = () => {
      if (id !== run) return;
      if (n >= steps.length) { finish(result, done, id); return; }
      show(steps[n], n);
      n++;
      setTimeout(next, STEP_MS);
    };
    next();
  }

  return { clear, draw };
}
