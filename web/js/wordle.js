// Wordle page.
//
// The server holds the hidden answer and scores each guess (POST /api/wordle/guess). For the solver it ranks
// the candidates still possible (GET /api/wordle/hint): for each allowed guess, the expected bits of its
// feedback, the worst-case bucket, and the expected words left. This page draws that ranking as bars.
// In AI PLAYS mode the AI types the top-ranked word, letter by letter, so what the bars show is what it plays.
//
// The answer never reaches the page until the game is over, so the page cannot cheat by reading it.

import { getJSON, postJSON } from './api.js';
import { banner, burst, pop, shake } from './fx.js';

const $ = (id) => document.getElementById(id);
const ROWS = 6;
const COLS = 5;
const KEY_ROWS = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm'];
const TYPE_MS = 170; // pause between letters when the AI types
const FLIP_MS = 260; // stagger between tiles when a row is revealed
const BEST_RANK = { x: 0, y: 1, g: 2 }; // a key keeps the best colour its letter has earned
const CLASS_OF = { g: 'g', y: 'y', '.': 'x' }; // server feedback symbols to tile classes
const STRATEGY_NAME = { entropy: 'max entropy', minimax: 'minimax', random: 'random consistent' };

const state = {
  meta: null,
  gameId: null,
  guesses: [],      // {word, feedback, bits_gained, expected_bits, remaining} for each guess made
  typed: '',        // letters in the current row
  hint: null,       // last /hint response for this game
  keyState: {},     // letter -> 'g' | 'y' | 'x', the best colour seen so far
  mode: 'you',      // 'you' or 'ai'
  busy: false,      // a request or an animation is running; input waits
  over: false,
  revealRow: -1,    // the row being revealed; render() leaves it alone
  aiOn: false,      // AI AUTO is running
  aiLoopRunning: false,
  tiles: [],        // tile elements, [row][col]
  rowEls: [],
  keys: new Map(),  // letter -> key element
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const speedMs = () => Number($('speed').value);
const strategy = () => $('strategy').value;

// ── Board and keyboard ───────────────────────────────────────────────────

function buildBoard() {
  const board = $('wd-board');
  board.innerHTML = '';
  state.tiles = [];
  state.rowEls = [];
  for (let r = 0; r < ROWS; r++) {
    const row = document.createElement('div');
    row.className = 'wd-row';
    const tiles = [];
    for (let c = 0; c < COLS; c++) {
      const el = document.createElement('div');
      el.className = 'wd-tile';
      el.setAttribute('aria-label', `row ${r + 1}, letter ${c + 1}, empty`);
      row.appendChild(el);
      tiles.push(el);
    }
    board.appendChild(row);
    state.tiles.push(tiles);
    state.rowEls.push(row);
  }
}

function buildKeys() {
  const wrap = $('wd-keys');
  wrap.innerHTML = '';
  KEY_ROWS.forEach((letters, i) => {
    const row = document.createElement('div');
    row.className = 'wd-keyrow';
    if (i === KEY_ROWS.length - 1) row.appendChild(keyButton('ENTER', 'wide', submit));
    for (const ch of letters) row.appendChild(keyButton(ch.toUpperCase(), '', () => addLetter(ch)));
    if (i === KEY_ROWS.length - 1) row.appendChild(keyButton('⌫', 'wide', backspace));
    wrap.appendChild(row);
  });
}

function keyButton(label, extra, onClick) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = `wd-key ${extra}`.trim();
  b.textContent = label;
  b.setAttribute('aria-label', label === '⌫' ? 'backspace' : label.toLowerCase());
  b.onclick = onClick;
  if (label.length === 1) state.keys.set(label.toLowerCase(), b);
  return b;
}

// Paint the board from state. The row being revealed is skipped, so its flip animation is not cut short.
function render() {
  state.tiles.forEach((tiles, r) => {
    if (r === state.revealRow) return;
    tiles.forEach((el, c) => {
      if (r < state.guesses.length) {
        const g = state.guesses[r];
        setTile(el, g.word[c], CLASS_OF[g.feedback[c]]);
      } else if (r === state.guesses.length) {
        const ch = state.typed[c];
        const cur = !state.over && !state.busy && c === state.typed.length && state.mode === 'you';
        setTile(el, ch || '', ch ? 'filled' : '', cur ? 'cur' : '');
      } else {
        setTile(el, '', '');
      }
    });
  });
  for (const [letter, el] of state.keys) {
    el.className = `wd-key ${state.keyState[letter] || ''}`.trim();
    el.disabled = state.busy || state.over || state.mode !== 'you';
  }
  $('wd-keys').querySelectorAll('.wide').forEach((b) => { b.disabled = state.busy || state.over || state.mode !== 'you'; });
  $('chip-guess').textContent = `${state.guesses.length} / ${ROWS}`;
  $('chip-status').textContent = statusText();
  $('btn-ai-step').disabled = state.busy || state.over || state.mode !== 'ai';
  $('btn-ai-auto').disabled = state.over || state.mode !== 'ai' || (state.busy && !state.aiOn);
  $('btn-ai-auto').setAttribute('aria-pressed', String(state.aiOn));
  $('btn-ai-auto').querySelector('.btn-txt').textContent = state.aiOn ? '■ STOP' : '▶▶ AI AUTO';
  $('mode-you').setAttribute('aria-pressed', String(state.mode === 'you'));
  $('mode-ai').setAttribute('aria-pressed', String(state.mode === 'ai'));
  $('mode-you').classList.toggle('act', state.mode === 'you');
  $('mode-ai').classList.toggle('act', state.mode === 'ai');
}

// Set a tile's letter and classes. Extra classes (like 'cur') are optional.
function setTile(el, letter, cls, extra = '') {
  el.textContent = letter.toUpperCase();
  el.className = `wd-tile ${cls} ${extra}`.trim();
  el.setAttribute('aria-label', letter ? `letter ${letter}${cls ? `, ${cls === 'g' ? 'green' : cls === 'y' ? 'yellow' : cls === 'x' ? 'gray' : 'typed'}` : ''}` : 'empty');
}

function statusText() {
  if (state.over) return state.guesses.at(-1)?.feedback === 'ggggg' ? 'SOLVED' : 'OUT OF GUESSES';
  if (state.busy) return state.aiOn || state.mode === 'ai' ? 'AI THINKING' : 'SCORING';
  return state.mode === 'ai' ? 'AI READY' : 'YOUR TURN';
}

// ── Typing ───────────────────────────────────────────────────────────────

function canType() {
  return state.mode === 'you' && !state.busy && !state.over;
}

function addLetter(ch) {
  if (!canType() || state.typed.length >= COLS) return;
  state.typed += ch;
  render();
  const row = state.rowEls[state.guesses.length];
  row.querySelectorAll('.wd-tile')[state.typed.length - 1].classList.add('bump');
}

function backspace() {
  if (!canType()) return;
  state.typed = state.typed.slice(0, -1);
  render();
}

function setMsg(text, cls = '') {
  const el = $('wd-msg');
  el.textContent = text;
  el.className = `wd-msg field-hint ${cls}`.trim();
}

function log(text, cls = 'log-info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

// ── Guessing ─────────────────────────────────────────────────────────────

// Score one word with the server. Shakes the row and says why on a bad word; otherwise reveals the row.
async function guess(word, source) {
  const row = state.guesses.length;
  let result;
  try {
    result = await postJSON('/api/wordle/guess', { game_id: state.gameId, word });
  } catch (err) {
    // A word the server rejects (not in the list, or a stale game): shake the row, say why, and let the player retry.
    shakeRow(row);
    setMsg(err.message, 'err');
    state.busy = false;
    render();
    return false;
  }
  state.guesses.push({ word, feedback: result.feedback, bits_gained: result.bits_gained,
                       expected_bits: result.expected_bits, remaining: result.remaining });
  state.typed = '';
  state.revealRow = row;
  state.busy = true;
  render();
  await reveal(row, word, result.feedback);
  state.revealRow = -1;
  for (let c = 0; c < COLS; c++) {
    const letter = word[c];
    const code = CLASS_OF[result.feedback[c]];
    if (!state.keyState[letter] || BEST_RANK[code] > BEST_RANK[state.keyState[letter]]) state.keyState[letter] = code;
  }
  const gained = result.bits_gained === null ? '—' : `${result.bits_gained.toFixed(2)}`;
  $('chip-last').textContent = `${gained} / ${result.expected_bits.toFixed(2)}`;
  $('chip-left').textContent = String(result.remaining);
  log(`${source}: ${word.toUpperCase()} → ${result.feedback}  gained ${gained} bits (solver expected ${result.expected_bits.toFixed(2)}), ${result.remaining} left`,
      result.solved ? 'log-best' : 'log-move');
  pop(state.rowEls[row], gained === '—' ? '' : `+${gained} bits`, '#00f5ff');
  state.over = result.over;
  if (result.solved) {
    burst(state.rowEls[row], { count: 90 });
    banner('SOLVED', `${result.guess_no} ${result.guess_no === 1 ? 'guess' : 'guesses'}`, '#00ff88');
    setMsg(`Solved in ${result.guess_no}. The word was ${word.toUpperCase()}.`, 'ok');
  } else if (result.over) {
    shake($('wd-board'), 'big');
    banner('OUT OF GUESSES', `the word was ${result.answer.toUpperCase()}`, '#ff00a0');
    setMsg(`The word was ${result.answer.toUpperCase()}. Start a new one.`, 'err');
    log(`the word was ${result.answer.toUpperCase()}`, 'log-err');
  } else {
    setMsg(`${result.remaining} words still possible.`);
  }
  state.busy = false;
  render();
  if (!state.over) await loadHint();
  render();
  return true;
}

// Flip each tile of the row in turn; the colour changes at the middle of each flip, when the tile is edge-on.
function reveal(row, word, feedback) {
  const tiles = state.tiles[row];
  return new Promise((resolve) => {
    tiles.forEach((el, c) => {
      const delay = c * FLIP_MS;
      setTile(el, word[c], 'x', 'flip');
      el.style.setProperty('--delay', `${delay}ms`);
      setTimeout(() => setTile(el, word[c], CLASS_OF[feedback[c]], 'flip'), delay + 240);
    });
    setTimeout(resolve, (COLS - 1) * FLIP_MS + 520);
  });
}

// Restart the shake animation on a row (the class is removed after it has played).
function shakeRow(row) {
  const el = state.rowEls[row];
  el.classList.remove('shake');
  void el.offsetWidth;
  el.classList.add('shake');
  setTimeout(() => el.classList.remove('shake'), 450);
}

function submit() {
  if (!canType()) return;
  if (state.typed.length < COLS) {
    shakeRow(state.guesses.length);
    setMsg('Not enough letters.', 'err');
    return;
  }
  state.busy = true;
  render();
  const word = state.typed;
  guess(word, 'you');
}

// ── Solver ───────────────────────────────────────────────────────────────

async function loadHint() {
  if (!state.gameId) return;
  const data = await getJSON('/api/wordle/hint', { game_id: state.gameId, strategy: strategy(), top: 8 });
  state.hint = data;
  renderHint(data);
}

// The bars: one per guess, length = expected bits. The top row is the one the solver would play.
function renderHint(data) {
  const bars = $('wd-bars');
  bars.innerHTML = '';
  const maxBits = Math.max(...data.options.map((o) => o.bits), 1e-9);
  data.options.forEach((o, i) => {
    const row = document.createElement('div');
    row.className = `wd-bar-row${i === 0 ? ' best' : ''}${o.candidate ? ' cand' : ''}`;
    row.innerHTML = `<span class="w"></span><span class="wd-bar-track"><span class="wd-bar-fill"></span></span>`
      + `<span class="v"></span><span class="v"></span>`;
    row.querySelector('.w').textContent = o.word.toUpperCase();
    row.querySelector('.wd-bar-fill').style.width = `${(o.bits / maxBits) * 100}%`;
    const [bitsEl, leftEl] = row.querySelectorAll('.v');
    bitsEl.textContent = `${o.bits.toFixed(2)} bits`;
    leftEl.textContent = `${o.expected_left.toFixed(1)} left`;
    row.title = `worst bucket ${o.worst}${o.candidate ? '; it could be the answer' : ''}`;
    bars.appendChild(row);
  });
  const top = data.options[0];
  const sName = STRATEGY_NAME[data.strategy];
  $('think-title').textContent = `${data.remaining} answers still possible. Ranked by ${sName} over ${state.meta.guesses} allowed guesses.`;
  if (data.strategy === 'random') {
    $('think-text').textContent = `Random consistent: the AI picks any word the feedback still allows. These are a few of them, in random order.`;
  } else if (top) {
    const note = data.strategy === 'minimax'
      ? `It minimises the largest bucket, so its worst case is ${top.worst} words.`
      : `Its ${top.bits.toFixed(2)} expected bits split the candidates into buckets of ${top.expected_left.toFixed(1)} words on average.`;
    $('think-text').textContent = `Best: ${top.word.toUpperCase()}${top.candidate ? ' (it could be the answer)' : ''}. ${note}`;
  }
  $('chip-strategy').textContent = sName.toUpperCase();
  $('left-title').textContent = `${data.remaining} WORDS STILL POSSIBLE`;
  const chips = $('wd-chips');
  chips.innerHTML = '';
  if (data.words) {
    for (const w of data.words) {
      const chip = document.createElement('span');
      chip.className = 'wd-chip';
      chip.textContent = w.toUpperCase();
      chips.appendChild(chip);
    }
  } else {
    chips.innerHTML = '<span class="field-hint">Too many to list. Make a guess to narrow them down.</span>';
  }
  $('chip-left').textContent = String(data.remaining);
}

// The AI's move: take the top-ranked word from the hint, type it letter by letter, then submit it.
async function aiMove() {
  if (state.over || state.busy || state.mode !== 'ai') return false;
  state.busy = true;
  render();
  try {
    if (!state.hint) await loadHint();
    const word = state.hint.options[0].word;
    setMsg(`The solver plays ${word.toUpperCase()}.`);
    state.typed = '';
    for (const ch of word) {
      state.typed += ch;
      render();
      await sleep(TYPE_MS);
    }
    state.busy = false;
    render();
    return await guess(word, 'AI');
  } catch (err) {
    state.busy = false;
    setMsg(err.message, 'err');
    log(`✗ ${err.message}`, 'log-err');
    return false;
  } finally {
    state.busy = false;
    render();
  }
}

async function aiAuto() {
  if (state.aiLoopRunning) return;
  state.aiLoopRunning = true;
  while (state.aiOn && !state.over) {
    const moved = await aiMove();
    if (!moved) break;
    if (state.over) break;
    await sleep(speedMs());
  }
  state.aiOn = false;
  state.aiLoopRunning = false;
  render();
}

// ── Game flow ────────────────────────────────────────────────────────────

async function newGame() {
  state.aiOn = false;
  state.gameId = null;
  try {
    const data = await postJSON('/api/wordle/new', {});
    state.gameId = data.game_id;
  } catch (err) {
    setMsg(`Server unreachable (${err.message}).`, 'err');
    log(`✗ Server unreachable (${err.message}).`, 'log-err');
    return;
  }
  Object.assign(state, { guesses: [], typed: '', hint: null, keyState: {}, over: false, busy: false, revealRow: -1 });
  $('log').innerHTML = '';
  $('chip-last').textContent = '—';
  $('chip-left').textContent = String(state.meta.answers);
  state.keys.forEach((el) => { el.className = 'wd-key'; });
  buildBoard();
  setMsg(state.mode === 'ai' ? 'The solver is ready. Press AI MOVE, or AI AUTO to watch it finish.'
                             : 'Type a five-letter word, or use the screen keyboard.');
  log(`new word: ${state.meta.answers} possible answers, ${state.meta.guesses} allowed guesses`);
  render();
  try {
    await loadHint();
  } catch (err) {
    setMsg(err.message, 'err');
  }
  render();
}

function setMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  state.typed = '';
  if (mode === 'you') state.aiOn = false;
  setMsg(mode === 'ai' ? 'The solver plays. Press AI MOVE for one guess, or AI AUTO to let it finish.'
                       : 'Your turn. Type a five-letter word.');
  render();
}

function onKey(e) {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const tag = e.target.tagName;
  if (tag === 'SELECT' || tag === 'INPUT' || tag === 'TEXTAREA') return;
  if (e.key === 'Enter') { e.preventDefault(); submit(); }
  else if (e.key === 'Backspace') { e.preventDefault(); backspace(); }
  else if (/^[a-zA-Z]$/.test(e.key)) addLetter(e.key.toLowerCase());
}

async function init() {
  buildBoard();
  buildKeys();
  try {
    state.meta = await getJSON('/api/wordle/meta');
  } catch (err) {
    $('chip-status').textContent = 'OFFLINE';
    setMsg(`Server unreachable (${err.message}).`, 'err');
    log(`✗ Server unreachable (${err.message}).`, 'log-err');
    return;
  }
  $('btn-new').onclick = () => { if (!state.busy) newGame(); };
  $('mode-you').onclick = () => setMode('you');
  $('mode-ai').onclick = () => setMode('ai');
  $('btn-ai-step').onclick = () => { if (!state.aiOn) aiMove(); };
  $('btn-ai-auto').onclick = () => {
    if (state.aiOn) { state.aiOn = false; render(); return; }
    state.aiOn = true;
    render();
    aiAuto();
  };
  $('strategy').onchange = async () => {
    if (!state.gameId || state.over) return;
    try { await loadHint(); } catch (err) { setMsg(err.message, 'err'); }
    render();
  };
  document.addEventListener('keydown', onKey);
  await newGame();
}

init();
