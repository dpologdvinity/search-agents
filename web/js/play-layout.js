// Board fitting for the game pages that use a .play-layout (the board column beside the agent panel).
//
// play.css sizes each board from --play-board-size, with a 100svh-based fallback for before this script runs.
// fitPlayLayouts() replaces that guess with a measured size: the window height minus everything that has to fit
// on screen with the board (the header above it, the hints and action rows below it), clamped to 140-600px. On
// phones the header alone takes a third of the screen, so a height fit left boards a few pixels per cell; there
// the board takes the game column's width instead (capped by the screen height, for phones held sideways) and the
// page scrolls to the actions. It refits whenever those parts change size.

// Selectors for the board element on each page; the size is fitted to the first one found in the game column.
const BOARDS = '.rt-map, .ck-board, .cp-stage, .nono-wrap, .lr-map, .ms-board, .qn-board, .qn-big, .rv-map, .gh-map, .tt-board, .np-grid';
const MIN_BOARD = 140;          // px; below this a board stops being playable, so the page scrolls instead
const MAX_BOARD = 600;          // px; larger boards only waste space on big screens
const MARGIN = 12;              // px kept free under the board column on desktop, and above and below a sideways phone's board
const PHONE_WIDTH = 760;        // px; at or below this the agent panel sits under the board, not beside it
// On touch screens the browser bar slides in and out while scrolling, which changes innerHeight. Refitting then
// would make the board jump as the user scrolls down to the agent panel, so touch screens refit on window
// resizes only when the width changes (rotation, split screen).
const TOUCH = matchMedia('(pointer: coarse)').matches;

/** The block whose bottom edge the space below the board is measured from. When the board shares a row with
 * other elements (Tetris's side panel), that is the row: measuring from the board itself would count a taller
 * neighbour as space below the board, which grows as the board shrinks, so the board would settle smaller than
 * the screen allows. A row is a non-wrapping horizontal flexbox between the board and the game column; a
 * wrapping one may put its items under the board, and those really are below it. */
function rowOf(board, column) {
  for (let el = board.parentElement; el && el !== column; el = el.parentElement) {
    const style = getComputedStyle(el);
    if (style.display.includes('flex') && style.flexDirection.startsWith('row') && style.flexWrap === 'nowrap') return el;
  }
  return board;
}

/** Fit every .play-layout under root now, and again whenever its header, column or the window changes size. */
export function fitPlayLayouts(root = document) {
  const layouts = [...root.querySelectorAll('.play-layout')].filter((l) => l.querySelector('.proj-game-col'));
  if (!layouts.length) return;
  let frame = 0;
  let height = window.innerHeight;   // the height boards are fitted to; see TOUCH
  let width = window.innerWidth;
  const fit = () => {
    frame = 0;
    for (const layout of layouts) {
      const top = layout.getBoundingClientRect().top + window.scrollY;
      layout.style.setProperty('--play-available-height', `${Math.max(180, height - top - MARGIN)}px`);
      const main = layout.querySelector('.proj-game-col');
      const board = main.querySelector(BOARDS);
      if (!board || !board.getClientRects().length) continue;   // hidden mode (e.g. one of the N-Queens tabs)
      let available;
      if (window.innerWidth <= PHONE_WIDTH) {
        const style = getComputedStyle(main);
        const width = main.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
        available = Math.min(width, height - 2 * MARGIN);
      } else {
        const rect = board.getBoundingClientRect();
        const below = main.getBoundingClientRect().bottom - rowOf(board, main).getBoundingClientRect().bottom;
        available = height - (rect.top + window.scrollY) - below - MARGIN;
      }
      layout.style.setProperty('--play-board-size', `${Math.max(MIN_BOARD, Math.min(MAX_BOARD, Math.floor(available)))}px`);
    }
  };
  const schedule = () => { if (!frame) frame = requestAnimationFrame(fit); };
  const observer = new ResizeObserver(schedule);
  for (const layout of layouts) observer.observe(layout.querySelector('.proj-game-col'));
  const header = root.querySelector('.compact-play .proj-header');
  if (header) observer.observe(header);
  window.addEventListener('resize', () => {
    if (TOUCH && window.innerWidth === width) return;
    width = window.innerWidth;
    height = window.innerHeight;
    schedule();
  });
  root.addEventListener('toggle', schedule, true);
  document.fonts.ready.then(schedule);
  schedule();
}
