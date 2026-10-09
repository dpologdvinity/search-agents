// Pseudocode boxes for the explanation sections. A box is <div class="pseudo"> holding a required
// <pre class="pseudo-simple"> and, when the algorithm has a longer version, a <pre class="pseudo-full" hidden>.
// tokenizePseudo() splits the text of a block into typed tokens so it can be coloured like Python source.
// enhancePseudo() colours every box on the page and adds the + toggle that shows the full version.

// The grammar, in the order the alternatives are tried. A comment runs from # to the end of the line. A procedure
// name is written in capitals, with hyphens between capitalised parts (ALPHA-BETA); the lookahead keeps mixed-case
// words such as "Man" out of it. Other identifiers are keywords, builtins or plain names. Numbers, Greek letters
// and the arrow ← get their own classes, and so does the real minus sign −, which the CSS draws in a font whose
// minus is long enough to read as minus. Anything else, including whitespace and the ASCII hyphen, is plain text.
const TOKEN = new RegExp([
  '(#[^\\n]*)',                                                    // 1 comment
  '([A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*(?![A-Za-z0-9_]))',         // 2 procedure name
  '([A-Za-z_][A-Za-z0-9_]*)',                                       // 3 identifier
  '(\\d+(?:\\.\\d+)?)',                                             // 4 number
  '([\\u0370-\\u03FF])',                                           // 5 Greek letter
  '(←)',                                                           // 6 arrow
  '(−)',                                                           // 7 minus sign (U+2212)
  '([\\s\\S])',                                                    // 8 anything else, one character at a time
].join('|'), 'gu');

const KEYWORDS = new Set([
  'function', 'if', 'else', 'otherwise', 'for', 'each', 'in', 'while', 'return', 'break', 'try', 'raise',
  'and', 'or', 'not', 'stop', 'store', 'constants',
]);
const BUILTINS = new Set(['max', 'min', 'sqrt', 'log', 'argmax', 'abs', 'len', 'sum']);

// Split source text into [{ type, text }] tokens. The types are 'comment', 'proc', 'keyword', 'builtin', 'number',
// 'greek', 'arrow', 'minus' and 'text'. Adjacent plain characters are merged into one 'text' token. The concatenated
// token texts always equal the input, so the highlighted box shows exactly the same characters.
export function tokenizePseudo(source) {
  const tokens = [];
  const push = (type, text) => {
    const last = tokens[tokens.length - 1];
    if (type === 'text' && last && last.type === 'text') last.text += text;
    else tokens.push({ type, text });
  };
  // A comment is split at each minus sign. The minus keeps the comment's colour and style, and it also takes the
  // minus font, so it reads as a real minus inside the comment too.
  const pushComment = (text) => {
    for (const part of text.split(/(−)/)) {
      if (part === '−') tokens.push({ type: 'minus', text: part, within: 'comment' });
      else if (part !== '') tokens.push({ type: 'comment', text: part });
    }
  };
  for (const match of source.matchAll(TOKEN)) {
    const [text, comment, proc, ident, number, greek, arrow, minus] = match;
    if (comment !== undefined) pushComment(text);
    else if (proc !== undefined) push('proc', text);
    else if (ident !== undefined) {
      if (KEYWORDS.has(ident)) push('keyword', text);
      else if (BUILTINS.has(ident)) push('builtin', text);
      else if (ident === 'infinity') push('number', text);
      else push('text', text);
    }
    else if (number !== undefined) push('number', text);
    else if (greek !== undefined) push('greek', text);
    else if (arrow !== undefined) push('arrow', text);
    else if (minus !== undefined) push('minus', text);
    else push('text', text);
  }
  return tokens;
}

// Replace the text of one <pre> with coloured spans. textContent and createElement are used, never innerHTML,
// so the source can never inject markup.
function highlightInto(pre) {
  const tokens = tokenizePseudo(pre.textContent);
  pre.textContent = '';
  for (const { type, text, within } of tokens) {
    if (type === 'text') {
      pre.append(text);
      continue;
    }
    const span = document.createElement('span');
    // A token inside another token (a minus inside a comment) carries both classes.
    span.className = within ? `${TOKEN_CLASS[type]} ${TOKEN_CLASS[within]}` : TOKEN_CLASS[type];
    if (type === 'minus' || type === 'arrow') {
      // These glyphs go in a one-column box in the page's monospace face, and the glyph itself is set in KaTeX_Main
      // inside it. The box keeps the columns; the glyph is the one that reads as a minus or an arrow.
      const glyph = document.createElement('span');
      glyph.className = 'tk-glyph';
      glyph.textContent = text;
      span.append(glyph);
    } else {
      span.textContent = text;
    }
    pre.append(span);
  }
}

// CSS class for each token type; the colours are in web/app.css.
const TOKEN_CLASS = {
  comment: 'tk-com', proc: 'tk-proc', keyword: 'tk-kw', builtin: 'tk-builtin', number: 'tk-num', greek: 'tk-greek', arrow: 'tk-arrow',
  minus: 'tk-op',
};

// The toggle is a bare plus sign, which becomes a minus sign while the full version is shown. Its name and tooltip
// say the action it will do, the same as the glyph does for a sighted reader.
const GLYPH_SHOW_FULL = '+';
const GLYPH_SHOW_SIMPLE = '−';
const NAME_SHOW_FULL = 'Show the full version';
const NAME_SHOW_SIMPLE = 'Show the simple version';
let nextId = 0;

// Colour every box under root and wire up its toggle. A box is enhanced once; running this again is harmless.
export function enhancePseudo(root = document) {
  for (const box of root.querySelectorAll('.pseudo')) {
    if (box.dataset.pseudoReady) continue;
    box.dataset.pseudoReady = '1';
    const simple = box.querySelector(':scope > .pseudo-simple');
    const full = box.querySelector(':scope > .pseudo-full');
    for (const pre of [simple, full]) {
      if (!pre) continue;
      // tabindex lets keyboard users scroll a box that overflows horizontally.
      pre.tabIndex = 0;
      highlightInto(pre);
    }
    if (!simple || !full) continue;

    if (!full.id) full.id = `pseudo-full-${++nextId}`;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'pseudo-toggle';
    button.setAttribute('aria-controls', full.id);
    button.textContent = GLYPH_SHOW_FULL;
    button.setAttribute('aria-label', NAME_SHOW_FULL);
    button.title = NAME_SHOW_FULL;
    button.setAttribute('aria-expanded', 'false');
    button.addEventListener('click', () => {
      // The full version is shown only while the button reports expanded. The glyph and name give the action it will do.
      const showingFull = button.getAttribute('aria-expanded') !== 'true';
      full.hidden = !showingFull;
      simple.hidden = showingFull;
      button.setAttribute('aria-expanded', String(showingFull));
      button.textContent = showingFull ? GLYPH_SHOW_SIMPLE : GLYPH_SHOW_FULL;
      button.setAttribute('aria-label', showingFull ? NAME_SHOW_SIMPLE : NAME_SHOW_FULL);
      button.title = showingFull ? NAME_SHOW_SIMPLE : NAME_SHOW_FULL;
    });
    // A box with code references keeps its toggle in the same corner bar, to the right of the references.
    const bar = box.querySelector(':scope > .pseudo-bar');
    if (bar) bar.append(button);
    else box.prepend(button);
  }
}
