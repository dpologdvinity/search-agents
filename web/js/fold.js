// Collapsible sections for the explanation pages: the HOW TO PLAY box and each topic under HOW IT WORKS.
//
// The pages keep plain markup (an h2 or h3 heading followed by its paragraphs), and this module wraps each part in a
// native <details> element at load time. <details> gives the open/close behaviour, keyboard support, and screen-reader
// state for free, and the browser's find-in-page opens a closed section when a match is inside it. Every section
// starts closed. Without JavaScript nothing is wrapped, so all the text stays visible.

const HOWTO_KEY = 'sa-howto-open'; // remembers whether the reader last left HOW TO PLAY open, across all pages

// Wrap a heading and the nodes after it in <details>, with the heading inside the <summary>.
function wrap(heading, nodes, className) {
  const details = document.createElement('details');
  details.className = className;
  const summary = document.createElement('summary');
  heading.before(details);
  summary.append(heading);
  details.append(summary, ...nodes);
  return details;
}

// HOW TO PLAY: the heading stays as the summary, and the paragraphs fold away. The reader's last choice is kept.
function foldHowTo(root) {
  for (const box of root.querySelectorAll('section.howto')) {
    const title = box.querySelector('.howto-title');
    if (!title || box.querySelector('details')) continue;
    const rest = [...box.childNodes].filter((n) => n !== title);
    const details = wrap(title, rest, 'howto-fold');
    let saved = null;
    try { saved = localStorage.getItem(HOWTO_KEY); } catch { /* storage blocked: start closed */ }
    details.open = saved === '1';
    details.addEventListener('toggle', () => {
      try { localStorage.setItem(HOWTO_KEY, details.open ? '1' : '0'); } catch { /* not saved */ }
    });
  }
}

// HOW IT WORKS: each h3.hiw-sub and everything up to the next h3.hiw-sub becomes one closed section. Text before the
// first topic (the section's introduction) stays visible. Returns the folds so the page can open them all at once.
function foldTopics(root) {
  const folds = [];
  for (const section of root.querySelectorAll('section.hiw')) {
    const headings = [...section.querySelectorAll(':scope > h3.hiw-sub')];
    for (const heading of headings) {
      const nodes = [];
      for (let n = heading.nextSibling; n && !(n.nodeType === 1 && n.matches('h3.hiw-sub')); n = n.nextSibling) {
        nodes.push(n);
      }
      folds.push(wrap(heading, nodes, 'hiw-fold'));
    }
    if (headings.length > 1) addOpenAll(section, folds.filter((f) => section.contains(f)));
  }
  return folds;
}

// A small button beside the HOW IT WORKS title that opens or closes every topic at once.
function addOpenAll(section, folds) {
  const title = section.querySelector(':scope > .section-title');
  if (!title) return;
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'fold-all';
  const sync = () => {
    const allOpen = folds.every((f) => f.open);
    button.textContent = allOpen ? '[ − close all ]' : '[ + open all ]';
    button.setAttribute('aria-label', allOpen ? 'Close every topic' : 'Open every topic');
  };
  button.addEventListener('click', () => {
    const open = !folds.every((f) => f.open);
    for (const f of folds) f.open = open;
    sync();
  });
  for (const f of folds) f.addEventListener('toggle', sync);
  title.append(' ', button);
  sync();
}

// Open the section holding the element a link points at (#id in the address), so deep links still land on text.
function openForHash() {
  const id = decodeURIComponent(location.hash.slice(1));
  const target = id && document.getElementById(id);
  for (let d = target && target.closest('details'); d; d = d.parentElement.closest('details')) d.open = true;
  if (target) target.scrollIntoView();
}

export function foldSections(root = document) {
  foldHowTo(root);
  const folds = foldTopics(root);
  // Printing a page prints every topic, then restores what the reader had open.
  let before = [];
  window.addEventListener('beforeprint', () => {
    before = folds.map((f) => f.open);
    for (const f of folds) f.open = true;
  });
  window.addEventListener('afterprint', () => folds.forEach((f, i) => { f.open = before[i] ?? f.open; }));
  window.addEventListener('hashchange', openForHash);
  openForHash();
}
