// Entry point for the explanation sections on the game pages. It typesets the formulas, sets up the pseudocode
// boxes, and folds HOW TO PLAY and each HOW IT WORKS topic into collapsible sections. Pages load it as a module after the deferred KaTeX script, so the page is parsed before this runs. If the
// document is still loading, it waits for DOMContentLoaded.
import { renderMath } from './math.js';
import { enhancePseudo } from './pseudo.js';
import { foldSections } from './fold.js';

function start() {
  renderMath(document);
  enhancePseudo(document);
  foldSections(document);
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
else start();
