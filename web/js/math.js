// Formula rendering for the explanation sections. Markup is <span class="math">TEX</span> for an inline formula
// and <div class="math math-block">TEX</div> for a display formula. The TeX source is the element's text, which the
// browser has already unescaped from the HTML. KaTeX (web/vendor/katex, loaded before this module) replaces the text
// with typeset maths. A formula that fails to parse keeps its source text, and the error goes to the console.

// Render every formula under root. A formula is rendered once; running this again is harmless.
export function renderMath(root = document) {
  if (!window.katex) {
    console.error('KaTeX is not loaded, so formulas stay as TeX source');
    return;
  }
  for (const el of root.querySelectorAll('.math')) {
    if (el.dataset.mathDone) continue;
    const source = el.textContent;
    try {
      window.katex.render(source, el, { displayMode: el.classList.contains('math-block'), throwOnError: true });
      el.dataset.mathDone = '1';
    } catch (err) {
      console.error('KaTeX could not render', source, err);
    }
  }
}
