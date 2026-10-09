# KaTeX (vendored)

Version 0.19.0 from the npm package `katex`, MIT licence (see `LICENSE`).

Copied from the package `dist/` directory, unmodified:

- `katex.min.js`: the browser build (UMD; sets `window.katex`)
- `katex.min.css`: the stylesheet, unmodified. It still lists `woff` and `ttf` fallbacks; browsers use the first format they support, so only `woff2` is shipped.
- `fonts/*.woff2`: the 20 font files, the only format copied.

Pages load it with `<link rel="stylesheet" href="vendor/katex/katex.min.css">` and `<script defer src="vendor/katex/katex.min.js"></script>`. Formulas are rendered by `web/js/math.js`.
