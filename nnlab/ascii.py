"""Draw the decision boundary of a trained network as text, so the terminal shows what the browser shows.

Each character cell takes the predicted probability at its centre. Pink regions use '#' (confident) and
'+' (leaning), cyan regions '.' (confident) and ':' (leaning). Data points overwrite the cells they sit
in: 'O' for pink, 'o' for cyan, and '@' or '%' for test points if the caller marks them.
"""

from __future__ import annotations

import numpy as np

from .data import featurize
from .mlp import Net, predict


def render(net: Net, points: list[dict], features, noise: float = 0.0, width: int = 60, height: int = 22,
           domain: float = 1.2) -> str:
    """Return the boundary and points as a multi-line string, with y increasing upwards."""
    xs = np.linspace(-domain, domain, width)
    ys = np.linspace(domain, -domain, height)  # top row is the largest y
    grid = np.array([[featurize(x, y, features) for x in xs] for y in ys], dtype=np.float64)
    p = predict(net, grid.reshape(-1, len(features))).reshape(height, width)

    chars = [[""] * width for _ in range(height)]
    for r in range(height):
        for c in range(width):
            v = p[r, c]
            if v >= 0.85:
                chars[r][c] = "#"
            elif v >= 0.5:
                chars[r][c] = "+"
            elif v >= 0.15:
                chars[r][c] = ":"
            else:
                chars[r][c] = "."

    for pt in points:
        x = pt["x"] + noise * pt["jx"]
        y = pt["y"] + noise * pt["jy"]
        c = int(round((x + domain) / (2 * domain) * (width - 1)))
        r = int(round((domain - y) / (2 * domain) * (height - 1)))
        if 0 <= r < height and 0 <= c < width:
            chars[r][c] = "O" if pt["label"] == 1 else "o"
    return "\n".join("".join(row) for row in chars)
