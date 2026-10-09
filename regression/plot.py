"""Text plots for the command line: a fitted curve over points, and a classifier's decision map.

Plain characters only, so the output is the same in any terminal and in test snapshots. Both plots
map data coordinates to grid cells with the same rule: the value's position inside the range, rounded
to the nearest cell, with the top row holding the largest y.
"""

from __future__ import annotations

import numpy as np

from .core import poly_features, poly_features_2d, sigmoid


def _cell(value: float, lo: float, hi: float, cells: int) -> int:
    """Index in [0, cells) for value in [lo, hi], values outside the range clamped to the edges."""
    t = (value - lo) / (hi - lo)
    return int(min(cells - 1, max(0, round(t * (cells - 1)))))


def curve_plot(x, y, test_mask, w, degree: int, width: int = 70, height: int = 20) -> list[str]:
    """Scatter of (x, y) with train points as 'o' and test points as 'x', plus the fitted polynomial as '.'.

    The frame is x in [-1, 1] and y in [-1, 1]; points and curve outside that range are clipped to the
    border so a wild extrapolation still shows its direction. Points are drawn after the curve so they
    stay visible where they sit on it.
    """
    grid = [[" "] * width for _ in range(height)]
    xs = np.linspace(-1.0, 1.0, width)
    ys = poly_features(xs, degree) @ np.asarray(w, dtype=float)
    for col in range(width):
        grid[height - 1 - _cell(ys[col], -1.0, 1.0, height)][col] = "."
    for xi, yi, is_test in zip(np.asarray(x), np.asarray(y), np.asarray(test_mask), strict=True):
        col = _cell(xi, -1.0, 1.0, width)
        row = height - 1 - _cell(yi, -1.0, 1.0, height)
        grid[row][col] = "x" if is_test else "o"
    border = "+" + "-" * width + "+"
    lines = [border]
    lines += ["|" + "".join(row) + "|" for row in grid]
    lines.append(border)
    lines.append("  x from -1 to 1 (left to right), y from -1 to 1 (bottom to top); o train, x test, . fitted curve")
    return lines


def decision_plot(xy, labels, w, degree: int, width: int = 60, height: int = 20) -> list[str]:
    """Map of the classifier over the bounding box of the data, with the points on top.

    Each cell shows '#' where the model predicts class 1 (p >= 0.5) and '.' where it predicts class 0.
    Data points are drawn as '1' or '0' for their true class. The box is the data's own range plus a
    10% margin, so the boundary is visible even for a small preset such as moons.
    """
    xy = np.asarray(xy, dtype=float)
    labels = np.asarray(labels)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    pad = 0.1 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    gx = np.linspace(lo[0], hi[0], width)
    gy = np.linspace(hi[1], lo[1], height)  # top row is the largest y
    GX, GY = np.meshgrid(gx, gy)
    cells = np.column_stack([GX.ravel(), GY.ravel()])
    p = sigmoid(poly_features_2d(cells, degree) @ np.asarray(w, dtype=float)).reshape(height, width)
    grid = [["#" if p[r, c] >= 0.5 else "." for c in range(width)] for r in range(height)]
    for (px, py), lab in zip(xy, labels, strict=True):
        col = _cell(px, lo[0], hi[0], width)
        row = _cell(py, lo[1], hi[1], height)
        # _cell counts from the bottom; the grid is written top-down, so flip the row.
        grid[height - 1 - row][col] = "1" if lab == 1 else "0"
    border = "+" + "-" * width + "+"
    lines = [border]
    lines += ["|" + "".join(row) + "|" for row in grid]
    lines.append(border)
    lines.append("  '#' predicts class 1, '.' predicts class 0; points show their true class (1 or 0)")
    return lines
