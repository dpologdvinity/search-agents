"""ASCII contour map with the optimizers' trails, for the terminal.

The loss is drawn on a log scale (log1p of the excess over the minimum), so the valleys stand out.
Each optimizer leaves a trail of its lowercase letter, its final position is the uppercase letter, the
start is 'o', known global minima are '*', and a diverged track ends in 'X' where it last stayed finite.
"""

from __future__ import annotations

import math

from .optimizers import LABELS, NAMES
from .race import Race

RAMP = " .:-=+*#%@"  # low to high, but drawn in reverse so valleys are the brightest characters
LETTER = {"sgd": "s", "momentum": "m", "nesterov": "n", "rmsprop": "r", "adam": "a", "adagrad": "d"}


def _cell(x: float, y: float, dom: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int] | None:
    """Character cell for a world point, or None when it is outside the view box."""
    x0, x1, y0, y1 = dom
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return None
    col = min(w - 1, int((x - x0) / (x1 - x0) * w))
    row = min(h - 1, int((y1 - y) / (y1 - y0) * h))  # y grows upward on the map
    return row, col


def contour_ascii(race: Race, width: int = 64, height: int | None = None) -> str:
    """Render the race's surface and trails as a block of text (no trailing newline)."""
    s = race.surface
    x0, x1, y0, y1 = s.domain
    if height is None:
        # Terminal cells are about twice as tall as they are wide, so halve the row count.
        height = max(8, round(width * (y1 - y0) / (x1 - x0) / 2))
    vals = [
        [s.f(x0 + (c + 0.5) * (x1 - x0) / width, y1 - (r + 0.5) * (y1 - y0) / height) for c in range(width)]
        for r in range(height)
    ]
    base = s.fmin if s.fmin is not None else min(min(row) for row in vals)
    logs = [[math.log1p(max(v - base, 0.0)) for v in row] for row in vals]
    hi = max(max(row) for row in logs) or 1.0
    grid = []
    for row in logs:
        line = []
        for v in row:
            k = int((v / hi) * (len(RAMP) - 1))
            line.append(RAMP[len(RAMP) - 1 - k])
        grid.append(line)

    for tr in race.tracks:
        letter = LETTER[tr.name]
        for (ax, ay), (bx, by) in zip(tr.traj, tr.traj[1:]):
            n = max(1, int(max(abs(bx - ax) / (x1 - x0) * width, abs(by - ay) / (y1 - y0) * height) * 2))
            for i in range(n + 1):
                t = i / n
                cell = _cell(ax + (bx - ax) * t, ay + (by - ay) * t, s.domain, width, height)
                if cell:
                    grid[cell[0]][cell[1]] = letter
        if tr.traj:
            cell = _cell(*tr.traj[-1], s.domain, width, height)
            if cell:
                grid[cell[0]][cell[1]] = "X" if tr.diverged else letter.upper()

    for mx, my in s.minima:
        cell = _cell(mx, my, s.domain, width, height)
        if cell:
            grid[cell[0]][cell[1]] = "*"
    cell = _cell(*race.surface.start, s.domain, width, height)
    if cell:
        grid[cell[0]][cell[1]] = "o"
    lines = ["".join(row) for row in grid]
    legend = "  ".join(f"{LETTER[n]}={LABELS[n]}" for n in NAMES if any(t.name == n for t in race.tracks))
    lines.append(f"o=start  *=global minimum  X=diverged  UPPER=final  {legend}")
    return "\n".join(lines)


def table(race: Race) -> str:
    """The summary table: one row per optimizer."""
    head = f"{'optimizer':<10} {'lr':>8} {'final x':>12} {'final y':>12} {'final loss':>14} {'to tol':>8}  status"
    out = [head, "-" * len(head)]
    for row in race.summary():
        if row["diverged_at"] is not None:
            status = f"DIVERGED at step {row['diverged_at']}"
        elif row["steps_to_tol"] is not None:
            status = "reached minimum"
        else:
            status = "still moving"
        to_tol = "-" if row["steps_to_tol"] is None else str(row["steps_to_tol"])
        out.append(
            f"{LABELS[row['name']]:<10} {row['lr']:>8.4g} {row['x']:>12.5g} {row['y']:>12.5g} "
            f"{row['loss']:>14.6g} {to_tol:>8}  {status}"
        )
    return "\n".join(out)
