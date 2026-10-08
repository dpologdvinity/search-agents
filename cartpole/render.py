"""ASCII view of the cart and pole for the terminal.

The track is WIDTH columns and the pole is drawn to scale in one direction only: 1 m of pole is
8 rows high and 8 columns sideways, so a 12-degree lean looks like a lean rather than a slant
line. Horizontal positions use the same 8 columns per metre, so the track's |x| <= 2.4 limit
is 19 columns either side of the centre. The cart is "[###]" and the pole is drawn sample by
sample from the hinge, with '/' or '\\' for the lean direction and '|' when nearly upright.
"""

from __future__ import annotations

import math

from .env import THETA_LIMIT, X_LIMIT

WIDTH = 60
HEIGHT = 12
COLS_PER_M = 8.0
ROWS_PER_M = 8.0
CENTRE = (WIDTH - 1) / 2
CLEAR = "\033[H\033[J"  # move the cursor home and clear the screen


def render_frame(x: float, theta: float, steps: int, action: int | None = None,
                 probs: tuple[float, float] | None = None, value: float | None = None) -> str:
    """One frame as text. probs and value are optional readouts from a learned agent."""
    grid = [[" "] * WIDTH for _ in range(HEIGHT)]
    ground, cart_row = HEIGHT - 1, HEIGHT - 2
    hinge_row = HEIGHT - 3
    for c in range(WIDTH):
        grid[ground][c] = "="
    # The track limits, shown as posts at x = +-2.4.
    for sign in (-1, 1):
        col = round(CENTRE + sign * X_LIMIT * COLS_PER_M)
        grid[ground - 1][col] = "|"
    cx = min(max(round(CENTRE + x * COLS_PER_M), 3), WIDTH - 4)
    for c in range(cx - 2, cx + 3):
        grid[cart_row][c] = "#"
    grid[cart_row][cx - 2], grid[cart_row][cx + 2] = "[", "]"
    # Pole: sample points from the hinge up to the tip.
    tilt = "|" if abs(theta) < math.radians(2) else ("/" if theta > 0 else "\\")
    for i in range(1, 9):
        t = i / 8
        r = hinge_row - round(t * ROWS_PER_M * math.cos(theta))
        c = round(cx + t * COLS_PER_M * math.sin(theta))
        if 0 <= r < HEIGHT and 0 <= c < WIDTH:
            grid[r][c] = tilt
    grid[hinge_row][cx] = "o"

    lines = ["".join(row).rstrip() for row in grid]
    deg = math.degrees(theta)
    status = f"step {steps:3d}   x {x:+6.2f} m   theta {deg:+6.1f} deg"
    if abs(theta) > THETA_LIMIT or abs(x) > X_LIMIT:
        status += "   FALLEN"
    lines.append(status)
    if action is not None:
        lines.append(f"push {'LEFT ' if action == 0 else 'RIGHT'}"
                     + (f"   P(left) {probs[0]:.2f}  P(right) {probs[1]:.2f}" if probs else "")
                     + (f"   V {value:.1f}" if value is not None else ""))
    return "\n".join(lines)
