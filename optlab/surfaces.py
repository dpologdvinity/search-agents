"""Loss landscapes with analytic gradients, for the optimizer race.

Each surface is a function f(x, y) with its gradient written out by hand; the tests check every gradient
against central differences. The Hessian is never written out: `hessian` differentiates the gradient
numerically, which is enough to tell a minimum from a saddle (a positive definite Hessian or not).

The "custom" surface is the paint-your-own landscape: a gentle bowl plus Gaussian bumps. A positive amplitude
is a hill and a negative one a valley. Every bump is smooth, so the gradient stays analytic.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

Grad = Callable[[float, float], tuple[float, float]]
Fn = Callable[[float, float], float]

TAU = 2.0 * math.pi
HESSIAN_STEP = 1e-5


@dataclass(frozen=True)
class Surface:
    """A landscape: loss, gradient, the view box, and what is known about its minima.

    `fmin` is the global minimum value when it is known in closed form (None for custom surfaces).
    `minima` lists the global minima only; local minima are found by the race itself.
    """

    name: str
    label: str
    f: Fn
    grad: Grad
    domain: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    start: tuple[float, float]
    fmin: float | None = None
    minima: tuple[tuple[float, float], ...] = field(default=())
    note: str = ""


def _bowl_f(x, y):
    return x * x + y * y


def _bowl_g(x, y):
    return (2.0 * x, 2.0 * y)


def _ravine_f(x, y):
    return 0.5 * (x * x + 25.0 * y * y)


def _ravine_g(x, y):
    return (x, 25.0 * y)


def _saddle_f(x, y):
    return x * x - y * y + 0.25 * (y * y * y * y)


def _saddle_g(x, y):
    return (2.0 * x, -2.0 * y + y * y * y)


def _rosen_f(x, y):
    return (1.0 - x) * (1.0 - x) + 100.0 * ((y - x * x) * (y - x * x))


def _rosen_g(x, y):
    r = y - x * x
    return (-2.0 * (1.0 - x) - 400.0 * x * r, 200.0 * r)


def _himmel_f(x, y):
    a = x * x + y - 11.0
    b = x + y * y - 7.0
    return a * a + b * b


def _himmel_g(x, y):
    a = x * x + y - 11.0
    b = x + y * y - 7.0
    return (4.0 * x * a + 2.0 * b, 2.0 * a + 4.0 * y * b)


def _rastrigin_f(x, y):
    return 20.0 + (x * x - 10.0 * math.cos(TAU * x)) + (y * y - 10.0 * math.cos(TAU * y))


def _rastrigin_g(x, y):
    return (2.0 * x + 10.0 * TAU * math.sin(TAU * x), 2.0 * y + 10.0 * TAU * math.sin(TAU * y))


_HIMMEL_MINIMA = (
    (3.0, 2.0),
    (-2.805118, 3.131312),
    (-3.779310, -3.283186),
    (3.584428, -1.848126),
)

SURFACES: dict[str, Surface] = {
    s.name: s
    for s in (
        Surface(
            "bowl",
            "Bowl",
            _bowl_f,
            _bowl_g,
            (-3.0, 3.0, -3.0, 3.0),
            (2.4, 1.6),
            0.0,
            ((0.0, 0.0),),
            "x^2 + y^2. Well conditioned: every direction has the same curvature.",
        ),
        Surface(
            "ravine",
            "Ravine",
            _ravine_f,
            _ravine_g,
            (-4.0, 4.0, -4.0, 4.0),
            (3.2, 0.9),
            0.0,
            ((0.0, 0.0),),
            "0.5 (x^2 + 25 y^2). Curvature 25 across the ravine, 1 along it: ill conditioned.",
        ),
        Surface(
            "saddle",
            "Saddle",
            _saddle_f,
            _saddle_g,
            (-2.5, 2.5, -2.5, 2.5),
            (1.8, 0.6),
            -1.0,
            ((0.0, math.sqrt(2.0)), (0.0, -math.sqrt(2.0))),
            "x^2 - y^2 + y^4/4. The origin is a saddle; the two minima sit at y = +-sqrt(2).",
        ),
        Surface(
            "rosenbrock",
            "Rosenbrock",
            _rosen_f,
            _rosen_g,
            (-2.0, 2.0, -1.0, 3.0),
            (-1.5, 2.0),
            0.0,
            ((1.0, 1.0),),
            "(1 - x)^2 + 100 (y - x^2)^2. A curved banana valley to one minimum.",
        ),
        Surface(
            "himmelblau",
            "Himmelblau",
            _himmel_f,
            _himmel_g,
            (-5.0, 5.0, -5.0, 5.0),
            (-4.0, 4.5),
            0.0,
            _HIMMEL_MINIMA,
            "(x^2 + y - 11)^2 + (x + y^2 - 7)^2. Four global minima, all at zero loss.",
        ),
        Surface(
            "rastrigin",
            "Rastrigin",
            _rastrigin_f,
            _rastrigin_g,
            (-5.12, 5.12, -5.12, 5.12),
            (4.1, 2.9),
            0.0,
            ((0.0, 0.0),),
            "20 + sum (x_i^2 - 10 cos 2 pi x_i). A grid of local minima around one global one.",
        ),
    )
}

# The starting bumps of the paint-your-own surface: one hill and one valley.
CUSTOM_DEFAULT_BUMPS = ((0.6, -0.4, 3.0, 0.7), (-1.2, 1.1, -2.5, 0.9))
CUSTOM_BASE = 0.1  # bowl coefficient under the bumps; it keeps the landscape bounded below
CUSTOM_DOMAIN = (-3.0, 3.0, -3.0, 3.0)
CUSTOM_START = (-2.2, -1.6)
MAX_BUMPS = 12


def custom(bumps=CUSTOM_DEFAULT_BUMPS) -> Surface:
    """The paint-your-own landscape: CUSTOM_BASE * (x^2 + y^2) plus Gaussian bumps (cx, cy, amp, sigma).

    Each bump adds amp * exp(-r^2 / (2 sigma^2)), with r the distance from its centre (cx, cy). Its gradient
    is -amp * exp(...) * (p - c) / sigma^2.
    """
    bumps = tuple((float(cx), float(cy), float(a), float(s)) for cx, cy, a, s in bumps)
    if len(bumps) > MAX_BUMPS:
        raise ValueError(f"at most {MAX_BUMPS} bumps")
    for _, _, _, s in bumps:
        if not s > 0:
            raise ValueError("bump sigma must be positive")

    def f(x, y):
        v = CUSTOM_BASE * (x * x + y * y)
        for cx, cy, a, s in bumps:
            dx, dy = x - cx, y - cy
            v += a * math.exp(-(dx * dx + dy * dy) / (2.0 * s * s))
        return v

    def g(x, y):
        gx = 2.0 * CUSTOM_BASE * x
        gy = 2.0 * CUSTOM_BASE * y
        for cx, cy, a, s in bumps:
            dx, dy = x - cx, y - cy
            e = a * math.exp(-(dx * dx + dy * dy) / (2.0 * s * s)) / (s * s)
            gx -= e * dx
            gy -= e * dy
        return (gx, gy)

    return Surface(
        "custom",
        "Paint your own",
        f,
        g,
        CUSTOM_DOMAIN,
        CUSTOM_START,
        None,
        (),
        "A bowl with your own hills (+) and valleys (-).",
    )


def get(name: str) -> Surface:
    """Look up a surface by name ("custom" gives the default paint-your-own landscape)."""
    if name == "custom":
        return custom()
    try:
        return SURFACES[name]
    except KeyError:
        raise KeyError(f"unknown surface {name!r}; choose from {', '.join(SURFACES)}") from None


def hessian(grad: Grad, x: float, y: float, h: float = HESSIAN_STEP) -> tuple[float, float, float]:
    """Symmetric 2x2 Hessian (hxx, hxy, hyy) from central differences of the gradient."""
    gxp, gxm = grad(x + h, y), grad(x - h, y)
    gyp, gym = grad(x, y + h), grad(x, y - h)
    hxx = (gxp[0] - gxm[0]) / (2.0 * h)
    hyx = (gxp[1] - gxm[1]) / (2.0 * h)
    hxy = (gyp[0] - gym[0]) / (2.0 * h)
    hyy = (gyp[1] - gym[1]) / (2.0 * h)
    return hxx, 0.5 * (hxy + hyx), hyy


def is_minimum(surface: Surface, x: float, y: float, gtol: float) -> bool:
    """True when the gradient norm is below gtol and the Hessian is positive definite.

    A zero gradient alone also matches saddles and maxima. The curvature test is what keeps the
    "reached minimum" badge honest: at a saddle the Hessian has a negative eigenvalue.
    """
    gx, gy = surface.grad(x, y)
    if not (math.sqrt(gx * gx + gy * gy) < gtol):
        return False
    hxx, hxy, hyy = hessian(surface.grad, x, y)
    return hxx * hyy - hxy * hxy > 0.0 and hxx + hyy > 0.0
