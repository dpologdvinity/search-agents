"""EVOLVING WALKERS: physics and genetic algorithm, the Python twin of web/js/walkers-core.js.

Every function here follows the JavaScript file operation for operation: the same constants, the same order of
floating-point operations, and the same order of random draws from the same mulberry32 stream. That is what makes
a seed produce the same evolution in the browser and in Python, and it is what tests/test_walkers_parity.py checks.

The physics runs on a whole population at once with NumPy (`simulate`). Each creature's nodes and springs are
padded to the batch maximum; padded springs have zero stiffness, damping and rest length, so their force is exactly
0.0, and padded nodes are masked out of every sum and every contact. Rows of the batch never interact: NumPy
operations are element-wise, and `np.add.at` only sums into the nodes of one creature.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------- constants (mirror walkers-core.js)
DT = 1 / 240
NODE_MIN = 2
NODE_MAX = 12
SPRING_MAX = 40
REST_MIN = 0.08
REST_MAX = 1.6
K_MIN = 20
K_MAX = 600
C_MIN = 0.2
C_MAX = 10
AMP_MAX = 0.4
FREQ_MIN = 0.4
FREQ_MAX = 4.0
POS_MAX = 1.5
VMAX = 30
AIR_DRAG = 0.1
REST_SPEED = 0.5
STATIC_RATIO = 1.2
FALL_Y = -1.5
SPIN_FREE = 1.0
SPIN_PENALTY = 1.0  # metres per radian beyond SPIN_FREE: tumbling must cost more than it earns
SPEED_BONUS = 0.5  # seconds: fitness gains this times the mean forward speed of the second half
SPAWN_X = 2.0
SPAWN_LIFT = 0.45
HOLE = -1000
DEFAULT_WORLD = {"terrain": "flat", "gravity": 9.8, "friction": 0.8, "restitution": 0.2, "duration": 10}
TERRAINS = ("flat", "hills", "steps", "gaps")
GA_DEFAULTS = {
    "popSize": 80,
    "mutationRate": 0.5,
    "crossoverRate": 0.6,
    "eliteMin": 5,
    "tournament": 3,
    "survivalFrac": 0.5,
    "speciesThreshold": 1.2,
}

PI = 3.141592653589793
TWO_PI = 6.283185307179586
HALF_PI = 1.5707963267948966
MASK32 = 0xFFFFFFFF


# ---------------------------------------------------------------- numbers
def wsin(x: float) -> float:
    """Sine from basic arithmetic, bit-identical to the JavaScript wsin.

    Range-reduces with floor, folds into [-pi/2, pi/2], then evaluates the Taylor series in nested form up to r^15.
    Libraries disagree in the last bit, and chaotic bodies amplify that, so both languages use this polynomial.
    """
    r = x - TWO_PI * math.floor((x + PI) / TWO_PI)
    if r > HALF_PI:
        r = PI - r
    elif r < -HALF_PI:
        r = -PI - r
    r2 = r * r
    return r * (
        1
        - (r2 / 6)
        * (1 - (r2 / 20) * (1 - (r2 / 42) * (1 - (r2 / 72) * (1 - (r2 / 110) * (1 - (r2 / 156) * (1 - r2 / 210))))))
    )


def wcos(x: float) -> float:
    """Cosine as wsin(x + pi/2), like the JavaScript version."""
    return wsin(x + HALF_PI)


def wsin_np(x: np.ndarray) -> np.ndarray:
    """Vector form of wsin with the same operations in the same order (element-wise identical results)."""
    r = x - TWO_PI * np.floor((x + PI) / TWO_PI)
    r = np.where(r > HALF_PI, PI - r, np.where(r < -HALF_PI, -PI - r, r))
    r2 = r * r
    return r * (
        1
        - (r2 / 6)
        * (1 - (r2 / 20) * (1 - (r2 / 42) * (1 - (r2 / 72) * (1 - (r2 / 110) * (1 - (r2 / 156) * (1 - r2 / 210))))))
    )


def wcos_np(x: np.ndarray) -> np.ndarray:
    """Vector form of wcos."""
    return wsin_np(x + HALF_PI)


def q4(v: float) -> float:
    """Round to the 1e-4 grid with round-half-up (floor(v*1e4 + 0.5)), as the genes are stored."""
    return math.floor(v * 10000 + 0.5) / 10000


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp v into [lo, hi]."""
    return lo if v < lo else hi if v > hi else v


def _imul(a: int, b: int) -> int:
    """32-bit multiply keeping the low 32 bits (Math.imul on unsigned representations)."""
    return (a * b) & MASK32


class Mulberry32:
    """The mulberry32 generator, bit-identical to the JavaScript mulberry32 in walkers-core.js.

    `next()` returns a float in [0, 1); `int(n)` gives 0..n-1; `range(lo, hi)` gives a float in [lo, hi).
    Every draw is one 32-bit step, so Python and JavaScript stay in step when they make the same calls.
    """

    def __init__(self, seed: int):
        self.a = int(seed) & MASK32

    def next(self) -> float:
        """Advance the generator and return the next uniform value in [0, 1)."""
        self.a = (self.a + 0x6D2B79F5) & MASK32
        t = self.a
        t = _imul(t ^ (t >> 15), t | 1)
        t ^= (t + _imul(t ^ (t >> 7), t | 61)) & MASK32
        return ((t ^ (t >> 14)) & MASK32) / 4294967296

    def int(self, n: int) -> int:
        """Uniform integer in 0..n-1."""
        return math.floor(self.next() * n)

    def range(self, lo: float, hi: float) -> float:
        """Uniform float in [lo, hi)."""
        return lo + (hi - lo) * self.next()


def mulberry32(seed: int) -> Mulberry32:
    """Factory matching the JavaScript name."""
    return Mulberry32(seed)


# ---------------------------------------------------------------- terrain
def ground_at(terrain: str, x: float) -> tuple[float, float]:
    """Ground height and slope under x (scalar form, used by tests and the replay).

    hills: a ramp-in from x=3 to x=5, then two sine waves. steps: 0.15 m risers every 1.2 m from x=3 (8 max).
    gaps: flat floor with 0.9 m holes every 3.2 m from x=4. Anything else is flat.
    """
    if terrain == "hills":
        ramp = (x - 3) / 2
        ramp = 0 if ramp < 0 else 1 if ramp > 1 else ramp
        dramp = 0.5 if (x > 3 and x < 5) else 0
        H = 0.35 * wsin(0.7 * x) + 0.12 * wsin(1.9 * x + 1)
        dH = 0.35 * 0.7 * wcos(0.7 * x) + 0.12 * 1.9 * wcos(1.9 * x + 1)
        return ramp * H, ramp * dH + dramp * H
    if terrain == "steps":
        k = min(8, math.floor((x - 3) / 1.2)) if x > 3 else 0
        return 0.15 * k, 0.0
    if terrain == "gaps":
        if x < 4:
            return 0.0, 0.0
        u = x - 4
        r = u - 3.2 * math.floor(u / 3.2)
        return (HOLE if r < 0.9 else 0.0), 0.0
    return 0.0, 0.0


def ground_np(terrain: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vector form of ground_at over an array of x positions. Same expressions as the scalar form."""
    zero = np.zeros_like(x)
    if terrain == "hills":
        ramp = (x - 3) / 2
        ramp = np.where(ramp < 0, 0.0, np.where(ramp > 1, 1.0, ramp))
        dramp = np.where((x > 3) & (x < 5), 0.5, 0.0)
        H = 0.35 * wsin_np(0.7 * x) + 0.12 * wsin_np(1.9 * x + 1)
        dH = 0.35 * 0.7 * wcos_np(0.7 * x) + 0.12 * 1.9 * wcos_np(1.9 * x + 1)
        return ramp * H, ramp * dH + dramp * H
    if terrain == "steps":
        k = np.where(x > 3, np.minimum(8, np.floor((x - 3) / 1.2)), 0.0)
        return 0.15 * k, zero
    if terrain == "gaps":
        u = x - 4
        r = u - 3.2 * np.floor(u / 3.2)
        h = np.where(r < 0.9, float(HOLE), 0.0)
        return np.where(x < 4, 0.0, h), zero
    return zero, zero


# ---------------------------------------------------------------- genome
def clone_genome(g: dict) -> dict:
    """Deep copy of a genome (plain dicts), with muscle stored as 0 or 1 as in the JavaScript."""
    return {
        "nodes": [{"x": n["x"], "y": n["y"]} for n in g["nodes"]],
        "springs": [
            {
                "a": s["a"],
                "b": s["b"],
                "rest": s["rest"],
                "k": s["k"],
                "c": s["c"],
                "muscle": 1 if s["muscle"] else 0,
                "amp": s["amp"],
                "freq": s["freq"],
                "phase": s["phase"],
            }
            for s in g["springs"]
        ],
    }


def _make_spring(a: int, b: int, rest: float, muscle: bool, rng: Mulberry32) -> dict:
    """A spring with random genes. Draw order k, c, amp, freq, phase is fixed (same as JavaScript)."""
    return {
        "a": min(a, b),
        "b": max(a, b),
        "rest": q4(clamp(rest, REST_MIN, REST_MAX)),
        "k": q4(rng.range(60, 250)),
        "c": q4(rng.range(0.5, 6)),
        "muscle": 1 if muscle else 0,
        "amp": q4(rng.range(0.1, 0.35)),
        "freq": q4(rng.range(0.8, 3.0)),
        "phase": q4(rng.range(0, TWO_PI)),
    }


def _dist(nodes: list, i: int, j: int) -> float:
    """Euclidean distance between nodes i and j of a node list."""
    dx = nodes[j]["x"] - nodes[i]["x"]
    dy = nodes[j]["y"] - nodes[i]["y"]
    return math.sqrt(dx * dx + dy * dy)


def random_genome(rng: Mulberry32) -> dict:
    """A random connected creature: 3 to 7 nodes, a spanning tree, a few short extra links, and at least one muscle."""
    n = 3 + rng.int(5)
    nodes = []
    for _ in range(n):
        x = q4(rng.range(-0.35, 0.35))
        y = q4(rng.range(0, 0.35))
        nodes.append({"x": x, "y": y})
    g = {"nodes": nodes, "springs": []}

    def has(a: int, b: int) -> bool:
        lo, hi = min(a, b), max(a, b)
        return any(s["a"] == lo and s["b"] == hi for s in g["springs"])

    for i in range(1, n):
        j = rng.int(i)
        g["springs"].append(_make_spring(i, j, _dist(nodes, i, j), rng.next() < 0.6, rng))
    for i in range(n):
        for j in range(i + 1, n):
            if has(i, j):
                continue
            d = _dist(nodes, i, j)
            if d < 0.8 and rng.next() < 0.3:
                g["springs"].append(_make_spring(i, j, d, rng.next() < 0.6, rng))
    if not any(s["muscle"] for s in g["springs"]):
        g["springs"][rng.int(len(g["springs"]))]["muscle"] = 1
    return repair(g)


def repair(g: dict) -> dict:
    """Deterministic repair (no random draws): quantise and clamp, drop bad or duplicate springs, keep one spring
    and one muscle at least, and connect every node. Unreached nodes join through the nearest reached node,
    the lowest-index unreached node first, and the lowest index on distance ties."""
    nodes = [
        {"x": q4(clamp(n["x"], -POS_MAX, POS_MAX)), "y": q4(clamp(n["y"], -POS_MAX, POS_MAX))}
        for n in g["nodes"][:NODE_MAX]
    ]
    while len(nodes) < NODE_MIN:
        nodes.append({"x": q4(0.1 * len(nodes)), "y": 0.1})
    n = len(nodes)
    seen = set()
    springs = []
    for s in g["springs"]:
        a = min(s["a"], s["b"])
        b = max(s["a"], s["b"])
        if a == b or a < 0 or b >= n:
            continue
        key = a * 64 + b
        if key in seen:
            continue
        seen.add(key)
        springs.append(
            {
                "a": a,
                "b": b,
                "rest": q4(clamp(s["rest"], REST_MIN, REST_MAX)),
                "k": q4(clamp(s["k"], K_MIN, K_MAX)),
                "c": q4(clamp(s["c"], C_MIN, C_MAX)),
                "muscle": 1 if s["muscle"] else 0,
                "amp": q4(clamp(s["amp"], 0, AMP_MAX)),
                "freq": q4(clamp(s["freq"], FREQ_MIN, FREQ_MAX)),
                "phase": q4(s["phase"]),
            }
        )
    out = {"nodes": nodes, "springs": springs}
    # Connectivity: grow the reached set from node 0 until everything is reached.
    while True:
        reached = [False] * n
        reached[0] = True
        grew = True
        while grew:
            grew = False
            for s in out["springs"]:
                if reached[s["a"]] != reached[s["b"]]:
                    reached[s["a"]] = True
                    reached[s["b"]] = True
                    grew = True
        if all(reached):
            break
        u = reached.index(False)
        best = -1
        best_d = math.inf
        for v in range(n):
            if not reached[v]:
                continue
            d = _dist(nodes, u, v)
            if d < best_d:
                best_d = d
                best = v
        out["springs"].append(
            {
                "a": min(u, best),
                "b": max(u, best),
                "rest": q4(clamp(best_d, REST_MIN, REST_MAX)),
                "k": 150,
                "c": 2,
                "muscle": 0,
                "amp": 0,
                "freq": 1,
                "phase": 0,
            }
        )
    if not out["springs"]:
        out["springs"].append(
            {
                "a": 0,
                "b": 1,
                "rest": q4(clamp(_dist(nodes, 0, 1), REST_MIN, REST_MAX)),
                "k": 150,
                "c": 2,
                "muscle": 0,
                "amp": 0,
                "freq": 1,
                "phase": 0,
            }
        )
    if not any(s["muscle"] for s in out["springs"]):
        out["springs"][0]["muscle"] = 1
    return out


def is_valid(g: dict) -> bool:
    """True when the genome is within limits, has unique in-range springs with a < b, has a muscle, and is connected."""
    n = len(g["nodes"])
    if n < NODE_MIN or n > NODE_MAX or len(g["springs"]) < 1 or len(g["springs"]) > SPRING_MAX:
        return False
    if not any(s["muscle"] for s in g["springs"]):
        return False
    seen = set()
    for s in g["springs"]:
        if not (s["a"] < s["b"]) or s["a"] < 0 or s["b"] >= n:
            return False
        key = s["a"] * 64 + s["b"]
        if key in seen:
            return False
        seen.add(key)
        if (
            s["rest"] < REST_MIN
            or s["rest"] > REST_MAX
            or s["k"] < K_MIN
            or s["k"] > K_MAX
            or s["c"] < C_MIN
            or s["c"] > C_MAX
        ):
            return False
    reached = [False] * n
    reached[0] = True
    grew = True
    while grew:
        grew = False
        for s in g["springs"]:
            if reached[s["a"]] != reached[s["b"]]:
                reached[s["a"]] = True
                reached[s["b"]] = True
                grew = True
    return all(reached)


# ---------------------------------------------------------------- text encoding
def _fmt4(v: float) -> str:
    """Four decimals with trailing zeros and a trailing dot removed, exactly as JavaScript's toFixed(4) path does.
    Adding 0.0 turns -0.0 into 0.0, which JavaScript never prints as a minus sign."""
    s = f"{v + 0.0:.4f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def encode(g: dict) -> str:
    """Text form: "w1|x,y;x,y;...|a,b,rest,k,c,muscle,amp,freq,phase;...". Shared with the page and the editor."""
    nodes = ";".join(f"{_fmt4(n['x'])},{_fmt4(n['y'])}" for n in g["nodes"])
    springs = ";".join(
        ",".join(
            [
                str(s["a"]),
                str(s["b"]),
                _fmt4(s["rest"]),
                _fmt4(s["k"]),
                _fmt4(s["c"]),
                str(1 if s["muscle"] else 0),
                _fmt4(s["amp"]),
                _fmt4(s["freq"]),
                _fmt4(s["phase"]),
            ]
        )
        for s in g["springs"]
    )
    return f"w1|{nodes}|{springs}"


def decode(text: str) -> dict:
    """Parse the text form. Raises ValueError on anything that is not a w1 creature with nine fields per spring."""
    parts = str(text).strip().split("|")
    if len(parts) != 3 or parts[0] != "w1":
        raise ValueError("not a walkers creature (expected w1|nodes|springs)")
    nodes = []
    for p in parts[1].split(";"):
        xy = [float(v) for v in p.split(",")]
        if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
            raise ValueError("bad node: " + p)
        nodes.append({"x": xy[0], "y": xy[1]})
    springs = []
    for p in parts[2].split(";"):
        f = p.split(",")
        if len(f) != 9:
            raise ValueError("bad spring: " + p)
        v = [float(x) for x in f]
        if not all(math.isfinite(x) for x in v):
            raise ValueError("bad spring: " + p)
        springs.append(
            {
                "a": int(f[0]),
                "b": int(f[1]),
                "rest": v[2],
                "k": v[3],
                "c": v[4],
                "muscle": 1 if v[5] else 0,
                "amp": v[6],
                "freq": v[7],
                "phase": v[8],
            }
        )
    return {"nodes": nodes, "springs": springs}


# ---------------------------------------------------------------- body and physics
def build_body(g: dict) -> dict:
    """Place a genome above the floor and flatten it into arrays.

    The body is centred on SPAWN_X and its lowest node sits SPAWN_LIFT up. The spine is the pair of nodes farthest
    apart at spawn (first pair on ties); its rotation counts toward the flip penalty.
    """
    N = len(g["nodes"])
    sx = 0.0
    min_y = math.inf
    for n in g["nodes"]:
        sx += n["x"]
        if n["y"] < min_y:
            min_y = n["y"]
    mx = sx / N
    px = [SPAWN_X + (n["x"] - mx) for n in g["nodes"]]
    py = [n["y"] - min_y + SPAWN_LIFT for n in g["nodes"]]
    sa = 0
    sb = 1 if N > 1 else 0
    best = -1.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = px[j] - px[i]
            dy = py[j] - py[i]
            d2 = dx * dx + dy * dy
            if d2 > best:
                best = d2
                sa = i
                sb = j
    springs = g["springs"]
    return {
        "N": N,
        "px": px,
        "py": py,
        "sa": sa,
        "sb": sb,
        "ia": [s["a"] for s in springs],
        "ib": [s["b"] for s in springs],
        "rest": [float(s["rest"]) for s in springs],
        "k": [float(s["k"]) for s in springs],
        "c": [float(s["c"]) for s in springs],
        "mus": [1 if s["muscle"] else 0 for s in springs],
        "amp": [float(s["amp"]) for s in springs],
        "w": [TWO_PI * s["freq"] for s in springs],
        "ph": [float(s["phase"]) for s in springs],
    }


def _round_steps(duration: float) -> int:
    """Math.round for positive numbers: the number of DT steps in the run."""
    return int(math.floor(duration / DT + 0.5))


def simulate(genomes: list, world: dict | None = None, sample_every: int = 0) -> list:
    """Run a batch of creatures for world["duration"] seconds with the fixed-step integrator of walkers-core.js.

    Each step: spring and muscle forces from the positions at the start of the step, summed in two ordered passes
    (all "a" ends in spring order, then all "b" ends); then velocities with gravity, drag and the velocity clamp;
    then positions; then ground contact (restitution above REST_SPEED, Coulomb friction with static and kinetic
    limits). A creature whose state turns non-finite is frozen and reported as exploded with fitness 0. A creature
    whose centre of mass drops below FALL_Y is scored where it fell.

    Returns one dict per creature: fitness, distance, exploded, fallen, spin, clamps, steps, N, and frames as an
    array of shape (frames, N, 2) when sample_every > 0 (frame 0 is the spawn pose, then one every sample_every steps).
    """
    world = {**DEFAULT_WORLD, **(world or {})}
    P = len(genomes)
    if P == 0:
        return []
    bodies = [build_body(g) for g in genomes]
    Nmax = max(b["N"] for b in bodies)
    Smax = max(max(len(b["ia"]) for b in bodies), 1)
    rows = np.arange(P)

    Nv = np.array([b["N"] for b in bodies], dtype=np.int64)
    nmask = np.arange(Nmax)[None, :] < Nv[:, None]
    px = np.zeros((P, Nmax))
    py = np.zeros((P, Nmax))
    for p, b in enumerate(bodies):
        px[p, : b["N"]] = b["px"]
        py[p, : b["N"]] = b["py"]
    vx = np.zeros((P, Nmax))
    vy = np.zeros((P, Nmax))

    ia = np.zeros((P, Smax), dtype=np.int64)
    ib = np.zeros((P, Smax), dtype=np.int64)
    rest = np.zeros((P, Smax))
    kk = np.zeros((P, Smax))
    cc = np.zeros((P, Smax))
    mus = np.zeros((P, Smax), dtype=bool)
    amp = np.zeros((P, Smax))
    ww = np.zeros((P, Smax))
    ph = np.zeros((P, Smax))
    for p, b in enumerate(bodies):
        S = len(b["ia"])
        if S:
            ia[p, :S] = b["ia"]
            ib[p, :S] = b["ib"]
            rest[p, :S] = b["rest"]
            kk[p, :S] = b["k"]
            cc[p, :S] = b["c"]
            mus[p, :S] = np.array(b["mus"], dtype=bool)
            amp[p, :S] = b["amp"]
            ww[p, :S] = b["w"]
            ph[p, :S] = b["ph"]
    # Flat indices into the (P*Nmax) node vector, so one np.add.at call sums each a-end (or b-end) per creature.
    idx_a = rows[:, None] * Nmax + ia  # (P, S) indices into the flattened (P, Nmax) node arrays
    idx_b = rows[:, None] * Nmax + ib
    flat_a = idx_a.ravel()
    flat_b = idx_b.ravel()

    # Spine per creature, found with the same pair order and strict ">" as the scalar code.
    sa = np.zeros(P, dtype=np.int64)
    sb = np.where(Nv > 1, 1, 0).astype(np.int64)
    best = np.full(P, -1.0)
    for i in range(Nmax):
        for j in range(i + 1, Nmax):
            valid = j < Nv
            dx = px[:, j] - px[:, i]
            dy = py[:, j] - py[:, i]
            d2 = dx * dx + dy * dy
            upd = valid & (d2 > best)
            best = np.where(upd, d2, best)
            sa = np.where(upd, i, sa)
            sb = np.where(upd, j, sb)

    def seq_sum(arr: np.ndarray) -> np.ndarray:
        """Sum the valid nodes of each row in index order, like the scalar loop (np.sum would pair them)."""
        # Padded columns are exactly 0.0 and come last, so the running sum over all columns equals the valid sum.
        return np.add.accumulate(arr, axis=1)[:, -1]

    com0 = seq_sum(px) / Nv
    ux = px[rows, sb] - px[rows, sa]
    uy = py[rows, sb] - py[rows, sa]
    ul = np.sqrt(ux * ux + uy * uy)
    ul = np.where(ul < 1e-9, 1e-9, ul)
    ux = ux / ul
    uy = uy / ul

    steps = _round_steps(world["duration"])
    grav = float(world["gravity"])
    muK = float(world["friction"])
    muS = float(world["friction"]) * STATIC_RATIO
    e = float(world["restitution"])
    terrain = world["terrain"]
    damp = 1 - AIR_DRAG * DT

    sample_every = int(sample_every or 0)
    frame_count = steps // sample_every + 1 if sample_every else 0
    frames = np.zeros((frame_count, P, Nmax, 2)) if sample_every else None
    fr = 0

    def record() -> None:
        nonlocal fr
        frames[fr, :, :, 0] = px
        frames[fr, :, :, 1] = py
        fr += 1

    if frames is not None:
        record()

    fallen = np.zeros(P, dtype=bool)
    exploded = np.zeros(P, dtype=bool)
    dist_fall = np.zeros(P)
    spin_fall = np.zeros(P)
    spin = np.zeros(P)
    com = com0.copy()
    com_mid = com0.copy()
    half = steps // 2
    clamps = np.zeros(P, dtype=np.int64)
    steps_run = np.zeros(P, dtype=np.int64)
    fx = np.zeros(P * Nmax)
    fy = np.zeros(P * Nmax)

    for step in range(steps):
        t = step * DT
        # (1) spring and muscle forces, from the positions at the start of the step.
        pax = np.take(px, idx_a)
        pay = np.take(py, idx_a)
        pbx = np.take(px, idx_b)
        pby = np.take(py, idx_b)
        vax = np.take(vx, idx_a)
        vay = np.take(vy, idx_a)
        vbx = np.take(vx, idx_b)
        vby = np.take(vy, idx_b)
        dx = pbx - pax
        dy = pby - pay
        L = np.sqrt(dx * dx + dy * dy)
        L = np.where(L < 1e-9, 1e-9, L)
        nx = dx / L
        ny = dy / L
        r = np.where(mus, rest * (1 + amp * wsin_np(ww * t + ph)), rest)
        f = kk * (L - r) + cc * ((vbx - vax) * nx + (vby - vay) * ny)
        Fx = (f * nx).ravel()
        Fy = (f * ny).ravel()
        # (2) accumulate: all a-ends in spring order, then all b-ends (the same order as the scalar loops).
        fx[:] = 0.0
        fy[:] = 0.0
        np.add.at(fx, flat_a, Fx)
        np.add.at(fy, flat_a, Fy)
        np.add.at(fx, flat_b, -Fx)
        np.add.at(fy, flat_b, -Fy)
        fx2 = fx.reshape(P, Nmax)
        fy2 = fy.reshape(P, Nmax)
        # (3) velocities: gravity, drag and the clamp.
        vxn = (vx + fx2 * DT) * damp
        vyn = (vy + (fy2 - grav) * DT) * damp
        over = (vxn > VMAX) | (vxn < -VMAX) | (vyn > VMAX) | (vyn < -VMAX)
        clamps += (over & nmask).sum(axis=1)
        vxn = np.where(vxn > VMAX, VMAX, np.where(vxn < -VMAX, -VMAX, vxn))
        vyn = np.where(vyn > VMAX, VMAX, np.where(vyn < -VMAX, -VMAX, vyn))
        # Padded nodes stay at rest so they never affect anything.
        vx = np.where(nmask, vxn, 0.0)
        vy = np.where(nmask, vyn, 0.0)
        # (4) positions.
        px = np.where(nmask, px + vx * DT, 0.0)
        py = np.where(nmask, py + vy * DT, 0.0)
        # (5) ground contact.
        h, sl = ground_np(terrain, px)
        inv = 1 / np.sqrt(1 + sl * sl)
        gnx = -sl * inv
        gny = inv
        gtx = gny
        gty = -gnx
        hit = nmask & (py < h)
        py = np.where(hit, h, py)
        vn = vx * gnx + vy * gny
        cond = hit & (vn < 0)
        vt = vx * gtx + vy * gty
        speed = -vn
        ee = np.where(speed > REST_SPEED, e, 0.0)
        vn_new = -ee * vn
        lim = (1 + ee) * speed
        vt_new = np.where(np.abs(vt) <= muS * lim, 0.0, np.where(vt > 0, vt - muK * lim, vt + muK * lim))
        vx = np.where(cond, vn_new * gnx + vt_new * gtx, vx)
        vy = np.where(cond, vn_new * gny + vt_new * gty, vy)
        # Non-finite state: freeze that creature (its row is reset so it cannot emit NaN into later steps).
        bad = ~(np.isfinite(px) & np.isfinite(py)).all(axis=1)
        newly = bad & ~exploded
        if newly.any():
            exploded |= bad
            px = np.where(bad[:, None], 0.0, px)
            py = np.where(bad[:, None], 0.0, py)
            vx = np.where(bad[:, None], 0.0, vx)
            vy = np.where(bad[:, None], 0.0, vy)
        alive = ~exploded
        steps_run = np.where(alive, step + 1, steps_run)
        # (6) bookkeeping for creatures that have not fallen: centre of mass, spine rotation, fall test.
        active = alive & ~fallen
        com_now = seq_sum(px) / Nv
        comy = seq_sum(py) / Nv
        dxs = px[rows, sb] - px[rows, sa]
        dys = py[rows, sb] - py[rows, sa]
        dl = np.sqrt(dxs * dxs + dys * dys)
        dl = np.where(dl < 1e-9, 1e-9, dl)
        dxs = dxs / dl
        dys = dys / dl
        spin_new = spin + np.abs(ux * dys - uy * dxs)
        spin = np.where(active, spin_new, spin)
        ux = np.where(active, dxs, ux)
        uy = np.where(active, dys, uy)
        com = np.where(active, com_now, com)
        if step + 1 == half:
            com_mid = np.where(active, com_now, com_mid)
        newfall = active & (comy < FALL_Y)
        dist_fall = np.where(newfall, com_now - com0, dist_fall)
        spin_fall = np.where(newfall, spin, spin_fall)
        fallen |= newfall
        if frames is not None and (step + 1) % sample_every == 0:
            record()

    distance = np.where(fallen, dist_fall, com - com0)
    spin_used = np.where(fallen, spin_fall, spin)
    distance = np.where(exploded, 0.0, distance)
    # Speed bonus: mean forward speed over the second half, (com at the end - com at the midpoint) / its seconds.
    # Bodies that fell, exploded, or ran fewer than two steps get none.
    speed_bonus = np.where(
        exploded | fallen | (steps <= half), 0.0, (SPEED_BONUS * (com - com_mid)) / ((steps - half) * DT)
    )
    fitness = np.where(
        exploded, 0.0, distance - SPIN_PENALTY * np.maximum(0.0, spin_used - SPIN_FREE) + speed_bonus
    )
    out = []
    for p, b in enumerate(bodies):
        N = b["N"]
        out.append(
            {
                "fitness": float(fitness[p]),
                "distance": float(distance[p]),
                "exploded": bool(exploded[p]),
                "fallen": bool(fallen[p]),
                "spin": float(spin_used[p]),
                "speedBonus": float(speed_bonus[p]) if not exploded[p] else 0.0,
                "clamps": int(clamps[p]),
                "steps": int(steps_run[p]),
                "N": N,
                "frames": frames[:fr, p, :N, :].copy() if frames is not None else None,
                "frameCount": fr if frames is not None else 0,
            }
        )
    return out


def simulate_one(g: dict, world: dict | None = None, sample_every: int = 0) -> dict:
    """Convenience wrapper: one creature through the batch simulator."""
    return simulate([g], world, sample_every)[0]


def actuation(s: dict, t: float) -> float:
    """Muscle actuation level at time t, in [-amp, amp]. The rest length is rest * (1 + level)."""
    return s["amp"] * wsin(TWO_PI * s["freq"] * t + s["phase"])


# ---------------------------------------------------------------- genetic operators
def _edge_keys(g: dict) -> set:
    """The set of node-index pairs (encoded a*64+b) a genome's springs connect."""
    return {s["a"] * 64 + s["b"] for s in g["springs"]}


def compatibility(a: dict, b: dict) -> float:
    """Speciation distance: 0.4 per node-count difference, 0.3 per spring-count difference, the muscle-fraction
    difference, and the Jaccard distance of the edge sets. Identical topologies give 0."""
    sa = len(a["springs"])
    sb = len(b["springs"])
    ma = sum(s["muscle"] for s in a["springs"]) / sa
    mb = sum(s["muscle"] for s in b["springs"]) / sb
    ea = _edge_keys(a)
    eb = _edge_keys(b)
    inter = sum(1 for key in ea if key in eb)
    uni = len(ea) + len(eb) - inter
    jac = 1 if uni == 0 else inter / uni
    return 0.4 * abs(len(a["nodes"]) - len(b["nodes"])) + 0.3 * abs(sa - sb) + abs(ma - mb) + (1 - jac)


def _tournament(members: list, k: int, rng: Mulberry32) -> dict:
    """Best of k random picks (with replacement); ties keep the earlier pick."""
    best = members[rng.int(len(members))]
    for _ in range(1, k):
        cand = members[rng.int(len(members))]
        if cand["fitness"] > best["fitness"]:
            best = cand
    return best


def crossover(a: dict, b: dict, rng: Mulberry32, fa: float, fb: float) -> dict:
    """Crossover aligned by node index, as in walkers-core.js. The fitter parent is dominant and sets the node count.

    Shared node indices take a parent's position with probability 1/2. Springs present in both parents (same endpoint
    pair) take either parent's genes with probability 1/2. Springs only the dominant parent has are kept; springs
    only the other parent has are dropped. The result is repaired, so it is always connected.
    """
    d, o = (b, a) if fb > fa else (a, b)
    nN = len(d["nodes"])
    nodes = []
    for i in range(nN):
        if i < len(o["nodes"]) and rng.next() < 0.5:
            nodes.append({"x": o["nodes"][i]["x"], "y": o["nodes"][i]["y"]})
        else:
            nodes.append({"x": d["nodes"][i]["x"], "y": d["nodes"][i]["y"]})
    o_by = {s["a"] * 64 + s["b"]: s for s in o["springs"]}
    springs = []
    for sd in d["springs"]:
        other = o_by.get(sd["a"] * 64 + sd["b"])
        if other is not None and rng.next() < 0.5:
            springs.append(dict(other))
        else:
            springs.append(dict(sd))
    return repair({"nodes": nodes, "springs": [dict(s) for s in springs]})


def _nearest(g: dict, i: int, exclude: int) -> int:
    """Nearest node to i, skipping i and `exclude`; lowest index on ties; -1 when none."""
    best = -1
    best_d = math.inf
    for j in range(len(g["nodes"])):
        if j == i or j == exclude:
            continue
        d = _dist(g["nodes"], i, j)
        if d < best_d:
            best_d = d
            best = j
    return best


def _has_spring(g: dict, a: int, b: int) -> bool:
    """True when a spring already joins nodes a and b."""
    lo, hi = min(a, b), max(a, b)
    return any(s["a"] == lo and s["b"] == hi for s in g["springs"])


def _add_node(g: dict, rng: Mulberry32) -> None:
    """Add a node near a random node, linked to it and to its nearest neighbour."""
    i = rng.int(len(g["nodes"]))
    base = g["nodes"][i]
    x = q4(clamp(base["x"] + rng.range(-0.3, 0.3), -POS_MAX, POS_MAX))
    y = q4(clamp(base["y"] + rng.range(-0.3, 0.3), -POS_MAX, POS_MAX))
    g["nodes"].append({"x": x, "y": y})
    idx = len(g["nodes"]) - 1
    g["springs"].append(_make_spring(i, idx, _dist(g["nodes"], i, idx), rng.next() < 0.5, rng))
    j = _nearest(g, idx, i)
    if j >= 0:
        g["springs"].append(_make_spring(j, idx, _dist(g["nodes"], j, idx), rng.next() < 0.5, rng))


def _remove_node(g: dict, rng: Mulberry32) -> None:
    """Delete a random node, its springs, and renumber the nodes above it."""
    r = rng.int(len(g["nodes"]))
    del g["nodes"][r]
    springs = [s for s in g["springs"] if s["a"] != r and s["b"] != r]
    g["springs"] = [
        {**s, "a": s["a"] - 1 if s["a"] > r else s["a"], "b": s["b"] - 1 if s["b"] > r else s["b"]} for s in springs
    ]


def _add_spring(g: dict, rng: Mulberry32) -> None:
    """Try up to 12 random node pairs and add the first new spring found."""
    n = len(g["nodes"])
    for _ in range(12):
        a = rng.int(n)
        b = rng.int(n)
        if a == b or _has_spring(g, a, b):
            continue
        g["springs"].append(_make_spring(a, b, _dist(g["nodes"], a, b), rng.next() < 0.5, rng))
        return


def _remove_spring(g: dict, rng: Mulberry32) -> None:
    """Delete a random spring."""
    del g["springs"][rng.int(len(g["springs"]))]


def _toggle_muscle(g: dict, rng: Mulberry32) -> None:
    """Flip a random spring between muscle and passive spring."""
    s = g["springs"][rng.int(len(g["springs"]))]
    s["muscle"] = 0 if s["muscle"] else 1


def mutate(g0: dict, rng: Mulberry32, rate: float) -> dict:
    """Mutation as in walkers-core.js: parameter jitter on each node and spring (chance rate/2), then four structural
    operators (add node rate*0.15, remove node rate*0.1, add spring rate*0.2, remove spring rate*0.15) and toggle
    muscle rate*0.2. The result is repaired."""
    g = clone_genome(g0)
    for n in g["nodes"]:
        if rng.next() < rate * 0.5:
            n["x"] = q4(clamp(n["x"] + rng.range(-0.06, 0.06), -POS_MAX, POS_MAX))
            n["y"] = q4(clamp(n["y"] + rng.range(-0.06, 0.06), -POS_MAX, POS_MAX))
    for s in g["springs"]:
        if rng.next() < rate * 0.5:
            s["rest"] = q4(clamp(s["rest"] * (1 + rng.range(-0.1, 0.1)), REST_MIN, REST_MAX))
            s["k"] = q4(clamp(s["k"] * (1 + rng.range(-0.2, 0.2)), K_MIN, K_MAX))
            s["c"] = q4(clamp(s["c"] * (1 + rng.range(-0.2, 0.2)), C_MIN, C_MAX))
            s["amp"] = q4(clamp(s["amp"] + rng.range(-0.05, 0.05), 0, AMP_MAX))
            s["freq"] = q4(clamp(s["freq"] * (1 + rng.range(-0.2, 0.2)), FREQ_MIN, FREQ_MAX))
            s["phase"] = q4(s["phase"] + rng.range(-0.5, 0.5))
    if rng.next() < rate * 0.15 and len(g["nodes"]) < NODE_MAX:
        _add_node(g, rng)
    if rng.next() < rate * 0.1 and len(g["nodes"]) > NODE_MIN:
        _remove_node(g, rng)
    if rng.next() < rate * 0.2 and len(g["springs"]) < SPRING_MAX:
        _add_spring(g, rng)
    if rng.next() < rate * 0.15 and len(g["springs"]) > 1:
        _remove_spring(g, rng)
    # A body can lose all its springs when a node is removed; the draw still happens so the streams stay aligned.
    if rng.next() < rate * 0.2 and len(g["springs"]) > 0:
        _toggle_muscle(g, rng)
    return repair(g)


# ---------------------------------------------------------------- the evolver
GA_KEYS = tuple(GA_DEFAULTS.keys())


def _sorted_sum(values: list) -> float:
    """Left-to-right float sum (the JavaScript reduce). Python's sum() may compensate, so it is not used here."""
    acc = 0.0
    for v in values:
        acc += v
    return acc


class Evolver:
    """One evolution run: population, speciation, breeding, and a lineage record for every evaluated creature.

    Mirrors walkers-core.js Evolver. Members are dicts with id, genome, parents, gen, user, fitness, distance,
    exploded, fallen, species, and a `record` dict that is appended to `records` at evaluation time.
    """

    def __init__(
        self,
        seed: int = 1,
        pop_size: int = GA_DEFAULTS["popSize"],
        mutation_rate: float = GA_DEFAULTS["mutationRate"],
        world: dict | None = None,
    ):
        self.rng = Mulberry32(seed)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.world = {**DEFAULT_WORLD, **(world or {})}
        self.generation = 1
        self.next_id = 1
        self.next_species_id = 1
        self.species: list = []
        self.injected: list = []
        self.records: list = []
        self.population: list = []
        for _ in range(pop_size):
            self.population.append(self.member(random_genome(self.rng), []))

    def member(self, genome: dict, parents: list, user: bool = False) -> dict:
        """A new population member, stamped with the current generation."""
        m = {
            "id": self.next_id,
            "genome": genome,
            "parents": parents,
            "gen": self.generation,
            "user": user,
            "fitness": 0.0,
            "distance": 0.0,
            "exploded": False,
            "fallen": False,
            "species": 0,
        }
        self.next_id += 1
        return m

    def inject(self, genome: dict) -> None:
        """Queue a user creature. It joins the next generation in place of a child from the back of the list."""
        self.injected.append(repair(genome))

    def evaluate(self) -> None:
        """Score the whole population in one batch and append a lineage record for each member."""
        results = simulate([m["genome"] for m in self.population], self.world)
        for m, r in zip(self.population, results):
            m["fitness"] = r["fitness"]
            m["distance"] = r["distance"]
            m["exploded"] = r["exploded"]
            m["fallen"] = r["fallen"]
            m["record"] = {
                "id": m["id"],
                "gen": m["gen"],
                "parents": m["parents"],
                "fitness": r["fitness"],
                "distance": r["distance"],
                "exploded": r["exploded"],
                "fallen": r["fallen"],
                "user": m["user"],
                "species": 0,
                "code": encode(m["genome"]),
            }
            self.records.append(m["record"])

    def speciate(self, threshold: float = GA_DEFAULTS["speciesThreshold"]) -> list:
        """Assign each member to the first species whose representative is within threshold, founding new species
        as needed. Then each surviving species draws a random member as its next representative (one draw per species,
        in species order). Returns the ids of species created this generation."""
        for sp in self.species:
            sp["members"] = []
        created = []
        for m in self.population:
            sp = next((s for s in self.species if compatibility(m["genome"], s["rep"]) < threshold), None)
            if sp is None:
                sp = {"id": self.next_species_id, "rep": m["genome"], "members": [], "created": self.generation}
                self.next_species_id += 1
                self.species.append(sp)
                created.append(sp["id"])
            sp["members"].append(m)
            m["species"] = sp["id"]
            m["record"]["species"] = sp["id"]
        self.species = [s for s in self.species if s["members"]]
        for sp in self.species:
            sp["rep"] = sp["members"][self.rng.int(len(sp["members"]))]["genome"]
        return created

    def breed(self) -> list:
        """Build the next population. Quotas come from fitness sharing (shifted fitness divided by species size),
        floored, with the remainder given to the largest fractional parts (earlier species first on ties). Within a
        species: an elite copy of its best when the species has eliteMin members or holds the global best, then
        children of tournament picks from the top half, by crossover or by cloning, then mutation. Injected user
        creatures take the last slots."""
        P = self.pop_size
        min_f = math.inf
        best_member = None
        for m in self.population:
            if m["fitness"] < min_f:
                min_f = m["fitness"]
            if best_member is None or m["fitness"] > best_member["fitness"]:
                best_member = m
        shares = []
        for sp in self.species:
            s = 0.0
            for m in sp["members"]:
                s += (m["fitness"] - min_f + 1) / len(sp["members"])
            shares.append(s)
        total = _sorted_sum(shares)
        exact = [(P * s) / total for s in shares]
        quota = [math.floor(x) for x in exact]
        rem = P - sum(quota)
        order = sorted(range(len(exact)), key=lambda i: (-(exact[i] - quota[i]), i))
        for r in range(rem):
            quota[order[r % len(order)]] += 1

        nxt: list = []
        for si, sp in enumerate(self.species):
            q = quota[si]
            if q <= 0:
                continue
            ranked = sorted(sp["members"], key=lambda m: (-m["fitness"], m["id"]))
            keep = max(1, math.ceil(len(ranked) * GA_DEFAULTS["survivalFrac"]))
            survivors = ranked[:keep]
            made = 0
            has_best = any(m is best_member for m in sp["members"])
            if len(sp["members"]) >= GA_DEFAULTS["eliteMin"] or has_best:
                nxt.append(self.member(clone_genome(ranked[0]["genome"]), [ranked[0]["id"]]))
                made += 1
            while made < q:
                if len(survivors) >= 2 and self.rng.next() < GA_DEFAULTS["crossoverRate"]:
                    a = _tournament(survivors, GA_DEFAULTS["tournament"], self.rng)
                    b = _tournament(survivors, GA_DEFAULTS["tournament"], self.rng)
                    child = crossover(a["genome"], b["genome"], self.rng, a["fitness"], b["fitness"])
                    parents = [a["id"], b["id"]]
                else:
                    a = _tournament(survivors, GA_DEFAULTS["tournament"], self.rng)
                    child = clone_genome(a["genome"])
                    parents = [a["id"]]
                nxt.append(self.member(mutate(child, self.rng, self.mutation_rate), parents))
                made += 1
        slots = min(len(self.injected), len(nxt))
        for k in range(slots):
            nxt[len(nxt) - 1 - k] = self.member(clone_genome(self.injected[k]), [], True)
        self.injected = self.injected[slots:]
        while len(nxt) > P:
            nxt.pop()
        while len(nxt) < P:
            nxt.append(self.member(random_genome(self.rng), []))
        return nxt

    def step(self, top_k: int = 0) -> dict:
        """One generation: evaluate, summarise, speciate, breed. With top_k > 0 the summary carries `top`: frames of
        the top_k evaluated members, taken before breeding (mirrors step(topK) in the JavaScript)."""
        self.evaluate()
        # Speciate before summarising, so the champion's species is set (no random draws happen in either).
        created = self.speciate()
        summary = self.summary()
        summary["newSpecies"] = created
        summary["speciesSizes"] = [{"id": sp["id"], "size": len(sp["members"])} for sp in self.species]
        if top_k > 0:
            summary["top"] = self.top_frames(top_k, 4)
        # The generation counter moves on before breeding, so the children carry the new generation number.
        self.generation += 1
        self.population = self.breed()
        return summary

    def summary(self) -> dict:
        """Statistics of the evaluated population: best, mean and median fitness (the mean sums in sorted order),
        the champion (highest fitness, lowest id on ties), user creatures, and the species count."""
        pop = self.population
        fits = sorted(m["fitness"] for m in pop)
        n = len(fits)
        mean = _sorted_sum(fits) / n
        median = fits[(n - 1) // 2] if n % 2 else (fits[n // 2 - 1] + fits[n // 2]) / 2
        champ = pop[0]
        for m in pop:
            if m["fitness"] > champ["fitness"] or (m["fitness"] == champ["fitness"] and m["id"] < champ["id"]):
                champ = m
        return {
            "gen": self.generation,
            "best": champ["fitness"],
            "mean": mean,
            "median": median,
            "exploded": sum(1 for m in pop if m["exploded"]),
            "champion": {
                "id": champ["id"],
                "fitness": champ["fitness"],
                "distance": champ["distance"],
                "parents": champ["parents"],
                "species": champ["species"],
                "code": champ["record"]["code"],
            },
            "user": [{"id": m["id"], "fitness": m["fitness"], "distance": m["distance"]} for m in pop if m["user"]],
            "species": len(self.species),
        }

    def top_frames(self, k: int = 4, sample_every: int = 4) -> list:
        """Frames of the k fittest members (positions every sample_every steps), for the live view."""
        ranked = sorted(self.population, key=lambda m: (-m["fitness"], m["id"]))[:k]
        results = simulate([m["genome"] for m in ranked], self.world, sample_every)
        return [
            {
                "id": m["id"],
                "fitness": r["fitness"],
                "N": r["N"],
                "frameCount": r["frameCount"],
                "sampleEvery": sample_every,
                "frames": r["frames"],
            }
            for m, r in zip(ranked, results)
        ]
