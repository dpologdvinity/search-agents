"""Chess rules for the endgames in this package: a white king and one white piece (queen or rook)
against a lone black king.

Squares are numbered 0..63 with a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63, so a square is
8 * rank + file with both counted from 0. A position is the tuple (wk, wp, bk, stm): the white king's
square, the white piece's square, the black king's square, and the side to move (WHITE = 0, BLACK = 1).
A move is (piece, from, to) where piece is 'K' (white king), 'k' (black king), 'Q' or 'R' (white piece).

Why this material: the black king is alone, so its only capture is taking the white piece, and that leaves
king against king, a dead draw. Every other move is quiet. A quiet move can be undone by one "unmove"
(a predecessor), which is exactly what the retrograde solver in retro.py walks backwards with.

Everything here is plain integer arithmetic. Lines between squares are precomputed once at import, so a
legality check is a handful of table lookups.
"""

from __future__ import annotations

WHITE, BLACK = 0, 1
PIECES = ("Q", "R")
_FILES = "abcdefgh"

# Direction steps (file, rank) for each sliding piece. A rook slides on files and ranks; a queen also
# on diagonals.
_DIRS = {
    "R": ((1, 0), (-1, 0), (0, 1), (0, -1)),
    "Q": ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)),
}


def square_name(sq: int) -> str:
    """'e4' style name of a square number."""
    return _FILES[sq & 7] + str((sq >> 3) + 1)


def parse_square(name: str) -> int:
    """Square number of an 'e4' style name; raises ValueError on anything else."""
    name = name.strip().lower()
    if len(name) != 2 or name[0] not in _FILES or name[1] not in "12345678":
        raise ValueError(f"not a square: {name!r} (use a1..h8)")
    return (int(name[1]) - 1) * 8 + _FILES.index(name[0])


def _rays(sq: int, dirs) -> tuple[tuple[int, ...], ...]:
    """For each direction, the squares a slider on sq reaches, nearest first, up to the board edge."""
    f, r = sq & 7, sq >> 3
    out = []
    for df, dr in dirs:
        ray = []
        ff, rr = f + df, r + dr
        while 0 <= ff < 8 and 0 <= rr < 8:
            ray.append(rr * 8 + ff)
            ff += df
            rr += dr
        out.append(tuple(ray))
    return tuple(out)


# RAYS[piece][sq] lists the rays a piece on sq can slide along. The king's neighbours are a separate table.
RAYS = {p: tuple(_rays(sq, dirs) for sq in range(64)) for p, dirs in _DIRS.items()}

KING_NEIGHBOURS: tuple[tuple[int, ...], ...] = tuple(
    tuple(sq for sq in range(64)
          if max(abs((sq & 7) - (s & 7)), abs((sq >> 3) - (s >> 3))) == 1)
    for s in range(64)
)
# The same neighbours as a bit mask, so "are these two kings touching" is one shift and mask.
_KING_MASK = tuple(sum(1 << n for n in KING_NEIGHBOURS[s]) for s in range(64))


def _pair_tables():
    """For every ordered pair of squares (a, b): is b on a rook line / diagonal from a, and which squares lie between.

    The index is a * 64 + b. BETWEEN is a bit mask of the squares strictly between a and b, which must be empty
    for a slider on a to reach b.
    """
    rook, diag, between = [False] * 4096, [False] * 4096, [0] * 4096
    for a in range(64):
        af, ar = a & 7, a >> 3
        for b in range(64):
            if a == b:
                continue
            bf, br = b & 7, b >> 3
            df, dr = bf - af, br - ar
            i = a * 64 + b
            rook[i] = df == 0 or dr == 0
            diag[i] = abs(df) == abs(dr)
            if not (rook[i] or diag[i]):
                continue
            sf = (df > 0) - (df < 0)
            sr = (dr > 0) - (dr < 0)
            mask, f, r = 0, af + sf, ar + sr
            while (f, r) != (bf, br):
                mask |= 1 << (r * 8 + f)
                f += sf
                r += sr
            between[i] = mask
    queen = [x or y for x, y in zip(rook, diag)]
    return {"R": rook, "Q": queen}, between


ALIGNED, BETWEEN = _pair_tables()


def adjacent(a: int, b: int) -> bool:
    """True when the two squares touch (including diagonally), which two kings may never do."""
    return (_KING_MASK[a] >> b) & 1 == 1


def attacks(piece: str, src: int, dst: int, occupied: int) -> bool:
    """Does a piece on src attack dst, given the bit mask `occupied` of squares that block sliding lines?

    Only squares strictly between src and dst can block, so the endpoints in `occupied` do not matter.
    """
    i = src * 64 + dst
    return ALIGNED[piece][i] and (BETWEEN[i] & occupied) == 0


def _slides(piece: str, src: int, blockers: int):
    """Squares a piece on src can move to: every square along its rays up to the first blocker (excluded)."""
    for ray in RAYS[piece][src]:
        for sq in ray:
            if (blockers >> sq) & 1:
                break
            yield sq


def is_legal(piece: str, pos) -> bool:
    """A position can occur in a game: distinct squares, kings not touching, and the side not to move not in check.

    With a lone black king only the black king can be in check, and only from the white piece. The white
    king never can be: kings may not touch, and black has no other piece.
    """
    wk, wp, bk, stm = pos
    if wk == wp or wk == bk or wp == bk or adjacent(wk, bk):
        return False
    return stm == BLACK or not attacks(piece, wp, bk, 1 << wk)


def in_check(piece: str, pos) -> bool:
    """Is the side to move in check? Only Black can be (see is_legal)."""
    wk, wp, bk, stm = pos
    return stm == BLACK and attacks(piece, wp, bk, 1 << wk)


def legal_moves(piece: str, pos) -> list[tuple[str, int, int]]:
    """Every legal move in pos, including the black king taking the white piece.

    White: the king steps to a free square not touching the black king; the piece slides along its rays and
    stops at the first king it meets (a piece never takes the king, so the king square is not a destination).
    Black: the king steps to a neighbour not touching the white king and not attacked by the white piece.
    The attack is checked with the white king as the only blocker, because the black king is moving off
    its own square and must not shield the squares behind it.
    """
    wk, wp, bk, stm = pos
    moves = []
    if stm == WHITE:
        for d in KING_NEIGHBOURS[wk]:
            if d != wp and not adjacent(d, bk):
                moves.append(("K", wk, d))
        for d in _slides(piece, wp, (1 << wk) | (1 << bk)):
            moves.append((piece, wp, d))
    else:
        guard = 1 << wk
        for d in KING_NEIGHBOURS[bk]:
            if adjacent(d, wk):
                continue  # the white king guards its neighbours
            if d == wp or not attacks(piece, wp, d, guard):  # taking the piece is safe: nothing recaptures
                moves.append(("k", bk, d))
    return moves


def apply_move(pos, move):
    """The position after `move`, with the other side to move.

    Returns None when the black king takes the white piece: king against king is a draw, so the result is
    not a position in the tablebase.
    """
    wk, wp, bk, stm = pos
    mover, _, dst = move
    if mover == "k" and dst == wp:
        return None
    if mover == "K":
        wk = dst
    elif mover == "k":
        bk = dst
    else:
        wp = dst
    return (wk, wp, bk, 1 - stm)


def predecessors(piece: str, pos) -> list[tuple[int, int, int, int]]:
    """Every legal position from which one quiet move reaches `pos`. This is the inverse of legal_moves.

    The retrograde solver walks these backwards from checkmates. Captures are never predecessors: a capture
    would have removed a piece, and the result would be king against king, which is not in the table.

    If White just moved (pos has Black to move), the mover was the white king or the white piece. The piece
    came back along the same rays it could have moved along, with the same blockers, so its origins are the
    squares it can slide to from where it stands now. A king origin is any neighbour. If Black just moved,
    its king came from a neighbouring square. Each candidate is kept only if it is a legal position.
    """
    wk, wp, bk, stm = pos
    out = []
    if stm == BLACK:
        for a in KING_NEIGHBOURS[wk]:
            if a != wp and a != bk:
                out.append((a, wp, bk, WHITE))
        for a in _slides(piece, wp, (1 << wk) | (1 << bk)):
            out.append((wk, a, bk, WHITE))
    else:
        for a in KING_NEIGHBOURS[bk]:
            if a != wk and a != wp:
                out.append((wk, wp, a, BLACK))
    return [p for p in out if is_legal(piece, p)]


def move_name(piece: str, pos, move, after) -> str:
    """Short algebraic name of a move, like 'Qd4', 'Kxd1' or 'Ke2'. Pieces are 'K', 'Q' or 'R'."""
    mover, src, dst = move
    letter = "K" if mover in ("K", "k") else mover
    capture = "x" if after is None else ""
    return f"{letter}{capture}{square_name(dst)}"


def to_fen(piece: str, pos) -> str:
    """FEN for the position (placement, side to move, and the default fields), with the white piece as given."""
    wk, wp, bk, stm = pos
    board = {wk: "K", wp: piece, bk: "k"}
    rows = []
    for rank in range(7, -1, -1):
        run, row = 0, ""
        for f in range(8):
            ch = board.get(rank * 8 + f)
            if ch is None:
                run += 1
                continue
            if run:
                row += str(run)
                run = 0
            row += ch
        if run:
            row += str(run)
        rows.append(row)
    side = "w" if stm == WHITE else "b"
    return "/".join(rows) + f" {side} - - 0 1"


def parse_fen(text: str) -> tuple[str, tuple[int, int, int, int]]:
    """Read a FEN with exactly a white king, a black king and one white queen or rook, and nothing else.

    Returns (piece, position). Raises ValueError with the reason when the setup is not a legal position.
    Only the placement and side-to-move fields are used; castling, en passant and counters are ignored.
    """
    parts = text.split()
    if not parts:
        raise ValueError("empty FEN")
    rows = parts[0].split("/")
    if len(rows) != 8:
        raise ValueError("FEN needs 8 ranks separated by '/'")
    found: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        rank = 7 - i
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
                continue
            if ch not in "KkQR":
                raise ValueError(f"only K, k, Q, R are allowed, got {ch!r}")
            if f > 7:
                raise ValueError("a rank has more than 8 squares")
            found.setdefault(ch, []).append(rank * 8 + f)
            f += 1
        if f != 8:
            raise ValueError(f"rank {rank + 1} does not have 8 squares")
    side = parts[1].lower() if len(parts) > 1 else "w"
    if side not in ("w", "b"):
        raise ValueError("side to move must be w or b")
    pieces = [p for p in ("Q", "R") if p in found]
    if len(pieces) != 1 or len(found.get("Q", [])) + len(found.get("R", [])) != 1:
        raise ValueError("need exactly one white queen or rook (Q or R)")
    if found.get("K") is None or len(found["K"]) != 1 or found.get("k") is None or len(found["k"]) != 1:
        raise ValueError("need exactly one white king (K) and one black king (k) and nothing else")
    piece = pieces[0]
    pos = (found["K"][0], found[piece][0], found["k"][0], WHITE if side == "w" else BLACK)
    if not is_legal(piece, pos):
        raise ValueError("illegal position: the kings touch, or the side not to move is in check")
    return piece, pos
