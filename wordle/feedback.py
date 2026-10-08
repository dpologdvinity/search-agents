"""The Wordle scoring rule, one guess against one answer.

A feedback pattern has five tiles, each gray (0), yellow (1: right letter, wrong place) or green (2: right
letter, right place). Patterns are numbered in base 3 with the first tile as the least significant digit, so
all-gray is 0 and all-green is 242. That gives 243 patterns, which fit in one byte for the lookup table.

The duplicate-letter rule is the subtle part. Greens are settled first and use up their answer letters. Then
yellows are handed out left to right, each one consuming an unused answer letter of the same kind. So a guess
with two E's against an answer with one E marks only the first E yellow; the second stays gray.
"""

from __future__ import annotations

WORD_LENGTH = 5
GRAY, YELLOW, GREEN = 0, 1, 2
N_PATTERNS = 3**WORD_LENGTH  # 243
ALL_GREEN = N_PATTERNS - 1  # 242
_SYMBOLS = {GRAY: ".", YELLOW: "y", GREEN: "g"}
_PARSE = {"g": GREEN, "y": YELLOW, ".": GRAY, "x": GRAY, "b": GRAY}  # x and b also mean gray


def score(guess: str, answer: str) -> int:
    """Pattern id for `guess` against `answer`. Plain Python: it scores single games, and it is the reference
    that the vectorised table in lexicon.py is tested against."""
    tiles = [GRAY] * WORD_LENGTH
    left: dict[str, int] = {}
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            tiles[i] = GREEN
        else:
            left[a] = left.get(a, 0) + 1  # answer letters that no green used up
    for i, g in enumerate(guess):
        if tiles[i] == GREEN:
            continue
        if left.get(g, 0) > 0:  # a yellow consumes one of the unused answer letters
            tiles[i] = YELLOW
            left[g] -= 1
    return pattern_id(tiles)


def pattern_id(tiles) -> int:
    """Base-3 number from five tile values, first tile least significant."""
    p = 0
    for t in reversed(tiles):
        p = p * 3 + int(t)
    return p


def tiles(pattern: int) -> tuple[int, ...]:
    """The five tile values of a pattern id, first tile first."""
    out = []
    for _ in range(WORD_LENGTH):
        out.append(pattern % 3)
        pattern //= 3
    return tuple(out)


def to_text(pattern: int) -> str:
    """Feedback as text: g = green, y = yellow, . = gray, so "gy..g" is green, yellow, gray, gray, green."""
    return "".join(_SYMBOLS[t] for t in tiles(pattern))


def from_text(text: str) -> int:
    """Parse feedback typed as g / y / . (x or b also mean gray). Spaces and commas are ignored."""
    cleaned = [c for c in text.lower() if c not in " ,"]
    if len(cleaned) != WORD_LENGTH or any(c not in _PARSE for c in cleaned):
        raise ValueError(f"feedback needs {WORD_LENGTH} of g (green), y (yellow) or . (gray), e.g. gy..g")
    return pattern_id([_PARSE[c] for c in cleaned])
