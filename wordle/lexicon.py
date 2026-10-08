"""The word lists, and the feedback table that makes the solver fast.

The solver needs the feedback pattern of every allowed guess against every possible answer. Computing a
pattern takes a few Python steps, and a strategy looks at millions of (guess, answer) pairs per game. So the
table is built once, as a NumPy array of shape (guesses, answers) with one byte per entry (a pattern id,
0..242, see feedback.py). A later lookup is `table[g, answers]`, which is a single gather.

Two word lists:
  answers  the words the hidden answer can be (3,568 five-letter words, from the common tier of SCOWL)
  guesses  every word the player may guess: the answers plus the rest of SCOWL's five-letter words, so
           every answer is also a legal guess. (The official Wordle lists are not used.)
Both files are plain text, one lowercase word per line. wordle/build_lists.py regenerates them from SCOWL.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .feedback import WORD_LENGTH

DATA_DIR = Path(__file__).resolve().parent / "data"
ANSWERS_FILE = DATA_DIR / "answers.txt"
GUESSES_FILE = DATA_DIR / "guesses.txt"

# Column weights that turn a (rows, 5) array of tile values into base-3 pattern ids (first tile least significant).
_WEIGHTS = np.array([3**i for i in range(WORD_LENGTH)], dtype=np.int32)


def encode(words) -> np.ndarray:
    """Words as an (n, 5) int16 array of letter codes 0..25 (a = 0)."""
    arr = np.array([[ord(c) - 97 for c in w] for w in words], dtype=np.int16)
    return arr.reshape(len(words), WORD_LENGTH)


def _pattern_row(guess: np.ndarray, answers: np.ndarray) -> np.ndarray:
    """Pattern ids of one guess against every answer at once (vectorised over answers).

    Greens first: a position where the guess letter equals the answer letter is green, and the answer letter
    there is blanked out (-1) so it cannot match a yellow. Then yellows, left to right over the guess: a
    non-green guess tile is yellow if an unused answer letter of that kind is left. Using that letter blanks
    the first such position, which is the one-letter-per-answer-letter rule from feedback.py.
    """
    green = answers == guess[None, :]  # (A, 5) bool
    rem = np.where(green, np.int16(-1), answers)  # unused answer letters; -1 = green or already used
    tiles = np.where(green, 2, 0).astype(np.int32)  # gray unless green so far
    for i in range(WORD_LENGTH):
        match = rem == guess[i]  # (A, 5): unused answer positions holding this guess letter
        yellow = ~green[:, i] & match.any(axis=1)
        tiles[:, i] += yellow
        rows = np.flatnonzero(yellow)
        if rows.size:
            rem[rows, match[rows].argmax(axis=1)] = -1  # consume the first matching unused letter
    return tiles @ _WEIGHTS


def build_table(guess_codes: np.ndarray, answer_codes: np.ndarray) -> np.ndarray:
    """The (guesses, answers) table of pattern ids as uint8. About 2 s for 6,748 x 3,568 on one core."""
    table = np.empty((guess_codes.shape[0], answer_codes.shape[0]), dtype=np.uint8)
    for g in range(guess_codes.shape[0]):
        table[g] = _pattern_row(guess_codes[g], answer_codes)
    return table


@dataclass(frozen=True)
class Lexicon:
    """The two word lists, their index maps, and the feedback table. Index i of `answers` is column i of
    `table`; index g of `guesses` is row g."""

    answers: tuple[str, ...]
    guesses: tuple[str, ...]
    table: np.ndarray  # (len(guesses), len(answers)) uint8 pattern ids
    guess_of_answer: np.ndarray  # (len(answers),) index of each answer in `guesses`
    answer_index: dict
    guess_index: dict
    # Results that depend only on the lexicon, such as the opening ranking. Stored here (not in a module
    # dict keyed by id()), so a discarded lexicon takes its cache with it.
    cache: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_words(cls, answers, guesses=None) -> Lexicon:
        """Build a lexicon from word lists. Every answer is added to the guesses if it is missing."""
        answers = tuple(sorted(set(answers)))
        guesses = tuple(sorted(set(guesses or ()) | set(answers)))
        table = build_table(encode(guesses), encode(answers))
        guess_index = {w: i for i, w in enumerate(guesses)}
        return cls(
            answers=answers,
            guesses=guesses,
            table=table,
            guess_of_answer=np.array([guess_index[w] for w in answers], dtype=np.int32),
            answer_index={w: i for i, w in enumerate(answers)},
            guess_index=guess_index,
        )

    def pattern(self, guess: str, answer_i: int) -> int:
        """Pattern id of a guess against answer number `answer_i`, from the table."""
        return int(self.table[self.guess_index[guess], answer_i])


def read_words(path) -> list[str]:
    """Words from a one-per-line file. Blank lines are skipped."""
    return [w for w in Path(path).read_text(encoding="utf-8", errors="replace").split() if w]


_LOCK = threading.Lock()
_LEXICON: Lexicon | None = None


def get_lexicon() -> Lexicon:
    """The bundled lexicon, built on first use and then shared. The first call takes a couple of seconds
    (the table); later calls return the same object. Thread-safe, so a server can call it from worker threads."""
    global _LEXICON
    with _LOCK:
        if _LEXICON is None:
            _LEXICON = Lexicon.from_words(read_words(ANSWERS_FILE), read_words(GUESSES_FILE))
        return _LEXICON
