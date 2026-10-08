"""Regenerate wordle/data/answers.txt and guesses.txt from SCOWL word lists.

    python -m wordle.build_lists --answers /usr/share/dict/american-english-small \
                                 --guesses /usr/share/dict/american-english-large

Sources: SCOWL (Spell Checker Oriented Word Lists) by Kevin Atkinson, packaged by Debian as wamerican-small,
wamerican and wamerican-large (installed as /usr/share/dict/american-english-{small,large}). These are the
lists behind /usr/share/dict/words. The bundled files were built from the small and large tiers because
/usr/share/dict/words was not present on the machine that made them. Licence: SCOWL's copyright notice grants
"permission to use, copy, modify, distribute and sell these word lists ... for any purpose ... without fee",
and the SCOWL README calls the combined work MIT-like. The full notice ships as wordle/data/SCOWL-COPYRIGHT.txt.

Why two tiers: SCOWL orders its tiers by how common the words are, so the small tier is the common core
that makes a fair answer pool, and the large tier adds the rarer five-letter words that players may guess
but that are unlikely to be answers. Only lowercase a-z words of exactly five letters are kept, so proper
nouns, abbreviations, possessives and accented words are dropped.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .feedback import WORD_LENGTH
from .lexicon import ANSWERS_FILE, GUESSES_FILE, read_words

DEFAULT_SOURCE = Path("/usr/share/dict/words")
_WORD = re.compile(rf"[a-z]{{{WORD_LENGTH}}}")


def five_letter(words) -> list[str]:
    """Lowercase a-z words of exactly five letters, deduplicated and sorted."""
    return sorted({w for w in words if _WORD.fullmatch(w)})


def build(answers_src: Path, guesses_src: Path, answers_out: Path = ANSWERS_FILE, guesses_out: Path = GUESSES_FILE):
    """Write the two lists. Every answer is also written into the guesses."""
    answers = five_letter(read_words(answers_src))
    guesses = sorted(set(five_letter(read_words(guesses_src))) | set(answers))
    answers_out.write_text("\n".join(answers) + "\n", encoding="ascii")
    guesses_out.write_text("\n".join(guesses) + "\n", encoding="ascii")
    return len(answers), len(guesses)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the Wordle answer and guess lists from SCOWL word lists.")
    parser.add_argument("--answers", type=Path, default=DEFAULT_SOURCE, help="common-word list for the answers")
    parser.add_argument("--guesses", type=Path, default=DEFAULT_SOURCE, help="wider list; adds allowed guesses")
    args = parser.parse_args(argv)
    for path in (args.answers, args.guesses):
        if not path.is_file():
            parser.error(f"{path} is not a file; install the Debian package wamerican-small / wamerican-large "
                         "or pass the path of a one-word-per-line list")
    n_answers, n_guesses = build(args.answers, args.guesses)
    print(f"wrote {n_answers} answers and {n_guesses} guesses to {ANSWERS_FILE.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
