"""The bundled public-domain corpora, read from markov/data/*.txt.

Each file has a header (its source and public-domain status), a line containing only `---`, then the text.
The web page reads the same files (web/data/markov/), and tests/test_markov.py checks the two copies match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPORA = ("alice", "pride", "sonnets", "constitution")
SEPARATOR = "\n---\n"


@dataclass(frozen=True)
class Corpus:
    name: str
    source: str  # the header's Source line
    body: str    # the text after the header


def parse_corpus(name: str, raw: str) -> Corpus:
    """Split a corpus file into its header and body. The header is ignored by the model."""
    head, sep, body = raw.partition(SEPARATOR)
    if not sep:
        raise ValueError(f"corpus {name!r} has no '---' line between its header and text")
    source = next((ln[len("Source:"):].strip() for ln in head.splitlines() if ln.startswith("Source:")), name)
    return Corpus(name, source, body)


def load_corpus(name: str) -> Corpus:
    """Load a bundled corpus by name, or any text file by path."""
    if name in CORPORA:
        path = DATA_DIR / f"{name}.txt"
    else:
        path = Path(name)
        if not path.is_file():
            raise ValueError(f"unknown corpus {name!r}: use one of {', '.join(CORPORA)} or a text file path")
    raw = path.read_text(encoding="utf-8")
    if SEPARATOR not in raw:
        return Corpus(path.stem, path.name, raw)
    return parse_corpus(path.stem, raw)
