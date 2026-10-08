"""The committed GA result: tetris/tuned.json, plus the hand-picked weights as a fallback.

tuned.json holds the nine weights in FEATURES order, the GA settings that produced them, the training
fitness, and the per-generation history that the web page plots. If the file is missing (a fresh
checkout before `python -m tetris evolve`), the hand-picked weights are used and the history is empty.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .features import FEATURES, HAND_WEIGHTS

TUNED_PATH = Path(__file__).with_name("tuned.json")


@lru_cache(maxsize=1)
def load_tuned() -> dict:
    """The tuned document, or the hand-picked fallback. Treat the result as read-only."""
    if not TUNED_PATH.exists():
        return {"features": list(FEATURES), "weights": list(HAND_WEIGHTS), "train_fitness": None,
                "config": None, "history": [], "source": "hand-picked fallback"}
    doc = json.loads(TUNED_PATH.read_text())
    assert doc["features"] == list(FEATURES), "tuned.json was written for a different feature set"
    return doc
