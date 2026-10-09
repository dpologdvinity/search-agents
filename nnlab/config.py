"""Default hyperparameters and option lists, shared by the CLI, the training loop and the server meta route.

web/js/nnlab-core.js and web/nnlab.html carry the same defaults. Each default is picked so that the
default preset (circles) separates in a few hundred epochs, and the reasons are written on the page.
"""

from __future__ import annotations

from .data import FEATURES, PRESETS
from .mlp import ACTIVATIONS

DEFAULTS = {
    "data": "circles",
    "n": 200,
    "seed": 0,
    "noise": 0.0,
    "features": ["x", "y"],
    "test_frac": 0.2,
    "hidden": [6, 6],
    "act": "tanh",
    "lr": 0.03,
    "batch": 16,
    "l2": 0.0,
    "optimizer": "adam",
    "epochs": 300,
}

OPTIMIZERS = ("adam", "sgd")
MAX_HIDDEN_LAYERS = 4
MAX_NEURONS = 8

OPTIONS = {
    "presets": list(PRESETS),
    "features": list(FEATURES),
    "activations": list(ACTIVATIONS),
    "optimizers": list(OPTIMIZERS),
}
