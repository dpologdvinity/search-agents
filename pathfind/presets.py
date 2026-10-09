"""Named maps for the lab's preset buttons and the server's presets route.

Each preset is a recipe, not a grid: the same seed rebuilds the same map in Python and in the browser.
"""

from __future__ import annotations

PRESETS: list[dict] = [
    {
        "id": "lanes",
        "name": "Lanes",
        "maze": "prim",
        "size": 41,
        "seed": 3,
        "swamp": 0,
        "diagonal": False,
        "note": "A perfect maze with one route between any two cells.",
    },
    {
        "id": "swamp",
        "name": "Swamp crossing",
        "maze": "prim",
        "size": 41,
        "seed": 7,
        "swamp": 18,
        "diagonal": False,
        "note": "The same kind of maze with swamp (cost 5) on 18% of the open cells.",
    },
    {
        "id": "rooms",
        "name": "Room block",
        "maze": "rooms",
        "size": 61,
        "seed": 11,
        "swamp": 0,
        "diagonal": True,
        "note": "Rooms joined by corridors with loops, searched with diagonal moves.",
    },
    {
        "id": "scatter",
        "name": "Scatter field",
        "maze": "scatter",
        "size": 41,
        "seed": 5,
        "swamp": 0,
        "diagonal": False,
        "density": 30,
        "note": "Random walls at 30%. Routes may not exist.",
    },
    {
        "id": "open",
        "name": "Open ground",
        "maze": "open",
        "size": 41,
        "seed": 1,
        "swamp": 12,
        "diagonal": True,
        "note": "Empty ground with swamp patches; the cost-only case.",
    },
]
