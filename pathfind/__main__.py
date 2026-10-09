"""Entry point for `python -m pathfind`. See pathfind/cli.py for the subcommands."""

import sys

from .cli import main

sys.exit(main())
