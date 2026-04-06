"""Run TrafficMind startup sanity checks and report results.

Usage::

    trafficmind-check                       # installed console script
    python -m trafficmind.cli              # module entry point
    python scripts/check_trafficmind.py           # uses current env
    TRAFFICMIND_PROFILE=staging python scripts/check_trafficmind.py

Exits 0 when all checks pass, 1 when any check fails.
"""

from __future__ import annotations

import pathlib
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trafficmind.cli import main


if __name__ == "__main__":
    main()
