"""
Pyranometer station registry.

Coordinates are the station's own reported position, used to select the
nearest grid cell / pixel in gridded forecast and nowcast products.
"""

from __future__ import annotations

STATIONS: dict[str, dict[str, float]] = {
    "risoe": {
        "lat": 55.694243,
        "lon": 12.101793,
        "alt": None,
    },
    "lyngby": {
        "lat": 55.79064,
        "lon": 12.52505,
        "alt": 50.0,
    },
}
