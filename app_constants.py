from __future__ import annotations

from typing import Any

WIND16 = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]

WIND16_ARROWS = {
    "N": "\u2191",
    "NNE": "\u2191",
    "NE": "\u2197",
    "ENE": "\u2197",
    "E": "\u2192",
    "ESE": "\u2198",
    "SE": "\u2198",
    "SSE": "\u2193",
    "S": "\u2193",
    "SSW": "\u2193",
    "SW": "\u2199",
    "WSW": "\u2199",
    "W": "\u2190",
    "WNW": "\u2196",
    "NW": "\u2196",
    "NNW": "\u2191",
}

DEFAULT_LOCATION: dict[str, Any] = {
    "lat": 0.0,
    "lon": 0.0,
    "label": "Location not set",
    "source": "unset",
    "resolved_at": "",
}

UI_THEME_LIGHT = "light"
UI_THEME_DARK = "dark"
UI_THEME_OPTIONS = {UI_THEME_LIGHT, UI_THEME_DARK}
