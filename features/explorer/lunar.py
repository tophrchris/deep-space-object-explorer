from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

LUNAR_PHASE_EMOJIS: dict[str, str] = {
    "new": "🌑",
    "first_quarter": "🌓",
    "full": "🌕",
    "third_quarter": "🌗",
}


def lunar_phase_emoji(phase_key: Any) -> str:
    key = str(phase_key or "").strip().lower()
    return LUNAR_PHASE_EMOJIS.get(key, "🌙")


def lunar_altitude_row_label(phase_key: Any) -> str:
    emoji = lunar_phase_emoji(phase_key)
    return f"Lunar Altitude {emoji}".strip()


def format_lunar_altitude_display(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(numeric):
        return "-"
    if numeric < 0.0:
        return "-"
    return f"{numeric:.1f} deg"


def format_lunar_percent_display(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(numeric):
        return "-"
    if f"{numeric:.0f}%" == "0%":
        return "-"
    return f"{numeric:.0f}%"
