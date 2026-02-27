from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


BROADBAND_PENALTY_MAG = 0.5
NARROWBAND_BONUS_MAG = 1.5

STATUS_IN_RANGE = "in_range"
STATUS_BROADBAND_BORDERLINE = "broadband_borderline"
STATUS_NARROWBAND_BOOSTED = "narrowband_boosted"
STATUS_OUT_OF_RANGE = "out_of_range"
STATUS_UNKNOWN = "unknown"


@dataclass(frozen=True)
class MagnitudeThresholds:
    selected_max: float
    broadband_effective_max: float
    narrowband_effective_max: float


def _safe_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def resolve_telescope_max_magnitude(
    telescope: dict[str, Any] | None,
    *,
    fallback: float = 10.5,
) -> float:
    if not isinstance(telescope, dict):
        return float(fallback)
    for field_name in ("estimated_max_magnitude", "max_mag"):
        parsed = _safe_finite_float(telescope.get(field_name))
        if parsed is not None and parsed > 0.0:
            return float(parsed)
    return float(fallback)


def slider_upper_bound(telescope_max_magnitude: float) -> float:
    return max(2.0, float(telescope_max_magnitude) + 2.0)


def resolve_magnitude_thresholds(selected_max: float) -> MagnitudeThresholds:
    selected = max(0.0, float(selected_max))
    return MagnitudeThresholds(
        selected_max=selected,
        broadband_effective_max=max(0.0, selected - BROADBAND_PENALTY_MAG),
        narrowband_effective_max=selected + NARROWBAND_BONUS_MAG,
    )


def classify_target_magnitude(
    magnitude_value: Any,
    *,
    selected_max: float,
    narrowband_filter_active: bool,
    target_is_narrowband: bool,
) -> tuple[bool, str, float | None]:
    magnitude = _safe_finite_float(magnitude_value)
    if magnitude is None:
        return False, STATUS_UNKNOWN, None

    thresholds = resolve_magnitude_thresholds(selected_max)
    if not narrowband_filter_active:
        if magnitude <= thresholds.selected_max:
            return True, STATUS_IN_RANGE, magnitude
        return False, STATUS_OUT_OF_RANGE, magnitude

    if target_is_narrowband:
        if magnitude <= thresholds.selected_max:
            return True, STATUS_IN_RANGE, magnitude
        if magnitude <= thresholds.narrowband_effective_max:
            return True, STATUS_NARROWBAND_BOOSTED, magnitude
        return False, STATUS_OUT_OF_RANGE, magnitude

    if magnitude <= thresholds.broadband_effective_max:
        return True, STATUS_IN_RANGE, magnitude
    if magnitude <= thresholds.selected_max:
        return True, STATUS_BROADBAND_BORDERLINE, magnitude
    return False, STATUS_OUT_OF_RANGE, magnitude
