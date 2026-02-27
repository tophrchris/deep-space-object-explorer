from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import numpy as np
import pandas as pd

from runtime.weather_service import (
    format_precipitation,
    format_snowfall,
    format_temperature,
    format_wind_speed,
)


class ConditionTip(TypedDict):
    id: str
    text: str
    display: str


@dataclass(frozen=True)
class ConditionTipsContext:
    period_label: str
    forecast_date_text: str
    hourly_weather_rows: list[dict[str, Any]]
    summary_row: dict[str, Any] | None
    temperature_unit: str
    use_12_hour: bool
    eclipse_visibility: dict[str, Any] | None = None


ConditionTipRule = Callable[[ConditionTipsContext], ConditionTip | None]


def _make_tip(tip_id: str, text: str, display: str = "bullet") -> ConditionTip:
    return {
        "id": str(tip_id).strip(),
        "text": str(text).strip(),
        "display": str(display or "bullet").strip().lower() or "bullet",
    }


def _nonnegative_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return max(0.0, numeric)


def _finite_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def _positive_float(value: Any) -> float | None:
    numeric = _nonnegative_float(value)
    if numeric is None or numeric <= 0.0:
        return None
    return numeric


def _collect_values(rows: list[dict[str, Any]], field: str, *, nonnegative: bool = True) -> list[float]:
    values: list[float] = []
    for row in rows:
        numeric = _nonnegative_float(row.get(field)) if nonnegative else _finite_float(row.get(field))
        if numeric is not None:
            values.append(float(numeric))
    return values


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def _sorted_hourly_rows(rows: list[dict[str, Any]]) -> list[tuple[pd.Timestamp, dict[str, Any]]]:
    ordered: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for row in rows:
        timestamp = _parse_timestamp(row.get("time_iso"))
        if timestamp is None:
            continue
        ordered.append((timestamp, row))
    ordered.sort(key=lambda item: item[0])
    return ordered


def _format_tip_time(timestamp: pd.Timestamp, *, use_12_hour: bool) -> str:
    if use_12_hour:
        label = timestamp.strftime("%I:%M %p")
        return label[1:] if label.startswith("0") else label
    return timestamp.strftime("%H:%M")


def _format_duration(duration: pd.Timedelta) -> str:
    total_minutes = max(0, int(round(duration.total_seconds() / 60.0)))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def _format_minutes_compact(total_minutes: float) -> str:
    minutes = max(0, int(round(float(total_minutes))))
    hours, rem_minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {rem_minutes:02d}m" if rem_minutes > 0 else f"{hours}h"
    return f"{rem_minutes}m"


def _sky_region_from_wind16(direction: str) -> str | None:
    key = str(direction or "").strip().upper()
    if key in {"N", "NNE", "NNW"}:
        return "northern sky"
    if key in {"NE", "ENE"}:
        return "northeastern sky"
    if key in {"E", "ESE"}:
        return "eastern sky"
    if key in {"SE", "SSE"}:
        return "southeastern sky"
    if key in {"S", "SSW"}:
        return "southern sky"
    if key in {"SW", "WSW"}:
        return "southwestern sky"
    if key in {"W", "WNW"}:
        return "western sky"
    if key in {"NW"}:
        return "northwestern sky"
    return None


def _temperature_delta_for_display(delta_c: float, temperature_unit: str) -> float:
    if str(temperature_unit).strip().lower() == "f":
        return float(delta_c) * 9.0 / 5.0
    return float(delta_c)


def _temperature_unit_symbol(temperature_unit: str) -> str:
    return "F" if str(temperature_unit).strip().lower() == "f" else "C"


def _tip_clear_window(context: ConditionTipsContext) -> ConditionTip | None:
    ordered_rows = _sorted_hourly_rows(context.hourly_weather_rows)
    if not ordered_rows:
        return _make_tip("clear_window", "Clear-sky window is unavailable.", display="muted")

    longest_start: pd.Timestamp | None = None
    longest_end: pd.Timestamp | None = None
    longest_duration = pd.Timedelta(0)

    current_start: pd.Timestamp | None = None
    current_end: pd.Timestamp | None = None

    def flush_current_run() -> None:
        nonlocal longest_start, longest_end, longest_duration, current_start, current_end
        if current_start is None or current_end is None:
            return
        duration = (current_end - current_start) + pd.Timedelta(hours=1)
        if duration > longest_duration:
            longest_duration = duration
            longest_start = current_start
            longest_end = current_end
        current_start = None
        current_end = None

    for timestamp, row in ordered_rows:
        cloud_cover = _nonnegative_float(row.get("cloud_cover"))
        is_clear = cloud_cover is not None and cloud_cover < 20.0
        if not is_clear:
            flush_current_run()
            continue

        if current_start is None or current_end is None:
            current_start = timestamp
            current_end = timestamp
            continue

        gap = timestamp - current_end
        if gap <= pd.Timedelta(hours=1):
            current_end = timestamp
            continue

        flush_current_run()
        current_start = timestamp
        current_end = timestamp

    flush_current_run()

    if longest_start is None or longest_end is None or longest_duration <= pd.Timedelta(0):
        return _make_tip("clear_window", "No clear-sky window under 20% cloud cover is expected.")

    window_start = _format_tip_time(longest_start, use_12_hour=context.use_12_hour)
    window_end = _format_tip_time(longest_end + pd.Timedelta(hours=1), use_12_hour=context.use_12_hour)
    duration_text = _format_duration(longest_duration)
    display = "opportunity" if longest_duration >= pd.Timedelta(hours=5) else "bullet"
    return _make_tip(
        "clear_window",
        f"Longest clear-sky window (<20% clouds): {window_start}-{window_end} ({duration_text}).",
        display,
    )


def _tip_sky_cover(context: ConditionTipsContext) -> ConditionTip | None:
    cloud_values = _collect_values(context.hourly_weather_rows, "cloud_cover")
    if not cloud_values:
        return _make_tip("sky_cover", "Sky cover outlook unavailable.", display="muted")

    avg_cloud = float(np.mean(cloud_values))
    if avg_cloud <= 25.0:
        return _make_tip("sky_cover", f"Skies look mostly clear ({avg_cloud:.0f}% average cloud cover).", "muted")
    if avg_cloud <= 50.0:
        return _make_tip("sky_cover", f"Skies are partly cloudy on average ({avg_cloud:.0f}%).", "muted")
    if avg_cloud <= 75.0:
        return _make_tip("sky_cover", f"Cloud cover is elevated ({avg_cloud:.0f}% average).", "muted")
    return _make_tip("sky_cover", f"Cloud cover is likely heavy ({avg_cloud:.0f}% average).", "muted")


def _tip_precipitation(context: ConditionTipsContext) -> ConditionTip | None:
    rows = context.hourly_weather_rows
    if not rows:
        return _make_tip("precip", "Precipitation outlook unavailable.", display="muted")

    precip_probabilities = _collect_values(rows, "precipitation_probability")
    rain_values = _collect_values(rows, "rain")
    shower_values = _collect_values(rows, "showers")
    snow_values = _collect_values(rows, "snowfall")

    max_prob = max(precip_probabilities) if precip_probabilities else 0.0
    total_rain_mm = float(np.sum(rain_values)) if rain_values else 0.0
    total_showers_mm = float(np.sum(shower_values)) if shower_values else 0.0
    total_snow_cm = float(np.sum(snow_values)) if snow_values else 0.0

    if total_snow_cm > 0.0:
        return _make_tip(
            "precip",
            (
                f"❄️ Snow is possible ({format_snowfall(total_snow_cm, context.temperature_unit)} total, "
                f"{max_prob:.0f}% peak probability)."
            ),
            "warning",
        )

    total_liquid_mm = total_rain_mm + total_showers_mm
    if total_liquid_mm > 0.0:
        liquid_warning_threshold_mm = (
            (0.1 / 0.0393701) if str(context.temperature_unit).strip().lower() == "f" else 0.1
        )
        if total_rain_mm > 0.0:
            precip_prefix = "⛈️ Rain/showers are possible"
        else:
            precip_prefix = "☔ Showers are possible"
        if total_liquid_mm > liquid_warning_threshold_mm:
            return _make_tip(
                "precip",
                (
                    f"{precip_prefix} ({format_precipitation(total_liquid_mm, context.temperature_unit)} total, "
                    f"{max_prob:.0f}% peak probability)."
                ),
                "warning",
            )
        return _make_tip(
            "precip",
            (
                f"Trace rain/showers possible ({format_precipitation(total_liquid_mm, context.temperature_unit)} total, "
                f"{max_prob:.0f}% peak probability)."
            ),
            "muted",
        )

    if max_prob >= 40.0:
        return _make_tip("precip", f"Precipitation probability peaks near {max_prob:.0f}%.", "muted")
    if max_prob >= 20.0:
        return _make_tip("precip", f"Low-end precipitation risk (up to {max_prob:.0f}%).", "muted")
    return _make_tip("precip", "Precipitation risk looks low.", "muted")


def _tip_wind(context: ConditionTipsContext) -> ConditionTip | None:
    gust_values = _collect_values(context.hourly_weather_rows, "wind_gusts_10m")
    if not gust_values:
        return _make_tip("wind", "Wind outlook unavailable.", display="muted")

    peak_gust_kmh = max(gust_values)
    peak_gust_text = format_wind_speed(peak_gust_kmh, context.temperature_unit)
    if peak_gust_kmh >= 45.0:
        return _make_tip(
            "wind",
            f"Wind gusts may be strong (up to {peak_gust_text}); **consider shorter sub-exposure times**.",
        )
    if peak_gust_kmh >= 30.0:
        return _make_tip("wind", f"Moderate gusts expected (up to {peak_gust_text}).", "muted")
    return _make_tip("wind", f"Winds look manageable (gusts up to {peak_gust_text}).", "muted")


def _tip_dewpoint(context: ConditionTipsContext) -> ConditionTip | None:
    rows = context.hourly_weather_rows
    spreads_c: list[float] = []
    for row in rows:
        temp_c = _finite_float(row.get("temperature_2m"))
        dew_c = _finite_float(row.get("dew_point_2m"))
        if temp_c is None or dew_c is None:
            continue
        spreads_c.append(abs(float(temp_c - dew_c)))

    if not spreads_c:
        return _make_tip("dewpoint", "Dewpoint spread unavailable.", display="muted")

    min_spread_c = min(spreads_c)
    min_spread_f = _temperature_delta_for_display(min_spread_c, "f")
    min_spread_display = _temperature_delta_for_display(min_spread_c, context.temperature_unit)
    unit_symbol = _temperature_unit_symbol(context.temperature_unit)
    if min_spread_f <= 2.0:
        return _make_tip(
            "dewpoint",
            (
                f"Dewpoint spread may narrow to {min_spread_display:.1f} {unit_symbol}; "
                "**high risk of dew accumulation, use Heater and Dew Shield to avoid fogging.**"
            ),
        )
    if min_spread_f <= 5.0:
        return _make_tip(
            "dewpoint",
            (
                f"Dewpoint spread may narrow to {min_spread_display:.1f} {unit_symbol}; "
                "**dew likely, consider dew shield and heater, battery permitting.**"
            ),
        )
    return _make_tip(
        "dewpoint",
        (
            f"Dewpoint spread bottoms near {min_spread_display:.1f} {unit_symbol}; "
            "**low risk of dew, heater/shield optional.**"
        ),
    )


def _tip_humidity(context: ConditionTipsContext) -> ConditionTip | None:
    humidity_values = _collect_values(context.hourly_weather_rows, "relative_humidity_2m")
    if not humidity_values:
        return _make_tip("humidity", "Relative humidity outlook unavailable.", display="muted")

    peak_humidity = max(humidity_values)
    if peak_humidity > 85.0:
        return _make_tip(
            "humidity",
            (
                f"Relative humidity may reach {peak_humidity:.0f}%; **prefer bright emission targets**, "
                "**avoid faint galaxies**, and **expect stronger gradients**."
            ),
            "warning",
        )
    if peak_humidity >= 65.0:
        return _make_tip(
            "humidity",
            f"Relative humidity may reach {peak_humidity:.0f}%; **use the anti-dew heater**.",
        )
    return _make_tip("humidity", f"Relative humidity remains moderate (peaks near {peak_humidity:.0f}%).", "muted")


def _tip_temperature_drop(context: ConditionTipsContext) -> ConditionTip | None:
    ordered_rows = _sorted_hourly_rows(context.hourly_weather_rows)
    if len(ordered_rows) < 2:
        return _make_tip("temp_drop", "Cooling trend is unavailable.", display="muted")

    temperature_points: list[tuple[pd.Timestamp, float]] = []
    for timestamp, row in ordered_rows:
        temp_c = _finite_float(row.get("temperature_2m"))
        if temp_c is None:
            continue
        temperature_points.append((timestamp, temp_c))

    if len(temperature_points) < 2:
        return _make_tip("temp_drop", "Cooling trend is unavailable.", display="muted")

    steepest_rate_display: float | None = None
    steepest_rate_f_per_hr: float = 0.0
    steepest_drop_display: float | None = None
    steepest_start: pd.Timestamp | None = None
    steepest_end: pd.Timestamp | None = None

    running_max_c = temperature_points[0][1]
    max_drop_c = 0.0
    for _, temp_c in temperature_points[1:]:
        if temp_c > running_max_c:
            running_max_c = temp_c
            continue
        drawdown_c = running_max_c - temp_c
        if drawdown_c > max_drop_c:
            max_drop_c = drawdown_c

    for idx in range(len(temperature_points) - 1):
        start_time, temp_start_c = temperature_points[idx]
        end_time, temp_end_c = temperature_points[idx + 1]
        duration_hours = (end_time - start_time).total_seconds() / 3600.0
        if duration_hours <= 0.0:
            continue

        drop_c = temp_start_c - temp_end_c
        if drop_c <= 0.0:
            continue

        drop_display = _temperature_delta_for_display(drop_c, context.temperature_unit)
        rate_display = drop_display / duration_hours
        rate_f_per_hr = (_temperature_delta_for_display(drop_c, "f")) / duration_hours

        if steepest_start is None or rate_f_per_hr > steepest_rate_f_per_hr:
            steepest_rate_display = rate_display
            steepest_rate_f_per_hr = rate_f_per_hr
            steepest_drop_display = drop_display
            steepest_start = start_time
            steepest_end = end_time

    max_drop_display = _temperature_delta_for_display(max_drop_c, context.temperature_unit)
    max_drop_f = _temperature_delta_for_display(max_drop_c, "f")
    unit_symbol = _temperature_unit_symbol(context.temperature_unit)
    should_refocus = (max_drop_f >= 5.0) or (steepest_rate_f_per_hr > 2.0)

    if (
        steepest_rate_display is None
        or steepest_drop_display is None
        or steepest_start is None
        or steepest_end is None
    ):
        return _make_tip(
            "temp_drop",
            (
                f"Max overnight cooling drop: {max_drop_display:.1f} {unit_symbol}. "
                "refocusing due to temperature drop not required."
            ),
            "muted",
        )

    start_label = _format_tip_time(steepest_start, use_12_hour=context.use_12_hour)
    end_label = _format_tip_time(steepest_end, use_12_hour=context.use_12_hour)
    if should_refocus:
        advice = "**Re-focus during the session to avoid bloated stars.**"
        display = "bullet"
    else:
        advice = "refocusing due to temperature drop not required."
        display = "muted"
    return _make_tip(
        "temp_drop",
        (
            f"Steepest cooling: {steepest_rate_display:.1f} {unit_symbol}/hr "
            f"from {start_label}-{end_label} ({steepest_drop_display:.1f} {unit_symbol} interval drop; "
            f"{max_drop_display:.1f} {unit_symbol} max overnight drop). "
            f"{advice}"
        ),
        display,
    )


def _tip_temperature(context: ConditionTipsContext) -> ConditionTip | None:
    summary = context.summary_row or {}
    temp_range_text = str(summary.get("Low-Hi") or summary.get("Temp Range") or "").strip()
    if temp_range_text and temp_range_text != "-":
        return _make_tip("temp", f"Temperature range: {temp_range_text}.", "muted")

    temperature_values = _collect_values(context.hourly_weather_rows, "temperature_2m", nonnegative=False)
    if not temperature_values:
        return None

    temp_unit = str(context.temperature_unit).strip().lower()
    if temp_unit == "f":
        low = (min(temperature_values) * 9.0 / 5.0) + 32.0
        high = (max(temperature_values) * 9.0 / 5.0) + 32.0
        unit = "F"
    else:
        low = min(temperature_values)
        high = max(temperature_values)
        unit = "C"
    return _make_tip("temp", f"Temperature range: {low:.0f}-{high:.0f} {unit}.", "muted")


def _tip_eclipse_opportunity(context: ConditionTipsContext) -> ConditionTip | None:
    payload = context.eclipse_visibility if isinstance(context.eclipse_visibility, dict) else {}
    if not bool(payload.get("has_visible_eclipse", False)):
        return None

    raw_events = payload.get("events", [])
    if not isinstance(raw_events, list):
        return None

    event_descriptions: list[str] = []
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        if not bool(event.get("has_visible_phase", False)):
            continue
        raw_phases = event.get("phases", [])
        if not isinstance(raw_phases, list):
            continue

        phase_bits: list[str] = []
        for raw_phase in raw_phases:
            if not isinstance(raw_phase, dict):
                continue
            phase_name = str(raw_phase.get("name", "")).strip().lower()
            if phase_name not in {"partial", "total"}:
                continue

            phase_start = _parse_timestamp(raw_phase.get("start_local_iso"))
            phase_end = _parse_timestamp(raw_phase.get("end_local_iso"))
            if phase_start is None or phase_end is None or phase_end <= phase_start:
                continue

            duration_minutes = _positive_float(raw_phase.get("duration_minutes"))
            if duration_minutes is None:
                duration_minutes = (phase_end - phase_start).total_seconds() / 60.0
            if duration_minutes is None or duration_minutes <= 0.0:
                continue

            start_text = _format_tip_time(phase_start, use_12_hour=context.use_12_hour)
            end_text = _format_tip_time(phase_end, use_12_hour=context.use_12_hour)
            duration_text = _format_minutes_compact(duration_minutes)
            azimuth_dir = str(raw_phase.get("azimuth_dir", "")).strip().upper()
            sky_region = _sky_region_from_wind16(azimuth_dir)
            phase_label = "Totality" if phase_name == "total" else "Partial phase"
            if sky_region:
                phase_bits.append(
                    f"{phase_label} runs {start_text}-{end_text} ({duration_text}) in the {sky_region}"
                )
            else:
                phase_bits.append(
                    f"{phase_label} runs {start_text}-{end_text} ({duration_text})"
                )

        if phase_bits:
            event_descriptions.append("; ".join(phase_bits))

    if not event_descriptions:
        return None

    if len(event_descriptions) == 1:
        return _make_tip("eclipse_opportunity", f"Lunar eclipse: {event_descriptions[0]}.", "opportunity")
    return _make_tip(
        "eclipse_opportunity",
        "Lunar eclipse phases: " + " | ".join(event_descriptions) + ".",
        "opportunity",
    )


def _tip_eclipse_obstruction_warning(context: ConditionTipsContext) -> ConditionTip | None:
    payload = context.eclipse_visibility if isinstance(context.eclipse_visibility, dict) else {}
    if not bool(payload.get("has_visible_eclipse", False)):
        return None
    if not bool(payload.get("has_obstructed_eclipse_phase", False)):
        return None

    raw_events = payload.get("events", [])
    obstructed_phase_labels: list[str] = []
    if isinstance(raw_events, list):
        for event in raw_events:
            if not isinstance(event, dict):
                continue
            if not bool(event.get("has_visible_phase", False)):
                continue
            raw_phases = event.get("phases", [])
            if not isinstance(raw_phases, list):
                continue
            for raw_phase in raw_phases:
                if not isinstance(raw_phase, dict):
                    continue
                if not bool(raw_phase.get("obstructed", False)):
                    continue
                phase_name = str(raw_phase.get("name", "")).strip().lower()
                phase_start = _parse_timestamp(raw_phase.get("start_local_iso"))
                phase_end = _parse_timestamp(raw_phase.get("end_local_iso"))
                if phase_start is not None and phase_end is not None and phase_end > phase_start:
                    start_text = _format_tip_time(phase_start, use_12_hour=context.use_12_hour)
                    end_text = _format_tip_time(phase_end, use_12_hour=context.use_12_hour)
                    label = "Totality" if phase_name == "total" else "Partial phase"
                    obstructed_phase_labels.append(f"{label} ({start_text}-{end_text})")
                elif phase_name == "total":
                    obstructed_phase_labels.append("Totality")
                elif phase_name == "partial":
                    obstructed_phase_labels.append("Partial phase")

    if obstructed_phase_labels:
        details = ", ".join(obstructed_phase_labels[:3])
        return _make_tip(
            "eclipse_obstructed",
            f"Eclipse is partly obstructed at this site (below local obstruction profile/horizon during {details}).",
            "warning",
        )
    return _make_tip(
        "eclipse_obstructed",
        "Eclipse is partly obstructed at this site (below local obstruction profile/horizon during one or more phases).",
        "warning",
    )


_CONDITION_RULES: tuple[ConditionTipRule, ...] = (
    _tip_eclipse_opportunity,
    _tip_eclipse_obstruction_warning,
    _tip_sky_cover,
    _tip_clear_window,
    _tip_precipitation,
    _tip_wind,
    _tip_dewpoint,
    _tip_humidity,
    _tip_temperature_drop,
    _tip_temperature,
)


def collect_condition_tips(context: ConditionTipsContext) -> list[ConditionTip]:
    if not context.hourly_weather_rows:
        return [_make_tip("weather_unavailable", "Hourly weather for this period is unavailable.", "muted")]

    tips: list[ConditionTip] = []
    for rule in _CONDITION_RULES:
        tip = rule(context)
        if tip is not None:
            tips.append(tip)
    return tips
