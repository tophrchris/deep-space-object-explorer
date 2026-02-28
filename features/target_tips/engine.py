from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, TypedDict

import numpy as np
import pandas as pd

from runtime.weather_service import (
    format_precipitation,
    format_snowfall,
    format_temperature,
    format_wind_speed,
)

_ALERT_SNOW = "\u2744\ufe0f"
_ALERT_RAIN = "\u26c8\ufe0f"
_ALERT_SHOWERS = "\u2614"
_ALERT_ALARM = "\U0001f6a8"
_ALERT_CAUTION = "\u26a0\ufe0f"


class TargetTip(TypedDict):
    id: str
    text: str
    display: str


@dataclass(frozen=True)
class TargetTipsContext:
    selected_id: str
    selected_label: str
    selected_target_data: pd.Series | dict[str, Any] | None
    selected_track: pd.DataFrame | None
    selected_row: dict[str, Any]
    nightly_weather_alert_emojis: list[str]
    hourly_weather_rows: list[dict[str, Any]]
    temperature_unit: str
    use_12_hour: bool
    local_now: datetime
    window_start: datetime
    window_end: datetime
    active_mount_choice: str


TargetTipRule = Callable[[TargetTipsContext], TargetTip | None]


def _normalize_12_hour_label(value: str) -> str:
    cleaned = str(value).strip()
    if cleaned.startswith("0"):
        cleaned = cleaned[1:]
    return cleaned.replace("AM", "am").replace("PM", "pm")


def _format_display_time(value: pd.Timestamp | datetime, use_12_hour: bool) -> str:
    timestamp = pd.Timestamp(value)
    rendered = timestamp.strftime("%I:%M %p" if use_12_hour else "%H:%M")
    return _normalize_12_hour_label(rendered) if use_12_hour else rendered


def _format_hour_bucket_label(hour_ts: pd.Timestamp, use_12_hour: bool) -> str:
    timestamp = pd.Timestamp(hour_ts)
    if use_12_hour:
        rendered = timestamp.strftime("%I%p")
        return rendered[1:] if rendered.startswith("0") else rendered
    return timestamp.strftime("%H")


def _hour_bins_from_mask(track: pd.DataFrame, mask: pd.Series) -> list[pd.Timestamp]:
    if "time_local" not in track.columns:
        return []

    selected_times = pd.to_datetime(track.loc[mask, "time_local"], errors="coerce").dropna()
    if selected_times.empty:
        return []

    unique_hours = selected_times.dt.floor("h").unique().tolist()
    hour_bins = sorted(pd.Timestamp(value) for value in unique_hours)
    return hour_bins


def describe_hour_collection(hour_bins: list[pd.Timestamp], use_12_hour: bool) -> str:
    if not hour_bins:
        return ""

    sorted_hours = sorted(pd.Timestamp(value) for value in hour_bins)
    if len(sorted_hours) == 1:
        return _format_hour_bucket_label(sorted_hours[0], use_12_hour=use_12_hour)

    is_continuous = all(
        (sorted_hours[idx + 1] - sorted_hours[idx]) == pd.Timedelta(hours=1)
        for idx in range(len(sorted_hours) - 1)
    )
    if is_continuous:
        start_label = _format_hour_bucket_label(sorted_hours[0], use_12_hour=use_12_hour)
        end_label = _format_hour_bucket_label(sorted_hours[-1], use_12_hour=use_12_hour)
        return f"{start_label}-{end_label}"

    return ", ".join(_format_hour_bucket_label(hour_ts, use_12_hour=use_12_hour) for hour_ts in sorted_hours)


def _positive_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


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


def build_culmination_weather_summary(
    culmination_time: Any,
    hourly_weather_rows: list[dict[str, Any]],
    *,
    temperature_unit: str,
    use_12_hour: bool,
) -> str:
    if culmination_time is None or pd.isna(culmination_time):
        return "Culmination weather: unavailable."

    try:
        culmination_ts = pd.Timestamp(culmination_time)
    except Exception:
        return "Culmination weather: unavailable."

    nearest_row: dict[str, Any] | None = None
    nearest_delta: pd.Timedelta | None = None
    for row in hourly_weather_rows:
        time_iso = str(row.get("time_iso", "")).strip()
        if not time_iso:
            continue
        try:
            row_ts = pd.Timestamp(time_iso)
            if culmination_ts.tzinfo is not None and row_ts.tzinfo is None:
                row_ts = row_ts.tz_localize(culmination_ts.tzinfo)
            elif culmination_ts.tzinfo is None and row_ts.tzinfo is not None:
                row_ts = row_ts.tz_localize(None)
            elif culmination_ts.tzinfo is not None and row_ts.tzinfo is not None:
                row_ts = row_ts.tz_convert(culmination_ts.tzinfo)
        except Exception:
            continue

        delta = abs(row_ts - culmination_ts)
        if nearest_delta is None or delta < nearest_delta:
            nearest_delta = delta
            nearest_row = row

    if nearest_row is None:
        return "Culmination weather: unavailable."

    time_text = _format_display_time(culmination_ts, use_12_hour=use_12_hour)
    conditions: list[str] = []

    cloud_cover = _nonnegative_float(nearest_row.get("cloud_cover"))
    if cloud_cover is not None:
        if cloud_cover >= 85.0:
            sky_text = "overcast"
        elif cloud_cover >= 65.0:
            sky_text = "mostly cloudy"
        elif cloud_cover >= 40.0:
            sky_text = "partly cloudy"
        elif cloud_cover >= 15.0:
            sky_text = "mostly clear"
        else:
            sky_text = "clear"
        conditions.append(f"{sky_text} ({cloud_cover:.0f}% cloud cover)")

    temperature_c = nearest_row.get("temperature_2m")
    if temperature_c is not None and not pd.isna(temperature_c):
        conditions.append(format_temperature(float(temperature_c), temperature_unit))

    wind_gust_kmh = nearest_row.get("wind_gusts_10m")
    if wind_gust_kmh is not None and not pd.isna(wind_gust_kmh):
        conditions.append(f"gusts {format_wind_speed(float(wind_gust_kmh), temperature_unit)}")

    precip_probability = _nonnegative_float(nearest_row.get("precipitation_probability"))
    if precip_probability is not None:
        conditions.append(f"{precip_probability:.0f}% precip chance")

    snowfall = _positive_float(nearest_row.get("snowfall"))
    rain = _positive_float(nearest_row.get("rain"))
    showers = _positive_float(nearest_row.get("showers"))
    if snowfall is not None:
        conditions.append(f"snow {format_snowfall(snowfall, temperature_unit)}")
    if rain is not None:
        conditions.append(f"rain {format_precipitation(rain, temperature_unit)}")
    if showers is not None:
        conditions.append(f"showers {format_precipitation(showers, temperature_unit)}")

    if not conditions:
        return f"Culmination weather ({time_text}): unavailable."
    return f"Culmination weather ({time_text}): {', '.join(conditions)}."


def build_emission_target_tip(target_data: pd.Series | dict[str, Any] | None) -> str:
    if target_data is None:
        return ""

    if isinstance(target_data, pd.Series):
        raw_map = target_data.to_dict()
    elif isinstance(target_data, dict):
        raw_map = dict(target_data)
    else:
        return ""

    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
            return ""
        if isinstance(value, (list, tuple, set)):
            items = [str(item).strip() for item in value if str(item).strip()]
            return "; ".join(items)
        text = str(value).strip()
        if text.lower() in {"nan", "none"}:
            return ""
        return text

    explicit_values: list[str] = []
    for explicit_key in ("emission_lines", "emissions"):
        if explicit_key in raw_map:
            explicit_text = _text(raw_map.get(explicit_key))
            if explicit_text:
                explicit_values.append(explicit_text)

    all_text_values = [_text(value) for value in raw_map.values()]
    normalized_text = " ".join(value.lower() for value in all_text_values if value)
    has_emission_word = bool(re.search(r"\bemissions?\b", normalized_text))
    has_explicit_emissions = bool(explicit_values)
    if not has_emission_word and not has_explicit_emissions:
        return ""

    if has_explicit_emissions:
        cleaned_tokens: list[str] = []
        for value in explicit_values:
            cleaned_value = re.sub(r"[\[\]]", "", value)
            parts = re.split(r"[;,|]+", cleaned_value)
            for part in parts:
                token = str(part).strip()
                if token and token not in cleaned_tokens:
                    cleaned_tokens.append(token)
        emissions_text = ", ".join(cleaned_tokens) if cleaned_tokens else ", ".join(explicit_values)
        return f"Emission target reminder: catalog emissions include {emissions_text}."
    return "Emission target reminder: this target is identified as emission-related."


def _format_target_tip_time(value: Any, use_12_hour: bool) -> str:
    if value is None or pd.isna(value):
        return "--"
    try:
        return _format_display_time(pd.Timestamp(value), use_12_hour=use_12_hour)
    except Exception:
        return "--"


def _make_target_tip(tip_id: str, text: str, display: str = "bullet") -> TargetTip:
    return {
        "id": str(tip_id).strip(),
        "text": str(text).strip(),
        "display": str(display).strip().lower() or "bullet",
    }


def _target_tip_best_window(context: TargetTipsContext) -> TargetTip | None:
    first_visible = context.selected_row.get("first_visible")
    last_visible = context.selected_row.get("last_visible")
    culmination = context.selected_row.get("culmination")

    if not pd.isna(first_visible) and not pd.isna(last_visible):
        return _make_target_tip(
            "best_window",
            f"Best window: {_format_target_tip_time(first_visible, context.use_12_hour)} "
            f"to {_format_target_tip_time(last_visible, context.use_12_hour)}",
        )
    if not pd.isna(culmination):
        return _make_target_tip(
            "best_window",
            f"Best around: {_format_target_tip_time(culmination, context.use_12_hour)}",
        )
    return None


def _target_tip_peak_view(context: TargetTipsContext) -> TargetTip | None:
    culmination = context.selected_row.get("culmination")
    if pd.isna(culmination):
        return None

    peak_altitude = str(context.selected_row.get("culmination_alt") or "--").strip() or "--"
    peak_direction = str(context.selected_row.get("culmination_dir") or "--").strip() or "--"
    return _make_target_tip(
        "peak_view",
        f"Peak view: {_format_target_tip_time(culmination, context.use_12_hour)} "
        f"at {peak_altitude} toward {peak_direction}",
    )


def _target_tip_culmination_weather(context: TargetTipsContext) -> TargetTip | None:
    culmination = context.selected_row.get("culmination")
    if pd.isna(culmination):
        return None
    return _make_target_tip(
        "culmination_weather",
        build_culmination_weather_summary(
            culmination,
            context.hourly_weather_rows,
            temperature_unit=context.temperature_unit,
            use_12_hour=context.use_12_hour,
        ),
    )


def _target_tip_emission_target(context: TargetTipsContext) -> TargetTip | None:
    emission_tip = build_emission_target_tip(context.selected_target_data)
    if not emission_tip:
        return None
    return _make_target_tip("emission_target", emission_tip, display="strong")


def _target_tip_visibility(context: TargetTipsContext) -> TargetTip | None:
    visible_total = str(context.selected_row.get("visible_total") or "--").strip() or "--"
    visible_remaining = str(context.selected_row.get("visible_remaining") or "--").strip() or "--"
    if visible_total == "--":
        return None

    if context.window_start <= context.local_now <= context.window_end and visible_remaining != "--":
        return _make_target_tip(
            "visibility",
            f"Visibility now: {visible_remaining} remaining tonight ({visible_total} total)",
        )
    return _make_target_tip("visibility", f"Total visible tonight: {visible_total}")


def _track_unobstructed_mask(track: pd.DataFrame, altitude: pd.Series) -> pd.Series:
    if "visible" in track.columns:
        return track["visible"].fillna(False).astype(bool)
    if "min_alt_required" in track.columns:
        required_altitude = pd.to_numeric(track["min_alt_required"], errors="coerce")
        return altitude >= required_altitude
    return pd.Series(True, index=track.index, dtype=bool)


def _target_tip_low_altitude_exposure(context: TargetTipsContext) -> TargetTip | None:
    if str(context.active_mount_choice or "").strip().lower() != "eq":
        return None
    track = context.selected_track
    if not isinstance(track, pd.DataFrame) or track.empty:
        return None
    if "time_local" not in track.columns or "alt" not in track.columns:
        return None

    altitude = pd.to_numeric(track["alt"], errors="coerce")
    unobstructed_mask = _track_unobstructed_mask(track, altitude)

    low_and_unobstructed = unobstructed_mask & altitude.notna() & (altitude < 30.0)
    if not bool(low_and_unobstructed.any()):
        return None

    hour_bins = _hour_bins_from_mask(track, low_and_unobstructed)
    if not hour_bins:
        return None

    hour_description = describe_hour_collection(hour_bins, use_12_hour=context.use_12_hour)
    return _make_target_tip(
        "low_altitude_exposure",
        (
            f"Tracking tip: unobstructed time below 30 deg occurs during {hour_description}; "
            "if using an EQ mount, use shorter exposures to avoid star trails."
        ),
    )


def _target_tip_high_altitude_altaz_exposure(context: TargetTipsContext) -> TargetTip | None:
    if str(context.active_mount_choice or "").strip().lower() != "altaz":
        return None
    track = context.selected_track
    if not isinstance(track, pd.DataFrame) or track.empty:
        return None
    if "time_local" not in track.columns or "alt" not in track.columns:
        return None

    altitude = pd.to_numeric(track["alt"], errors="coerce")
    unobstructed_mask = _track_unobstructed_mask(track, altitude)

    high_and_unobstructed = unobstructed_mask & altitude.notna() & (altitude > 75.0)
    if not bool(high_and_unobstructed.any()):
        return None

    hour_bins = _hour_bins_from_mask(track, high_and_unobstructed)
    if not hour_bins:
        return None

    hour_description = describe_hour_collection(hour_bins, use_12_hour=context.use_12_hour)
    return _make_target_tip(
        "high_altitude_altaz_exposure",
        (
            f"Tracking tip: unobstructed time above 75 deg occurs during {hour_description}; "
            "if using an Alt/Az mount, use shorter exposures to avoid star trails."
        ),
    )


def _target_tip_weather_alert(context: TargetTipsContext) -> TargetTip | None:
    weather_tip_by_emoji: dict[str, tuple[str, str]] = {
        _ALERT_SNOW: ("Weather alert: Snow expected overnight.", "strong"),
        _ALERT_RAIN: ("Weather alert: Rain expected overnight.", "strong"),
        _ALERT_SHOWERS: ("Weather alert: Showers possible overnight.", "bullet"),
        _ALERT_ALARM: ("Weather alert: Precipitation probability is elevated (>20%).", "strong"),
        _ALERT_CAUTION: ("Weather alert: Low precipitation probability (1-20%).", "bullet"),
    }
    if context.nightly_weather_alert_emojis:
        top_alert = str(context.nightly_weather_alert_emojis[0])
        payload = weather_tip_by_emoji.get(top_alert)
        if payload:
            text, display = payload
            return _make_target_tip("weather_alert", text, display=display)
        return None
    return _make_target_tip(
        "weather_alert",
        "Weather alert: No precipitation signals for tonight.",
        display="muted",
    )


# Tip registry: add new rule functions here to extend Target Tips behavior.
TARGET_TIP_RULES: tuple[TargetTipRule, ...] = (
    _target_tip_best_window,
    _target_tip_peak_view,
    _target_tip_culmination_weather,
    _target_tip_emission_target,
    _target_tip_visibility,
    _target_tip_low_altitude_exposure,
    _target_tip_high_altitude_altaz_exposure,
    _target_tip_weather_alert,
)


def collect_target_tips(context: TargetTipsContext) -> list[TargetTip]:
    tips: list[TargetTip] = []
    seen_ids: set[str] = set()
    for rule in TARGET_TIP_RULES:
        try:
            tip = rule(context)
        except Exception:
            continue
        if tip is None:
            continue

        tip_id = str(tip.get("id", "")).strip() or f"tip_{len(tips) + 1}"
        tip_text = str(tip.get("text", "")).strip()
        tip_display = str(tip.get("display", "bullet")).strip().lower() or "bullet"
        if not tip_text or tip_id in seen_ids:
            continue

        seen_ids.add(tip_id)
        tips.append({"id": tip_id, "text": tip_text, "display": tip_display})
    return tips
