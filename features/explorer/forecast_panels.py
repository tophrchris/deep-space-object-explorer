from __future__ import annotations

import html
import re
from datetime import datetime
from functools import lru_cache
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
import streamlit as st
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from app_constants import WIND16
from app_theme import apply_dataframe_styler_theme
from features.explorer.lunar import (
    format_lunar_altitude_display,
    format_lunar_percent_display,
    lunar_altitude_row_label,
    lunar_phase_emoji,
)
from features.explorer.night_rating import compute_night_rating_details
from runtime.lunar_ephemeris import (
    build_hourly_lunar_altitude_map,
    compute_lunar_night_summary,
    compute_lunar_phase_for_night,
)
from runtime.weather_service import (
    EXTENDED_FORECAST_HOURLY_FIELDS,
    fetch_hourly_weather,
    format_precipitation,
    format_snowfall,
    format_temperature,
    format_wind_speed,
)


@lru_cache(maxsize=None)
def _ui_name(name: str) -> Any:
    # Lazily resolve remaining UI-owned helpers/constants to avoid circular import
    # failures while the Streamlit monolith is still being decomposed.
    from ui import streamlit_app as ui_app

    return getattr(ui_app, name)


def _ideal_text_color_for_hex(*args: Any, **kwargs: Any):
    return _ui_name("_ideal_text_color_for_hex")(*args, **kwargs)


def _interpolate_cloud_cover_color(*args: Any, **kwargs: Any):
    return _ui_name("_interpolate_cloud_cover_color")(*args, **kwargs)


def _interpolate_color_stops(*args: Any, **kwargs: Any):
    return _ui_name("_interpolate_color_stops")(*args, **kwargs)


def _interpolate_temperature_color_f(*args: Any, **kwargs: Any):
    return _ui_name("_interpolate_temperature_color_f")(*args, **kwargs)


def apparent_size_sort_key_arcmin(*args: Any, **kwargs: Any):
    return _ui_name("apparent_size_sort_key_arcmin")(*args, **kwargs)


def astronomical_night_window(*args: Any, **kwargs: Any):
    return _ui_name("astronomical_night_window")(*args, **kwargs)


def format_apparent_size_display(*args: Any, **kwargs: Any):
    return _ui_name("format_apparent_size_display")(*args, **kwargs)


def format_description_preview(*args: Any, **kwargs: Any):
    return _ui_name("format_description_preview")(*args, **kwargs)


def format_display_time(*args: Any, **kwargs: Any):
    return _ui_name("format_display_time")(*args, **kwargs)


def format_emissions_display(*args: Any, **kwargs: Any):
    return _ui_name("format_emissions_display")(*args, **kwargs)


def format_hour_label(*args: Any, **kwargs: Any):
    return _ui_name("format_hour_label")(*args, **kwargs)


def format_weather_forecast_date(*args: Any, **kwargs: Any):
    return _ui_name("format_weather_forecast_date")(*args, **kwargs)


def normalize_object_type_group(*args: Any, **kwargs: Any):
    return _ui_name("normalize_object_type_group")(*args, **kwargs)


def normalize_weather_forecast_day_offset(*args: Any, **kwargs: Any):
    return _ui_name("normalize_weather_forecast_day_offset")(*args, **kwargs)


def object_type_group_color(*args: Any, **kwargs: Any):
    return _ui_name("object_type_group_color")(*args, **kwargs)


def weather_forecast_window(*args: Any, **kwargs: Any):
    return _ui_name("weather_forecast_window")(*args, **kwargs)


def _get_st_mui_table():
    return _ui_name("st_mui_table")

def _hex_to_rgb_local(value: str) -> tuple[int, int, int] | None:
    text = str(value or "").strip()
    if not re.fullmatch(r"#?[0-9a-fA-F]{6}", text):
        return None
    hex_text = text[1:] if text.startswith("#") else text
    try:
        return (int(hex_text[0:2], 16), int(hex_text[2:4], 16), int(hex_text[4:6], 16))
    except ValueError:
        return None


def _rgb_to_hex_local(red: int, green: int, blue: int) -> str:
    return f"#{max(0, min(255, int(red))):02X}{max(0, min(255, int(green))):02X}{max(0, min(255, int(blue))):02X}"


def _neutral_gray_from_hex(color_hex: str) -> str:
    rgb = _hex_to_rgb_local(color_hex)
    if rgb is None:
        return "#2A2A2A"
    red, green, blue = rgb
    luminance = int(round(((red * 299) + (green * 587) + (blue * 114)) / 1000.0))
    return _rgb_to_hex_local(luminance, luminance, luminance)


def _lunar_phase_style_bucket_from_label(row_label: Any) -> str:
    label = str(row_label or "").strip()
    if "🌑" in label:
        return "new"
    if "🌕" in label:
        return "full"
    if "🌓" in label or "🌗" in label:
        return "quarter"
    return "quarter"


def _lunar_phase_style_bucket_from_phase_key(phase_key: Any) -> str:
    key = str(phase_key or "").strip().lower()
    if key == "new":
        return "new"
    if key == "full":
        return "full"
    if key in {"first_quarter", "third_quarter"}:
        return "quarter"
    return "quarter"


def _lunar_gray_scale_palette() -> dict[str, str]:
    # Match cloud-cover brightness endpoints, but force grayscale for lunar altitude.
    low_gray = _neutral_gray_from_hex(_interpolate_cloud_cover_color(0.0))
    new_moon_high_gray = _neutral_gray_from_hex(_interpolate_cloud_cover_color(100.0))
    # Full moon is brighter than new moon, but remains a light gray (not pure white).
    full_moon_high_gray = _interpolate_color_stops(
        60.0,
        [
            (0.0, new_moon_high_gray),
            (100.0, "#FFFFFF"),
        ],
    )
    # First/third quarter use a midpoint between the low-end gray and full-moon gray.
    quarter_moon_high_gray = _interpolate_color_stops(
        50.0,
        [
            (0.0, low_gray),
            (100.0, full_moon_high_gray),
        ],
    )
    return {
        "low": low_gray,
        "new": new_moon_high_gray,
        "quarter": quarter_moon_high_gray,
        "full": full_moon_high_gray,
    }


def _lunar_gradient_style_for_altitude(
    altitude_deg: Any,
    *,
    phase_bucket: str,
) -> str:
    palette = _lunar_gray_scale_palette()
    low_gray = palette["low"]
    if phase_bucket == "new":
        high_gray = palette["new"]
    elif phase_bucket == "full":
        high_gray = palette["full"]
    else:
        high_gray = palette["quarter"]

    try:
        numeric_altitude = float(altitude_deg)
    except (TypeError, ValueError):
        numeric_altitude = 0.0
    if not np.isfinite(numeric_altitude):
        numeric_altitude = 0.0

    clamped_altitude = max(0.0, min(90.0, numeric_altitude))
    background_color = _interpolate_color_stops(
        clamped_altitude,
        [
            (0.0, low_gray),
            (90.0, high_gray),
        ],
    )
    text_color = _ideal_text_color_for_hex(background_color)
    return f"background-color: {background_color}; color: {text_color};"


def lunar_altitude_cell_style(raw_value: Any, row_label: Any) -> str:
    text = str(raw_value).strip() if raw_value is not None and not pd.isna(raw_value) else ""
    if not text or text == "-":
        return _lunar_gradient_style_for_altitude(0.0, phase_bucket=_lunar_phase_style_bucket_from_label(row_label))

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return _lunar_gradient_style_for_altitude(0.0, phase_bucket=_lunar_phase_style_bucket_from_label(row_label))

    try:
        parsed_altitude = float(match.group(0))
    except ValueError:
        parsed_altitude = 0.0

    return _lunar_gradient_style_for_altitude(
        parsed_altitude,
        phase_bucket=_lunar_phase_style_bucket_from_label(row_label),
    )


def lunar_forecast_column_cell_style(
    raw_value: Any,
    *,
    lunar_phase_key: Any,
    lunar_max_altitude_deg: Any,
) -> str:
    text = str(raw_value).strip() if raw_value is not None and not pd.isna(raw_value) else ""
    if not text:
        return ""
    return _lunar_gradient_style_for_altitude(
        lunar_max_altitude_deg,
        phase_bucket=_lunar_phase_style_bucket_from_phase_key(lunar_phase_key),
    )


def full_moon_scale_legend_html() -> str:
    palette = _lunar_gray_scale_palette()
    low_gray = palette["low"]
    quarter_gray = palette["quarter"]
    full_gray = palette["full"]

    # 12-box legend: darkest -> quarter marker -> lightest.
    swatch_colors: list[str] = []
    total_boxes = 12
    quarter_index = 5
    for idx in range(total_boxes):
        if idx <= quarter_index:
            color = _interpolate_color_stops(
                float(idx),
                [
                    (0.0, low_gray),
                    (float(quarter_index), quarter_gray),
                ],
            )
        else:
            color = _interpolate_color_stops(
                float(idx),
                [
                    (float(quarter_index), quarter_gray),
                    (float(total_boxes - 1), full_gray),
                ],
            )
        swatch_colors.append(color)

    def _swatch(color: str, *, margin_right: str = "0.12rem") -> str:
        return (
            "<span style='display:inline-block; width:0.58rem; height:0.58rem; border-radius:2px; "
            f"margin-right:{margin_right}; border:1px solid rgba(17,24,39,0.28); background:{color};'></span>"
        )

    left_swatches = "".join(
        _swatch(color, margin_right=("0.12rem" if idx < (quarter_index) else "0"))
        for idx, color in enumerate(swatch_colors[: quarter_index + 1])
    )
    right_swatches = "".join(
        _swatch(color, margin_right=("0.12rem" if idx < (len(swatch_colors[quarter_index + 1 :]) - 1) else "0"))
        for idx, color in enumerate(swatch_colors[quarter_index + 1 :])
    )

    return (
        "<div style='font-size:0.88rem; color:#6b7280; margin:0.02rem 0 0.12rem;'>"
        "🌙 lunar phase brightness scale: "
        "<span style='margin-left:0.04rem; margin-right:0.14rem;'>🌑</span>"
        f"{left_swatches}"
        "<span style='margin-left:0.14rem; margin-right:0.14rem;'>🌓</span>"
        f"{right_swatches}"
        "<span style='margin-left:0.14rem;'>🌕</span>"
        "</div>"
    )


def site_conditions_legends_table_html() -> str:
    def _swatch(color: str, *, margin_right: str = "0.12rem") -> str:
        return (
            "<span style='display:inline-block; width:0.58rem; height:0.58rem; border-radius:2px; "
            f"margin-right:{margin_right}; border:1px solid rgba(17,24,39,0.28); background:{color};'></span>"
        )

    # Night-time clarity key (Good -> Bad).
    clarity_swatches: list[str] = []
    clarity_boxes = 10
    for idx in range(clarity_boxes):
        clear_percent = 100.0 - (float(idx) * (100.0 / float(max(1, clarity_boxes - 1))))
        cloud_equivalent = max(0.0, min(100.0, 100.0 - clear_percent))
        color = _interpolate_cloud_cover_color(cloud_equivalent)
        clarity_swatches.append(_swatch(color))
    clarity_key_html = (
        "<div style='display:inline-flex; align-items:center; justify-content:center; flex-wrap:wrap;'>"
        "<span style='margin-right:0.2rem;'>Good</span>"
        f"{''.join(clarity_swatches)}"
        "<span style='margin-left:0.12rem;'>Bad</span>"
        "</div>"
    )

    # Dew risk key (Dry -> Dew).
    dew_swatches: list[str] = []
    spread_values = list(range(6, -1, -1))
    for idx, spread_value in enumerate(spread_values):
        color = _interpolate_color_stops(
            float(spread_value),
            [
                (0.0, "#FDBA74"),
                (6.0, "#FFFFFF"),
            ],
        )
        margin_right = "0.12rem" if idx < (len(spread_values) - 1) else "0"
        dew_swatches.append(_swatch(color, margin_right=margin_right))
    dew_key_html = (
        "<div style='display:inline-flex; align-items:center; justify-content:center; flex-wrap:wrap;'>"
        "<span style='margin-right:0.2rem;'>Dry</span>"
        f"{''.join(dew_swatches)}"
        "<span style='margin-left:0.12rem;'>Dew</span>"
        "</div>"
    )

    # Lunar brightness key (phase-aware grayscale reference).
    palette = _lunar_gray_scale_palette()
    low_gray = palette["low"]
    quarter_gray = palette["quarter"]
    full_gray = palette["full"]
    lunar_colors: list[str] = []
    lunar_boxes = 12
    quarter_index = 5
    for idx in range(lunar_boxes):
        if idx <= quarter_index:
            color = _interpolate_color_stops(
                float(idx),
                [
                    (0.0, low_gray),
                    (float(quarter_index), quarter_gray),
                ],
            )
        else:
            color = _interpolate_color_stops(
                float(idx),
                [
                    (float(quarter_index), quarter_gray),
                    (float(lunar_boxes - 1), full_gray),
                ],
            )
        lunar_colors.append(color)
    lunar_left = "".join(
        _swatch(color, margin_right=("0.12rem" if idx < quarter_index else "0"))
        for idx, color in enumerate(lunar_colors[: quarter_index + 1])
    )
    lunar_right_colors = lunar_colors[quarter_index + 1 :]
    lunar_right = "".join(
        _swatch(color, margin_right=("0.12rem" if idx < (len(lunar_right_colors) - 1) else "0"))
        for idx, color in enumerate(lunar_right_colors)
    )
    lunar_key_html = (
        "<div style='display:inline-flex; align-items:center; justify-content:center; flex-wrap:wrap;'>"
        "<span style='margin-right:0.14rem;'>🌑</span>"
        f"{lunar_left}"
        "<span style='margin-left:0.14rem; margin-right:0.14rem;'>🌓</span>"
        f"{lunar_right}"
        "<span style='margin-left:0.14rem;'>🌕</span>"
        "</div>"
    )

    weather_alert_key_html = (
        "<div style='display:inline-flex; align-items:center; justify-content:center; flex-wrap:wrap; gap:0.45rem 0.7rem;'>"
        "<span style='white-space:nowrap;'>❄️ Snow</span>"
        "<span style='white-space:nowrap;'>⛈️ Rain</span>"
        "<span style='white-space:nowrap;'>☔ Showers</span>"
        "<span style='white-space:nowrap;'>⚠️ Low</span>"
        "<span style='white-space:nowrap;'>🚨 High</span>"
        "</div>"
    )

    rows = [
        ("Weather Alerts", weather_alert_key_html),
        ("Night Clarity", clarity_key_html),
        ("Dew Risk", dew_key_html),
        ("Lunar Altitude", lunar_key_html),
    ]
    body_rows = "".join(
        (
            "<tr>"
            f"<td style='padding:0.04rem 0.6rem 0.04rem 0; text-align:right; white-space:nowrap; "
            "vertical-align:middle; color:#6b7280; font-weight:500;'>"
            f"{html.escape(label)}</td>"
            "<td style='padding:0.04rem 0; text-align:center; width:100%; vertical-align:middle;'>"
            f"{key_html}</td>"
            "</tr>"
        )
        for label, key_html in rows
    )
    return (
        "<table role='presentation' style='width:100%; border-collapse:collapse; "
        "font-size:0.88rem; margin:0.04rem 0 0.12rem;'>"
        f"<tbody>{body_rows}</tbody>"
        "</table>"
    )


def cloud_cover_cell_style(raw_value: Any) -> str:
    if raw_value is None or pd.isna(raw_value):
        return ""

    text = str(raw_value).strip()
    if not text or text == "-":
        return ""

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return ""

    try:
        cloud_cover_percent = float(match.group(0))
    except ValueError:
        return ""

    background_color = _interpolate_cloud_cover_color(cloud_cover_percent)
    text_color = "#111827" if cloud_cover_percent >= 65.0 else "#FFFFFF"
    return f"background-color: {background_color}; color: {text_color};"


def clarity_percentage_cell_style(raw_value: Any) -> str:
    if raw_value is None or pd.isna(raw_value):
        return ""

    text = str(raw_value).strip()
    if not text or text == "-":
        return ""

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return ""

    try:
        clarity_percent = float(match.group(0))
    except ValueError:
        return ""

    clamped_clear = max(0.0, min(100.0, clarity_percent))
    cloud_equivalent = 100.0 - clamped_clear
    background_color = _interpolate_cloud_cover_color(cloud_equivalent)
    text_color = "#111827" if cloud_equivalent >= 65.0 else "#FFFFFF"
    return f"background-color: {background_color}; color: {text_color};"


def visibility_condition_cell_style(raw_value: Any) -> str:
    if raw_value is None or pd.isna(raw_value):
        return ""

    condition = str(raw_value).strip().lower()
    if not condition or condition == "-":
        return ""

    clear_color = "#FFFFFF"
    fog_color = _interpolate_cloud_cover_color(100.0)
    gradient_stops = [(0.0, clear_color), (100.0, fog_color)]
    visibility_color_by_condition = {
        "clear": clear_color,
        "misty": _interpolate_color_stops(33.0, gradient_stops),
        "high haze": _interpolate_color_stops(66.0, gradient_stops),
        "fog": fog_color,
    }
    background_color = visibility_color_by_condition.get(condition)
    if not background_color:
        return ""

    text_color = _ideal_text_color_for_hex(background_color)
    return f"background-color: {background_color}; color: {text_color};"


def dewpoint_spread_cell_style(raw_value: Any) -> str:
    if raw_value is None or pd.isna(raw_value):
        return ""

    text = str(raw_value).strip()
    if not text or text == "-":
        return ""

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return ""

    try:
        spread_value = float(match.group(0))
    except ValueError:
        return ""

    if spread_value > 6.0:
        return ""
    clamped_spread = max(0.0, spread_value)
    background_color = _interpolate_color_stops(
        clamped_spread,
        [
            (0.0, "#FDBA74"),  # light orange at 0°
            (6.0, "#FFFFFF"),  # fade to white by 6°
        ],
    )

    text_color = _ideal_text_color_for_hex(background_color)
    return f"background-color: {background_color}; color: {text_color};"


def _temperature_f_from_display_value(raw_value: Any, temperature_unit: str) -> float | None:
    if raw_value is None or pd.isna(raw_value):
        return None

    text = str(raw_value).strip()
    if not text or text == "-":
        return None

    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([FC])", text, flags=re.IGNORECASE)
    if match is not None:
        numeric = float(match.group(1))
        unit = str(match.group(2)).upper()
        return numeric if unit == "F" else ((numeric * 9.0 / 5.0) + 32.0)

    number_match = re.search(r"-?\d+(?:\.\d+)?", text)
    if number_match is None:
        return None

    numeric = float(number_match.group(0))
    return numeric if str(temperature_unit).strip().lower() == "f" else ((numeric * 9.0 / 5.0) + 32.0)


def temperature_cell_style(raw_value: Any, temperature_unit: str) -> str:
    temp_f = _temperature_f_from_display_value(raw_value, temperature_unit)
    if temp_f is None:
        return ""
    text_color = _interpolate_temperature_color_f(temp_f)
    return f"color: {text_color}; font-weight: 700;"


def format_visibility_value(distance_meters: float | None, temperature_unit: str) -> str:
    if distance_meters is None or pd.isna(distance_meters):
        return "-"
    numeric = max(0.0, float(distance_meters))
    if str(temperature_unit).strip().lower() == "f":
        return f"{(numeric * 0.000621371):.1f} mi"
    return f"{(numeric * 0.001):.1f} km"


def format_visibility_condition(distance_meters: float | None) -> str:
    if distance_meters is None or pd.isna(distance_meters):
        return "-"
    miles = max(0.0, float(distance_meters)) * 0.000621371
    if miles > 6.0:
        return "Clear"
    if miles >= 4.0:
        return "misty"
    if miles >= 2.0:
        return "high haze"
    return "fog"


def _dewpoint_spread_celsius(temperature_celsius: Any, dewpoint_celsius: Any) -> float | None:
    if temperature_celsius is None or dewpoint_celsius is None:
        return None
    try:
        temperature_value = float(temperature_celsius)
        dewpoint_value = float(dewpoint_celsius)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(temperature_value) or not np.isfinite(dewpoint_value):
        return None
    return max(0.0, temperature_value - dewpoint_value)


def format_weather_matrix_value(metric_key: str, raw_value: Any, temperature_unit: str) -> str:
    if raw_value is None or pd.isna(raw_value):
        return "-"

    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return "-"

    if metric_key in {"temperature_2m", "dew_point_2m"}:
        return format_temperature(numeric, temperature_unit)
    if metric_key == "dewpoint_spread":
        if str(temperature_unit).strip().lower() == "f":
            return f"{(numeric * 9.0 / 5.0):.0f} F"
        return f"{numeric:.0f} C"
    if metric_key == "precipitation_probability":
        probability = max(0.0, float(numeric))
        if probability > 20.0:
            return "🚨"
        if probability >= 1.0:
            return "⚠️"
        return ""
    if metric_key in {"rain", "showers"}:
        return format_precipitation(numeric, temperature_unit)
    if metric_key == "snowfall":
        return format_snowfall(numeric, temperature_unit)
    if metric_key in {"relative_humidity_2m", "cloud_cover"}:
        return f"{numeric:.0f}%"
    if metric_key == "visibility":
        return format_visibility_condition(numeric)
    if metric_key == "wind_gusts_10m":
        return format_wind_speed(numeric, temperature_unit)
    return f"{numeric:.2f}"


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


def resolve_weather_alert_indicator(hour_row: dict[str, Any], temperature_unit: str) -> tuple[str, str]:
    rain = _positive_float(hour_row.get("rain"))
    showers = _positive_float(hour_row.get("showers"))
    snowfall = _positive_float(hour_row.get("snowfall"))
    precip_probability = _nonnegative_float(hour_row.get("precipitation_probability"))

    tooltip_parts: list[str] = []
    if precip_probability is not None:
        tooltip_parts.append(f"Precip probability: {precip_probability:.0f}%")
    else:
        tooltip_parts.append("Precip probability: -")

    if rain is not None:
        tooltip_parts.append(f"Rain: {format_precipitation(rain, temperature_unit)}")
    if showers is not None:
        tooltip_parts.append(f"Showers: {format_precipitation(showers, temperature_unit)}")
    if snowfall is not None:
        tooltip_parts.append(f"Snowfall: {format_snowfall(snowfall, temperature_unit)}")

    # One icon per hour with explicit precedence:
    # 1) actual precip (snow > rain > showers), otherwise
    # 2) precip-probability warning icon.
    if snowfall is not None:
        emoji = "❄️"
    elif rain is not None:
        emoji = "⛈️"
    elif showers is not None:
        emoji = "☔"
    elif precip_probability is not None and precip_probability > 20.0:
        emoji = "🚨"
    elif precip_probability is not None and precip_probability >= 1.0:
        emoji = "⚠️"
    else:
        return "", ""

    tooltip_text = " | ".join(tooltip_parts)
    return emoji, tooltip_text


def build_weather_alert_indicator_html(hour_row: dict[str, Any], temperature_unit: str) -> str:
    emoji, tooltip_text = resolve_weather_alert_indicator(hour_row, temperature_unit)
    if not emoji:
        return ""
    return (
        f'<span title="{html.escape(tooltip_text)}" '
        'style="display:inline-block; margin-left:4px;">'
        f"{emoji}</span>"
    )


def collect_night_weather_alert_emojis(rows: list[dict[str, Any]], temperature_unit: str) -> list[str]:
    weather_alert_rain_priority = tuple(_ui_name("WEATHER_ALERT_RAIN_PRIORITY"))
    seen: set[str] = set()
    for row in rows:
        emoji, _ = resolve_weather_alert_indicator(row, temperature_unit)
        if emoji:
            seen.add(emoji)

    # Render one rain emoji per night using explicit priority:
    # snow > rain > showers > alarm > caution.
    # A caution-only night is intentionally excluded from animation.
    if seen == {"⚠️"}:
        return []

    for candidate in weather_alert_rain_priority:
        if candidate in seen:
            return [candidate]
    return []


def normalize_hour_key(value: Any) -> str | None:
    try:
        timestamp = pd.Timestamp(value).floor("h")
    except Exception:
        return None

    if timestamp.tzinfo is not None:
        try:
            timestamp = timestamp.tz_convert("UTC")
        except Exception:
            pass
    return timestamp.isoformat()


def build_hourly_weather_maps(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, Any]]]:
    temperatures: dict[str, float] = {}
    cloud_cover_by_hour: dict[str, float] = {}
    weather_by_hour: dict[str, dict[str, Any]] = {}
    for weather_row in rows:
        time_iso = str(weather_row.get("time_iso", "")).strip()
        if not time_iso:
            continue
        hour_key = normalize_hour_key(time_iso)
        if not hour_key:
            continue

        temperature_value = weather_row.get("temperature_2m")
        if temperature_value is not None and not pd.isna(temperature_value):
            temperatures[hour_key] = float(temperature_value)

        cloud_cover_value = weather_row.get("cloud_cover")
        if cloud_cover_value is not None and not pd.isna(cloud_cover_value):
            cloud_cover_by_hour[hour_key] = float(cloud_cover_value)

        weather_by_hour[hour_key] = weather_row
    return temperatures, cloud_cover_by_hour, weather_by_hour

def _extract_finite_weather_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw_value = row.get(field)
        if raw_value is None or pd.isna(raw_value):
            continue
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return values




def compute_hourly_target_recommendations(
    catalog: pd.DataFrame,
    *,
    lat: float,
    lon: float,
    hour_start_local: pd.Timestamp | datetime,
    obstructions: dict[str, float],
    object_type_groups: list[str],
    exclude_ids: set[str] | None = None,
    max_results: int = 5,
) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame()

    selected_groups = {
        normalize_object_type_group(value)
        for value in object_type_groups
        if str(value).strip()
    }
    if not selected_groups:
        return pd.DataFrame()

    group_series = catalog["object_type_group"].map(normalize_object_type_group)
    candidate_mask = group_series.isin(selected_groups)

    excluded = {str(item).strip() for item in (exclude_ids or set()) if str(item).strip()}
    if excluded:
        candidate_mask &= ~catalog["primary_id"].astype(str).isin(excluded)

    candidate_columns = [
        "primary_id",
        "common_name",
        "description",
        "object_type",
        "object_type_group",
        "emission_lines",
        "ang_size_maj_arcmin",
        "ang_size_min_arcmin",
        "ra_deg",
        "dec_deg",
    ]
    candidates = catalog.loc[candidate_mask, candidate_columns].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates["ra_deg"] = pd.to_numeric(candidates["ra_deg"], errors="coerce")
    candidates["dec_deg"] = pd.to_numeric(candidates["dec_deg"], errors="coerce")
    candidates = candidates[np.isfinite(candidates["ra_deg"]) & np.isfinite(candidates["dec_deg"])]
    if candidates.empty:
        return pd.DataFrame()

    hour_start_ts = pd.Timestamp(hour_start_local)
    hour_end_ts = hour_start_ts + pd.Timedelta(hours=1)
    sample_times_local = pd.date_range(
        start=hour_start_ts,
        end=hour_end_ts,
        freq="10min",
        inclusive="both",
    )
    if sample_times_local.empty:
        return pd.DataFrame()

    target_count = len(candidates)
    time_count = len(sample_times_local)
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    sample_times_utc = sample_times_local.tz_convert("UTC")

    ra_values = candidates["ra_deg"].to_numpy(dtype=float)
    dec_values = candidates["dec_deg"].to_numpy(dtype=float)
    repeated_ra = np.tile(ra_values, time_count)
    repeated_dec = np.tile(dec_values, time_count)
    repeated_times = np.repeat(sample_times_utc.to_pydatetime(), target_count)

    coords = SkyCoord(ra=repeated_ra * u.deg, dec=repeated_dec * u.deg)
    frame = AltAz(obstime=Time(repeated_times), location=location)
    altaz = coords.transform_to(frame)

    altitude_matrix = np.asarray(altaz.alt.deg, dtype=float).reshape(time_count, target_count)
    azimuth_matrix = (np.asarray(altaz.az.deg, dtype=float).reshape(time_count, target_count)) % 360.0
    wind_index_matrix = (((azimuth_matrix + 11.25) // 22.5).astype(int)) % 16
    obstruction_thresholds = np.array(
        [float(obstructions.get(direction, 20.0)) for direction in WIND16],
        dtype=float,
    )
    min_required_matrix = obstruction_thresholds[wind_index_matrix]
    visible_matrix = (altitude_matrix >= 0.0) & (altitude_matrix >= min_required_matrix)

    visible_any = np.any(visible_matrix, axis=0)
    if not np.any(visible_any):
        return pd.DataFrame()

    visible_altitudes = np.where(visible_matrix, altitude_matrix, -np.inf)
    max_visible_altitude = np.max(visible_altitudes, axis=0)
    peak_index_by_target = np.argmax(visible_altitudes, axis=0)
    peak_time_local = np.array(
        [sample_times_local[int(index)] for index in peak_index_by_target],
        dtype=object,
    )
    direction_during_hour = np.array(["--"] * target_count, dtype=object)
    for target_index in range(target_count):
        visible_time_indexes = np.where(visible_matrix[:, target_index])[0]
        if visible_time_indexes.size == 0:
            continue

        start_idx = int(visible_time_indexes[0])
        end_idx = int(visible_time_indexes[-1])
        start_wind = WIND16[int(wind_index_matrix[start_idx, target_index])]
        end_wind = WIND16[int(wind_index_matrix[end_idx, target_index])]
        if start_wind == end_wind:
            direction_during_hour[target_index] = start_wind
        else:
            direction_during_hour[target_index] = f"{start_wind}->{end_wind}"

    recommended = candidates.copy()
    recommended["visible_in_hour"] = visible_any
    recommended = recommended[recommended["visible_in_hour"]].copy()
    if recommended.empty:
        return pd.DataFrame()

    visible_indices = np.where(visible_any)[0]
    recommended["max_alt_hour"] = np.round(max_visible_altitude[visible_indices], 1)
    recommended["peak_time_local"] = peak_time_local[visible_indices]
    recommended["direction_during_hour"] = direction_during_hour[visible_indices]
    recommended["object_type"] = recommended["object_type"].fillna("").astype(str).str.strip()
    recommended["object_type_group"] = recommended["object_type_group"].map(normalize_object_type_group)
    recommended["emissions"] = recommended["emission_lines"].apply(format_emissions_display)
    recommended["description_preview"] = recommended["description"].apply(
        lambda value: format_description_preview(value, max_chars=100)
    )
    recommended["apparent_size"] = recommended.apply(
        lambda row: format_apparent_size_display(
            row.get("ang_size_maj_arcmin"),
            row.get("ang_size_min_arcmin"),
        ),
        axis=1,
    )
    recommended["apparent_size_sort_arcmin"] = recommended.apply(
        lambda row: apparent_size_sort_key_arcmin(
            row.get("ang_size_maj_arcmin"),
            row.get("ang_size_min_arcmin"),
        ),
        axis=1,
    )

    primary_ids = recommended["primary_id"].astype(str)
    common_names = recommended["common_name"].fillna("").astype(str).str.strip()
    recommended["target"] = np.where(
        common_names != "",
        primary_ids + " - " + common_names,
        primary_ids,
    )
    recommended["line_color"] = recommended["object_type_group"].map(
        lambda group: object_type_group_color(normalize_object_type_group(group))
    )

    return (
        recommended.sort_values(
            by=["apparent_size_sort_arcmin", "max_alt_hour", "primary_id"],
            ascending=[False, False, True],
        )
        .head(max(1, int(max_results)))
        .loc[
            :,
            [
                "primary_id",
                "target",
                "description_preview",
                "object_type",
                "object_type_group",
                "emissions",
                "apparent_size",
                "max_alt_hour",
                "peak_time_local",
                "direction_during_hour",
                "line_color",
            ],
        ]
        .reset_index(drop=True)
    )


def build_hourly_weather_matrix(
    rows: list[dict[str, Any]],
    *,
    use_12_hour: bool,
    temperature_unit: str,
    lunar_hourly_altitude_by_hour: dict[str, float] | None = None,
    lunar_phase_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weather_matrix_rows = _ui_name("WEATHER_MATRIX_ROWS")
    metric_label_by_key = {str(key): str(label) for key, label in weather_matrix_rows}
    visible_metric_keys_in_order = [
        "cloud_cover",
        "visibility",
        "wind_gusts_10m",
        "dewpoint_spread",
        "temperature_2m",
    ]
    ordered_weather_matrix_rows = [
        (metric_key, metric_label_by_key[metric_key])
        for metric_key in visible_metric_keys_in_order
        if metric_key in metric_label_by_key
    ]
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    by_hour: dict[pd.Timestamp, dict[str, Any]] = {}
    for row in rows:
        time_iso = str(row.get("time_iso", "")).strip()
        if not time_iso:
            continue
        try:
            timestamp = pd.Timestamp(time_iso)
        except Exception:
            continue
        by_hour[timestamp] = row

    if not by_hour:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ordered_times = sorted(by_hour.keys())
    hour_labels = [format_hour_label(timestamp, use_12_hour=use_12_hour) for timestamp in ordered_times]

    matrix_rows: dict[str, list[str]] = {}
    tooltip_rows: dict[str, list[str]] = {}
    indicator_rows: dict[str, list[str]] = {}
    lunar_row_inserted = False

    def _insert_lunar_row() -> None:
        nonlocal lunar_row_inserted
        if lunar_row_inserted:
            return
        if not isinstance(lunar_hourly_altitude_by_hour, dict):
            return

        row_label = lunar_altitude_row_label(lunar_phase_key)
        matrix_rows[row_label] = [
            format_lunar_altitude_display(
                lunar_hourly_altitude_by_hour.get(pd.Timestamp(timestamp).floor("h").isoformat())
            )
            for timestamp in ordered_times
        ]
        tooltip_rows[row_label] = [
            (
                f"Moon average altitude during hour: {format_lunar_altitude_display(avg_alt)}"
                if (avg_alt := lunar_hourly_altitude_by_hour.get(pd.Timestamp(timestamp).floor('h').isoformat())) is not None
                else ""
            )
            for timestamp in ordered_times
        ]
        indicator_rows[row_label] = ["" for _ in ordered_times]
        lunar_row_inserted = True

    for metric_key, metric_label in ordered_weather_matrix_rows:
        if metric_key == "dewpoint_spread":
            matrix_rows[metric_label] = [
                format_weather_matrix_value(
                    "relative_humidity_2m",
                    by_hour[timestamp].get("relative_humidity_2m"),
                    temperature_unit=temperature_unit,
                )
                for timestamp in ordered_times
            ]
        else:
            matrix_rows[metric_label] = [
                format_weather_matrix_value(
                    metric_key,
                    by_hour[timestamp].get(metric_key),
                    temperature_unit=temperature_unit,
                )
                for timestamp in ordered_times
            ]
        if metric_key == "cloud_cover":
            indicator_rows[metric_label] = [
                build_weather_alert_indicator_html(by_hour[timestamp], temperature_unit=temperature_unit)
                for timestamp in ordered_times
            ]
        else:
            indicator_rows[metric_label] = ["" for _ in ordered_times]
        if metric_key == "dewpoint_spread":
            tooltip_rows[metric_label] = [
                (
                    (
                        "Dewpoint spread: "
                        + format_weather_matrix_value(
                            "dewpoint_spread",
                            _dewpoint_spread_celsius(
                                by_hour[timestamp].get("temperature_2m"),
                                by_hour[timestamp].get("dew_point_2m"),
                            ),
                            temperature_unit=temperature_unit,
                        )
                    )
                    if (
                        _dewpoint_spread_celsius(
                            by_hour[timestamp].get("temperature_2m"),
                            by_hour[timestamp].get("dew_point_2m"),
                        )
                        is not None
                    )
                    else ""
                )
                for timestamp in ordered_times
            ]
        elif metric_key == "precipitation_probability":
            tooltip_rows[metric_label] = [
                (
                    f"Precip probability: {float(by_hour[timestamp].get(metric_key)):.0f}%"
                    if by_hour[timestamp].get(metric_key) is not None and not pd.isna(by_hour[timestamp].get(metric_key))
                    else ""
                )
                for timestamp in ordered_times
            ]
        elif metric_key == "visibility":
            tooltip_rows[metric_label] = [
                (
                    f"Visibility: {format_visibility_value(by_hour[timestamp].get(metric_key), temperature_unit)}"
                    if by_hour[timestamp].get(metric_key) is not None and not pd.isna(by_hour[timestamp].get(metric_key))
                    else ""
                )
                for timestamp in ordered_times
            ]
        else:
            tooltip_rows[metric_label] = ["" for _ in ordered_times]

        if metric_key == "cloud_cover":
            _insert_lunar_row()

    if not lunar_row_inserted:
        _insert_lunar_row()

    return (
        pd.DataFrame.from_dict(matrix_rows, orient="index", columns=hour_labels),
        pd.DataFrame.from_dict(tooltip_rows, orient="index", columns=hour_labels),
        pd.DataFrame.from_dict(indicator_rows, orient="index", columns=hour_labels),
    )




def render_hourly_weather_matrix(
    frame: pd.DataFrame,
    *,
    temperature_unit: str,
    tooltip_frame: pd.DataFrame | None = None,
    indicator_frame: pd.DataFrame | None = None,
    hour_column_states: dict[str, str] | None = None,
) -> str | None:
    mui_table = _get_st_mui_table()
    if frame.empty:
        st.info("No hourly weather data available.")
        return None

    normalized_hour_column_states: dict[str, str] = {}
    if isinstance(hour_column_states, dict):
        for raw_column_name, raw_state in hour_column_states.items():
            column_name = str(raw_column_name).strip()
            state = str(raw_state).strip().lower()
            if not column_name or state not in {"past", "current"}:
                continue
            normalized_hour_column_states[column_name] = state

    def _hour_column_inline_style(column_name: Any) -> str:
        state = normalized_hour_column_states.get(str(column_name).strip(), "")
        if state == "past":
            return "background-color: rgba(148, 163, 184, 0.15); color: #475569;"
        if state == "current":
            return (
                "box-shadow: inset 2px 0 0 #FACC15, inset -2px 0 0 #FACC15;"
                " background-color: rgba(250, 204, 21, 0.05);"
            )
        return ""

    aligned_tooltips: pd.DataFrame | None = None
    if tooltip_frame is not None and not tooltip_frame.empty:
        aligned_tooltips = tooltip_frame.reindex(index=frame.index, columns=frame.columns, fill_value="")
        if "Element" in aligned_tooltips.columns:
            aligned_tooltips["Element"] = ""
    aligned_indicators: pd.DataFrame | None = None
    if indicator_frame is not None and not indicator_frame.empty:
        aligned_indicators = indicator_frame.reindex(index=frame.index, columns=frame.columns, fill_value="")
        if "Element" in aligned_indicators.columns:
            aligned_indicators["Element"] = ""

    if mui_table is not None:
        def _display_element_label(raw_element: str) -> str:
            label = str(raw_element or "").strip()
            if label.lower().startswith("lunar altitude"):
                return "Lunar Altitude"
            return label

        def _mui_cell_html(
            text: str,
            style_parts: list[str],
            *,
            title_text: str = "",
            extra_html: str = "",
            full_bleed: bool = False,
        ) -> str:
            safe_text = html.escape(text) if text else "&nbsp;"
            content_html = safe_text
            if extra_html:
                spacer = "&nbsp;" if content_html != "&nbsp;" else ""
                content_html = f"{content_html}{spacer}{extra_html}"

            local_style_parts = [part for part in style_parts if str(part).strip()]
            if full_bleed:
                local_style_parts.extend(
                    [
                        "display:block;",
                        "width:calc(100% + 16px);",
                        "margin:-6px -8px;",
                        "padding:6px 8px;",
                        "box-sizing:border-box;",
                    ]
                )
            style = " ".join(local_style_parts)
            title_attr = f" title=\"{html.escape(title_text)}\"" if title_text else ""
            return f"<div{title_attr} style='{style}'>{content_html}</div>"

        mui_frame = frame.copy()
        if "Element" in mui_frame.columns:
            mui_frame = mui_frame.rename(columns={"Element": ""})
        for row_idx, row in frame.iterrows():
            element = str(row.get("Element", "")).strip()
            element_key = element.lower()
            element_display = _display_element_label(element)
            for column in frame.columns:
                raw_value = row.get(column)
                cell_text = str(raw_value).strip() if raw_value is not None and not pd.isna(raw_value) else ""
                tooltip_text = ""
                if aligned_tooltips is not None:
                    tooltip_text = str(aligned_tooltips.at[row_idx, column]).strip()

                if str(column) == "Element":
                    mui_frame.at[row_idx, ""] = _mui_cell_html(
                        element_display,
                        ["font-weight: 600;", "white-space: nowrap;", "text-align: right;"],
                    )
                    continue

                style_parts = ["white-space: nowrap;", "text-align: center;"]
                if element_key == "cloud cover":
                    cloud_style = cloud_cover_cell_style(raw_value)
                    if cloud_style:
                        style_parts.append(cloud_style)
                elif element_key == "temperature":
                    temp_style = temperature_cell_style(raw_value, temperature_unit=temperature_unit)
                    if temp_style:
                        style_parts.append(temp_style)
                elif element_key == "humidity":
                    dew_style = dewpoint_spread_cell_style(tooltip_text)
                    if dew_style:
                        style_parts.append(dew_style)
                elif element_key == "visibility":
                    visibility_style = visibility_condition_cell_style(raw_value)
                    if visibility_style:
                        style_parts.append(visibility_style)
                elif element_key.startswith("lunar altitude"):
                    lunar_style = lunar_altitude_cell_style(raw_value, element)
                    if lunar_style:
                        style_parts.append(lunar_style)

                indicator_html = ""
                if aligned_indicators is not None and element_key == "cloud cover":
                    indicator_html = str(aligned_indicators.at[row_idx, column]).strip()

                full_bleed = element_key in {"cloud cover", "humidity", "visibility"} or element_key.startswith(
                    "lunar altitude"
                )
                mui_frame.at[row_idx, column] = _mui_cell_html(
                    cell_text,
                    style_parts,
                    title_text=tooltip_text,
                    extra_html=indicator_html,
                    full_bleed=full_bleed,
                )

        dynamic_column_css_rules: list[str] = []
        for column_index, column_name in enumerate(frame.columns, start=1):
            normalized_name = str(column_name).strip()
            if normalized_name == "Element":
                continue
            state = normalized_hour_column_states.get(normalized_name, "")
            if state == "past":
                dynamic_column_css_rules.append(
                    (
                        f".MuiTableHead-root .MuiTableCell-root:nth-child({column_index})"
                        "{ background: rgba(148, 163, 184, 0.15) !important; color: #64748b !important; }"
                    )
                )
            elif state == "current":
                dynamic_column_css_rules.append(
                    (
                        f".MuiTableHead-root .MuiTableCell-root:nth-child({column_index})"
                        "{ background: rgba(250, 204, 21, 0.08) !important; color: #854D0E !important; "
                        "box-shadow: inset 2px 0 0 #FACC15, inset -2px 0 0 #FACC15, inset 0 2px 0 #FACC15, inset 0 -2px 0 #FACC15 !important; }"
                    )
                )

        mui_custom_css = """
.MuiTableCell-root {
  padding: 6px 8px !important;
  border-bottom: none !important;
  border-top: none !important;
  border-right: 1px solid rgba(148, 163, 184, 0.28) !important;
}
.MuiTableCell-root:last-child {
  border-right: none !important;
}
.MuiTableBody-root .MuiTableRow-root .MuiTableCell-root:not(:first-child),
.MuiTableHead-root .MuiTableCell-root:not(:first-child) {
  text-align: center !important;
}
.MuiTableHead-root .MuiTableCell-root:first-child {
  position: sticky !important;
  left: 0 !important;
  z-index: 4 !important;
  background: #f3f4f6 !important;
}
.MuiTableBody-root .MuiTableRow-root .MuiTableCell-root:first-child {
  position: sticky !important;
  left: 0 !important;
  z-index: 3 !important;
  background: #ffffff !important;
}
.MuiTablePagination-root {
  display: none !important;
}
""" + ("\n".join(dynamic_column_css_rules) if dynamic_column_css_rules else "")
        clicked_cell = mui_table(
            mui_frame,
            enablePagination=True,
            paginationSizes=[24],
            customCss=mui_custom_css,
            showHeaders=True,
            key="hourly_weather_mui_table",
            stickyHeader=False,
            showIndex=False,
            enable_sorting=False,
            return_clicked_cell=True,
            paperStyle={
                "width": "100%",
                "overflow": "visible",
                "paddingBottom": "0px",
                "border": "1px solid rgba(148, 163, 184, 0.35)",
            },
        )
        if isinstance(clicked_cell, dict):
            raw_column = (
                clicked_cell.get("column")
                if "column" in clicked_cell
                else clicked_cell.get("col", clicked_cell.get("field"))
            )
            clicked_column_name = ""
            if isinstance(raw_column, str):
                clicked_column_name = raw_column
            else:
                try:
                    parsed_column_index = int(raw_column)
                except (TypeError, ValueError):
                    parsed_column_index = -1
                if 0 <= parsed_column_index < len(mui_frame.columns):
                    clicked_column_name = str(mui_frame.columns[parsed_column_index])

            clicked_column_name = "Element" if clicked_column_name == "" else clicked_column_name
            if clicked_column_name and clicked_column_name != "Element" and clicked_column_name in frame.columns:
                return clicked_column_name
        return None

    def _display_element_label(raw_element: str) -> str:
        label = str(raw_element or "").strip()
        if label.lower().startswith("lunar altitude"):
            return "Lunar Altitude"
        return label

    header_cells_parts: list[str] = []
    for column in frame.columns:
        column_name = str(column)
        header_style = (
            "padding: 6px 8px; border-bottom: 1px solid #d1d5db; "
            "background: #f3f4f6; color: #6b7280; text-align: left; white-space: nowrap;"
        )
        if column_name != "Element":
            header_style = header_style.replace("text-align: left;", "text-align: center;")
            column_state_style = _hour_column_inline_style(column_name)
            if column_state_style:
                header_style += column_state_style
            if normalized_hour_column_states.get(column_name) == "current":
                header_style += " box-shadow: inset 2px 0 0 #FACC15, inset -2px 0 0 #FACC15, inset 0 2px 0 #FACC15, inset 0 -2px 0 #FACC15;"
        header_cells_parts.append(f'<th style="{header_style}">{html.escape(column_name)}</th>')
    header_cells = "".join(header_cells_parts)

    body_rows: list[str] = []
    for row_idx, row in frame.iterrows():
        element = str(row.get("Element", "")).strip()
        element_key = element.lower()
        element_display = _display_element_label(element)
        row_cells = [
            (
                '<td style="padding: 6px 8px; border-bottom: 1px solid #d1d5db; '
                'font-weight: 600; white-space: nowrap; text-align: right;">'
                f"{html.escape(element_display) if element_display else '&nbsp;'}</td>"
            )
        ]

        for column in frame.columns:
            if str(column) == "Element":
                continue

            raw_value = row.get(column)
            text_value = str(raw_value).strip()
            display_value = html.escape(text_value) if text_value else "&nbsp;"
            tooltip = ""
            if aligned_tooltips is not None:
                tooltip = str(aligned_tooltips.at[row_idx, column]).strip()

            cell_style = (
                "padding: 6px 8px; border-bottom: 1px solid #d1d5db; "
                "white-space: nowrap; text-align: center;"
            )
            if element_key == "cloud cover":
                cell_style += cloud_cover_cell_style(raw_value)
            elif element_key == "temperature":
                cell_style += temperature_cell_style(raw_value, temperature_unit=temperature_unit)
            elif element_key == "humidity":
                cell_style += dewpoint_spread_cell_style(tooltip)
            elif element_key == "visibility":
                cell_style += visibility_condition_cell_style(raw_value)
            elif element_key.startswith("lunar altitude"):
                cell_style += lunar_altitude_cell_style(raw_value, element)
            title_attr = f' title="{html.escape(tooltip)}"' if tooltip else ""

            indicator_html = ""
            if aligned_indicators is not None:
                indicator_html = str(aligned_indicators.at[row_idx, column]).strip()
            cell_content = display_value
            if element_key == "cloud cover" and indicator_html:
                spacer = "&nbsp;" if cell_content != "&nbsp;" else ""
                cell_content = f"{cell_content}{spacer}{indicator_html}"

            row_cells.append(f'<td{title_attr} style="{cell_style}">{cell_content}</td>')

        body_rows.append("<tr>" + "".join(row_cells) + "</tr>")

    table_html = (
        '<div style="overflow-x: auto; max-width: 100%;">'
        '<table style="border-collapse: collapse; width: max-content; min-width: 100%; font-size: 0.855rem;">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
    return None


def build_astronomy_forecast_summary(
    lat: float,
    lon: float,
    *,
    tz_name: str,
    temperature_unit: str,
    browser_locale: str | None = None,
    browser_month_day_pattern: str | None = None,
    nights: int | None = None,
) -> pd.DataFrame:
    astronomy_forecast_nights = int(_ui_name("ASTRONOMY_FORECAST_NIGHTS"))
    night_count = max(1, int(astronomy_forecast_nights if nights is None else nights))
    windows: list[tuple[datetime, datetime]] = []
    for day_offset in range(night_count):
        window_start, window_end, _ = astronomical_night_window(lat, lon, day_offset=day_offset)
        windows.append((window_start, window_end))

    if not windows:
        return pd.DataFrame()

    all_rows = fetch_hourly_weather(
        lat=lat,
        lon=lon,
        tz_name=tz_name,
        start_local_iso=windows[0][0].isoformat(),
        end_local_iso=windows[-1][1].isoformat(),
        hourly_fields=EXTENDED_FORECAST_HOURLY_FIELDS,
    )

    rows_with_time: list[tuple[pd.Timestamp, dict[str, Any]]] = []
    for row in all_rows:
        time_iso = str(row.get("time_iso", "")).strip()
        if not time_iso:
            continue
        try:
            timestamp = pd.Timestamp(time_iso)
        except Exception:
            continue
        rows_with_time.append((timestamp, row))

    def _finite(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return float(numeric)

    def _format_ratio_percent(pass_count: int, total_count: int) -> str:
        if total_count <= 0:
            return "-"
        percent = (float(pass_count) / float(total_count)) * 100.0
        return f"{percent:.0f}%"

    summary_rows: list[dict[str, Any]] = []
    previous_lunar_phase_key: str | None = None
    for day_offset, (window_start, window_end) in enumerate(windows):
        night_rows = [
            row
            for timestamp, row in rows_with_time
            if window_start <= timestamp <= window_end
        ]

        temperature_values = _extract_finite_weather_values(night_rows, "temperature_2m")
        alert_emojis = collect_night_weather_alert_emojis(night_rows, temperature_unit)
        rating_details = compute_night_rating_details(
            night_rows,
            temperature_unit=temperature_unit,
        )
        rating_emoji = str((rating_details or {}).get("emoji", "")).strip()

        label = format_weather_forecast_date(
            window_start,
            browser_locale=browser_locale,
            browser_month_day_pattern=browser_month_day_pattern,
        )
        if "," in label:
            day_name, date_part = label.split(",", 1)
            label = f"{day_name[:3]}, {date_part.strip()}"
        lunar_phase_payload = compute_lunar_phase_for_night(
            tz_name=tz_name,
            start_local_iso=window_start.isoformat(),
            end_local_iso=window_end.isoformat(),
        )
        lunar_phase_key = str((lunar_phase_payload or {}).get("phase_key", "")).strip().lower() or None
        lunar_phase_display_prefix = ""
        if lunar_phase_key:
            if previous_lunar_phase_key is None:
                lunar_phase_display_prefix = lunar_phase_emoji(lunar_phase_key)
            elif lunar_phase_key != previous_lunar_phase_key:
                previous_phase_emoji = lunar_phase_emoji(previous_lunar_phase_key)
                current_phase_emoji = lunar_phase_emoji(lunar_phase_key)
                lunar_phase_display_prefix = f"{previous_phase_emoji} \u2192 {current_phase_emoji}"
        previous_lunar_phase_key = lunar_phase_key

        if rating_emoji:
            label = f"{label} {rating_emoji}"

        temp_unit = str(temperature_unit).strip().lower()
        temp_unit_suffix = "F" if temp_unit == "f" else "C"
        low_temp = (
            format_temperature(min(temperature_values), temperature_unit)
            if temperature_values
            else "-"
        )
        high_temp = (
            format_temperature(max(temperature_values), temperature_unit)
            if temperature_values
            else "-"
        )
        if temperature_values:
            if temp_unit == "f":
                low_temp_numeric = (min(temperature_values) * 9.0 / 5.0) + 32.0
                high_temp_numeric = (max(temperature_values) * 9.0 / 5.0) + 32.0
            else:
                low_temp_numeric = min(temperature_values)
                high_temp_numeric = max(temperature_values)
            temp_range_display = f"{low_temp_numeric:.0f}-{high_temp_numeric:.0f} {temp_unit_suffix}"
        else:
            temp_range_display = "-"
        avg_temp_display = (
            format_temperature(float(np.mean(temperature_values)), temperature_unit)
            if temperature_values
            else "-"
        )

        clear_hour_count = 0
        calm_hour_count = 0
        crisp_hour_count = 0
        total_hour_count = len(night_rows)
        for weather_row in night_rows:
            cloud_cover = _finite(weather_row.get("cloud_cover"))
            visibility_m = _finite(weather_row.get("visibility"))
            visibility_miles = visibility_m * 0.000621371 if visibility_m is not None else None
            if cloud_cover is not None and visibility_miles is not None:
                if cloud_cover < 30.0 and visibility_miles >= 4.0:
                    clear_hour_count += 1

            gust_kmh = _finite(weather_row.get("wind_gusts_10m"))
            if gust_kmh is None:
                gust_kmh = _finite(weather_row.get("wind_speed_10m"))
            if gust_kmh is not None and (gust_kmh * 0.621371) <= 12.0:
                calm_hour_count += 1

            humidity = _finite(weather_row.get("relative_humidity_2m"))
            temperature_c = _finite(weather_row.get("temperature_2m"))
            dewpoint_c = _finite(weather_row.get("dew_point_2m"))
            if humidity is not None and temperature_c is not None and dewpoint_c is not None:
                spread_c = max(0.0, temperature_c - dewpoint_c)
                spread_value = spread_c * 9.0 / 5.0 if temp_unit == "f" else spread_c
                if humidity <= 65.0 and spread_value >= 5.0:
                    crisp_hour_count += 1

        clear_percent_text = _format_ratio_percent(clear_hour_count, total_hour_count)
        calm_percent_text = _format_ratio_percent(calm_hour_count, total_hour_count)
        crisp_percent_text = _format_ratio_percent(crisp_hour_count, total_hour_count)
        primary_alert_emoji = str(alert_emojis[0]) if alert_emojis else ""
        clear_display = clear_percent_text
        if primary_alert_emoji:
            clear_display = (
                f"{clear_percent_text} {primary_alert_emoji}"
                if clear_percent_text != "-"
                else primary_alert_emoji
            )

        sunset_start, sunrise_end, _ = weather_forecast_window(
            lat,
            lon,
            day_offset=day_offset,
        )
        sunset_text = re.sub(
            r"\b(am|pm)\b",
            lambda match: str(match.group(1)).upper(),
            format_display_time(sunset_start, use_12_hour=True),
            flags=re.IGNORECASE,
        )
        sunrise_text = re.sub(
            r"\b(am|pm)\b",
            lambda match: str(match.group(1)).upper(),
            format_display_time(sunrise_end, use_12_hour=True),
            flags=re.IGNORECASE,
        )
        dark_total_minutes = max(0, int(round((window_end - window_start).total_seconds() / 60.0)))
        dark_hours, dark_minutes = divmod(dark_total_minutes, 60)
        dark_duration = f"{dark_hours}h {dark_minutes:02d}m"
        lunar_summary = compute_lunar_night_summary(
            lat=lat,
            lon=lon,
            tz_name=tz_name,
            start_local_iso=window_start.isoformat(),
            end_local_iso=window_end.isoformat(),
            sample_minutes=10,
        )
        lunar_below_horizon_pct = (lunar_summary or {}).get("moon_below_horizon_pct")
        lunar_avg_altitude_deg = (lunar_summary or {}).get("moon_avg_altitude_deg")
        lunar_max_altitude_deg = (lunar_summary or {}).get("moon_max_altitude_deg")
        lunar_display = format_lunar_percent_display(lunar_below_horizon_pct)
        if lunar_phase_display_prefix:
            if lunar_display == "-":
                lunar_display = lunar_phase_display_prefix
            else:
                lunar_display = f"{lunar_phase_display_prefix} {lunar_display}"

        summary_rows.append(
            {
                "day_offset": day_offset,
                "Night": label,
                "Lunar": lunar_display,
                "Dark Time": dark_duration,
                "lunar_phase_key": lunar_phase_key or "",
                "lunar_avg_altitude_deg": lunar_avg_altitude_deg,
                "lunar_max_altitude_deg": lunar_max_altitude_deg,
                "Sunset": sunset_text,
                "Sunrise": sunrise_text,
                "Clear": clear_display,
                "Calm": calm_percent_text,
                "Crisp": crisp_percent_text,
                "Low-Hi": temp_range_display,
                "Temp Range": temp_range_display,
                "temp_low_display": low_temp,
                "temp_high_display": high_temp,
                "Avg. Temp": avg_temp_display,
                "Warnings": primary_alert_emoji,
            }
        )

    return pd.DataFrame(summary_rows)


def render_astronomy_forecast_summary(
    frame: pd.DataFrame,
    *,
    temperature_unit: str,
    selected_day_offset: int,
) -> None:
    astronomy_forecast_nights = int(_ui_name("ASTRONOMY_FORECAST_NIGHTS"))
    weather_forecast_day_offset_state_key = str(_ui_name("WEATHER_FORECAST_DAY_OFFSET_STATE_KEY"))
    weather_forecast_period_state_key = str(_ui_name("WEATHER_FORECAST_PERIOD_STATE_KEY"))
    weather_forecast_period_tonight = _ui_name("WEATHER_FORECAST_PERIOD_TONIGHT")
    weather_forecast_period_tomorrow = _ui_name("WEATHER_FORECAST_PERIOD_TOMORROW")
    mui_table = _get_st_mui_table()
    if frame.empty:
        st.info("No 5-night forecast data available.")
        return

    ordered_columns = [
        "Night",
        "Clear",
        "Lunar",
        "Calm",
        "Crisp",
        "Dark Time",
        "Avg. Temp",
    ]
    source_frame = frame.copy()
    if "day_offset" not in source_frame.columns:
        source_frame["day_offset"] = 0
    if "Dark Time" not in source_frame.columns and "Dark" in source_frame.columns:
        source_frame["Dark Time"] = source_frame["Dark"]
    display_frame = source_frame.reindex(columns=ordered_columns, fill_value="").reset_index(drop=True)
    source_frame = source_frame.reset_index(drop=True)

    if mui_table is not None:
        center_columns = {"Clear", "Calm", "Crisp", "Lunar", "Dark Time"}

        def _mui_cell_html(text: str, style_parts: list[str], *, full_bleed: bool = False) -> str:
            safe_text = html.escape(text) if text else "&nbsp;"
            local_style_parts = [part for part in style_parts if str(part).strip()]
            if full_bleed:
                local_style_parts.extend(
                    [
                        "display:block;",
                        "width:calc(100% + 16px);",
                        "margin:-6px -8px;",
                        "padding:6px 8px;",
                        "box-sizing:border-box;",
                    ]
                )
            style = " ".join(local_style_parts)
            return f"<div style='{style}'>{safe_text}</div>"

        mui_frame = display_frame.copy()
        for row_idx, row in display_frame.iterrows():
            row_day_offset = normalize_weather_forecast_day_offset(
                source_frame.at[int(row_idx), "day_offset"],
                max_offset=astronomy_forecast_nights - 1,
            )
            row_is_selected = row_day_offset == selected_day_offset
            for column_name in ordered_columns:
                raw_cell_value = row.get(column_name)
                cell_text = "" if raw_cell_value is None or pd.isna(raw_cell_value) else str(raw_cell_value).strip()
                style_parts = ["white-space: nowrap;"]
                if row_is_selected:
                    style_parts.append("background-color: rgba(37, 99, 235, 0.14);")
                if column_name == "Night" and row_is_selected:
                    style_parts.append("font-weight: 700;")
                if column_name in center_columns:
                    style_parts.append("text-align: center;")
                if column_name == "Clear":
                    clear_style = clarity_percentage_cell_style(cell_text)
                    if clear_style:
                        style_parts.append(clear_style)
                if column_name == "Lunar":
                    lunar_style = lunar_forecast_column_cell_style(
                        raw_cell_value,
                        lunar_phase_key=(
                            source_frame.at[int(row_idx), "lunar_phase_key"]
                            if "lunar_phase_key" in source_frame.columns
                            else ""
                        ),
                        lunar_max_altitude_deg=(
                            source_frame.at[int(row_idx), "lunar_max_altitude_deg"]
                            if "lunar_max_altitude_deg" in source_frame.columns
                            else (
                                source_frame.at[int(row_idx), "lunar_avg_altitude_deg"]
                                if "lunar_avg_altitude_deg" in source_frame.columns
                                else None
                            )
                        ),
                    )
                    if lunar_style:
                        style_parts.append(lunar_style)
                if column_name == "Avg. Temp":
                    avg_style = temperature_cell_style(cell_text, temperature_unit=temperature_unit)
                    if avg_style:
                        style_parts.append(avg_style)
                mui_frame.at[int(row_idx), column_name] = _mui_cell_html(
                    cell_text,
                    style_parts,
                    full_bleed=(column_name in {"Clear", "Lunar"}),
                )
        if "Night" in mui_frame.columns:
            mui_frame = mui_frame.rename(columns={"Night": ""})

        mui_custom_css = """
.MuiTableCell-root {
  padding: 6px 8px !important;
}
.MuiTableHead-root .MuiTableCell-root {
  vertical-align: bottom !important;
  padding-top: 10px !important;
  padding-bottom: 4px !important;
}
.MuiTableCell-root:nth-child(2),
.MuiTableCell-root:nth-child(3),
.MuiTableCell-root:nth-child(4),
.MuiTableCell-root:nth-child(5),
.MuiTableCell-root:nth-child(6) {
  text-align: center !important;
}
.MuiTableHead-root .MuiTableCell-root:nth-child(2),
.MuiTableHead-root .MuiTableCell-root:nth-child(3),
.MuiTableHead-root .MuiTableCell-root:nth-child(4),
.MuiTableHead-root .MuiTableCell-root:nth-child(5),
.MuiTableHead-root .MuiTableCell-root:nth-child(6) {
  text-align: center !important;
}
.MuiTableHead-root .MuiTableCell-root:first-child {
  position: sticky !important;
  left: 0 !important;
  z-index: 4 !important;
  background: #f3f4f6 !important;
}
.MuiTableBody-root .MuiTableRow-root .MuiTableCell-root:first-child {
  position: sticky !important;
  left: 0 !important;
  z-index: 3 !important;
  background: #ffffff !important;
}
.MuiTablePagination-root {
  display: none !important;
}
"""
        clicked_cell = mui_table(
            mui_frame,
            enablePagination=True,
            customCss=mui_custom_css,
            paginationSizes=[10],
            showHeaders=True,
            key="astronomy_forecast_mui_table",
            stickyHeader=False,
            showIndex=False,
            enable_sorting=False,
            return_clicked_cell=True,
            paperStyle={
                "width": "100%",
                "overflow": "visible",
                "paddingBottom": "0px",
                "border": "1px solid rgba(148, 163, 184, 0.35)",
            },
        )

        selected_index: int | None = None
        if isinstance(clicked_cell, dict):
            try:
                parsed_row_index = int(clicked_cell.get("row"))
            except (TypeError, ValueError):
                parsed_row_index = -1
            if 0 <= parsed_row_index < len(source_frame):
                selected_index = parsed_row_index

        if selected_index is None:
            return

        selected_offset = normalize_weather_forecast_day_offset(
            source_frame.at[selected_index, "day_offset"],
            max_offset=astronomy_forecast_nights - 1,
        )
        selection_token = f"{selected_index}:{selected_offset}"
        last_selection_token = str(st.session_state.get("astronomy_forecast_last_selection_token", ""))
        if selection_token == last_selection_token:
            return

        st.session_state["astronomy_forecast_last_selection_token"] = selection_token
        if selected_offset != selected_day_offset:
            st.session_state[weather_forecast_day_offset_state_key] = selected_offset
            st.session_state[weather_forecast_period_state_key] = (
                weather_forecast_period_tonight if selected_offset <= 0 else weather_forecast_period_tomorrow
            )
            st.rerun()
        return

    def _style_forecast_row(row: pd.Series) -> list[str]:
        row_idx = int(row.name)
        row_day_offset = normalize_weather_forecast_day_offset(
            source_frame.at[row_idx, "day_offset"],
            max_offset=astronomy_forecast_nights - 1,
        )
        row_is_selected = row_day_offset == selected_day_offset

        styles: list[str] = []
        for column in display_frame.columns:
            style_parts = ["white-space: nowrap;"]
            if row_is_selected:
                style_parts.append("background-color: rgba(37, 99, 235, 0.14);")
            if column == "Night" and row_is_selected:
                style_parts.append("font-weight: 700;")
            if column in {"Lunar", "Dark Time", "Clear", "Calm", "Crisp"}:
                style_parts.append("text-align: center !important;")
                style_parts.append("justify-content: center !important;")
            if column == "Clear":
                style_parts.append(clarity_percentage_cell_style(row.get(column)))
            if column == "Lunar":
                style_parts.append(
                    lunar_forecast_column_cell_style(
                        row.get(column),
                        lunar_phase_key=(
                            source_frame.at[row_idx, "lunar_phase_key"]
                            if "lunar_phase_key" in source_frame.columns
                            else ""
                        ),
                        lunar_max_altitude_deg=(
                            source_frame.at[row_idx, "lunar_max_altitude_deg"]
                            if "lunar_max_altitude_deg" in source_frame.columns
                            else (
                                source_frame.at[row_idx, "lunar_avg_altitude_deg"]
                                if "lunar_avg_altitude_deg" in source_frame.columns
                                else None
                            )
                        ),
                    )
                )
            if column == "Avg. Temp":
                style_parts.append(temperature_cell_style(row.get(column), temperature_unit=temperature_unit))
            styles.append(" ".join(part for part in style_parts if str(part).strip()))
        return styles

    base_styler = display_frame.style.apply(_style_forecast_row, axis=1)
    base_styler = base_styler.set_properties(
        subset=["Clear", "Calm", "Crisp", "Lunar", "Dark Time"],
        **{
            "text-align": "center !important",
            "justify-content": "center !important",
        },
    )
    styled = apply_dataframe_styler_theme(base_styler)

    table_event = st.dataframe(
        styled,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-cell",
        key="astronomy_forecast_table",
        column_config={
            "Night": st.column_config.TextColumn("", width="small"),
            "Clear": st.column_config.TextColumn(width="small"),
            "Calm": st.column_config.TextColumn(width="small"),
            "Crisp": st.column_config.TextColumn(width="small"),
            "Lunar": st.column_config.TextColumn(width="small"),
            "Dark Time": st.column_config.TextColumn(width="small"),
            "Avg. Temp": st.column_config.TextColumn(width="small"),
        },
    )

    selected_rows: list[int] = []
    selected_cells: list[Any] = []
    if table_event is not None:
        try:
            selected_rows = list(table_event.selection.rows)
        except Exception:
            if isinstance(table_event, dict):
                selection_payload = table_event.get("selection", {})
                selected_rows = list(selection_payload.get("rows", []))
        try:
            selected_cells = list(table_event.selection.cells)
        except Exception:
            if isinstance(table_event, dict):
                selection_payload = table_event.get("selection", {})
                selected_cells = list(selection_payload.get("cells", []))

    selected_index: int | None = None
    if selected_rows:
        try:
            parsed_row_index = int(selected_rows[0])
            if 0 <= parsed_row_index < len(source_frame):
                selected_index = parsed_row_index
        except (TypeError, ValueError):
            selected_index = None
    elif selected_cells:
        first_cell = selected_cells[0]
        parsed_row_index: int | None = None
        if isinstance(first_cell, (tuple, list)) and first_cell:
            try:
                parsed_row_index = int(first_cell[0])
            except (TypeError, ValueError):
                parsed_row_index = None
        elif isinstance(first_cell, dict):
            try:
                parsed_row_index = int(first_cell.get("row"))
            except (TypeError, ValueError):
                parsed_row_index = None
        if parsed_row_index is not None and 0 <= parsed_row_index < len(source_frame):
            selected_index = parsed_row_index

    if selected_index is None:
        return

    selected_offset = normalize_weather_forecast_day_offset(
        source_frame.at[selected_index, "day_offset"],
        max_offset=astronomy_forecast_nights - 1,
    )
    selection_token = f"{selected_index}:{selected_offset}"
    last_selection_token = str(st.session_state.get("astronomy_forecast_last_selection_token", ""))
    if selection_token == last_selection_token:
        return

    st.session_state["astronomy_forecast_last_selection_token"] = selection_token
    if selected_offset != selected_day_offset:
        st.session_state[weather_forecast_day_offset_state_key] = selected_offset
        st.session_state[weather_forecast_period_state_key] = (
            weather_forecast_period_tonight if selected_offset <= 0 else weather_forecast_period_tomorrow
        )
        st.rerun()
