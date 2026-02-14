from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT_SECONDS = 12
DEFAULT_HOURLY_FIELDS = ("temperature_2m",)
EXTENDED_FORECAST_HOURLY_FIELDS = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation_probability",
    "rain",
    "showers",
    "snowfall",
    "cloud_cover",
    "wind_gusts_10m",
)
FAHRENHEIT_COUNTRY_CODES = {"US", "BS", "BZ", "KY", "PW", "FM", "MH", "LR"}
KMH_TO_MPH = 0.621371
MM_TO_IN = 0.0393701
CM_TO_IN = 0.393701


def infer_temperature_unit_from_locale(locale_value: str | None) -> str:
    if not locale_value:
        return "f"

    normalized = str(locale_value).replace("-", "_")
    parts = [part for part in normalized.split("_") if part]
    country = ""
    if len(parts) >= 2:
        candidate = parts[-1].upper()
        if len(candidate) == 2 and candidate.isalpha():
            country = candidate

    if country in FAHRENHEIT_COUNTRY_CODES:
        return "f"
    return "c"


def resolve_temperature_unit(preference: str, locale_value: str | None) -> str:
    pref = str(preference).strip().lower()
    if pref == "f":
        return "f"
    if pref == "c":
        return "c"
    return infer_temperature_unit_from_locale(locale_value)


def format_temperature(temp_celsius: float | None, unit: str) -> str:
    if temp_celsius is None or pd.isna(temp_celsius):
        return "-"

    if unit == "f":
        converted = (float(temp_celsius) * 9.0 / 5.0) + 32.0
        return f"{converted:.0f} F"
    return f"{float(temp_celsius):.0f} C"


def format_wind_speed(speed_kmh: float | None, temperature_unit: str) -> str:
    if speed_kmh is None or pd.isna(speed_kmh):
        return "-"

    numeric = float(speed_kmh)
    if str(temperature_unit).strip().lower() == "f":
        return f"{(numeric * KMH_TO_MPH):.0f} mph"
    return f"{numeric:.0f} km/h"


def format_precipitation(depth_mm: float | None, temperature_unit: str) -> str:
    if depth_mm is None or pd.isna(depth_mm):
        return "-"

    numeric = float(depth_mm)
    if str(temperature_unit).strip().lower() == "f":
        return f"{(numeric * MM_TO_IN):.2f} in"
    return f"{numeric:.1f} mm"


def format_snowfall(depth_cm: float | None, temperature_unit: str) -> str:
    if depth_cm is None or pd.isna(depth_cm):
        return "-"

    numeric = float(depth_cm)
    if str(temperature_unit).strip().lower() == "f":
        return f"{(numeric * CM_TO_IN):.2f} in"
    return f"{numeric:.1f} cm"


def _normalize_hourly_fields(hourly_fields: Iterable[str] | None) -> tuple[str, ...]:
    if hourly_fields is None:
        return DEFAULT_HOURLY_FIELDS

    normalized: list[str] = []
    for field in hourly_fields:
        cleaned = str(field).strip()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized) if normalized else DEFAULT_HOURLY_FIELDS


def _parse_local_timestamp(value: Any, tz_name: str) -> pd.Timestamp | None:
    try:
        parsed = pd.Timestamp(value)
    except Exception:
        return None

    try:
        tzinfo = ZoneInfo(tz_name)
        if parsed.tzinfo is None:
            return parsed.tz_localize(tzinfo)
        return parsed.tz_convert(tzinfo)
    except Exception:
        return None


def _coerce_numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False, ttl=15 * 60)
def fetch_hourly_weather(
    lat: float,
    lon: float,
    tz_name: str,
    start_local_iso: str,
    end_local_iso: str,
    hourly_fields: tuple[str, ...] = DEFAULT_HOURLY_FIELDS,
) -> list[dict[str, Any]]:
    fields = _normalize_hourly_fields(hourly_fields)
    start_local = pd.Timestamp(start_local_iso)
    end_local = pd.Timestamp(end_local_iso)

    try:
        tzinfo = ZoneInfo(tz_name)
    except Exception:
        tzinfo = ZoneInfo("UTC")

    # Ensure the API response includes both sides of overnight windows:
    # - `past_days` captures prior-day sunset hours after midnight
    # - `forecast_days` captures next-morning hours before sunrise
    if start_local.tzinfo is None:
        start_local_in_tz = start_local.tz_localize(tzinfo)
    else:
        start_local_in_tz = start_local.tz_convert(tzinfo)
    if end_local.tzinfo is None:
        end_local_in_tz = end_local.tz_localize(tzinfo)
    else:
        end_local_in_tz = end_local.tz_convert(tzinfo)

    today_local = datetime.now(tzinfo).date()
    start_date = start_local_in_tz.date()
    end_date = end_local_in_tz.date()
    past_days = max(0, (today_local - start_date).days)
    forecast_days = max(1, (end_date - today_local).days + 1)

    try:
        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(fields),
                "timezone": tz_name,
                "past_days": past_days,
                "forecast_days": forecast_days,
            },
            timeout=OPEN_METEO_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not isinstance(times, list) or not times:
        return []

    values_by_field: dict[str, list[Any]] = {}
    for field in fields:
        raw_values = hourly.get(field, [])
        values_by_field[field] = raw_values if isinstance(raw_values, list) else []

    rows: list[dict[str, Any]] = []
    for idx, raw_time in enumerate(times):
        parsed = _parse_local_timestamp(raw_time, tz_name=tz_name)
        if parsed is None or parsed < start_local or parsed > end_local:
            continue

        row: dict[str, Any] = {"time_iso": parsed.floor("h").isoformat()}
        for field in fields:
            field_values = values_by_field.get(field, [])
            raw_value = field_values[idx] if idx < len(field_values) else None
            row[field] = _coerce_numeric(raw_value)
        rows.append(row)

    return rows


def fetch_hourly_temperatures(
    lat: float, lon: float, tz_name: str, start_local_iso: str, end_local_iso: str
) -> dict[str, float]:
    rows = fetch_hourly_weather(
        lat=lat,
        lon=lon,
        tz_name=tz_name,
        start_local_iso=start_local_iso,
        end_local_iso=end_local_iso,
        hourly_fields=EXTENDED_FORECAST_HOURLY_FIELDS,
    )

    mapping: dict[str, float] = {}
    for row in rows:
        time_iso = str(row.get("time_iso", "")).strip()
        temperature = row.get("temperature_2m")
        if not time_iso or temperature is None or pd.isna(temperature):
            continue
        mapping[time_iso] = float(temperature)
    return mapping
