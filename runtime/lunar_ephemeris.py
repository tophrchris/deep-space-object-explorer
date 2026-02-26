from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import astropy.units as u
import numpy as np
import pandas as pd
import streamlit as st
from astral import moon as astral_moon
from astropy.coordinates import AltAz, EarthLocation, get_body
from astropy.time import Time

from app_constants import WIND16

_LUNAR_PHASE_ANCHORS: dict[str, tuple[float, str]] = {
    "new": (0.0, "New Moon"),
    "first_quarter": (7.38, "First Quarter"),
    "full": (14.77, "Full Moon"),
    "third_quarter": (22.15, "Third Quarter"),
}
_SYNODIC_MONTH_DAYS = 29.53


def _empty_moon_track() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["time_local", "alt", "az", "wind16", "min_alt_required", "visible"]
    )


def _parse_local_timestamp(value: str, tz_name: str) -> pd.Timestamp | None:
    try:
        parsed = pd.Timestamp(value)
    except Exception:
        return None

    try:
        tzinfo = ZoneInfo(str(tz_name or "UTC"))
    except Exception:
        tzinfo = ZoneInfo("UTC")

    try:
        if parsed.tzinfo is None:
            return parsed.tz_localize(tzinfo)
        return parsed.tz_convert(tzinfo)
    except Exception:
        return None


def _az_to_wind16(azimuth_deg: float) -> str:
    try:
        azimuth = float(azimuth_deg) % 360.0
    except (TypeError, ValueError):
        return "N"
    idx = int(((azimuth + 11.25) // 22.5) % 16)
    return WIND16[idx]


def _phase_key_from_age_days(age_days: float) -> tuple[str, str]:
    try:
        age = float(age_days)
    except (TypeError, ValueError):
        return "new", _LUNAR_PHASE_ANCHORS["new"][1]
    if not np.isfinite(age):
        return "new", _LUNAR_PHASE_ANCHORS["new"][1]

    best_key = "new"
    best_distance = float("inf")
    for phase_key, (anchor, _label) in _LUNAR_PHASE_ANCHORS.items():
        distance = abs(age - anchor)
        distance = min(distance, _SYNODIC_MONTH_DAYS - distance)
        if distance < best_distance:
            best_distance = distance
            best_key = phase_key
    return best_key, _LUNAR_PHASE_ANCHORS[best_key][1]


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def compute_moon_track(
    *,
    lat: float,
    lon: float,
    tz_name: str,
    start_local_iso: str,
    end_local_iso: str,
    sample_minutes: int = 10,
) -> pd.DataFrame:
    start_local = _parse_local_timestamp(start_local_iso, tz_name)
    end_local = _parse_local_timestamp(end_local_iso, tz_name)
    if start_local is None or end_local is None:
        return _empty_moon_track()
    if end_local < start_local:
        return _empty_moon_track()

    cadence_minutes = max(1, int(sample_minutes))
    times_local = pd.date_range(
        start=start_local,
        end=end_local,
        freq=f"{cadence_minutes}min",
        inclusive="both",
    )
    if times_local.empty:
        return _empty_moon_track()

    try:
        location = EarthLocation(lat=float(lat) * u.deg, lon=float(lon) * u.deg)
    except Exception:
        return _empty_moon_track()

    times_utc = times_local.tz_convert("UTC")
    astropy_times = Time(times_utc.to_pydatetime())

    try:
        moon_icrs = get_body("moon", astropy_times, location=location)
        altaz_frame = AltAz(obstime=astropy_times, location=location)
        moon_altaz = moon_icrs.transform_to(altaz_frame)
    except Exception:
        return _empty_moon_track()

    altitudes = np.asarray(moon_altaz.alt.deg, dtype=float)
    azimuths = np.asarray(moon_altaz.az.deg % 360.0, dtype=float)
    visible = altitudes >= 0.0
    winds = [_az_to_wind16(value) for value in azimuths.tolist()]

    return pd.DataFrame(
        {
            "time_local": times_local,
            "alt": altitudes,
            "az": azimuths,
            "wind16": winds,
            "min_alt_required": np.zeros(len(times_local), dtype=float),
            "visible": visible,
        }
    )


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def build_hourly_lunar_altitude_map(
    *,
    lat: float,
    lon: float,
    tz_name: str,
    start_local_iso: str,
    end_local_iso: str,
    sample_minutes: int = 10,
) -> dict[str, float]:
    moon_track = compute_moon_track(
        lat=lat,
        lon=lon,
        tz_name=tz_name,
        start_local_iso=start_local_iso,
        end_local_iso=end_local_iso,
        sample_minutes=sample_minutes,
    )
    if moon_track.empty:
        return {}

    working = moon_track.copy()
    time_values = pd.to_datetime(working["time_local"], errors="coerce")
    altitude_values = pd.to_numeric(working["alt"], errors="coerce")
    valid_mask = time_values.notna() & altitude_values.notna()
    if not bool(valid_mask.any()):
        return {}

    working = working.loc[valid_mask].copy()
    working["time_local"] = pd.to_datetime(working["time_local"], errors="coerce")
    working["alt"] = pd.to_numeric(working["alt"], errors="coerce")
    working = working.dropna(subset=["time_local", "alt"])
    if working.empty:
        return {}

    working["hour_local"] = working["time_local"].dt.floor("h")
    grouped = working.groupby("hour_local", sort=True)["alt"].mean()
    result: dict[str, float] = {}
    for hour_ts, avg_alt in grouped.items():
        try:
            numeric = float(avg_alt)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            continue
        result[pd.Timestamp(hour_ts).isoformat()] = numeric
    return result


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 30)
def compute_lunar_phase_for_night(
    *,
    tz_name: str,
    start_local_iso: str,
    end_local_iso: str,
) -> dict[str, str | float]:
    start_local = _parse_local_timestamp(start_local_iso, tz_name)
    end_local = _parse_local_timestamp(end_local_iso, tz_name)
    if start_local is None or end_local is None:
        return {"phase_key": "new", "phase_label": "New Moon", "phase_age_days": 0.0}
    if end_local < start_local:
        start_local, end_local = end_local, start_local

    midpoint = start_local + (end_local - start_local) / 2
    try:
        phase_age_days = float(astral_moon.phase(midpoint.date()))
    except Exception:
        phase_age_days = 0.0
    phase_key, phase_label = _phase_key_from_age_days(phase_age_days)
    return {
        "phase_key": phase_key,
        "phase_label": phase_label,
        "phase_age_days": phase_age_days,
    }


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def compute_lunar_night_summary(
    *,
    lat: float,
    lon: float,
    tz_name: str,
    start_local_iso: str,
    end_local_iso: str,
    sample_minutes: int = 10,
) -> dict[str, int | float | None]:
    hourly_altitude_map = build_hourly_lunar_altitude_map(
        lat=lat,
        lon=lon,
        tz_name=tz_name,
        start_local_iso=start_local_iso,
        end_local_iso=end_local_iso,
        sample_minutes=sample_minutes,
    )
    if not hourly_altitude_map:
        return {
            "hour_count": 0,
            "moon_below_horizon_hours": 0,
            "moon_below_horizon_pct": None,
            "moon_avg_altitude_deg": None,
            "moon_max_altitude_deg": None,
        }

    finite_values = [
        float(value)
        for value in hourly_altitude_map.values()
        if value is not None and np.isfinite(float(value))
    ]
    if not finite_values:
        return {
            "hour_count": 0,
            "moon_below_horizon_hours": 0,
            "moon_below_horizon_pct": None,
            "moon_avg_altitude_deg": None,
            "moon_max_altitude_deg": None,
        }

    hour_count = int(len(finite_values))
    below_horizon_hours = int(sum(1 for value in finite_values if float(value) < 0.0))
    pct = (float(below_horizon_hours) / float(hour_count)) * 100.0 if hour_count > 0 else None
    avg_altitude: float | None = None
    max_altitude: float | None = None

    # For Lunar-column coloring, use the exact astronomical-dark window samples
    # (not the per-hour bucket averages) so partial edge hours are weighted
    # correctly.
    moon_track = compute_moon_track(
        lat=lat,
        lon=lon,
        tz_name=tz_name,
        start_local_iso=start_local_iso,
        end_local_iso=end_local_iso,
        sample_minutes=sample_minutes,
    )
    if isinstance(moon_track, pd.DataFrame) and not moon_track.empty:
        track_times = pd.to_datetime(moon_track.get("time_local"), errors="coerce")
        track_altitudes = pd.to_numeric(moon_track.get("alt"), errors="coerce")
        sample_mask = track_times.notna() & track_altitudes.notna()
        start_local = _parse_local_timestamp(start_local_iso, tz_name)
        end_local = _parse_local_timestamp(end_local_iso, tz_name)
        if start_local is not None and end_local is not None:
            if end_local < start_local:
                start_local, end_local = end_local, start_local
            sample_mask &= (track_times >= start_local) & (track_times <= end_local)
        if bool(sample_mask.any()):
            sample_values = track_altitudes.loc[sample_mask].astype(float).to_numpy(dtype=float)
            finite_sample_values = sample_values[np.isfinite(sample_values)]
            if finite_sample_values.size > 0:
                avg_altitude = float(np.mean(finite_sample_values))
                max_altitude = float(np.max(finite_sample_values))

    if avg_altitude is None and hour_count > 0:
        avg_altitude = float(np.mean(finite_values))
    if max_altitude is None and hour_count > 0:
        max_altitude = float(np.max(finite_values))
    return {
        "hour_count": hour_count,
        "moon_below_horizon_hours": below_horizon_hours,
        "moon_below_horizon_pct": pct,
        "moon_avg_altitude_deg": avg_altitude,
        "moon_max_altitude_deg": max_altitude,
    }
