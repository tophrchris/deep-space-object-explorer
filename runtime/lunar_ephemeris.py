from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import astropy.units as u
import numpy as np
import pandas as pd
import streamlit as st
from astral import moon as astral_moon
from astropy.coordinates import AltAz, EarthLocation, get_body
from astropy.time import Time

from app_constants import WIND16
from runtime.lunar_eclipse_catalog import LUNAR_UMBRAL_ECLIPSE_EVENTS

_LUNAR_PHASE_ANCHORS: dict[str, tuple[float, str]] = {
    "new": (0.0, "New Moon"),
    "first_quarter": (7.38, "First Quarter"),
    "full": (14.77, "Full Moon"),
    "third_quarter": (22.15, "Third Quarter"),
}
_SYNODIC_MONTH_DAYS = 29.53


def _build_umbral_eclipse_intervals_utc() -> tuple[
    tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, str, float, float | None], ...
]:
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, str, float, float | None]] = []
    for greatest_utc_iso, eclipse_kind, partial_minutes, total_minutes in LUNAR_UMBRAL_ECLIPSE_EVENTS:
        try:
            greatest_utc = pd.Timestamp(greatest_utc_iso)
        except Exception:
            continue
        if greatest_utc.tzinfo is None:
            greatest_utc = greatest_utc.tz_localize("UTC")
        else:
            greatest_utc = greatest_utc.tz_convert("UTC")

        try:
            partial_duration_minutes = float(partial_minutes)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(partial_duration_minutes) or partial_duration_minutes <= 0.0:
            continue

        half_duration = pd.Timedelta(minutes=(partial_duration_minutes / 2.0))
        interval_start_utc = greatest_utc - half_duration
        interval_end_utc = greatest_utc + half_duration
        intervals.append(
            (
                interval_start_utc,
                interval_end_utc,
                greatest_utc,
                str(eclipse_kind or "").strip().upper()[:1] or "P",
                float(partial_duration_minutes),
                (float(total_minutes) if total_minutes is not None else None),
            )
        )
    return tuple(intervals)


_UMBRAL_ECLIPSE_INTERVALS_UTC = _build_umbral_eclipse_intervals_utc()


def _normalize_wind16_obstructions(obstructions: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(obstructions, dict):
        return None
    normalized: dict[str, float] = {}
    for direction in WIND16:
        raw_value = obstructions.get(direction, 20.0)
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            numeric = 20.0
        if not np.isfinite(numeric):
            numeric = 20.0
        normalized[direction] = float(max(0.0, min(90.0, numeric)))
    return normalized


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


def _empty_lunar_eclipse_visibility() -> dict[str, Any]:
    return {
        "has_visible_eclipse": False,
        "has_visible_total_eclipse": False,
        "has_obstructed_eclipse_phase": False,
        "visible_event_kinds": [],
        "hourly_visible_minutes_by_hour": {},
        "visible_minutes_total": 0.0,
        "events": [],
    }


def _segment_visible_fraction(alt_start: float, alt_end: float) -> float:
    if (not np.isfinite(alt_start)) or (not np.isfinite(alt_end)):
        return 0.0
    if alt_start >= 0.0 and alt_end >= 0.0:
        return 1.0
    if alt_start < 0.0 and alt_end < 0.0:
        return 0.0

    delta = alt_end - alt_start
    if abs(delta) < 1e-9:
        return 1.0 if alt_start >= 0.0 else 0.0

    crossing_fraction = (-alt_start) / delta
    crossing_fraction = float(max(0.0, min(1.0, crossing_fraction)))
    if alt_start < 0.0 <= alt_end:
        return 1.0 - crossing_fraction
    return crossing_fraction


def _interpolate_alt_az_at_time(track_frame: pd.DataFrame, target_time: pd.Timestamp) -> tuple[float, float] | None:
    if not isinstance(track_frame, pd.DataFrame) or track_frame.empty:
        return None
    times = pd.to_datetime(track_frame.get("time_local"), errors="coerce")
    altitudes = pd.to_numeric(track_frame.get("alt"), errors="coerce")
    azimuths = pd.to_numeric(track_frame.get("az"), errors="coerce")
    valid_mask = times.notna() & altitudes.notna() & azimuths.notna()
    if not bool(valid_mask.any()):
        return None
    working = pd.DataFrame(
        {
            "time_local": times.loc[valid_mask],
            "alt": altitudes.loc[valid_mask].astype(float),
            "az": azimuths.loc[valid_mask].astype(float),
        }
    ).sort_values("time_local")
    if working.empty:
        return None

    target = pd.Timestamp(target_time)
    first_time = pd.Timestamp(working["time_local"].iloc[0])
    last_time = pd.Timestamp(working["time_local"].iloc[-1])
    if target <= first_time:
        return float(working["alt"].iloc[0]), float(working["az"].iloc[0] % 360.0)
    if target >= last_time:
        return float(working["alt"].iloc[-1]), float(working["az"].iloc[-1] % 360.0)

    left_candidates = working.loc[working["time_local"] <= target]
    right_candidates = working.loc[working["time_local"] >= target]
    if left_candidates.empty or right_candidates.empty:
        return None

    left = left_candidates.iloc[-1]
    right = right_candidates.iloc[0]
    t_left = pd.Timestamp(left["time_local"])
    t_right = pd.Timestamp(right["time_local"])
    if t_right <= t_left:
        return float(left["alt"]), float(float(left["az"]) % 360.0)

    total_seconds = (t_right - t_left).total_seconds()
    if total_seconds <= 0.0:
        return float(left["alt"]), float(float(left["az"]) % 360.0)
    frac = float((target - t_left).total_seconds() / total_seconds)
    frac = max(0.0, min(1.0, frac))

    alt_left = float(left["alt"])
    alt_right = float(right["alt"])
    interpolated_alt = alt_left + ((alt_right - alt_left) * frac)

    az_left = float(left["az"]) % 360.0
    az_right = float(right["az"]) % 360.0
    az_delta = ((az_right - az_left + 540.0) % 360.0) - 180.0
    interpolated_az = (az_left + (az_delta * frac)) % 360.0
    return float(interpolated_alt), float(interpolated_az)


def _visible_minutes_within_interval(track_frame: pd.DataFrame, interval_start: pd.Timestamp, interval_end: pd.Timestamp) -> float:
    if interval_end <= interval_start:
        return 0.0
    if not isinstance(track_frame, pd.DataFrame) or track_frame.empty:
        return 0.0
    times = pd.to_datetime(track_frame.get("time_local"), errors="coerce")
    altitudes = pd.to_numeric(track_frame.get("alt"), errors="coerce")
    min_required_source = track_frame.get("min_alt_required")
    if min_required_source is None:
        min_required = pd.Series(np.zeros(len(track_frame), dtype=float), index=track_frame.index, dtype=float)
    else:
        min_required = pd.to_numeric(min_required_source, errors="coerce").fillna(0.0)
    valid_mask = times.notna() & altitudes.notna() & min_required.notna()
    if not bool(valid_mask.any()):
        return 0.0
    working = pd.DataFrame(
        {
            "time_local": times.loc[valid_mask],
            "alt": altitudes.loc[valid_mask].astype(float),
            "min_alt_required": min_required.loc[valid_mask].astype(float),
        }
    ).sort_values("time_local")
    if len(working) < 2:
        return 0.0

    visible_minutes = 0.0
    time_values = working["time_local"].tolist()
    altitude_values = working["alt"].tolist()
    required_values = working["min_alt_required"].tolist()
    for idx in range(len(time_values) - 1):
        segment_start = pd.Timestamp(time_values[idx])
        segment_end = pd.Timestamp(time_values[idx + 1])
        if segment_end <= interval_start or segment_start >= interval_end:
            continue
        clipped_start = max(segment_start, interval_start)
        clipped_end = min(segment_end, interval_end)
        if clipped_end <= clipped_start:
            continue
        clipped_minutes = (clipped_end - clipped_start).total_seconds() / 60.0
        if clipped_minutes <= 0.0:
            continue
        clearance_start = float(altitude_values[idx]) - max(0.0, float(required_values[idx]))
        clearance_end = float(altitude_values[idx + 1]) - max(0.0, float(required_values[idx + 1]))
        visible_fraction = _segment_visible_fraction(clearance_start, clearance_end)
        if visible_fraction <= 0.0:
            continue
        visible_minutes += clipped_minutes * visible_fraction
    return float(max(0.0, visible_minutes))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def compute_moon_track(
    *,
    lat: float,
    lon: float,
    tz_name: str,
    start_local_iso: str,
    end_local_iso: str,
    sample_minutes: int = 10,
    obstructions: dict[str, float] | None = None,
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
    winds = [_az_to_wind16(value) for value in azimuths.tolist()]
    normalized_obstructions = _normalize_wind16_obstructions(obstructions)
    if normalized_obstructions is None:
        min_required = np.zeros(len(times_local), dtype=float)
    else:
        min_required = np.asarray(
            [float(normalized_obstructions.get(direction, 20.0)) for direction in winds],
            dtype=float,
        )
    visible = (altitudes >= 0.0) & (altitudes >= min_required)

    return pd.DataFrame(
        {
            "time_local": times_local,
            "alt": altitudes,
            "az": azimuths,
            "wind16": winds,
            "min_alt_required": min_required,
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


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 30)
def compute_lunar_eclipse_visibility_for_night(
    *,
    lat: float,
    lon: float,
    tz_name: str,
    start_local_iso: str,
    end_local_iso: str,
    sample_minutes: int = 1,
    obstructions: dict[str, float] | None = None,
) -> dict[str, Any]:
    start_local = _parse_local_timestamp(start_local_iso, tz_name)
    end_local = _parse_local_timestamp(end_local_iso, tz_name)
    if start_local is None or end_local is None:
        return _empty_lunar_eclipse_visibility()
    if end_local < start_local:
        start_local, end_local = end_local, start_local

    if not _UMBRAL_ECLIPSE_INTERVALS_UTC:
        return _empty_lunar_eclipse_visibility()
    normalized_obstructions = _normalize_wind16_obstructions(obstructions)

    try:
        tzinfo = ZoneInfo(str(tz_name or "UTC"))
    except Exception:
        tzinfo = ZoneInfo("UTC")

    overlapping_intervals: list[
        tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, str, float, float | None]
    ] = []
    for (
        interval_start_utc,
        interval_end_utc,
        greatest_utc,
        eclipse_kind,
        partial_minutes,
        total_minutes,
    ) in _UMBRAL_ECLIPSE_INTERVALS_UTC:
        interval_start_local = pd.Timestamp(interval_start_utc).tz_convert(tzinfo)
        interval_end_local = pd.Timestamp(interval_end_utc).tz_convert(tzinfo)
        overlap_start = max(start_local, interval_start_local)
        overlap_end = min(end_local, interval_end_local)
        if overlap_end <= overlap_start:
            continue
        overlapping_intervals.append(
            (
                interval_start_local,
                interval_end_local,
                pd.Timestamp(greatest_utc).tz_convert(tzinfo),
                eclipse_kind,
                float(partial_minutes),
                total_minutes,
            )
        )

    if not overlapping_intervals:
        return _empty_lunar_eclipse_visibility()

    cadence_minutes = max(1, int(sample_minutes))
    moon_track_start = min(item[0] for item in overlapping_intervals) - pd.Timedelta(minutes=cadence_minutes)
    moon_track_end = max(item[1] for item in overlapping_intervals) + pd.Timedelta(minutes=cadence_minutes)
    moon_track = compute_moon_track(
        lat=lat,
        lon=lon,
        tz_name=tz_name,
        start_local_iso=moon_track_start.isoformat(),
        end_local_iso=moon_track_end.isoformat(),
        sample_minutes=cadence_minutes,
        obstructions=normalized_obstructions,
    )
    if moon_track.empty:
        return _empty_lunar_eclipse_visibility()

    track_times = pd.to_datetime(moon_track.get("time_local"), errors="coerce")
    track_altitudes = pd.to_numeric(moon_track.get("alt"), errors="coerce")
    track_azimuths = pd.to_numeric(moon_track.get("az"), errors="coerce")
    track_required = pd.to_numeric(moon_track.get("min_alt_required"), errors="coerce").fillna(0.0)
    valid_mask = track_times.notna() & track_altitudes.notna() & track_azimuths.notna() & track_required.notna()
    if not bool(valid_mask.any()):
        return _empty_lunar_eclipse_visibility()

    track_frame = pd.DataFrame(
        {
            "time_local": track_times.loc[valid_mask],
            "alt": track_altitudes.loc[valid_mask],
            "az": track_azimuths.loc[valid_mask],
            "min_alt_required": track_required.loc[valid_mask],
        }
    ).sort_values("time_local")
    if len(track_frame) < 2:
        return _empty_lunar_eclipse_visibility()

    time_values = track_frame["time_local"].tolist()
    altitude_values = track_frame["alt"].astype(float).tolist()
    required_values = track_frame["min_alt_required"].astype(float).tolist()
    hourly_visible_minutes_by_hour: dict[str, float] = {}
    visible_event_kinds: set[str] = set()
    visible_minutes_total = 0.0
    visible_total_eclipse = False
    has_obstructed_eclipse_phase = False
    event_payloads: list[dict[str, Any]] = []

    for interval_start, interval_end, greatest_time, eclipse_kind, partial_minutes, total_minutes in overlapping_intervals:
        interval_visible_minutes = 0.0

        overlap_start = max(start_local, interval_start)
        overlap_end = min(end_local, interval_end)
        if overlap_end <= overlap_start:
            continue

        for idx in range(len(time_values) - 1):
            segment_start = pd.Timestamp(time_values[idx])
            segment_end = pd.Timestamp(time_values[idx + 1])
            if segment_end <= overlap_start or segment_start >= overlap_end:
                continue

            clipped_start = max(segment_start, overlap_start)
            clipped_end = min(segment_end, overlap_end)
            if clipped_end <= clipped_start:
                continue

            clipped_minutes = (clipped_end - clipped_start).total_seconds() / 60.0
            if clipped_minutes <= 0.0:
                continue

            clearance_start = float(altitude_values[idx]) - max(0.0, float(required_values[idx]))
            clearance_end = float(altitude_values[idx + 1]) - max(0.0, float(required_values[idx + 1]))
            visible_fraction = _segment_visible_fraction(clearance_start, clearance_end)
            if visible_fraction <= 0.0:
                continue

            clipped_visible_minutes = clipped_minutes * visible_fraction
            if clipped_visible_minutes <= 0.0:
                continue

            bucket_cursor = pd.Timestamp(clipped_start)
            while bucket_cursor < clipped_end:
                hour_start = bucket_cursor.floor("h")
                hour_end = hour_start + pd.Timedelta(hours=1)
                bucket_end = min(clipped_end, hour_end)
                if bucket_end <= bucket_cursor:
                    break
                bucket_minutes = (bucket_end - bucket_cursor).total_seconds() / 60.0
                if bucket_minutes > 0.0:
                    hour_key = hour_start.isoformat()
                    hourly_visible_minutes_by_hour[hour_key] = (
                        float(hourly_visible_minutes_by_hour.get(hour_key, 0.0)) + (bucket_minutes * visible_fraction)
                    )
                bucket_cursor = bucket_end

            interval_visible_minutes += clipped_visible_minutes
            visible_minutes_total += clipped_visible_minutes

        phase_entries: list[dict[str, Any]] = []
        partial_start = interval_start
        partial_end = interval_end
        partial_duration = max(0.0, float(partial_minutes))
        partial_midpoint = partial_start + (partial_end - partial_start) / 2
        partial_pos = _interpolate_alt_az_at_time(track_frame, partial_midpoint)
        partial_az = float(partial_pos[1]) if partial_pos is not None else None
        partial_visible_minutes = _visible_minutes_within_interval(track_frame, partial_start, partial_end)
        partial_obstructed = (partial_visible_minutes + 0.1) < partial_duration
        partial_fully_obstructed = partial_visible_minutes <= 0.1
        phase_entries.append(
            {
                "name": "partial",
                "start_local_iso": partial_start.isoformat(),
                "end_local_iso": partial_end.isoformat(),
                "duration_minutes": partial_duration,
                "visible_minutes": float(max(0.0, partial_visible_minutes)),
                "obstructed": bool(partial_obstructed),
                "fully_obstructed": bool(partial_fully_obstructed),
                "azimuth_deg": partial_az,
                "azimuth_dir": (_az_to_wind16(partial_az) if partial_az is not None else ""),
            }
        )

        if total_minutes is not None:
            total_duration = max(0.0, float(total_minutes))
            total_half = pd.Timedelta(minutes=(total_duration / 2.0))
            total_start = greatest_time - total_half
            total_end = greatest_time + total_half
            total_midpoint = total_start + (total_end - total_start) / 2
            total_pos = _interpolate_alt_az_at_time(track_frame, total_midpoint)
            total_az = float(total_pos[1]) if total_pos is not None else None
            total_visible_minutes = _visible_minutes_within_interval(track_frame, total_start, total_end)
            total_obstructed = (total_visible_minutes + 0.1) < total_duration
            total_fully_obstructed = total_visible_minutes <= 0.1
            phase_entries.append(
                {
                    "name": "total",
                    "start_local_iso": total_start.isoformat(),
                    "end_local_iso": total_end.isoformat(),
                    "duration_minutes": total_duration,
                    "visible_minutes": float(max(0.0, total_visible_minutes)),
                    "obstructed": bool(total_obstructed),
                    "fully_obstructed": bool(total_fully_obstructed),
                    "azimuth_deg": total_az,
                    "azimuth_dir": (_az_to_wind16(total_az) if total_az is not None else ""),
                }
            )

        event_has_obstructed_phase = any(bool(item.get("obstructed", False)) for item in phase_entries)
        event_has_visible_phase = any(float(item.get("visible_minutes", 0.0)) > 0.1 for item in phase_entries)
        has_obstructed_eclipse_phase = has_obstructed_eclipse_phase or event_has_obstructed_phase
        event_payloads.append(
            {
                "kind": str(eclipse_kind or "P").strip().upper()[:1] or "P",
                "greatest_time_local_iso": pd.Timestamp(greatest_time).isoformat(),
                "phases": phase_entries,
                "has_obstructed_phase": bool(event_has_obstructed_phase),
                "has_visible_phase": bool(event_has_visible_phase),
            }
        )

        if interval_visible_minutes > 0.0:
            visible_event_kinds.add(str(eclipse_kind or "P").strip().upper()[:1] or "P")
            if str(eclipse_kind or "").strip().upper().startswith("T") and total_minutes is not None:
                visible_total_eclipse = True

    cleaned_hourly_visible_minutes: dict[str, float] = {}
    for hour_key, raw_minutes in hourly_visible_minutes_by_hour.items():
        try:
            numeric_minutes = float(raw_minutes)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric_minutes) or numeric_minutes <= 0.0:
            continue
        cleaned_hourly_visible_minutes[str(hour_key)] = numeric_minutes

    if not cleaned_hourly_visible_minutes:
        return _empty_lunar_eclipse_visibility()

    ordered_visible_kinds = sorted(
        visible_event_kinds,
        key=lambda kind: 0 if kind == "T" else 1,
    )
    return {
        "has_visible_eclipse": True,
        "has_visible_total_eclipse": bool(visible_total_eclipse),
        "has_obstructed_eclipse_phase": bool(has_obstructed_eclipse_phase),
        "visible_event_kinds": ordered_visible_kinds,
        "hourly_visible_minutes_by_hour": cleaned_hourly_visible_minutes,
        "visible_minutes_total": float(max(0.0, visible_minutes_total)),
        "events": event_payloads,
    }
