from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app_constants import WIND16
from app_theme import resolve_plot_theme_colors
from features.explorer.lunar import lunar_phase_emoji
from runtime.weather_service import format_temperature

_ECLIPSE_ACCENT_HEX = "#7A1E2C"
_ECLIPSE_ACCENT_SHADOW = "rgba(122, 30, 44, 0.44)"


@lru_cache(maxsize=None)
def _ui_name(name: str) -> Any:
    # Lazily resolve remaining UI-owned helpers/constants to avoid circular import
    # failures while the Streamlit monolith is still being decomposed.
    from ui import streamlit_app as ui_app

    return getattr(ui_app, name)


def _interpolate_cloud_cover_color(*args: Any, **kwargs: Any):
    return _ui_name("_interpolate_cloud_cover_color")(*args, **kwargs)


def _interpolate_temperature_color_f(*args: Any, **kwargs: Any):
    return _ui_name("_interpolate_temperature_color_f")(*args, **kwargs)


def _muted_rgba_from_hex(*args: Any, **kwargs: Any):
    return _ui_name("_muted_rgba_from_hex")(*args, **kwargs)


def format_display_time(*args: Any, **kwargs: Any):
    return _ui_name("format_display_time")(*args, **kwargs)


def format_hour_label(*args: Any, **kwargs: Any):
    return _ui_name("format_hour_label")(*args, **kwargs)


def mount_warning_zone_altitude_bounds(*args: Any, **kwargs: Any):
    return _ui_name("mount_warning_zone_altitude_bounds")(*args, **kwargs)


def mount_warning_zone_plot_style(*args: Any, **kwargs: Any):
    return _ui_name("mount_warning_zone_plot_style")(*args, **kwargs)


def normalize_12_hour_label(*args: Any, **kwargs: Any):
    return _ui_name("normalize_12_hour_label")(*args, **kwargs)


def normalize_hour_key(*args: Any, **kwargs: Any):
    return _ui_name("normalize_hour_key")(*args, **kwargs)


def resolve_weather_alert_indicator(*args: Any, **kwargs: Any):
    return _ui_name("resolve_weather_alert_indicator")(*args, **kwargs)

def build_path_hovertext(
    target_label: str,
    emission_details: str,
    time_values: np.ndarray,
    altitude_values: np.ndarray | None = None,
) -> np.ndarray:
    emissions = str(emission_details or "").strip()
    emissions_line = f"<br>Emissions: {emissions}" if emissions else ""
    total = int(len(time_values))

    if altitude_values is None:
        altitude = np.full(total, np.nan, dtype=float)
    else:
        altitude = np.asarray(altitude_values, dtype=float)
        if altitude.shape[0] != total:
            aligned = np.full(total, np.nan, dtype=float)
            copy_count = min(total, int(altitude.shape[0]))
            if copy_count > 0:
                aligned[:copy_count] = altitude[:copy_count]
            altitude = aligned

    hover_values: list[str] = []
    for idx, value in enumerate(time_values):
        time_text = str(value).strip()
        if time_text:
            altitude_line = ""
            altitude_value = float(altitude[idx])
            if np.isfinite(altitude_value):
                altitude_line = f"<br>Altitude: {altitude_value:.1f} deg"
            hover_values.append(f"{target_label}{emissions_line}<br>Time: {time_text}{altitude_line}")
        else:
            hover_values.append("")
    return np.asarray(hover_values, dtype=object)


def _moon_culmination_point(moon_track: pd.DataFrame) -> dict[str, Any] | None:
    if not isinstance(moon_track, pd.DataFrame) or moon_track.empty:
        return None

    moon_times_source = moon_track["time_local"] if "time_local" in moon_track.columns else pd.Series(index=moon_track.index, dtype=object)
    moon_alt_source = moon_track["alt"] if "alt" in moon_track.columns else pd.Series(index=moon_track.index, dtype=float)
    moon_az_source = moon_track["az"] if "az" in moon_track.columns else pd.Series(np.nan, index=moon_track.index, dtype=float)
    moon_times = pd.to_datetime(moon_times_source, errors="coerce")
    moon_altitudes = pd.to_numeric(moon_alt_source, errors="coerce")
    moon_azimuths = pd.to_numeric(moon_az_source, errors="coerce")
    valid_mask = moon_times.notna() & moon_altitudes.notna()
    valid_mask &= moon_azimuths.notna()
    if not bool(valid_mask.any()):
        return None

    moon_times = moon_times.loc[valid_mask]
    moon_altitudes = moon_altitudes.loc[valid_mask].astype(float)
    moon_azimuths = moon_azimuths.loc[valid_mask].astype(float)

    altitude_values = moon_altitudes.to_numpy(dtype=float)
    finite_mask = np.isfinite(altitude_values)
    if not bool(finite_mask.any()):
        return None

    moon_times = moon_times.iloc[finite_mask]
    moon_altitudes = moon_altitudes.iloc[finite_mask]
    moon_azimuths = moon_azimuths.iloc[finite_mask]
    if moon_times.empty or moon_altitudes.empty:
        return None

    max_idx = int(np.argmax(moon_altitudes.to_numpy(dtype=float)))
    culmination_time = pd.Timestamp(moon_times.iloc[max_idx])
    culmination_alt = float(moon_altitudes.iloc[max_idx])
    culmination_az = float(moon_azimuths.iloc[max_idx]) if not moon_azimuths.empty else float("nan")
    if not np.isfinite(culmination_alt):
        return None
    return {
        "time_local": culmination_time,
        "alt": culmination_alt,
        "az": culmination_az,
    }


def _moon_culmination_hovertext(
    culmination_point: dict[str, Any],
    *,
    use_12_hour: bool,
) -> str:
    culmination_time = pd.Timestamp(culmination_point["time_local"])
    time_text = format_display_time(culmination_time, use_12_hour=use_12_hour)
    altitude_text = f"{float(culmination_point['alt']):.1f} deg"
    return f"Moon<br>Culmination at {time_text}<br>Altitude: {altitude_text}"


def _interpolate_moon_track_alt_az(moon_track: pd.DataFrame, target_time: pd.Timestamp) -> tuple[float, float] | None:
    if not isinstance(moon_track, pd.DataFrame) or moon_track.empty:
        return None
    times = pd.to_datetime(moon_track.get("time_local"), errors="coerce")
    altitudes = pd.to_numeric(moon_track.get("alt"), errors="coerce")
    azimuths = pd.to_numeric(moon_track.get("az"), errors="coerce")
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

    time_ns = pd.to_datetime(working["time_local"]).astype("int64").to_numpy(dtype=np.int64)
    altitude_values = working["alt"].to_numpy(dtype=float)
    azimuth_values = np.mod(working["az"].to_numpy(dtype=float), 360.0)
    if time_ns.size < 1 or altitude_values.size < 1 or azimuth_values.size < 1:
        return None

    target_timestamp = pd.Timestamp(target_time)
    target_ns = int(target_timestamp.value)
    if target_ns < int(time_ns[0]) or target_ns > int(time_ns[-1]):
        return None

    right_idx = int(np.searchsorted(time_ns, target_ns, side="left"))
    left_idx = max(0, right_idx - 1)
    if right_idx >= len(time_ns):
        return float(altitude_values[-1]), float(azimuth_values[-1])

    left_ns = int(time_ns[left_idx])
    right_ns = int(time_ns[right_idx])
    if right_ns <= left_ns:
        return float(altitude_values[left_idx]), float(azimuth_values[left_idx])

    fraction = float((target_ns - left_ns) / float(right_ns - left_ns))
    fraction = max(0.0, min(1.0, fraction))
    alt_left = float(altitude_values[left_idx])
    alt_right = float(altitude_values[right_idx])
    interpolated_alt = alt_left + ((alt_right - alt_left) * fraction)

    az_left = float(azimuth_values[left_idx]) % 360.0
    az_right = float(azimuth_values[right_idx]) % 360.0
    az_delta = ((az_right - az_left + 540.0) % 360.0) - 180.0
    interpolated_az = (az_left + (az_delta * fraction)) % 360.0
    return float(interpolated_alt), float(interpolated_az)


def _eclipse_phase_plot_points(
    moon_track: pd.DataFrame | None,
    eclipse_visibility: dict[str, Any] | None,
    *,
    use_12_hour: bool,
) -> list[dict[str, Any]]:
    if not isinstance(moon_track, pd.DataFrame) or moon_track.empty:
        return []
    payload = eclipse_visibility if isinstance(eclipse_visibility, dict) else {}
    raw_events = payload.get("events", [])
    if not isinstance(raw_events, list) or not raw_events:
        return []

    points: list[dict[str, Any]] = []
    for event in raw_events:
        if not isinstance(event, dict):
            continue
        raw_phases = event.get("phases", [])
        if not isinstance(raw_phases, list):
            continue
        for phase in raw_phases:
            if not isinstance(phase, dict):
                continue
            phase_name = str(phase.get("name", "")).strip().lower()
            if phase_name not in {"partial", "total"}:
                continue
            phase_prefix = "T" if phase_name == "total" else "P"
            phase_label = "Total" if phase_name == "total" else "Partial"
            for boundary in ("start", "end"):
                phase_time = pd.to_datetime(phase.get(f"{boundary}_local_iso"), errors="coerce")
                if phase_time is None or pd.isna(phase_time):
                    continue
                interpolated = _interpolate_moon_track_alt_az(moon_track, pd.Timestamp(phase_time))
                if interpolated is None:
                    continue
                altitude_deg, azimuth_deg = interpolated
                if not np.isfinite(altitude_deg) or not np.isfinite(azimuth_deg):
                    continue
                boundary_title = "Start" if boundary == "start" else "End"
                time_label = format_display_time(pd.Timestamp(phase_time), use_12_hour=use_12_hour)
                points.append(
                    {
                        "time_local": pd.Timestamp(phase_time),
                        "alt": float(altitude_deg),
                        "az": float(azimuth_deg),
                        "text": f"{phase_prefix} {boundary.lower()}",
                        "hover": (
                            f"{phase_label} {boundary_title}<br>{time_label}<br>"
                            f"Azimuth: {float(azimuth_deg):.1f} deg<br>"
                            f"Altitude: {float(altitude_deg):.1f} deg"
                        ),
                    }
                )

    points.sort(key=lambda item: pd.Timestamp(item["time_local"]))
    return points


def _moon_path_colors(eclipse_visibility: dict[str, Any] | None) -> tuple[str, str]:
    payload = eclipse_visibility if isinstance(eclipse_visibility, dict) else {}
    if bool(payload.get("has_visible_eclipse", False)):
        return _ECLIPSE_ACCENT_SHADOW, _ECLIPSE_ACCENT_HEX
    return "rgba(0, 0, 0, 0.45)", "#ffffff"


def split_path_on_az_wrap(track: pd.DataFrame, use_12_hour: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if track.empty:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=object)

    az_values = track["az"].to_numpy(dtype=float)
    alt_values = track["alt"].to_numpy(dtype=float)
    time_values = [
        format_display_time(pd.Timestamp(value), use_12_hour=use_12_hour) for value in track["time_local"].tolist()
    ]

    x_values: list[float] = []
    y_values: list[float] = []
    hover_times: list[str] = []

    for idx, azimuth in enumerate(az_values):
        if idx > 0 and abs(float(azimuth) - float(az_values[idx - 1])) > 180.0:
            x_values.append(np.nan)
            y_values.append(np.nan)
            hover_times.append("")

        x_values.append(float(azimuth))
        y_values.append(float(alt_values[idx]))
        hover_times.append(time_values[idx])

    return (
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        np.asarray(hover_times, dtype=object),
    )


def iter_labeled_events(events: dict[str, pd.Series | None]) -> list[tuple[str, pd.Series]]:
    event_labels = _ui_name("EVENT_LABELS")
    suppress_culmination = False
    first_visible = events.get("first_visible")
    culmination = events.get("culmination")
    if first_visible is not None and culmination is not None:
        try:
            first_visible_time = pd.Timestamp(first_visible["time_local"])
            culmination_time = pd.Timestamp(culmination["time_local"])
            suppress_culmination = abs(culmination_time - first_visible_time) <= pd.Timedelta(minutes=15)
        except Exception:
            suppress_culmination = False

    labeled: list[tuple[str, pd.Series]] = []
    for event_key, event_label in event_labels:
        event = events.get(event_key)
        if event is None:
            continue
        if event_key == "culmination" and suppress_culmination:
            continue
        labeled.append((event_label, event))
    return labeled


def sample_direction_indices(length: int, max_markers: int = 6) -> list[int]:
    if length < 3:
        return []
    step = max(1, length // (max_markers + 1))
    return list(range(step, length - 1, step))


def direction_marker_segments_cartesian(
    track: pd.DataFrame, max_markers: int
) -> list[tuple[float, float, float, float]]:
    if track.empty:
        return []

    az_values = track["az"].to_numpy(dtype=float)
    alt_values = track["alt"].to_numpy(dtype=float)
    indices = [idx for idx in sample_direction_indices(len(track), max_markers=max_markers) if idx < len(track) - 1]

    segments: list[tuple[float, float, float, float]] = []
    for idx in indices:
        next_idx = idx + 1
        az_start = float(az_values[idx])
        az_end = float(az_values[next_idx])
        alt_start = float(alt_values[idx])
        alt_end = float(alt_values[next_idx])
        if abs(az_end - az_start) > 180.0:
            # Cartesian path view splits across azimuth wrap, so skip wrap-crossing marker segments.
            continue
        if abs(az_end - az_start) < 1e-9 and abs(alt_end - alt_start) < 1e-9:
            continue
        segments.append((az_start, alt_start, az_end, alt_end))
    return segments


def direction_marker_segments_radial(
    track: pd.DataFrame, radial_values: np.ndarray, max_markers: int
) -> list[tuple[float, float, float, float]]:
    if track.empty:
        return []

    theta_values = track["az"].to_numpy(dtype=float)
    r_values = np.asarray(radial_values, dtype=float)
    indices = [idx for idx in sample_direction_indices(len(track), max_markers=max_markers) if idx < len(track) - 1]

    segments: list[tuple[float, float, float, float]] = []
    for idx in indices:
        next_idx = idx + 1
        theta_start = float(theta_values[idx])
        theta_end = float(theta_values[next_idx])
        r_start = float(r_values[idx])
        r_end = float(r_values[next_idx])
        if abs(theta_end - theta_start) > 180.0:
            continue
        if abs(theta_end - theta_start) < 1e-9 and abs(r_end - r_start) < 1e-9:
            continue
        segments.append((theta_start, r_start, theta_end, r_end))
    return segments


def track_event_index(track: pd.DataFrame, event: pd.Series | None) -> int | None:
    if event is None or track.empty:
        return None
    try:
        position = track.index.get_loc(event.name)
    except Exception:
        return None

    if isinstance(position, slice):
        return int(position.start) if position.start is not None else None
    if isinstance(position, np.ndarray):
        if position.size == 0:
            return None
        return int(position[0])
    try:
        return int(position)
    except Exception:
        return None


def endpoint_marker_segments_cartesian(
    track: pd.DataFrame, events: dict[str, pd.Series | None]
) -> tuple[tuple[float, float, float, float] | None, tuple[float, float, float, float] | None]:
    rise = events.get("rise")
    set_event = events.get("set")
    rise_index = track_event_index(track, rise)
    set_index = track_event_index(track, set_event)

    rise_segment: tuple[float, float, float, float] | None = None
    if rise is not None and rise_index is not None:
        neighbor = rise_index + 1 if rise_index + 1 < len(track) else None
        if neighbor is not None:
            az_rise = float(track.iloc[rise_index]["az"])
            alt_rise = float(track.iloc[rise_index]["alt"])
            az_next = float(track.iloc[neighbor]["az"])
            alt_next = float(track.iloc[neighbor]["alt"])
            if abs(az_next - az_rise) <= 180.0 and (
                abs(az_next - az_rise) >= 1e-9 or abs(alt_next - alt_rise) >= 1e-9
            ):
                # Reverse points so the tail marker is drawn at the rise location.
                rise_segment = (az_next, alt_next, az_rise, alt_rise)

    set_segment: tuple[float, float, float, float] | None = None
    if set_event is not None and set_index is not None:
        neighbor = set_index - 1 if set_index - 1 >= 0 else None
        if neighbor is not None:
            az_prev = float(track.iloc[neighbor]["az"])
            alt_prev = float(track.iloc[neighbor]["alt"])
            az_set = float(track.iloc[set_index]["az"])
            alt_set = float(track.iloc[set_index]["alt"])
            if abs(az_set - az_prev) <= 180.0 and (
                abs(az_set - az_prev) >= 1e-9 or abs(alt_set - alt_prev) >= 1e-9
            ):
                set_segment = (az_prev, alt_prev, az_set, alt_set)

    return rise_segment, set_segment


def endpoint_marker_segments_radial(
    track: pd.DataFrame,
    events: dict[str, pd.Series | None],
    radial_values: np.ndarray,
) -> tuple[tuple[float, float, float, float] | None, tuple[float, float, float, float] | None]:
    rise = events.get("rise")
    set_event = events.get("set")
    rise_index = track_event_index(track, rise)
    set_index = track_event_index(track, set_event)

    rise_segment: tuple[float, float, float, float] | None = None
    if rise is not None and rise_index is not None and rise_index + 1 < len(track):
        neighbor = rise_index + 1
        theta_rise = float(track.iloc[rise_index]["az"])
        r_rise = float(radial_values[rise_index])
        theta_next = float(track.iloc[neighbor]["az"])
        r_next = float(radial_values[neighbor])
        if abs(theta_next - theta_rise) <= 180.0 and (
            abs(theta_next - theta_rise) >= 1e-9 or abs(r_next - r_rise) >= 1e-9
        ):
            rise_segment = (theta_next, r_next, theta_rise, r_rise)

    set_segment: tuple[float, float, float, float] | None = None
    if set_event is not None and set_index is not None and set_index - 1 >= 0:
        neighbor = set_index - 1
        theta_prev = float(track.iloc[neighbor]["az"])
        r_prev = float(radial_values[neighbor])
        theta_set = float(track.iloc[set_index]["az"])
        r_set = float(radial_values[set_index])
        if abs(theta_set - theta_prev) <= 180.0 and (
            abs(theta_set - theta_prev) >= 1e-9 or abs(r_set - r_prev) >= 1e-9
        ):
            set_segment = (theta_prev, r_prev, theta_set, r_set)

    return rise_segment, set_segment


def terminal_segment_from_path_arrays(
    x_values: np.ndarray | pd.Series,
    y_values: np.ndarray | pd.Series,
) -> tuple[float, float, float, float] | None:
    x_numeric = np.asarray(x_values, dtype=float)
    y_numeric = np.asarray(y_values, dtype=float)
    if x_numeric.size < 2 or y_numeric.size < 2:
        return None

    finite_mask = np.isfinite(x_numeric) & np.isfinite(y_numeric)
    finite_indices = np.flatnonzero(finite_mask)
    if finite_indices.size < 2:
        return None

    for idx in range(finite_indices.size - 1, 0, -1):
        prev_idx = int(finite_indices[idx - 1])
        curr_idx = int(finite_indices[idx])
        if curr_idx - prev_idx != 1:
            continue
        x0 = float(x_numeric[prev_idx])
        y0 = float(y_numeric[prev_idx])
        x1 = float(x_numeric[curr_idx])
        y1 = float(y_numeric[curr_idx])
        if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9:
            continue
        return (x0, y0, x1, y1)
    return None


def obstruction_step_profile(obstructions: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    # Hard-floor profile aligned to 16-wind bin boundaries.
    boundaries = [0.0] + [11.25 + (22.5 * idx) for idx in range(16)] + [360.0]
    segment_dirs = ["N"] + WIND16[1:] + ["N"]
    segment_alts = [float(obstructions.get(direction, 20.0)) for direction in segment_dirs]

    x_values: list[float] = []
    y_values: list[float] = []
    for idx, altitude in enumerate(segment_alts):
        left = float(boundaries[idx])
        right = float(boundaries[idx + 1])
        x_values.extend([left, right])
        y_values.extend([altitude, altitude])
        if idx < len(segment_alts) - 1:
            next_altitude = float(segment_alts[idx + 1])
            x_values.append(right)
            y_values.append(next_altitude)

    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def visible_track_segments(track: pd.DataFrame) -> list[pd.DataFrame]:
    if track.empty or "visible" not in track.columns:
        return []

    visible_mask = track["visible"].fillna(False).astype(bool).to_numpy()
    if not visible_mask.any():
        return []

    segments: list[pd.DataFrame] = []
    start_idx: int | None = None
    for idx, is_visible in enumerate(visible_mask):
        if is_visible and start_idx is None:
            start_idx = idx
            continue
        if (not is_visible) and start_idx is not None:
            segment = track.iloc[start_idx:idx].copy()
            if not segment.empty:
                segments.append(segment)
            start_idx = None
    if start_idx is not None:
        segment = track.iloc[start_idx:].copy()
        if not segment.empty:
            segments.append(segment)
    return segments


def _track_start_timestamp(track: pd.DataFrame | None) -> pd.Timestamp | None:
    if not isinstance(track, pd.DataFrame) or track.empty:
        return None
    times = pd.to_datetime(track.get("time_local"), errors="coerce")
    if not isinstance(times, pd.Series):
        return None
    valid_times = times.dropna()
    if valid_times.empty:
        return None
    return pd.Timestamp(valid_times.iloc[0])


def _build_sequence_numbers(start_entries: list[tuple[str, pd.Timestamp | None]]) -> dict[str, int]:
    normalized_entries: list[tuple[str, pd.Timestamp, int]] = []
    for index, (entry_key, entry_start) in enumerate(start_entries):
        if entry_start is None:
            continue
        try:
            normalized_entries.append((str(entry_key), pd.Timestamp(entry_start), index))
        except Exception:
            continue
    normalized_entries.sort(key=lambda item: (item[1], item[2]))
    return {entry_key: sequence_idx + 1 for sequence_idx, (entry_key, _entry_start, _entry_index) in enumerate(normalized_entries)}


def _first_cartesian_track_point(track: pd.DataFrame | None) -> tuple[float, float] | None:
    if not isinstance(track, pd.DataFrame) or track.empty:
        return None
    azimuth = pd.to_numeric(track.get("az"), errors="coerce")
    altitude = pd.to_numeric(track.get("alt"), errors="coerce")
    if not isinstance(azimuth, pd.Series) or not isinstance(altitude, pd.Series):
        return None
    valid_mask = azimuth.notna() & altitude.notna()
    if not bool(valid_mask.any()):
        return None
    first_index = valid_mask[valid_mask].index[0]
    return float(azimuth.loc[first_index]), float(altitude.loc[first_index])


def _first_radial_track_point(track: pd.DataFrame | None, *, dome_view: bool) -> tuple[float, float] | None:
    point = _first_cartesian_track_point(track)
    if point is None:
        return None
    azimuth_value, altitude_value = point
    clipped_altitude = max(0.0, min(90.0, altitude_value))
    radial_value = (90.0 - clipped_altitude) if dome_view else clipped_altitude
    return float(azimuth_value), float(radial_value)


def _coerce_schedule_window(value: Any) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start_ts = pd.Timestamp(value[0])
        end_ts = pd.Timestamp(value[1])
    except Exception:
        return None
    if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
        return None
    return pd.Timestamp(start_ts), pd.Timestamp(end_ts)


def _clip_track_to_schedule_window(
    track: pd.DataFrame,
    schedule_window: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> pd.DataFrame:
    if not isinstance(track, pd.DataFrame) or track.empty:
        return pd.DataFrame()
    normalized_window = _coerce_schedule_window(schedule_window)
    if normalized_window is None or "time_local" not in track.columns:
        return pd.DataFrame()

    schedule_start, schedule_end = normalized_window
    times = pd.to_datetime(track["time_local"], errors="coerce")
    if times.empty:
        return pd.DataFrame()

    tz_info = times.dt.tz
    if tz_info is not None:
        if schedule_start.tzinfo is None:
            schedule_start = schedule_start.tz_localize(tz_info)
        else:
            schedule_start = schedule_start.tz_convert(tz_info)
        if schedule_end.tzinfo is None:
            schedule_end = schedule_end.tz_localize(tz_info)
        else:
            schedule_end = schedule_end.tz_convert(tz_info)
    else:
        if schedule_start.tzinfo is not None:
            schedule_start = schedule_start.tz_localize(None)
        if schedule_end.tzinfo is not None:
            schedule_end = schedule_end.tz_localize(None)

    mask = times.notna() & (times >= schedule_start) & (times <= schedule_end)
    if not bool(mask.any()):
        return pd.DataFrame(columns=track.columns)
    clipped = track.loc[mask].copy()
    return clipped if not clipped.empty else pd.DataFrame(columns=track.columns)


def distribute_non_overlapping_values(
    values: list[float],
    *,
    lower: float,
    upper: float,
    min_gap: float,
) -> list[float]:
    if not values:
        return []

    numeric_values = np.asarray(values, dtype=float)
    total = int(numeric_values.size)
    if total <= 1:
        clipped_single = float(np.clip(numeric_values[0], lower, upper))
        return [clipped_single]

    bounded_lower = float(min(lower, upper))
    bounded_upper = float(max(lower, upper))
    span = max(0.0, bounded_upper - bounded_lower)
    effective_gap = min(max(0.0, float(min_gap)), span / float(total - 1))

    sorted_indices = np.argsort(numeric_values)
    sorted_values = np.clip(numeric_values[sorted_indices], bounded_lower, bounded_upper).astype(float)
    adjusted = sorted_values.copy()

    for idx in range(1, total):
        adjusted[idx] = max(adjusted[idx], adjusted[idx - 1] + effective_gap)

    overflow = adjusted[-1] - bounded_upper
    if overflow > 0:
        adjusted -= overflow

    for idx in range(total - 2, -1, -1):
        adjusted[idx] = min(adjusted[idx], adjusted[idx + 1] - effective_gap)

    underflow = bounded_lower - adjusted[0]
    if underflow > 0:
        adjusted += underflow

    adjusted = np.clip(adjusted, bounded_lower, bounded_upper)
    unsorted_adjusted = np.empty_like(adjusted)
    unsorted_adjusted[sorted_indices] = adjusted
    return [float(value) for value in unsorted_adjusted]


def build_unobstructed_altitude_area_plot(
    target_tracks: list[dict[str, Any]],
    *,
    use_12_hour: bool,
    temperature_by_hour: dict[str, float] | None = None,
    weather_by_hour: dict[str, dict[str, Any]] | None = None,
    temperature_unit: str = "f",
    mount_choice: str = "none",
    moon_track: pd.DataFrame | None = None,
    moon_phase_key: str | None = None,
    eclipse_visibility: dict[str, Any] | None = None,
) -> go.Figure:
    unobstructed_area_constant_obstruction_alt_deg = float(
        _ui_name("UNOBSTRUCTED_AREA_CONSTANT_OBSTRUCTION_ALT_DEG")
    )
    object_type_group_color_default = str(_ui_name("OBJECT_TYPE_GROUP_COLOR_DEFAULT"))
    path_line_width_overlay_default = float(_ui_name("PATH_LINE_WIDTH_OVERLAY_DEFAULT"))
    path_line_width_primary_default = float(_ui_name("PATH_LINE_WIDTH_PRIMARY_DEFAULT"))
    theme_colors = resolve_plot_theme_colors()
    fig = go.Figure()
    plotted_any = False
    plotted_times: list[pd.Timestamp] = []
    obstruction_ceiling = max(0.0, min(90.0, unobstructed_area_constant_obstruction_alt_deg))

    if obstruction_ceiling > 0.0:
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="y",
            x0=0.0,
            x1=1.0,
            y0=0.0,
            y1=obstruction_ceiling,
            fillcolor=theme_colors["obstruction_fill"],
            line={"width": 0},
            layer="below",
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="y",
            x0=0.0,
            x1=1.0,
            y0=obstruction_ceiling,
            y1=obstruction_ceiling,
            line={"width": 1, "color": theme_colors["obstruction_line"]},
            layer="below",
        )
    mount_warning_zone = mount_warning_zone_altitude_bounds(mount_choice)
    if mount_warning_zone is not None:
        warning_y0, warning_y1, _warning_label = mount_warning_zone
        warning_fill, warning_line = mount_warning_zone_plot_style()
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="y",
            x0=0.0,
            x1=1.0,
            y0=warning_y0,
            y1=warning_y1,
            fillcolor=warning_fill,
            line={"width": 0},
            layer="below",
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="y",
            x0=0.0,
            x1=1.0,
            y0=warning_y1 if warning_y0 <= 0.0 else warning_y0,
            y1=warning_y1 if warning_y0 <= 0.0 else warning_y0,
            line={"width": 1, "color": warning_line, "dash": "dot"},
            layer="below",
        )

    non_selected_tracks = [target for target in target_tracks if not bool(target.get("is_selected", False))]
    selected_tracks = [target for target in target_tracks if bool(target.get("is_selected", False))]
    ordered_tracks = [*non_selected_tracks, *selected_tracks]
    sequence_entries: list[tuple[str, pd.Timestamp | None]] = []
    for target_index, target_track in enumerate(ordered_tracks):
        sequence_entries.append((f"target:{target_index}", _track_start_timestamp(target_track.get("track"))))
    sequence_numbers = _build_sequence_numbers(sequence_entries)

    for target_index, target_track in enumerate(ordered_tracks):
        track = target_track.get("track")
        if not isinstance(track, pd.DataFrame) or track.empty:
            continue

        is_selected = bool(target_track.get("is_selected", False))
        target_label = str(target_track.get("label", "List target")).strip() or "List target"
        target_color = (
            str(target_track.get("color", object_type_group_color_default)).strip() or object_type_group_color_default
        )
        base_line_width = float(target_track.get("line_width", path_line_width_overlay_default))
        target_line_width = (
            max(base_line_width, path_line_width_primary_default + 1.2)
            if is_selected
            else max(1.6, min(base_line_width, path_line_width_overlay_default))
        )
        schedule_line_width = target_line_width
        target_line_color = target_color if is_selected else _muted_rgba_from_hex(target_color, alpha=0.68)
        target_emissions = str(target_track.get("emission_lines_display") or "").strip()
        target_fill_color = _muted_rgba_from_hex(target_color, alpha=(0.30 if is_selected else 0.10))
        target_schedule_window = _coerce_schedule_window(target_track.get("schedule_window"))
        target_sequence_number = sequence_numbers.get(f"target:{target_index}")
        target_sequence_drawn = False

        for segment in visible_track_segments(track):
            segment_times = pd.to_datetime(segment["time_local"], errors="coerce")
            segment_altitudes = pd.to_numeric(segment["alt"], errors="coerce")
            if segment_times.empty or segment_altitudes.empty:
                continue

            finite_mask = segment_times.notna() & segment_altitudes.notna()
            if not bool(finite_mask.any()):
                continue

            segment_times = segment_times.loc[finite_mask]
            segment_altitudes = segment_altitudes.loc[finite_mask].astype(float)
            altitude_values = segment_altitudes.to_numpy(dtype=float)
            finite_altitude_mask = np.isfinite(altitude_values)
            if not bool(finite_altitude_mask.any()):
                continue

            segment_times = segment_times.iloc[finite_altitude_mask]
            altitude_values = altitude_values[finite_altitude_mask]
            if segment_times.empty or altitude_values.size == 0:
                continue

            plotted_times.extend([pd.Timestamp(value) for value in segment_times.tolist()])

            hover_times = np.asarray(
                [format_display_time(pd.Timestamp(value), use_12_hour=use_12_hour) for value in segment_times.tolist()],
                dtype=object,
            )
            hover_text = build_path_hovertext(target_label, target_emissions, hover_times, altitude_values)
            fig.add_trace(
                go.Scatter(
                    x=segment_times,
                    y=altitude_values,
                    mode="lines",
                    showlegend=False,
                    line={"width": target_line_width, "color": target_line_color},
                    fill="tozeroy",
                    fillcolor=target_fill_color,
                    hovertext=hover_text,
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )
            if (target_sequence_number is not None) and (not target_sequence_drawn):
                fig.add_trace(
                    go.Scatter(
                        x=[pd.Timestamp(segment_times.iloc[0])],
                        y=[float(altitude_values[0])],
                        mode="markers+text",
                        text=[str(target_sequence_number)],
                        textposition="middle center",
                        textfont={"size": 10, "color": "#FFFFFF"},
                        marker={
                            "size": 16,
                            "color": target_line_color,
                            "line": {"width": 1, "color": "#FFFFFF"},
                        },
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                target_sequence_drawn = True
            if target_schedule_window is not None:
                scheduled_segment = _clip_track_to_schedule_window(segment, target_schedule_window)
                if not scheduled_segment.empty:
                    scheduled_times = pd.to_datetime(scheduled_segment["time_local"], errors="coerce")
                    scheduled_altitudes = pd.to_numeric(scheduled_segment["alt"], errors="coerce")
                    scheduled_mask = scheduled_times.notna() & scheduled_altitudes.notna()
                    if bool(scheduled_mask.any()):
                        scheduled_times = scheduled_times.loc[scheduled_mask]
                        scheduled_altitudes = scheduled_altitudes.loc[scheduled_mask].astype(float)
                        scheduled_alt_values = scheduled_altitudes.to_numpy(dtype=float)
                        scheduled_alt_mask = np.isfinite(scheduled_alt_values)
                        if bool(scheduled_alt_mask.any()):
                            scheduled_times = scheduled_times.iloc[scheduled_alt_mask]
                            scheduled_alt_values = scheduled_alt_values[scheduled_alt_mask]
                            if not scheduled_times.empty and scheduled_alt_values.size > 0:
                                scheduled_hover_times = np.asarray(
                                    [
                                        format_display_time(pd.Timestamp(value), use_12_hour=use_12_hour)
                                        for value in scheduled_times.tolist()
                                    ],
                                    dtype=object,
                                )
                                scheduled_hover = build_path_hovertext(
                                    target_label,
                                    target_emissions,
                                    scheduled_hover_times,
                                    scheduled_alt_values,
                                )
                                fig.add_trace(
                                    go.Scatter(
                                        x=scheduled_times,
                                        y=scheduled_alt_values,
                                        mode="lines",
                                        showlegend=False,
                                        line={"width": schedule_line_width, "color": target_line_color},
                                        hovertext=scheduled_hover,
                                        hovertemplate="%{hovertext}<extra></extra>",
                                    )
                                )
            plotted_any = True

    moon_culmination_point = _moon_culmination_point(moon_track) if isinstance(moon_track, pd.DataFrame) else None
    if isinstance(moon_track, pd.DataFrame) and not moon_track.empty:
        moon_times = pd.to_datetime(moon_track.get("time_local"), errors="coerce")
        moon_altitudes = pd.to_numeric(moon_track.get("alt"), errors="coerce")
        if moon_times is not None and moon_altitudes is not None:
            moon_valid_mask = moon_times.notna() & moon_altitudes.notna()
            if bool(moon_valid_mask.any()):
                moon_times = moon_times.loc[moon_valid_mask]
                moon_altitudes = moon_altitudes.loc[moon_valid_mask].astype(float)
                moon_altitude_values = moon_altitudes.to_numpy(dtype=float)
                moon_finite_mask = np.isfinite(moon_altitude_values)
                if bool(moon_finite_mask.any()):
                    moon_times = moon_times.iloc[moon_finite_mask]
                    moon_altitude_values = moon_altitude_values[moon_finite_mask]
                    if not moon_times.empty and moon_altitude_values.size > 0:
                        moon_shadow_color, moon_line_color = _moon_path_colors(eclipse_visibility)
                        plotted_times.extend([pd.Timestamp(value) for value in moon_times.tolist()])
                        moon_hover_times = np.asarray(
                            [
                                format_display_time(pd.Timestamp(value), use_12_hour=use_12_hour)
                                for value in moon_times.tolist()
                            ],
                            dtype=object,
                        )
                        moon_hover = build_path_hovertext("Moon", "", moon_hover_times, moon_altitude_values)
                        fig.add_trace(
                            go.Scatter(
                                x=moon_times,
                                y=moon_altitude_values,
                                mode="lines",
                                showlegend=False,
                                line={"width": 3.0, "color": moon_shadow_color},
                                hoverinfo="skip",
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=moon_times,
                                y=moon_altitude_values,
                                mode="lines",
                                name="Moon",
                                showlegend=False,
                                line={"width": 1.8, "color": moon_line_color},
                                hovertext=moon_hover,
                                hovertemplate="%{hovertext}<extra></extra>",
                            )
                        )
                        eclipse_phase_points = _eclipse_phase_plot_points(
                            moon_track,
                            eclipse_visibility,
                            use_12_hour=use_12_hour,
                        )
                        if eclipse_phase_points:
                            phase_text_positions = [
                                ("top center" if (idx % 2 == 0) else "bottom center")
                                for idx in range(len(eclipse_phase_points))
                            ]
                            fig.add_trace(
                                go.Scatter(
                                    x=[pd.Timestamp(point["time_local"]) for point in eclipse_phase_points],
                                    y=[float(point["alt"]) for point in eclipse_phase_points],
                                    mode="markers+text",
                                    text=[str(point["text"]) for point in eclipse_phase_points],
                                    textposition=phase_text_positions,
                                    textfont={"size": 10, "color": _ECLIPSE_ACCENT_HEX},
                                    marker={
                                        "size": 7,
                                        "color": _ECLIPSE_ACCENT_HEX,
                                        "line": {"width": 1, "color": "#FFFFFF"},
                                    },
                                    showlegend=False,
                                    hovertext=[str(point["hover"]) for point in eclipse_phase_points],
                                    hovertemplate="%{hovertext}<extra></extra>",
                                )
                            )
                        if moon_culmination_point is not None:
                            moon_emoji = lunar_phase_emoji(moon_phase_key)
                            moon_culmination_hover = _moon_culmination_hovertext(
                                moon_culmination_point,
                                use_12_hour=use_12_hour,
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=[pd.Timestamp(moon_culmination_point["time_local"])],
                                    y=[float(moon_culmination_point["alt"])],
                                    mode="text",
                                    text=[moon_emoji],
                                    textposition="middle center",
                                    textfont={"size": 18},
                                    showlegend=False,
                                    hovertext=[moon_culmination_hover],
                                    hovertemplate="%{hovertext}<extra></extra>",
                                )
                            )

    title = "Unobstructed Altitude Coverage"
    if not plotted_any:
        fig.add_annotation(
            text="No unobstructed intervals for these targets tonight.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": theme_colors["muted_text"]},
        )

    hourly_points: list[pd.Timestamp] = []
    if temperature_by_hour:
        for hour_key in temperature_by_hour.keys():
            try:
                hourly_points.append(pd.Timestamp(hour_key))
            except Exception:
                continue
    if weather_by_hour:
        for hour_key in weather_by_hour.keys():
            try:
                hourly_points.append(pd.Timestamp(hour_key))
            except Exception:
                continue

    if hourly_points:
        unique_hour_points = sorted({pd.Timestamp(value).floor("h") for value in hourly_points})
        min_plot_time = min(plotted_times) if plotted_times else None
        max_plot_time = max(plotted_times) if plotted_times else None

        temp_x: list[pd.Timestamp] = []
        temp_y: list[float] = []
        temp_text: list[str] = []
        temp_colors: list[str] = []
        temp_hover: list[str] = []

        alert_x: list[pd.Timestamp] = []
        alert_y: list[float] = []
        alert_text: list[str] = []
        alert_hover: list[str] = []

        for hour_timestamp in unique_hour_points:
            if min_plot_time is not None and hour_timestamp < (min_plot_time.floor("h") - pd.Timedelta(hours=1)):
                continue
            if max_plot_time is not None and hour_timestamp > (max_plot_time.ceil("h") + pd.Timedelta(hours=1)):
                continue

            hour_key = normalize_hour_key(hour_timestamp) or hour_timestamp.isoformat()
            temp_value = None if not temperature_by_hour else temperature_by_hour.get(hour_key)
            if temp_value is not None and not pd.isna(temp_value):
                temp_x.append(hour_timestamp)
                temp_y.append(-4.5)
                temp_text.append(format_temperature(temp_value, temperature_unit))
                temp_f = (float(temp_value) * 9.0 / 5.0) + 32.0
                temp_colors.append(_interpolate_temperature_color_f(temp_f))
                temp_hover.append(
                    f"{format_display_time(hour_timestamp, use_12_hour=use_12_hour)}<br>"
                    f"Temperature: {format_temperature(temp_value, temperature_unit)}"
                )

            weather_row = {} if not weather_by_hour else weather_by_hour.get(hour_key, {})
            emoji, tooltip_text = resolve_weather_alert_indicator(weather_row, temperature_unit)
            if emoji:
                alert_x.append(hour_timestamp)
                alert_y.append(-11.0)
                alert_text.append(emoji)
                if tooltip_text:
                    alert_hover.append(
                        f"{format_display_time(hour_timestamp, use_12_hour=use_12_hour)}<br>{tooltip_text}"
                    )
                else:
                    alert_hover.append(format_display_time(hour_timestamp, use_12_hour=use_12_hour))

        if temp_text:
            fig.add_trace(
                go.Scatter(
                    x=temp_x,
                    y=temp_y,
                    mode="text",
                    text=temp_text,
                    textposition="middle center",
                    textfont={"color": temp_colors},
                    showlegend=False,
                    hovertext=temp_hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )

        if alert_text:
            fig.add_trace(
                go.Scatter(
                    x=alert_x,
                    y=alert_y,
                    mode="text",
                    text=alert_text,
                    textposition="middle center",
                    showlegend=False,
                    hovertext=alert_hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        height=360,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        showlegend=False,
        plot_bgcolor=theme_colors["plot_bg"],
        paper_bgcolor=theme_colors["paper_bg"],
        font={"color": theme_colors["text"]},
        xaxis_title="Time",
        yaxis_title="Altitude (deg)",
    )

    x_axis_settings: dict[str, Any] = {
        "type": "date",
        "showgrid": True,
        "gridcolor": theme_colors["grid"],
        "gridwidth": 1,
        "dtick": 60 * 60 * 1000,
    }
    if plotted_times:
        min_time = min(plotted_times)
        max_time = max(plotted_times)
        tick_start = min_time.floor("h")
        tick_end = max_time.ceil("h")
        hourly_ticks = pd.date_range(start=tick_start, end=tick_end, freq="1h")
        if len(hourly_ticks) > 0:
            x_axis_settings["tickmode"] = "array"
            x_axis_settings["tickvals"] = [pd.Timestamp(value).isoformat() for value in hourly_ticks]
            if use_12_hour:
                x_axis_settings["ticktext"] = [
                    normalize_12_hour_label(pd.Timestamp(value).strftime("%I%p")) for value in hourly_ticks
                ]
            else:
                x_axis_settings["ticktext"] = [pd.Timestamp(value).strftime("%H") for value in hourly_ticks]
        x_axis_settings["tick0"] = tick_start.isoformat()
        x_axis_settings["range"] = [
            (min_time - pd.Timedelta(minutes=10)).isoformat(),
            (max_time + pd.Timedelta(minutes=10)).isoformat(),
        ]

    fig.update_xaxes(
        **x_axis_settings,
    )
    fig.update_yaxes(
        range=[-12, 90],
        tickvals=[0, 15, 30, 45, 60, 75, 90],
        showgrid=True,
        gridcolor=theme_colors["grid"],
        gridwidth=1,
    )
    return fig


def build_path_plot(
    track: pd.DataFrame,
    events: dict[str, pd.Series | None],
    obstructions: dict[str, float],
    selected_label: str,
    selected_emissions: str,
    selected_color: str,
    selected_line_width: float,
    use_12_hour: bool,
    selected_schedule_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    overlay_tracks: list[dict[str, Any]] | None = None,
    mount_choice: str = "none",
    moon_track: pd.DataFrame | None = None,
    moon_phase_key: str | None = None,
    eclipse_visibility: dict[str, Any] | None = None,
) -> go.Figure:
    path_endpoint_marker_size_primary = int(_ui_name("PATH_ENDPOINT_MARKER_SIZE_PRIMARY"))
    path_endpoint_marker_size_overlay = int(_ui_name("PATH_ENDPOINT_MARKER_SIZE_OVERLAY"))
    path_line_width_overlay_default = float(_ui_name("PATH_LINE_WIDTH_OVERLAY_DEFAULT"))
    theme_colors = resolve_plot_theme_colors()
    fig = go.Figure()
    sequence_marker_entries: list[dict[str, Any]] = []
    sequence_entries: list[tuple[str, pd.Timestamp | None]] = [("selected", _track_start_timestamp(track))]
    if overlay_tracks:
        for overlay_index, target_track in enumerate(overlay_tracks):
            sequence_entries.append((f"overlay:{overlay_index}", _track_start_timestamp(target_track.get("track"))))
    sequence_numbers = _build_sequence_numbers(sequence_entries)

    for azimuth in (90.0, 180.0, 270.0):
        fig.add_shape(
            type="line",
            x0=azimuth,
            y0=0.0,
            x1=azimuth,
            y1=90.0,
            xref="x",
            yref="y",
            line={"color": theme_colors["cardinal_grid"], "width": 1, "dash": "dot"},
            layer="below",
        )

    obstruction_x, obstruction_y = obstruction_step_profile(obstructions)
    fig.add_trace(
        go.Scatter(
            x=obstruction_x,
            y=obstruction_y,
            mode="lines",
            name="Obstructed region",
            line={"width": 1, "color": theme_colors["obstruction_line"]},
            fill="tozeroy",
            fillcolor=theme_colors["obstruction_fill"],
            hoverinfo="skip",
        )
    )
    mount_warning_zone = mount_warning_zone_altitude_bounds(mount_choice)
    if mount_warning_zone is not None:
        warning_y0, warning_y1, warning_label = mount_warning_zone
        warning_fill, warning_line = mount_warning_zone_plot_style()
        fig.add_trace(
            go.Scatter(
                x=[0.0, 360.0, 360.0, 0.0, 0.0],
                y=[warning_y0, warning_y0, warning_y1, warning_y1, warning_y0],
                mode="lines",
                showlegend=False,
                line={"width": 0, "color": warning_line},
                fill="toself",
                fillcolor=warning_fill,
                hoverinfo="skip",
                name=warning_label,
            )
        )
        threshold_alt = warning_y1 if warning_y0 <= 0.0 else warning_y0
        fig.add_trace(
            go.Scatter(
                x=[0.0, 360.0],
                y=[threshold_alt, threshold_alt],
                mode="lines",
                showlegend=False,
                line={"width": 1, "color": warning_line, "dash": "dot"},
                hoverinfo="skip",
            )
        )

    moon_culmination_point = _moon_culmination_point(moon_track) if isinstance(moon_track, pd.DataFrame) else None
    if isinstance(moon_track, pd.DataFrame) and not moon_track.empty:
        moon_path_x, moon_path_y, moon_path_times = split_path_on_az_wrap(moon_track, use_12_hour=use_12_hour)
        if moon_path_x.size > 0:
            moon_shadow_color, moon_line_color = _moon_path_colors(eclipse_visibility)
            moon_hover = build_path_hovertext("Moon", "", moon_path_times, moon_path_y)
            fig.add_trace(
                go.Scatter(
                    x=moon_path_x,
                    y=moon_path_y,
                    mode="lines",
                    showlegend=False,
                    line={"width": 3.2, "color": moon_shadow_color},
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=moon_path_x,
                    y=moon_path_y,
                    mode="lines",
                    name="Moon",
                    showlegend=False,
                    line={"width": 1.9, "color": moon_line_color},
                    hovertext=moon_hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )
            eclipse_phase_points = _eclipse_phase_plot_points(
                moon_track,
                eclipse_visibility,
                use_12_hour=use_12_hour,
            )
            if eclipse_phase_points:
                phase_text_positions = [
                    ("top center" if (idx % 2 == 0) else "bottom center")
                    for idx in range(len(eclipse_phase_points))
                ]
                fig.add_trace(
                    go.Scatter(
                        x=[float(point["az"]) for point in eclipse_phase_points],
                        y=[float(point["alt"]) for point in eclipse_phase_points],
                        mode="markers+text",
                        text=[str(point["text"]) for point in eclipse_phase_points],
                        textposition=phase_text_positions,
                        textfont={"size": 10, "color": _ECLIPSE_ACCENT_HEX},
                        marker={
                            "size": 7,
                            "color": _ECLIPSE_ACCENT_HEX,
                            "line": {"width": 1, "color": "#FFFFFF"},
                        },
                        showlegend=False,
                        hovertext=[str(point["hover"]) for point in eclipse_phase_points],
                        hovertemplate="%{hovertext}<extra></extra>",
                    )
                )

    path_x, path_y, path_times = split_path_on_az_wrap(track, use_12_hour=use_12_hour)
    selected_hover = build_path_hovertext(selected_label, selected_emissions, path_times, path_y)
    fig.add_trace(
        go.Scatter(
            x=path_x,
            y=path_y,
            mode="lines",
            name=selected_label,
            line={"width": selected_line_width, "color": selected_color},
            hovertext=selected_hover,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    selected_sequence_number = sequence_numbers.get("selected")
    selected_sequence_point = _first_cartesian_track_point(track)
    if selected_sequence_number is not None and selected_sequence_point is not None:
        sequence_marker_entries.append(
            {
                "x": float(selected_sequence_point[0]),
                "y": float(selected_sequence_point[1]),
                "label": str(selected_sequence_number),
                "color": selected_color,
                "size": 24,
            }
        )
    selected_schedule_segment = _clip_track_to_schedule_window(track, selected_schedule_window)
    if not selected_schedule_segment.empty:
        schedule_path_x, schedule_path_y, schedule_path_times = split_path_on_az_wrap(
            selected_schedule_segment,
            use_12_hour=use_12_hour,
        )
        if schedule_path_x.size > 0:
            selected_schedule_hover = build_path_hovertext(
                selected_label,
                selected_emissions,
                schedule_path_times,
                schedule_path_y,
            )
            fig.add_trace(
                go.Scatter(
                    x=schedule_path_x,
                    y=schedule_path_y,
                    mode="lines",
                    showlegend=False,
                    line={"width": selected_line_width, "color": selected_color},
                    hovertext=selected_schedule_hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )
    rise_segment, set_segment = endpoint_marker_segments_cartesian(track, events)
    if set_segment is None:
        set_segment = terminal_segment_from_path_arrays(path_x, path_y)
    if rise_segment is not None:
        fig.add_trace(
            go.Scatter(
                x=[rise_segment[0], rise_segment[2]],
                y=[rise_segment[1], rise_segment[3]],
                mode="markers",
                showlegend=False,
                marker={
                    "size": [0, path_endpoint_marker_size_primary],
                    "symbol": "line-ew",
                    "angleref": "previous",
                    "color": selected_color,
                    "line": {"width": 2, "color": selected_color},
                },
                hoverinfo="skip",
            )
        )
    if set_segment is not None:
        fig.add_trace(
            go.Scatter(
                x=[set_segment[0], set_segment[2]],
                y=[set_segment[1], set_segment[3]],
                mode="markers",
                showlegend=False,
                marker={
                    "size": [0, path_endpoint_marker_size_primary + 1],
                    "symbol": "triangle-up",
                    "angleref": "previous",
                    "color": selected_color,
                    "line": {"width": 1, "color": selected_color},
                },
                hoverinfo="skip",
            )
        )
    selected_events = iter_labeled_events(events)
    if selected_events:
        event_x = [float(event["az"]) for _, event in selected_events]
        event_y = [float(event["alt"]) for _, event in selected_events]
        event_text = [label for label, _ in selected_events]
        event_custom = np.asarray(
            [[label, format_display_time(pd.Timestamp(event["time_local"]), use_12_hour=use_12_hour)] for label, event in selected_events],
            dtype=object,
        )
        fig.add_trace(
            go.Scatter(
                x=event_x,
                y=event_y,
                mode="markers+text",
                text=event_text,
                textposition="top center",
                textfont={"color": selected_color},
                marker={"size": 8, "color": selected_color},
                showlegend=False,
                customdata=event_custom,
                hovertemplate=f"{selected_label}<br>%{{customdata[0]}} at %{{customdata[1]}}<extra></extra>",
            )
        )

    if overlay_tracks:
        for overlay_index, target_track in enumerate(overlay_tracks):
            overlay_track = target_track.get("track")
            if not isinstance(overlay_track, pd.DataFrame) or overlay_track.empty:
                continue
            path_x, path_y, path_times = split_path_on_az_wrap(overlay_track, use_12_hour=use_12_hour)
            if path_x.size == 0:
                continue

            target_label = str(target_track.get("label", "List target"))
            target_color = str(target_track.get("color", "#22c55e"))
            target_line_width = float(target_track.get("line_width", path_line_width_overlay_default))
            target_schedule_window = _coerce_schedule_window(target_track.get("schedule_window"))
            target_emissions = str(target_track.get("emission_lines_display") or "").strip()
            overlay_hover = build_path_hovertext(target_label, target_emissions, path_times, path_y)
            fig.add_trace(
                go.Scatter(
                    x=path_x,
                    y=path_y,
                    mode="lines",
                    name=target_label,
                    showlegend=False,
                    line={"width": target_line_width, "color": target_color},
                    hovertext=overlay_hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )
            overlay_sequence_number = sequence_numbers.get(f"overlay:{overlay_index}")
            overlay_sequence_point = _first_cartesian_track_point(overlay_track)
            if overlay_sequence_number is not None and overlay_sequence_point is not None:
                sequence_marker_entries.append(
                    {
                        "x": float(overlay_sequence_point[0]),
                        "y": float(overlay_sequence_point[1]),
                        "label": str(overlay_sequence_number),
                        "color": target_color,
                        "size": 20,
                    }
                )
            overlay_schedule_segment = _clip_track_to_schedule_window(overlay_track, target_schedule_window)
            if not overlay_schedule_segment.empty:
                schedule_overlay_x, schedule_overlay_y, schedule_overlay_times = split_path_on_az_wrap(
                    overlay_schedule_segment,
                    use_12_hour=use_12_hour,
                )
                if schedule_overlay_x.size > 0:
                    overlay_schedule_hover = build_path_hovertext(
                        target_label,
                        target_emissions,
                        schedule_overlay_times,
                        schedule_overlay_y,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=schedule_overlay_x,
                            y=schedule_overlay_y,
                            mode="lines",
                            showlegend=False,
                            line={
                                "width": target_line_width,
                                "color": target_color,
                            },
                            hovertext=overlay_schedule_hover,
                            hovertemplate="%{hovertext}<extra></extra>",
                        )
                    )
            overlay_rise_segment, overlay_set_segment = endpoint_marker_segments_cartesian(
                overlay_track, target_track.get("events", {})
            )
            if overlay_set_segment is None:
                overlay_set_segment = terminal_segment_from_path_arrays(path_x, path_y)
            if overlay_rise_segment is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[overlay_rise_segment[0], overlay_rise_segment[2]],
                        y=[overlay_rise_segment[1], overlay_rise_segment[3]],
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": [0, path_endpoint_marker_size_overlay],
                            "symbol": "line-ew",
                            "angleref": "previous",
                            "color": target_color,
                            "line": {"width": 2, "color": target_color},
                        },
                        hoverinfo="skip",
                    )
                )

            if overlay_set_segment is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[overlay_set_segment[0], overlay_set_segment[2]],
                        y=[overlay_set_segment[1], overlay_set_segment[3]],
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": [0, path_endpoint_marker_size_overlay + 1],
                            "symbol": "triangle-up",
                            "angleref": "previous",
                            "color": target_color,
                            "line": {"width": 1, "color": target_color},
                        },
                        hoverinfo="skip",
                    )
                )
            overlay_events = iter_labeled_events(target_track.get("events", {}))
            if overlay_events:
                event_x = [float(event["az"]) for _, event in overlay_events]
                event_y = [float(event["alt"]) for _, event in overlay_events]
                event_text = [label for label, _ in overlay_events]
                event_custom = np.asarray(
                    [[label, format_display_time(pd.Timestamp(event["time_local"]), use_12_hour=use_12_hour)] for label, event in overlay_events],
                    dtype=object,
                )
                fig.add_trace(
                    go.Scatter(
                        x=event_x,
                        y=event_y,
                        mode="markers+text",
                        text=event_text,
                        textposition="top center",
                        textfont={"color": target_color},
                        marker={"size": 7, "color": target_color},
                        showlegend=False,
                        customdata=event_custom,
                        hovertemplate=f"{target_label}<br>%{{customdata[0]}} at %{{customdata[1]}}<extra></extra>",
                    )
                )

    if moon_culmination_point is not None and np.isfinite(float(moon_culmination_point.get("az", np.nan))):
        moon_emoji = lunar_phase_emoji(moon_phase_key)
        moon_culmination_hover = _moon_culmination_hovertext(moon_culmination_point, use_12_hour=use_12_hour)
        fig.add_trace(
            go.Scatter(
                x=[float(moon_culmination_point["az"])],
                y=[float(moon_culmination_point["alt"])],
                mode="text",
                text=[moon_emoji],
                textposition="middle center",
                textfont={"size": 18},
                showlegend=False,
                hovertext=[moon_culmination_hover],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )
    for marker_entry in sequence_marker_entries:
        fig.add_trace(
            go.Scatter(
                x=[float(marker_entry["x"])],
                y=[float(marker_entry["y"])],
                mode="markers+text",
                text=[str(marker_entry["label"])],
                textposition="middle center",
                textfont={"size": 14, "color": "#FFFFFF"},
                marker={
                    "size": int(marker_entry.get("size", 20)),
                    "color": str(marker_entry.get("color", "#111827")),
                    "line": {"width": 2, "color": "#FFFFFF"},
                },
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Target Paths",
        height=360,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        showlegend=False,
        plot_bgcolor=theme_colors["plot_bg"],
        paper_bgcolor=theme_colors["paper_bg"],
        font={"color": theme_colors["text"]},
        xaxis_title="Azimuth",
        yaxis_title="Altitude (deg)",
    )
    fig.update_xaxes(tickvals=[i * 22.5 for i in range(16)], ticktext=WIND16, range=[0, 360])
    fig.update_yaxes(
        range=[-12, 90],
        tickvals=[0, 15, 30, 45, 60, 75, 90],
        showgrid=True,
        gridcolor=theme_colors["grid"],
        gridwidth=1,
    )
    return fig


def build_path_plot_radial(
    track: pd.DataFrame,
    events: dict[str, pd.Series | None],
    obstructions: dict[str, float],
    dome_view: bool,
    selected_label: str,
    selected_emissions: str,
    selected_color: str,
    selected_line_width: float,
    use_12_hour: bool,
    selected_schedule_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    overlay_tracks: list[dict[str, Any]] | None = None,
    mount_choice: str = "none",
    moon_track: pd.DataFrame | None = None,
    moon_phase_key: str | None = None,
    eclipse_visibility: dict[str, Any] | None = None,
) -> go.Figure:
    path_endpoint_marker_size_primary = int(_ui_name("PATH_ENDPOINT_MARKER_SIZE_PRIMARY"))
    path_endpoint_marker_size_overlay = int(_ui_name("PATH_ENDPOINT_MARKER_SIZE_OVERLAY"))
    path_line_width_overlay_default = float(_ui_name("PATH_LINE_WIDTH_OVERLAY_DEFAULT"))
    theme_colors = resolve_plot_theme_colors()
    fig = go.Figure()
    sequence_entries: list[tuple[str, pd.Timestamp | None]] = [("selected", _track_start_timestamp(track))]
    if overlay_tracks:
        for overlay_index, target_track in enumerate(overlay_tracks):
            sequence_entries.append((f"overlay:{overlay_index}", _track_start_timestamp(target_track.get("track"))))
    sequence_numbers = _build_sequence_numbers(sequence_entries)

    obstruction_theta, obstruction_alt = obstruction_step_profile(obstructions)

    if dome_view:
        obstruction_boundary_r = [90.0 - value for value in obstruction_alt]
        obstruction_baseline_r = [90.0 for _ in obstruction_theta]
        track_r = (90.0 - track["alt"]).clip(lower=0.0, upper=90.0)
        radial_ticktext = ["90", "60", "30", "0"]
        radial_title = "Altitude (deg, center=90)"
    else:
        obstruction_boundary_r = obstruction_alt
        obstruction_baseline_r = [0.0 for _ in obstruction_theta]
        track_r = track["alt"].clip(lower=0.0, upper=90.0)
        radial_ticktext = ["0", "30", "60", "90"]
        radial_title = "Altitude (deg, center=0)"

    fig.add_trace(
        go.Scatterpolar(
            theta=obstruction_theta,
            r=obstruction_boundary_r,
            mode="lines",
            line={"width": 1, "color": theme_colors["obstruction_line"]},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            theta=obstruction_theta,
            r=obstruction_baseline_r,
            mode="lines",
            name="Obstructed region",
            line={"width": 0},
            fill="tonext",
            fillcolor=theme_colors["obstruction_fill"],
            hoverinfo="skip",
        )
    )

    for azimuth in (90.0, 180.0, 270.0):
        fig.add_trace(
            go.Scatterpolar(
                theta=[azimuth, azimuth],
                r=[0.0, 90.0],
                mode="lines",
                line={"color": theme_colors["cardinal_grid"], "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    mount_warning_zone = mount_warning_zone_altitude_bounds(mount_choice)
    if mount_warning_zone is not None:
        warning_alt_low, warning_alt_high, _warning_label = mount_warning_zone
        warning_fill, warning_line = mount_warning_zone_plot_style()
        warning_theta = [float(value) for value in range(0, 361, 5)]
        if dome_view:
            warning_boundary_r = [max(0.0, min(90.0, 90.0 - warning_alt_high)) for _ in warning_theta]
            warning_baseline_r = [max(0.0, min(90.0, 90.0 - warning_alt_low)) for _ in warning_theta]
        else:
            warning_boundary_r = [max(0.0, min(90.0, warning_alt_low)) for _ in warning_theta]
            warning_baseline_r = [max(0.0, min(90.0, warning_alt_high)) for _ in warning_theta]

        fig.add_trace(
            go.Scatterpolar(
                theta=warning_theta,
                r=warning_boundary_r,
                mode="lines",
                line={"width": 1, "color": warning_line, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                theta=warning_theta,
                r=warning_baseline_r,
                mode="lines",
                line={"width": 0},
                fill="tonext",
                fillcolor=warning_fill,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    moon_culmination_point = _moon_culmination_point(moon_track) if isinstance(moon_track, pd.DataFrame) else None
    if isinstance(moon_track, pd.DataFrame) and not moon_track.empty:
        moon_altitudes = pd.to_numeric(moon_track.get("alt"), errors="coerce")
        moon_azimuths = pd.to_numeric(moon_track.get("az"), errors="coerce")
        moon_times = pd.to_datetime(moon_track.get("time_local"), errors="coerce")
        moon_valid_mask = moon_altitudes.notna() & moon_azimuths.notna() & moon_times.notna()
        if bool(moon_valid_mask.any()):
            moon_altitudes = moon_altitudes.loc[moon_valid_mask].astype(float)
            moon_azimuths = moon_azimuths.loc[moon_valid_mask].astype(float)
            moon_times = moon_times.loc[moon_valid_mask]
            moon_altitude_values = moon_altitudes.to_numpy(dtype=float)
            moon_azimuth_values = moon_azimuths.to_numpy(dtype=float)
            moon_finite_mask = np.isfinite(moon_altitude_values) & np.isfinite(moon_azimuth_values)
            if bool(moon_finite_mask.any()):
                moon_altitude_values = moon_altitude_values[moon_finite_mask]
                moon_azimuth_values = moon_azimuth_values[moon_finite_mask]
                moon_times = moon_times.iloc[moon_finite_mask]
                if moon_altitude_values.size > 0 and not moon_times.empty:
                    moon_shadow_color, moon_line_color = _moon_path_colors(eclipse_visibility)
                    moon_alt_series = pd.Series(moon_altitude_values)
                    moon_r = (90.0 - moon_alt_series.clip(lower=0.0, upper=90.0)) if dome_view else moon_alt_series.clip(lower=0.0, upper=90.0)
                    moon_time_values = np.asarray(
                        [
                            format_display_time(pd.Timestamp(value), use_12_hour=use_12_hour)
                            for value in moon_times.tolist()
                        ],
                        dtype=object,
                    )
                    moon_hover = build_path_hovertext("Moon", "", moon_time_values, moon_altitude_values)
                    fig.add_trace(
                        go.Scatterpolar(
                            theta=moon_azimuth_values,
                            r=moon_r,
                            mode="lines",
                            showlegend=False,
                            line={"width": 3.2, "color": moon_shadow_color},
                            hoverinfo="skip",
                        )
                    )
                    fig.add_trace(
                        go.Scatterpolar(
                            theta=moon_azimuth_values,
                            r=moon_r,
                            mode="lines",
                            name="Moon",
                            showlegend=False,
                            line={"width": 1.9, "color": moon_line_color},
                            hovertext=moon_hover,
                            hovertemplate="%{hovertext}<extra></extra>",
                        )
                    )
                    eclipse_phase_points = _eclipse_phase_plot_points(
                        moon_track,
                        eclipse_visibility,
                        use_12_hour=use_12_hour,
                    )
                    if eclipse_phase_points:
                        phase_text_positions = [
                            ("top center" if (idx % 2 == 0) else "bottom center")
                            for idx in range(len(eclipse_phase_points))
                        ]
                        phase_r = [
                            (
                                max(0.0, min(90.0, 90.0 - float(point["alt"])))
                                if dome_view
                                else max(0.0, min(90.0, float(point["alt"])))
                            )
                            for point in eclipse_phase_points
                        ]
                        fig.add_trace(
                            go.Scatterpolar(
                                theta=[float(point["az"]) for point in eclipse_phase_points],
                                r=phase_r,
                                mode="markers+text",
                                text=[str(point["text"]) for point in eclipse_phase_points],
                                textposition=phase_text_positions,
                                textfont={"size": 10, "color": _ECLIPSE_ACCENT_HEX},
                                marker={
                                    "size": 7,
                                    "color": _ECLIPSE_ACCENT_HEX,
                                    "line": {"width": 1, "color": "#FFFFFF"},
                                },
                                showlegend=False,
                                hovertext=[str(point["hover"]) for point in eclipse_phase_points],
                                hovertemplate="%{hovertext}<extra></extra>",
                            )
                        )

    selected_time_values = np.asarray(
        [format_display_time(pd.Timestamp(value), use_12_hour=use_12_hour) for value in track["time_local"].tolist()],
        dtype=object,
    )
    selected_hover = build_path_hovertext(
        selected_label,
        selected_emissions,
        selected_time_values,
        track["alt"].to_numpy(dtype=float),
    )
    fig.add_trace(
        go.Scatterpolar(
            theta=track["az"],
            r=track_r,
            mode="lines",
            name=selected_label,
            line={"width": selected_line_width, "color": selected_color},
            hovertext=selected_hover,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    selected_sequence_number = sequence_numbers.get("selected")
    selected_sequence_point = _first_radial_track_point(track, dome_view=dome_view)
    if selected_sequence_number is not None and selected_sequence_point is not None:
        fig.add_trace(
            go.Scatterpolar(
                theta=[selected_sequence_point[0]],
                r=[selected_sequence_point[1]],
                mode="markers+text",
                text=[str(selected_sequence_number)],
                textposition="middle center",
                textfont={"size": 10, "color": "#FFFFFF"},
                marker={
                    "size": 16,
                    "color": selected_color,
                    "line": {"width": 1, "color": "#FFFFFF"},
                },
                showlegend=False,
                hoverinfo="skip",
            )
        )
    selected_schedule_segment = _clip_track_to_schedule_window(track, selected_schedule_window)
    if not selected_schedule_segment.empty:
        schedule_altitudes = pd.to_numeric(selected_schedule_segment.get("alt"), errors="coerce")
        schedule_azimuths = pd.to_numeric(selected_schedule_segment.get("az"), errors="coerce")
        schedule_times = pd.to_datetime(selected_schedule_segment.get("time_local"), errors="coerce")
        schedule_mask = schedule_altitudes.notna() & schedule_azimuths.notna() & schedule_times.notna()
        if bool(schedule_mask.any()):
            schedule_altitudes = schedule_altitudes.loc[schedule_mask].astype(float)
            schedule_azimuths = schedule_azimuths.loc[schedule_mask].astype(float)
            schedule_times = schedule_times.loc[schedule_mask]
            schedule_alt_values = schedule_altitudes.to_numpy(dtype=float)
            schedule_az_values = schedule_azimuths.to_numpy(dtype=float)
            schedule_finite_mask = np.isfinite(schedule_alt_values) & np.isfinite(schedule_az_values)
            if bool(schedule_finite_mask.any()):
                schedule_alt_values = schedule_alt_values[schedule_finite_mask]
                schedule_az_values = schedule_az_values[schedule_finite_mask]
                schedule_times = schedule_times.iloc[schedule_finite_mask]
                if schedule_alt_values.size > 0 and not schedule_times.empty:
                    schedule_alt_series = pd.Series(schedule_alt_values)
                    schedule_r = (
                        (90.0 - schedule_alt_series.clip(lower=0.0, upper=90.0))
                        if dome_view
                        else schedule_alt_series.clip(lower=0.0, upper=90.0)
                    )
                    schedule_time_values = np.asarray(
                        [
                            format_display_time(pd.Timestamp(value), use_12_hour=use_12_hour)
                            for value in schedule_times.tolist()
                        ],
                        dtype=object,
                    )
                    selected_schedule_hover = build_path_hovertext(
                        selected_label,
                        selected_emissions,
                        schedule_time_values,
                        schedule_alt_values,
                    )
                    fig.add_trace(
                        go.Scatterpolar(
                            theta=schedule_az_values,
                            r=schedule_r,
                            mode="lines",
                            showlegend=False,
                            line={"width": selected_line_width, "color": selected_color},
                            hovertext=selected_schedule_hover,
                            hovertemplate="%{hovertext}<extra></extra>",
                        )
                    )
    radial_values = np.asarray(track_r, dtype=float)
    rise_segment, set_segment = endpoint_marker_segments_radial(track, events, radial_values)
    if set_segment is None:
        set_segment = terminal_segment_from_path_arrays(track["az"], track_r)
    if rise_segment is not None:
        fig.add_trace(
            go.Scatterpolar(
                theta=[rise_segment[0], rise_segment[2]],
                r=[rise_segment[1], rise_segment[3]],
                mode="markers",
                showlegend=False,
                marker={
                    "size": [0, path_endpoint_marker_size_primary],
                    "symbol": "line-ew",
                    "angleref": "previous",
                    "color": selected_color,
                    "line": {"width": 2, "color": selected_color},
                },
                hoverinfo="skip",
            )
        )
    if set_segment is not None:
        fig.add_trace(
            go.Scatterpolar(
                theta=[set_segment[0], set_segment[2]],
                r=[set_segment[1], set_segment[3]],
                mode="markers",
                showlegend=False,
                marker={
                    "size": [0, path_endpoint_marker_size_primary + 1],
                    "symbol": "triangle-up",
                    "angleref": "previous",
                    "color": selected_color,
                    "line": {"width": 1, "color": selected_color},
                },
                hoverinfo="skip",
            )
        )
    selected_events = iter_labeled_events(events)
    if selected_events:
        event_theta = [float(event["az"]) for _, event in selected_events]
        event_r = [
            max(0.0, min(90.0, 90.0 - float(event["alt"]))) if dome_view else max(0.0, min(90.0, float(event["alt"])))
            for _, event in selected_events
        ]
        event_text = [label for label, _ in selected_events]
        event_custom = np.asarray(
            [[label, format_display_time(pd.Timestamp(event["time_local"]), use_12_hour=use_12_hour)] for label, event in selected_events],
            dtype=object,
        )
        fig.add_trace(
            go.Scatterpolar(
                theta=event_theta,
                r=event_r,
                mode="markers+text",
                text=event_text,
                textposition="top center",
                textfont={"color": selected_color},
                marker={"size": 8, "color": selected_color},
                showlegend=False,
                customdata=event_custom,
                hovertemplate=f"{selected_label}<br>%{{customdata[0]}} at %{{customdata[1]}}<extra></extra>",
            )
        )

    if overlay_tracks:
        for overlay_index, target_track in enumerate(overlay_tracks):
            overlay_track = target_track.get("track")
            if not isinstance(overlay_track, pd.DataFrame) or overlay_track.empty:
                continue

            target_label = str(target_track.get("label", "List target"))
            target_color = str(target_track.get("color", "#22c55e"))
            target_line_width = float(target_track.get("line_width", path_line_width_overlay_default))
            target_schedule_window = _coerce_schedule_window(target_track.get("schedule_window"))
            target_emissions = str(target_track.get("emission_lines_display") or "").strip()
            overlay_alt = overlay_track["alt"].clip(lower=0.0, upper=90.0)
            overlay_r = (90.0 - overlay_alt) if dome_view else overlay_alt
            overlay_time_values = np.asarray(
                [
                    format_display_time(pd.Timestamp(value), use_12_hour=use_12_hour)
                    for value in overlay_track["time_local"].tolist()
                ],
                dtype=object,
            )
            overlay_hover = build_path_hovertext(
                target_label,
                target_emissions,
                overlay_time_values,
                overlay_track["alt"].to_numpy(dtype=float),
            )
            fig.add_trace(
                go.Scatterpolar(
                    theta=overlay_track["az"],
                    r=overlay_r,
                    mode="lines",
                    name=target_label,
                    showlegend=False,
                    line={"width": target_line_width, "color": target_color},
                    hovertext=overlay_hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )
            overlay_sequence_number = sequence_numbers.get(f"overlay:{overlay_index}")
            overlay_sequence_point = _first_radial_track_point(overlay_track, dome_view=dome_view)
            if overlay_sequence_number is not None and overlay_sequence_point is not None:
                fig.add_trace(
                    go.Scatterpolar(
                        theta=[overlay_sequence_point[0]],
                        r=[overlay_sequence_point[1]],
                        mode="markers+text",
                        text=[str(overlay_sequence_number)],
                        textposition="middle center",
                        textfont={"size": 10, "color": "#FFFFFF"},
                        marker={
                            "size": 14,
                            "color": target_color,
                            "line": {"width": 1, "color": "#FFFFFF"},
                        },
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            overlay_schedule_segment = _clip_track_to_schedule_window(overlay_track, target_schedule_window)
            if not overlay_schedule_segment.empty:
                schedule_overlay_altitudes = pd.to_numeric(overlay_schedule_segment.get("alt"), errors="coerce")
                schedule_overlay_azimuths = pd.to_numeric(overlay_schedule_segment.get("az"), errors="coerce")
                schedule_overlay_times = pd.to_datetime(overlay_schedule_segment.get("time_local"), errors="coerce")
                schedule_overlay_mask = (
                    schedule_overlay_altitudes.notna()
                    & schedule_overlay_azimuths.notna()
                    & schedule_overlay_times.notna()
                )
                if bool(schedule_overlay_mask.any()):
                    schedule_overlay_altitudes = schedule_overlay_altitudes.loc[schedule_overlay_mask].astype(float)
                    schedule_overlay_azimuths = schedule_overlay_azimuths.loc[schedule_overlay_mask].astype(float)
                    schedule_overlay_times = schedule_overlay_times.loc[schedule_overlay_mask]
                    schedule_overlay_alt_values = schedule_overlay_altitudes.to_numpy(dtype=float)
                    schedule_overlay_az_values = schedule_overlay_azimuths.to_numpy(dtype=float)
                    schedule_overlay_finite_mask = (
                        np.isfinite(schedule_overlay_alt_values) & np.isfinite(schedule_overlay_az_values)
                    )
                    if bool(schedule_overlay_finite_mask.any()):
                        schedule_overlay_alt_values = schedule_overlay_alt_values[schedule_overlay_finite_mask]
                        schedule_overlay_az_values = schedule_overlay_az_values[schedule_overlay_finite_mask]
                        schedule_overlay_times = schedule_overlay_times.iloc[schedule_overlay_finite_mask]
                        if schedule_overlay_alt_values.size > 0 and not schedule_overlay_times.empty:
                            schedule_overlay_alt_series = pd.Series(schedule_overlay_alt_values)
                            schedule_overlay_r = (
                                (90.0 - schedule_overlay_alt_series.clip(lower=0.0, upper=90.0))
                                if dome_view
                                else schedule_overlay_alt_series.clip(lower=0.0, upper=90.0)
                            )
                            schedule_overlay_time_values = np.asarray(
                                [
                                    format_display_time(pd.Timestamp(value), use_12_hour=use_12_hour)
                                    for value in schedule_overlay_times.tolist()
                                ],
                                dtype=object,
                            )
                            overlay_schedule_hover = build_path_hovertext(
                                target_label,
                                target_emissions,
                                schedule_overlay_time_values,
                                schedule_overlay_alt_values,
                            )
                            fig.add_trace(
                                go.Scatterpolar(
                                    theta=schedule_overlay_az_values,
                                    r=schedule_overlay_r,
                                    mode="lines",
                                    showlegend=False,
                                    line={
                                        "width": target_line_width,
                                        "color": target_color,
                                    },
                                    hovertext=overlay_schedule_hover,
                                    hovertemplate="%{hovertext}<extra></extra>",
                                )
                            )
            overlay_radial_values = np.asarray(overlay_r, dtype=float)
            overlay_rise_segment, overlay_set_segment = endpoint_marker_segments_radial(
                overlay_track, target_track.get("events", {}), overlay_radial_values
            )
            if overlay_set_segment is None:
                overlay_set_segment = terminal_segment_from_path_arrays(overlay_track["az"], overlay_r)
            if overlay_rise_segment is not None:
                fig.add_trace(
                    go.Scatterpolar(
                        theta=[overlay_rise_segment[0], overlay_rise_segment[2]],
                        r=[overlay_rise_segment[1], overlay_rise_segment[3]],
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": [0, path_endpoint_marker_size_overlay],
                            "symbol": "line-ew",
                            "angleref": "previous",
                            "color": target_color,
                            "line": {"width": 2, "color": target_color},
                        },
                        hoverinfo="skip",
                    )
                )

            if overlay_set_segment is not None:
                fig.add_trace(
                    go.Scatterpolar(
                        theta=[overlay_set_segment[0], overlay_set_segment[2]],
                        r=[overlay_set_segment[1], overlay_set_segment[3]],
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": [0, path_endpoint_marker_size_overlay + 1],
                            "symbol": "triangle-up",
                            "angleref": "previous",
                            "color": target_color,
                            "line": {"width": 1, "color": target_color},
                        },
                        hoverinfo="skip",
                    )
                )
            overlay_events = iter_labeled_events(target_track.get("events", {}))
            if overlay_events:
                event_theta = [float(event["az"]) for _, event in overlay_events]
                event_r = [
                    max(0.0, min(90.0, 90.0 - float(event["alt"])))
                    if dome_view
                    else max(0.0, min(90.0, float(event["alt"])))
                    for _, event in overlay_events
                ]
                event_text = [label for label, _ in overlay_events]
                event_custom = np.asarray(
                    [[label, format_display_time(pd.Timestamp(event["time_local"]), use_12_hour=use_12_hour)] for label, event in overlay_events],
                    dtype=object,
                )
                fig.add_trace(
                    go.Scatterpolar(
                        theta=event_theta,
                        r=event_r,
                        mode="markers+text",
                        text=event_text,
                        textposition="top center",
                        textfont={"color": target_color},
                        marker={"size": 7, "color": target_color},
                        showlegend=False,
                        customdata=event_custom,
                        hovertemplate=f"{target_label}<br>%{{customdata[0]}} at %{{customdata[1]}}<extra></extra>",
                    )
                )

    if moon_culmination_point is not None and np.isfinite(float(moon_culmination_point.get("az", np.nan))):
        moon_emoji = lunar_phase_emoji(moon_phase_key)
        moon_alt = float(moon_culmination_point["alt"])
        moon_r = max(0.0, min(90.0, 90.0 - moon_alt)) if dome_view else max(0.0, min(90.0, moon_alt))
        moon_culmination_hover = _moon_culmination_hovertext(moon_culmination_point, use_12_hour=use_12_hour)
        fig.add_trace(
            go.Scatterpolar(
                theta=[float(moon_culmination_point["az"])],
                r=[moon_r],
                mode="text",
                text=[moon_emoji],
                textposition="middle center",
                textfont={"size": 18},
                showlegend=False,
                hovertext=[moon_culmination_hover],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Target Paths",
        height=660,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        showlegend=False,
        paper_bgcolor=theme_colors["paper_bg"],
        font={"color": theme_colors["text"]},
        polar={
            "bgcolor": theme_colors["plot_bg"],
            "angularaxis": {
                "rotation": 90,
                "direction": "clockwise",
                "tickmode": "array",
                "tickvals": [i * 22.5 for i in range(16)],
                "ticktext": WIND16,
                "gridcolor": theme_colors["grid"],
                "linecolor": theme_colors["grid"],
                "tickfont": {"color": theme_colors["text"]},
            },
            "radialaxis": {
                "range": [0, 90],
                "tickmode": "array",
                "tickvals": [0, 30, 60, 90],
                "ticktext": radial_ticktext,
                "title": radial_title,
                "gridcolor": theme_colors["grid"],
                "linecolor": theme_colors["grid"],
                "tickfont": {"color": theme_colors["text"]},
            },
        },
    )
    return fig


def build_night_plot(
    track: pd.DataFrame,
    temperature_by_hour: dict[str, float],
    cloud_cover_by_hour: dict[str, float] | None,
    weather_by_hour: dict[str, dict[str, Any]] | None,
    temperature_unit: str,
    target_label: str | None = None,
    period_label: str | None = None,
    use_12_hour: bool = False,
) -> go.Figure:
    theme_colors = resolve_plot_theme_colors()
    if track.empty:
        return go.Figure()

    rows: list[dict[str, Any]] = []
    grouped = track.set_index("time_local").resample("1h")
    for hour, chunk in grouped:
        if chunk.empty:
            continue

        max_row = chunk.loc[chunk["alt"].idxmax()]
        hour_iso = normalize_hour_key(hour) or pd.Timestamp(hour).floor("h").isoformat()
        temp = temperature_by_hour.get(hour_iso)
        cloud_cover = cloud_cover_by_hour.get(hour_iso) if cloud_cover_by_hour else None
        weather_hour_row = weather_by_hour.get(hour_iso, {}) if weather_by_hour else {}
        alert_emoji, alert_tooltip = resolve_weather_alert_indicator(weather_hour_row, temperature_unit)

        rows.append(
            {
                "hour": hour,
                "hour_label": format_hour_label(hour, use_12_hour=use_12_hour),
                "max_alt": max(float(max_row["alt"]), 0.0),
                "obstructed_alt": min(
                    max(float(max_row["alt"]), 0.0),
                    max(float(max_row.get("min_alt_required", 0.0)), 0.0),
                ),
                "wind16": str(max_row["wind16"]),
                "temp": temp,
                "cloud_cover": cloud_cover,
                "weather_alert_emoji": alert_emoji,
                "weather_alert_tooltip": alert_tooltip,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return go.Figure()
    frame["clear_alt"] = (frame["max_alt"] - frame["obstructed_alt"]).clip(lower=0.0)
    fallback_cloud_color = _interpolate_cloud_cover_color(50.0)
    frame["cloud_color"] = frame["cloud_cover"].apply(
        lambda value: (
            _interpolate_cloud_cover_color(float(value))
            if value is not None and not pd.isna(value)
            else fallback_cloud_color
        )
    )

    temp_labels = []
    temp_label_colors = []
    for _, row in frame.iterrows():
        temp_value = row["temp"]
        temp_str = format_temperature(temp_value, temperature_unit)
        temp_labels.append(temp_str)
        if temp_value is None or pd.isna(temp_value):
            temp_label_colors.append(theme_colors["muted_text"])
        else:
            temp_f = (float(temp_value) * 9.0 / 5.0) + 32.0
            temp_label_colors.append(_interpolate_temperature_color_f(temp_f))

    obstructed_hover = []
    clear_hover = []
    for _, row in frame.iterrows():
        hour_str = format_display_time(pd.Timestamp(row["hour"]), use_12_hour=use_12_hour)
        max_alt = float(row["max_alt"])
        obstructed_alt = float(row["obstructed_alt"])
        clear_alt = float(row["clear_alt"])
        cloud_cover_text = (
            f"{float(row['cloud_cover']):.0f}%"
            if row.get("cloud_cover") is not None and not pd.isna(row.get("cloud_cover"))
            else "-"
        )
        obstructed_hover.append(
            f"{row['hour_label']} ({hour_str})<br>Max Alt {max_alt:.1f} deg<br>"
            f"Cloud Cover {cloud_cover_text}<br>Obstructed {obstructed_alt:.1f} deg"
        )
        clear_hover.append(
            f"{row['hour_label']} ({hour_str})<br>Max Alt {max_alt:.1f} deg<br>"
            f"Cloud Cover {cloud_cover_text}<br>Visible {clear_alt:.1f} deg"
        )

    visible_colors = [str(color) for color in frame["cloud_color"].tolist()]
    wind_labels = [str(row["wind16"]) if float(row["clear_alt"]) > 0.0 else "" for _, row in frame.iterrows()]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["hour_label"],
            y=frame["clear_alt"],
            base=frame["obstructed_alt"],
            hovertext=clear_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            name="Visible",
            marker={"color": visible_colors},
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["hour_label"],
            y=frame["obstructed_alt"],
            hovertext=obstructed_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            name="Obstructed",
            marker={
                "color": theme_colors["obstruction_fill"],
                "line": {"color": theme_colors["obstruction_line"], "width": 1},
            },
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["hour_label"],
            y=frame["max_alt"],
            mode="text",
            text=wind_labels,
            textposition="top center",
            textfont={"color": theme_colors["text"]},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["hour_label"],
            y=np.full(len(frame), -4.5),
            mode="text",
            text=temp_labels,
            textposition="middle center",
            textfont={"color": temp_label_colors},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    alert_x: list[str] = []
    alert_y: list[float] = []
    alert_text: list[str] = []
    alert_hover: list[str] = []
    alert_y_level = -8.0
    for _, row in frame.iterrows():
        emoji = str(row.get("weather_alert_emoji") or "").strip()
        if not emoji:
            continue
        alert_x.append(str(row["hour_label"]))
        alert_y.append(alert_y_level)
        alert_text.append(emoji)
        tooltip_text = str(row.get("weather_alert_tooltip") or "").strip()
        hour_text = format_display_time(pd.Timestamp(row["hour"]), use_12_hour=use_12_hour)
        if tooltip_text:
            alert_hover.append(f"{row['hour_label']} ({hour_text})<br>{tooltip_text}")
        else:
            alert_hover.append(f"{row['hour_label']} ({hour_text})")
    if alert_text:
        fig.add_trace(
            go.Scatter(
                x=alert_x,
                y=alert_y,
                mode="text",
                text=alert_text,
                textposition="middle center",
                showlegend=False,
                hovertext=alert_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    title = "Hourly Forecast"
    cleaned_period_label = str(period_label or "").strip()
    if cleaned_period_label:
        title = f"{title} ({cleaned_period_label})"
    cleaned_label = str(target_label or "").strip()
    if cleaned_label:
        title = f"Hourly Forecast - {cleaned_label}"
        if cleaned_period_label:
            title = f"{title} ({cleaned_period_label})"

    fig.update_layout(
        title=title,
        height=400,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        barmode="overlay",
        bargap=0.06,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
        plot_bgcolor=theme_colors["plot_bg"],
        paper_bgcolor=theme_colors["paper_bg"],
        font={"color": theme_colors["text"]},
        yaxis_title="Altitude (deg)",
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=frame["hour_label"].tolist(),
        showgrid=True,
        gridwidth=1,
        gridcolor=theme_colors["grid"],
    )
    fig.update_yaxes(range=[-12, 90], tickvals=[0, 15, 30, 45, 60, 75, 90])
    return fig
