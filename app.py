from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import astropy.units as u
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from astral import LocationInfo
from astral.sun import sun
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from catalog_ingestion import load_unified_catalog
from geopy.geocoders import Nominatim
from streamlit_js_eval import get_browser_language, get_geolocation, get_local_storage, set_local_storage
from streamlit_autorefresh import st_autorefresh
from timezonefinder import TimezoneFinder

st.set_page_config(page_title="DSO Explorer", page_icon="✨", layout="wide")

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

DEFAULT_LOCATION = {
    "lat": 40.3573,
    "lon": -74.6672,
    "label": "Princeton, NJ",
    "source": "default",
    "resolved_at": "",
}

CATALOG_SEED_PATH = Path("data/dso_catalog_seed.csv")
CATALOG_CACHE_PATH = Path("data/dso_catalog_cache.parquet")
CATALOG_META_PATH = Path("data/dso_catalog_cache_meta.json")
BROWSER_PREFS_STORAGE_KEY = "dso_explorer_prefs_v1"
SETTINGS_EXPORT_FORMAT_VERSION = 1

TEMPERATURE_UNIT_OPTIONS = {
    "Auto (browser)": "auto",
    "Fahrenheit": "f",
    "Celsius": "c",
}
FAHRENHEIT_COUNTRY_CODES = {"US", "BS", "BZ", "KY", "PW", "FM", "MH", "LR"}
EVENT_LABELS: list[tuple[str, str]] = [
    ("rise", "Rise"),
    ("set", "Set"),
    ("first_visible", "First Visible"),
    ("last_visible", "Last Visible"),
    ("culmination", "Culmination"),
]
PATH_LINE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]
OBSTRUCTION_FILL_COLOR = "rgba(226, 232, 240, 0.40)"
OBSTRUCTION_LINE_COLOR = "rgba(148, 163, 184, 0.95)"


def default_preferences() -> dict[str, Any]:
    return {
        "favorites": [],
        "set_list": [],
        "obstructions": {direction: 20.0 for direction in WIND16},
        "location": copy.deepcopy(DEFAULT_LOCATION),
        "temperature_unit": "auto",
    }


def ensure_preferences_shape(raw: dict[str, Any]) -> dict[str, Any]:
    prefs = default_preferences()
    if isinstance(raw, dict):
        prefs["favorites"] = [str(x) for x in raw.get("favorites", []) if str(x).strip()]
        prefs["set_list"] = [str(x) for x in raw.get("set_list", []) if str(x).strip()]
        temp_unit = str(raw.get("temperature_unit", "auto")).strip().lower()
        prefs["temperature_unit"] = temp_unit if temp_unit in {"auto", "f", "c"} else "auto"

        obs = raw.get("obstructions", {})
        if isinstance(obs, dict):
            for key in WIND16:
                value = obs.get(key, 20.0)
                try:
                    prefs["obstructions"][key] = float(value)
                except (TypeError, ValueError):
                    prefs["obstructions"][key] = 20.0

        loc = raw.get("location", {})
        if isinstance(loc, dict):
            merged = copy.deepcopy(DEFAULT_LOCATION)
            merged.update({k: loc.get(k, merged[k]) for k in merged})
            prefs["location"] = merged

    return prefs


def encode_preferences_for_storage(prefs: dict[str, Any]) -> str:
    compact_json = json.dumps(ensure_preferences_shape(prefs), separators=(",", ":"), ensure_ascii=True)
    return base64.urlsafe_b64encode(compact_json.encode("utf-8")).decode("ascii")


def decode_preferences_from_storage(raw_value: str) -> dict[str, Any] | None:
    try:
        decoded_json = base64.urlsafe_b64decode(str(raw_value).encode("ascii")).decode("utf-8")
        payload = json.loads(decoded_json)
        if not isinstance(payload, dict):
            return None
        return ensure_preferences_shape(payload)
    except Exception:
        return None


def load_preferences() -> dict[str, Any]:
    raw_value = get_local_storage(BROWSER_PREFS_STORAGE_KEY, component_key="browser_prefs_read")
    if isinstance(raw_value, str) and raw_value.strip():
        decoded = decode_preferences_from_storage(raw_value)
        if decoded is not None:
            st.session_state.pop("prefs_persistence_notice", None)
            return decoded
    return default_preferences()


def save_preferences(prefs: dict[str, Any]) -> bool:
    try:
        encoded = encode_preferences_for_storage(prefs)
        payload_hash = hashlib.sha1(encoded.encode("ascii")).hexdigest()[:12]
        set_local_storage(
            BROWSER_PREFS_STORAGE_KEY,
            encoded,
            component_key=f"browser_prefs_write_{payload_hash}",
        )
        st.session_state.pop("prefs_persistence_notice", None)
        return True
    except Exception:
        st.session_state["prefs_persistence_notice"] = (
            "Browser-local preference storage is unavailable. Using session-only preferences."
        )
        return False


def build_settings_export_payload(prefs: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "dso_explorer_settings",
        "version": SETTINGS_EXPORT_FORMAT_VERSION,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "preferences": ensure_preferences_shape(prefs),
    }


def parse_settings_import_payload(raw_text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    candidate = payload
    if isinstance(payload.get("preferences"), dict):
        candidate = payload["preferences"]

    if not isinstance(candidate, dict):
        return None
    return ensure_preferences_shape(candidate)


def persist_and_rerun(prefs: dict[str, Any]) -> None:
    st.session_state["prefs"] = prefs
    save_preferences(prefs)
    st.rerun()


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


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


def canonicalize_designation(query: str) -> str:
    compact = normalize_text(query)

    match = re.match(r"^(messier|m)(\d+)$", compact)
    if match:
        return f"M{int(match.group(2))}"

    match = re.match(r"^(ngc)(\d+)$", compact)
    if match:
        return f"NGC {int(match.group(2))}"

    match = re.match(r"^(ic)(\d+)$", compact)
    if match:
        return f"IC {int(match.group(2))}"

    match = re.match(r"^(sh2|sharpless)(\d+)$", compact)
    if match:
        return f"Sh2-{int(match.group(2))}"

    return query.strip()


def az_to_wind16(az_deg: float) -> str:
    idx = int(((az_deg % 360.0) + 11.25) // 22.5) % 16
    return WIND16[idx]


def wind_bin_center(direction: str) -> float:
    idx = WIND16.index(direction)
    return idx * 22.5


def build_search_index(df: pd.DataFrame) -> pd.DataFrame:
    indexed = df.copy()
    for col in ["common_name", "object_type", "constellation", "aliases"]:
        if col in indexed:
            indexed[col] = indexed[col].fillna("")

    indexed["primary_id_norm"] = indexed["primary_id"].map(normalize_text)
    indexed["aliases_norm"] = indexed["aliases"].map(normalize_text)
    indexed["search_blob_norm"] = (
        indexed["primary_id"].fillna("")
        + " "
        + indexed["common_name"].fillna("")
        + " "
        + indexed["aliases"].fillna("")
        + " "
        + indexed["catalog"].fillna("")
    ).map(normalize_text)

    return indexed


def load_catalog(force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, metadata = load_unified_catalog(
        seed_path=CATALOG_SEED_PATH,
        cache_path=CATALOG_CACHE_PATH,
        metadata_path=CATALOG_META_PATH,
        force_refresh=force_refresh,
    )
    return build_search_index(frame), metadata


def search_catalog(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query.strip():
        return df.copy()

    canonical = canonicalize_designation(query)
    query_norm = normalize_text(query)
    canonical_norm = normalize_text(canonical)

    exact_mask = (df["primary_id_norm"] == query_norm) | (df["primary_id_norm"] == canonical_norm)
    alias_exact = df["aliases_norm"].str.contains(query_norm, regex=False) | df[
        "aliases_norm"
    ].str.contains(canonical_norm, regex=False)
    partial_mask = df["search_blob_norm"].str.contains(query_norm, regex=False)

    results = df[exact_mask | alias_exact | partial_mask].copy()
    results["_rank"] = np.where(exact_mask.loc[results.index] | alias_exact.loc[results.index], 0, 1)
    results = results.sort_values(by=["_rank", "catalog", "primary_id"], ascending=[True, True, True])
    return results.drop(columns=["_rank"])


def compute_altaz_now(targets: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    if targets.empty:
        enriched = targets.copy()
        enriched["alt_now"] = pd.Series(dtype=float)
        enriched["az_now"] = pd.Series(dtype=float)
        enriched["wind16"] = pd.Series(dtype=str)
        return enriched

    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    obstime = Time(datetime.now(timezone.utc))
    coords = SkyCoord(
        ra=targets["ra_deg"].to_numpy(dtype=float) * u.deg,
        dec=targets["dec_deg"].to_numpy(dtype=float) * u.deg,
    )
    frame = AltAz(obstime=obstime, location=location)
    altaz = coords.transform_to(frame)

    enriched = targets.copy()
    enriched["alt_now"] = np.round(altaz.alt.deg, 1)
    enriched["az_now"] = np.round(altaz.az.deg % 360.0, 1)
    enriched["wind16"] = [az_to_wind16(value) for value in enriched["az_now"]]
    return enriched


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def resolve_timezone(lat: float, lon: float) -> str:
    finder = TimezoneFinder()
    tz_name = finder.timezone_at(lat=lat, lng=lon)
    return tz_name or "UTC"


def tonight_window(lat: float, lon: float) -> tuple[datetime, datetime, ZoneInfo]:
    tz_name = resolve_timezone(lat, lon)
    tzinfo = ZoneInfo(tz_name)
    local_now = datetime.now(tzinfo)
    base_date = local_now.date()

    try:
        loc = LocationInfo(latitude=lat, longitude=lon, timezone=tz_name)
        sun_today = sun(loc.observer, date=base_date, tzinfo=tzinfo)
        sun_tomorrow = sun(loc.observer, date=base_date + timedelta(days=1), tzinfo=tzinfo)
        start = sun_today["sunset"]
        end = sun_tomorrow["sunrise"]
    except Exception:
        start = datetime.combine(base_date, time(18, 0), tzinfo=tzinfo)
        end = start + timedelta(hours=12)

    if end <= start:
        end = start + timedelta(hours=12)

    return start, end, tzinfo


def compute_track(
    ra_deg: float,
    dec_deg: float,
    lat: float,
    lon: float,
    start_local: datetime,
    end_local: datetime,
    obstructions: dict[str, float],
) -> pd.DataFrame:
    times_local = pd.date_range(start=start_local, end=end_local, freq="10min", inclusive="both")
    times_utc = times_local.tz_convert("UTC")

    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    frame = AltAz(obstime=Time(times_utc.to_pydatetime()), location=location)
    altaz = target.transform_to(frame)

    altitudes = altaz.alt.deg
    azimuths = altaz.az.deg % 360.0
    winds = [az_to_wind16(value) for value in azimuths]
    min_required = np.array([obstructions.get(wind, 20.0) for wind in winds], dtype=float)
    visible = (altitudes >= 0) & (altitudes >= min_required)

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


def extract_events(track: pd.DataFrame) -> dict[str, pd.Series | None]:
    events: dict[str, pd.Series | None] = {
        "rise": None,
        "set": None,
        "first_visible": None,
        "last_visible": None,
        "culmination": None,
    }

    if track.empty:
        return events

    above_horizon = track[track["alt"] >= 0]
    visible = track[track["visible"]]
    events["culmination"] = track.loc[track["alt"].idxmax()]

    if not above_horizon.empty:
        events["rise"] = above_horizon.iloc[0]
        events["set"] = above_horizon.iloc[-1]

    if not visible.empty:
        events["first_visible"] = visible.iloc[0]
        events["last_visible"] = visible.iloc[-1]

    return events


def format_time(series: pd.Series | None) -> str:
    if series is None:
        return "--"
    return pd.Timestamp(series["time_local"]).strftime("%H:%M")


def compute_total_visible_time(track: pd.DataFrame) -> timedelta:
    if track.empty or "visible" not in track:
        return timedelta(0)

    visible_mask = track["visible"].fillna(False).astype(bool).to_numpy()
    if not visible_mask.any():
        return timedelta(0)

    times = pd.to_datetime(track["time_local"])
    diffs = times.diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    step = positive_diffs.median() if not positive_diffs.empty else pd.Timedelta(minutes=10)

    total = pd.Timedelta(0)
    idx = 0
    while idx < len(visible_mask):
        if not visible_mask[idx]:
            idx += 1
            continue
        start_idx = idx
        while idx + 1 < len(visible_mask) and visible_mask[idx + 1]:
            idx += 1
        end_idx = idx
        total += (times.iloc[end_idx] - times.iloc[start_idx]) + step
        idx += 1

    if total < pd.Timedelta(0):
        return timedelta(0)
    return total.to_pytimedelta()


def format_duration_hm(duration: timedelta) -> str:
    total_minutes = max(0, int(round(duration.total_seconds() / 60.0)))
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


def build_sky_position_summary_rows(
    selected_id: str,
    selected_label: str,
    selected_color: str,
    selected_events: dict[str, pd.Series | None],
    selected_track: pd.DataFrame,
    set_list_tracks: list[dict[str, Any]],
    pinned_ids: set[str],
) -> list[dict[str, Any]]:
    def _build_row(
        primary_id: str,
        label: str,
        color: str,
        events: dict[str, pd.Series | None],
        is_pinned: bool,
    ) -> dict[str, Any]:
        culmination = events.get("culmination")
        culmination_dir = str(culmination["wind16"]) if culmination is not None else "--"
        return {
            "primary_id": primary_id,
            "line_color": color,
            "target": label,
            "rise": format_time(events.get("rise")),
            "first_visible": format_time(events.get("first_visible")),
            "culmination": format_time(events.get("culmination")),
            "last_visible": format_time(events.get("last_visible")),
            "set": format_time(events.get("set")),
            "visible_total": "--",
            "culmination_dir": culmination_dir,
            "is_pinned": is_pinned,
        }

    selected_row = _build_row(selected_id, selected_label, selected_color, selected_events, selected_id in pinned_ids)
    selected_row["visible_total"] = format_duration_hm(compute_total_visible_time(selected_track))
    rows = [selected_row]
    for target_track in set_list_tracks:
        primary_id = str(target_track.get("primary_id", ""))
        target_row = _build_row(
            primary_id,
            str(target_track.get("label", "Set List target")),
            str(target_track.get("color", "#22c55e")),
            target_track.get("events", {}),
            primary_id in pinned_ids,
        )
        overlay_track = target_track.get("track")
        if isinstance(overlay_track, pd.DataFrame):
            target_row["visible_total"] = format_duration_hm(compute_total_visible_time(overlay_track))
        rows.append(target_row)
    return rows


def render_sky_position_summary_table(rows: list[dict[str, Any]], prefs: dict[str, Any]) -> None:
    if not rows:
        return

    st.markdown("#### Sky Position Summary")
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return

    summary_df["line_swatch"] = "■"
    summary_df["set_list_state"] = summary_df["is_pinned"].map(lambda value: "Pinned" if bool(value) else "Unpinned")

    display = summary_df[
        [
            "line_swatch",
            "target",
            "rise",
            "first_visible",
            "culmination",
            "last_visible",
            "set",
            "visible_total",
            "culmination_dir",
            "set_list_state",
        ]
    ].rename(
        columns={
            "line_swatch": "Line",
            "target": "Target",
            "rise": "Rise",
            "first_visible": "First Visible",
            "culmination": "Culmination",
            "last_visible": "Last Visible",
            "set": "Set",
            "visible_total": "Visible",
            "culmination_dir": "Culm Dir",
            "set_list_state": "Set List",
        }
    )

    def _style_summary_row(row: pd.Series) -> list[str]:
        styles = ["" for _ in row]
        color = str(summary_df.loc[row.name, "line_color"]).strip()
        if color:
            line_idx = row.index.get_loc("Line")
            styles[line_idx] = f"color: {color}; font-weight: 700;"
        return styles

    styled = display.style.apply(_style_summary_row, axis=1)

    table_event = st.dataframe(
        styled,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="sky_summary_table",
        column_config={
            "Line": st.column_config.TextColumn(width="small"),
            "Target": st.column_config.TextColumn(width="large"),
            "Rise": st.column_config.TextColumn(width="small"),
            "First Visible": st.column_config.TextColumn(width="small"),
            "Culmination": st.column_config.TextColumn(width="small"),
            "Last Visible": st.column_config.TextColumn(width="small"),
            "Set": st.column_config.TextColumn(width="small"),
            "Visible": st.column_config.TextColumn(width="small"),
            "Culm Dir": st.column_config.TextColumn(width="small"),
            "Set List": st.column_config.TextColumn(width="small"),
        },
    )

    selected_rows: list[int] = []
    if table_event is not None:
        try:
            selected_rows = list(table_event.selection.rows)
        except Exception:
            if isinstance(table_event, dict):
                selected_rows = list(table_event.get("selection", {}).get("rows", []))

    selected_ids: list[str] = []
    selected_targets: list[str] = []
    for selected_index_raw in selected_rows:
        selected_index = int(selected_index_raw)
        if selected_index < 0 or selected_index >= len(summary_df):
            continue
        selected_row = summary_df.iloc[selected_index]
        primary_id = str(selected_row.get("primary_id", ""))
        if not primary_id:
            continue
        if primary_id in selected_ids:
            continue
        selected_ids.append(primary_id)
        selected_targets.append(str(selected_row.get("target", primary_id)))

    action_cols = st.columns([5, 2], gap="small")
    if selected_targets:
        action_cols[0].caption(f"Selected: {', '.join(selected_targets)}")
    else:
        action_cols[0].caption("Select one or more summary rows to toggle Set List.")

    if action_cols[1].button(
        "Toggle Selected",
        use_container_width=True,
        disabled=not bool(selected_ids),
        key="sky_summary_toggle_selected",
    ):
        next_set_list = list(prefs["set_list"])
        for primary_id in selected_ids:
            next_set_list = (
                remove_if_present(next_set_list, primary_id)
                if primary_id in next_set_list
                else add_if_missing(next_set_list, primary_id)
            )
        prefs["set_list"] = next_set_list
        persist_and_rerun(prefs)


def format_hour_label(value: pd.Timestamp | datetime) -> str:
    hour = int(pd.Timestamp(value).hour)
    suffix = "am" if hour < 12 else "pm"
    display = hour % 12
    if display == 0:
        display = 12
    return f"{display}{suffix}"


def split_path_on_az_wrap(track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if track.empty:
        return np.array([], dtype=float), np.array([], dtype=float), np.empty((0, 3), dtype=object)

    az_values = track["az"].to_numpy(dtype=float)
    alt_values = track["alt"].to_numpy(dtype=float)
    time_values = track["time_local"].dt.strftime("%H:%M").to_numpy(dtype=object)
    wind_values = track["wind16"].astype(str).to_numpy(dtype=object)
    min_values = track["min_alt_required"].to_numpy(dtype=float)

    x_values: list[float] = []
    y_values: list[float] = []
    custom_rows: list[list[object]] = []

    for idx, azimuth in enumerate(az_values):
        if idx > 0 and abs(float(azimuth) - float(az_values[idx - 1])) > 180.0:
            x_values.append(np.nan)
            y_values.append(np.nan)
            custom_rows.append(["", "", np.nan])

        x_values.append(float(azimuth))
        y_values.append(float(alt_values[idx]))
        custom_rows.append([time_values[idx], wind_values[idx], float(min_values[idx])])

    return (
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        np.asarray(custom_rows, dtype=object),
    )


def iter_labeled_events(events: dict[str, pd.Series | None]) -> list[tuple[str, pd.Series]]:
    labeled: list[tuple[str, pd.Series]] = []
    for event_key, event_label in EVENT_LABELS:
        event = events.get(event_key)
        if event is None:
            continue
        labeled.append((event_label, event))
    return labeled


def sample_direction_indices(length: int, max_markers: int = 6) -> list[int]:
    if length < 3:
        return []
    step = max(1, length // (max_markers + 1))
    return list(range(step, length - 1, step))


def direction_markers_cartesian(track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if track.empty:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=object),
        )

    az_values = track["az"].to_numpy(dtype=float)
    alt_values = track["alt"].to_numpy(dtype=float)
    time_values = track["time_local"].dt.strftime("%H:%M").to_numpy(dtype=object)
    indices = [idx for idx in sample_direction_indices(len(track)) if idx < len(track) - 1]

    x_values: list[float] = []
    y_values: list[float] = []
    angles: list[float] = []
    labels: list[object] = []

    for idx in indices:
        next_idx = idx + 1
        dx = ((float(az_values[next_idx]) - float(az_values[idx]) + 180.0) % 360.0) - 180.0
        dy = float(alt_values[next_idx]) - float(alt_values[idx])
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue

        x_values.append(float(az_values[idx]))
        y_values.append(float(alt_values[idx]))
        angles.append(float(np.degrees(np.arctan2(dy, dx))))
        labels.append(f"{time_values[idx]} -> {time_values[next_idx]}")

    return (
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        np.asarray(angles, dtype=float),
        np.asarray(labels, dtype=object),
    )


def direction_markers_radial(track: pd.DataFrame, radial_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if track.empty:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=object),
        )

    theta_values = track["az"].to_numpy(dtype=float)
    r_values = np.asarray(radial_values, dtype=float)
    time_values = track["time_local"].dt.strftime("%H:%M").to_numpy(dtype=object)
    indices = [idx for idx in sample_direction_indices(len(track)) if idx < len(track) - 1]

    theta_markers: list[float] = []
    radial_markers: list[float] = []
    angles: list[float] = []
    labels: list[object] = []

    for idx in indices:
        next_idx = idx + 1
        current_theta = np.deg2rad(theta_values[idx])
        next_theta = np.deg2rad(theta_values[next_idx])
        current_x = float(r_values[idx]) * float(np.sin(current_theta))
        current_y = float(r_values[idx]) * float(np.cos(current_theta))
        next_x = float(r_values[next_idx]) * float(np.sin(next_theta))
        next_y = float(r_values[next_idx]) * float(np.cos(next_theta))
        dx = next_x - current_x
        dy = next_y - current_y
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue

        theta_markers.append(float(theta_values[idx]))
        radial_markers.append(float(r_values[idx]))
        angles.append(float(np.degrees(np.arctan2(dy, dx))))
        labels.append(f"{time_values[idx]} -> {time_values[next_idx]}")

    return (
        np.asarray(theta_markers, dtype=float),
        np.asarray(radial_markers, dtype=float),
        np.asarray(angles, dtype=float),
        np.asarray(labels, dtype=object),
    )


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


def build_path_plot(
    track: pd.DataFrame,
    events: dict[str, pd.Series | None],
    obstructions: dict[str, float],
    selected_label: str,
    selected_color: str,
    set_list_tracks: list[dict[str, Any]] | None = None,
) -> go.Figure:
    fig = go.Figure()
    obstruction_x, obstruction_y = obstruction_step_profile(obstructions)
    fig.add_trace(
        go.Scatter(
            x=obstruction_x,
            y=obstruction_y,
            mode="lines",
            name="Obstructed region",
            line={"width": 1, "color": OBSTRUCTION_LINE_COLOR},
            fill="tozeroy",
            fillcolor=OBSTRUCTION_FILL_COLOR,
            hoverinfo="skip",
        )
    )

    path_x, path_y, path_custom = split_path_on_az_wrap(track)
    fig.add_trace(
        go.Scatter(
            x=path_x,
            y=path_y,
            mode="lines",
            name=selected_label,
            line={"width": 3, "color": selected_color},
            customdata=path_custom,
            hovertemplate="Az %{x:.1f} deg<br>Alt %{y:.1f} deg<br>Time %{customdata[0]}<br>Dir %{customdata[1]}<br>Obs %{customdata[2]:.0f} deg<extra></extra>",
        )
    )

    direction_x, direction_y, direction_angles, direction_labels = direction_markers_cartesian(track)
    if direction_x.size:
        fig.add_trace(
            go.Scatter(
                x=direction_x,
                y=direction_y,
                mode="markers",
                showlegend=False,
                marker={
                    "size": 12,
                    "symbol": "arrow-right",
                    "color": selected_color,
                    "angle": direction_angles,
                    "line": {"width": 0},
                },
                customdata=np.stack([direction_labels], axis=-1),
                hovertemplate=f"{selected_label}<br>Moving: %{{customdata[0]}}<extra></extra>",
            )
        )

    selected_events = iter_labeled_events(events)
    if selected_events:
        event_x = [float(event["az"]) for _, event in selected_events]
        event_y = [float(event["alt"]) for _, event in selected_events]
        event_text = [label for label, _ in selected_events]
        event_custom = np.asarray(
            [[label, pd.Timestamp(event["time_local"]).strftime("%H:%M")] for label, event in selected_events],
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

    if set_list_tracks:
        for target_track in set_list_tracks:
            overlay_track = target_track.get("track")
            if not isinstance(overlay_track, pd.DataFrame) or overlay_track.empty:
                continue
            path_x, path_y, path_custom = split_path_on_az_wrap(overlay_track)
            if path_x.size == 0:
                continue

            target_label = str(target_track.get("label", "Set List target"))
            target_color = str(target_track.get("color", "#22c55e"))
            overlay_custom = np.stack(
                [
                    np.full(path_x.shape[0], target_label, dtype=object),
                    path_custom[:, 0],
                    path_custom[:, 1],
                ],
                axis=-1,
            )
            fig.add_trace(
                go.Scatter(
                    x=path_x,
                    y=path_y,
                    mode="lines",
                    name=target_label,
                    showlegend=False,
                    line={"width": 2.2, "color": target_color},
                    customdata=overlay_custom,
                    hovertemplate="%{customdata[0]}<br>Az %{x:.1f} deg<br>Alt %{y:.1f} deg<br>Time %{customdata[1]}<br>Dir %{customdata[2]}<extra></extra>",
                )
            )

            overlay_direction_x, overlay_direction_y, overlay_direction_angles, overlay_direction_labels = (
                direction_markers_cartesian(overlay_track)
            )
            if overlay_direction_x.size:
                fig.add_trace(
                    go.Scatter(
                        x=overlay_direction_x,
                        y=overlay_direction_y,
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": 11,
                            "symbol": "arrow-right",
                            "color": target_color,
                            "angle": overlay_direction_angles,
                            "line": {"width": 0},
                        },
                        customdata=np.stack([overlay_direction_labels], axis=-1),
                        hovertemplate=f"{target_label}<br>Moving: %{{customdata[0]}}<extra></extra>",
                    )
                )

            overlay_events = iter_labeled_events(target_track.get("events", {}))
            if overlay_events:
                event_x = [float(event["az"]) for _, event in overlay_events]
                event_y = [float(event["alt"]) for _, event in overlay_events]
                event_text = [label for label, _ in overlay_events]
                event_custom = np.asarray(
                    [[label, pd.Timestamp(event["time_local"]).strftime("%H:%M")] for label, event in overlay_events],
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

    fig.update_layout(
        title="Sky Position",
        height=330,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        showlegend=False,
        xaxis_title="Azimuth",
        yaxis_title="Altitude (deg)",
    )
    fig.update_xaxes(tickvals=[i * 22.5 for i in range(16)], ticktext=WIND16, range=[0, 360])
    fig.update_yaxes(range=[0, 90])
    return fig


def build_path_plot_radial(
    track: pd.DataFrame,
    events: dict[str, pd.Series | None],
    obstructions: dict[str, float],
    dome_view: bool,
    selected_label: str,
    selected_color: str,
    set_list_tracks: list[dict[str, Any]] | None = None,
) -> go.Figure:
    fig = go.Figure()

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
            line={"width": 1, "color": OBSTRUCTION_LINE_COLOR},
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
            fillcolor=OBSTRUCTION_FILL_COLOR,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            theta=track["az"],
            r=track_r,
            mode="lines",
            name=selected_label,
            line={"width": 3, "color": selected_color},
            customdata=np.stack(
                [track["time_local"].dt.strftime("%H:%M"), track["wind16"], track["min_alt_required"], track["alt"]],
                axis=-1,
            ),
            hovertemplate="Az %{theta:.1f} deg<br>Alt %{customdata[3]:.1f} deg<br>Time %{customdata[0]}<br>Dir %{customdata[1]}<br>Obs %{customdata[2]:.0f} deg<extra></extra>",
        )
    )

    selected_direction_theta, selected_direction_r, selected_direction_angles, selected_direction_labels = (
        direction_markers_radial(track, np.asarray(track_r, dtype=float))
    )
    if selected_direction_theta.size:
        fig.add_trace(
            go.Scatterpolar(
                theta=selected_direction_theta,
                r=selected_direction_r,
                mode="markers",
                showlegend=False,
                marker={
                    "size": 12,
                    "symbol": "arrow-right",
                    "color": selected_color,
                    "angle": selected_direction_angles,
                    "line": {"width": 0},
                },
                customdata=np.stack([selected_direction_labels], axis=-1),
                hovertemplate=f"{selected_label}<br>Moving: %{{customdata[0]}}<extra></extra>",
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
            [[label, pd.Timestamp(event["time_local"]).strftime("%H:%M")] for label, event in selected_events],
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

    if set_list_tracks:
        for target_track in set_list_tracks:
            overlay_track = target_track.get("track")
            if not isinstance(overlay_track, pd.DataFrame) or overlay_track.empty:
                continue

            target_label = str(target_track.get("label", "Set List target"))
            target_color = str(target_track.get("color", "#22c55e"))
            overlay_alt = overlay_track["alt"].clip(lower=0.0, upper=90.0)
            overlay_r = (90.0 - overlay_alt) if dome_view else overlay_alt
            overlay_custom = np.stack(
                [
                    np.full(len(overlay_track), target_label, dtype=object),
                    overlay_track["time_local"].dt.strftime("%H:%M"),
                    overlay_track["wind16"].astype(str),
                    overlay_track["alt"],
                ],
                axis=-1,
            )
            fig.add_trace(
                go.Scatterpolar(
                    theta=overlay_track["az"],
                    r=overlay_r,
                    mode="lines",
                    name=target_label,
                    showlegend=False,
                    line={"width": 2.2, "color": target_color},
                    customdata=overlay_custom,
                    hovertemplate="%{customdata[0]}<br>Az %{theta:.1f} deg<br>Alt %{customdata[3]:.1f} deg<br>Time %{customdata[1]}<br>Dir %{customdata[2]}<extra></extra>",
                )
            )

            overlay_direction_theta, overlay_direction_r, overlay_direction_angles, overlay_direction_labels = (
                direction_markers_radial(overlay_track, np.asarray(overlay_r, dtype=float))
            )
            if overlay_direction_theta.size:
                fig.add_trace(
                    go.Scatterpolar(
                        theta=overlay_direction_theta,
                        r=overlay_direction_r,
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": 11,
                            "symbol": "arrow-right",
                            "color": target_color,
                            "angle": overlay_direction_angles,
                            "line": {"width": 0},
                        },
                        customdata=np.stack([overlay_direction_labels], axis=-1),
                        hovertemplate=f"{target_label}<br>Moving: %{{customdata[0]}}<extra></extra>",
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
                    [[label, pd.Timestamp(event["time_local"]).strftime("%H:%M")] for label, event in overlay_events],
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

    fig.update_layout(
        title="Sky Position",
        height=330,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        showlegend=False,
        polar={
            "angularaxis": {
                "rotation": 90,
                "direction": "clockwise",
                "tickmode": "array",
                "tickvals": [i * 22.5 for i in range(16)],
                "ticktext": WIND16,
            },
            "radialaxis": {
                "range": [0, 90],
                "tickmode": "array",
                "tickvals": [0, 30, 60, 90],
                "ticktext": radial_ticktext,
                "title": radial_title,
            },
        },
    )
    return fig


@st.cache_data(show_spinner=False, ttl=15 * 60)
def fetch_hourly_temperatures(
    lat: float, lon: float, tz_name: str, start_local_iso: str, end_local_iso: str
) -> dict[str, float]:
    start_local = pd.Timestamp(start_local_iso)
    end_local = pd.Timestamp(end_local_iso)

    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m",
                "timezone": tz_name,
            },
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()

        hourly_times = payload.get("hourly", {}).get("time", [])
        hourly_temps = payload.get("hourly", {}).get("temperature_2m", [])
        if not hourly_times or not hourly_temps:
            return {}

        mapping: dict[str, float] = {}
        tzinfo = ZoneInfo(tz_name)
        for raw_time, temp in zip(hourly_times, hourly_temps):
            parsed = pd.Timestamp(raw_time).tz_localize(tzinfo)
            if start_local <= parsed <= end_local:
                mapping[parsed.floor("h").isoformat()] = float(temp)

        return mapping
    except Exception:
        return {}


def build_night_plot(
    track: pd.DataFrame,
    temperature_by_hour: dict[str, float],
    temperature_unit: str,
    target_label: str | None = None,
) -> go.Figure:
    if track.empty:
        return go.Figure()

    rows: list[dict[str, Any]] = []
    grouped = track.set_index("time_local").resample("1h")
    for hour, chunk in grouped:
        if chunk.empty:
            continue

        max_row = chunk.loc[chunk["alt"].idxmax()]
        temp = temperature_by_hour.get(hour.isoformat())

        rows.append(
            {
                "hour": hour,
                "hour_label": format_hour_label(hour),
                "max_alt": max(float(max_row["alt"]), 0.0),
                "obstructed_alt": min(
                    max(float(max_row["alt"]), 0.0),
                    max(float(max_row.get("min_alt_required", 0.0)), 0.0),
                ),
                "wind16": str(max_row["wind16"]),
                "temp": temp,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return go.Figure()
    frame["clear_alt"] = (frame["max_alt"] - frame["obstructed_alt"]).clip(lower=0.0)

    labels = []
    for _, row in frame.iterrows():
        temp_str = format_temperature(row["temp"], temperature_unit)
        labels.append(f"{row['wind16']}<br>{temp_str}")

    obstructed_hover = []
    clear_hover = []
    for _, row in frame.iterrows():
        hour_str = pd.Timestamp(row["hour"]).strftime("%H:%M")
        max_alt = float(row["max_alt"])
        obstructed_alt = float(row["obstructed_alt"])
        clear_alt = float(row["clear_alt"])
        obstructed_hover.append(
            f"{row['hour_label']} ({hour_str})<br>Max Alt {max_alt:.1f} deg<br>Obstructed {obstructed_alt:.1f} deg"
        )
        clear_hover.append(
            f"{row['hour_label']} ({hour_str})<br>Max Alt {max_alt:.1f} deg<br>Clear {clear_alt:.1f} deg"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["hour_label"],
            y=frame["obstructed_alt"],
            hovertext=obstructed_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            name="Obstructed",
            marker={"color": "rgba(239, 68, 68, 0.55)"},
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["hour_label"],
            y=frame["clear_alt"],
            hovertext=clear_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            name="Clear",
            marker={"color": "rgba(34, 197, 94, 0.65)"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["hour_label"],
            y=frame["max_alt"] + 1.5,
            mode="text",
            text=labels,
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    title = "Hourly Forecast"
    cleaned_label = str(target_label or "").strip()
    if cleaned_label:
        title = f"Hourly Forecast - {cleaned_label}"

    fig.update_layout(
        title=title,
        height=300,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        barmode="stack",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
        xaxis_title="Hour",
        yaxis_title="Altitude (deg)",
    )
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=frame["hour_label"].tolist())
    fig.update_yaxes(range=[0, 95])
    return fig


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def resolve_manual_location(query: str) -> dict[str, Any] | None:
    cleaned = query.strip()
    if not cleaned:
        return None

    geocoder = Nominatim(user_agent="dso-explorer-prototype")
    attempts = [cleaned]
    if cleaned.isdigit() and len(cleaned) == 5:
        attempts = [f"{cleaned}, USA", cleaned]

    for candidate in attempts:
        try:
            match = geocoder.geocode(candidate, exactly_one=True, timeout=10)
            if match:
                return {
                    "lat": float(match.latitude),
                    "lon": float(match.longitude),
                    "label": match.address.split(",")[0],
                    "source": "manual",
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                }
        except Exception:
            continue

    return None


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def reverse_geocode_label(lat: float, lon: float) -> str:
    geocoder = Nominatim(user_agent="dso-explorer-prototype")

    try:
        match = geocoder.reverse((lat, lon), exactly_one=True, timeout=10)
        if not match:
            return f"{lat:.4f}, {lon:.4f}"

        address = match.raw.get("address", {})
        locality = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("hamlet")
            or address.get("county")
        )
        region = address.get("state")

        parts = [part for part in [locality, region] if part]
        if parts:
            return ", ".join(parts)

        title = str(match.address).split(",")[0].strip()
        return title or f"{lat:.4f}, {lon:.4f}"
    except Exception:
        return f"{lat:.4f}, {lon:.4f}"


def apply_browser_geolocation_payload(prefs: dict[str, Any], payload: Any) -> None:
    try:
        if isinstance(payload, dict):
            coords = payload.get("coords")
            if isinstance(coords, dict):
                lat = float(coords.get("latitude"))
                lon = float(coords.get("longitude"))
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    raise ValueError("Coordinates out of range")

                prefs["location"] = {
                    "lat": lat,
                    "lon": lon,
                    "label": reverse_geocode_label(lat, lon),
                    "source": "browser",
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                }
                st.session_state["prefs"] = prefs
                save_preferences(prefs)
                st.session_state["location_notice"] = "Browser geolocation applied."
                return

            error = payload.get("error")
            if isinstance(error, dict):
                code = str(error.get("code", "")).strip()
                if code == "1":
                    st.session_state["location_notice"] = (
                        "Location permission denied - keeping previous location."
                    )
                elif code == "2":
                    st.session_state["location_notice"] = (
                        "Location unavailable - keeping previous location."
                    )
                elif code == "3":
                    st.session_state["location_notice"] = (
                        "Location request timed out - keeping previous location."
                    )
                else:
                    message = str(error.get("message", "")).strip()
                    detail = f": {message}" if message else "."
                    st.session_state["location_notice"] = (
                        f"Could not resolve browser geolocation{detail} Keeping previous location."
                    )
                return
    except Exception:
        st.session_state["location_notice"] = (
            "Could not parse browser geolocation - keeping previous location."
        )
        return

    st.session_state["location_notice"] = (
        "Could not read browser geolocation response - keeping previous location."
    )


@st.cache_data(show_spinner=False, ttl=15 * 60)
def approximate_location_from_ip() -> dict[str, Any] | None:
    try:
        response = requests.get("https://ipapi.co/json/", timeout=8)
        response.raise_for_status()
        payload = response.json()

        lat = payload.get("latitude")
        lon = payload.get("longitude")
        city = payload.get("city")
        region = payload.get("region")
        if lat is None or lon is None:
            return None

        label_parts = [part for part in [city, region] if part]
        label = ", ".join(label_parts) if label_parts else "IP-based estimate"

        return {
            "lat": float(lat),
            "lon": float(lon),
            "label": label,
            "source": "ip",
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def fetch_free_use_image(search_phrase: str) -> dict[str, str] | None:
    if not search_phrase.strip():
        return None

    try:
        search_response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "list": "search",
                "srnamespace": 6,
                "srlimit": 1,
                "srsearch": search_phrase,
            },
            timeout=10,
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        results = search_data.get("query", {}).get("search", [])
        if not results:
            return None

        title = results[0].get("title")
        if not title:
            return None

        image_response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": 1000,
            },
            timeout=10,
        )
        image_response.raise_for_status()
        image_data = image_response.json()

        pages = image_data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        imageinfo = (page.get("imageinfo") or [{}])[0]
        if not imageinfo:
            return None

        metadata = imageinfo.get("extmetadata", {})
        license_label = metadata.get("LicenseShortName", {}).get("value", "Unknown")

        return {
            "image_url": imageinfo.get("thumburl") or imageinfo.get("url", ""),
            "source_url": imageinfo.get("descriptionurl", ""),
            "license_label": re.sub(r"<[^>]+>", "", license_label),
            "title": title,
        }
    except Exception:
        return None


def add_if_missing(values: list[str], item: str) -> list[str]:
    if item not in values:
        values.append(item)
    return values


def remove_if_present(values: list[str], item: str) -> list[str]:
    return [value for value in values if value != item]


def move_item(values: list[str], index: int, step: int) -> list[str]:
    new_index = index + step
    if new_index < 0 or new_index >= len(values):
        return values
    updated = values.copy()
    updated[index], updated[new_index] = updated[new_index], updated[index]
    return updated


def sanitize_saved_lists(catalog: pd.DataFrame, prefs: dict[str, Any]) -> None:
    available_ids = set(catalog["primary_id"].tolist())
    next_favorites = [item for item in prefs["favorites"] if item in available_ids]
    next_set_list = [item for item in prefs["set_list"] if item in available_ids]

    if next_favorites != prefs["favorites"] or next_set_list != prefs["set_list"]:
        prefs["favorites"] = next_favorites
        prefs["set_list"] = next_set_list
        st.session_state["prefs"] = prefs
        save_preferences(prefs)


def subset_by_id_list(frame: pd.DataFrame, ordered_ids: list[str]) -> pd.DataFrame:
    if not ordered_ids:
        return frame.iloc[0:0].copy()

    id_rank = {identifier: idx for idx, identifier in enumerate(ordered_ids)}
    subset = frame[frame["primary_id"].isin(id_rank)].copy()
    if subset.empty:
        return subset

    subset["_rank"] = subset["primary_id"].map(id_rank)
    subset = subset.sort_values(by="_rank", ascending=True).drop(columns=["_rank"]).reset_index(drop=True)
    return subset


def render_target_table(
    targets: pd.DataFrame,
    *,
    table_key: str,
    empty_message: str,
    auto_select_first: bool = False,
) -> str | None:
    display = targets[["primary_id", "common_name", "catalog", "object_type", "alt_now", "az_now", "wind16"]].copy()
    display = display.rename(
        columns={
            "primary_id": "ID",
            "common_name": "Name",
            "catalog": "Catalog",
            "object_type": "Type",
            "alt_now": "Alt(now)",
            "az_now": "Az(now)",
            "wind16": "Dir",
        }
    )

    table_event = st.dataframe(
        display,
        use_container_width=True,
        height=360,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
    )

    ids = targets["primary_id"].tolist()
    if not ids:
        st.info(empty_message)
        return None

    selected_id = st.session_state.get("selected_id")
    if auto_select_first and (not selected_id or selected_id not in ids):
        st.session_state["selected_id"] = ids[0]

    selected_rows: list[int] = []
    if table_event is not None:
        try:
            selected_rows = list(table_event.selection.rows)
        except Exception:
            if isinstance(table_event, dict):
                selected_rows = list(table_event.get("selection", {}).get("rows", []))

    if selected_rows:
        selected_index = int(selected_rows[0])
        if 0 <= selected_index < len(ids):
            st.session_state["selected_id"] = ids[selected_index]

    selected_id = st.session_state.get("selected_id")
    if selected_id in ids:
        st.caption(f"Selected: {selected_id}")
        return str(selected_id)

    return None


def render_main_tabs(catalog: pd.DataFrame, prefs: dict[str, Any]) -> None:
    with st.container(border=True):
        st.subheader("Targets")
        query = st.text_input(
            "Search (M / NGC / IC / Sh2 + common names)",
            placeholder="M31, NGC 7000, IC1805, Sh2-132, Orion Nebula",
            key="targets_search_query",
        )
        location = prefs["location"]
        filtered = search_catalog(catalog, query)
        results = compute_altaz_now(filtered, lat=float(location["lat"]), lon=float(location["lon"]))
        favorites = compute_altaz_now(
            subset_by_id_list(catalog, prefs["favorites"]),
            lat=float(location["lat"]),
            lon=float(location["lon"]),
        )
        set_list = compute_altaz_now(
            subset_by_id_list(catalog, prefs["set_list"]),
            lat=float(location["lat"]),
            lon=float(location["lon"]),
        )

        tab_results, tab_favorites, tab_set_list = st.tabs(
            [
                f"Results ({len(results)})",
                f"Favorites ({len(favorites)})",
                f"Set List ({len(set_list)})",
            ]
        )

        with tab_results:
            st.caption("Click a row to open target detail.")
            render_target_table(
                results,
                table_key="results_table",
                empty_message="No targets matched that search.",
                auto_select_first=True,
            )

        with tab_favorites:
            render_target_table(
                favorites,
                table_key="favorites_table",
                empty_message="No favorites yet.",
            )

        with tab_set_list:
            selected_in_set = render_target_table(
                set_list,
                table_key="set_list_table",
                empty_message="Set list is empty.",
            )

            if not set_list.empty:
                active_id = selected_in_set or st.session_state.get("selected_id")
                if active_id in prefs["set_list"]:
                    active_idx = prefs["set_list"].index(str(active_id))
                    controls = st.columns(3)
                    if controls[0].button(
                        "Move Up",
                        disabled=active_idx == 0,
                        key="set_list_move_up_main",
                        use_container_width=True,
                    ):
                        prefs["set_list"] = move_item(prefs["set_list"], active_idx, -1)
                        persist_and_rerun(prefs)
                    if controls[1].button(
                        "Move Down",
                        disabled=active_idx == len(prefs["set_list"]) - 1,
                        key="set_list_move_down_main",
                        use_container_width=True,
                    ):
                        prefs["set_list"] = move_item(prefs["set_list"], active_idx, 1)
                        persist_and_rerun(prefs)
                    if controls[2].button(
                        "Remove",
                        key="set_list_remove_main",
                        use_container_width=True,
                    ):
                        prefs["set_list"] = remove_if_present(prefs["set_list"], str(active_id))
                        persist_and_rerun(prefs)


def resolve_selected_row(catalog: pd.DataFrame, prefs: dict[str, Any]) -> pd.Series | None:
    selected_id = st.session_state.get("selected_id")
    if not selected_id:
        return None

    target = catalog[catalog["primary_id"] == selected_id]
    if target.empty:
        return None

    location = prefs["location"]
    enriched = compute_altaz_now(target.iloc[:1], lat=float(location["lat"]), lon=float(location["lon"]))
    if enriched.empty:
        return None
    return enriched.iloc[0]


def render_sidebar(catalog_meta: dict[str, Any], prefs: dict[str, Any], browser_locale: str | None) -> None:
    with st.sidebar:
        st.header("Controls")

        persistence_notice = st.session_state.get("prefs_persistence_notice", "")
        if persistence_notice:
            st.warning(persistence_notice)

        location_notice = st.session_state.pop("location_notice", "")
        if location_notice:
            st.info(location_notice)

        st.subheader("Location")
        location = prefs["location"]
        st.markdown(f"**{location['label']}**")
        st.caption(f"Lat {location['lat']:.4f}, Lon {location['lon']:.4f} ({location['source']})")

        manual_location = st.text_input("Manual ZIP / place", key="manual_location")
        if st.button("Resolve location", use_container_width=True):
            resolved = resolve_manual_location(manual_location)
            if resolved:
                prefs["location"] = resolved
                persist_and_rerun(prefs)
            else:
                st.warning("Couldn't find that location - keeping previous location.")

        if st.button("Use browser location (permission)", use_container_width=True):
            st.session_state["request_browser_geo"] = True
            st.session_state["browser_geo_request_id"] = int(st.session_state.get("browser_geo_request_id", 0)) + 1

        if st.session_state.get("request_browser_geo"):
            request_id = int(st.session_state.get("browser_geo_request_id", 1))
            geolocation_payload = get_geolocation(component_key=f"browser_geo_request_{request_id}")
            if geolocation_payload is None:
                st.caption("Requesting browser geolocation permission...")
            else:
                apply_browser_geolocation_payload(prefs, geolocation_payload)
                st.session_state["request_browser_geo"] = False
                st.rerun()

        if st.button("Use my location (IP fallback)", use_container_width=True):
            resolved = approximate_location_from_ip()
            if resolved:
                prefs["location"] = resolved
                persist_and_rerun(prefs)
            else:
                st.warning("Location unavailable - keeping previous location.")

        st.subheader("Display")
        labels = list(TEMPERATURE_UNIT_OPTIONS.keys())
        reverse_options = {value: label for label, value in TEMPERATURE_UNIT_OPTIONS.items()}
        current_pref = str(prefs.get("temperature_unit", "auto")).lower()
        if current_pref not in reverse_options:
            current_pref = "auto"
        selected_label = st.selectbox(
            "Temperature units",
            options=labels,
            index=labels.index(reverse_options[current_pref]),
            key="temperature_unit_preference",
        )
        selected_pref = TEMPERATURE_UNIT_OPTIONS[selected_label]
        if selected_pref != current_pref:
            prefs["temperature_unit"] = selected_pref
            persist_and_rerun(prefs)

        effective_unit = resolve_temperature_unit(selected_pref, browser_locale)
        source_note = "browser locale" if selected_pref == "auto" else "manual setting"
        st.caption(f"Active temperature unit: {effective_unit.upper()} ({source_note})")

        st.subheader("Settings Backup")
        export_payload = build_settings_export_payload(prefs)
        export_text = json.dumps(export_payload, indent=2)
        export_filename = f"dso-explorer-settings-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
        st.download_button(
            "Export settings JSON",
            data=export_text,
            file_name=export_filename,
            mime="application/json",
            use_container_width=True,
        )

        uploaded_settings = st.file_uploader(
            "Import settings JSON",
            type=["json"],
            key="settings_import_file",
            help="Imports location, obstructions, favorites, set list, and display preferences.",
        )
        if st.button("Import settings", use_container_width=True, key="settings_import_apply"):
            if uploaded_settings is None:
                st.warning("Choose a JSON file first.")
            else:
                raw_bytes = uploaded_settings.getvalue()
                try:
                    raw_text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    st.warning("Could not read that file as UTF-8 JSON.")
                else:
                    imported_prefs = parse_settings_import_payload(raw_text)
                    if imported_prefs is None:
                        st.warning("Invalid settings file format.")
                    else:
                        st.session_state["location_notice"] = "Settings imported."
                        persist_and_rerun(imported_prefs)

        st.subheader("Catalog Ingestion")
        st.caption(
            f"Rows: {int(catalog_meta.get('row_count', 0))} | "
            f"Mode: {catalog_meta.get('load_mode', 'unknown')}"
        )
        catalog_counts = catalog_meta.get("catalog_counts", {})
        if isinstance(catalog_counts, dict) and catalog_counts:
            counts_line = " | ".join(f"{catalog_name}: {count}" for catalog_name, count in sorted(catalog_counts.items()))
            st.caption(counts_line)
        loaded_at = str(catalog_meta.get("loaded_at_utc", "")).strip()
        if loaded_at:
            st.caption(f"Loaded: {loaded_at}")
        if st.button("Refresh catalog cache", use_container_width=True):
            st.session_state["force_catalog_refresh"] = True
            st.rerun()

        st.subheader("Obstructions (16-bin)")
        obstruction_frame = pd.DataFrame(
            {
                "Direction": WIND16,
                "Min Altitude (deg)": [prefs["obstructions"].get(direction, 20.0) for direction in WIND16],
            }
        )
        edited = st.data_editor(
            obstruction_frame,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            disabled=["Direction"],
            key="obstruction_editor",
        )

        edited_values = edited["Min Altitude (deg)"].tolist()
        next_obstructions = {
            direction: float(max(0.0, min(90.0, edited_values[idx]))) for idx, direction in enumerate(WIND16)
        }
        if next_obstructions != prefs["obstructions"]:
            prefs["obstructions"] = next_obstructions
            save_preferences(prefs)


def render_detail_panel(
    selected: pd.Series | None,
    catalog: pd.DataFrame,
    prefs: dict[str, Any],
    temperature_unit: str,
) -> None:
    with st.container(border=True):
        st.subheader("Detail")

        if selected is None:
            st.info("Select a target from results to view detail and plots.")
            st.plotly_chart(go.Figure(), use_container_width=True)
            st.plotly_chart(go.Figure(), use_container_width=True)
            return

        target_id = str(selected["primary_id"])
        is_favorite = target_id in prefs["favorites"]
        in_set_list = target_id in prefs["set_list"]

        header_cols = st.columns([2, 1, 1])
        title = target_id
        if selected.get("common_name"):
            title = f"{target_id} - {selected['common_name']}"
        header_cols[0].markdown(f"### {title}")

        favorite_label = "Remove Favorite" if is_favorite else "Add Favorite"
        if header_cols[1].button(favorite_label, use_container_width=True):
            prefs["favorites"] = (
                remove_if_present(prefs["favorites"], target_id)
                if is_favorite
                else add_if_missing(prefs["favorites"], target_id)
            )
            persist_and_rerun(prefs)

        set_list_label = "Remove Set List" if in_set_list else "Add to Set List"
        if header_cols[2].button(set_list_label, use_container_width=True):
            prefs["set_list"] = (
                remove_if_present(prefs["set_list"], target_id)
                if in_set_list
                else add_if_missing(prefs["set_list"], target_id)
            )
            persist_and_rerun(prefs)

        st.caption(f"Catalog: {selected['catalog']} | Type: {selected.get('object_type') or '-'}")

        image_cols = st.columns([1.3, 1])
        search_phrase = selected.get("common_name") or selected.get("primary_id")
        image_data = fetch_free_use_image(str(search_phrase))

        with image_cols[0]:
            if image_data and image_data.get("image_url"):
                st.image(image_data["image_url"], use_container_width=True)
                attribution = image_data.get("source_url")
                license_label = image_data.get("license_label", "Unknown")
                if attribution:
                    st.caption(f"Source: [Wikimedia Commons]({attribution}) | License: {license_label}")
                else:
                    st.caption(f"License: {license_label}")
            else:
                st.info("No free-use image found for this target.")

        with image_cols[1]:
            property_rows = pd.DataFrame(
                [
                    {
                        "Property": "RA / Dec",
                        "Value": f"{float(selected['ra_deg']):.4f} deg / {float(selected['dec_deg']):.4f} deg",
                    },
                    {"Property": "Constellation", "Value": str(selected.get("constellation") or "-")},
                    {
                        "Property": "Alt / Az (now)",
                        "Value": f"{float(selected['alt_now']):.1f} deg / {float(selected['az_now']):.1f} deg",
                    },
                    {"Property": "16-wind", "Value": str(selected["wind16"])},
                ]
            )
            st.dataframe(property_rows, hide_index=True, use_container_width=True, height=180)

        location = prefs["location"]
        window_start, window_end, tzinfo = tonight_window(location["lat"], location["lon"])
        selected_common_name = str(selected.get("common_name") or "").strip()
        selected_label = f"{target_id} - {selected_common_name}" if selected_common_name else target_id
        selected_color = PATH_LINE_COLORS[0]

        set_list_tracks: list[dict[str, Any]] = []
        set_list_targets = subset_by_id_list(catalog, prefs["set_list"])
        color_cursor = 1
        for _, set_list_target in set_list_targets.iterrows():
            set_list_target_id = str(set_list_target["primary_id"])
            if set_list_target_id == target_id:
                continue

            try:
                set_list_ra = float(set_list_target["ra_deg"])
                set_list_dec = float(set_list_target["dec_deg"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(set_list_ra) or not np.isfinite(set_list_dec):
                continue

            try:
                set_list_track = compute_track(
                    ra_deg=set_list_ra,
                    dec_deg=set_list_dec,
                    lat=float(location["lat"]),
                    lon=float(location["lon"]),
                    start_local=window_start,
                    end_local=window_end,
                    obstructions=prefs["obstructions"],
                )
            except Exception:
                continue
            if set_list_track.empty:
                continue

            set_list_common_name = str(set_list_target.get("common_name") or "").strip()
            set_list_label = (
                f"{set_list_target_id} - {set_list_common_name}" if set_list_common_name else set_list_target_id
            )
            set_list_tracks.append(
                {
                    "primary_id": set_list_target_id,
                    "common_name": set_list_common_name,
                    "label": set_list_label,
                    "color": PATH_LINE_COLORS[color_cursor % len(PATH_LINE_COLORS)],
                    "track": set_list_track,
                    "events": extract_events(set_list_track),
                }
            )
            color_cursor += 1

        try:
            selected_ra = float(selected["ra_deg"])
            selected_dec = float(selected["dec_deg"])
        except (TypeError, ValueError):
            st.warning("Selected target is missing valid coordinates, so path/forecast plots are unavailable.")
            return
        if not np.isfinite(selected_ra) or not np.isfinite(selected_dec):
            st.warning("Selected target has non-finite coordinates, so path/forecast plots are unavailable.")
            return

        track = compute_track(
            ra_deg=selected_ra,
            dec_deg=selected_dec,
            lat=float(location["lat"]),
            lon=float(location["lon"]),
            start_local=window_start,
            end_local=window_end,
            obstructions=prefs["obstructions"],
        )
        events = extract_events(track)

        st.caption(
            f"Tonight ({tzinfo.key}): {window_start.strftime('%H:%M')} -> {window_end.strftime('%H:%M')} | "
            f"Rise {format_time(events['rise'])} | First-visible {format_time(events['first_visible'])} | "
            f"Culmination {format_time(events['culmination'])} | Last-visible {format_time(events['last_visible'])}"
        )

        path_style = st.segmented_control(
            "Sky Position Style",
            options=["Line", "Radial"],
            default="Line",
            key="path_style_preference",
        )
        if path_style == "Radial":
            dome_view = st.toggle("Dome View", value=True, key="dome_view_preference")
            path_figure = build_path_plot_radial(
                track=track,
                events=events,
                obstructions=prefs["obstructions"],
                dome_view=dome_view,
                selected_label=selected_label,
                selected_color=selected_color,
                set_list_tracks=set_list_tracks,
            )
        else:
            path_figure = build_path_plot(
                track=track,
                events=events,
                obstructions=prefs["obstructions"],
                selected_label=selected_label,
                selected_color=selected_color,
                set_list_tracks=set_list_tracks,
            )

        st.plotly_chart(
            path_figure,
            use_container_width=True,
        )
        summary_rows = build_sky_position_summary_rows(
            selected_id=target_id,
            selected_label=selected_label,
            selected_color=selected_color,
            selected_events=events,
            selected_track=track,
            set_list_tracks=set_list_tracks,
            pinned_ids=set(str(item) for item in prefs["set_list"]),
        )
        render_sky_position_summary_table(summary_rows, prefs)

        temperatures = fetch_hourly_temperatures(
            lat=float(location["lat"]),
            lon=float(location["lon"]),
            tz_name=tzinfo.key,
            start_local_iso=window_start.isoformat(),
            end_local_iso=window_end.isoformat(),
        )
        st.plotly_chart(
            build_night_plot(
                track=track,
                temperature_by_hour=temperatures,
                temperature_unit=temperature_unit,
                target_label=selected_label,
            ),
            use_container_width=True,
        )


def main() -> None:
    st_autorefresh(interval=60_000, key="altaz_refresh")

    if "prefs_bootstrap_runs" not in st.session_state:
        st.session_state["prefs_bootstrap_runs"] = 0
    if "prefs" not in st.session_state:
        st.session_state["prefs"] = default_preferences()

    if int(st.session_state.get("prefs_bootstrap_runs", 0)) < 2:
        st.session_state["prefs"] = load_preferences()
        st.session_state["prefs_bootstrap_runs"] = int(st.session_state.get("prefs_bootstrap_runs", 0)) + 1
        if int(st.session_state.get("prefs_bootstrap_runs", 0)) < 2:
            st.rerun()

    prefs = ensure_preferences_shape(st.session_state["prefs"])
    st.session_state["prefs"] = prefs

    browser_language_raw = get_browser_language(component_key="browser_language_pref")
    if isinstance(browser_language_raw, str) and browser_language_raw.strip():
        st.session_state["browser_language"] = browser_language_raw.strip()
    browser_language = st.session_state.get("browser_language")
    effective_temperature_unit = resolve_temperature_unit(
        str(prefs.get("temperature_unit", "auto")),
        browser_language,
    )

    force_catalog_refresh = bool(st.session_state.pop("force_catalog_refresh", False))
    catalog, catalog_meta = load_catalog(force_refresh=force_catalog_refresh)
    sanitize_saved_lists(catalog=catalog, prefs=prefs)
    render_sidebar(catalog_meta=catalog_meta, prefs=prefs, browser_locale=browser_language)

    now_utc = datetime.now(timezone.utc)
    st.markdown(
        """
        <style>
            .small-note {font-size: 0.9rem; color: #666;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    header_cols = st.columns([3, 1])
    header_cols[0].title("DSO Explorer")
    header_cols[1].markdown(
        f"<p class='small-note'>Alt/Az auto-refresh 60s<br>Updated: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>",
        unsafe_allow_html=True,
    )
    st.caption(f"Catalog rows loaded: {int(catalog_meta.get('row_count', len(catalog)))}")

    use_phone_layout = st.toggle("Phone layout preview", value=False)

    if use_phone_layout:
        render_main_tabs(catalog, prefs)
        selected_row = resolve_selected_row(catalog, prefs)
        with st.container(border=True):
            st.markdown("### Detail bottom sheet")
            render_detail_panel(
                selected=selected_row,
                catalog=catalog,
                prefs=prefs,
                temperature_unit=effective_temperature_unit,
            )
    else:
        result_col, detail_col = st.columns([35, 65])
        with result_col:
            render_main_tabs(catalog, prefs)

        selected_row = resolve_selected_row(catalog, prefs)
        with detail_col:
            render_detail_panel(
                selected=selected_row,
                catalog=catalog,
                prefs=prefs,
                temperature_unit=effective_temperature_unit,
            )


if __name__ == "__main__":
    main()
