from __future__ import annotations

import base64
import copy
import hashlib
import html
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
from catalog_service import (
    CATALOG_MODE_CURATED_PARQUET,
    CATALOG_MODE_LEGACY,
    get_object_by_id,
    load_catalog_data,
    search_catalog,
)
from list_subsystem import (
    AUTO_RECENT_LIST_ID,
    all_listed_ids_in_order,
    clean_primary_id_list,
    default_list_meta,
    default_list_order,
    default_lists_payload,
    editable_list_ids_in_order,
    get_active_preview_list_id,
    get_list_ids,
    get_list_name,
    is_system_list,
    list_ids_in_order,
    normalize_list_preferences,
    push_target_to_auto_recent_list,
    set_active_preview_list_id,
    toggle_target_in_list,
)
from list_search import searchbox_target_options, subset_by_id_list
from list_settings_ui import render_lists_settings_section
from geopy.geocoders import ArcGIS, Nominatim, Photon
try:
    from streamlit_searchbox import st_searchbox
except Exception:
    st_searchbox = None
try:
    from streamlit_vertical_slider import vertical_slider
except Exception:
    try:
        from streamlit_extras.vertical_slider import vertical_slider
    except Exception:
        vertical_slider = None
try:
    from streamlit_extras.let_it_rain import rain as let_it_rain
except Exception:
    let_it_rain = None
from streamlit_js_eval import (
    get_browser_language,
    get_geolocation,
    get_local_storage,
    set_local_storage,
    streamlit_js_eval,
)
from streamlit_autorefresh import st_autorefresh
from prefs_cookie_backup import (
    bootstrap_cookie_backup,
    get_cookie_backup_notice,
    read_preferences_cookie_backup,
    set_cookie_backup_runtime_enabled,
    write_preferences_cookie_backup,
)
from target_tips.ui import render_target_tips_panel
from timezonefinder import TimezoneFinder
from weather_service import (
    EXTENDED_FORECAST_HOURLY_FIELDS,
    fetch_hourly_weather,
    format_precipitation,
    format_snowfall,
    format_temperature,
    format_wind_speed,
    resolve_temperature_unit,
)

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
WIND16_ARROWS = {
    "N": "↑",
    "NNE": "↑",
    "NE": "↗",
    "ENE": "↗",
    "E": "→",
    "ESE": "↘",
    "SE": "↘",
    "SSE": "↓",
    "S": "↓",
    "SSW": "↓",
    "SW": "↙",
    "WSW": "↙",
    "W": "←",
    "WNW": "↖",
    "NW": "↖",
    "NNW": "↑",
}

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
CURATED_CATALOG_PATH = Path("data/dso_catalog_curated.parquet")
CATALOG_ENRICHED_PATH = Path("data/DSO_CATALOG_ENRICHED.CSV")
# Catalog rollout feature flag:
# - "legacy": current loader (`catalog_ingestion.load_unified_catalog`)
# - "curated_parquet": read directly from `CURATED_CATALOG_PATH` and fallback to legacy on validation/load errors
CATALOG_LOADER_MODES = (CATALOG_MODE_LEGACY, CATALOG_MODE_CURATED_PARQUET)
CATALOG_LOADER_MODE = CATALOG_MODE_LEGACY
BROWSER_PREFS_STORAGE_KEY = "dso_explorer_prefs_v2"
ENABLE_COOKIE_BACKUP = True
PREFS_BOOTSTRAP_MAX_RUNS = 6
PREFS_BOOTSTRAP_RETRY_INTERVAL_MS = 250
SETTINGS_EXPORT_FORMAT_VERSION = 2

TEMPERATURE_UNIT_OPTIONS = {
    "Auto (browser)": "auto",
    "Fahrenheit": "f",
    "Celsius": "c",
}
WEATHER_MATRIX_ROWS: list[tuple[str, str]] = [
    ("temperature_2m", "Temperature"),
    ("cloud_cover", "Cloud Cover"),
    ("wind_gusts_10m", "Wind Gusts"),
    ("relative_humidity_2m", "Rel Humidity"),
]
WEATHER_ALERT_INDICATOR_LEGEND_ITEMS = "❄️ Snow | ⛈️ Rain | ☔ Showers | ⚠️ 1-20% | 🚨 >20%"
WEATHER_ALERT_INDICATOR_LEGEND_CAPTION = f"Weather Alert Indicator: {WEATHER_ALERT_INDICATOR_LEGEND_ITEMS}"
WEATHER_ALERT_RAIN_PRIORITY = ["❄️", "⛈️", "☔", "🚨", "⚠️"]
WEATHER_ALERT_RAIN_INTERVAL_SECONDS = 5 * 60
WEATHER_ALERT_RAIN_DURATION_SECONDS = 30
WEATHER_ALERT_RAIN_BUCKET_STATE_KEY = "weather_alert_rain_last_bucket"
OBJECT_TYPE_GROUP_COLOR_DEFAULT = "#94A3B8"
OBJECT_TYPE_GROUP_COLOR_RANGE_THEMES: dict[str, dict[str, tuple[str, str]]] = {
    "default": {
        "Galaxies": ("#8B5CF6", "#EC4899"),      # purple -> pink
        "Clusters": ("#16A34A", "#EAB308"),      # green -> yellow
        "Bright Nebula": ("#DC2626", "#F97316"), # red -> orange
        "Dark Nebula": ("#2563EB", "#F97316"),   # blue -> orange
        "Stars": ("#EAB308", "#16A34A"),         # yellow -> green
        "other": ("#94A3B8", "#94A3B8"),         # grey
    }
}
# Approximate cloud-cover legend gradient inspired by the provided scale:
# Overcast/90% -> light gray, 70-50% -> cyan/blue, 30-0% -> deeper blue.
CLOUD_COVER_COLOR_STOPS: list[tuple[float, str]] = [
    (0.0, "#0B2A70"),
    (10.0, "#194896"),
    (20.0, "#245EAF"),
    (30.0, "#3278C5"),
    (40.0, "#4D95D7"),
    (50.0, "#67B6E6"),
    (60.0, "#83CFEA"),
    (70.0, "#9BE6EE"),
    (80.0, "#BBC9D2"),
    (90.0, "#CAD1D6"),
    (100.0, "#D8D8D8"),
]
OBSTRUCTION_SLIDER_COLOR_STOPS: list[tuple[float, str]] = [
    (0.0, "#22C55E"),
    (20.0, "#EAB308"),
    (50.0, "#EF4444"),
    (90.0, "#EF4444"),
]
TEMPERATURE_COLOR_STOPS_F: list[tuple[float, str]] = [
    (0.0, "#0B2A70"),
    (20.0, "#1E4F9C"),
    (40.0, "#2D7FC1"),
    (60.0, "#66A9D9"),
    (75.0, "#F2C96D"),
    (88.0, "#F28C45"),
    (100.0, "#C62828"),
]
HOUR12_COUNTRY_CODES = {
    "AU",
    "CA",
    "CO",
    "EG",
    "HN",
    "IN",
    "IE",
    "MY",
    "NZ",
    "PH",
    "PK",
    "SA",
    "SV",
    "US",
}
EVENT_LABELS: list[tuple[str, str]] = [
    ("first_visible", "First"),
    ("last_visible", "Last"),
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
OBSTRUCTION_FILL_COLOR = "rgba(181, 186, 192, 0.40)"
OBSTRUCTION_LINE_COLOR = "rgba(148, 163, 184, 0.95)"
UNOBSTRUCTED_AREA_CONSTANT_OBSTRUCTION_ALT_DEG = 20.0
CARDINAL_GRIDLINE_COLOR = "rgba(100, 116, 139, 0.45)"
DETAIL_PANE_STACK_BREAKPOINT_PX = 800
PATH_PLOT_BACKGROUND_COLOR = "#E2F0FB"
PATH_PLOT_HORIZONTAL_GRID_COLOR = "rgba(255, 255, 255, 0.95)"
PATH_DIRECTION_ARROW_COLOR = "#111111"
PATH_DIRECTION_ARROW_SIZE_PRIMARY = 10
PATH_DIRECTION_ARROW_SIZE_OVERLAY = 9
PATH_DIRECTION_MARKERS_PRIMARY = 4
PATH_DIRECTION_MARKERS_OVERLAY = 3
PATH_ENDPOINT_MARKER_SIZE_PRIMARY = 13
PATH_ENDPOINT_MARKER_SIZE_OVERLAY = 11
PATH_HIGHLIGHT_WIDTH_MULTIPLIER = 5.0
PATH_LINE_WIDTH_PRIMARY_DEFAULT = 3.0
PATH_LINE_WIDTH_OVERLAY_DEFAULT = 2.2
PATH_LINE_WIDTH_SELECTION_MULTIPLIER = 3.0


def resolve_object_type_group_color_ranges(theme_name: str | None = None) -> dict[str, tuple[str, str]]:
    selected_theme = str(theme_name or "").strip().lower() or "default"
    theme_colors = (
        OBJECT_TYPE_GROUP_COLOR_RANGE_THEMES.get(selected_theme)
        or OBJECT_TYPE_GROUP_COLOR_RANGE_THEMES["default"]
    )
    return dict(theme_colors)


def _blend_hex_colors(start_hex: str, end_hex: str, t: float) -> str:
    clamped_t = max(0.0, min(1.0, float(t)))
    try:
        start_r, start_g, start_b = _hex_to_rgb(start_hex)
    except Exception:
        start_r, start_g, start_b = _hex_to_rgb(OBJECT_TYPE_GROUP_COLOR_DEFAULT)
    try:
        end_r, end_g, end_b = _hex_to_rgb(end_hex)
    except Exception:
        end_r, end_g, end_b = (start_r, start_g, start_b)

    red = int(round(start_r + (end_r - start_r) * clamped_t))
    green = int(round(start_g + (end_g - start_g) * clamped_t))
    blue = int(round(start_b + (end_b - start_b) * clamped_t))
    return _rgb_to_hex((red, green, blue))


def object_type_group_color(
    group_label: str | None,
    *,
    step_fraction: float = 0.0,
    theme_name: str | None = None,
) -> str:
    group = str(group_label or "").strip()
    ranges = resolve_object_type_group_color_ranges(theme_name)
    start_end = ranges.get(group) or ranges.get("other") or (OBJECT_TYPE_GROUP_COLOR_DEFAULT, OBJECT_TYPE_GROUP_COLOR_DEFAULT)
    start_color, end_color = start_end
    return _blend_hex_colors(start_color, end_color, step_fraction)


def target_line_color(primary_id: str) -> str:
    cleaned = str(primary_id or "").strip().upper()
    if not cleaned:
        return PATH_LINE_COLORS[0]
    digest = hashlib.md5(cleaned.encode("utf-8")).digest()
    palette_index = int.from_bytes(digest[:4], byteorder="big", signed=False) % len(PATH_LINE_COLORS)
    return PATH_LINE_COLORS[palette_index]


def default_preferences() -> dict[str, Any]:
    return {
        "lists": default_lists_payload(),
        "list_order": default_list_order(),
        "list_meta": default_list_meta(),
        "active_preview_list_id": AUTO_RECENT_LIST_ID,
        "obstructions": {direction: 20.0 for direction in WIND16},
        "location": copy.deepcopy(DEFAULT_LOCATION),
        "temperature_unit": "auto",
    }


def ensure_preferences_shape(raw: dict[str, Any]) -> dict[str, Any]:
    prefs = default_preferences()
    if isinstance(raw, dict):
        prefs.update(normalize_list_preferences(raw))

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


# Optional cookie backup is isolated in `prefs_cookie_backup.py`.
# Toggle `ENABLE_COOKIE_BACKUP` above to disable it without code removal.
# Full removal later: delete the `prefs_cookie_backup` import, remove the
# integration calls below, remove `extra-streamlit-components` from requirements,
# and delete `prefs_cookie_backup.py`.


def load_preferences() -> tuple[dict[str, Any], bool]:
    retry_needed = False
    raw_local = get_local_storage(BROWSER_PREFS_STORAGE_KEY, component_key="browser_prefs_read")
    local_exists_probe = streamlit_js_eval(
        js_expressions=(
            "Object.prototype.hasOwnProperty.call(window.localStorage, "
            + json.dumps(BROWSER_PREFS_STORAGE_KEY)
            + ")"
        ),
        key="browser_prefs_local_exists_probe",
    )

    if raw_local is None and local_exists_probe is None:
        retry_needed = True

    if isinstance(raw_local, str) and raw_local.strip():
        decoded = decode_preferences_from_storage(raw_local)
        if decoded is not None:
            st.session_state.pop("prefs_persistence_notice", None)
            return decoded, False
    elif local_exists_probe is True:
        # Local key appears to exist but returned value is unavailable in this pass.
        retry_needed = True

    raw_cookie = read_preferences_cookie_backup()
    if isinstance(raw_cookie, str) and raw_cookie.strip():
        decoded_cookie = decode_preferences_from_storage(raw_cookie)
        if decoded_cookie is not None:
            try:
                payload_hash = hashlib.sha1(raw_cookie.encode("ascii")).hexdigest()[:12]
                set_local_storage(
                    BROWSER_PREFS_STORAGE_KEY,
                    raw_cookie,
                    component_key=f"browser_prefs_rehydrate_{payload_hash}",
                )
            except Exception:
                pass

            st.session_state.pop("prefs_persistence_notice", None)
            return decoded_cookie, False

    return default_preferences(), retry_needed


def save_preferences(prefs: dict[str, Any]) -> bool:
    try:
        encoded = encode_preferences_for_storage(prefs)
    except Exception:
        st.session_state["prefs_persistence_notice"] = (
            "Browser-local preference storage is unavailable. Using session-only preferences."
        )
        return False

    payload_hash = hashlib.sha1(encoded.encode("ascii")).hexdigest()[:12]
    local_saved = False
    try:
        set_local_storage(
            BROWSER_PREFS_STORAGE_KEY,
            encoded,
            component_key=f"browser_prefs_write_{payload_hash}",
        )
        local_saved = True
    except Exception:
        local_saved = False

    cookie_saved = write_preferences_cookie_backup(encoded)

    if local_saved:
        st.session_state.pop("prefs_persistence_notice", None)
        return True

    if cookie_saved:
        st.session_state["prefs_persistence_notice"] = (
            "Browser-local preference storage is unavailable. Using cookie backup preferences."
        )
        return True

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


def infer_12_hour_clock_from_locale(locale_value: str | None) -> bool:
    if not locale_value:
        return False

    normalized = str(locale_value).replace("-", "_")
    parts = [part for part in normalized.split("_") if part]
    if len(parts) < 2:
        return False

    candidate = parts[-1].upper()
    if len(candidate) == 2 and candidate.isalpha():
        return candidate in HOUR12_COUNTRY_CODES
    return False


def resolve_12_hour_clock(locale_value: str | None, hour_cycle_value: str | None) -> bool:
    cycle = str(hour_cycle_value or "").strip().lower()
    if cycle in {"h11", "h12"}:
        return True
    if cycle in {"h23", "h24"}:
        return False
    return infer_12_hour_clock_from_locale(locale_value)


def normalize_12_hour_label(value: str) -> str:
    cleaned = str(value).strip()
    if cleaned.startswith("0"):
        cleaned = cleaned[1:]
    return cleaned.replace("AM", "am").replace("PM", "pm")


def format_display_time(value: pd.Timestamp | datetime, use_12_hour: bool, include_seconds: bool = False) -> str:
    timestamp = pd.Timestamp(value)
    if include_seconds:
        fmt = "%I:%M:%S %p" if use_12_hour else "%H:%M:%S"
    else:
        fmt = "%I:%M %p" if use_12_hour else "%H:%M"
    rendered = timestamp.strftime(fmt)
    return normalize_12_hour_label(rendered) if use_12_hour else rendered


def az_to_wind16(az_deg: float) -> str:
    idx = int(((az_deg % 360.0) + 11.25) // 22.5) % 16
    return WIND16[idx]


def wind_bin_center(direction: str) -> float:
    idx = WIND16.index(direction)
    return idx * 22.5


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
        sun_yesterday = sun(loc.observer, date=base_date - timedelta(days=1), tzinfo=tzinfo)
        sun_today = sun(loc.observer, date=base_date, tzinfo=tzinfo)
        sun_tomorrow = sun(loc.observer, date=base_date + timedelta(days=1), tzinfo=tzinfo)
        today_sunrise = sun_today["sunrise"]

        # Keep showing the in-progress observing night after midnight
        # until local sunrise.
        if local_now < today_sunrise:
            start = sun_yesterday["sunset"]
            end = today_sunrise
        else:
            start = sun_today["sunset"]
            end = sun_tomorrow["sunrise"]
    except Exception:
        fallback_sunrise_today = datetime.combine(base_date, time(6, 0), tzinfo=tzinfo)
        if local_now < fallback_sunrise_today:
            start = datetime.combine(base_date - timedelta(days=1), time(18, 0), tzinfo=tzinfo)
            end = fallback_sunrise_today
        else:
            start = datetime.combine(base_date, time(18, 0), tzinfo=tzinfo)
            end = datetime.combine(base_date + timedelta(days=1), time(6, 0), tzinfo=tzinfo)

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


def format_time(series: pd.Series | None, use_12_hour: bool) -> str:
    if series is None:
        return "--"
    return format_display_time(pd.Timestamp(series["time_local"]), use_12_hour=use_12_hour)


def event_time_value(series: pd.Series | None) -> pd.Timestamp | pd.NaTType:
    if series is None:
        return pd.NaT
    try:
        return pd.Timestamp(series["time_local"])
    except Exception:
        return pd.NaT


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


def compute_remaining_visible_time(track: pd.DataFrame, now: pd.Timestamp | datetime | None = None) -> timedelta:
    if track.empty or "visible" not in track:
        return timedelta(0)

    visible_mask = track["visible"].fillna(False).astype(bool).to_numpy()
    if not visible_mask.any():
        return timedelta(0)

    times = pd.to_datetime(track["time_local"])
    if times.empty:
        return timedelta(0)

    diffs = times.diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    step = positive_diffs.median() if not positive_diffs.empty else pd.Timedelta(minutes=10)

    first_time = pd.Timestamp(times.iloc[0])
    if now is None:
        now_ts = pd.Timestamp.now(tz=first_time.tzinfo) if first_time.tzinfo is not None else pd.Timestamp.now()
    else:
        now_ts = pd.Timestamp(now)
        if first_time.tzinfo is not None and now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize(first_time.tzinfo)
        elif first_time.tzinfo is None and now_ts.tzinfo is not None:
            now_ts = now_ts.tz_localize(None)
        elif first_time.tzinfo is not None and now_ts.tzinfo is not None:
            now_ts = now_ts.tz_convert(first_time.tzinfo)

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
        interval_start = pd.Timestamp(times.iloc[start_idx])
        interval_end = pd.Timestamp(times.iloc[end_idx]) + step
        effective_start = max(interval_start, now_ts)
        if interval_end > effective_start:
            total += interval_end - effective_start
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
    selected_type_group: str,
    selected_color: str,
    selected_events: dict[str, pd.Series | None],
    selected_track: pd.DataFrame,
    overlay_tracks: list[dict[str, Any]],
    list_member_ids: set[str],
    now_local: pd.Timestamp | datetime | None = None,
    row_order_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    def _build_row(
        primary_id: str,
        label: str,
        type_group: str,
        color: str,
        events: dict[str, pd.Series | None],
        is_in_list: bool,
    ) -> dict[str, Any]:
        culmination = events.get("culmination")
        culmination_dir = str(culmination["wind16"]) if culmination is not None else "--"
        culmination_alt = f"{float(culmination['alt']):.1f} deg" if culmination is not None else "--"
        return {
            "primary_id": primary_id,
            "line_color": color,
            "target": label,
            "object_type_group": type_group or "other",
            "rise": event_time_value(events.get("rise")),
            "first_visible": event_time_value(events.get("first_visible")),
            "culmination": event_time_value(events.get("culmination")),
            "culmination_alt": culmination_alt,
            "last_visible": event_time_value(events.get("last_visible")),
            "set": event_time_value(events.get("set")),
            "visible_total": "--",
            "visible_remaining": "--",
            "culmination_dir": culmination_dir,
            "is_in_list": is_in_list,
        }

    selected_row = _build_row(
        selected_id,
        selected_label,
        selected_type_group,
        selected_color,
        selected_events,
        selected_id in list_member_ids,
    )
    selected_row["visible_total"] = format_duration_hm(compute_total_visible_time(selected_track))
    selected_row["visible_remaining"] = format_duration_hm(compute_remaining_visible_time(selected_track, now=now_local))
    rows = [selected_row]
    for target_track in overlay_tracks:
        primary_id = str(target_track.get("primary_id", ""))
        target_row = _build_row(
            primary_id,
            str(target_track.get("label", "List target")),
            str(target_track.get("object_type_group", "other")),
            str(target_track.get("color", "#22c55e")),
            target_track.get("events", {}),
            primary_id in list_member_ids,
        )
        overlay_track = target_track.get("track")
        if isinstance(overlay_track, pd.DataFrame):
            target_row["visible_total"] = format_duration_hm(compute_total_visible_time(overlay_track))
            target_row["visible_remaining"] = format_duration_hm(
                compute_remaining_visible_time(overlay_track, now=now_local)
            )
        rows.append(target_row)

    if row_order_ids:
        order_map = {str(primary_id): idx for idx, primary_id in enumerate(row_order_ids)}
        default_rank = len(order_map)
        rows = sorted(rows, key=lambda row: order_map.get(str(row.get("primary_id", "")), default_rank))
    return rows


def render_sky_position_summary_table(
    rows: list[dict[str, Any]],
    prefs: dict[str, Any],
    use_12_hour: bool,
    *,
    preview_list_id: str,
    preview_list_name: str,
    allow_list_membership_toggle: bool,
    show_remaining: bool = False,
    now_local: pd.Timestamp | datetime | None = None,
) -> None:
    if not rows:
        st.session_state["sky_summary_highlight_primary_id"] = ""
        return

    st.markdown("#### Targets")
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        st.session_state["sky_summary_highlight_primary_id"] = ""
        return

    summary_df["line_swatch"] = "■"
    if allow_list_membership_toggle:
        summary_df["list_action"] = summary_df["is_in_list"].map(lambda value: "Remove" if bool(value) else "Add")
    else:
        summary_df["list_action"] = "Auto"
    summary_df["visible_remaining_display"] = "--"
    if show_remaining:
        for row_index, row in summary_df.iterrows():
            first_visible = row.get("first_visible")
            total_duration = str(row.get("visible_total") or "").strip()
            remaining = str(row.get("visible_remaining") or "").strip()
            if (
                (not total_duration or total_duration == "--")
                or (not remaining or remaining == "--")
                or pd.isna(first_visible)
            ):
                continue
            try:
                first_visible_ts = pd.Timestamp(first_visible)
                if now_local is None:
                    now_ts = (
                        pd.Timestamp.now(tz=first_visible_ts.tzinfo)
                        if first_visible_ts.tzinfo is not None
                        else pd.Timestamp.now()
                    )
                else:
                    now_ts = pd.Timestamp(now_local)
                    if first_visible_ts.tzinfo is not None and now_ts.tzinfo is None:
                        now_ts = now_ts.tz_localize(first_visible_ts.tzinfo)
                    elif first_visible_ts.tzinfo is None and now_ts.tzinfo is not None:
                        now_ts = now_ts.tz_localize(None)
                    elif first_visible_ts.tzinfo is not None and now_ts.tzinfo is not None:
                        now_ts = now_ts.tz_convert(first_visible_ts.tzinfo)

                if now_ts > first_visible_ts:
                    summary_df.at[row_index, "visible_remaining_display"] = remaining
                else:
                    summary_df.at[row_index, "visible_remaining_display"] = total_duration
            except Exception:
                continue
    display_columns = [
        "line_swatch",
        "target",
        "object_type_group",
        "first_visible",
        "culmination",
        "last_visible",
        "visible_total",
    ]
    if show_remaining:
        display_columns.append("visible_remaining_display")
    display_columns.extend(["culmination_alt", "culmination_dir", "list_action"])

    display = summary_df[display_columns].rename(
        columns={
            "line_swatch": "Line",
            "target": "Target",
            "object_type_group": "Type",
            "first_visible": "First Visible",
            "culmination": "Peak",
            "last_visible": "Last Visible",
            "visible_total": "Duration",
            "visible_remaining_display": "Remaining",
            "culmination_alt": "Max Alt",
            "culmination_dir": "Direction",
            "list_action": "List",
        }
    )

    def _style_summary_row(row: pd.Series) -> list[str]:
        styles = ["" for _ in row]
        color = str(summary_df.loc[row.name, "line_color"]).strip()
        row_primary_id = str(summary_df.loc[row.name, "primary_id"]).strip()
        selected_detail_id = str(st.session_state.get("selected_id") or "").strip()
        if row_primary_id and selected_detail_id and row_primary_id == selected_detail_id:
            selected_bg = _muted_rgba_from_hex(color, alpha=0.16)
            for idx in range(len(styles)):
                styles[idx] = f"background-color: {selected_bg};"
        if color:
            line_idx = row.index.get_loc("Line")
            base_style = styles[line_idx]
            if base_style and not base_style.endswith(";"):
                base_style = f"{base_style};"
            styles[line_idx] = f"{base_style} color: {color}; font-weight: 700;"
        return styles

    styled = display.style.apply(_style_summary_row, axis=1)

    column_config: dict[str, Any] = {
        "Line": st.column_config.TextColumn(width="small"),
        "Target": st.column_config.TextColumn(width="large"),
        "Type": st.column_config.TextColumn(width="small"),
        "First Visible": st.column_config.DatetimeColumn(
            width="small",
            format=("h:mm a" if use_12_hour else "HH:mm"),
        ),
        "Peak": st.column_config.DatetimeColumn(
            width="small",
            format=("h:mm a" if use_12_hour else "HH:mm"),
        ),
        "Max Alt": st.column_config.TextColumn(width="small"),
        "Last Visible": st.column_config.DatetimeColumn(
            width="small",
            format=("h:mm a" if use_12_hour else "HH:mm"),
        ),
        "Duration": st.column_config.TextColumn(width="small"),
        "Direction": st.column_config.TextColumn(width="small"),
        "List": st.column_config.TextColumn(width="small"),
    }
    if show_remaining:
        column_config["Remaining"] = st.column_config.TextColumn(width="small")

    table_event = st.dataframe(
        styled,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-cell",
        key="sky_summary_table",
        column_config=column_config,
    )

    selected_rows: list[int] = []
    selected_columns: list[Any] = []
    selected_cells: list[Any] = []
    if table_event is not None:
        try:
            selected_rows = list(table_event.selection.rows)
            selected_columns = list(table_event.selection.columns)
            selected_cells = list(table_event.selection.cells)
        except Exception:
            if isinstance(table_event, dict):
                selection_payload = table_event.get("selection", {})
                selected_rows = list(selection_payload.get("rows", []))
                selected_columns = list(selection_payload.get("columns", []))
                selected_cells = list(selection_payload.get("cells", []))

    selected_index: int | None = None
    selected_column_index: int | None = None
    if selected_cells:
        first_cell = selected_cells[0]
        raw_row: Any = None
        raw_column: Any = None
        if isinstance(first_cell, (list, tuple)) and len(first_cell) >= 2:
            raw_row = first_cell[0]
            raw_column = first_cell[1]
        elif isinstance(first_cell, dict):
            raw_row = first_cell.get("row")
            raw_column = first_cell.get("column")

        if raw_row is not None:
            try:
                parsed_row_index = int(raw_row)
                if 0 <= parsed_row_index < len(summary_df):
                    selected_index = parsed_row_index
            except (TypeError, ValueError):
                selected_index = None

        if raw_column is not None:
            try:
                selected_column_index = int(raw_column)
            except (TypeError, ValueError):
                raw_column_name = str(raw_column)
                if raw_column_name in display.columns:
                    selected_column_index = int(display.columns.get_loc(raw_column_name))

    if selected_rows:
        try:
            parsed_row_index = int(selected_rows[0])
            if 0 <= parsed_row_index < len(summary_df):
                selected_index = parsed_row_index
        except (TypeError, ValueError):
            selected_index = None

    if selected_column_index is None and selected_columns:
        raw_column = selected_columns[0]
        try:
            selected_column_index = int(raw_column)
        except (TypeError, ValueError):
            raw_column_name = str(raw_column)
            if raw_column_name in display.columns:
                selected_column_index = int(display.columns.get_loc(raw_column_name))

    selected_primary_id = ""
    if selected_index is not None:
        selected_primary_id = str(summary_df.iloc[selected_index].get("primary_id", ""))

    selection_token = (
        f"{selected_index}:{selected_column_index}"
        if selected_index is not None and selected_column_index is not None
        else ""
    )
    last_selection_token = str(st.session_state.get("sky_summary_last_selection_token", ""))
    selection_changed = bool(selection_token) and selection_token != last_selection_token
    st.session_state["sky_summary_last_selection_token"] = selection_token

    current_highlight_id = str(st.session_state.get("sky_summary_highlight_primary_id", ""))
    if selection_changed and selected_primary_id and selected_primary_id != current_highlight_id:
        st.session_state["sky_summary_highlight_primary_id"] = selected_primary_id

    list_col_index = int(display.columns.get_loc("List"))

    if (
        selection_changed
        and selected_index is not None
        and selected_column_index == list_col_index
        and allow_list_membership_toggle
    ):
        action_token = f"{selected_index}:{selected_column_index}"
        last_action_token = str(st.session_state.get("sky_summary_list_action_token", ""))
        if action_token != last_action_token:
            selected_row = summary_df.iloc[selected_index]
            primary_id = str(selected_row.get("primary_id", ""))
            if primary_id:
                if toggle_target_in_list(prefs, preview_list_id, primary_id):
                    st.session_state["sky_summary_list_action_token"] = action_token
                    persist_and_rerun(prefs)
                st.session_state["sky_summary_list_action_token"] = action_token
    else:
        st.session_state["sky_summary_list_action_token"] = ""

    if allow_list_membership_toggle:
        st.caption(
            f"Search results choose the detail target. Use this table to highlight rows and update '{preview_list_name}'."
        )
    else:
        st.caption(
            f"Search results choose the detail target. '{preview_list_name}' is auto-managed from recent search selections."
        )


def format_hour_label(value: pd.Timestamp | datetime, use_12_hour: bool) -> str:
    timestamp = pd.Timestamp(value)
    if use_12_hour:
        return normalize_12_hour_label(timestamp.strftime("%I%p"))
    return timestamp.strftime("%H")


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    cleaned = str(value).strip().lstrip("#")
    if len(cleaned) != 6:
        raise ValueError("Expected 6-digit hex color")
    return int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    red, green, blue = rgb
    return f"#{red:02X}{green:02X}{blue:02X}"


def _muted_rgba_from_hex(value: str, alpha: float = 0.14) -> str:
    try:
        red, green, blue = _hex_to_rgb(value)
    except Exception:
        red, green, blue = (59, 130, 246)
    clamped_alpha = max(0.0, min(1.0, float(alpha)))
    return f"rgba({red}, {green}, {blue}, {clamped_alpha:.3f})"


def _interpolate_color_stops(value: float, color_stops: list[tuple[float, str]]) -> str:
    clamped = max(float(color_stops[0][0]), min(float(color_stops[-1][0]), float(value)))

    lower_stop = color_stops[0]
    upper_stop = color_stops[-1]
    for idx in range(len(color_stops) - 1):
        left = color_stops[idx]
        right = color_stops[idx + 1]
        if left[0] <= clamped <= right[0]:
            lower_stop = left
            upper_stop = right
            break

    lower_value, lower_hex = lower_stop
    upper_value, upper_hex = upper_stop
    if abs(upper_value - lower_value) < 1e-9:
        return lower_hex

    t = (clamped - lower_value) / (upper_value - lower_value)
    l_r, l_g, l_b = _hex_to_rgb(lower_hex)
    u_r, u_g, u_b = _hex_to_rgb(upper_hex)
    red = int(round(l_r + (u_r - l_r) * t))
    green = int(round(l_g + (u_g - l_g) * t))
    blue = int(round(l_b + (u_b - l_b) * t))
    return _rgb_to_hex((red, green, blue))


def _interpolate_cloud_cover_color(cloud_cover_percent: float) -> str:
    return _interpolate_color_stops(cloud_cover_percent, CLOUD_COVER_COLOR_STOPS)


def _interpolate_temperature_color_f(temp_f: float) -> str:
    return _interpolate_color_stops(temp_f, TEMPERATURE_COLOR_STOPS_F)


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


def format_weather_matrix_value(metric_key: str, raw_value: Any, temperature_unit: str) -> str:
    if raw_value is None or pd.isna(raw_value):
        return "-"

    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return "-"

    if metric_key == "temperature_2m":
        return format_temperature(numeric, temperature_unit)
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

    for candidate in WEATHER_ALERT_RAIN_PRIORITY:
        if candidate in seen:
            return [candidate]
    return []


def build_hourly_weather_matrix(
    rows: list[dict[str, Any]],
    *,
    use_12_hour: bool,
    temperature_unit: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    for metric_key, metric_label in WEATHER_MATRIX_ROWS:
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
        if metric_key == "precipitation_probability":
            tooltip_rows[metric_label] = [
                (
                    f"Precip probability: {float(by_hour[timestamp].get(metric_key)):.0f}%"
                    if by_hour[timestamp].get(metric_key) is not None and not pd.isna(by_hour[timestamp].get(metric_key))
                    else ""
                )
                for timestamp in ordered_times
            ]
        else:
            tooltip_rows[metric_label] = ["" for _ in ordered_times]

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
) -> None:
    if frame.empty:
        st.info("No hourly weather data available.")
        return

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

    header_cells = "".join(
        f'<th style="padding: 6px 8px; border: 1px solid #d1d5db; '
        f'background: #f3f4f6; color: #6b7280; text-align: left; white-space: nowrap;">'
        f"{html.escape(str(column))}</th>"
        for column in frame.columns
    )

    body_rows: list[str] = []
    for row_idx, row in frame.iterrows():
        element = str(row.get("Element", "")).strip()
        element_key = element.lower()
        row_cells = [
            (
                '<td style="padding: 6px 8px; border: 1px solid #d1d5db; '
                'font-weight: 600; white-space: nowrap; text-align: left;">'
                f"{html.escape(element) if element else '&nbsp;'}</td>"
            )
        ]

        for column in frame.columns:
            if str(column) == "Element":
                continue

            raw_value = row.get(column)
            text_value = str(raw_value).strip()
            display_value = html.escape(text_value) if text_value else "&nbsp;"

            cell_style = (
                "padding: 6px 8px; border: 1px solid #d1d5db; "
                "white-space: nowrap; text-align: center;"
            )
            if element_key == "cloud cover":
                cell_style += cloud_cover_cell_style(raw_value)
            elif element_key == "temperature":
                cell_style += temperature_cell_style(raw_value, temperature_unit=temperature_unit)

            tooltip = ""
            if aligned_tooltips is not None:
                tooltip = str(aligned_tooltips.at[row_idx, column]).strip()
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
    for event_key, event_label in EVENT_LABELS:
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
            if abs(az_set - az_prev) <= 180.0 and (abs(az_set - az_prev) >= 1e-9 or abs(alt_set - alt_prev) >= 1e-9):
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
) -> go.Figure:
    fig = go.Figure()
    plotted_any = False
    plotted_times: list[pd.Timestamp] = []
    label_candidates: list[dict[str, Any]] = []
    obstruction_ceiling = max(0.0, min(90.0, float(UNOBSTRUCTED_AREA_CONSTANT_OBSTRUCTION_ALT_DEG)))

    fig.add_shape(
        type="rect",
        xref="paper",
        yref="y",
        x0=0.0,
        x1=1.0,
        y0=0.0,
        y1=obstruction_ceiling,
        fillcolor=OBSTRUCTION_FILL_COLOR,
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
        line={"width": 1, "color": OBSTRUCTION_LINE_COLOR},
        layer="below",
    )

    non_selected_tracks = [target for target in target_tracks if not bool(target.get("is_selected", False))]
    selected_tracks = [target for target in target_tracks if bool(target.get("is_selected", False))]
    ordered_tracks = [*non_selected_tracks, *selected_tracks]

    for target_track in ordered_tracks:
        track = target_track.get("track")
        if not isinstance(track, pd.DataFrame) or track.empty:
            continue

        is_selected = bool(target_track.get("is_selected", False))
        target_label = str(target_track.get("label", "List target")).strip() or "List target"
        target_color = str(target_track.get("color", OBJECT_TYPE_GROUP_COLOR_DEFAULT)).strip() or OBJECT_TYPE_GROUP_COLOR_DEFAULT
        base_line_width = float(target_track.get("line_width", PATH_LINE_WIDTH_OVERLAY_DEFAULT))
        target_line_width = (
            max(base_line_width, PATH_LINE_WIDTH_PRIMARY_DEFAULT + 1.2)
            if is_selected
            else max(1.6, min(base_line_width, PATH_LINE_WIDTH_OVERLAY_DEFAULT))
        )
        target_line_color = target_color if is_selected else _muted_rgba_from_hex(target_color, alpha=0.68)
        target_emissions = str(target_track.get("emission_lines_display") or "").strip()
        target_fill_color = _muted_rgba_from_hex(target_color, alpha=(0.30 if is_selected else 0.10))
        latest_visible_time: pd.Timestamp | None = None
        latest_visible_altitude: float | None = None

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
            segment_latest_time = pd.Timestamp(segment_times.iloc[-1])
            segment_latest_altitude = float(altitude_values[-1])
            if latest_visible_time is None or segment_latest_time > latest_visible_time:
                latest_visible_time = segment_latest_time
                latest_visible_altitude = segment_latest_altitude

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
            plotted_any = True
        if latest_visible_time is not None and latest_visible_altitude is not None:
            label_candidates.append(
                {
                    "label": target_label,
                    "color": target_color,
                    "is_selected": is_selected,
                    "anchor_time": latest_visible_time,
                    "anchor_altitude": float(max(0.0, min(90.0, latest_visible_altitude))),
                }
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
            font={"size": 13, "color": "#334155"},
        )
    else:
        anchored_altitudes = [float(candidate["anchor_altitude"]) for candidate in label_candidates]
        placed_altitudes = distribute_non_overlapping_values(
            anchored_altitudes,
            lower=2.0,
            upper=88.0,
            min_gap=4.0,
        )
        for idx, candidate in enumerate(label_candidates):
            candidate_color = str(candidate["color"])
            is_selected_label = bool(candidate["is_selected"])
            label_text = html.escape(str(candidate["label"]))
            if is_selected_label:
                label_text = f"<b>{label_text}</b>"
            placed_altitude = float(placed_altitudes[idx])
            anchor_altitude = float(candidate["anchor_altitude"])

            fig.add_trace(
                go.Scatter(
                    x=[candidate["anchor_time"]],
                    y=[anchor_altitude],
                    mode="markers",
                    showlegend=False,
                    marker={
                        "size": 5 if is_selected_label else 4,
                        "color": candidate_color,
                        "line": {"width": 0},
                    },
                    hoverinfo="skip",
                )
            )
            if abs(placed_altitude - anchor_altitude) >= 0.2:
                fig.add_trace(
                    go.Scatter(
                        x=[candidate["anchor_time"], candidate["anchor_time"]],
                        y=[anchor_altitude, placed_altitude],
                        mode="lines",
                        showlegend=False,
                        line={
                            "width": 1,
                            "color": _muted_rgba_from_hex(candidate_color, alpha=0.70),
                            "dash": "dot",
                        },
                        hoverinfo="skip",
                    )
                )
            fig.add_annotation(
                x=1.004,
                y=placed_altitude,
                xref="paper",
                yref="y",
                text=label_text,
                showarrow=False,
                xanchor="left",
                align="left",
                font={
                    "size": 11 if is_selected_label else 10,
                    "color": candidate_color if is_selected_label else _muted_rgba_from_hex(candidate_color, alpha=0.86),
                },
                bgcolor="rgba(255, 255, 255, 0.45)",
                bordercolor="rgba(148, 163, 184, 0.35)",
                borderwidth=1,
                borderpad=2,
            )

    fig.update_layout(
        title=title,
        height=360,
        margin={"l": 10, "r": 170, "t": 70, "b": 10},
        showlegend=False,
        plot_bgcolor=PATH_PLOT_BACKGROUND_COLOR,
        xaxis_title="Time",
        yaxis_title="Altitude (deg)",
    )

    x_axis_settings: dict[str, Any] = {
        "type": "date",
        "showgrid": True,
        "gridcolor": PATH_PLOT_HORIZONTAL_GRID_COLOR,
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
        range=[0, 90],
        tickvals=[0, 15, 30, 45, 60, 75, 90],
        showgrid=True,
        gridcolor=PATH_PLOT_HORIZONTAL_GRID_COLOR,
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
    overlay_tracks: list[dict[str, Any]] | None = None,
) -> go.Figure:
    fig = go.Figure()

    for azimuth in (90.0, 180.0, 270.0):
        fig.add_shape(
            type="line",
            x0=azimuth,
            y0=0.0,
            x1=azimuth,
            y1=90.0,
            xref="x",
            yref="y",
            line={"color": CARDINAL_GRIDLINE_COLOR, "width": 1, "dash": "dot"},
            layer="below",
        )

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
    rise_segment, set_segment = endpoint_marker_segments_cartesian(track, events)
    if rise_segment is not None:
        fig.add_trace(
            go.Scatter(
                x=[rise_segment[0], rise_segment[2]],
                y=[rise_segment[1], rise_segment[3]],
                mode="markers",
                showlegend=False,
                marker={
                    "size": [0, PATH_ENDPOINT_MARKER_SIZE_PRIMARY],
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
                    "size": [0, PATH_ENDPOINT_MARKER_SIZE_PRIMARY],
                    "symbol": "arrow-right",
                    "angleref": "previous",
                    "color": selected_color,
                    "line": {"width": 0},
                },
                hoverinfo="skip",
            )
        )

    direction_segments = direction_marker_segments_cartesian(track, max_markers=PATH_DIRECTION_MARKERS_PRIMARY)
    for segment in direction_segments:
        x_start, y_start, x_end, y_end = segment
        fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[y_start, y_end],
                mode="markers",
                showlegend=False,
                marker={
                    "size": [0, PATH_DIRECTION_ARROW_SIZE_PRIMARY],
                    "symbol": "arrow-right",
                    "angleref": "previous",
                    "color": PATH_DIRECTION_ARROW_COLOR,
                    "line": {"width": 0},
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
        for target_track in overlay_tracks:
            overlay_track = target_track.get("track")
            if not isinstance(overlay_track, pd.DataFrame) or overlay_track.empty:
                continue
            path_x, path_y, path_times = split_path_on_az_wrap(overlay_track, use_12_hour=use_12_hour)
            if path_x.size == 0:
                continue

            target_label = str(target_track.get("label", "List target"))
            target_color = str(target_track.get("color", "#22c55e"))
            target_line_width = float(target_track.get("line_width", PATH_LINE_WIDTH_OVERLAY_DEFAULT))
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
            overlay_rise_segment, overlay_set_segment = endpoint_marker_segments_cartesian(
                overlay_track, target_track.get("events", {})
            )
            if overlay_rise_segment is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[overlay_rise_segment[0], overlay_rise_segment[2]],
                        y=[overlay_rise_segment[1], overlay_rise_segment[3]],
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": [0, PATH_ENDPOINT_MARKER_SIZE_OVERLAY],
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
                            "size": [0, PATH_ENDPOINT_MARKER_SIZE_OVERLAY],
                            "symbol": "arrow-right",
                            "angleref": "previous",
                            "color": target_color,
                            "line": {"width": 0},
                        },
                        hoverinfo="skip",
                    )
                )

            overlay_direction_segments = direction_marker_segments_cartesian(
                overlay_track, max_markers=PATH_DIRECTION_MARKERS_OVERLAY
            )
            for segment in overlay_direction_segments:
                x_start, y_start, x_end, y_end = segment
                fig.add_trace(
                    go.Scatter(
                        x=[x_start, x_end],
                        y=[y_start, y_end],
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": [0, PATH_DIRECTION_ARROW_SIZE_OVERLAY],
                            "symbol": "arrow-right",
                            "angleref": "previous",
                            "color": PATH_DIRECTION_ARROW_COLOR,
                            "line": {"width": 0},
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

    fig.update_layout(
        title="Target Paths",
        height=330,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        showlegend=False,
        plot_bgcolor=PATH_PLOT_BACKGROUND_COLOR,
        xaxis_title="Azimuth",
        yaxis_title="Altitude (deg)",
    )
    fig.update_xaxes(tickvals=[i * 22.5 for i in range(16)], ticktext=WIND16, range=[0, 360])
    fig.update_yaxes(
        range=[0, 90],
        showgrid=True,
        gridcolor=PATH_PLOT_HORIZONTAL_GRID_COLOR,
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
    overlay_tracks: list[dict[str, Any]] | None = None,
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

    for azimuth in (90.0, 180.0, 270.0):
        fig.add_trace(
            go.Scatterpolar(
                theta=[azimuth, azimuth],
                r=[0.0, 90.0],
                mode="lines",
                line={"color": CARDINAL_GRIDLINE_COLOR, "width": 1, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
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
    radial_values = np.asarray(track_r, dtype=float)
    rise_segment, set_segment = endpoint_marker_segments_radial(track, events, radial_values)
    if rise_segment is not None:
        fig.add_trace(
            go.Scatterpolar(
                theta=[rise_segment[0], rise_segment[2]],
                r=[rise_segment[1], rise_segment[3]],
                mode="markers",
                showlegend=False,
                marker={
                    "size": [0, PATH_ENDPOINT_MARKER_SIZE_PRIMARY],
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
                    "size": [0, PATH_ENDPOINT_MARKER_SIZE_PRIMARY],
                    "symbol": "arrow-right",
                    "angleref": "previous",
                    "color": selected_color,
                    "line": {"width": 0},
                },
                hoverinfo="skip",
            )
        )

    selected_direction_segments = direction_marker_segments_radial(
        track, np.asarray(track_r, dtype=float), max_markers=PATH_DIRECTION_MARKERS_PRIMARY
    )
    for segment in selected_direction_segments:
        theta_start, r_start, theta_end, r_end = segment
        fig.add_trace(
            go.Scatterpolar(
                theta=[theta_start, theta_end],
                r=[r_start, r_end],
                mode="markers",
                showlegend=False,
                marker={
                    "size": [0, PATH_DIRECTION_ARROW_SIZE_PRIMARY],
                    "symbol": "arrow-right",
                    "angleref": "previous",
                    "color": PATH_DIRECTION_ARROW_COLOR,
                    "line": {"width": 0},
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
        for target_track in overlay_tracks:
            overlay_track = target_track.get("track")
            if not isinstance(overlay_track, pd.DataFrame) or overlay_track.empty:
                continue

            target_label = str(target_track.get("label", "List target"))
            target_color = str(target_track.get("color", "#22c55e"))
            target_line_width = float(target_track.get("line_width", PATH_LINE_WIDTH_OVERLAY_DEFAULT))
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
            overlay_radial_values = np.asarray(overlay_r, dtype=float)
            overlay_rise_segment, overlay_set_segment = endpoint_marker_segments_radial(
                overlay_track, target_track.get("events", {}), overlay_radial_values
            )
            if overlay_rise_segment is not None:
                fig.add_trace(
                    go.Scatterpolar(
                        theta=[overlay_rise_segment[0], overlay_rise_segment[2]],
                        r=[overlay_rise_segment[1], overlay_rise_segment[3]],
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": [0, PATH_ENDPOINT_MARKER_SIZE_OVERLAY],
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
                            "size": [0, PATH_ENDPOINT_MARKER_SIZE_OVERLAY],
                            "symbol": "arrow-right",
                            "angleref": "previous",
                            "color": target_color,
                            "line": {"width": 0},
                        },
                        hoverinfo="skip",
                    )
                )

            overlay_direction_segments = direction_marker_segments_radial(
                overlay_track, np.asarray(overlay_r, dtype=float), max_markers=PATH_DIRECTION_MARKERS_OVERLAY
            )
            for segment in overlay_direction_segments:
                theta_start, r_start, theta_end, r_end = segment
                fig.add_trace(
                    go.Scatterpolar(
                        theta=[theta_start, theta_end],
                        r=[r_start, r_end],
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": [0, PATH_DIRECTION_ARROW_SIZE_OVERLAY],
                            "symbol": "arrow-right",
                            "angleref": "previous",
                            "color": PATH_DIRECTION_ARROW_COLOR,
                            "line": {"width": 0},
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

    fig.update_layout(
        title="Target Paths",
        height=660,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        showlegend=False,
        polar={
            "bgcolor": PATH_PLOT_BACKGROUND_COLOR,
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
                "gridcolor": PATH_PLOT_HORIZONTAL_GRID_COLOR,
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
    use_12_hour: bool = False,
) -> go.Figure:
    if track.empty:
        return go.Figure()

    rows: list[dict[str, Any]] = []
    grouped = track.set_index("time_local").resample("1h")
    for hour, chunk in grouped:
        if chunk.empty:
            continue

        max_row = chunk.loc[chunk["alt"].idxmax()]
        hour_iso = hour.isoformat()
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
            temp_label_colors.append("#6B7280")
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
                "color": OBSTRUCTION_FILL_COLOR,
                "line": {"color": OBSTRUCTION_LINE_COLOR, "width": 1},
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
            textfont={"color": "#111111"},
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
    cleaned_label = str(target_label or "").strip()
    if cleaned_label:
        title = f"Hourly Forecast - {cleaned_label}"

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
        yaxis_title="Altitude (deg)",
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=frame["hour_label"].tolist(),
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(148, 163, 184, 0.35)",
    )
    max_altitude = float(frame["max_alt"].max()) if "max_alt" in frame.columns and not frame["max_alt"].empty else 0.0
    if not np.isfinite(max_altitude):
        max_altitude = 0.0
    y_axis_max = min(90.0, max_altitude + 10.0)
    y_axis_max = max(10.0, y_axis_max)

    tickvals = [value for value in [0, 15, 30, 45, 60, 75, 90] if value <= y_axis_max + 1e-9]
    y_axis_max_rounded = round(y_axis_max, 1)
    if y_axis_max_rounded not in tickvals:
        tickvals.append(y_axis_max_rounded)
    tickvals = sorted(set(tickvals))

    fig.update_yaxes(range=[-12, y_axis_max], tickvals=tickvals)
    return fig


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def resolve_manual_location(query: str) -> dict[str, Any] | None:
    cleaned = query.strip()
    if not cleaned:
        return None

    def _valid_lat_lon(lat_value: Any, lon_value: Any) -> tuple[float, float] | None:
        try:
            lat = float(lat_value)
            lon = float(lon_value)
        except (TypeError, ValueError):
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        return lat, lon

    def _payload(lat: float, lon: float, label: str) -> dict[str, Any]:
        clean_label = str(label or "").strip() or f"{lat:.4f}, {lon:.4f}"
        return {
            "lat": lat,
            "lon": lon,
            "label": clean_label,
            "source": "manual",
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }

    coord_match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", cleaned)
    if coord_match:
        coords = _valid_lat_lon(coord_match.group(1), coord_match.group(2))
        if coords is not None:
            lat, lon = coords
            return _payload(lat, lon, reverse_geocode_label(lat, lon))

    zip_match = re.match(r"^\s*(\d{5})(?:-\d{4})?\s*$", cleaned)
    if zip_match:
        zip_code = zip_match.group(1)
        try:
            response = requests.get(f"https://api.zippopotam.us/us/{zip_code}", timeout=8)
            if response.ok:
                payload = response.json()
                places = payload.get("places") or []
                if places:
                    first = places[0]
                    coords = _valid_lat_lon(first.get("latitude"), first.get("longitude"))
                    if coords is not None:
                        lat, lon = coords
                        place_name = str(first.get("place name") or "").strip()
                        state = str(first.get("state abbreviation") or first.get("state") or "").strip()
                        label_parts = [part for part in [place_name, state] if part]
                        return _payload(lat, lon, ", ".join(label_parts) if label_parts else f"ZIP {zip_code}")
        except Exception:
            pass

    attempts = [cleaned]
    if "," in cleaned:
        attempts.append(re.sub(r"\s+", " ", cleaned.replace(",", " ")).strip())
    attempts = list(dict.fromkeys([candidate for candidate in attempts if candidate]))

    provider_factories: list[tuple[str, Any]] = [
        ("nominatim", lambda: Nominatim(user_agent="dso-explorer-prototype")),
        ("arcgis", lambda: ArcGIS(timeout=10)),
        ("photon", lambda: Photon(user_agent="dso-explorer-prototype")),
    ]
    for candidate in attempts:
        for _, provider_factory in provider_factories:
            try:
                geocoder = provider_factory()
                match = geocoder.geocode(candidate, exactly_one=True, timeout=10)
                if not match:
                    continue
                coords = _valid_lat_lon(getattr(match, "latitude", None), getattr(match, "longitude", None))
                if coords is None:
                    continue
                lat, lon = coords
                raw_label = str(getattr(match, "address", "") or "").split(",")[0].strip()
                return _payload(lat, lon, raw_label or candidate)
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
    def _valid_lat_lon(lat_value: Any, lon_value: Any) -> tuple[float, float] | None:
        try:
            lat = float(lat_value)
            lon = float(lon_value)
        except (TypeError, ValueError):
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        return lat, lon

    def _location_payload(lat: float, lon: float, city: Any, region: Any, country: Any) -> dict[str, Any]:
        city_str = str(city or "").strip()
        region_str = str(region or "").strip()
        country_str = str(country or "").strip()
        label_parts = [part for part in [city_str, region_str, country_str] if part]
        label = ", ".join(label_parts) if label_parts else "IP-based estimate"
        return {
            "lat": lat,
            "lon": lon,
            "label": label,
            "source": "ip",
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }

    try:
        response = requests.get("https://ipapi.co/json/", timeout=8)
        if response.ok:
            payload = response.json()
            if not bool(payload.get("error")):
                coords = _valid_lat_lon(payload.get("latitude"), payload.get("longitude"))
                if coords is not None:
                    lat, lon = coords
                    return _location_payload(
                        lat,
                        lon,
                        payload.get("city"),
                        payload.get("region"),
                        payload.get("country_name") or payload.get("country"),
                    )
    except Exception:
        pass

    try:
        response = requests.get("https://ipwho.is/", timeout=8)
        if response.ok:
            payload = response.json()
            if bool(payload.get("success", True)):
                coords = _valid_lat_lon(payload.get("latitude"), payload.get("longitude"))
                if coords is not None:
                    lat, lon = coords
                    return _location_payload(lat, lon, payload.get("city"), payload.get("region"), payload.get("country"))
    except Exception:
        pass

    try:
        response = requests.get("https://ipinfo.io/json", timeout=8)
        if response.ok:
            payload = response.json()
            loc = str(payload.get("loc") or "").strip()
            if "," in loc:
                lat_raw, lon_raw = loc.split(",", 1)
                coords = _valid_lat_lon(lat_raw, lon_raw)
                if coords is not None:
                    lat, lon = coords
                    return _location_payload(lat, lon, payload.get("city"), payload.get("region"), payload.get("country"))
    except Exception:
        pass

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


def sanitize_saved_lists(catalog: pd.DataFrame, prefs: dict[str, Any]) -> None:
    available_ids = set(catalog["primary_id"].tolist())
    current = ensure_preferences_shape(prefs)
    current_lists = current.get("lists", {})
    if not isinstance(current_lists, dict):
        return

    next_lists: dict[str, list[str]] = {}
    for raw_list_id, raw_ids in current_lists.items():
        list_id = str(raw_list_id).strip()
        if not list_id:
            continue
        cleaned_ids = [item for item in clean_primary_id_list(raw_ids) if item in available_ids]
        next_lists[list_id] = cleaned_ids

    next_payload = dict(current)
    next_payload["lists"] = next_lists
    next = ensure_preferences_shape(next_payload)
    if next != current:
        prefs.clear()
        prefs.update(next)
        st.session_state["prefs"] = prefs
        save_preferences(prefs)


def render_searchbox_results(
    catalog: pd.DataFrame,
    *,
    prefs: dict[str, Any] | None = None,
    lat: float,
    lon: float,
    listed_ids: list[str] | set[str] | None = None,
) -> str | None:
    if catalog.empty:
        st.info("No targets matched that search.")
        return None

    current_selected = str(st.session_state.get("selected_id") or "").strip()

    if st_searchbox is None:
        st.warning("`streamlit-searchbox` is unavailable; install dependencies to enable searchbox UI.")
        return None

    picked = st_searchbox(
        search_function=lambda search_term: searchbox_target_options(
            search_term,
            catalog=catalog,
            lat=lat,
            lon=lon,
            listed_ids=listed_ids,
            max_options=20,
            wind16_arrows=WIND16_ARROWS,
            search_catalog_fn=search_catalog,
            compute_altaz_now_fn=compute_altaz_now,
        ),
        label="Search",
        placeholder="Type to search targets (M31, NGC 7000, Orion Nebula...)",
        key="targets_searchbox_component",
        help="Type to filter suggestions, then use arrow keys + Enter to select.",
        clear_on_submit=True,
        edit_after_submit="option",
        rerun_on_update=True,
        debounce=150,
    )

    picked_value = str(picked).strip() if picked is not None else ""
    if picked_value:
        last_applied_pick = str(st.session_state.get("targets_searchbox_last_applied_pick", "")).strip()
        if picked_value != last_applied_pick:
            st.session_state["selected_id"] = picked_value
            st.session_state["targets_searchbox_last_applied_pick"] = picked_value
            if isinstance(prefs, dict) and push_target_to_auto_recent_list(prefs, picked_value):
                st.session_state["prefs"] = prefs
                save_preferences(prefs)
        st.caption(f"Selected: {picked_value}")
        return picked_value
    else:
        st.session_state["targets_searchbox_last_applied_pick"] = ""

    search_state = st.session_state.get("targets_searchbox_component")
    search_term = str(search_state.get("search", "")).strip() if isinstance(search_state, dict) else ""
    options_js = search_state.get("options_js", []) if isinstance(search_state, dict) else []
    options_count = len(options_js) if isinstance(options_js, list) else 0

    if search_term and options_count == 0:
        st.caption("No targets matched that search.")
        return None

    if current_selected:
        st.caption(f"No search selection. Detail remains on selected target: {current_selected}")
        return None

    if search_term:
        st.caption("Use arrow keys and Enter to choose a target.")
        return None

    st.caption("No target selected. Type to search and choose a target.")
    return None


def resolve_selected_row(catalog: pd.DataFrame, prefs: dict[str, Any]) -> pd.Series | None:
    selected_id = st.session_state.get("selected_id")
    selected = get_object_by_id(catalog, str(selected_id) if selected_id else "")
    if selected is None:
        return None

    location = prefs["location"]
    target = pd.DataFrame([selected])
    enriched = compute_altaz_now(target, lat=float(location["lat"]), lon=float(location["lon"]))
    if enriched.empty:
        return None
    return enriched.iloc[0]


def render_settings_page(catalog_meta: dict[str, Any], prefs: dict[str, Any], browser_locale: str | None) -> None:
    st.title("Settings")
    st.caption("Manage location, display preferences, catalog cache, obstructions, and settings backup.")

    persistence_notice = st.session_state.get("prefs_persistence_notice", "")
    if persistence_notice:
        st.warning(persistence_notice)

    cookie_backup_notice = get_cookie_backup_notice()
    if cookie_backup_notice:
        st.caption(cookie_backup_notice)

    location_notice = st.session_state.pop("location_notice", "")
    if location_notice:
        st.info(location_notice)

    st.subheader("Location")
    location = prefs["location"]
    st.markdown(f"**{location['label']}**")
    st.caption(f"Lat {location['lat']:.4f}, Lon {location['lon']:.4f} ({location['source']})")
    st.map(
        pd.DataFrame({"lat": [float(location["lat"])], "lon": [float(location["lon"])]}),
        zoom=8,
        height=300,
        use_container_width=True,
    )

    manual_location = st.text_input("Manual ZIP / place", key="manual_location")
    if st.button("Resolve location", use_container_width=True):
        if not manual_location.strip():
            st.warning("Enter a ZIP, place, or coordinates (lat, lon).")
        else:
            resolved = resolve_manual_location(manual_location)
            if resolved:
                prefs["location"] = resolved
                st.session_state["location_notice"] = f"Location resolved: {resolved['label']}."
                persist_and_rerun(prefs)
            else:
                st.warning("Couldn't find that location - keeping previous location.")

    location_cols = st.columns(2)
    if location_cols[0].button("Use browser location (permission)", use_container_width=True):
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

    if location_cols[1].button("Use my location (IP fallback)", use_container_width=True):
        resolved = approximate_location_from_ip()
        if resolved:
            prefs["location"] = resolved
            st.session_state["location_notice"] = f"Approximate location applied: {resolved['label']}."
            persist_and_rerun(prefs)
        else:
            st.warning("Location unavailable - keeping previous location.")

    st.divider()
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

    st.divider()
    render_lists_settings_section(prefs, persist_and_rerun_fn=persist_and_rerun)

    st.divider()
    st.subheader("Catalog")
    requested_mode = str(catalog_meta.get("feature_mode_requested", CATALOG_LOADER_MODE))
    active_mode = str(catalog_meta.get("feature_mode_active", catalog_meta.get("load_mode", "unknown")))
    st.caption(f"Loader mode: requested `{requested_mode}` | active `{active_mode}`")
    st.caption(f"Available loader modes: {', '.join(CATALOG_LOADER_MODES)}")
    st.caption(
        f"Rows: {int(catalog_meta.get('row_count', 0))} | "
        f"Mode: {catalog_meta.get('load_mode', 'unknown')}"
    )
    catalog_counts = catalog_meta.get("catalog_counts", {})
    if isinstance(catalog_counts, dict) and catalog_counts:
        counts_line = " | ".join(f"{catalog_name}: {count}" for catalog_name, count in sorted(catalog_counts.items()))
        st.caption(counts_line)
    catalog_filters = catalog_meta.get("filters", {})
    if isinstance(catalog_filters, dict):
        catalogs_count = len(catalog_filters.get("catalogs", []))
        object_types_count = len(catalog_filters.get("object_types", []))
        constellations_count = len(catalog_filters.get("constellations", []))
        st.caption(
            "Filter options: "
            f"catalogs={catalogs_count} | object types={object_types_count} | constellations={constellations_count}"
        )
    loaded_at = str(catalog_meta.get("loaded_at_utc", "")).strip()
    if loaded_at:
        st.caption(f"Loaded: {loaded_at}")
    validation = catalog_meta.get("validation", {})
    if isinstance(validation, dict):
        row_count = int(validation.get("row_count", 0))
        unique_ids = int(validation.get("unique_primary_id_count", 0))
        duplicate_ids = int(validation.get("duplicate_primary_id_count", 0))
        blank_ids = int(validation.get("blank_primary_id_count", 0))
        st.caption(
            "Validation: "
            f"rows={row_count} | unique_ids={unique_ids} | duplicate_ids={duplicate_ids} | blank_ids={blank_ids}"
        )
        warnings = validation.get("warnings", [])
        if isinstance(warnings, list):
            for warning in warnings:
                warning_text = str(warning).strip()
                if warning_text:
                    st.warning(warning_text)
    if st.button("Refresh catalog cache", use_container_width=True):
        st.session_state["force_catalog_refresh"] = True
        st.rerun()

    st.divider()
    st.subheader("Obstructions")
    if vertical_slider is None:
        st.warning(
            "`streamlit-vertical-slider` is required for the vertical obstruction sliders. "
            "Falling back to table editor."
        )
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
    else:
        mobile_obstruction_layout = int(st.session_state.get("browser_viewport_width", 1920)) < 900
        with st.container():
            st.markdown('<div id="obstruction-slider-scroll-anchor"></div>', unsafe_allow_html=True)
            if mobile_obstruction_layout:
                st.markdown(
                    """
                    <style>
                    @media (max-width: 900px) {
                      div[data-testid="stVerticalBlock"]:has(#obstruction-slider-scroll-anchor) {
                        overflow-x: auto;
                        overflow-y: visible;
                        -webkit-overflow-scrolling: touch;
                        padding-bottom: 0.4rem;
                      }
                      div[data-testid="stVerticalBlock"]:has(#obstruction-slider-scroll-anchor) > div[data-testid="stHorizontalBlock"] {
                        min-width: 950px;
                      }
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("Swipe horizontally to adjust all direction sliders.")
            st.caption("Minimum altitude by direction (deg)")
            header_cols = st.columns(len(WIND16), gap="small")
            for idx, direction in enumerate(WIND16):
                header_cols[idx].markdown(
                    f"<div style='text-align:center; font-size:0.8rem;'><strong>{direction}</strong></div>",
                    unsafe_allow_html=True,
                )

            slider_cols = st.columns(len(WIND16), gap="small")
            value_cols = st.columns(len(WIND16), gap="small")
            next_obstructions: dict[str, float] = {}
            for idx, direction in enumerate(WIND16):
                default_val = int(round(float(prefs["obstructions"].get(direction, 20.0))))
                state_key = f"obstruction_slider_{direction}"
                preview_value_raw = st.session_state.get(state_key, default_val)
                try:
                    preview_value = float(preview_value_raw)
                except (TypeError, ValueError):
                    preview_value = float(default_val)
                preview_clamped = float(max(0.0, min(90.0, preview_value)))
                slider_color = _interpolate_color_stops(preview_clamped, OBSTRUCTION_SLIDER_COLOR_STOPS)
                with slider_cols[idx]:
                    raw_value = vertical_slider(
                        key=state_key,
                        default_value=default_val,
                        min_value=0,
                        max_value=90,
                        step=1,
                        height=220,
                        track_color="#E2E8F0",
                        slider_color=slider_color,
                        thumb_color=slider_color,
                    )
                try:
                    slider_value = float(raw_value)
                except (TypeError, ValueError):
                    slider_value = float(default_val)
                clamped_value = float(max(0.0, min(90.0, slider_value)))
                next_obstructions[direction] = clamped_value
                with value_cols[idx]:
                    st.markdown(
                        f"<div style='text-align:center; font-size:0.8rem; color:#64748b;'>{int(round(clamped_value))} deg</div>",
                        unsafe_allow_html=True,
                    )

            if next_obstructions != prefs["obstructions"]:
                prefs["obstructions"] = next_obstructions
                save_preferences(prefs)

    st.divider()
    st.subheader("Settings Backup / Restore")
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
        help="Imports location, obstructions, lists, and display preferences.",
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


def render_detail_panel(
    selected: pd.Series | None,
    catalog: pd.DataFrame,
    prefs: dict[str, Any],
    temperature_unit: str,
    use_12_hour: bool,
    detail_stack_vertical: bool,
) -> None:
    def clean_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and not np.isfinite(value):
            return ""
        text = str(value).strip()
        if text.lower() in {"nan", "none"}:
            return ""
        return text

    def format_numeric(value: Any) -> str:
        if value is None:
            return ""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return clean_text(value)
        if not np.isfinite(numeric):
            return ""
        return f"{numeric:.6g}"

    def parse_numeric(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    if selected is None:
        with st.container(border=True):
            st.info("Select a target from results to view detail and plots.")
            st.plotly_chart(go.Figure(), use_container_width=True, key="detail_empty_path_plot")
            st.plotly_chart(go.Figure(), use_container_width=True, key="detail_empty_night_plot")
        return

    target_id = str(selected["primary_id"])
    active_preview_list_id = get_active_preview_list_id(prefs)
    active_preview_list_name = get_list_name(prefs, active_preview_list_id)
    active_preview_list_ids = get_list_ids(prefs, active_preview_list_id)
    active_preview_list_members = set(active_preview_list_ids)
    preview_list_is_system = is_system_list(prefs, active_preview_list_id)

    title = target_id
    if selected.get("common_name"):
        title = f"{target_id} - {selected['common_name']}"

    catalog_image_url = clean_text(selected.get("image_url"))
    image_source_url = clean_text(selected.get("image_attribution_url"))
    image_license = clean_text(selected.get("license_label"))
    info_url = clean_text(selected.get("info_url")) or image_source_url
    if not image_source_url:
        image_source_url = info_url

    image_url = catalog_image_url
    if not image_url:
        search_phrase = selected.get("common_name") or selected.get("primary_id")
        image_data = fetch_free_use_image(str(search_phrase))
        if image_data and image_data.get("image_url"):
            image_url = clean_text(image_data.get("image_url"))
            image_source_url = clean_text(image_data.get("source_url")) or image_source_url
            image_license = clean_text(image_data.get("license_label")) or image_license

    dist_value = format_numeric(selected.get("dist_value"))
    dist_unit = clean_text(selected.get("dist_unit"))
    redshift = format_numeric(selected.get("redshift"))
    ang_size_maj_arcmin_value = parse_numeric(selected.get("ang_size_maj_arcmin"))
    ang_size_min_arcmin_value = parse_numeric(selected.get("ang_size_min_arcmin"))
    ang_size_maj_arcmin = format_numeric(ang_size_maj_arcmin_value)
    ang_size_min_arcmin = format_numeric(ang_size_min_arcmin_value)

    if ang_size_maj_arcmin and ang_size_min_arcmin:
        ang_size_arcmin_display = f"{ang_size_maj_arcmin} x {ang_size_min_arcmin} arcmin"
    elif ang_size_maj_arcmin:
        ang_size_arcmin_display = f"{ang_size_maj_arcmin} arcmin"
    elif ang_size_min_arcmin:
        ang_size_arcmin_display = f"{ang_size_min_arcmin} arcmin"
    else:
        ang_size_arcmin_display = ""

    show_ang_size_in_degrees = (
        (ang_size_maj_arcmin_value is not None and ang_size_maj_arcmin_value >= 60.0)
        or (ang_size_min_arcmin_value is not None and ang_size_min_arcmin_value >= 60.0)
    )
    if show_ang_size_in_degrees:
        ang_size_maj_deg = (
            format_numeric(ang_size_maj_arcmin_value / 60.0)
            if ang_size_maj_arcmin_value is not None
            else ""
        )
        ang_size_min_deg = (
            format_numeric(ang_size_min_arcmin_value / 60.0)
            if ang_size_min_arcmin_value is not None
            else ""
        )
        if ang_size_maj_deg and ang_size_min_deg:
            ang_size_display = f"{ang_size_maj_deg} x {ang_size_min_deg} deg"
        elif ang_size_maj_deg:
            ang_size_display = f"{ang_size_maj_deg} deg"
        elif ang_size_min_deg:
            ang_size_display = f"{ang_size_min_deg} deg"
        else:
            ang_size_display = ""
        ang_size_tooltip = ang_size_arcmin_display
    else:
        ang_size_display = ang_size_arcmin_display
        ang_size_tooltip = ""
    morphology = clean_text(selected.get("morphology"))
    emission_details = clean_text(selected.get("emission_lines"))
    emission_details_display = re.sub(r"[\[\]]", "", emission_details)
    description = clean_text(selected.get("description"))

    with st.container(border=True):
        st.markdown(f"### {title}")
        st.caption(f"Catalog: {selected['catalog']} | Type: {selected.get('object_type') or '-'}")

        if detail_stack_vertical:
            image_container = st.container()
            description_container = st.container()
            property_container = st.container()
            forecast_container = st.container()
        else:
            detail_cols = st.columns([1, 1, 1, 2])
            image_container = detail_cols[0]
            description_container = detail_cols[1]
            property_container = detail_cols[2]
            forecast_container = detail_cols[3]

        with image_container:
            if image_url:
                image_url_html = html.escape(image_url, quote=True)
                image_tag = (
                    '<div style="width:200px; height:200px; max-width:100%; display:flex; align-items:center; justify-content:center;">'
                    f'<img src="{image_url_html}" '
                    'style="max-width:200px; max-height:200px; width:auto; height:auto; object-fit:contain; object-position:center;" />'
                    "</div>"
                )
                st.markdown(image_tag, unsafe_allow_html=True)
            else:
                st.info("No image URL available for this target.")

        with description_container:
            st.markdown("**Description**")
            st.write(description or "-")
            if image_source_url:
                st.caption(f"Image source: [Open link]({image_source_url})")
            if info_url:
                st.caption(f"Background: [Open object page]({info_url})")
            if image_license:
                st.caption(f"License/Credit: {image_license}")

        with property_container:
            editable_list_ids = editable_list_ids_in_order(prefs)
            if not editable_list_ids:
                st.caption("No editable lists available yet.")
            else:
                preferred_action_list_id = (
                    active_preview_list_id if active_preview_list_id in editable_list_ids else editable_list_ids[0]
                )
                action_select_key = "detail_add_to_list_select"
                current_action_selection = str(st.session_state.get(action_select_key, "")).strip()
                if current_action_selection not in editable_list_ids:
                    st.session_state[action_select_key] = preferred_action_list_id
                    current_action_selection = preferred_action_list_id

                selected_action_list_id = st.selectbox(
                    "Add to list...",
                    options=editable_list_ids,
                    index=editable_list_ids.index(current_action_selection),
                    key=action_select_key,
                    format_func=lambda list_id: get_list_name(prefs, list_id),
                )
                selected_action_list_name = get_list_name(prefs, selected_action_list_id)
                selected_action_list_members = set(get_list_ids(prefs, selected_action_list_id))
                is_in_selected_action_list = target_id in selected_action_list_members
                list_action_label = "Remove" if is_in_selected_action_list else "Add"
                if st.button(list_action_label, use_container_width=True, key="detail_add_to_list_apply"):
                    if toggle_target_in_list(prefs, selected_action_list_id, target_id):
                        st.session_state["selected_id"] = target_id
                        persist_and_rerun(prefs)
                st.caption(
                    f"{'In' if is_in_selected_action_list else 'Not in'} list: {selected_action_list_name}"
                )

            property_items = [
                {
                    "Property": "RA / Dec",
                    "Value": f"{float(selected['ra_deg']):.4f} deg / {float(selected['dec_deg']):.4f} deg",
                },
                {"Property": "Constellation", "Value": str(selected.get("constellation") or "-")},
                {
                    "Property": "Alt / Az (now)",
                    "Value": f"{float(selected['alt_now']):.1f} deg / {float(selected['az_now']):.1f} deg",
                },
                {"Property": "Direction", "Value": str(selected["wind16"])},
                {"Property": "Distance Value", "Value": dist_value or "-"},
                {"Property": "Distance Unit", "Value": dist_unit or "-"},
                {"Property": "Redshift", "Value": redshift or "-"},
                {"Property": "Angular Size", "Value": ang_size_display or "-", "Tooltip": ang_size_tooltip},
                {"Property": "Morphology", "Value": morphology or "-"},
                {"Property": "Emissions Details", "Value": emission_details_display or "-"},
            ]
            property_rows = pd.DataFrame(
                [
                    row
                    for row in property_items
                    if (clean_text(row.get("Value", "")) and clean_text(row.get("Value", "")) != "-")
                ]
            )
            if not property_rows.empty:
                table_rows_html: list[str] = []
                for _, row in property_rows.iterrows():
                    property_label = html.escape(str(row.get("Property", "")))
                    value_text = clean_text(row.get("Value")) or "-"
                    tooltip_text = clean_text(row.get("Tooltip"))
                    value_html = html.escape(value_text)
                    if tooltip_text and tooltip_text != value_text:
                        tooltip_html = html.escape(tooltip_text, quote=True)
                        value_html = (
                            f'<span title="{tooltip_html}" style="text-decoration: underline dotted; cursor: help;">'
                            f"{value_html}</span>"
                        )
                    table_rows_html.append(
                        "<tr>"
                        f'<td style="padding:0.35rem 0.5rem; vertical-align:top; border-bottom:1px solid rgba(120,120,120,0.18);">{property_label}</td>'
                        f'<td style="padding:0.35rem 0.5rem; vertical-align:top; border-bottom:1px solid rgba(120,120,120,0.18);">{value_html}</td>'
                        "</tr>"
                    )
                attributes_table_html = (
                    '<table style="width:100%; border-collapse:collapse; font-size:0.92rem;">'
                    "<thead><tr>"
                    '<th style="text-align:left; padding:0.35rem 0.5rem; border-bottom:1px solid rgba(120,120,120,0.28);">Property</th>'
                    '<th style="text-align:left; padding:0.35rem 0.5rem; border-bottom:1px solid rgba(120,120,120,0.28);">Value</th>'
                    "</tr></thead>"
                    f"<tbody>{''.join(table_rows_html)}</tbody>"
                    "</table>"
                )
                st.markdown(attributes_table_html, unsafe_allow_html=True)
        with forecast_container:
            forecast_placeholder = st.empty()
            forecast_legend_placeholder = st.empty()

    location = prefs["location"]
    window_start, window_end, tzinfo = tonight_window(location["lat"], location["lon"])
    selected_common_name = str(selected.get("common_name") or "").strip()
    selected_label = f"{target_id} - {selected_common_name}" if selected_common_name else target_id
    selected_group = str(selected.get("object_type_group") or "").strip() or "other"
    summary_highlight_id = str(st.session_state.get("sky_summary_highlight_primary_id", "")).strip()
    selected_line_width = (
        (PATH_LINE_WIDTH_PRIMARY_DEFAULT * PATH_LINE_WIDTH_SELECTION_MULTIPLIER)
        if summary_highlight_id == target_id
        else PATH_LINE_WIDTH_PRIMARY_DEFAULT
    )

    available_preview_list_ids = list_ids_in_order(prefs, include_auto_recent=True)
    if not available_preview_list_ids:
        available_preview_list_ids = [AUTO_RECENT_LIST_ID]
    if active_preview_list_id not in available_preview_list_ids:
        active_preview_list_id = AUTO_RECENT_LIST_ID
        active_preview_list_name = get_list_name(prefs, active_preview_list_id)
        active_preview_list_ids = get_list_ids(prefs, active_preview_list_id)
        active_preview_list_members = set(active_preview_list_ids)
        preview_list_is_system = is_system_list(prefs, active_preview_list_id)

    preview_tracks: list[dict[str, Any]] = []
    preview_targets = subset_by_id_list(catalog, active_preview_list_ids)
    for _, preview_target in preview_targets.iterrows():
        preview_target_id = str(preview_target["primary_id"])
        if preview_target_id == target_id:
            continue

        try:
            preview_ra = float(preview_target["ra_deg"])
            preview_dec = float(preview_target["dec_deg"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(preview_ra) or not np.isfinite(preview_dec):
            continue

        try:
            preview_track = compute_track(
                ra_deg=preview_ra,
                dec_deg=preview_dec,
                lat=float(location["lat"]),
                lon=float(location["lon"]),
                start_local=window_start,
                end_local=window_end,
                obstructions=prefs["obstructions"],
            )
        except Exception:
            continue
        if preview_track.empty:
            continue

        preview_common_name = str(preview_target.get("common_name") or "").strip()
        preview_label = f"{preview_target_id} - {preview_common_name}" if preview_common_name else preview_target_id
        preview_emission_details = re.sub(r"[\[\]]", "", clean_text(preview_target.get("emission_lines")))
        preview_group = str(preview_target.get("object_type_group") or "").strip() or "other"
        preview_tracks.append(
            {
                "primary_id": preview_target_id,
                "common_name": preview_common_name,
                "label": preview_label,
                "object_type_group": preview_group,
                "emission_lines_display": preview_emission_details,
                "line_width": (
                    (PATH_LINE_WIDTH_OVERLAY_DEFAULT * PATH_LINE_WIDTH_SELECTION_MULTIPLIER)
                    if summary_highlight_id == preview_target_id
                    else PATH_LINE_WIDTH_OVERLAY_DEFAULT
                ),
                "track": preview_track,
                "events": extract_events(preview_track),
            }
        )

    # Evenly distribute same-group targets across each group's start->end gradient.
    group_total_counts: dict[str, int] = {selected_group: 1}
    for target_track in preview_tracks:
        group_key = str(target_track.get("object_type_group") or "").strip() or "other"
        group_total_counts[group_key] = group_total_counts.get(group_key, 0) + 1

    group_seen_counts: dict[str, int] = {}

    def _next_group_plot_color(group_label: str | None) -> str:
        group_key = str(group_label or "").strip() or "other"
        index_in_group = group_seen_counts.get(group_key, 0)
        group_seen_counts[group_key] = index_in_group + 1
        total_in_group = max(1, int(group_total_counts.get(group_key, 1)))
        step_fraction = 0.0 if total_in_group <= 1 else (float(index_in_group) / float(total_in_group - 1))
        return object_type_group_color(group_key, step_fraction=step_fraction)

    selected_color = _next_group_plot_color(selected_group)
    for target_track in preview_tracks:
        group_key = str(target_track.get("object_type_group") or "").strip() or "other"
        target_track["color"] = _next_group_plot_color(group_key)

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
    hourly_weather_rows = fetch_hourly_weather(
        lat=float(location["lat"]),
        lon=float(location["lon"]),
        tz_name=tzinfo.key,
        start_local_iso=window_start.isoformat(),
        end_local_iso=window_end.isoformat(),
        hourly_fields=EXTENDED_FORECAST_HOURLY_FIELDS,
    )
    nightly_weather_alert_emojis = collect_night_weather_alert_emojis(hourly_weather_rows, temperature_unit)

    with st.container(border=True):
        st.markdown("### Night Sky Preview")
        current_preview_idx = available_preview_list_ids.index(active_preview_list_id)
        selected_preview_list_id = st.selectbox(
            "Preview List",
            options=available_preview_list_ids,
            index=current_preview_idx,
            format_func=lambda list_id: get_list_name(prefs, list_id),
            key="night_sky_preview_list_select",
        )
        if selected_preview_list_id != active_preview_list_id:
            if set_active_preview_list_id(prefs, selected_preview_list_id):
                persist_and_rerun(prefs)

        active_preview_list_id = get_active_preview_list_id(prefs)
        active_preview_list_name = get_list_name(prefs, active_preview_list_id)
        active_preview_list_ids = get_list_ids(prefs, active_preview_list_id)
        active_preview_list_members = set(active_preview_list_ids)
        preview_list_is_system = is_system_list(prefs, active_preview_list_id)

        st.caption(
            f"Tonight ({tzinfo.key}): "
            f"{format_display_time(window_start, use_12_hour=use_12_hour)} -> "
            f"{format_display_time(window_end, use_12_hour=use_12_hour)} | "
            f"Rise {format_time(events['rise'], use_12_hour=use_12_hour)} | "
            f"First-visible {format_time(events['first_visible'], use_12_hour=use_12_hour)} | "
            f"Culmination {format_time(events['culmination'], use_12_hour=use_12_hour)} | "
            f"Last-visible {format_time(events['last_visible'], use_12_hour=use_12_hour)}"
        )
        st.caption(f"Overlaying list: {active_preview_list_name} ({len(preview_tracks)} companion targets)")

        path_style = st.segmented_control(
            "Target Paths Style",
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
                selected_emissions=emission_details_display,
                selected_color=selected_color,
                selected_line_width=selected_line_width,
                use_12_hour=use_12_hour,
                overlay_tracks=preview_tracks,
            )
        else:
            path_figure = build_path_plot(
                track=track,
                events=events,
                obstructions=prefs["obstructions"],
                selected_label=selected_label,
                selected_emissions=emission_details_display,
                selected_color=selected_color,
                selected_line_width=selected_line_width,
                use_12_hour=use_12_hour,
                overlay_tracks=preview_tracks,
            )

        if let_it_rain is not None and nightly_weather_alert_emojis:
            now_local = datetime.now(tzinfo)
            current_bucket = int(now_local.timestamp() // WEATHER_ALERT_RAIN_INTERVAL_SECONDS)
            last_bucket = st.session_state.get(WEATHER_ALERT_RAIN_BUCKET_STATE_KEY)
            if last_bucket != current_bucket:
                st.session_state[WEATHER_ALERT_RAIN_BUCKET_STATE_KEY] = current_bucket
                for alert_emoji in nightly_weather_alert_emojis:
                    let_it_rain(
                        emoji=alert_emoji,
                        font_size=34,
                        falling_speed=5,
                        animation_length=WEATHER_ALERT_RAIN_DURATION_SECONDS,
                    )
        else:
            st.session_state.pop(WEATHER_ALERT_RAIN_BUCKET_STATE_KEY, None)

        st.plotly_chart(
            path_figure,
            use_container_width=True,
            key="detail_path_plot",
        )
        summary_rows = build_sky_position_summary_rows(
            selected_id=target_id,
            selected_label=selected_label,
            selected_type_group=selected_group,
            selected_color=selected_color,
            selected_events=events,
            selected_track=track,
            overlay_tracks=preview_tracks,
            list_member_ids=active_preview_list_members,
            now_local=pd.Timestamp(datetime.now(tzinfo)),
            row_order_ids=(
                [target_id] + [str(item) for item in active_preview_list_ids if str(item) != target_id]
                if target_id not in active_preview_list_members
                else [str(item) for item in active_preview_list_ids]
            ),
        )
        local_now = datetime.now(tzinfo)
        show_remaining_column = window_start <= local_now <= window_end
        summary_col, tips_col = st.columns([3, 1], gap="medium")
        with summary_col:
            render_sky_position_summary_table(
                summary_rows,
                prefs,
                use_12_hour=use_12_hour,
                preview_list_id=active_preview_list_id,
                preview_list_name=active_preview_list_name,
                allow_list_membership_toggle=(not preview_list_is_system),
                show_remaining=show_remaining_column,
                now_local=pd.Timestamp(local_now),
            )
            unobstructed_area_tracks = [
                {
                    "is_selected": True,
                    "label": selected_label,
                    "color": selected_color,
                    "line_width": selected_line_width,
                    "emission_lines_display": emission_details_display,
                    "track": track,
                },
                *[{**preview_track, "is_selected": False} for preview_track in preview_tracks],
            ]
            st.plotly_chart(
                build_unobstructed_altitude_area_plot(
                    unobstructed_area_tracks,
                    use_12_hour=use_12_hour,
                ),
                use_container_width=True,
                key="detail_unobstructed_area_plot",
            )
        with tips_col:
            summary_ids = {
                str(row.get("primary_id", "")).strip()
                for row in summary_rows
                if str(row.get("primary_id", "")).strip()
            }
            tips_focus_id = target_id
            summary_highlight_id = str(st.session_state.get("sky_summary_highlight_primary_id", "")).strip()
            if summary_highlight_id and summary_highlight_id in summary_ids:
                tips_focus_id = summary_highlight_id

            tips_track_by_id: dict[str, pd.DataFrame | None] = {target_id: track}
            for preview_track_payload in preview_tracks:
                preview_track_id = str(preview_track_payload.get("primary_id", "")).strip()
                if not preview_track_id:
                    continue
                preview_track_df = preview_track_payload.get("track")
                tips_track_by_id[preview_track_id] = (
                    preview_track_df if isinstance(preview_track_df, pd.DataFrame) else None
                )

            tips_data_by_id: dict[str, pd.Series | dict[str, Any] | None] = {target_id: selected}
            for _, preview_target in preview_targets.iterrows():
                preview_target_id = str(preview_target.get("primary_id", "")).strip()
                if preview_target_id:
                    tips_data_by_id[preview_target_id] = preview_target

            tips_label_by_id: dict[str, str] = {}
            for row in summary_rows:
                row_id = str(row.get("primary_id", "")).strip()
                if not row_id:
                    continue
                row_label = str(row.get("target", "")).strip()
                if row_label:
                    tips_label_by_id[row_id] = row_label
            tips_label_by_id.setdefault(target_id, selected_label)

            tips_focus_label = tips_label_by_id.get(tips_focus_id, target_id)
            tips_focus_data = tips_data_by_id.get(tips_focus_id)
            tips_focus_track = tips_track_by_id.get(tips_focus_id)

            with st.container(border=True):
                render_target_tips_panel(
                    tips_focus_id,
                    tips_focus_label,
                    tips_focus_data,
                    tips_focus_track,
                    summary_rows,
                    nightly_weather_alert_emojis,
                    hourly_weather_rows,
                    temperature_unit=temperature_unit,
                    use_12_hour=use_12_hour,
                    local_now=local_now,
                    window_start=window_start,
                    window_end=window_end,
                )
    temperatures: dict[str, float] = {}
    cloud_cover_by_hour: dict[str, float] = {}
    weather_by_hour: dict[str, dict[str, Any]] = {}
    for weather_row in hourly_weather_rows:
        time_iso = str(weather_row.get("time_iso", "")).strip()
        if not time_iso:
            continue
        try:
            hour_key = pd.Timestamp(time_iso).floor("h").isoformat()
        except Exception:
            hour_key = time_iso
        temperature_value = weather_row.get("temperature_2m")
        if temperature_value is not None and not pd.isna(temperature_value):
            temperatures[hour_key] = float(temperature_value)
        cloud_cover_value = weather_row.get("cloud_cover")
        if cloud_cover_value is not None and not pd.isna(cloud_cover_value):
            cloud_cover_by_hour[hour_key] = float(cloud_cover_value)
        weather_by_hour[hour_key] = weather_row

    forecast_placeholder.plotly_chart(
        build_night_plot(
            track=track,
            temperature_by_hour=temperatures,
            cloud_cover_by_hour=cloud_cover_by_hour,
            weather_by_hour=weather_by_hour,
            temperature_unit=temperature_unit,
            target_label=selected_label,
            use_12_hour=use_12_hour,
        ),
        use_container_width=True,
        key="detail_night_plot",
    )
    forecast_legend_placeholder.caption(WEATHER_ALERT_INDICATOR_LEGEND_CAPTION)


def render_explorer_page(
    catalog: pd.DataFrame,
    catalog_meta: dict[str, Any],
    prefs: dict[str, Any],
    temperature_unit: str,
    use_12_hour: bool,
    detail_stack_vertical: bool,
) -> None:
    persistence_notice = st.session_state.get("prefs_persistence_notice", "")
    if persistence_notice:
        st.warning(persistence_notice)

    cookie_backup_notice = get_cookie_backup_notice()
    if cookie_backup_notice:
        st.caption(cookie_backup_notice)

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
        (
            "<p class='small-note'>Alt/Az auto-refresh 60s<br>"
            f"Updated: {now_utc.strftime('%Y-%m-%d')} "
            f"{format_display_time(now_utc, use_12_hour=use_12_hour, include_seconds=True)} UTC"
            "</p>"
        ),
        unsafe_allow_html=True,
    )
    st.caption(f"Catalog rows loaded: {int(catalog_meta.get('row_count', len(catalog)))}")
    location = prefs["location"]
    location_lat = float(location["lat"])
    location_lon = float(location["lon"])
    listed_ids = all_listed_ids_in_order(prefs)
    top_cols = st.columns([3, 2], gap="medium")
    with top_cols[0]:
        st.caption("Type to filter suggestions, then use arrow keys + Enter to select.")
        render_searchbox_results(
            catalog,
            prefs=prefs,
            lat=location_lat,
            lon=location_lon,
            listed_ids=listed_ids,
        )
    with top_cols[1]:
        st.caption("Tonight weather by hour")
        window_start, window_end, tzinfo = tonight_window(location_lat, location_lon)
        hourly_weather_rows = fetch_hourly_weather(
            lat=location_lat,
            lon=location_lon,
            tz_name=tzinfo.key,
            start_local_iso=window_start.isoformat(),
            end_local_iso=window_end.isoformat(),
            hourly_fields=EXTENDED_FORECAST_HOURLY_FIELDS,
        )
        weather_matrix, weather_tooltips, weather_indicators = build_hourly_weather_matrix(
            hourly_weather_rows,
            use_12_hour=use_12_hour,
            temperature_unit=temperature_unit,
        )
        weather_display = weather_matrix.reset_index().rename(columns={"index": "Element"})
        weather_tooltip_display = weather_tooltips.reset_index().rename(columns={"index": "Element"})
        weather_indicator_display = weather_indicators.reset_index().rename(columns={"index": "Element"})
        render_hourly_weather_matrix(
            weather_display,
            temperature_unit=temperature_unit,
            tooltip_frame=weather_tooltip_display,
            indicator_frame=weather_indicator_display,
        )
        st.caption(WEATHER_ALERT_INDICATOR_LEGEND_CAPTION)

    selected_row = resolve_selected_row(catalog, prefs)
    active_preview_list_ids = get_list_ids(prefs, get_active_preview_list_id(prefs))
    if selected_row is not None or bool(active_preview_list_ids):
        render_detail_panel(
            selected=selected_row,
            catalog=catalog,
            prefs=prefs,
            temperature_unit=temperature_unit,
            use_12_hour=use_12_hour,
            detail_stack_vertical=detail_stack_vertical,
        )


def main() -> None:
    st_autorefresh(interval=60_000, key="altaz_refresh")
    set_cookie_backup_runtime_enabled(ENABLE_COOKIE_BACKUP)
    bootstrap_cookie_backup()

    if "prefs_bootstrap_runs" not in st.session_state:
        st.session_state["prefs_bootstrap_runs"] = 0
    if "prefs_bootstrapped" not in st.session_state:
        st.session_state["prefs_bootstrapped"] = False
    if "prefs" not in st.session_state:
        st.session_state["prefs"] = default_preferences()

    if not bool(st.session_state.get("prefs_bootstrapped", False)):
        runs = int(st.session_state.get("prefs_bootstrap_runs", 0))
        loaded_prefs, retry_needed = load_preferences()
        st.session_state["prefs"] = loaded_prefs
        runs += 1
        st.session_state["prefs_bootstrap_runs"] = runs

        if retry_needed and runs < PREFS_BOOTSTRAP_MAX_RUNS:
            st_autorefresh(
                interval=PREFS_BOOTSTRAP_RETRY_INTERVAL_MS,
                limit=1,
                key=f"prefs_bootstrap_wait_{runs}",
            )
            st.stop()

        st.session_state["prefs_bootstrapped"] = True

    prefs = ensure_preferences_shape(st.session_state["prefs"])
    st.session_state["prefs"] = prefs

    browser_language_raw = get_browser_language(component_key="browser_language_pref")
    if isinstance(browser_language_raw, str) and browser_language_raw.strip():
        st.session_state["browser_language"] = browser_language_raw.strip()
    browser_language = st.session_state.get("browser_language")
    browser_hour_cycle_raw = streamlit_js_eval(
        js_expressions=(
            "new Intl.DateTimeFormat(window.navigator.language, "
            "{hour: 'numeric', minute: '2-digit'}).resolvedOptions().hourCycle"
        ),
        key="browser_hour_cycle_pref",
    )
    if isinstance(browser_hour_cycle_raw, str) and browser_hour_cycle_raw.strip():
        st.session_state["browser_hour_cycle"] = browser_hour_cycle_raw.strip()
    browser_hour_cycle = st.session_state.get("browser_hour_cycle")
    use_12_hour = resolve_12_hour_clock(browser_language, browser_hour_cycle)
    viewport_width_raw = streamlit_js_eval(
        js_expressions="window.innerWidth",
        key="browser_viewport_width_probe",
    )
    if isinstance(viewport_width_raw, (int, float)) and float(viewport_width_raw) > 0:
        st.session_state["browser_viewport_width"] = int(float(viewport_width_raw))
    elif isinstance(viewport_width_raw, str):
        try:
            parsed_width = float(viewport_width_raw.strip())
            if parsed_width > 0:
                st.session_state["browser_viewport_width"] = int(parsed_width)
        except (TypeError, ValueError):
            pass
    browser_viewport_width = int(st.session_state.get("browser_viewport_width", 1920))
    detail_stack_vertical = browser_viewport_width < DETAIL_PANE_STACK_BREAKPOINT_PX

    effective_temperature_unit = resolve_temperature_unit(
        str(prefs.get("temperature_unit", "auto")),
        browser_language,
    )

    force_catalog_refresh = bool(st.session_state.pop("force_catalog_refresh", False))
    catalog, catalog_meta = load_catalog_data(
        seed_path=CATALOG_SEED_PATH,
        cache_path=CATALOG_CACHE_PATH,
        metadata_path=CATALOG_META_PATH,
        enriched_path=CATALOG_ENRICHED_PATH,
        force_refresh=force_catalog_refresh,
        mode=CATALOG_LOADER_MODE,
        curated_path=CURATED_CATALOG_PATH,
    )
    sanitize_saved_lists(catalog=catalog, prefs=prefs)

    def explorer_page() -> None:
        render_explorer_page(
            catalog=catalog,
            catalog_meta=catalog_meta,
            prefs=prefs,
            temperature_unit=effective_temperature_unit,
            use_12_hour=use_12_hour,
            detail_stack_vertical=detail_stack_vertical,
        )

    def settings_page() -> None:
        render_settings_page(
            catalog_meta=catalog_meta,
            prefs=prefs,
            browser_locale=browser_language,
        )

    if hasattr(st, "navigation") and hasattr(st, "Page"):
        navigation = st.navigation(
            [
                st.Page(explorer_page, title="Explorer", icon="✨", default=True),
                st.Page(settings_page, title="Settings", icon="⚙️"),
            ]
        )
        navigation.run()
        return

    selected_page = st.sidebar.radio("Page", ["Explorer", "Settings"], key="app_page_selector")
    if selected_page == "Settings":
        settings_page()
        return
    explorer_page()


if __name__ == "__main__":
    main()
