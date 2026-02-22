from __future__ import annotations

import copy
import hashlib
import html
import json
import re
import uuid
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from time import perf_counter
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
from app_constants import (
    DEFAULT_LOCATION,
    DEFAULT_SITE_ID,
    DEFAULT_SITE_NAME,
    UI_THEME_AURA_DRACULA,
    UI_THEME_BLUE_LIGHT,
    UI_THEME_DARK,
    UI_THEME_LIGHT,
    UI_THEME_MONOKAI_ST3,
    WIND16,
)
from app_preferences import (
    PREFS_BOOTSTRAP_MAX_RUNS,
    PREFS_BOOTSTRAP_RETRY_INTERVAL_MS,
    build_settings_export_payload,
    default_preferences,
    ensure_preferences_shape,
    load_preferences,
    parse_settings_import_payload,
    persist_and_rerun,
    save_preferences,
)
from app_theme import (
    apply_dataframe_styler_theme,
    apply_ui_theme_css,
    is_dark_ui_theme,
    resolve_theme_palette,
    resolve_plot_theme_colors,
)
from condition_tips.ui import render_condition_tips_panel
from dso_enricher.catalog_service import (
    get_object_by_id,
    load_catalog_from_cache,
    search_catalog,
)
from lists.list_subsystem import (
    AUTO_RECENT_LIST_ID,
    all_listed_ids_in_order,
    clean_primary_id_list,
    editable_list_ids_in_order,
    get_active_preview_list_id,
    get_list_ids,
    get_list_name,
    is_system_list,
    list_ids_in_order,
    push_target_to_auto_recent_list,
    set_active_preview_list_id,
    toggle_target_in_list,
)
from lists.list_search import subset_by_id_list
from lists.list_settings_ui import render_lists_settings_section
from geopy.geocoders import ArcGIS, Nominatim, Photon
try:
    import folium
    from branca.element import MacroElement, Template
    from streamlit_folium import st_folium
except Exception:
    folium = None
    MacroElement = None
    Template = None
    st_folium = None
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
try:
    from st_mui_table import st_mui_table
except Exception:
    st_mui_table = None
try:
    from streamlit_modal import Modal
except Exception:
    try:
        from streamlit_modal_compat import Modal
    except Exception:
        Modal = None
from streamlit_js_eval import get_geolocation, streamlit_js_eval
from streamlit_autorefresh import st_autorefresh
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

CATALOG_CACHE_PATH = Path("data/dso_catalog_cache.parquet")
EQUIPMENT_CATALOG_PATH = Path("data/equipment/equipment_catalog.json")

TEMPERATURE_UNIT_OPTIONS = {
    "Auto (browser)": "auto",
    "Fahrenheit": "f",
    "Celsius": "c",
}


def _catalog_cache_fingerprint(cache_path: Path) -> tuple[int, int]:
    try:
        stat_result = cache_path.stat()
    except OSError:
        return (0, 0)
    return (int(stat_result.st_mtime_ns), int(stat_result.st_size))


@st.cache_resource(show_spinner=False)
def _load_catalog_resource_cached(
    cache_path_str: str,
    cache_mtime_ns: int,
    cache_size_bytes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trace_cache_event(
        f"Hydrating app catalog cache for {cache_path_str} "
        f"(mtime_ns={cache_mtime_ns}, bytes={cache_size_bytes})"
    )
    del cache_mtime_ns, cache_size_bytes
    cached_catalog, cached_meta = load_catalog_from_cache(cache_path=Path(cache_path_str))
    meta_payload = dict(cached_meta or {})
    meta_payload["app_cache"] = "streamlit_resource"
    return cached_catalog, meta_payload


def load_catalog_app_cached(cache_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_path = cache_path.expanduser().resolve()
    cache_mtime_ns, cache_size_bytes = _catalog_cache_fingerprint(resolved_path)
    return _load_catalog_resource_cached(
        str(resolved_path),
        cache_mtime_ns,
        cache_size_bytes,
    )


RECOMMENDATION_CACHE_SAMPLE_MINUTES = 10
RECOMMENDATION_CLOUD_COVER_THRESHOLD = 30.0
RECOMMENDATION_QUERY_SESSION_CACHE_LIMIT = 8


def trace_cache_event(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[DSO CACHE {timestamp}] {message}", flush=True)


def normalize_emission_band_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "", str(value or "").strip().upper())
    if not token:
        return ""
    synonyms = {
        "HALPHA": "HA",
        "HABETA": "HB",
        "HBETA": "HB",
        "O3": "OIII",
        "N2": "NII",
        "S2": "SII",
    }
    return synonyms.get(token, token)


def parse_emission_band_set(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        raw_parts = [str(item) for item in value]
    else:
        cleaned = str(value or "").replace("[", " ").replace("]", " ")
        raw_parts = re.split(r"[;,/|]+", cleaned)
    return {
        normalize_emission_band_token(part)
        for part in raw_parts
        if normalize_emission_band_token(part)
    }


@st.cache_resource(show_spinner=False)
def _load_catalog_recommendation_features_cached(
    cache_path_str: str,
    cache_mtime_ns: int,
    cache_size_bytes: int,
) -> pd.DataFrame:
    trace_cache_event(
        f"Hydrating catalog recommendation feature cache for {cache_path_str} "
        f"(mtime_ns={cache_mtime_ns}, bytes={cache_size_bytes})"
    )
    catalog_frame, _ = load_catalog_app_cached(Path(cache_path_str))
    features = catalog_frame.copy()

    features["primary_id"] = features["primary_id"].fillna("").astype(str).str.strip()
    features["common_name"] = features["common_name"].fillna("").astype(str).str.strip()
    features["object_type_group_norm"] = features["object_type_group"].map(normalize_object_type_group)
    features["ra_deg_num"] = pd.to_numeric(features["ra_deg"], errors="coerce")
    features["dec_deg_num"] = pd.to_numeric(features["dec_deg"], errors="coerce")
    features["has_valid_coords"] = np.isfinite(features["ra_deg_num"]) & np.isfinite(features["dec_deg_num"])
    features["emission_band_tokens"] = features["emission_lines"].apply(
        lambda value: tuple(sorted(parse_emission_band_set(value)))
    )
    features["apparent_size"] = features.apply(
        lambda row: format_apparent_size_display(
            row.get("ang_size_maj_arcmin"),
            row.get("ang_size_min_arcmin"),
        ),
        axis=1,
    )
    features["apparent_size_sort_arcmin"] = features.apply(
        lambda row: apparent_size_sort_key_arcmin(
            row.get("ang_size_maj_arcmin"),
            row.get("ang_size_min_arcmin"),
        ),
        axis=1,
    )
    features["target_name"] = np.where(
        features["common_name"] != "",
        features["primary_id"] + " - " + features["common_name"],
        features["primary_id"],
    )
    return features


def load_catalog_recommendation_features(cache_path: Path) -> pd.DataFrame:
    resolved_path = cache_path.expanduser().resolve()
    cache_mtime_ns, cache_size_bytes = _catalog_cache_fingerprint(resolved_path)
    return _load_catalog_recommendation_features_cached(
        str(resolved_path),
        cache_mtime_ns,
        cache_size_bytes,
    )


@st.cache_resource(show_spinner=False)
def _load_site_date_altaz_bundle_cached(
    cache_path_str: str,
    cache_mtime_ns: int,
    cache_size_bytes: int,
    lat: float,
    lon: float,
    window_start_iso: str,
    window_end_iso: str,
    sample_minutes: int,
) -> dict[str, Any]:
    trace_cache_event(
        "Hydrating site/date alt-az cache "
        f"(lat={lat:.5f}, lon={lon:.5f}, window={window_start_iso}->{window_end_iso}, step={sample_minutes}m)"
    )
    feature_frame = _load_catalog_recommendation_features_cached(
        cache_path_str,
        cache_mtime_ns,
        cache_size_bytes,
    )
    valid_frame = feature_frame[feature_frame["has_valid_coords"]].copy()
    if valid_frame.empty:
        return {
            "primary_ids": tuple(),
            "primary_id_to_col": {},
            "sample_times_local_iso": tuple(),
            "sample_hour_keys": tuple(),
            "altitude_matrix": np.empty((0, 0), dtype=float),
            "wind_index_matrix": np.empty((0, 0), dtype=np.uint8),
            "peak_idx_by_target": np.empty((0,), dtype=np.int32),
            "peak_altitude": np.empty((0,), dtype=float),
            "peak_time_local_iso": tuple(),
            "peak_direction": tuple(),
        }

    sample_times_local = pd.date_range(
        start=pd.Timestamp(window_start_iso),
        end=pd.Timestamp(window_end_iso),
        freq=f"{int(sample_minutes)}min",
        inclusive="both",
    )
    if sample_times_local.empty:
        return {
            "primary_ids": tuple(valid_frame["primary_id"].tolist()),
            "primary_id_to_col": {primary_id: idx for idx, primary_id in enumerate(valid_frame["primary_id"].tolist())},
            "sample_times_local_iso": tuple(),
            "sample_hour_keys": tuple(),
            "altitude_matrix": np.empty((0, len(valid_frame)), dtype=float),
            "wind_index_matrix": np.empty((0, len(valid_frame)), dtype=np.uint8),
            "peak_idx_by_target": np.zeros((len(valid_frame),), dtype=np.int32),
            "peak_altitude": np.zeros((len(valid_frame),), dtype=float),
            "peak_time_local_iso": tuple("" for _ in range(len(valid_frame))),
            "peak_direction": tuple("--" for _ in range(len(valid_frame))),
        }

    sample_times_utc = sample_times_local.tz_convert("UTC")
    time_count = len(sample_times_local)
    target_count = len(valid_frame)
    location_obj = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)

    repeated_ra = np.tile(valid_frame["ra_deg_num"].to_numpy(dtype=float), time_count)
    repeated_dec = np.tile(valid_frame["dec_deg_num"].to_numpy(dtype=float), time_count)
    repeated_times = np.repeat(sample_times_utc.to_pydatetime(), target_count)

    coords = SkyCoord(ra=repeated_ra * u.deg, dec=repeated_dec * u.deg)
    frame = AltAz(obstime=Time(repeated_times), location=location_obj)
    altaz = coords.transform_to(frame)

    altitude_matrix = np.asarray(altaz.alt.deg, dtype=float).reshape(time_count, target_count)
    azimuth_matrix = np.asarray(altaz.az.deg % 360.0, dtype=float).reshape(time_count, target_count)
    wind_index_matrix = (((azimuth_matrix + 11.25) // 22.5).astype(int)) % 16
    wind_index_matrix = wind_index_matrix.astype(np.uint8)

    peak_idx_by_target = np.argmax(altitude_matrix, axis=0).astype(np.int32)
    peak_altitude = altitude_matrix[peak_idx_by_target, np.arange(target_count)]
    peak_time_local_iso = tuple(
        pd.Timestamp(sample_times_local[int(index)]).isoformat()
        for index in peak_idx_by_target
    )
    peak_direction = tuple(
        WIND16[int(wind_index_matrix[int(index), target_idx])]
        for target_idx, index in enumerate(peak_idx_by_target)
    )

    primary_ids = tuple(valid_frame["primary_id"].tolist())
    primary_id_to_col = {
        primary_id: idx
        for idx, primary_id in enumerate(primary_ids)
    }
    sample_times_local_iso = tuple(pd.Timestamp(value).isoformat() for value in sample_times_local.tolist())
    sample_hour_keys = tuple(
        normalize_hour_key(pd.Timestamp(value).floor("h")) or pd.Timestamp(value).floor("h").isoformat()
        for value in sample_times_local
    )

    return {
        "primary_ids": primary_ids,
        "primary_id_to_col": primary_id_to_col,
        "sample_times_local_iso": sample_times_local_iso,
        "sample_hour_keys": sample_hour_keys,
        "altitude_matrix": altitude_matrix,
        "wind_index_matrix": wind_index_matrix,
        "peak_idx_by_target": peak_idx_by_target,
        "peak_altitude": peak_altitude,
        "peak_time_local_iso": peak_time_local_iso,
        "peak_direction": peak_direction,
    }


def load_site_date_altaz_bundle(
    cache_path: Path,
    *,
    lat: float,
    lon: float,
    window_start: datetime,
    window_end: datetime,
    sample_minutes: int = RECOMMENDATION_CACHE_SAMPLE_MINUTES,
) -> dict[str, Any]:
    resolved_path = cache_path.expanduser().resolve()
    cache_mtime_ns, cache_size_bytes = _catalog_cache_fingerprint(resolved_path)
    return _load_site_date_altaz_bundle_cached(
        str(resolved_path),
        cache_mtime_ns,
        cache_size_bytes,
        float(lat),
        float(lon),
        pd.Timestamp(window_start).isoformat(),
        pd.Timestamp(window_end).isoformat(),
        int(sample_minutes),
    )


@st.cache_data(show_spinner=False, ttl=15 * 60)
def load_site_date_weather_mask_bundle(
    *,
    lat: float,
    lon: float,
    tz_name: str,
    window_start_iso: str,
    window_end_iso: str,
    sample_hour_keys: tuple[str, ...],
    cloud_cover_threshold: float = RECOMMENDATION_CLOUD_COVER_THRESHOLD,
) -> dict[str, Any]:
    trace_cache_event(
        "Hydrating site/date weather-mask cache "
        f"(lat={lat:.5f}, lon={lon:.5f}, window={window_start_iso}->{window_end_iso}, "
        f"hours={len(sample_hour_keys)}, cloud<{cloud_cover_threshold:.1f})"
    )
    weather_rows = fetch_hourly_weather(
        lat=lat,
        lon=lon,
        tz_name=tz_name,
        start_local_iso=window_start_iso,
        end_local_iso=window_end_iso,
        hourly_fields=EXTENDED_FORECAST_HOURLY_FIELDS,
    )
    cloud_cover_by_hour: dict[str, float] = {}
    for weather_row in weather_rows:
        hour_key = normalize_hour_key(weather_row.get("time_iso"))
        if not hour_key:
            continue
        try:
            cloud_value = float(weather_row.get("cloud_cover"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(cloud_value):
            cloud_cover_by_hour[hour_key] = float(cloud_value)

    cloud_ok_mask = tuple(
        (hour_key in cloud_cover_by_hour) and (float(cloud_cover_by_hour[hour_key]) < float(cloud_cover_threshold))
        for hour_key in sample_hour_keys
    )
    return {
        "weather_rows": weather_rows,
        "cloud_cover_by_hour": cloud_cover_by_hour,
        "cloud_ok_mask": cloud_ok_mask,
    }
WEATHER_MATRIX_ROWS: list[tuple[str, str]] = [
    ("temperature_2m", "Temperature"),
    ("dewpoint_spread", "Dewpoint Spread"),
    ("cloud_cover", "Cloud Cover"),
    ("visibility", "Visibility"),
    ("wind_gusts_10m", "Wind Gusts"),
]
WEATHER_ALERT_INDICATOR_LEGEND_ITEMS = "❄️ Snow | ⛈️ Rain | ☔ Showers | ⚠️ 1-20% | 🚨 >20%"
WEATHER_ALERT_INDICATOR_LEGEND_CAPTION = f"Weather Alert Indicator: {WEATHER_ALERT_INDICATOR_LEGEND_ITEMS}"
WEATHER_ALERT_RAIN_PRIORITY = ["❄️", "⛈️", "☔", "🚨", "⚠️"]
WEATHER_ALERT_RAIN_INTERVAL_SECONDS = 5 * 60
WEATHER_ALERT_RAIN_DURATION_SECONDS = 30
WEATHER_ALERT_RAIN_BUCKET_STATE_KEY = "weather_alert_rain_last_bucket"
TARGET_DETAIL_MODAL_OPEN_REQUEST_KEY = "target_detail_modal_open_request"
TARGET_DETAIL_MODAL_LAST_TARGET_STATE_KEY = "target_detail_modal_last_target_id"
WEATHER_FORECAST_PERIOD_STATE_KEY = "weather_forecast_period"
WEATHER_FORECAST_PERIOD_TONIGHT = "tonight"
WEATHER_FORECAST_PERIOD_TOMORROW = "tomorrow"
WEATHER_FORECAST_DAY_OFFSET_STATE_KEY = "weather_forecast_day_offset"
ASTRONOMY_FORECAST_NIGHTS = 5
NIGHT_RATING_FACTOR_WEIGHTS: dict[str, float] = {
    "precipitation": 0.05,
    "cloud_coverage": 0.45,
    "visibility": 0.25,
    "wind": 0.15,
    "dew_risk": 0.10,
}
NIGHT_RATING_EMOJIS: dict[int, str] = {
    1: "🚩🚩",
    2: "🚩",
    3: "⭐️",
    4: "⭐️⭐️",
    5: "⭐️⭐️🌟",
}
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
MONTH_DAY_COUNTRY_CODES = {
    "BS",
    "BZ",
    "CA",
    "FM",
    "KY",
    "LR",
    "MH",
    "PH",
    "PW",
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
UNOBSTRUCTED_AREA_CONSTANT_OBSTRUCTION_ALT_DEG = 20.0
DETAIL_PANE_STACK_BREAKPOINT_PX = 800
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
CARDINAL_DIRECTIONS = ("N", "E", "S", "W")
OBSTRUCTION_INPUT_MODE_NESW = "N/E/S/W (4 sliders)"
OBSTRUCTION_INPUT_MODE_WIND16 = "WIND16 (16 sliders)"
OBSTRUCTION_INPUT_MODE_HRZ = "Import .hrz file"
OBSTRUCTION_INPUT_MODES = [
    OBSTRUCTION_INPUT_MODE_NESW,
    OBSTRUCTION_INPUT_MODE_WIND16,
    OBSTRUCTION_INPUT_MODE_HRZ,
]


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


def eval_js_hidden(js_expression: str, *, key: str, want_output: bool = True) -> Any:
    # Keep streamlit_js_eval utility probes from reserving visible layout height.
    wrapped_expression = "(setFrameHeight(0), (" + str(js_expression) + "))"
    return streamlit_js_eval(js_expressions=wrapped_expression, key=key, want_output=want_output)


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


def normalize_weather_forecast_period(value: Any) -> str:
    candidate = str(value or "").strip().lower()
    if candidate == WEATHER_FORECAST_PERIOD_TOMORROW:
        return WEATHER_FORECAST_PERIOD_TOMORROW
    return WEATHER_FORECAST_PERIOD_TONIGHT


def resolve_weather_forecast_period() -> str:
    state_period = normalize_weather_forecast_period(st.session_state.get(WEATHER_FORECAST_PERIOD_STATE_KEY, ""))
    period = state_period
    st.session_state[WEATHER_FORECAST_PERIOD_STATE_KEY] = period
    return period


def normalize_weather_forecast_day_offset(
    value: Any,
    *,
    max_offset: int = ASTRONOMY_FORECAST_NIGHTS - 1,
) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        candidate = 0
    if candidate < 0:
        return 0
    safe_max = max(0, int(max_offset))
    if candidate > safe_max:
        return safe_max
    return candidate


def resolve_weather_forecast_day_offset(
    *,
    max_offset: int = ASTRONOMY_FORECAST_NIGHTS - 1,
) -> int:
    state_raw = st.session_state.get(WEATHER_FORECAST_DAY_OFFSET_STATE_KEY, "")
    if str(state_raw).strip():
        day_offset = normalize_weather_forecast_day_offset(state_raw, max_offset=max_offset)
    else:
        legacy_period = normalize_weather_forecast_period(
            st.session_state.get(WEATHER_FORECAST_PERIOD_STATE_KEY, "")
        )
        legacy_offset = 1 if legacy_period == WEATHER_FORECAST_PERIOD_TOMORROW else 0
        day_offset = normalize_weather_forecast_day_offset(legacy_offset, max_offset=max_offset)

    st.session_state[WEATHER_FORECAST_DAY_OFFSET_STATE_KEY] = day_offset
    st.session_state[WEATHER_FORECAST_PERIOD_STATE_KEY] = (
        WEATHER_FORECAST_PERIOD_TONIGHT if day_offset <= 0 else WEATHER_FORECAST_PERIOD_TOMORROW
    )
    return day_offset


def describe_weather_forecast_period(day_offset: int) -> str:
    if day_offset <= 0:
        return "Tonight"
    if day_offset == 1:
        return "Tomorrow"
    return f"Night +{day_offset}"


def format_hourly_weather_title_period(
    day_offset: int,
    window_start: datetime,
    *,
    browser_locale: str | None = None,
    browser_month_day_pattern: str | None = None,
) -> str:
    if day_offset <= 0:
        return "Tonight"
    if day_offset == 1:
        return "Tomorrow night"

    forecast_date = format_weather_forecast_date(
        window_start,
        browser_locale=browser_locale,
        browser_month_day_pattern=browser_month_day_pattern,
    )
    if "," in forecast_date:
        day_name, date_part = forecast_date.split(",", 1)
        return f"{day_name} night, {date_part.strip()}"

    day_name = pd.Timestamp(window_start).strftime("%A")
    return f"{day_name} night, {forecast_date}"


def format_recommendation_night_title(day_offset: int, window_start: datetime) -> str:
    if day_offset <= 0:
        return "Tonight"
    if day_offset == 1:
        return "Tomorrow Night"
    day_name = pd.Timestamp(window_start).strftime("%A")
    return f"{day_name} Night"


def compute_night_rating_details(
    hourly_weather_rows: list[dict[str, Any]],
    *,
    temperature_unit: str,
) -> dict[str, Any] | None:
    if not hourly_weather_rows:
        return None

    def _finite(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return float(numeric)

    def _average(values: list[float]) -> float | None:
        if not values:
            return None
        return float(sum(values)) / float(len(values))

    def _score_precip_accum_mm(accum_mm: float) -> float:
        if accum_mm <= 0.0:
            return 1.0
        if accum_mm <= 0.1:
            return 0.80
        if accum_mm <= 0.5:
            return 0.45
        if accum_mm <= 1.5:
            return 0.15
        return 0.0

    def _score_cloud_cover(cloud_cover_pct: float) -> float:
        if cloud_cover_pct <= 5.0:
            return 1.0
        if cloud_cover_pct <= 15.0:
            return 0.92
        if cloud_cover_pct <= 30.0:
            return 0.75
        if cloud_cover_pct <= 50.0:
            return 0.50
        if cloud_cover_pct <= 70.0:
            return 0.25
        return 0.05

    def _score_visibility_meters(distance_meters: float) -> float:
        miles = max(0.0, float(distance_meters)) * 0.000621371
        if miles > 6.0:
            return 1.0
        if miles >= 4.0:
            return 0.75
        if miles >= 2.0:
            return 0.40
        return 0.10

    def _score_wind_mph(wind_mph: float) -> float:
        if wind_mph <= 8.0:
            return 1.0
        if wind_mph <= 12.0:
            return 0.85
        if wind_mph <= 18.0:
            return 0.65
        if wind_mph <= 25.0:
            return 0.40
        return 0.20

    def _score_relative_humidity(rh_pct: float) -> float:
        if rh_pct <= 65.0:
            return 1.0
        if rh_pct <= 75.0:
            return 0.90
        if rh_pct <= 85.0:
            return 0.75
        if rh_pct <= 92.0:
            return 0.50
        return 0.30

    def _score_dew_spread_c(spread_c: float) -> float:
        if spread_c >= 5.0:
            return 1.0
        if spread_c >= 3.0:
            return 0.80
        if spread_c >= 2.0:
            return 0.60
        if spread_c >= 1.0:
            return 0.40
        return 0.20

    precip_scores: list[float] = []
    cloud_scores: list[float] = []
    visibility_scores: list[float] = []
    wind_scores: list[float] = []
    dew_scores: list[float] = []
    precip_accum_mm_values: list[float] = []
    wind_mph_values: list[float] = []

    for row in hourly_weather_rows:
        rain_mm = _finite(row.get("rain")) or 0.0
        showers_mm = _finite(row.get("showers")) or 0.0
        snowfall_cm = _finite(row.get("snowfall")) or 0.0
        precip_accum_mm = max(0.0, rain_mm + showers_mm + (snowfall_cm * 10.0))
        precip_accum_mm_values.append(precip_accum_mm)
        precip_scores.append(_score_precip_accum_mm(precip_accum_mm))

        cloud_cover = _finite(row.get("cloud_cover"))
        if cloud_cover is not None:
            cloud_scores.append(_score_cloud_cover(cloud_cover))

        visibility_meters = _finite(row.get("visibility"))
        if visibility_meters is not None:
            visibility_scores.append(_score_visibility_meters(visibility_meters))

        gust_kmh = _finite(row.get("wind_gusts_10m"))
        if gust_kmh is None:
            gust_kmh = _finite(row.get("wind_speed_10m"))
        if gust_kmh is not None:
            wind_mph = gust_kmh * 0.621371
            wind_mph_values.append(wind_mph)
            wind_scores.append(_score_wind_mph(wind_mph))

        dew_component_scores: list[float] = []
        humidity_pct = _finite(row.get("relative_humidity_2m"))
        if humidity_pct is not None:
            dew_component_scores.append(_score_relative_humidity(humidity_pct))
        temp_c = _finite(row.get("temperature_2m"))
        dew_c = _finite(row.get("dew_point_2m"))
        if temp_c is not None and dew_c is not None:
            spread_c = abs(temp_c - dew_c)
            dew_component_scores.append(_score_dew_spread_c(spread_c))
        if dew_component_scores:
            dew_scores.append(min(dew_component_scores))

    factor_scores: dict[str, float | None] = {
        "precipitation": _average(precip_scores),
        "cloud_coverage": _average(cloud_scores),
        "visibility": _average(visibility_scores),
        "wind": _average(wind_scores),
        "dew_risk": _average(dew_scores),
    }
    weighted_total = 0.0
    available_weight = 0.0
    for factor_name, factor_weight in NIGHT_RATING_FACTOR_WEIGHTS.items():
        factor_score = factor_scores.get(factor_name)
        if factor_score is None:
            continue
        weighted_total += float(factor_weight) * float(factor_score)
        available_weight += float(factor_weight)

    if available_weight <= 0.0:
        return None

    normalized_score = weighted_total / available_weight
    raw_rating = int(np.clip(np.ceil(normalized_score * 5.0), 1, 5))
    rating = raw_rating
    rating_caps: list[dict[str, Any]] = []

    if precip_accum_mm_values:
        serious_precip_hours = sum(1 for value in precip_accum_mm_values if value >= 0.1)
        heavy_precip_hours = sum(1 for value in precip_accum_mm_values if value >= 0.5)
        total_precip_mm = float(sum(precip_accum_mm_values))
        precip_cap: int | None = None
        precip_reason = ""
        if heavy_precip_hours > 0 or total_precip_mm >= 2.0:
            precip_cap = 2
            precip_reason = "Heavy precipitation accumulation risk."
        elif serious_precip_hours > 0 or total_precip_mm >= 0.5:
            precip_cap = 3
            precip_reason = "Serious precipitation accumulation risk."
        elif total_precip_mm > 0.0:
            precip_cap = 4
            precip_reason = "Light precipitation accumulation risk."

        if precip_cap is not None and rating > precip_cap:
            rating = precip_cap
            rating_caps.append({"max_rating": precip_cap, "reason": precip_reason})

    cloud_score = factor_scores.get("cloud_coverage")
    visibility_score = factor_scores.get("visibility")
    if (
        wind_mph_values
        and cloud_score is not None
        and visibility_score is not None
        and cloud_score >= 0.80
        and visibility_score >= 0.80
    ):
        max_wind_mph = max(wind_mph_values)
        if max_wind_mph >= 20.0 and rating > 4:
            rating = 4
            rating_caps.append(
                {
                    "max_rating": 4,
                    "reason": "Strong wind prevents a 5/5 despite clear sky and visibility.",
                }
            )

    dew_score = factor_scores.get("dew_risk")
    if dew_score is not None and dew_score < 0.75 and rating > 4:
        rating = 4
        rating_caps.append(
            {
                "max_rating": 4,
                "reason": "Elevated dew risk prevents a 5/5 night.",
            }
        )

    emoji = NIGHT_RATING_EMOJIS.get(rating, "⭐️")
    factor_rows: list[dict[str, Any]] = []
    factor_definitions = (
        ("precipitation", "Precip accumulation risk", precip_scores),
        ("cloud_coverage", "Cloud cover quality", cloud_scores),
        ("visibility", "Visibility quality", visibility_scores),
        ("wind", "Wind stability", wind_scores),
        ("dew_risk", "Dew risk (humidity/spread)", dew_scores),
    )
    for factor_key, factor_label, factor_values in factor_definitions:
        factor_weight = float(NIGHT_RATING_FACTOR_WEIGHTS.get(factor_key, 0.0))
        factor_score = factor_scores.get(factor_key)
        data_hours = len(factor_values)
        pass_hours = sum(1 for value in factor_values if float(value) >= 0.70)
        factor_rows.append(
            {
                "key": factor_key,
                "label": factor_label,
                "weight": factor_weight,
                "score": factor_score,
                "data_hours": data_hours,
                "pass_hours": pass_hours,
                "weighted_contribution": (
                    factor_weight * float(factor_score) if factor_score is not None else None
                ),
            }
        )

    return {
        "rating": rating,
        "raw_rating": raw_rating,
        "emoji": emoji,
        "normalized_score": normalized_score,
        "available_weight": available_weight,
        "factors": factor_rows,
        "caps": rating_caps,
    }


def compute_night_rating(
    hourly_weather_rows: list[dict[str, Any]],
    *,
    temperature_unit: str,
) -> tuple[int, str] | None:
    details = compute_night_rating_details(
        hourly_weather_rows,
        temperature_unit=temperature_unit,
    )
    if details is None:
        return None
    return int(details["rating"]), str(details["emoji"])


def format_night_rating_tooltip(rating_details: dict[str, Any] | None) -> str:
    if not isinstance(rating_details, dict):
        return ""

    try:
        rating = int(rating_details.get("rating", 0))
    except (TypeError, ValueError):
        rating = 0
    try:
        raw_rating = int(rating_details.get("raw_rating", rating))
    except (TypeError, ValueError):
        raw_rating = rating
    emoji = str(rating_details.get("emoji", "")).strip()
    normalized_score = rating_details.get("normalized_score")
    available_weight = rating_details.get("available_weight")
    try:
        weighted_pct_text = f"{float(normalized_score) * 100.0:.0f}%"
    except (TypeError, ValueError):
        weighted_pct_text = "-"
    try:
        available_weight_pct_text = f"{float(available_weight) * 100.0:.0f}%"
    except (TypeError, ValueError):
        available_weight_pct_text = "-"

    lines: list[str] = [
        f"Night rating: {rating}/5 {emoji}".strip(),
        f"Weighted score: {weighted_pct_text} (weights in use: {available_weight_pct_text})",
    ]
    if raw_rating != rating:
        lines.append(f"Raw rating before caps: {raw_rating}/5")
    caps = rating_details.get("caps", [])
    if isinstance(caps, list) and caps:
        lines.append("Caps applied:")
        for cap in caps:
            if not isinstance(cap, dict):
                continue
            reason = str(cap.get("reason", "")).strip()
            try:
                max_rating = int(cap.get("max_rating", 0))
            except (TypeError, ValueError):
                max_rating = 0
            if max_rating > 0 and reason:
                lines.append(f"- max {max_rating}/5: {reason}")
            elif reason:
                lines.append(f"- {reason}")

    lines.append("Factors:")

    raw_factors = rating_details.get("factors", [])
    if isinstance(raw_factors, list):
        for factor in raw_factors:
            if not isinstance(factor, dict):
                continue
            label = str(factor.get("label", "")).strip()
            if not label:
                continue
            try:
                weight_pct = float(factor.get("weight", 0.0)) * 100.0
            except (TypeError, ValueError):
                weight_pct = 0.0
            score = factor.get("score")
            if score is None:
                lines.append(f"- {label} (w {weight_pct:.0f}%): no data")
                continue
            try:
                score_pct = float(score) * 100.0
            except (TypeError, ValueError):
                lines.append(f"- {label} (w {weight_pct:.0f}%): no data")
                continue
            pass_hours = int(factor.get("pass_hours", 0))
            data_hours = int(factor.get("data_hours", 0))
            lines.append(
                f"- {label} (w {weight_pct:.0f}%): {score_pct:.0f}% ({pass_hours}/{data_hours} hrs)"
            )

    return "\n".join(lines)


def format_condition_tips_title(
    day_offset: int,
    window_start: datetime,
    *,
    rating_emoji: str | None = None,
) -> str:
    if day_offset <= 0:
        base_title = "Tonight's Conditions"
    elif day_offset == 1:
        base_title = "Tomorrow's Conditions"
    else:
        day_name = pd.Timestamp(window_start).strftime("%A")
        base_title = f"{day_name}'s Conditions"

    if not str(rating_emoji or "").strip():
        return base_title

    return f"{base_title} {str(rating_emoji).strip()}"


def weather_forecast_window(
    lat: float,
    lon: float,
    *,
    day_offset: int = 0,
) -> tuple[datetime, datetime, ZoneInfo]:
    tonight_start, tonight_end, tzinfo = tonight_window(lat, lon)
    if day_offset <= 0:
        return tonight_start, tonight_end, tzinfo

    tz_name = tzinfo.key
    sunset_date = tonight_start.date() + timedelta(days=day_offset)
    try:
        loc = LocationInfo(latitude=lat, longitude=lon, timezone=tz_name)
        sun_on_day = sun(loc.observer, date=sunset_date, tzinfo=tzinfo)
        sun_next_day = sun(loc.observer, date=sunset_date + timedelta(days=1), tzinfo=tzinfo)
        start = sun_on_day["sunset"]
        end = sun_next_day["sunrise"]
    except Exception:
        start = datetime.combine(sunset_date, time(18, 0), tzinfo=tzinfo)
        end = datetime.combine(sunset_date + timedelta(days=1), time(6, 0), tzinfo=tzinfo)

    if end <= start:
        end = start + timedelta(hours=12)
    return start, end, tzinfo


def astronomical_night_window(
    lat: float,
    lon: float,
    *,
    day_offset: int = 0,
) -> tuple[datetime, datetime, ZoneInfo]:
    forecast_start, forecast_end, tzinfo = weather_forecast_window(lat, lon, day_offset=day_offset)
    tz_name = tzinfo.key
    sunset_date = forecast_start.date()

    try:
        loc = LocationInfo(latitude=lat, longitude=lon, timezone=tz_name)
        sun_on_day = sun(
            loc.observer,
            date=sunset_date,
            tzinfo=tzinfo,
            dawn_dusk_depression=18,
        )
        sun_next_day = sun(
            loc.observer,
            date=sunset_date + timedelta(days=1),
            tzinfo=tzinfo,
            dawn_dusk_depression=18,
        )
        start = sun_on_day["dusk"]
        end = sun_next_day["dawn"]
    except Exception:
        start = forecast_start
        end = forecast_end

    if end <= start:
        start = forecast_start
        end = forecast_end
    return start, end, tzinfo


def infer_month_day_order_from_locale(locale_value: str | None) -> str:
    if not locale_value:
        return "md"

    normalized = str(locale_value).replace("-", "_")
    parts = [part for part in normalized.split("_") if part]
    country = ""
    if len(parts) >= 2:
        candidate = parts[-1].upper()
        if len(candidate) == 2 and candidate.isalpha():
            country = candidate

    if country in MONTH_DAY_COUNTRY_CODES:
        return "md"
    return "dm"


def parse_browser_month_day_pattern(value: Any) -> tuple[str, str] | None:
    raw = str(value or "").strip()
    if not raw:
        return None

    order: list[str] = []
    separator = "/"
    for token in raw.split("|"):
        if ":" not in token:
            continue
        token_type, token_value = token.split(":", 1)
        normalized_type = token_type.strip().lower()
        if normalized_type == "month":
            order.append("month")
            continue
        if normalized_type == "day":
            order.append("day")
            continue
        if normalized_type == "literal":
            literal = token_value
            if literal and len(literal) <= 3:
                separator = literal

    if order[:2] == ["month", "day"]:
        return "md", separator
    if order[:2] == ["day", "month"]:
        return "dm", separator
    return None


def format_weather_forecast_date(
    value: datetime,
    *,
    browser_locale: str | None = None,
    browser_month_day_pattern: str | None = None,
) -> str:
    local_date = pd.Timestamp(value).date()
    date_part = format_weather_forecast_month_day(
        value,
        browser_locale=browser_locale,
        browser_month_day_pattern=browser_month_day_pattern,
    )
    return f"{local_date.strftime('%A')}, {date_part}"


def format_weather_forecast_month_day(
    value: datetime,
    *,
    browser_locale: str | None = None,
    browser_month_day_pattern: str | None = None,
) -> str:
    local_date = pd.Timestamp(value).date()
    parsed_pattern = parse_browser_month_day_pattern(browser_month_day_pattern)
    if parsed_pattern is not None:
        month_day_order, separator = parsed_pattern
    else:
        month_day_order = infer_month_day_order_from_locale(browser_locale)
        separator = "/"

    date_part = (
        f"{local_date.month}{separator}{local_date.day}"
        if month_day_order == "md"
        else f"{local_date.day}{separator}{local_date.month}"
    )
    return date_part


def resolve_location_display_name(location: dict[str, Any]) -> str:
    label = str(location.get("label") or "").strip()
    if label:
        return label
    try:
        lat = float(location.get("lat"))
        lon = float(location.get("lon"))
    except (TypeError, ValueError):
        return "your location"
    return f"{lat:.4f}, {lon:.4f}"


def is_location_configured(location: dict[str, Any] | None) -> bool:
    if not isinstance(location, dict):
        return False

    source = str(location.get("source", "")).strip().lower()
    if source in {"", "unset", "default"}:
        return False

    try:
        lat = float(location.get("lat"))
        lon = float(location.get("lon"))
    except (TypeError, ValueError):
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def is_default_site_name(label: Any) -> bool:
    normalized = str(label or "").strip().lower()
    return normalized in {"", "location not set", "observation site"}


def resolve_location_source_badge(source: Any) -> tuple[str, str] | None:
    normalized = str(source or "").strip().lower()
    if normalized in {"manual", "search", "map"}:
        return ("found", "manual")
    if normalized == "browser":
        return ("Browser", "browser")
    if normalized == "ip":
        return ("IP", "ip")
    return None


def default_site_definition(name: str = DEFAULT_SITE_NAME) -> dict[str, Any]:
    site_name = str(name or "").strip() or DEFAULT_SITE_NAME
    location = copy.deepcopy(DEFAULT_LOCATION)
    location["label"] = site_name
    return {
        "name": site_name,
        "location": location,
        "obstructions": {direction: 20.0 for direction in WIND16},
    }


def site_ids_in_order(prefs: dict[str, Any]) -> list[str]:
    sites = prefs.get("sites", {})
    if not isinstance(sites, dict):
        return []

    ordered: list[str] = []
    raw_order = prefs.get("site_order", [])
    if isinstance(raw_order, (list, tuple)):
        for raw_site_id in raw_order:
            site_id = str(raw_site_id).strip()
            if site_id and site_id in sites and site_id not in ordered:
                ordered.append(site_id)

    for raw_site_id in sites.keys():
        site_id = str(raw_site_id).strip()
        if site_id and site_id not in ordered:
            ordered.append(site_id)
    return ordered


def get_active_site_id(prefs: dict[str, Any]) -> str:
    ordered = site_ids_in_order(prefs)
    if not ordered:
        return DEFAULT_SITE_ID
    candidate = str(prefs.get("active_site_id", "")).strip()
    return candidate if candidate in ordered else ordered[0]


def get_site_definition(prefs: dict[str, Any], site_id: str) -> dict[str, Any]:
    sites = prefs.get("sites", {})
    if not isinstance(sites, dict):
        return default_site_definition()
    site = sites.get(site_id)
    if isinstance(site, dict):
        return site
    return default_site_definition()


def sync_active_site_to_legacy_fields(prefs: dict[str, Any]) -> None:
    ordered = site_ids_in_order(prefs)
    if not ordered:
        default_site = default_site_definition()
        prefs["sites"] = {DEFAULT_SITE_ID: default_site}
        prefs["site_order"] = [DEFAULT_SITE_ID]
        prefs["active_site_id"] = DEFAULT_SITE_ID
        ordered = [DEFAULT_SITE_ID]

    active_site_id = get_active_site_id(prefs)
    active_site = get_site_definition(prefs, active_site_id)
    site_name = str(active_site.get("name") or "").strip() or DEFAULT_SITE_NAME
    location = copy.deepcopy(active_site.get("location", DEFAULT_LOCATION))
    if not str(location.get("label") or "").strip():
        location["label"] = site_name
    obstructions_raw = active_site.get("obstructions", {})
    obstructions = {
        direction: clamp_obstruction_altitude(
            obstructions_raw.get(direction, 20.0) if isinstance(obstructions_raw, dict) else 20.0,
            default=20.0,
        )
        for direction in WIND16
    }

    prefs["active_site_id"] = active_site_id
    prefs["location"] = location
    prefs["obstructions"] = obstructions


def persist_legacy_fields_to_active_site(prefs: dict[str, Any]) -> None:
    sites = prefs.get("sites", {})
    if not isinstance(sites, dict):
        sites = {}
    active_site_id = get_active_site_id(prefs)
    current_site = get_site_definition(prefs, active_site_id)

    location = prefs.get("location", {})
    merged_location = copy.deepcopy(DEFAULT_LOCATION)
    if isinstance(location, dict):
        for key in merged_location:
            if key in location:
                merged_location[key] = location.get(key, merged_location[key])
    location_label = str(merged_location.get("label") or "").strip()

    obstructions = {
        direction: clamp_obstruction_altitude(
            prefs.get("obstructions", {}).get(direction, 20.0) if isinstance(prefs.get("obstructions"), dict) else 20.0,
            default=20.0,
        )
        for direction in WIND16
    }

    site_name = str(current_site.get("name") or "").strip()
    if location_label:
        site_name = location_label
    if not site_name:
        site_name = DEFAULT_SITE_NAME
        merged_location["label"] = site_name

    sites[active_site_id] = {
        "name": site_name,
        "location": merged_location,
        "obstructions": obstructions,
    }
    prefs["sites"] = sites

    ordered = site_ids_in_order(prefs)
    if active_site_id not in ordered:
        ordered.append(active_site_id)
    prefs["site_order"] = ordered
    prefs["active_site_id"] = active_site_id
    prefs["location"] = copy.deepcopy(merged_location)
    prefs["obstructions"] = copy.deepcopy(obstructions)


def set_active_site(prefs: dict[str, Any], site_id: str) -> bool:
    ordered = site_ids_in_order(prefs)
    if site_id not in ordered:
        return False
    changed = str(prefs.get("active_site_id", "")).strip() != site_id
    prefs["active_site_id"] = site_id
    sync_active_site_to_legacy_fields(prefs)
    return changed


def duplicate_site(prefs: dict[str, Any], site_id: str) -> str | None:
    source_site = get_site_definition(prefs, site_id)
    source_name = str(source_site.get("name") or "").strip() or DEFAULT_SITE_NAME
    sites = prefs.get("sites", {})
    if not isinstance(sites, dict):
        sites = {}

    existing_names = {
        str(site.get("name") or "").strip()
        for site in sites.values()
        if isinstance(site, dict) and str(site.get("name") or "").strip()
    }
    copy_index = 1
    candidate_name = f"{source_name} - copy {copy_index}"
    while candidate_name in existing_names:
        copy_index += 1
        candidate_name = f"{source_name} - copy {copy_index}"

    duplicated = copy.deepcopy(source_site)
    duplicated["name"] = candidate_name
    duplicated_location = duplicated.get("location", {})
    if isinstance(duplicated_location, dict):
        duplicated_location["label"] = candidate_name
        duplicated["location"] = duplicated_location

    new_site_id = f"site_{uuid.uuid4().hex[:8]}"
    sites[new_site_id] = duplicated
    prefs["sites"] = sites

    ordered = site_ids_in_order(prefs)
    if site_id in ordered:
        insert_idx = ordered.index(site_id) + 1
        ordered.insert(insert_idx, new_site_id)
    else:
        ordered.append(new_site_id)
    prefs["site_order"] = ordered
    return new_site_id


def create_site(prefs: dict[str, Any], name: str | None = None) -> str | None:
    sites = prefs.get("sites", {})
    if not isinstance(sites, dict):
        sites = {}

    base_name = str(name or "").strip() or DEFAULT_SITE_NAME
    existing_names = {
        str(site.get("name") or "").strip().lower()
        for site in sites.values()
        if isinstance(site, dict) and str(site.get("name") or "").strip()
    }
    candidate_name = base_name
    suffix = 2
    while candidate_name.strip().lower() in existing_names:
        candidate_name = f"{base_name} {suffix}"
        suffix += 1

    new_site_id = f"site_{uuid.uuid4().hex[:8]}"
    sites[new_site_id] = default_site_definition(candidate_name)
    prefs["sites"] = sites

    ordered = site_ids_in_order(prefs)
    if new_site_id not in ordered:
        ordered.append(new_site_id)
    prefs["site_order"] = ordered
    return new_site_id


def delete_site(prefs: dict[str, Any], site_id: str) -> bool:
    ordered = site_ids_in_order(prefs)
    if site_id not in ordered or len(ordered) <= 1:
        return False

    sites = prefs.get("sites", {})
    if not isinstance(sites, dict):
        return False
    sites.pop(site_id, None)
    prefs["sites"] = sites

    ordered = [item for item in ordered if item != site_id]
    prefs["site_order"] = ordered
    if str(prefs.get("active_site_id", "")).strip() == site_id:
        prefs["active_site_id"] = ordered[0]
    sync_active_site_to_legacy_fields(prefs)
    return True


def get_site_name(prefs: dict[str, Any], site_id: str) -> str:
    site = get_site_definition(prefs, site_id)
    name = str(site.get("name") or "").strip()
    if name:
        return name
    location = site.get("location", {})
    if isinstance(location, dict):
        location_label = str(location.get("label") or "").strip()
        if location_label:
            return location_label
    return "Observation Site"


def load_equipment_catalog(catalog_path: str = str(EQUIPMENT_CATALOG_PATH)) -> dict[str, Any]:
    try:
        raw_payload = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    except Exception:
        return {"categories": []}

    categories: list[dict[str, Any]] = []
    raw_categories = raw_payload.get("categories", [])
    if not isinstance(raw_categories, list):
        return {"categories": []}

    for raw_category in raw_categories:
        if not isinstance(raw_category, dict):
            continue
        category_id = str(raw_category.get("id") or "").strip()
        if not category_id:
            continue
        label = str(raw_category.get("label") or category_id).strip() or category_id
        description = str(raw_category.get("description") or "").strip()

        display_columns: list[dict[str, str]] = []
        raw_columns = raw_category.get("display_columns", [])
        if isinstance(raw_columns, list):
            for raw_column in raw_columns:
                if not isinstance(raw_column, dict):
                    continue
                field = str(raw_column.get("field") or "").strip()
                if not field:
                    continue
                column_label = str(raw_column.get("label") or field).strip() or field
                display_columns.append({"field": field, "label": column_label})

        items: list[dict[str, Any]] = []
        raw_items = raw_category.get("items", [])
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                item_id = str(raw_item.get("id") or "").strip()
                item_name = str(raw_item.get("name") or "").strip()
                if not item_id or not item_name:
                    continue
                item_payload = {"id": item_id, "name": item_name}
                for key, value in raw_item.items():
                    if key in {"id", "name"}:
                        continue
                    item_payload[str(key)] = value
                items.append(item_payload)

        categories.append(
            {
                "id": category_id,
                "label": label,
                "description": description,
                "display_columns": display_columns,
                "items": items,
            }
        )

    return {"categories": categories}


def _normalize_mount_choice(value: Any, *, default_choice: str = "altaz") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"eq", "equatorial", "equatorial-mount"}:
        return "eq"
    if normalized in {"altaz", "alt/az", "alt-az", "alt az"}:
        return "altaz"
    return default_choice


def mount_choice_label(choice: str) -> str:
    return "EQ" if str(choice).strip().lower() == "eq" else "Alt/Az"


def build_owned_equipment_context(prefs: dict[str, Any]) -> dict[str, Any]:
    equipment_catalog = load_equipment_catalog()
    categories = equipment_catalog.get("categories", [])
    category_items_by_id: dict[str, dict[str, dict[str, Any]]] = {}
    if isinstance(categories, list):
        for category in categories:
            if not isinstance(category, dict):
                continue
            category_id = str(category.get("id", "")).strip()
            if not category_id:
                continue
            items = category.get("items", [])
            if not isinstance(items, list):
                continue
            item_lookup = {
                str(item.get("id", "")).strip(): item
                for item in items
                if isinstance(item, dict) and str(item.get("id", "")).strip()
            }
            category_items_by_id[category_id] = item_lookup

    owned_equipment = prefs.get("equipment", {})
    if not isinstance(owned_equipment, dict):
        owned_equipment = {}

    telescope_lookup = category_items_by_id.get("telescopes", {})
    filter_lookup = category_items_by_id.get("filters", {})

    owned_telescope_ids = [
        item_id
        for item_id in [str(item).strip() for item in owned_equipment.get("telescopes", []) if str(item).strip()]
        if item_id in telescope_lookup
    ]
    owned_filter_ids = [
        item_id
        for item_id in [str(item).strip() for item in owned_equipment.get("filters", []) if str(item).strip()]
        if item_id in filter_lookup
    ]
    owned_accessory_ids = {
        str(item).strip()
        for item in owned_equipment.get("accessories", [])
        if str(item).strip()
    }
    owned_telescopes = [telescope_lookup[item_id] for item_id in owned_telescope_ids]
    owned_filters = [filter_lookup[item_id] for item_id in owned_filter_ids]

    return {
        "category_items_by_id": category_items_by_id,
        "telescope_lookup": telescope_lookup,
        "filter_lookup": filter_lookup,
        "owned_telescope_ids": owned_telescope_ids,
        "owned_filter_ids": owned_filter_ids,
        "owned_accessory_ids": owned_accessory_ids,
        "owned_telescopes": owned_telescopes,
        "owned_filters": owned_filters,
        "eq_owned": "equatorial-mount" in owned_accessory_ids,
    }


def sync_active_equipment_settings(
    prefs: dict[str, Any],
    equipment_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = equipment_context if isinstance(equipment_context, dict) else build_owned_equipment_context(prefs)

    owned_telescope_ids = list(context.get("owned_telescope_ids", []))
    owned_filter_ids = list(context.get("owned_filter_ids", []))
    telescope_lookup = context.get("telescope_lookup", {})
    filter_lookup = context.get("filter_lookup", {})

    default_mount_choice = "eq" if bool(context.get("eq_owned", False)) else "altaz"
    current_mount_choice = _normalize_mount_choice(
        prefs.get("active_mount_choice", default_mount_choice),
        default_choice=default_mount_choice,
    )

    current_telescope_id = str(prefs.get("active_telescope_id", "")).strip()
    if owned_telescope_ids:
        active_telescope_id = (
            current_telescope_id if current_telescope_id in owned_telescope_ids else owned_telescope_ids[0]
        )
    else:
        active_telescope_id = ""

    current_filter_id = str(prefs.get("active_filter_id", "__none__")).strip()
    valid_filter_ids = {"__none__"} | set(owned_filter_ids)
    active_filter_id = current_filter_id if current_filter_id in valid_filter_ids else "__none__"

    changed = False
    if str(prefs.get("active_telescope_id", "")).strip() != active_telescope_id:
        prefs["active_telescope_id"] = active_telescope_id
        changed = True
    if str(prefs.get("active_filter_id", "__none__")).strip() != active_filter_id:
        prefs["active_filter_id"] = active_filter_id
        changed = True
    if _normalize_mount_choice(
        prefs.get("active_mount_choice", default_mount_choice),
        default_choice=default_mount_choice,
    ) != current_mount_choice:
        prefs["active_mount_choice"] = current_mount_choice
        changed = True

    active_telescope = telescope_lookup.get(active_telescope_id) if active_telescope_id else None
    active_filter = filter_lookup.get(active_filter_id) if active_filter_id != "__none__" else None

    return {
        "changed": changed,
        "active_telescope_id": active_telescope_id,
        "active_filter_id": active_filter_id,
        "active_mount_choice": current_mount_choice,
        "active_telescope": active_telescope,
        "active_filter": active_filter,
        **context,
    }


def format_equipment_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "; ".join(parts) if parts else "-"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "-"
        return f"{value:.6g}"
    text = str(value).strip()
    return text if text else "-"


def clamp_obstruction_altitude(value: Any, *, default: float = 20.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    return float(max(0.0, min(90.0, numeric)))


def expand_cardinal_obstructions_to_wind16(cardinal_values: dict[str, Any]) -> dict[str, float]:
    north = clamp_obstruction_altitude(cardinal_values.get("N"), default=20.0)
    east = clamp_obstruction_altitude(cardinal_values.get("E"), default=20.0)
    south = clamp_obstruction_altitude(cardinal_values.get("S"), default=20.0)
    west = clamp_obstruction_altitude(cardinal_values.get("W"), default=20.0)

    def _lerp(start: float, end: float, t: float) -> float:
        return start + ((end - start) * t)

    expanded: dict[str, float] = {}
    for direction in WIND16:
        angle = wind_bin_center(direction)
        if angle < 90.0:
            fraction = angle / 90.0
            raw_value = _lerp(north, east, fraction)
        elif angle < 180.0:
            fraction = (angle - 90.0) / 90.0
            raw_value = _lerp(east, south, fraction)
        elif angle < 270.0:
            fraction = (angle - 180.0) / 90.0
            raw_value = _lerp(south, west, fraction)
        else:
            fraction = (angle - 270.0) / 90.0
            raw_value = _lerp(west, north, fraction)
        expanded[direction] = clamp_obstruction_altitude(raw_value, default=20.0)
    return expanded


def parse_hrz_obstruction_points(raw_text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for raw_line in str(raw_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith(";") or line.startswith("//"):
            continue

        numeric_tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
        if len(numeric_tokens) < 2:
            continue

        try:
            azimuth = float(numeric_tokens[0]) % 360.0
            altitude = clamp_obstruction_altitude(numeric_tokens[1], default=0.0)
        except (TypeError, ValueError):
            continue
        points.append((azimuth, altitude))
    return points


def reduce_hrz_points_to_wind16(
    points: list[tuple[float, float]],
    *,
    fallback: dict[str, Any] | None = None,
) -> tuple[dict[str, float], list[str]]:
    maxima_by_direction: dict[str, float | None] = {direction: None for direction in WIND16}
    for azimuth, altitude in points:
        direction = az_to_wind16(float(azimuth))
        clamped_altitude = clamp_obstruction_altitude(altitude, default=0.0)
        previous = maxima_by_direction.get(direction)
        if previous is None or clamped_altitude > previous:
            maxima_by_direction[direction] = clamped_altitude

    reduced: dict[str, float] = {}
    missing_directions: list[str] = []
    fallback_values = fallback or {}
    for direction in WIND16:
        maximum = maxima_by_direction.get(direction)
        if maximum is None:
            missing_directions.append(direction)
            reduced[direction] = clamp_obstruction_altitude(fallback_values.get(direction), default=20.0)
        else:
            reduced[direction] = float(maximum)
    return reduced, missing_directions


def wind16_obstructions_to_hrz_text(obstructions: dict[str, Any]) -> str:
    wind16_values = {
        direction: clamp_obstruction_altitude(obstructions.get(direction), default=20.0)
        for direction in WIND16
    }
    lines: list[str] = []
    for azimuth in range(360):
        direction = az_to_wind16(float(azimuth))
        altitude = wind16_values.get(direction, 20.0)
        lines.append(f"{azimuth} {altitude:.1f}")
    return "\n".join(lines) + "\n"


def sync_slider_state_value(state_key: str, target_value: float) -> None:
    rounded_target = int(round(clamp_obstruction_altitude(target_value, default=20.0)))
    sync_key = f"{state_key}_synced"
    if int(st.session_state.get(sync_key, -1)) != rounded_target:
        st.session_state[state_key] = rounded_target
        st.session_state[sync_key] = rounded_target


def build_location_selection_map(lat: float, lon: float, *, zoom_start: int) -> Any | None:
    if folium is None or MacroElement is None or Template is None:
        return None

    map_obj = folium.Map(
        location=[float(lat), float(lon)],
        zoom_start=int(zoom_start),
        control_scale=True,
    )
    folium.Marker(
        [float(lat), float(lon)],
        tooltip="Current site location",
    ).add_to(map_obj)
    bridge = MacroElement()
    bridge._template = Template(
        """
        {% macro script(this, kwargs) %}
        const map = {{ this._parent.get_name() }};
        let dsoRightClickMarker = null;

        map.on("contextmenu", function(evt) {
            if (dsoRightClickMarker) {
                map.removeLayer(dsoRightClickMarker);
            }
            dsoRightClickMarker = L.marker(evt.latlng).addTo(map);
            map.fire("click", { latlng: evt.latlng });
        });
        {% endmacro %}
        """
    )
    map_obj.add_child(bridge)
    return map_obj


def apply_resolved_location(prefs: dict[str, Any], resolved_location: dict[str, Any]) -> tuple[str, bool]:
    active_site = get_site_definition(prefs, get_active_site_id(prefs))
    current_site_name = str(active_site.get("name") or "").strip()
    resolved_label = str(resolved_location.get("label") or "").strip()

    keep_existing_site_name = bool(current_site_name) and not is_default_site_name(current_site_name)
    next_label = current_site_name if keep_existing_site_name else resolved_label
    if not next_label:
        next_label = current_site_name or resolved_label or DEFAULT_SITE_NAME

    lat = float(resolved_location["lat"])
    lon = float(resolved_location["lon"])
    source = str(resolved_location.get("source") or "manual").strip() or "manual"
    resolved_at = str(resolved_location.get("resolved_at") or datetime.now(timezone.utc).isoformat()).strip()

    prefs["location"] = {
        "lat": lat,
        "lon": lon,
        "label": next_label,
        "source": source,
        "resolved_at": resolved_at,
    }
    persist_legacy_fields_to_active_site(prefs)

    result_label = resolved_label or f"{lat:.4f}, {lon:.4f}"
    return result_label, keep_existing_site_name


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


def format_duration_hhmm(duration: timedelta) -> str:
    total_minutes = max(0, int(round(duration.total_seconds() / 60.0)))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def format_ra_hms(ra_deg: float | None) -> str:
    if ra_deg is None:
        return "-"
    try:
        ra_value = float(ra_deg)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(ra_value):
        return "-"

    total_seconds = int(round((ra_value % 360.0) * 240.0))
    total_seconds %= 24 * 3600
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"


def format_dec_dms(dec_deg: float | None) -> str:
    if dec_deg is None:
        return "-"
    try:
        dec_value = float(dec_deg)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(dec_value):
        return "-"

    clamped_dec = max(-90.0, min(90.0, dec_value))
    sign = "-" if clamped_dec < 0 else "+"
    total_seconds = int(round(abs(clamped_dec) * 3600.0))
    degrees, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{sign}{degrees:02d}° {minutes:02d}' {seconds:02d}\""


def build_sky_position_summary_rows(
    selected_id: str | None,
    selected_label: str | None,
    selected_type_group: str | None,
    selected_color: str | None,
    selected_events: dict[str, pd.Series | None] | None,
    selected_track: pd.DataFrame | None,
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

    rows: list[dict[str, Any]] = []
    selected_primary_id = str(selected_id or "").strip()
    if selected_primary_id and isinstance(selected_track, pd.DataFrame):
        selected_row = _build_row(
            selected_primary_id,
            str(selected_label or selected_primary_id),
            str(selected_type_group or "other"),
            str(selected_color or "#22c55e"),
            selected_events or {},
            selected_primary_id in list_member_ids,
        )
        selected_row["visible_total"] = format_duration_hm(compute_total_visible_time(selected_track))
        selected_row["visible_remaining"] = format_duration_hm(
            compute_remaining_visible_time(selected_track, now=now_local)
        )
        rows.append(selected_row)
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
    is_dark_theme = is_dark_ui_theme()
    theme_palette = resolve_theme_palette()
    dark_table_styles = theme_palette.get("dataframe_styler", {})
    dark_td_bg = str(dark_table_styles.get("td_bg", "#0F172A"))
    dark_td_text = str(dark_table_styles.get("td_text", "#E5E7EB"))

    def _style_summary_row(row: pd.Series) -> list[str]:
        base_cell_style = f"background-color: {dark_td_bg}; color: {dark_td_text};" if is_dark_theme else ""
        styles = [base_cell_style for _ in row]
        color = str(summary_df.loc[row.name, "line_color"]).strip()
        row_primary_id = str(summary_df.loc[row.name, "primary_id"]).strip()
        selected_detail_id = str(st.session_state.get("selected_id") or "").strip()
        highlighted_summary_id = str(st.session_state.get("sky_summary_highlight_primary_id", "")).strip()
        row_is_selected = (
            row_primary_id
            and (
                (selected_detail_id and row_primary_id == selected_detail_id)
                or (highlighted_summary_id and row_primary_id == highlighted_summary_id)
            )
        )
        if row_is_selected:
            selected_bg = _muted_rgba_from_hex(color, alpha=0.16)
            for idx in range(len(styles)):
                base_style = styles[idx]
                if base_style and not base_style.endswith(";"):
                    base_style = f"{base_style};"
                styles[idx] = f"{base_style} background-color: {selected_bg};"
        if color:
            line_idx = row.index.get_loc("Line")
            base_style = styles[line_idx]
            if base_style and not base_style.endswith(";"):
                base_style = f"{base_style};"
            styles[line_idx] = f"{base_style} color: {color}; font-weight: 700;"
        return styles

    styled = apply_dataframe_styler_theme(display.style.apply(_style_summary_row, axis=1))

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
    # Keep the last non-empty selection token so unrelated reruns (for example,
    # style toggles) do not turn a stale row/cell selection into a "new" click.
    if selection_token:
        st.session_state["sky_summary_last_selection_token"] = selection_token

    current_highlight_id = str(st.session_state.get("sky_summary_highlight_primary_id", ""))
    if selection_changed and selected_primary_id and selected_primary_id != current_highlight_id:
        st.session_state["sky_summary_highlight_primary_id"] = selected_primary_id

    list_col_index = int(display.columns.get_loc("List"))

    if (
        selection_changed
        and selected_primary_id
        and selected_column_index is not None
        and selected_column_index != list_col_index
    ):
        current_selected_id = str(st.session_state.get("selected_id") or "").strip()
        if selected_primary_id != current_selected_id:
            st.session_state["selected_id"] = selected_primary_id
            st.session_state[TARGET_DETAIL_MODAL_OPEN_REQUEST_KEY] = True
            st.rerun()

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
            was_in_list = bool(selected_row.get("is_in_list", False))
            if primary_id:
                if toggle_target_in_list(prefs, preview_list_id, primary_id):
                    selected_detail_id = str(st.session_state.get("selected_id") or "").strip()
                    if was_in_list and selected_detail_id and selected_detail_id == primary_id:
                        st.session_state["selected_id"] = ""
                        highlighted_id = str(st.session_state.get("sky_summary_highlight_primary_id", "")).strip()
                        if highlighted_id == primary_id:
                            st.session_state["sky_summary_highlight_primary_id"] = ""
                    st.session_state["sky_summary_list_action_token"] = action_token
                    persist_and_rerun(prefs)
                st.session_state["sky_summary_list_action_token"] = action_token
    else:
        st.session_state["sky_summary_list_action_token"] = ""

    if allow_list_membership_toggle:
        st.caption(
            f"Recommended targets choose the detail target. Use this table to highlight rows and update '{preview_list_name}'."
        )
    else:
        st.caption(
            f"Recommended targets choose the detail target. '{preview_list_name}' is auto-managed from recent selections."
        )


def normalize_object_type_group(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return "other"
    return text


def format_emissions_display(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return "-"
    cleaned = re.sub(r"[\[\]]", "", text).strip()
    return cleaned or "-"


def format_apparent_size_display(major_arcmin: Any, minor_arcmin: Any) -> str:
    def _coerce(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric <= 0.0:
            return None
        return numeric

    def _fmt(value: float) -> str:
        return f"{value:.6g}"

    major_value = _coerce(major_arcmin)
    minor_value = _coerce(minor_arcmin)
    if major_value is None and minor_value is None:
        return "-"

    show_degrees = (
        (major_value is not None and major_value >= 60.0)
        or (minor_value is not None and minor_value >= 60.0)
    )
    if show_degrees:
        major_deg = (major_value / 60.0) if major_value is not None else None
        minor_deg = (minor_value / 60.0) if minor_value is not None else None
        if major_deg is not None and minor_deg is not None:
            return f"{_fmt(major_deg)} x {_fmt(minor_deg)} deg"
        if major_deg is not None:
            return f"{_fmt(major_deg)} deg"
        return f"{_fmt(float(minor_deg))} deg"

    if major_value is not None and minor_value is not None:
        return f"{_fmt(major_value)} x {_fmt(minor_value)} arcmin"
    if major_value is not None:
        return f"{_fmt(major_value)} arcmin"
    return f"{_fmt(float(minor_value))} arcmin"


def apparent_size_sort_key_arcmin(major_arcmin: Any, minor_arcmin: Any) -> float:
    def _coerce(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric <= 0.0:
            return None
        return numeric

    major_value = _coerce(major_arcmin)
    minor_value = _coerce(minor_arcmin)
    if major_value is None and minor_value is None:
        return -1.0
    if major_value is None:
        major_value = minor_value
    if minor_value is None:
        minor_value = major_value
    if major_value is None or minor_value is None:
        return -1.0
    return float(major_value * minor_value)


def format_description_preview(value: Any, max_chars: int = 100) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return "-"
    collapsed = re.sub(r"\s+", " ", text)
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[:max_chars].rstrip()}..."


def build_full_dark_hour_starts(window_start: datetime, window_end: datetime) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(window_start)
    end_ts = pd.Timestamp(window_end)
    if end_ts <= start_ts:
        return []

    first_full_hour_start = start_ts.ceil("h")
    last_full_hour_start = (end_ts - pd.Timedelta(hours=1)).floor("h")
    if last_full_hour_start < first_full_hour_start:
        return []

    return [
        pd.Timestamp(hour_start)
        for hour_start in pd.date_range(
            start=first_full_hour_start,
            end=last_full_hour_start,
            freq="1h",
        )
    ]


def format_hour_window_label(hour_start: pd.Timestamp | datetime, use_12_hour: bool) -> str:
    start_ts = pd.Timestamp(hour_start)
    end_ts = start_ts + pd.Timedelta(hours=1)

    if use_12_hour:
        start_label = normalize_12_hour_label(start_ts.strftime("%I%p"))
        end_label = normalize_12_hour_label(end_ts.strftime("%I%p"))
        if (
            len(start_label) >= 2
            and len(end_label) >= 2
            and start_label[-2:] in {"am", "pm"}
            and end_label[-2:] in {"am", "pm"}
            and start_label[-2:] == end_label[-2:]
        ):
            return f"{start_label[:-2]}-{end_label}"
        return f"{start_label}-{end_label}"

    return f"{start_ts.strftime('%H')}-{end_ts.strftime('%H')}"


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


def render_target_recommendations(
    catalog: pd.DataFrame,
    prefs: dict[str, Any],
    *,
    active_preview_list_ids: list[str],
    window_start: datetime,
    window_end: datetime,
    tzinfo: ZoneInfo,
    use_12_hour: bool,
    weather_forecast_day_offset: int = 0,
) -> None:
    del active_preview_list_ids
    recommendation_night_title = format_recommendation_night_title(weather_forecast_day_offset, window_start)
    st.markdown(f"### Recommended Targets for {recommendation_night_title}")

    if catalog.empty:
        st.info("Catalog is empty.")
        return

    def _parse_emission_band_set(value: Any) -> set[str]:
        return parse_emission_band_set(value)

    def _filter_match_tier(target_bands: set[str], reference_bands: set[str]) -> int:
        if not reference_bands:
            return 0
        overlap_count = len(target_bands & reference_bands)
        if overlap_count <= 0:
            return 0
        if overlap_count >= len(reference_bands):
            return 2
        return 1

    def _safe_positive_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric <= 0.0:
            return None
        return float(numeric)

    location = prefs["location"]
    location_lat = float(location["lat"])
    location_lon = float(location["lon"])
    obstructions = prefs["obstructions"]

    equipment_context = build_owned_equipment_context(prefs)
    active_equipment = sync_active_equipment_settings(prefs, equipment_context)
    if bool(active_equipment.get("changed", False)):
        st.session_state["prefs"] = prefs
        save_preferences(prefs)

    telescope_lookup = dict(active_equipment.get("telescope_lookup", {}))
    filter_lookup = dict(active_equipment.get("filter_lookup", {}))
    owned_telescopes = list(active_equipment.get("owned_telescopes", []))
    owned_filters = list(active_equipment.get("owned_filters", []))

    active_telescope_id = str(active_equipment.get("active_telescope_id", "")).strip()
    active_telescope = active_equipment.get("active_telescope")
    if not isinstance(active_telescope, dict):
        active_telescope = telescope_lookup.get(active_telescope_id) if active_telescope_id else None

    active_filter_id = str(active_equipment.get("active_filter_id", "__none__")).strip() or "__none__"
    active_filter = active_equipment.get("active_filter")
    if not isinstance(active_filter, dict):
        active_filter = filter_lookup.get(active_filter_id) if active_filter_id != "__none__" else None

    active_mount_choice = _normalize_mount_choice(
        active_equipment.get("active_mount_choice", "altaz"),
        default_choice="altaz",
    )

    hour_starts = build_full_dark_hour_starts(window_start, window_end)
    hour_options = [hour_start.isoformat() for hour_start in hour_starts]
    hour_labels = {
        option: format_hour_window_label(pd.Timestamp(option), use_12_hour=use_12_hour)
        for option in hour_options
    }
    hour_option_to_key = {option: normalize_hour_key(option) for option in hour_options}

    catalog_fingerprint = _catalog_cache_fingerprint(CATALOG_CACHE_PATH.expanduser().resolve())

    weather_rows = fetch_hourly_weather(
        lat=location_lat,
        lon=location_lon,
        tz_name=tzinfo.key,
        start_local_iso=pd.Timestamp(window_start).isoformat(),
        end_local_iso=pd.Timestamp(window_end).isoformat(),
        hourly_fields=EXTENDED_FORECAST_HOURLY_FIELDS,
    )
    weather_by_hour: dict[str, dict[str, Any]] = {}
    for weather_row in weather_rows:
        hour_key = normalize_hour_key(weather_row.get("time_iso"))
        if not hour_key:
            continue
        weather_by_hour[hour_key] = weather_row

    def _metric_or_inf(weather_row: dict[str, Any] | None, field_name: str) -> float:
        if not isinstance(weather_row, dict):
            return float("inf")
        try:
            value = float(weather_row.get(field_name))
        except (TypeError, ValueError):
            return float("inf")
        if not np.isfinite(value):
            return float("inf")
        return float(value)

    best_visible_hour_option: str | None = None
    if hour_options:
        ranked_hours: list[tuple[float, float, float, int, str]] = []
        for option_idx, hour_option in enumerate(hour_options):
            hour_key = hour_option_to_key.get(hour_option, "")
            hour_weather = weather_by_hour.get(hour_key)
            cloud_cover_rank = _metric_or_inf(hour_weather, "cloud_cover")
            wind_rank = _metric_or_inf(hour_weather, "wind_gusts_10m")
            if not np.isfinite(wind_rank):
                wind_rank = _metric_or_inf(hour_weather, "wind_speed_10m")
            humidity_rank = _metric_or_inf(hour_weather, "relative_humidity_2m")
            ranked_hours.append((cloud_cover_rank, wind_rank, humidity_rank, option_idx, hour_option))
        best_visible_hour_option = min(ranked_hours)[-1]

    groups_series = catalog["object_type_group"].map(normalize_object_type_group)
    group_options = [str(group).strip() for group in groups_series.value_counts().index.tolist() if str(group).strip()]
    if not group_options:
        group_options = ["other"]

    visible_hour_key = "recommended_targets_visible_hours"
    object_type_key = "recommended_targets_object_types"
    keyword_key = "recommended_targets_keyword"
    include_telescope_key = "recommended_targets_include_telescope_details"
    include_filter_key = "recommended_targets_include_filter_criteria"
    include_mount_key = "recommended_targets_include_mount_adaptation"
    min_size_enabled_key = "recommended_targets_min_size_enabled"
    min_size_value_key = "recommended_targets_min_size_pct"
    page_size_key = "recommended_targets_page_size"
    page_number_key = "recommended_targets_page_number"
    sort_field_key = "recommended_targets_sort_field"
    sort_direction_key = "recommended_targets_sort_direction"
    sort_signature_key = "recommended_targets_sort_signature"
    signature_key = "recommended_targets_criteria_signature"
    selection_token_key = "recommended_targets_selection_token"
    query_cache_key = "recommended_targets_query_cache"
    table_instance_key = "recommended_targets_table_instance"

    if visible_hour_key not in st.session_state:
        st.session_state[visible_hour_key] = [best_visible_hour_option] if best_visible_hour_option else []

    raw_visible_hours = st.session_state.get(visible_hour_key, [])
    if isinstance(raw_visible_hours, str):
        normalized_visible_hours = [raw_visible_hours.strip()] if raw_visible_hours.strip() else []
    elif isinstance(raw_visible_hours, (list, tuple, set)):
        normalized_visible_hours = [str(item).strip() for item in raw_visible_hours if str(item).strip()]
    else:
        normalized_visible_hours = []
    had_raw_visible_hours = len(normalized_visible_hours) > 0
    normalized_visible_hours = [item for item in normalized_visible_hours if item in hour_options]
    if not normalized_visible_hours and had_raw_visible_hours and best_visible_hour_option:
        normalized_visible_hours = [best_visible_hour_option]
    st.session_state[visible_hour_key] = normalized_visible_hours

    raw_groups = st.session_state.get(object_type_key, [])
    if isinstance(raw_groups, str):
        normalized_groups = [raw_groups.strip()] if raw_groups.strip() else []
    elif isinstance(raw_groups, (list, tuple, set)):
        normalized_groups = [str(item).strip() for item in raw_groups if str(item).strip()]
    else:
        normalized_groups = []
    normalized_groups = [group for group in normalized_groups if group in group_options]
    st.session_state[object_type_key] = normalized_groups

    if not isinstance(st.session_state.get(keyword_key), str):
        st.session_state[keyword_key] = ""

    if include_telescope_key not in st.session_state:
        st.session_state[include_telescope_key] = active_telescope is not None
    if include_filter_key not in st.session_state:
        st.session_state[include_filter_key] = active_filter is not None
    if include_mount_key not in st.session_state:
        st.session_state[include_mount_key] = True
    if min_size_enabled_key not in st.session_state:
        st.session_state[min_size_enabled_key] = False
    if min_size_value_key not in st.session_state:
        st.session_state[min_size_value_key] = 0
    if page_size_key not in st.session_state:
        st.session_state[page_size_key] = 20
    elif int(st.session_state.get(page_size_key, 20)) not in {10, 20, 100}:
        st.session_state[page_size_key] = 20
    if not isinstance(st.session_state.get(page_number_key), int):
        st.session_state[page_number_key] = 1
    if not isinstance(st.session_state.get(table_instance_key), int):
        st.session_state[table_instance_key] = 0
    if str(st.session_state.get(sort_direction_key, "Descending")).strip() not in {"Descending", "Ascending"}:
        st.session_state[sort_direction_key] = "Descending"

    criteria_col_1, criteria_col_2, criteria_col_3 = st.columns([3, 3, 3], gap="small")
    search_notes_placeholder: Any | None = None
    sort_controls_placeholder: Any | None = None
    with criteria_col_1:
        keyword_query = st.text_input(
            "Keyword",
            key=keyword_key,
            placeholder="id, name, alias, description, emissions",
        )
        st.caption("Optional. Leave blank to search all targets.")
        search_clicked = st.button(
            "Search",
            key="recommended_targets_search_button",
            type="primary",
            use_container_width=True,
        )
        if search_clicked:
            st.session_state["selected_id"] = ""
            st.session_state[selection_token_key] = ""
            st.session_state.pop(TARGET_DETAIL_MODAL_OPEN_REQUEST_KEY, None)
            st.session_state[table_instance_key] = int(st.session_state.get(table_instance_key, 0)) + 1
        _ = search_clicked
        st.caption("Adjust criteria and click Search to refresh recommendations.")
        sort_controls_placeholder = st.empty()
        search_notes_placeholder = st.empty()

    with criteria_col_2:
        selected_visible_hours = st.multiselect(
            "Visible Hour",
            options=hour_options,
            default=normalized_visible_hours,
            key=visible_hour_key,
            format_func=lambda option: hour_labels.get(option, option),
            help=(
                "Defaults to the best hour (lowest cloud cover, then wind, then relative humidity, then earliest). "
                "If empty, all hours in the selected night are considered."
            ),
        )
        selected_object_types = st.multiselect(
            "Object Type",
            options=group_options,
            default=normalized_groups,
            key=object_type_key,
            help="Optional. If empty, any object type can be returned.",
        )

    selected_telescope: dict[str, Any] | None = None
    include_telescope_details = False
    include_filter_criteria = False
    include_mount_adaptation = False
    use_minimum_size = False
    minimum_size_pct: float | None = None
    telescope_fov_maj: float | None = None
    telescope_fov_min: float | None = None
    telescope_fov_area: float | None = None

    with criteria_col_3:
        if isinstance(active_telescope, dict):
            active_telescope_name = str(active_telescope.get("name", "Selected telescope")).strip() or "Selected telescope"
            include_telescope_details = st.checkbox(
                "Include telescope details in recommendations",
                key=include_telescope_key,
                help=f"Equipped telescope: {active_telescope_name}",
            )

        if owned_filters:
            active_filter_name = (
                str(active_filter.get("name", active_filter_id)).strip()
                if isinstance(active_filter, dict)
                else "None"
            )
            include_filter_criteria = st.checkbox(
                "Include filter choice as criteria for recommendations",
                key=include_filter_key,
                help=f"Equipped filter: {active_filter_name or 'None'}",
            )

        include_mount_adaptation = st.checkbox(
            "Adapt visibility calculations to selected mount",
            key=include_mount_key,
            help=f"Active mount choice: {mount_choice_label(active_mount_choice)}",
        )
        if include_mount_adaptation:
            mount_note_text = ""
            if active_mount_choice == "eq":
                mount_note_text = "increased likelihood of star trails when target is below 30 degrees"
            elif active_mount_choice == "altaz":
                mount_note_text = "increased likelihood of field rotation artifacts above 80 degrees"
            if mount_note_text:
                st.caption(mount_note_text)
        if include_telescope_details and isinstance(active_telescope, dict):
            selected_telescope = active_telescope

        telescope_fov_maj = _safe_positive_float(selected_telescope.get("fov_maj_deg")) if selected_telescope else None
        telescope_fov_min = _safe_positive_float(selected_telescope.get("fov_min_deg")) if selected_telescope else None
        telescope_fov_area = (
            float(telescope_fov_maj * telescope_fov_min)
            if telescope_fov_maj is not None and telescope_fov_min is not None
            else None
        )

        if selected_telescope is not None and telescope_fov_area is not None and telescope_fov_area > 0.0:
            use_minimum_size = st.checkbox("Enable Minimum Size", key=min_size_enabled_key)
            slider_disabled = not use_minimum_size
            min_size_slider_col, _ = st.columns([1, 1], gap="small")
            with min_size_slider_col:
                min_size_value = st.slider(
                    "Minimum Size (% of FOV)",
                    min_value=0,
                    max_value=250,
                    value=int(st.session_state.get(min_size_value_key, 0)),
                    step=1,
                    key=min_size_value_key,
                    disabled=slider_disabled,
                )
                st.markdown(
                    """
                    <div style="display:flex; justify-content:space-between; margin-top:0.2rem; color:#6b7280; font-size:0.72rem;">
                      <span style="text-align:center;">|<br/>0.5x</span>
                      <span style="text-align:center;">|<br/>1x</span>
                      <span style="text-align:center;">|<br/>1.5x</span>
                      <span style="text-align:center;">|<br/>&gt;2x</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if use_minimum_size:
                minimum_size_pct = float(min_size_value)
        else:
            st.caption("Minimum Size filter requires a selected telescope with FOV dimensions.")

    mount_mode = active_mount_choice if include_mount_adaptation else "none"
    selected_filter_item = active_filter if (include_filter_criteria and isinstance(active_filter, dict)) else None
    selected_filter_bands = _parse_emission_band_set(selected_filter_item.get("emission_bands", [])) if selected_filter_item else set()
    filter_reference_band_sets: list[set[str]] = []

    if search_notes_placeholder is not None:
        with search_notes_placeholder.container():
            if include_filter_criteria and selected_filter_item is not None:
                filter_name = str(selected_filter_item.get("name", "selected filter")).strip() or "selected filter"
                st.caption(f"Camera filter applied: {filter_name} (hard-filter: any emission-band overlap required).")
            elif include_filter_criteria and active_filter_id == "__none__":
                st.caption("Filter criteria enabled, but Camera Filter is set to None.")

    sort_options: list[tuple[str, str]] = [
        ("ranking", "Recommended"),
        ("target_name", "Target Name"),
        ("visible_minutes", "Duration of visibility"),
        ("object_type", "Object Type"),
        ("emissions", "Emissions"),
        ("apparent_size_sort_arcmin", "Apparent size"),
        ("peak_time_local", "Peak"),
        ("peak_altitude", "Altitude at Peak"),
        ("peak_direction", "Direction"),
    ]
    if selected_telescope is not None:
        sort_options.insert(6, ("framing_percent", "Framing"))
    sort_option_labels = {option_value: option_label for option_value, option_label in sort_options}
    sort_option_values = [option_value for option_value, _ in sort_options]
    if str(st.session_state.get(sort_field_key, "ranking")).strip() not in sort_option_labels:
        st.session_state[sort_field_key] = "ranking"

    if sort_controls_placeholder is not None:
        with sort_controls_placeholder.container():
            sort_col, sort_order_col = st.columns([2, 1], gap="small")
            sort_field = sort_col.selectbox(
                "Sort results by",
                options=sort_option_values,
                key=sort_field_key,
                format_func=lambda option_value: sort_option_labels.get(option_value, option_value),
            )
            sort_direction = sort_order_col.segmented_control(
                "Order",
                options=["Descending", "Ascending"],
                key=sort_direction_key,
            )
    else:
        sort_field = str(st.session_state.get(sort_field_key, "ranking"))
        sort_direction = str(st.session_state.get(sort_direction_key, "Descending"))

    sort_direction = str(sort_direction or st.session_state.get(sort_direction_key, "Descending")).strip()
    if sort_direction not in {"Descending", "Ascending"}:
        sort_direction = "Descending"

    sort_signature = f"{sort_field}|{sort_direction}"
    if str(st.session_state.get(sort_signature_key, "")) != sort_signature:
        st.session_state[sort_signature_key] = sort_signature
        st.session_state[page_number_key] = 1

    query_started_at = perf_counter()
    query_progress_placeholder = st.empty()
    query_progress = query_progress_placeholder.progress(
        1,
        text="Searching recommendations: preparing query...",
    )

    def update_query_progress(value: int, text: str) -> None:
        clamped_value = max(0, min(100, int(value)))
        query_progress.progress(clamped_value, text=text)

    def clear_query_progress() -> None:
        query_progress_placeholder.empty()

    selected_visible_hour_keys = {
        key
        for option in selected_visible_hours
        for key in [hour_option_to_key.get(option)]
        if key
    }

    criteria_signature = json.dumps(
        {
            "lat": round(location_lat, 6),
            "lon": round(location_lon, 6),
            "visible_hours": sorted(selected_visible_hour_keys),
            "object_types": sorted(selected_object_types),
            "keyword": str(keyword_query).strip().lower(),
            "include_telescope": bool(include_telescope_details),
            "include_filter": bool(include_filter_criteria),
            "include_mount": bool(include_mount_adaptation),
            "filter_id": active_filter_id if include_filter_criteria else "__disabled__",
            "mount_mode": mount_mode,
            "telescope_id": str(selected_telescope.get("id", "")) if isinstance(selected_telescope, dict) else "",
            "min_size_enabled": bool(use_minimum_size),
            "min_size_pct": minimum_size_pct if minimum_size_pct is not None else "",
            "catalog_mtime_ns": int(catalog_fingerprint[0]),
            "catalog_size_bytes": int(catalog_fingerprint[1]),
            "obstructions": {direction: float(obstructions.get(direction, 20.0)) for direction in WIND16},
            "cloud_cover_threshold": RECOMMENDATION_CLOUD_COVER_THRESHOLD,
            "sample_minutes": RECOMMENDATION_CACHE_SAMPLE_MINUTES,
            "result_limit_mode": "unlimited",
            "visibility_fallback_mode": "issue93_v1",
            "window_start": pd.Timestamp(window_start).isoformat(),
            "window_end": pd.Timestamp(window_end).isoformat(),
        },
        sort_keys=True,
    )
    if str(st.session_state.get(signature_key, "")) != criteria_signature:
        st.session_state[signature_key] = criteria_signature
        st.session_state[page_number_key] = 1
        st.session_state[selection_token_key] = ""

    query_cache_store = st.session_state.get(query_cache_key)
    if not isinstance(query_cache_store, dict):
        query_cache_store = {}
    query_cache_payload = query_cache_store.get(criteria_signature)
    recommended = pd.DataFrame()
    total_results_uncapped = 0
    empty_query_message: str | None = None

    if isinstance(query_cache_payload, dict):
        trace_cache_event(f"Session recommendation query cache hit ({criteria_signature[:12]}...)")
        cached_status = str(query_cache_payload.get("status", "ok")).strip().lower()
        if cached_status == "empty":
            empty_query_message = str(query_cache_payload.get("message") or "No targets match the current criteria.")
        else:
            cached_recommended = query_cache_payload.get("recommended")
            if isinstance(cached_recommended, pd.DataFrame):
                recommended = cached_recommended.copy()
            total_results_uncapped = int(query_cache_payload.get("total_results_uncapped", len(recommended)))
        update_query_progress(100, "Searching recommendations: ready (session cache).")
        clear_query_progress()
    else:
        trace_cache_event(f"Hydrating session recommendation query cache ({criteria_signature[:12]}...)")
        update_query_progress(12, "Searching recommendations: loading catalog features...")
        recommendation_feature_catalog = load_catalog_recommendation_features(CATALOG_CACHE_PATH)

        update_query_progress(20, "Searching recommendations: loading site/date altitude cache...")
        altaz_bundle = load_site_date_altaz_bundle(
            CATALOG_CACHE_PATH,
            lat=location_lat,
            lon=location_lon,
            window_start=window_start,
            window_end=window_end,
            sample_minutes=RECOMMENDATION_CACHE_SAMPLE_MINUTES,
        )
        sample_hour_keys_for_weather = tuple(
            str(hour_key or "").strip()
            for hour_key in altaz_bundle.get("sample_hour_keys", ())
        )
        update_query_progress(24, "Searching recommendations: loading weather masks...")
        weather_bundle = load_site_date_weather_mask_bundle(
            lat=location_lat,
            lon=location_lon,
            tz_name=tzinfo.key,
            window_start_iso=pd.Timestamp(window_start).isoformat(),
            window_end_iso=pd.Timestamp(window_end).isoformat(),
            sample_hour_keys=sample_hour_keys_for_weather,
            cloud_cover_threshold=RECOMMENDATION_CLOUD_COVER_THRESHOLD,
        )
        update_query_progress(28, "Searching recommendations: evaluating weather conditions...")

        cloud_cover_by_hour: dict[str, float] = {}
        cloud_cover_payload = weather_bundle.get("cloud_cover_by_hour", {})
        if isinstance(cloud_cover_payload, dict):
            for hour_key_raw, cloud_value_raw in cloud_cover_payload.items():
                hour_key = str(hour_key_raw or "").strip()
                if not hour_key:
                    continue
                try:
                    cloud_value = float(cloud_value_raw)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(cloud_value):
                    cloud_cover_by_hour[hour_key] = float(cloud_value)

        working_catalog = recommendation_feature_catalog
        cleaned_keyword = str(keyword_query).strip()
        if cleaned_keyword:
            working_catalog = search_catalog(recommendation_feature_catalog, cleaned_keyword)

        selected_object_type_values = [str(value).strip() for value in (selected_object_types or []) if str(value).strip()]
        valid_group_set = {
            normalize_object_type_group(value)
            for value in group_options
            if str(value).strip()
        }
        selected_group_set = {
            normalize_object_type_group(value)
            for value in selected_object_type_values
            if normalize_object_type_group(value) in valid_group_set
        }
        if selected_object_type_values:
            effective_group_set = selected_group_set if selected_group_set else set(valid_group_set)
        else:
            effective_group_set = set(valid_group_set)
        if effective_group_set:
            working_catalog = working_catalog[
                working_catalog["object_type_group_norm"].isin(effective_group_set)
            ]

        if working_catalog.empty:
            empty_query_message = "No targets match the current criteria."
        else:
            update_query_progress(28, "Searching recommendations: preparing coordinate candidates...")
            primary_id_to_col_raw = altaz_bundle.get("primary_id_to_col", {})
            primary_id_to_col = primary_id_to_col_raw if isinstance(primary_id_to_col_raw, dict) else {}
            candidate_row_positions: list[int] = []
            candidate_col_positions: list[int] = []
            working_primary_ids = working_catalog["primary_id"].fillna("").astype(str).tolist()
            for row_idx, primary_id in enumerate(working_primary_ids):
                column_index = primary_id_to_col.get(primary_id)
                if column_index is None:
                    continue
                try:
                    candidate_col_positions.append(int(column_index))
                    candidate_row_positions.append(int(row_idx))
                except (TypeError, ValueError):
                    continue

            if not candidate_row_positions:
                empty_query_message = "No targets with valid coordinates match the current criteria."
            else:
                candidates = working_catalog.iloc[candidate_row_positions].copy().reset_index(drop=True)

                altitude_matrix_full = np.asarray(altaz_bundle.get("altitude_matrix", np.empty((0, 0))), dtype=float)
                wind_index_matrix_full = np.asarray(altaz_bundle.get("wind_index_matrix", np.empty((0, 0))), dtype=np.uint8)
                sample_hour_keys = [
                    str(hour_key or "").strip()
                    for hour_key in altaz_bundle.get("sample_hour_keys", ())
                ]
                if altitude_matrix_full.ndim != 2 or altitude_matrix_full.shape[0] <= 0:
                    empty_query_message = "No valid time samples available for this night."
                else:
                    time_count = int(altitude_matrix_full.shape[0])
                    if len(sample_hour_keys) != time_count:
                        sample_hour_keys = sample_hour_keys[:time_count]
                        if len(sample_hour_keys) < time_count:
                            sample_hour_keys.extend([""] * (time_count - len(sample_hour_keys)))

                    cloud_ok_mask = np.asarray(
                        weather_bundle.get("cloud_ok_mask", ()),
                        dtype=bool,
                    )
                    if cloud_ok_mask.size != time_count:
                        cloud_ok_mask = np.array(
                            [
                                (hour_key in cloud_cover_by_hour)
                                and (float(cloud_cover_by_hour[hour_key]) < float(RECOMMENDATION_CLOUD_COVER_THRESHOLD))
                                for hour_key in sample_hour_keys
                            ],
                            dtype=bool,
                        )

                    if selected_visible_hour_keys:
                        selected_hours_mask = np.array(
                            [hour_key in selected_visible_hour_keys for hour_key in sample_hour_keys],
                            dtype=bool,
                        )
                    else:
                        selected_hours_mask = np.ones(time_count, dtype=bool)

                    candidate_col_indices = np.asarray(candidate_col_positions, dtype=int)
                    altitude_matrix = altitude_matrix_full[:, candidate_col_indices]
                    wind_index_matrix = wind_index_matrix_full[:, candidate_col_indices]

                    obstruction_thresholds = np.array(
                        [float(obstructions.get(direction, 20.0)) for direction in WIND16],
                        dtype=float,
                    )
                    min_required_matrix = obstruction_thresholds[wind_index_matrix]
                    visible_matrix = (altitude_matrix >= 0.0) & (altitude_matrix >= min_required_matrix)

                    if mount_mode == "eq":
                        mount_mask = altitude_matrix >= 30.0
                    elif mount_mode == "altaz":
                        mount_mask = altitude_matrix <= 80.0
                    else:
                        mount_mask = np.ones_like(altitude_matrix, dtype=bool)

                    qualified_matrix_full_night = (
                        visible_matrix
                        & mount_mask
                        & cloud_ok_mask[:, np.newaxis]
                    )
                    qualified_matrix_selected_hours = (
                        visible_matrix
                        & mount_mask
                        & cloud_ok_mask[:, np.newaxis]
                        & selected_hours_mask[:, np.newaxis]
                    )

                    sample_minutes = RECOMMENDATION_CACHE_SAMPLE_MINUTES
                    selected_visible_minutes = np.sum(qualified_matrix_selected_hours, axis=0).astype(int) * sample_minutes
                    full_night_visible_minutes = np.sum(qualified_matrix_full_night, axis=0).astype(int) * sample_minutes
                    has_duration = selected_visible_minutes > 0
                    update_query_progress(72, "Searching recommendations: evaluating filter and visibility matches...")

                    peak_altitude_all = np.asarray(altaz_bundle.get("peak_altitude", np.empty((0,))), dtype=float)
                    peak_time_local_all = np.asarray(altaz_bundle.get("peak_time_local_iso", ()), dtype=object)
                    peak_direction_all = np.asarray(altaz_bundle.get("peak_direction", ()), dtype=object)
                    max_col_index = int(candidate_col_indices.max()) if candidate_col_indices.size else -1
                    if (
                        peak_altitude_all.ndim == 1
                        and peak_altitude_all.size > max_col_index
                        and peak_time_local_all.size > max_col_index
                        and peak_direction_all.size > max_col_index
                    ):
                        peak_altitude = peak_altitude_all[candidate_col_indices]
                        peak_time_local = peak_time_local_all[candidate_col_indices]
                        peak_direction = peak_direction_all[candidate_col_indices]
                    else:
                        peak_idx_by_target = np.argmax(altitude_matrix, axis=0)
                        peak_altitude = altitude_matrix[peak_idx_by_target, np.arange(len(candidate_col_indices))]
                        sample_times_local = np.asarray(altaz_bundle.get("sample_times_local_iso", ()), dtype=object)
                        peak_time_local = np.array(
                            [sample_times_local[int(index)] for index in peak_idx_by_target],
                            dtype=object,
                        )
                        peak_direction = np.array(
                            [
                                WIND16[int(wind_index_matrix[int(index), target_idx])]
                                for target_idx, index in enumerate(peak_idx_by_target)
                            ],
                            dtype=object,
                        )

                    emission_token_values = candidates.get("emission_band_tokens")
                    if emission_token_values is None:
                        target_emission_sets = [
                            _parse_emission_band_set(value)
                            for value in candidates.get("emission_lines", pd.Series(dtype=object)).tolist()
                        ]
                    else:
                        target_emission_sets = [set(value) for value in emission_token_values.tolist()]

                    if selected_filter_item is not None and selected_filter_bands:
                        filter_match_tier = np.array(
                            [_filter_match_tier(target_bands, selected_filter_bands) for target_bands in target_emission_sets],
                            dtype=int,
                        )
                        filter_visibility_mask = filter_match_tier > 0
                    else:
                        if filter_reference_band_sets:
                            filter_match_tier = np.array(
                                [
                                    max(_filter_match_tier(target_bands, reference_bands) for reference_bands in filter_reference_band_sets)
                                    for target_bands in target_emission_sets
                                ],
                                dtype=int,
                            )
                        else:
                            filter_match_tier = np.zeros(len(candidate_col_indices), dtype=int)
                        filter_visibility_mask = np.ones(len(candidate_col_indices), dtype=bool)

                    above_horizon_matrix = altitude_matrix >= 0.0
                    selected_window_matrix = selected_hours_mask[:, np.newaxis]
                    above_horizon_selected = np.any(above_horizon_matrix & selected_window_matrix, axis=0)
                    obstructed_selected = np.any(
                        above_horizon_matrix
                        & (~visible_matrix)
                        & selected_window_matrix,
                        axis=0,
                    )
                    cloud_blocked_selected = np.any(
                        visible_matrix
                        & mount_mask
                        & (~cloud_ok_mask[:, np.newaxis])
                        & selected_window_matrix,
                        axis=0,
                    )
                    visibility_reason = np.full(len(candidate_col_indices), "obstructed", dtype=object)
                    visibility_reason[~above_horizon_selected] = "below horizon"
                    visibility_reason[
                        above_horizon_selected
                        & (~obstructed_selected)
                        & cloud_blocked_selected
                    ] = "cloud cover"

                    eligible_visible_mask = has_duration & filter_visibility_mask
                    no_visible_targets = not bool(np.any(eligible_visible_mask))
                    keyword_search_active = bool(str(cleaned_keyword).strip())
                    include_fallback_mask = np.zeros(len(candidate_col_indices), dtype=bool)
                    if keyword_search_active:
                        include_fallback_mask = (~eligible_visible_mask) & filter_visibility_mask
                    elif no_visible_targets:
                        include_fallback_mask = (
                            (~eligible_visible_mask)
                            & filter_visibility_mask
                            & (visibility_reason != "below horizon")
                        )

                    eligible_mask = eligible_visible_mask | include_fallback_mask
                    if not np.any(filter_visibility_mask):
                        empty_query_message = "No targets match the selected camera filter."
                    elif not np.any(eligible_mask):
                        if no_visible_targets:
                            empty_query_message = "No targets above the horizon meet the current visibility/weather/mount constraints."
                        else:
                            empty_query_message = "No targets meet the current visibility/weather/mount constraints."
                    else:
                        eligible_indices = np.where(eligible_mask)[0]
                        recommended = candidates.iloc[eligible_indices].copy()
                        recommended["visible_minutes"] = full_night_visible_minutes[eligible_indices]
                        recommended["selected_visible_minutes"] = selected_visible_minutes[eligible_indices]
                        recommended["filter_match_tier"] = filter_match_tier[eligible_indices]
                        recommended["visibility_reason"] = visibility_reason[eligible_indices]
                        recommended["peak_altitude"] = np.round(peak_altitude[eligible_indices], 1)
                        recommended["peak_time_local"] = peak_time_local[eligible_indices]
                        recommended["peak_direction"] = peak_direction[eligible_indices]
                        recommended["object_type"] = recommended["object_type"].fillna("").astype(str).str.strip()
                        recommended["object_type_group"] = recommended["object_type_group"].map(normalize_object_type_group)
                        recommended["emissions"] = recommended["emission_lines"].apply(format_emissions_display)
                        if "apparent_size" not in recommended.columns:
                            recommended["apparent_size"] = recommended.apply(
                                lambda row: format_apparent_size_display(
                                    row.get("ang_size_maj_arcmin"),
                                    row.get("ang_size_min_arcmin"),
                                ),
                                axis=1,
                            )
                        if "apparent_size_sort_arcmin" not in recommended.columns:
                            recommended["apparent_size_sort_arcmin"] = recommended.apply(
                                lambda row: apparent_size_sort_key_arcmin(
                                    row.get("ang_size_maj_arcmin"),
                                    row.get("ang_size_min_arcmin"),
                                ),
                                axis=1,
                            )
                        if "target_name" not in recommended.columns:
                            primary_ids = recommended["primary_id"].astype(str)
                            common_names = recommended["common_name"].fillna("").astype(str).str.strip()
                            recommended["target_name"] = np.where(
                                common_names != "",
                                primary_ids + " - " + common_names,
                                primary_ids,
                            )
                        recommended["visibility_duration"] = recommended["visible_minutes"].map(
                            lambda minutes: f"{int(minutes) // 60:02d}:{int(minutes) % 60:02d}"
                        )
                        non_visible_reason_mask = recommended["selected_visible_minutes"] <= 0
                        recommended.loc[non_visible_reason_mask, "visibility_duration"] = (
                            recommended.loc[non_visible_reason_mask, "visibility_reason"].astype(str)
                        )
                        recommended["visibility_bin_15"] = np.floor_divide(
                            recommended["selected_visible_minutes"],
                            15,
                        ).astype(int)
                        recommended["peak_alt_band_10"] = np.floor(
                            np.clip(recommended["peak_altitude"], 0.0, 90.0) / 10.0
                        ).astype(int)

                        recommended["framing_percent"] = np.nan
                        if selected_telescope is not None and telescope_fov_area is not None and telescope_fov_area > 0.0:
                            target_maj_deg = pd.to_numeric(recommended["ang_size_maj_arcmin"], errors="coerce") / 60.0
                            target_min_deg = pd.to_numeric(recommended["ang_size_min_arcmin"], errors="coerce") / 60.0
                            target_maj_deg = target_maj_deg.where(target_maj_deg > 0.0)
                            target_min_deg = target_min_deg.where(target_min_deg > 0.0)
                            target_maj_deg = target_maj_deg.fillna(target_min_deg)
                            target_min_deg = target_min_deg.fillna(target_maj_deg)
                            target_area_deg2 = target_maj_deg * target_min_deg
                            framing_percent = (target_area_deg2 / float(telescope_fov_area)) * 100.0
                            recommended["framing_percent"] = framing_percent

                            if minimum_size_pct is not None:
                                recommended = recommended[
                                    recommended["framing_percent"].apply(
                                        lambda value: value is not None and not pd.isna(value) and np.isfinite(float(value))
                                        and float(value) >= float(minimum_size_pct)
                                    )
                                ]

                        if recommended.empty:
                            empty_query_message = "No targets remain after applying size/framing criteria."
                        else:
                            if selected_telescope is not None:
                                recommended["sort_size_metric"] = recommended["framing_percent"].apply(
                                    lambda value: (
                                        float(value)
                                        if value is not None and not pd.isna(value) and np.isfinite(float(value))
                                        else -1.0
                                    )
                                )
                            else:
                                recommended["sort_size_metric"] = recommended["apparent_size_sort_arcmin"].apply(
                                    lambda value: (
                                        float(value)
                                        if value is not None and not pd.isna(value) and np.isfinite(float(value))
                                        else -1.0
                                    )
                                )

                            recommended = recommended.sort_values(
                                by=[
                                    "filter_match_tier",
                                    "visibility_bin_15",
                                    "peak_alt_band_10",
                                    "sort_size_metric",
                                    "selected_visible_minutes",
                                    "peak_altitude",
                                    "primary_id",
                                ],
                                ascending=[False, False, False, False, False, False, True],
                            ).reset_index(drop=True)
                            total_results_uncapped = int(len(recommended))

        update_query_progress(88, "Searching recommendations: applying limits and sorting...")
        update_query_progress(100, "Searching recommendations: ready.")
        clear_query_progress()

        if empty_query_message:
            query_cache_store[criteria_signature] = {
                "status": "empty",
                "message": empty_query_message,
            }
        else:
            query_cache_store[criteria_signature] = {
                "status": "ok",
                "recommended": recommended.copy(),
                "total_results_uncapped": int(total_results_uncapped),
            }
        while len(query_cache_store) > RECOMMENDATION_QUERY_SESSION_CACHE_LIMIT:
            oldest_signature = next(iter(query_cache_store))
            del query_cache_store[oldest_signature]
        st.session_state[query_cache_key] = query_cache_store

    if empty_query_message:
        st.info(empty_query_message)
        return

    if total_results_uncapped <= 0:
        total_results_uncapped = int(len(recommended))

    if sort_field == "ranking":
        if sort_direction == "Ascending":
            recommended = recommended.iloc[::-1].reset_index(drop=True)
    else:
        sort_ascending = sort_direction == "Ascending"
        recommended = recommended.sort_values(
            by=[sort_field, "target_name", "primary_id"],
            ascending=[sort_ascending, True, True],
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)

    page_size = int(st.session_state.get(page_size_key, 20))
    if page_size not in {10, 20, 100}:
        page_size = 20
        st.session_state[page_size_key] = page_size
    total_results = int(len(recommended))
    query_elapsed_seconds = max(0.0, perf_counter() - query_started_at)
    query_elapsed_label = (
        f"{query_elapsed_seconds * 1000.0:.0f} ms"
        if query_elapsed_seconds < 1.0
        else f"{query_elapsed_seconds:.2f} s"
    )
    total_pages = max(1, int(np.ceil(float(total_results) / float(page_size))))
    current_page = int(st.session_state.get(page_number_key, 1))
    if current_page < 1 or current_page > total_pages:
        current_page = 1
        st.session_state[page_number_key] = current_page
    page_number = current_page
    results_meta_col, query_meta_col = st.columns([4, 2], gap="small")
    results_meta_col.caption(f"{total_results} targets | page {page_number}/{total_pages}")
    query_meta_col.caption(f"Query time: {query_elapsed_label}")

    start_index = (page_number - 1) * int(page_size)
    end_index = min(total_results, start_index + int(page_size))
    page_frame = recommended.iloc[start_index:end_index].copy().reset_index(drop=True)

    page_frame["Peak"] = page_frame["peak_time_local"].apply(
        lambda value: format_display_time(pd.Timestamp(value), use_12_hour=use_12_hour)
        if value is not None and not pd.isna(value)
        else "--"
    )
    page_frame["Altitude at Peak"] = page_frame["peak_altitude"].apply(
        lambda value: f"{float(value):.1f} deg" if value is not None and not pd.isna(value) else "--"
    )
    page_frame["Direction"] = page_frame["peak_direction"].fillna("--").astype(str)

    display_columns = [
        "target_name",
        "visibility_duration",
        "object_type",
        "emissions",
        "apparent_size",
    ]
    if selected_telescope is not None:
        display_columns.append("framing_percent")
    display_columns.extend(["Peak", "Altitude at Peak", "Direction"])

    rename_columns = {
        "target_name": "Target Name",
        "visibility_duration": "Duration of visibility",
        "object_type": "Object Type",
        "emissions": "Emissions",
        "apparent_size": "Apparent size",
        "framing_percent": "Framing",
    }
    display_table = page_frame[display_columns].rename(columns=rename_columns)

    column_config: dict[str, Any] = {
        "Target Name": st.column_config.TextColumn(width="large"),
        "Duration of visibility": st.column_config.TextColumn(width="small"),
        "Object Type": st.column_config.TextColumn(width="small"),
        "Emissions": st.column_config.TextColumn(width="small"),
        "Apparent size": st.column_config.TextColumn(width="small"),
        "Peak": st.column_config.TextColumn(width="small"),
        "Altitude at Peak": st.column_config.TextColumn(width="small"),
        "Direction": st.column_config.TextColumn(width="small"),
    }
    if "Framing" in display_table.columns:
        column_config["Framing"] = st.column_config.NumberColumn(width="small", format="%.0f%%")

    recommendation_event = st.dataframe(
        apply_dataframe_styler_theme(display_table.style),
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"recommended_targets_table_{int(st.session_state.get(table_instance_key, 0))}",
        column_config=column_config,
    )

    per_page_col, page_col = st.columns([1, 1], gap="small")
    per_page_col.selectbox(
        "Results per page",
        options=[10, 20, 100],
        key=page_size_key,
    )
    page_col.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=page_number,
        step=1,
        key=page_number_key,
    )

    selected_rows: list[int] = []
    if recommendation_event is not None:
        try:
            selected_rows = list(recommendation_event.selection.rows)
        except Exception:
            if isinstance(recommendation_event, dict):
                selection_payload = recommendation_event.get("selection", {})
                selected_rows = list(selection_payload.get("rows", []))

    if not selected_rows:
        st.session_state[selection_token_key] = ""
        return

    selected_index = None
    try:
        parsed_index = int(selected_rows[0])
        if 0 <= parsed_index < len(page_frame):
            selected_index = parsed_index
    except (TypeError, ValueError):
        selected_index = None

    if selected_index is None:
        st.session_state[selection_token_key] = ""
        return

    selected_primary_id = str(page_frame.iloc[selected_index].get("primary_id", "")).strip()
    if not selected_primary_id:
        return

    selection_token = f"{page_number}:{selected_index}:{selected_primary_id}"
    if str(st.session_state.get(selection_token_key, "")) == selection_token:
        return

    st.session_state[selection_token_key] = selection_token
    current_selected_id = str(st.session_state.get("selected_id") or "").strip()
    if selected_primary_id != current_selected_id:
        st.session_state["selected_id"] = selected_primary_id
        st.session_state[TARGET_DETAIL_MODAL_OPEN_REQUEST_KEY] = True
        st.rerun()


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


def _ideal_text_color_for_hex(value: str) -> str:
    try:
        red, green, blue = _hex_to_rgb(value)
    except Exception:
        return "#111827"
    yiq = ((red * 299) + (green * 587) + (blue * 114)) / 1000.0
    return "#111827" if yiq >= 155.0 else "#FFFFFF"


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


def cloud_cover_color_legend_html() -> str:
    # One swatch per configured interval in CLOUD_COVER_COLOR_STOPS.
    legend_stops: list[tuple[str, float]] = []
    for idx in range(max(0, len(CLOUD_COVER_COLOR_STOPS) - 1)):
        lower_value = float(CLOUD_COVER_COLOR_STOPS[idx][0])
        upper_value = float(CLOUD_COVER_COLOR_STOPS[idx + 1][0])
        mid_value = (lower_value + upper_value) / 2.0
        label = f"{int(round(lower_value))}-{int(round(upper_value))}%"
        legend_stops.append((label, mid_value))

    if not legend_stops:
        legend_stops = [("0-100%", 50.0)]

    swatches: list[str] = []
    for label, percent in legend_stops:
        color = _interpolate_cloud_cover_color(percent)
        swatches.append(
            "<span style='display:inline-flex; align-items:center; margin-right:0.4rem;'>"
            f"<span style='display:inline-block; width:0.58rem; height:0.58rem; border-radius:2px; "
            f"margin-right:0.18rem; border:1px solid rgba(17,24,39,0.28); background:{color};'></span>"
            f"{html.escape(label)}"
            "</span>"
        )

    return (
        "<div style='font-size:0.68rem; color:#6b7280; margin:0.08rem 0 0.12rem;'>"
        "☁️: "
        f"{''.join(swatches)}"
        "</div>"
    )


def night_time_clarity_scale_legend_html() -> str:
    swatches: list[str] = []
    total_boxes = 10
    for idx in range(total_boxes):
        clear_percent = 100.0 - (float(idx) * (100.0 / float(max(1, total_boxes - 1))))
        cloud_equivalent = max(0.0, min(100.0, 100.0 - clear_percent))
        color = _interpolate_cloud_cover_color(cloud_equivalent)
        swatches.append(
            "<span style='display:inline-block; width:0.58rem; height:0.58rem; border-radius:2px; "
            f"margin-right:0.12rem; border:1px solid rgba(17,24,39,0.28); background:{color};'></span>"
        )

    return (
        "<div style='font-size:0.88rem; color:#6b7280; margin:0.08rem 0 0.12rem;'>"
        "☁️ night time clarity scale: "
        "<span style='margin-right:0.2rem;'>Good</span>"
        f"{''.join(swatches)}"
        "<span style='margin-left:0.12rem;'>Bad</span>"
        "</div>"
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
        if metric_key == "dewpoint_spread":
            matrix_rows[metric_label] = [
                format_weather_matrix_value(
                    metric_key,
                    _dewpoint_spread_celsius(
                        by_hour[timestamp].get("temperature_2m"),
                        by_hour[timestamp].get("dew_point_2m"),
                    ),
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
                    f"RH: {float(by_hour[timestamp].get('relative_humidity_2m')):.0f}%"
                    if by_hour[timestamp].get("relative_humidity_2m") is not None
                    and not pd.isna(by_hour[timestamp].get("relative_humidity_2m"))
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

    if st_mui_table is not None:
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
            for column in frame.columns:
                raw_value = row.get(column)
                cell_text = str(raw_value).strip() if raw_value is not None and not pd.isna(raw_value) else ""

                if str(column) == "Element":
                    mui_frame.at[row_idx, ""] = _mui_cell_html(
                        cell_text,
                        ["font-weight: 600;", "white-space: nowrap;", "text-align: left;"],
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
                elif element_key == "visibility":
                    visibility_style = visibility_condition_cell_style(raw_value)
                    if visibility_style:
                        style_parts.append(visibility_style)

                tooltip_text = ""
                if aligned_tooltips is not None:
                    tooltip_text = str(aligned_tooltips.at[row_idx, column]).strip()

                indicator_html = ""
                if aligned_indicators is not None and element_key == "cloud cover":
                    indicator_html = str(aligned_indicators.at[row_idx, column]).strip()

                full_bleed = element_key in {"cloud cover", "visibility"}
                mui_frame.at[row_idx, column] = _mui_cell_html(
                    cell_text,
                    style_parts,
                    title_text=tooltip_text,
                    extra_html=indicator_html,
                    full_bleed=full_bleed,
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
"""
        st_mui_table(
            mui_frame,
            enablePagination=True,
            paginationSizes=[24],
            customCss=mui_custom_css,
            showHeaders=True,
            key="hourly_weather_mui_table",
            stickyHeader=False,
            showIndex=False,
            enable_sorting=False,
            return_clicked_cell=False,
            paperStyle={
                "width": "100%",
                "overflow": "visible",
                "paddingBottom": "0px",
                "border": "1px solid rgba(148, 163, 184, 0.35)",
            },
        )
        return

    header_cells = "".join(
        f'<th style="padding: 6px 8px; border-bottom: 1px solid #d1d5db; '
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
                '<td style="padding: 6px 8px; border-bottom: 1px solid #d1d5db; '
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
                "padding: 6px 8px; border-bottom: 1px solid #d1d5db; "
                "white-space: nowrap; text-align: center;"
            )
            if element_key == "cloud cover":
                cell_style += cloud_cover_cell_style(raw_value)
            elif element_key == "temperature":
                cell_style += temperature_cell_style(raw_value, temperature_unit=temperature_unit)
            elif element_key == "visibility":
                cell_style += visibility_condition_cell_style(raw_value)

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


def build_astronomy_forecast_summary(
    lat: float,
    lon: float,
    *,
    tz_name: str,
    temperature_unit: str,
    browser_locale: str | None = None,
    browser_month_day_pattern: str | None = None,
    nights: int = ASTRONOMY_FORECAST_NIGHTS,
) -> pd.DataFrame:
    night_count = max(1, int(nights))
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

        summary_rows.append(
            {
                "day_offset": day_offset,
                "Night": label,
                "Dark": dark_duration,
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
    if frame.empty:
        st.info("No 5-night forecast data available.")
        return

    ordered_columns = [
        "Night",
        "Clear",
        "Calm",
        "Crisp",
        "Dark",
        "Sunset",
        "Sunrise",
        "Avg. Temp",
    ]
    source_frame = frame.copy()
    if "day_offset" not in source_frame.columns:
        source_frame["day_offset"] = 0
    display_frame = source_frame.reindex(columns=ordered_columns, fill_value="").reset_index(drop=True)
    source_frame = source_frame.reset_index(drop=True)

    if st_mui_table is not None:
        center_columns = {"Clear", "Calm", "Crisp", "Dark"}

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
                max_offset=ASTRONOMY_FORECAST_NIGHTS - 1,
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
                if column_name == "Avg. Temp":
                    avg_style = temperature_cell_style(cell_text, temperature_unit=temperature_unit)
                    if avg_style:
                        style_parts.append(avg_style)
                mui_frame.at[int(row_idx), column_name] = _mui_cell_html(
                    cell_text,
                    style_parts,
                    full_bleed=(column_name == "Clear"),
                )

        mui_custom_css = """
.MuiTableCell-root {
  padding: 6px 8px !important;
}
.MuiTableCell-root:nth-child(2),
.MuiTableCell-root:nth-child(3),
.MuiTableCell-root:nth-child(4),
.MuiTableCell-root:nth-child(5) {
  text-align: center !important;
}
.MuiTableHead-root .MuiTableCell-root:nth-child(2),
.MuiTableHead-root .MuiTableCell-root:nth-child(3),
.MuiTableHead-root .MuiTableCell-root:nth-child(4),
.MuiTableHead-root .MuiTableCell-root:nth-child(5) {
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
        clicked_cell = st_mui_table(
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
            max_offset=ASTRONOMY_FORECAST_NIGHTS - 1,
        )
        selection_token = f"{selected_index}:{selected_offset}"
        last_selection_token = str(st.session_state.get("astronomy_forecast_last_selection_token", ""))
        if selection_token == last_selection_token:
            return

        st.session_state["astronomy_forecast_last_selection_token"] = selection_token
        if selected_offset != selected_day_offset:
            st.session_state[WEATHER_FORECAST_DAY_OFFSET_STATE_KEY] = selected_offset
            st.session_state[WEATHER_FORECAST_PERIOD_STATE_KEY] = (
                WEATHER_FORECAST_PERIOD_TONIGHT if selected_offset <= 0 else WEATHER_FORECAST_PERIOD_TOMORROW
            )
            st.rerun()
        return

    def _style_forecast_row(row: pd.Series) -> list[str]:
        row_idx = int(row.name)
        row_day_offset = normalize_weather_forecast_day_offset(
            source_frame.at[row_idx, "day_offset"],
            max_offset=ASTRONOMY_FORECAST_NIGHTS - 1,
        )
        row_is_selected = row_day_offset == selected_day_offset

        styles: list[str] = []
        for column in display_frame.columns:
            style_parts = ["white-space: nowrap;"]
            if row_is_selected:
                style_parts.append("background-color: rgba(37, 99, 235, 0.14);")
            if column == "Night" and row_is_selected:
                style_parts.append("font-weight: 700;")
            if column in {"Dark", "Clear", "Calm", "Crisp"}:
                style_parts.append("text-align: center !important;")
                style_parts.append("justify-content: center !important;")
            if column == "Clear":
                style_parts.append(clarity_percentage_cell_style(row.get(column)))
            if column == "Avg. Temp":
                style_parts.append(temperature_cell_style(row.get(column), temperature_unit=temperature_unit))
            styles.append(" ".join(part for part in style_parts if str(part).strip()))
        return styles

    base_styler = display_frame.style.apply(_style_forecast_row, axis=1)
    base_styler = base_styler.set_properties(
        subset=["Clear", "Calm", "Crisp", "Dark"],
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
            "Night": st.column_config.TextColumn(width="small"),
            "Clear": st.column_config.TextColumn(width="small"),
            "Calm": st.column_config.TextColumn(width="small"),
            "Crisp": st.column_config.TextColumn(width="small"),
            "Dark": st.column_config.TextColumn(width="small"),
            "Sunset": st.column_config.TextColumn(width="small"),
            "Sunrise": st.column_config.TextColumn(width="small"),
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
        max_offset=ASTRONOMY_FORECAST_NIGHTS - 1,
    )
    selection_token = f"{selected_index}:{selected_offset}"
    last_selection_token = str(st.session_state.get("astronomy_forecast_last_selection_token", ""))
    if selection_token == last_selection_token:
        return

    st.session_state["astronomy_forecast_last_selection_token"] = selection_token
    if selected_offset != selected_day_offset:
        st.session_state[WEATHER_FORECAST_DAY_OFFSET_STATE_KEY] = selected_offset
        st.session_state[WEATHER_FORECAST_PERIOD_STATE_KEY] = (
            WEATHER_FORECAST_PERIOD_TONIGHT if selected_offset <= 0 else WEATHER_FORECAST_PERIOD_TOMORROW
        )
        st.rerun()


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
) -> go.Figure:
    theme_colors = resolve_plot_theme_colors()
    fig = go.Figure()
    plotted_any = False
    plotted_times: list[pd.Timestamp] = []
    obstruction_ceiling = max(0.0, min(90.0, float(UNOBSTRUCTED_AREA_CONSTANT_OBSTRUCTION_ALT_DEG)))

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
            plotted_any = True

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
    overlay_tracks: list[dict[str, Any]] | None = None,
) -> go.Figure:
    theme_colors = resolve_plot_theme_colors()
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
                    "size": [0, PATH_ENDPOINT_MARKER_SIZE_PRIMARY + 1],
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
                            "size": [0, PATH_ENDPOINT_MARKER_SIZE_OVERLAY + 1],
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
    overlay_tracks: list[dict[str, Any]] | None = None,
) -> go.Figure:
    theme_colors = resolve_plot_theme_colors()
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
                    "size": [0, PATH_ENDPOINT_MARKER_SIZE_PRIMARY + 1],
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
                            "size": [0, PATH_ENDPOINT_MARKER_SIZE_OVERLAY + 1],
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
            "source": "search",
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

                resolved_label, kept_site_name = apply_resolved_location(
                    prefs,
                    {
                        "lat": lat,
                        "lon": lon,
                        "label": reverse_geocode_label(lat, lon),
                        "source": "browser",
                        "resolved_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                st.session_state["location_notice"] = (
                    f"Browser geolocation applied: {resolved_label}. Site name unchanged."
                    if kept_site_name
                    else f"Browser geolocation applied: {resolved_label}."
                )
                st.session_state["prefs"] = prefs
                save_preferences(prefs)
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


def resolve_selected_row(catalog: pd.DataFrame) -> pd.Series | None:
    selected_id = st.session_state.get("selected_id")
    selected = get_object_by_id(catalog, str(selected_id) if selected_id else "")
    return selected


def render_location_settings_section(prefs: dict[str, Any]) -> None:
    st.subheader("Location")
    st.caption("Set location manually or with browser geolocation first. IP-based location is used only as fallback.")
    sync_active_site_to_legacy_fields(prefs)
    active_site_id = get_active_site_id(prefs)
    active_site = get_site_definition(prefs, active_site_id)
    location = prefs["location"]
    location_label = str(active_site.get("name") or location.get("label") or "").strip() or DEFAULT_SITE_NAME
    label_editor_key = f"location_name_inline_edit_{active_site_id}"
    label_editor_sync_key = f"location_name_inline_edit_synced_label_{active_site_id}"

    if str(st.session_state.get(label_editor_sync_key, "")).strip() != location_label:
        st.session_state[label_editor_key] = location_label
        st.session_state[label_editor_sync_key] = location_label

    def _apply_location_label_edit() -> None:
        edited_value = str(st.session_state.get(label_editor_key, "")).strip()
        if not edited_value:
            st.session_state[label_editor_key] = location_label
            return
        if edited_value == location_label:
            return
        prefs["location"]["label"] = edited_value
        persist_legacy_fields_to_active_site(prefs)
        st.session_state["location_notice"] = f"Site name updated: {edited_value}."
        persist_and_rerun(prefs)

    name_weight = max(2.0, min(8.0, len(location_label) / 4.0))
    name_col, glyph_col, _name_spacer_col = st.columns([name_weight, 0.8, 12.0], gap="small")
    with name_col:
        st.markdown(f"**{location_label}**")
    with glyph_col:
        if hasattr(st, "popover"):
            with st.popover("✏️", help="Edit site name", use_container_width=True):
                st.text_input(
                    "Site name",
                    key=label_editor_key,
                    label_visibility="collapsed",
                    on_change=_apply_location_label_edit,
                )
                st.caption("Press Enter to save.")
        else:
            st.text_input(
                "Site name",
                key=label_editor_key,
                label_visibility="collapsed",
                on_change=_apply_location_label_edit,
            )

    lat_lon_text = f"Lat {location['lat']:.4f}, Lon {location['lon']:.4f}"
    source_badge = resolve_location_source_badge(location.get("source"))
    if source_badge is None:
        st.caption(lat_lon_text)
    else:
        badge_label, badge_kind = source_badge
        st.markdown(
            (
                "<div class='dso-location-meta'>"
                f"<span>{html.escape(lat_lon_text)}</span>"
                f"<span class='dso-location-source-badge dso-location-source-badge--{badge_kind}'>"
                f"{html.escape(badge_label)}"
                "</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    form_col, _form_spacer_col = st.columns([3, 7], gap="large")
    with form_col:
        with st.form(f"site_location_search_form_{active_site_id}"):
            manual_location = st.text_input(
                "Location",
                key=f"manual_location_{active_site_id}",
                placeholder="enter an address, zip code, or landmark",
            )
            search_col, browser_col = st.columns(2, gap="small")
            with search_col:
                search_submitted = st.form_submit_button("Search", use_container_width=True)
            with browser_col:
                browser_geo_submitted = st.form_submit_button("🧭", use_container_width=True)

    if search_submitted:
        location_query = manual_location.strip()
        if not location_query:
            st.warning("Enter an address, zip code, or landmark.")
        else:
            resolved = resolve_manual_location(location_query)
            if resolved:
                resolved_label, kept_site_name = apply_resolved_location(prefs, resolved)
                st.session_state["location_notice"] = (
                    f"Location resolved: {resolved_label}. Site name unchanged."
                    if kept_site_name
                    else f"Location resolved: {resolved_label}."
                )
                persist_and_rerun(prefs)
            else:
                st.warning("Couldn't find that location - keeping previous location.")

    if browser_geo_submitted:
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

    try:
        current_map_lat = float(location.get("lat", 0.0))
    except (TypeError, ValueError):
        current_map_lat = 0.0
    try:
        current_map_lon = float(location.get("lon", 0.0))
    except (TypeError, ValueError):
        current_map_lon = 0.0
    current_map_lat = float(max(-90.0, min(90.0, current_map_lat)))
    current_map_lon = float(max(-180.0, min(180.0, current_map_lon)))

    interactive_map = build_location_selection_map(
        current_map_lat,
        current_map_lon,
        zoom_start=8 if is_location_configured(location) else 2,
    )
    if interactive_map is not None and st_folium is not None:
        st.caption("Right-click any point on the map to set the site location.")
        map_event = st_folium(
            interactive_map,
            height=320,
            use_container_width=True,
            key=f"site_location_selector_map_{active_site_id}",
        )
        clicked = map_event.get("last_clicked") if isinstance(map_event, dict) else None
        if isinstance(clicked, dict):
            clicked_lat_raw = clicked.get("lat")
            clicked_lon_raw = clicked.get("lng")
            try:
                clicked_lat = float(clicked_lat_raw)
                clicked_lon = float(clicked_lon_raw)
            except (TypeError, ValueError):
                clicked_lat = current_map_lat
                clicked_lon = current_map_lon
            clicked_lat = float(max(-90.0, min(90.0, clicked_lat)))
            clicked_lon = float(max(-180.0, min(180.0, clicked_lon)))

            if abs(clicked_lat - current_map_lat) > 1e-6 or abs(clicked_lon - current_map_lon) > 1e-6:
                before_location = dict(prefs["location"])
                resolved_label, kept_site_name = apply_resolved_location(
                    prefs,
                    {
                        "lat": clicked_lat,
                        "lon": clicked_lon,
                        "label": reverse_geocode_label(clicked_lat, clicked_lon),
                        "source": "map",
                        "resolved_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                if prefs["location"] != before_location:
                    st.session_state["location_notice"] = (
                        f"Location set from map: {resolved_label}. Site name unchanged."
                        if kept_site_name
                        else f"Location set from map: {resolved_label}."
                    )
                    persist_and_rerun(prefs)
    else:
        if is_location_configured(location):
            st.map(
                pd.DataFrame({"lat": [current_map_lat], "lon": [current_map_lon]}),
                zoom=8,
                height=320,
                use_container_width=True,
            )
        else:
            st.info("No location set yet. Enter one manually or use browser geolocation; IP-based estimate is fallback.")


def render_obstructions_settings_section(prefs: dict[str, Any]) -> None:
    st.subheader("Obstructions")
    st.caption("Choose an input method. All obstruction data is normalized and stored as WIND16.")
    sync_active_site_to_legacy_fields(prefs)
    active_site_id = get_active_site_id(prefs)
    current_obstructions = {
        direction: clamp_obstruction_altitude(prefs["obstructions"].get(direction), default=20.0)
        for direction in WIND16
    }
    location_label = str(prefs.get("location", {}).get("label", "")).strip()
    filename_base = re.sub(r"[^a-zA-Z0-9._-]+", "-", location_label).strip("-").lower() or "observation-site"
    st.download_button(
        "Export WIND16 as .hrz",
        data=wind16_obstructions_to_hrz_text(current_obstructions),
        file_name=f"{filename_base}-obstructions.hrz",
        mime="text/plain",
        use_container_width=False,
        key=f"obstruction_export_hrz_button_{active_site_id}",
    )

    def _apply_wind16_obstructions(next_values: dict[str, Any], *, success_message: str) -> None:
        normalized = {
            direction: clamp_obstruction_altitude(next_values.get(direction), default=current_obstructions[direction])
            for direction in WIND16
        }
        if normalized != prefs["obstructions"]:
            prefs["obstructions"] = normalized
            persist_legacy_fields_to_active_site(prefs)
            save_preferences(prefs)
            st.success(success_message)
        else:
            st.info("No obstruction changes detected.")

    input_mode = st.radio(
        "Obstruction input method",
        options=OBSTRUCTION_INPUT_MODES,
        horizontal=True,
        key=f"obstruction_input_mode_{active_site_id}",
    )

    if input_mode == OBSTRUCTION_INPUT_MODE_NESW:
        st.caption("Use coarse cardinal values; the app expands them to all 16 wind directions.")
        wind16_average = float(
            np.mean([clamp_obstruction_altitude(current_obstructions.get(direction), default=20.0) for direction in WIND16])
        )
        for direction in CARDINAL_DIRECTIONS:
            sync_slider_state_value(
                f"obstruction_cardinal_slider_{active_site_id}_{direction}",
                wind16_average,
            )

        with st.form(f"obstruction_cardinal_form_{active_site_id}"):
            slider_cols = st.columns(len(CARDINAL_DIRECTIONS), gap="small")
            cardinal_values: dict[str, float] = {}
            for idx, direction in enumerate(CARDINAL_DIRECTIONS):
                with slider_cols[idx]:
                    cardinal_values[direction] = float(
                        st.slider(
                            direction,
                            min_value=0,
                            max_value=90,
                            step=1,
                            key=f"obstruction_cardinal_slider_{active_site_id}_{direction}",
                        )
                    )
            apply_cardinals = st.form_submit_button("Apply N/E/S/W to WIND16", use_container_width=True)

        if apply_cardinals:
            expanded = expand_cardinal_obstructions_to_wind16(cardinal_values)
            _apply_wind16_obstructions(
                expanded,
                success_message="Applied cardinal obstructions and expanded to WIND16.",
            )
        return

    if input_mode == OBSTRUCTION_INPUT_MODE_HRZ:
        st.caption("Upload a horizon file (.hrz, APCC/N.I.N.A compatible).")
        uploaded_hrz = st.file_uploader(
            "Horizon file (.hrz)",
            type=["hrz"],
            accept_multiple_files=False,
            key=f"obstruction_hrz_file_{active_site_id}",
        )
        apply_hrz = st.button(
            "Apply .hrz to WIND16",
            use_container_width=False,
            disabled=uploaded_hrz is None,
            key=f"obstruction_hrz_apply_button_{active_site_id}",
        )

        if apply_hrz and uploaded_hrz is not None:
            raw_bytes = uploaded_hrz.getvalue()
            try:
                raw_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raw_text = raw_bytes.decode("latin-1", errors="ignore")

            points = parse_hrz_obstruction_points(raw_text)
            if not points:
                st.warning("No valid azimuth/altitude points were found in that .hrz file.")
                return

            reduced, missing_directions = reduce_hrz_points_to_wind16(
                points,
                fallback=current_obstructions,
            )
            _apply_wind16_obstructions(
                reduced,
                success_message="Applied .hrz horizon profile and reduced it to WIND16 maxima.",
            )
            st.caption(f"Parsed {len(points)} horizon points from `{uploaded_hrz.name}`.")
            if missing_directions:
                missing_list = ", ".join(missing_directions)
                st.warning(
                    f"No samples mapped to: {missing_list}. Existing WIND16 values were kept for those directions."
                )
        return

    if vertical_slider is None:
        st.warning(
            "`streamlit-vertical-slider` is required for the vertical WIND16 sliders. "
            "Falling back to table editor."
        )
        obstruction_frame = pd.DataFrame(
            {
                "Direction": WIND16,
                "Min Altitude (deg)": [current_obstructions.get(direction, 20.0) for direction in WIND16],
            }
        )
        edited = st.data_editor(
            obstruction_frame,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            disabled=["Direction"],
            key=f"obstruction_editor_{active_site_id}",
        )

        edited_values = edited["Min Altitude (deg)"].tolist()
        next_obstructions = {
            direction: clamp_obstruction_altitude(edited_values[idx], default=current_obstructions[direction])
            for idx, direction in enumerate(WIND16)
        }
        if next_obstructions != prefs["obstructions"]:
            prefs["obstructions"] = next_obstructions
            persist_legacy_fields_to_active_site(prefs)
            save_preferences(prefs)
        return

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
            default_val = int(round(current_obstructions.get(direction, 20.0)))
            state_key = f"obstruction_slider_{active_site_id}_{direction}"
            sync_slider_state_value(state_key, default_val)
            preview_value_raw = st.session_state.get(state_key, default_val)
            try:
                preview_value = float(preview_value_raw)
            except (TypeError, ValueError):
                preview_value = float(default_val)
            preview_clamped = clamp_obstruction_altitude(preview_value, default=float(default_val))
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
            clamped_value = clamp_obstruction_altitude(raw_value, default=float(default_val))
            next_obstructions[direction] = clamped_value
            with value_cols[idx]:
                st.markdown(
                    f"<div style='text-align:center; font-size:0.8rem; color:#64748b;'>{int(round(clamped_value))} deg</div>",
                    unsafe_allow_html=True,
                )

        if next_obstructions != prefs["obstructions"]:
            prefs["obstructions"] = next_obstructions
            persist_legacy_fields_to_active_site(prefs)
            save_preferences(prefs)


def render_sites_page(prefs: dict[str, Any]) -> None:
    st.title("Observation Sites")
    st.caption("Manage sites. One site is active at a time and used across Explorer weather/visibility calculations.")

    persistence_notice = st.session_state.get("prefs_persistence_notice", "")
    if persistence_notice:
        st.warning(persistence_notice)

    location_notice = st.session_state.pop("location_notice", "")
    if location_notice:
        st.info(location_notice)

    site_ids = site_ids_in_order(prefs)
    if not site_ids:
        site_ids = [DEFAULT_SITE_ID]
    active_site_id = get_active_site_id(prefs)
    if active_site_id not in site_ids and site_ids:
        active_site_id = site_ids[0]
        set_active_site(prefs, active_site_id)

    with st.container(border=True):
        st.subheader("Sites")
        rows: list[dict[str, str]] = []
        for site_id in site_ids:
            site = get_site_definition(prefs, site_id)
            location = site.get("location", {})
            lat_text = "-"
            lon_text = "-"
            try:
                lat_text = f"{float(location.get('lat')):.4f}"
                lon_text = f"{float(location.get('lon')):.4f}"
            except (TypeError, ValueError):
                pass
            source_badge = resolve_location_source_badge(location.get("source"))
            source_label = source_badge[0] if source_badge is not None else "-"
            rows.append(
                {
                    "Active": "●" if site_id == active_site_id else "",
                    "Name": get_site_name(prefs, site_id),
                    "Latitude": lat_text,
                    "Longitude": lon_text,
                    "Source": source_label,
                }
            )
        site_frame = pd.DataFrame(rows)
        table_height = max(72, min(280, 36 * (len(site_frame) + 1)))
        st.dataframe(site_frame, hide_index=True, use_container_width=True, height=table_height)

        select_key = "sites_selected_site_id"
        pending_select_key = "sites_selected_site_id_pending"
        pending_selected = str(st.session_state.pop(pending_select_key, "")).strip()
        if pending_selected in site_ids:
            st.session_state[select_key] = pending_selected
        selected_default = str(st.session_state.get(select_key, "")).strip()
        if selected_default not in site_ids and site_ids:
            selected_default = active_site_id
            st.session_state[select_key] = selected_default
        selected_site_id = st.selectbox(
            "Selected site",
            options=site_ids,
            index=site_ids.index(selected_default) if selected_default in site_ids else 0,
            key=select_key,
            format_func=lambda site_id: get_site_name(prefs, site_id),
        )

        add_col, edit_col, duplicate_col, delete_col = st.columns(4, gap="small")
        if add_col.button("Add new site", use_container_width=True, key="sites_add_button"):
            created_site_id = create_site(prefs)
            if created_site_id:
                set_active_site(prefs, created_site_id)
                st.session_state[pending_select_key] = created_site_id
                st.session_state["location_notice"] = f"Created new site: {get_site_name(prefs, created_site_id)}."
                persist_and_rerun(prefs)

        if edit_col.button("Edit", use_container_width=True, key="sites_edit_button"):
            changed = set_active_site(prefs, selected_site_id)
            if changed:
                st.session_state["location_notice"] = f"Active site set to: {get_site_name(prefs, selected_site_id)}."
                persist_and_rerun(prefs)

        if duplicate_col.button("Duplicate", use_container_width=True, key="sites_duplicate_button"):
            duplicated_site_id = duplicate_site(prefs, selected_site_id)
            if duplicated_site_id:
                set_active_site(prefs, duplicated_site_id)
                st.session_state[pending_select_key] = duplicated_site_id
                st.session_state["location_notice"] = (
                    f"Created duplicate site: {get_site_name(prefs, duplicated_site_id)}."
                )
                persist_and_rerun(prefs)

        can_delete = len(site_ids) > 1
        if delete_col.button(
            "Delete",
            use_container_width=True,
            key="sites_delete_button",
            disabled=not can_delete,
        ):
            st.session_state["sites_delete_pending_id"] = selected_site_id

        pending_delete_id = str(st.session_state.get("sites_delete_pending_id", "")).strip()
        if pending_delete_id and pending_delete_id in site_ids:
            st.warning(f"Delete site '{get_site_name(prefs, pending_delete_id)}'? This cannot be undone.")
            confirm_col, cancel_col = st.columns(2, gap="small")
            if confirm_col.button("Confirm Delete", use_container_width=True, key="sites_delete_confirm_button"):
                deleted_name = get_site_name(prefs, pending_delete_id)
                if delete_site(prefs, pending_delete_id):
                    st.session_state.pop("sites_delete_pending_id", None)
                    st.session_state["location_notice"] = f"Deleted site: {deleted_name}."
                    persist_and_rerun(prefs)
            if cancel_col.button("Cancel", use_container_width=True, key="sites_delete_cancel_button"):
                st.session_state.pop("sites_delete_pending_id", None)

    sync_active_site_to_legacy_fields(prefs)
    active_site_name = get_site_name(prefs, get_active_site_id(prefs))
    st.caption(f"Editing active site: {active_site_name}")

    with st.container(border=True):
        render_location_settings_section(prefs)

    with st.container(border=True):
        render_obstructions_settings_section(prefs)


def render_lists_page(prefs: dict[str, Any]) -> None:
    st.title("Lists")
    st.caption("Manage target lists used across explorer search, overlays, and detail actions.")

    persistence_notice = st.session_state.get("prefs_persistence_notice", "")
    if persistence_notice:
        st.warning(persistence_notice)

    with st.container(border=True):
        render_lists_settings_section(
            prefs,
            persist_and_rerun_fn=persist_and_rerun,
            show_subheader=False,
        )


def render_equipment_page(prefs: dict[str, Any]) -> None:
    st.title("Equipment")
    st.caption("Store your telescopes/accessories/filters. Recommendation integration will come in a later update.")

    persistence_notice = st.session_state.get("prefs_persistence_notice", "")
    if persistence_notice:
        st.warning(persistence_notice)

    catalog = load_equipment_catalog()
    categories = catalog.get("categories", [])
    if not isinstance(categories, list) or not categories:
        st.warning(f"No equipment catalog entries found at `{EQUIPMENT_CATALOG_PATH}`.")
        return

    current_equipment_raw = prefs.get("equipment", {})
    if not isinstance(current_equipment_raw, dict):
        current_equipment_raw = {}

    def _normalize_selected_ids(raw_values: Any, allowed_ids: set[str]) -> list[str]:
        normalized: list[str] = []
        if not isinstance(raw_values, (list, tuple, set)):
            return normalized
        for raw_value in raw_values:
            item_id = str(raw_value).strip()
            if item_id and item_id in allowed_ids and item_id not in normalized:
                normalized.append(item_id)
        return normalized

    next_equipment: dict[str, list[str]] = {}
    current_equipment_known: dict[str, list[str]] = {}

    with st.container(border=True):
        st.subheader("Owned Equipment")
        st.caption("Use the Owned column to select one or more items per table. Changes are saved automatically.")

        for category in categories:
            category_id = str(category.get("id", "")).strip()
            if not category_id:
                continue
            label = str(category.get("label", category_id)).strip() or category_id
            description = str(category.get("description", "")).strip()
            items = category.get("items", [])
            if not isinstance(items, list) or not items:
                continue

            item_by_id: dict[str, dict[str, Any]] = {}
            ordered_item_ids: list[str] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id", "")).strip()
                item_name = str(item.get("name", "")).strip()
                if not item_id or not item_name:
                    continue
                item_by_id[item_id] = item
                ordered_item_ids.append(item_id)
            if not ordered_item_ids:
                continue

            allowed_ids = set(ordered_item_ids)
            existing_selected = _normalize_selected_ids(current_equipment_raw.get(category_id, []), allowed_ids)
            current_equipment_known[category_id] = existing_selected

            st.markdown(f"#### {label}")
            if description:
                st.caption(description)

            display_columns = category.get("display_columns", [])
            parsed_columns = [
                column
                for column in display_columns
                if isinstance(column, dict) and str(column.get("field", "")).strip()
            ]
            existing_set = set(existing_selected)
            table_rows: list[dict[str, Any]] = []
            for item_id in ordered_item_ids:
                item = item_by_id[item_id]
                row: dict[str, Any] = {
                    "Owned": item_id in existing_set,
                    "Name": str(item.get("name", item_id)).strip() or item_id,
                }
                for column in parsed_columns:
                    field = str(column.get("field", "")).strip()
                    column_label = str(column.get("label", field)).strip() or field
                    row[column_label] = format_equipment_value(item.get(field))
                table_rows.append(row)

            table_frame = pd.DataFrame(table_rows)
            table_height = max(72, min(360, 36 * (len(table_frame) + 1)))
            editable_cols = {"Owned"}
            disabled_cols = [column for column in table_frame.columns if column not in editable_cols]

            editor_kwargs: dict[str, Any] = {
                "hide_index": True,
                "use_container_width": False,
                "disabled": disabled_cols,
                "num_rows": "fixed",
                "height": table_height,
                "key": f"equipment_table_{category_id}",
            }
            if hasattr(st, "column_config") and hasattr(st.column_config, "CheckboxColumn"):
                editor_kwargs["column_config"] = {
                    "Owned": st.column_config.CheckboxColumn("Owned", width="small"),
                }

            edited_frame = st.data_editor(table_frame, **editor_kwargs)
            owned_values = (
                edited_frame["Owned"].tolist()
                if isinstance(edited_frame, pd.DataFrame) and "Owned" in edited_frame.columns
                else [False] * len(ordered_item_ids)
            )
            next_equipment[category_id] = [
                ordered_item_ids[idx]
                for idx, owned in enumerate(owned_values[: len(ordered_item_ids)])
                if bool(owned)
            ]

    if next_equipment != current_equipment_known:
        prefs["equipment"] = next_equipment
        st.session_state["prefs"] = prefs
        save_preferences(prefs)


def render_settings_page(catalog_meta: dict[str, Any], prefs: dict[str, Any], browser_locale: str | None) -> None:
    st.title("Settings")
    st.caption("Manage display preferences, catalog metadata, and settings backup.")

    persistence_notice = st.session_state.get("prefs_persistence_notice", "")
    if persistence_notice:
        st.warning(persistence_notice)
    import_notice = str(st.session_state.pop("settings_import_notice", "")).strip()
    if import_notice:
        st.success(import_notice)
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
    st.subheader("Catalog")
    st.caption(
        f"Rows: {int(catalog_meta.get('row_count', 0))} | "
        f"Source: {catalog_meta.get('source', str(CATALOG_CACHE_PATH))}"
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
        help="Imports sites, obstructions, equipment, lists, and display preferences.",
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
                    imported_site_count = len(site_ids_in_order(imported_prefs))
                    imported_list_count = len(list_ids_in_order(imported_prefs, include_auto_recent=True))
                    imported_equipment_payload = imported_prefs.get("equipment", {})
                    imported_equipment_count = 0
                    if isinstance(imported_equipment_payload, dict):
                        imported_equipment_count = sum(
                            len(items)
                            for items in imported_equipment_payload.values()
                            if isinstance(items, list)
                        )
                    st.session_state["location_notice"] = "Settings imported."
                    st.session_state["settings_import_notice"] = (
                        "Settings imported: "
                        f"{imported_site_count} site(s), "
                        f"{imported_list_count} list(s), "
                        f"{imported_equipment_count} equipment item(s)."
                    )
                    persist_and_rerun(imported_prefs)


def render_detail_panel(
    selected: pd.Series | None,
    catalog: pd.DataFrame,
    prefs: dict[str, Any],
    temperature_unit: str,
    use_12_hour: bool,
    detail_stack_vertical: bool,
    weather_forecast_day_offset: int = 0,
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

    try:
        normalized_forecast_day_offset = int(weather_forecast_day_offset)
    except (TypeError, ValueError):
        normalized_forecast_day_offset = 0
    if normalized_forecast_day_offset < 0:
        normalized_forecast_day_offset = 0
    if normalized_forecast_day_offset > (ASTRONOMY_FORECAST_NIGHTS - 1):
        normalized_forecast_day_offset = ASTRONOMY_FORECAST_NIGHTS - 1

    location = prefs["location"]
    location_lat = float(location["lat"])
    location_lon = float(location["lon"])
    recommendation_window_start, recommendation_window_end, recommendation_tzinfo = weather_forecast_window(
        location_lat,
        location_lon,
        day_offset=normalized_forecast_day_offset,
    )
    active_preview_list_id_for_recommendations = get_active_preview_list_id(prefs)
    active_preview_list_ids_for_recommendations = get_list_ids(prefs, active_preview_list_id_for_recommendations)

    with st.container(border=True):
        render_target_recommendations(
            catalog,
            prefs,
            active_preview_list_ids=active_preview_list_ids_for_recommendations,
            window_start=recommendation_window_start,
            window_end=recommendation_window_end,
            tzinfo=recommendation_tzinfo,
            use_12_hour=use_12_hour,
            weather_forecast_day_offset=normalized_forecast_day_offset,
        )

    if selected is None:
        with st.container(border=True):
            st.info("No target selected. Showing preview list targets.")

        window_start, window_end, tzinfo = weather_forecast_window(
            location_lat,
            location_lon,
            day_offset=normalized_forecast_day_offset,
        )
        forecast_period_label = describe_weather_forecast_period(normalized_forecast_day_offset)

        active_preview_list_id = get_active_preview_list_id(prefs)
        active_preview_list_name = get_list_name(prefs, active_preview_list_id)
        active_preview_list_ids = get_list_ids(prefs, active_preview_list_id)
        active_preview_list_members = set(active_preview_list_ids)
        preview_list_is_system = is_system_list(prefs, active_preview_list_id)

        available_preview_list_ids = list_ids_in_order(prefs, include_auto_recent=True)
        if active_preview_list_id not in available_preview_list_ids:
            active_preview_list_id = get_active_preview_list_id(prefs)
            active_preview_list_name = get_list_name(prefs, active_preview_list_id)
            active_preview_list_ids = get_list_ids(prefs, active_preview_list_id)
            active_preview_list_members = set(active_preview_list_ids)
            preview_list_is_system = is_system_list(prefs, active_preview_list_id)

        hourly_weather_rows = fetch_hourly_weather(
            lat=location_lat,
            lon=location_lon,
            tz_name=tzinfo.key,
            start_local_iso=window_start.isoformat(),
            end_local_iso=window_end.isoformat(),
            hourly_fields=EXTENDED_FORECAST_HOURLY_FIELDS,
        )
        nightly_weather_alert_emojis = collect_night_weather_alert_emojis(hourly_weather_rows, temperature_unit)
        temperatures, _, weather_by_hour = build_hourly_weather_maps(hourly_weather_rows)

        with st.container(border=True):
            st.markdown("### Night Sky Preview")

            st.caption(
                f"{forecast_period_label} ({tzinfo.key}): "
                f"{format_display_time(window_start, use_12_hour=use_12_hour)} -> "
                f"{format_display_time(window_end, use_12_hour=use_12_hour)}"
            )

            summary_highlight_id = str(st.session_state.get("sky_summary_highlight_primary_id", "")).strip()
            preview_tracks: list[dict[str, Any]] = []
            preview_targets = subset_by_id_list(catalog, active_preview_list_ids)
            for _, preview_target in preview_targets.iterrows():
                preview_target_id = str(preview_target["primary_id"])
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
                        lat=location_lat,
                        lon=location_lon,
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

            group_total_counts: dict[str, int] = {}
            for track_payload in preview_tracks:
                group_key = str(track_payload.get("object_type_group") or "").strip() or "other"
                group_total_counts[group_key] = group_total_counts.get(group_key, 0) + 1

            group_seen_counts: dict[str, int] = {}

            def _next_group_plot_color(group_label: str | None) -> str:
                group_key = str(group_label or "").strip() or "other"
                index_in_group = group_seen_counts.get(group_key, 0)
                group_seen_counts[group_key] = index_in_group + 1
                total_in_group = max(1, int(group_total_counts.get(group_key, 1)))
                step_fraction = 0.0 if total_in_group <= 1 else (float(index_in_group) / float(total_in_group - 1))
                return object_type_group_color(group_key, step_fraction=step_fraction)

            for track_payload in preview_tracks:
                group_key = str(track_payload.get("object_type_group") or "").strip() or "other"
                track_payload["color"] = _next_group_plot_color(group_key)

            st.caption(f"Preview list: {active_preview_list_name} ({len(preview_tracks)} targets)")

            summary_rows = build_sky_position_summary_rows(
                selected_id=None,
                selected_label=None,
                selected_type_group=None,
                selected_color=None,
                selected_events=None,
                selected_track=None,
                overlay_tracks=preview_tracks,
                list_member_ids=active_preview_list_members,
                now_local=pd.Timestamp(datetime.now(tzinfo)),
                row_order_ids=[str(item) for item in active_preview_list_ids],
            )

            local_now = datetime.now(tzinfo)
            show_remaining_column = window_start <= local_now <= window_end
            plots_container = st.container()
            summary_container = st.container()

            unobstructed_area_tracks: list[dict[str, Any]] = []
            focused_preview_track: dict[str, Any] | None = None
            with summary_container:
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
                    summary_highlight_id = str(st.session_state.get("sky_summary_highlight_primary_id", "")).strip()
                    unobstructed_area_tracks = [
                        {
                            **preview_track,
                            "is_selected": (
                                bool(summary_highlight_id)
                                and str(preview_track.get("primary_id", "")).strip() == summary_highlight_id
                            ),
                        }
                        for preview_track in preview_tracks
                    ]
                    focused_preview_track = next(
                        (
                            preview_track
                            for preview_track in unobstructed_area_tracks
                            if bool(preview_track.get("is_selected", False))
                        ),
                        (unobstructed_area_tracks[0] if unobstructed_area_tracks else None),
                    )
                with tips_col:
                    with st.container(border=True):
                        render_target_tips_panel(
                            "",
                            "No target selected",
                            None,
                            None,
                            summary_rows,
                            nightly_weather_alert_emojis,
                            hourly_weather_rows,
                            temperature_unit=temperature_unit,
                            use_12_hour=use_12_hour,
                            local_now=local_now,
                            window_start=window_start,
                            window_end=window_end,
                        )

            with plots_container:
                path_figure: go.Figure | None = None
                if focused_preview_track is not None:
                    overlay_tracks_for_path = [
                        payload
                        for payload in unobstructed_area_tracks
                        if str(payload.get("primary_id", "")).strip()
                        != str(focused_preview_track.get("primary_id", "")).strip()
                    ]
                    path_style = st.segmented_control(
                        "Target Paths Style",
                        options=["Line", "Radial"],
                        default="Line",
                        key="path_style_preference",
                    )
                    if path_style == "Radial":
                        dome_view = st.toggle("Dome View", value=True, key="dome_view_preference")
                        path_figure = build_path_plot_radial(
                            track=focused_preview_track.get("track"),
                            events=focused_preview_track.get("events", {}),
                            obstructions=prefs["obstructions"],
                            dome_view=dome_view,
                            selected_label=str(focused_preview_track.get("label", "Preview target")),
                            selected_emissions=str(focused_preview_track.get("emission_lines_display") or ""),
                            selected_color=str(focused_preview_track.get("color", OBJECT_TYPE_GROUP_COLOR_DEFAULT)),
                            selected_line_width=float(
                                focused_preview_track.get("line_width", PATH_LINE_WIDTH_PRIMARY_DEFAULT)
                            ),
                            use_12_hour=use_12_hour,
                            overlay_tracks=overlay_tracks_for_path,
                        )
                    else:
                        path_figure = build_path_plot(
                            track=focused_preview_track.get("track"),
                            events=focused_preview_track.get("events", {}),
                            obstructions=prefs["obstructions"],
                            selected_label=str(focused_preview_track.get("label", "Preview target")),
                            selected_emissions=str(focused_preview_track.get("emission_lines_display") or ""),
                            selected_color=str(focused_preview_track.get("color", OBJECT_TYPE_GROUP_COLOR_DEFAULT)),
                            selected_line_width=float(
                                focused_preview_track.get("line_width", PATH_LINE_WIDTH_PRIMARY_DEFAULT)
                            ),
                            use_12_hour=use_12_hour,
                            overlay_tracks=overlay_tracks_for_path,
                        )

                path_col, area_col = st.columns([1, 1], gap="small")
                with path_col:
                    if path_figure is None:
                        st.info("No preview tracks available for path rendering.")
                    else:
                        st.plotly_chart(
                            path_figure,
                            use_container_width=True,
                            key="preview_path_plot",
                        )
                with area_col:
                    st.plotly_chart(
                        build_unobstructed_altitude_area_plot(
                            unobstructed_area_tracks,
                            use_12_hour=use_12_hour,
                            temperature_by_hour=temperatures,
                            weather_by_hour=weather_by_hour,
                            temperature_unit=temperature_unit,
                        ),
                        use_container_width=True,
                        key="preview_unobstructed_area_plot",
                    )
                    st.caption(WEATHER_ALERT_INDICATOR_LEGEND_CAPTION)
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
    forecast_placeholder: Any | None = None
    forecast_legend_placeholder: Any | None = None
    forecast_cloud_cover_legend_placeholder: Any | None = None

    detail_modal = Modal(title, key="target_detail_modal") if Modal is not None else None
    if detail_modal is not None:
        st.markdown(
            """
            <style>
            div[data-modal-container='true'][key='target_detail_modal'] > div:first-child {
                width: 80vw !important;
                max-width: 80vw !important;
            }
            div[data-modal-container='true'][key='target_detail_modal'] > div:first-child > div:first-child > div:first-child {
                max-width: 80vw !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        last_modal_target = str(st.session_state.get(TARGET_DETAIL_MODAL_LAST_TARGET_STATE_KEY, "")).strip()
        if target_id != last_modal_target:
            st.session_state[TARGET_DETAIL_MODAL_LAST_TARGET_STATE_KEY] = target_id
            st.session_state[TARGET_DETAIL_MODAL_OPEN_REQUEST_KEY] = True
        if bool(st.session_state.pop(TARGET_DETAIL_MODAL_OPEN_REQUEST_KEY, False)):
            detail_modal.open()

        detail_header_cols = st.columns([4, 1], gap="small")
        detail_header_cols[0].caption(f"Selected target: {title}")
        if detail_header_cols[1].button(
            "Open details",
            key="target_detail_modal_reopen_button",
            use_container_width=True,
        ):
            detail_modal.open()

    render_detail_pane = detail_modal is None or detail_modal.is_open()
    detail_container_context = (
        (detail_modal.container() if detail_modal is not None else st.container(border=True))
        if render_detail_pane
        else None
    )
    if detail_container_context is not None:
        with detail_container_context:
            if detail_modal is None:
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
                if image_source_url:
                    st.caption(f"Image source: [Open link]({image_source_url})")
                if info_url:
                    st.caption(f"Background: [Open object page]({info_url})")
                if image_license:
                    st.caption(f"License/Credit: {image_license}")

            with description_container:
                st.markdown("**Description**")
                st.write(description or "-")

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
                            st.session_state[TARGET_DETAIL_MODAL_OPEN_REQUEST_KEY] = True
                            persist_and_rerun(prefs)
                    st.caption(
                        f"{'In' if is_in_selected_action_list else 'Not in'} list: {selected_action_list_name}"
                    )

                ra_deg_value = parse_numeric(selected.get("ra_deg"))
                dec_deg_value = parse_numeric(selected.get("dec_deg"))
                ra_sexagesimal = format_ra_hms(ra_deg_value)
                dec_sexagesimal = format_dec_dms(dec_deg_value)
                ra_decimal = f"{ra_deg_value:.4f} deg" if ra_deg_value is not None else "-"
                dec_decimal = f"{dec_deg_value:.4f} deg" if dec_deg_value is not None else "-"

                def _format_coordinate_value_html(primary_value: str, secondary_value: str) -> str:
                    primary_html = html.escape(str(primary_value or "-"))
                    secondary_html = html.escape(str(secondary_value or "-"))
                    return (
                        f"{primary_html} "
                        f'<span style="font-size:0.82em; color:var(--dso-muted-text-color);">({secondary_html})</span>'
                    )

                property_items = [
                    {
                        "Property": "RA",
                        "Value": ra_sexagesimal,
                        "ValueHtml": _format_coordinate_value_html(ra_sexagesimal, ra_decimal),
                    },
                    {
                        "Property": "DEC",
                        "Value": dec_sexagesimal,
                        "ValueHtml": _format_coordinate_value_html(dec_sexagesimal, dec_decimal),
                    },
                    {"Property": "Constellation", "Value": clean_text(selected.get("constellation")) or "-"},
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
                        raw_value_html = clean_text(row.get("ValueHtml"))
                        value_html = raw_value_html if raw_value_html else html.escape(value_text)
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
                forecast_cloud_cover_legend_placeholder = st.empty()

    window_start, window_end, tzinfo = tonight_window(location_lat, location_lon)
    forecast_window_start, forecast_window_end, _ = weather_forecast_window(
        location_lat,
        location_lon,
        day_offset=normalized_forecast_day_offset,
    )
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
    if active_preview_list_id not in available_preview_list_ids:
        fallback_preview_list_id = available_preview_list_ids[0] if available_preview_list_ids else AUTO_RECENT_LIST_ID
        set_active_preview_list_id(prefs, fallback_preview_list_id)
        active_preview_list_id = fallback_preview_list_id
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
                lat=location_lat,
                lon=location_lon,
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
        lat=location_lat,
        lon=location_lon,
        start_local=window_start,
        end_local=window_end,
        obstructions=prefs["obstructions"],
    )
    forecast_track = track
    if normalized_forecast_day_offset > 0:
        try:
            forecast_track = compute_track(
                ra_deg=selected_ra,
                dec_deg=selected_dec,
                lat=location_lat,
                lon=location_lon,
                start_local=forecast_window_start,
                end_local=forecast_window_end,
                obstructions=prefs["obstructions"],
            )
        except Exception:
            forecast_track = track
    events = extract_events(track)
    hourly_weather_rows = fetch_hourly_weather(
        lat=location_lat,
        lon=location_lon,
        tz_name=tzinfo.key,
        start_local_iso=forecast_window_start.isoformat(),
        end_local_iso=forecast_window_end.isoformat(),
        hourly_fields=EXTENDED_FORECAST_HOURLY_FIELDS,
    )
    forecast_hourly_weather_rows = hourly_weather_rows
    nightly_weather_alert_emojis = collect_night_weather_alert_emojis(forecast_hourly_weather_rows, temperature_unit)
    temperatures, cloud_cover_by_hour, weather_by_hour = build_hourly_weather_maps(forecast_hourly_weather_rows)
    if normalized_forecast_day_offset <= 0:
        detail_hourly_period_label = "Tonight"
    elif normalized_forecast_day_offset == 1:
        detail_hourly_period_label = "Tomorrow"
    else:
        detail_hourly_period_label = pd.Timestamp(forecast_window_start).strftime("%A")

    with st.container(border=True):
        st.markdown("### Night Sky Preview")
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

        should_animate_weather_alerts = normalized_forecast_day_offset == 0
        if let_it_rain is not None and should_animate_weather_alerts and nightly_weather_alert_emojis:
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
        highlight_for_area = summary_highlight_id if summary_highlight_id else target_id
        unobstructed_area_tracks = [
            {
                "is_selected": target_id == highlight_for_area,
                "label": selected_label,
                "color": selected_color,
                "line_width": selected_line_width,
                "emission_lines_display": emission_details_display,
                "track": track,
            },
            *[
                {
                    **preview_track,
                    "is_selected": str(preview_track.get("primary_id", "")).strip() == highlight_for_area,
                }
                for preview_track in preview_tracks
            ],
        ]
        path_col, area_col = st.columns([1, 1], gap="small")
        with path_col:
            st.plotly_chart(
                path_figure,
                use_container_width=True,
                key="detail_path_plot",
            )
        with area_col:
            st.plotly_chart(
                build_unobstructed_altitude_area_plot(
                    unobstructed_area_tracks,
                    use_12_hour=use_12_hour,
                    temperature_by_hour=temperatures,
                    weather_by_hour=weather_by_hour,
                    temperature_unit=temperature_unit,
                ),
                use_container_width=True,
                key="detail_unobstructed_area_plot",
            )
            st.caption(WEATHER_ALERT_INDICATOR_LEGEND_CAPTION)

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

    if (
        forecast_placeholder is not None
        and forecast_legend_placeholder is not None
        and forecast_cloud_cover_legend_placeholder is not None
    ):
        forecast_placeholder.plotly_chart(
            build_night_plot(
                track=forecast_track,
                temperature_by_hour=temperatures,
                cloud_cover_by_hour=cloud_cover_by_hour,
                weather_by_hour=weather_by_hour,
                temperature_unit=temperature_unit,
                target_label=selected_label,
                period_label=detail_hourly_period_label,
                use_12_hour=use_12_hour,
            ),
            use_container_width=True,
            key="detail_night_plot",
        )
        forecast_legend_placeholder.caption(WEATHER_ALERT_INDICATOR_LEGEND_CAPTION)
        forecast_cloud_cover_legend_placeholder.markdown(cloud_cover_color_legend_html(), unsafe_allow_html=True)


def render_sidebar_active_settings(
    prefs: dict[str, Any],
    *,
    theme_label_to_id: dict[str, str],
) -> str:
    current_theme = str(prefs.get("ui_theme", UI_THEME_LIGHT)).strip().lower()
    theme_id_to_label = {value: key for key, value in theme_label_to_id.items()}
    if current_theme not in theme_id_to_label:
        current_theme = UI_THEME_LIGHT

    site_ids = site_ids_in_order(prefs)
    if not site_ids:
        site_ids = [DEFAULT_SITE_ID]
    active_site_id = get_active_site_id(prefs)
    if active_site_id not in site_ids and site_ids:
        active_site_id = site_ids[0]

    available_list_ids = list_ids_in_order(prefs, include_auto_recent=True)
    if not available_list_ids:
        available_list_ids = [AUTO_RECENT_LIST_ID]
    active_preview_list_id = get_active_preview_list_id(prefs)
    if active_preview_list_id not in available_list_ids:
        active_preview_list_id = available_list_ids[0]

    equipment_context = build_owned_equipment_context(prefs)
    active_equipment = sync_active_equipment_settings(prefs, equipment_context)
    if bool(active_equipment.get("changed", False)):
        st.session_state["prefs"] = prefs
        save_preferences(prefs)

    owned_telescope_ids = list(active_equipment.get("owned_telescope_ids", []))
    owned_filter_ids = list(active_equipment.get("owned_filter_ids", []))
    telescope_lookup = dict(active_equipment.get("telescope_lookup", {}))
    filter_lookup = dict(active_equipment.get("filter_lookup", {}))
    active_telescope_id = str(active_equipment.get("active_telescope_id", "")).strip()
    active_filter_id = str(active_equipment.get("active_filter_id", "__none__")).strip() or "__none__"
    active_mount_choice = _normalize_mount_choice(
        active_equipment.get("active_mount_choice", "altaz"),
        default_choice="altaz",
    )

    with st.sidebar:
        st.markdown("### Observation settings")
        selected_site_id = st.selectbox(
            "Site",
            options=site_ids,
            index=site_ids.index(active_site_id) if active_site_id in site_ids else 0,
            format_func=lambda site_id: get_site_name(prefs, site_id),
            key="sidebar_active_site_selector",
        )
        if selected_site_id != active_site_id and set_active_site(prefs, selected_site_id):
            persist_and_rerun(prefs)

        selected_list_id = st.selectbox(
            "List",
            options=available_list_ids,
            index=available_list_ids.index(active_preview_list_id),
            format_func=lambda list_id: get_list_name(prefs, list_id),
            key="sidebar_active_list_selector",
        )
        if selected_list_id != active_preview_list_id:
            if set_active_preview_list_id(prefs, selected_list_id):
                persist_and_rerun(prefs)

        if owned_filter_ids:
            filter_options = ["__none__"] + owned_filter_ids
            selected_filter_option = st.selectbox(
                "Camera Filter",
                options=filter_options,
                index=filter_options.index(active_filter_id) if active_filter_id in filter_options else 0,
                format_func=lambda item_id: (
                    "None"
                    if item_id == "__none__"
                    else str(filter_lookup.get(item_id, {}).get("name", item_id))
                ),
                key="sidebar_active_filter_selector",
            )
            if selected_filter_option != active_filter_id:
                prefs["active_filter_id"] = selected_filter_option
                persist_and_rerun(prefs)

        mount_selection_label = st.segmented_control(
            "Mount Choice",
            options=["EQ", "Alt/Az"],
            default=mount_choice_label(active_mount_choice),
            key="sidebar_active_mount_selector",
        )
        mount_selection_label = str(mount_selection_label or mount_choice_label(active_mount_choice)).strip()
        selected_mount_choice = "eq" if mount_selection_label == "EQ" else "altaz"
        if selected_mount_choice != active_mount_choice:
            prefs["active_mount_choice"] = selected_mount_choice
            persist_and_rerun(prefs)

        if owned_telescope_ids:
            st.markdown("### Equipment")
            if len(owned_telescope_ids) == 1:
                only_telescope = telescope_lookup.get(owned_telescope_ids[0], {})
                only_name = str(only_telescope.get("name", "Selected telescope")).strip() or "Selected telescope"
                st.caption(f"Telescope: {only_name}")
            else:
                selected_telescope_id = st.selectbox(
                    "Telescope",
                    options=owned_telescope_ids,
                    index=(
                        owned_telescope_ids.index(active_telescope_id)
                        if active_telescope_id in owned_telescope_ids
                        else 0
                    ),
                    format_func=lambda item_id: str(telescope_lookup.get(item_id, {}).get("name", item_id)),
                    key="sidebar_active_telescope_selector",
                )
                if selected_telescope_id != active_telescope_id:
                    prefs["active_telescope_id"] = selected_telescope_id
                    persist_and_rerun(prefs)

        theme_container = st.container()
        with theme_container:
            st.markdown("<div class='dso-sidebar-theme-anchor'></div>", unsafe_allow_html=True)
            st.markdown("### Appearance")
            selected_theme_label = st.selectbox(
                "Theme",
                options=list(theme_label_to_id.keys()),
                index=list(theme_label_to_id.values()).index(current_theme),
                key="ui_theme_selector",
            )
    selected_ui_theme = theme_label_to_id[selected_theme_label]
    if selected_ui_theme != current_theme:
        prefs["ui_theme"] = selected_ui_theme
        persist_and_rerun(prefs)
    return selected_ui_theme


def render_explorer_page(
    catalog: pd.DataFrame,
    catalog_meta: dict[str, Any],
    prefs: dict[str, Any],
    temperature_unit: str,
    use_12_hour: bool,
    detail_stack_vertical: bool,
    browser_locale: str | None = None,
    browser_month_day_pattern: str | None = None,
) -> None:
    persistence_notice = st.session_state.get("prefs_persistence_notice", "")
    if persistence_notice:
        st.warning(persistence_notice)

    now_utc = datetime.now(timezone.utc)
    refresh_status_html = (
        "<p class='small-note' style='text-align: right;'>Alt/Az auto-refresh 15m<br>"
        f"Updated: {now_utc.strftime('%Y-%m-%d')} "
        f"{format_display_time(now_utc, use_12_hour=use_12_hour, include_seconds=True)} UTC"
        "</p>"
    )

    site_ids = site_ids_in_order(prefs)
    active_site_id = get_active_site_id(prefs)
    if active_site_id not in site_ids and site_ids:
        active_site_id = site_ids[0]
        set_active_site(prefs, active_site_id)
    st.title("Observation Planner")
    st.caption(f"Catalog rows loaded: {int(catalog_meta.get('row_count', len(catalog)))}")

    location = prefs["location"]
    if not is_location_configured(location):
        st.warning("Observation site location is not set. Open the Sites page to set location and obstructions.")
        return
    location_lat = float(location["lat"])
    location_lon = float(location["lon"])
    weather_forecast_day_offset = resolve_weather_forecast_day_offset(
        max_offset=ASTRONOMY_FORECAST_NIGHTS - 1
    )
    weather_window_start, weather_window_end, weather_tzinfo = weather_forecast_window(
        location_lat,
        location_lon,
        day_offset=weather_forecast_day_offset,
    )
    forecast_title_label = describe_weather_forecast_period(weather_forecast_day_offset)
    forecast_date_text = format_weather_forecast_date(
        weather_window_start,
        browser_locale=browser_locale,
        browser_month_day_pattern=browser_month_day_pattern,
    )
    hourly_title_period = format_hourly_weather_title_period(
        weather_forecast_day_offset,
        weather_window_start,
        browser_locale=browser_locale,
        browser_month_day_pattern=browser_month_day_pattern,
    )
    astronomy_summary = build_astronomy_forecast_summary(
        location_lat,
        location_lon,
        tz_name=weather_tzinfo.key,
        temperature_unit=temperature_unit,
        browser_locale=browser_locale,
        browser_month_day_pattern=browser_month_day_pattern,
        nights=ASTRONOMY_FORECAST_NIGHTS,
    )
    hourly_weather_rows = fetch_hourly_weather(
        lat=location_lat,
        lon=location_lon,
        tz_name=weather_tzinfo.key,
        start_local_iso=weather_window_start.isoformat(),
        end_local_iso=weather_window_end.isoformat(),
        hourly_fields=EXTENDED_FORECAST_HOURLY_FIELDS,
    )
    night_rating_details = compute_night_rating_details(
        hourly_weather_rows,
        temperature_unit=temperature_unit,
    )
    condition_tips_title = format_condition_tips_title(
        weather_forecast_day_offset,
        weather_window_start,
        rating_emoji=(str(night_rating_details.get("emoji", "")).strip() if night_rating_details else None),
    )
    condition_tips_tooltip = format_night_rating_tooltip(night_rating_details)
    weather_matrix, weather_tooltips, weather_indicators = build_hourly_weather_matrix(
        hourly_weather_rows,
        use_12_hour=use_12_hour,
        temperature_unit=temperature_unit,
    )
    selected_summary_row: dict[str, Any] | None = None
    if not astronomy_summary.empty:
        for _, summary_row in astronomy_summary.iterrows():
            row_day_offset = normalize_weather_forecast_day_offset(
                summary_row.get("day_offset", 0),
                max_offset=ASTRONOMY_FORECAST_NIGHTS - 1,
            )
            if row_day_offset == weather_forecast_day_offset:
                selected_summary_row = dict(summary_row)
                break

    with st.container(border=True):
        st.markdown(f"### Site Conditions for {resolve_location_display_name(location)}")

        if detail_stack_vertical:
            five_day_container = st.container()
            hourly_container = st.container()
            conditions_container = st.container()
        else:
            top_cols = st.columns([3, 4, 3], gap="medium")
            five_day_container = top_cols[0]
            hourly_container = top_cols[1]
            conditions_container = top_cols[2]

        with five_day_container:
            st.markdown("5-night astronomy forecast.")
            render_astronomy_forecast_summary(
                astronomy_summary,
                temperature_unit=temperature_unit,
                selected_day_offset=weather_forecast_day_offset,
            )
            st.caption("Click any row to set the active weather night across the page.")
            st.caption(WEATHER_ALERT_INDICATOR_LEGEND_CAPTION)
            st.markdown(night_time_clarity_scale_legend_html(), unsafe_allow_html=True)

        with hourly_container:
            st.markdown(
                f"Hourly weather for {hourly_title_period}."
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

        with conditions_container:
            with st.container(border=True):
                render_condition_tips_panel(
                    title=condition_tips_title,
                    title_tooltip=condition_tips_tooltip,
                    period_label=forecast_title_label,
                    forecast_date_text=forecast_date_text,
                    hourly_weather_rows=hourly_weather_rows,
                    summary_row=selected_summary_row,
                    temperature_unit=temperature_unit,
                    use_12_hour=use_12_hour,
                )

    selected_row = resolve_selected_row(catalog)
    active_preview_list_ids = get_list_ids(prefs, get_active_preview_list_id(prefs))
    if selected_row is not None or bool(active_preview_list_ids):
        render_detail_panel(
            selected=selected_row,
            catalog=catalog,
            prefs=prefs,
            temperature_unit=temperature_unit,
            use_12_hour=use_12_hour,
            detail_stack_vertical=detail_stack_vertical,
            weather_forecast_day_offset=weather_forecast_day_offset,
        )
    st.markdown(refresh_status_html, unsafe_allow_html=True)


def main() -> None:
    st_autorefresh(interval=900_000, key="altaz_refresh")

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
    sync_active_site_to_legacy_fields(prefs)
    st.session_state["prefs"] = prefs

    if not is_location_configured(prefs.get("location")):
        if not bool(st.session_state.get("initial_location_fallback_attempted", False)):
            browser_applied = False
            initial_geo_payload = get_geolocation(component_key="initial_site_browser_geo")
            if isinstance(initial_geo_payload, dict):
                coords = initial_geo_payload.get("coords")
                if isinstance(coords, dict):
                    apply_browser_geolocation_payload(prefs, initial_geo_payload)
                    browser_applied = is_location_configured(prefs.get("location"))

            st.session_state["initial_location_fallback_attempted"] = True
            if not browser_applied:
                resolved = approximate_location_from_ip()
                if resolved:
                    resolved_label, kept_site_name = apply_resolved_location(prefs, resolved)
                    st.session_state["prefs"] = prefs
                    save_preferences(prefs)
                    st.session_state["location_notice"] = (
                        f"Approximate location applied: {resolved_label}. Site name unchanged."
                        if kept_site_name
                        else f"Approximate location applied: {resolved_label}."
                    )
                else:
                    st.session_state["location_notice"] = (
                        "Location not set. Open the Sites page to set it manually or via browser geolocation (IP estimate is fallback)."
                    )

    theme_label_to_id = {
        "Light": UI_THEME_LIGHT,
        "Blue Light": UI_THEME_BLUE_LIGHT,
        "Dark": UI_THEME_DARK,
        "Aura Dracula": UI_THEME_AURA_DRACULA,
        "Monokai ST3": UI_THEME_MONOKAI_ST3,
    }
    current_theme = str(prefs.get("ui_theme", UI_THEME_LIGHT)).strip().lower()
    if current_theme not in set(theme_label_to_id.values()):
        current_theme = UI_THEME_LIGHT
    apply_ui_theme_css(current_theme)

    browser_language_raw = eval_js_hidden("window.navigator.language", key="browser_language_pref")
    if isinstance(browser_language_raw, str) and browser_language_raw.strip():
        st.session_state["browser_language"] = browser_language_raw.strip()
    browser_language = st.session_state.get("browser_language")
    browser_hour_cycle_raw = eval_js_hidden(
        (
            "new Intl.DateTimeFormat(window.navigator.language, "
            "{hour: 'numeric', minute: '2-digit'}).resolvedOptions().hourCycle"
        ),
        key="browser_hour_cycle_pref",
    )
    if isinstance(browser_hour_cycle_raw, str) and browser_hour_cycle_raw.strip():
        st.session_state["browser_hour_cycle"] = browser_hour_cycle_raw.strip()
    browser_hour_cycle = st.session_state.get("browser_hour_cycle")
    browser_month_day_pattern_raw = eval_js_hidden(
        (
            "new Intl.DateTimeFormat(window.navigator.language, {month: 'numeric', day: 'numeric'})"
            ".formatToParts(new Date(2024, 1, 15))"
            ".map((part) => `${part.type}:${part.value}`)"
            ".join('|')"
        ),
        key="browser_month_day_pattern_pref",
    )
    if isinstance(browser_month_day_pattern_raw, str) and browser_month_day_pattern_raw.strip():
        st.session_state["browser_month_day_pattern"] = browser_month_day_pattern_raw.strip()
    browser_month_day_pattern = st.session_state.get("browser_month_day_pattern")
    use_12_hour = resolve_12_hour_clock(browser_language, browser_hour_cycle)
    viewport_width_raw = eval_js_hidden("window.innerWidth", key="browser_viewport_width_probe")
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

    catalog, catalog_meta = load_catalog_app_cached(CATALOG_CACHE_PATH)
    sanitize_saved_lists(catalog=catalog, prefs=prefs)

    def explorer_page() -> None:
        render_explorer_page(
            catalog=catalog,
            catalog_meta=catalog_meta,
            prefs=prefs,
            temperature_unit=effective_temperature_unit,
            use_12_hour=use_12_hour,
            detail_stack_vertical=detail_stack_vertical,
            browser_locale=browser_language,
            browser_month_day_pattern=browser_month_day_pattern,
        )

    def sites_page() -> None:
        render_sites_page(prefs=prefs)

    def equipment_page() -> None:
        render_equipment_page(prefs=prefs)

    def lists_page() -> None:
        render_lists_page(prefs=prefs)

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
                st.Page(sites_page, title="Sites", icon="📍"),
                st.Page(equipment_page, title="Equipment", icon="🧰"),
                st.Page(lists_page, title="Lists", icon="📚"),
                st.Page(settings_page, title="Settings", icon="⚙️"),
            ]
        )
        render_sidebar_active_settings(
            prefs=prefs,
            theme_label_to_id=theme_label_to_id,
        )
        navigation.run()
        return

    selected_page = st.sidebar.radio(
        "Page",
        ["Explorer", "Sites", "Equipment", "Lists", "Settings"],
        key="app_page_selector",
    )
    render_sidebar_active_settings(
        prefs=prefs,
        theme_label_to_id=theme_label_to_id,
    )
    if selected_page == "Sites":
        sites_page()
        return
    if selected_page == "Equipment":
        equipment_page()
        return
    if selected_page == "Lists":
        lists_page()
        return
    if selected_page == "Settings":
        settings_page()
        return
    explorer_page()


if __name__ == "__main__":
    main()
