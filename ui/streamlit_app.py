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
    CLOUD_SYNC_PROVIDER_GOOGLE,
    CLOUD_SYNC_PROVIDER_NONE,
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
from features.condition_tips.ui import render_condition_tips_panel
from features.equipment.context import (
    _normalize_mount_choice,
    build_owned_equipment_context,
    mount_choice_label,
    sync_active_equipment_settings,
)
from features.equipment.page import render_equipment_page
from features.explorer.page import render_explorer_page
from catalog_runtime.catalog_service import (
    canonicalize_designation,
    get_object_by_id,
    load_catalog_from_cache,
    normalize_text,
    search_catalog,
)
from runtime.google_drive_sync import (
    DEFAULT_SETTINGS_FILENAME,
    find_settings_file,
    read_settings_payload,
    upsert_settings_file,
)
from features.lists.list_subsystem import (
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
from features.lists.list_search import subset_by_id_list
from features.lists.page import render_lists_page
from features.settings.page import SettingsPageDeps, render_settings_page as render_settings_page_feature
from features.sites.page import SitesPageDeps, render_sites_page as render_sites_page_feature
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
        from ui.streamlit_modal_compat import Modal
    except Exception:
        Modal = None
try:
    import authlib  # type: ignore
    AUTHLIB_AVAILABLE = True
except Exception:
    AUTHLIB_AVAILABLE = False
from streamlit_js_eval import get_geolocation, streamlit_js_eval
from streamlit_autorefresh import st_autorefresh
from features.target_tips.ui import render_target_tips_panel
from timezonefinder import TimezoneFinder
from runtime.weather_service import (
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_utc_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_user_logged_in() -> bool:
    try:
        return bool(getattr(st.user, "is_logged_in", False))
    except Exception:
        return False


def _get_user_claim(name: str) -> str:
    key = str(name or "").strip()
    if not key:
        return ""
    try:
        value = st.user.get(key)
    except Exception:
        value = None
    return str(value or "").strip()


def _get_google_access_token() -> str:
    if not _is_user_logged_in():
        return ""
    try:
        tokens = getattr(st.user, "tokens", None)
        if tokens is None:
            return ""
        raw = tokens.get("access") or tokens.get("access_token")
    except Exception:
        return ""
    return str(raw or "").strip()


def _session_value_to_jsonable(value: Any, *, depth: int = 0) -> Any:
    from runtime.session_snapshot import (
        _session_value_to_jsonable as _session_value_to_jsonable_feature,
    )

    return _session_value_to_jsonable_feature(value, depth=depth)


def build_syncable_session_snapshot() -> dict[str, Any]:
    from runtime.session_snapshot import (
        build_syncable_session_snapshot as build_syncable_session_snapshot_feature,
    )

    return build_syncable_session_snapshot_feature()


def apply_session_snapshot(snapshot: Any) -> int:
    from runtime.session_snapshot import (
        apply_session_snapshot as apply_session_snapshot_feature,
    )

    return apply_session_snapshot_feature(snapshot)


def _build_cloud_settings_payload(prefs: dict[str, Any], owner_sub: str) -> dict[str, Any]:
    from runtime.session_snapshot import (
        _build_cloud_settings_payload as _build_cloud_settings_payload_feature,
    )

    return _build_cloud_settings_payload_feature(prefs, owner_sub)


def _payload_updated_at_utc(payload: dict[str, Any], *, fallback_modified_time: str = "") -> datetime:
    payload_time = _parse_utc_timestamp(payload.get("updated_at_utc"))
    if payload_time > datetime.fromtimestamp(0, tz=timezone.utc):
        return payload_time
    return _parse_utc_timestamp(fallback_modified_time)


def maybe_sync_prefs_with_google_drive(prefs: dict[str, Any]) -> None:
    from runtime.google_drive_sync_runtime import (
        maybe_sync_prefs_with_google_drive as maybe_sync_prefs_with_google_drive_feature,
    )

    return maybe_sync_prefs_with_google_drive_feature(prefs)


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
def load_catalog_recommendation_features(cache_path: Path) -> pd.DataFrame:
    from runtime.recommendation_cache import (
        load_catalog_recommendation_features as load_catalog_recommendation_features_feature,
    )

    return load_catalog_recommendation_features_feature(cache_path)


def load_site_date_altaz_bundle(
    cache_path: Path,
    *,
    lat: float,
    lon: float,
    window_start: datetime,
    window_end: datetime,
    sample_minutes: int = RECOMMENDATION_CACHE_SAMPLE_MINUTES,
) -> dict[str, Any]:
    from runtime.recommendation_cache import (
        load_site_date_altaz_bundle as load_site_date_altaz_bundle_feature,
    )

    return load_site_date_altaz_bundle_feature(
        cache_path,
        lat=lat,
        lon=lon,
        window_start=window_start,
        window_end=window_end,
        sample_minutes=sample_minutes,
    )


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
    from runtime.weather_mask_cache import (
        load_site_date_weather_mask_bundle as load_site_date_weather_mask_bundle_feature,
    )

    return load_site_date_weather_mask_bundle_feature(
        lat=lat,
        lon=lon,
        tz_name=tz_name,
        window_start_iso=window_start_iso,
        window_end_iso=window_end_iso,
        sample_hour_keys=sample_hour_keys,
        cloud_cover_threshold=cloud_cover_threshold,
    )


WEATHER_MATRIX_ROWS: list[tuple[str, str]] = [
    ("temperature_2m", "Temperature"),
    ("dewpoint_spread", "Humidity"),
    ("cloud_cover", "Cloud Cover"),
    ("visibility", "Visibility"),
    ("wind_gusts_10m", "Wind Gusts"),
]
WEATHER_ALERT_INDICATOR_LEGEND_ITEMS = "❄️ Snow | ⛈️ Rain | ☔ Showers | ⚠️ Low | 🚨 High"
WEATHER_ALERT_INDICATOR_LEGEND_CAPTION = f"Weather Alert Indicator: {WEATHER_ALERT_INDICATOR_LEGEND_ITEMS}"
WEATHER_ALERT_RAIN_PRIORITY = ["❄️", "⛈️", "☔", "🚨", "⚠️"]
WEATHER_ALERT_RAIN_INTERVAL_SECONDS = 5 * 60
WEATHER_ALERT_RAIN_DURATION_SECONDS = 30
WEATHER_ALERT_RAIN_BUCKET_STATE_KEY = "weather_alert_rain_last_bucket"
TARGET_DETAIL_MODAL_OPEN_REQUEST_KEY = "target_detail_modal_open_request"
TARGET_DETAIL_MODAL_LAST_TARGET_STATE_KEY = "target_detail_modal_last_target_id"
GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY = "google_drive_sync_bootstrapped"
GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY = "cloud_sync_pending"
GOOGLE_DRIVE_SYNC_LAST_ACCOUNT_STATE_KEY = "google_drive_sync_last_account_sub"
GOOGLE_DRIVE_SYNC_LAST_APPLIED_TOKEN_STATE_KEY = "google_drive_sync_last_applied_token"
GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY = "google_drive_sync_manual_action"
GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY = "google_drive_sync_last_action"
GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY = "google_drive_sync_last_compare_summary"
GOOGLE_DRIVE_SYNC_LAST_REMOTE_FILE_MODIFIED_STATE_KEY = "google_drive_sync_last_remote_file_modified_utc"
GOOGLE_DRIVE_SYNC_LAST_REMOTE_PAYLOAD_UPDATED_STATE_KEY = "google_drive_sync_last_remote_payload_updated_utc"
GOOGLE_DRIVE_SYNC_STATE_STATE_KEY = "google_drive_sync_state"
GOOGLE_DRIVE_SYNC_DEFERRED_ACTION_STATE_KEY = "google_drive_sync_deferred_action"
GOOGLE_DRIVE_SYNC_MERGE_CANDIDATE_STATE_KEY = "google_drive_sync_merge_candidate"
GOOGLE_DRIVE_SYNC_MERGE_RESOLUTION_ACTION_STATE_KEY = "google_drive_sync_merge_resolution_action"
GOOGLE_DRIVE_SYNC_MERGE_SELECTION_STATE_KEY = "google_drive_sync_merge_selection"
GOOGLE_DRIVE_SYNC_SESSION_SNAPSHOT_VERSION = 1
WEATHER_FORECAST_PERIOD_STATE_KEY = "weather_forecast_period"
WEATHER_FORECAST_PERIOD_TONIGHT = "tonight"
WEATHER_FORECAST_PERIOD_TOMORROW = "tomorrow"
WEATHER_FORECAST_DAY_OFFSET_STATE_KEY = "weather_forecast_day_offset"
ASTRONOMY_FORECAST_NIGHTS = 10
NIGHT_RATING_FACTOR_WEIGHTS: dict[str, float] = {
    "precipitation": 0.05,
    "precip_probability": 0.10,
    "cloud_coverage": 0.45,
    "visibility": 0.15,
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
EQ_MOUNT_WARNING_MAX_ALT_DEG = 30.0
ALTAZ_MOUNT_WARNING_MIN_ALT_DEG = 80.0
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


def mount_warning_zone_altitude_bounds(mount_choice: Any) -> tuple[float, float, str] | None:
    normalized = str(mount_choice or "").strip().lower()
    if normalized == "eq":
        return (0.0, float(EQ_MOUNT_WARNING_MAX_ALT_DEG), "EQ warning zone (<30 deg)")
    if normalized == "altaz":
        return (float(ALTAZ_MOUNT_WARNING_MIN_ALT_DEG), 90.0, "Alt/Az warning zone (>80 deg)")
    return None


def mount_warning_zone_plot_style() -> tuple[str, str]:
    # Light gray fill to indicate mount-specific caution zones without overpowering the target paths.
    return ("rgba(148, 163, 184, 0.16)", "rgba(148, 163, 184, 0.45)")


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
    from features.explorer.night_rating import (
        compute_night_rating_details as compute_night_rating_details_feature,
    )

    return compute_night_rating_details_feature(
        hourly_weather_rows,
        temperature_unit=temperature_unit,
    )


def compute_night_rating(
    hourly_weather_rows: list[dict[str, Any]],
    *,
    temperature_unit: str,
) -> tuple[int, str] | None:
    from features.explorer.night_rating import (
        compute_night_rating as compute_night_rating_feature,
    )

    return compute_night_rating_feature(
        hourly_weather_rows,
        temperature_unit=temperature_unit,
    )


def format_night_rating_tooltip(rating_details: dict[str, Any] | None) -> str:
    from features.explorer.night_rating import (
        format_night_rating_tooltip as format_night_rating_tooltip_feature,
    )

    return format_night_rating_tooltip_feature(rating_details)


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
    from features.sites.site_state import (
        default_site_definition as default_site_definition_feature,
    )

    return default_site_definition_feature(name)


def site_ids_in_order(prefs: dict[str, Any]) -> list[str]:
    from features.sites.site_state import (
        site_ids_in_order as site_ids_in_order_feature,
    )

    return site_ids_in_order_feature(prefs)


def get_active_site_id(prefs: dict[str, Any]) -> str:
    from features.sites.site_state import (
        get_active_site_id as get_active_site_id_feature,
    )

    return get_active_site_id_feature(prefs)


def get_site_definition(prefs: dict[str, Any], site_id: str) -> dict[str, Any]:
    from features.sites.site_state import (
        get_site_definition as get_site_definition_feature,
    )

    return get_site_definition_feature(prefs, site_id)


def sync_active_site_to_legacy_fields(prefs: dict[str, Any]) -> None:
    from features.sites.site_state import (
        sync_active_site_to_legacy_fields as sync_active_site_to_legacy_fields_feature,
    )

    return sync_active_site_to_legacy_fields_feature(prefs)


def persist_legacy_fields_to_active_site(prefs: dict[str, Any]) -> None:
    from features.sites.site_state import (
        persist_legacy_fields_to_active_site as persist_legacy_fields_to_active_site_feature,
    )

    return persist_legacy_fields_to_active_site_feature(prefs)


def set_active_site(prefs: dict[str, Any], site_id: str) -> bool:
    from features.sites.site_state import (
        set_active_site as set_active_site_feature,
    )

    return set_active_site_feature(prefs, site_id)


def duplicate_site(prefs: dict[str, Any], site_id: str) -> str | None:
    from features.sites.site_state import (
        duplicate_site as duplicate_site_feature,
    )

    return duplicate_site_feature(prefs, site_id)


def create_site(prefs: dict[str, Any], name: str | None = None) -> str | None:
    from features.sites.site_state import (
        create_site as create_site_feature,
    )

    return create_site_feature(prefs, name=name)


def delete_site(prefs: dict[str, Any], site_id: str) -> bool:
    from features.sites.site_state import (
        delete_site as delete_site_feature,
    )

    return delete_site_feature(prefs, site_id)


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
    from features.sites.location_actions import (
        apply_resolved_location as apply_resolved_location_feature,
    )

    return apply_resolved_location_feature(prefs, resolved_location)


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
    selected_metadata: dict[str, Any] | None = None,
    now_local: pd.Timestamp | datetime | None = None,
    row_order_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    from features.explorer.summary_rows import (
        build_sky_position_summary_rows as build_sky_position_summary_rows_feature,
    )

    return build_sky_position_summary_rows_feature(
        selected_id=selected_id,
        selected_label=selected_label,
        selected_type_group=selected_type_group,
        selected_color=selected_color,
        selected_events=selected_events,
        selected_track=selected_track,
        overlay_tracks=overlay_tracks,
        list_member_ids=list_member_ids,
        selected_metadata=selected_metadata,
        now_local=now_local,
        row_order_ids=row_order_ids,
    )


def render_sky_position_summary_table(*args: Any, **kwargs: Any) -> None:
    from features.explorer.summary_table import render_sky_position_summary_table as render_sky_position_summary_table_feature

    return render_sky_position_summary_table_feature(*args, **kwargs)

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
        return f"{value:.2f}".rstrip("0").rstrip(".")

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


def compute_hourly_target_recommendations(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import compute_hourly_target_recommendations as compute_hourly_target_recommendations_feature

    return compute_hourly_target_recommendations_feature(*args, **kwargs)

def render_target_recommendations(
    catalog: pd.DataFrame,
    prefs: dict[str, Any],
    active_preview_list_ids: list[str],
    window_start: datetime,
    window_end: datetime,
    tzinfo: Any,
    use_12_hour: bool,
    weather_forecast_day_offset: int = 0,
) -> None:
    from features.explorer.recommendations import render_target_recommendations as render_target_recommendations_feature

    return render_target_recommendations_feature(
        catalog=catalog,
        prefs=prefs,
        active_preview_list_ids=active_preview_list_ids,
        window_start=window_start,
        window_end=window_end,
        tzinfo=tzinfo,
        use_12_hour=use_12_hour,
        weather_forecast_day_offset=weather_forecast_day_offset,
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


def dew_risk_scale_legend_html() -> str:
    swatches: list[str] = []
    spread_values = list(range(6, -1, -1))
    for idx, spread_value in enumerate(spread_values):
        color = _interpolate_color_stops(
            float(spread_value),
            [
                (0.0, "#FDBA74"),  # dew-prone (orange)
                (6.0, "#FFFFFF"),  # dry (white)
            ],
        )
        margin_right = "0.12rem" if idx < (len(spread_values) - 1) else "0"
        swatches.append(
            "<span style='display:inline-block; width:0.58rem; height:0.58rem; border-radius:2px; "
            f"margin-right:{margin_right}; border:1px solid rgba(17,24,39,0.28); background:{color};'></span>"
        )

    return (
        "<div style='font-size:0.88rem; color:#6b7280; margin:0.02rem 0 0.12rem;'>"
        "💧 dew risk scale: "
        "<span style='margin-right:0.2rem;'>Dry</span>"
        f"{''.join(swatches)}"
        "<span style='margin-left:0.12rem;'>Dew</span>"
        "</div>"
    )


def full_moon_scale_legend_html(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        full_moon_scale_legend_html as full_moon_scale_legend_html_feature,
    )

    return full_moon_scale_legend_html_feature(*args, **kwargs)


def site_conditions_legends_table_html(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        site_conditions_legends_table_html as site_conditions_legends_table_html_feature,
    )

    return site_conditions_legends_table_html_feature(*args, **kwargs)


def cloud_cover_cell_style(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        cloud_cover_cell_style as cloud_cover_cell_style_feature,
    )

    return cloud_cover_cell_style_feature(*args, **kwargs)


def clarity_percentage_cell_style(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        clarity_percentage_cell_style as clarity_percentage_cell_style_feature,
    )

    return clarity_percentage_cell_style_feature(*args, **kwargs)


def visibility_condition_cell_style(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        visibility_condition_cell_style as visibility_condition_cell_style_feature,
    )

    return visibility_condition_cell_style_feature(*args, **kwargs)


def dewpoint_spread_cell_style(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        dewpoint_spread_cell_style as dewpoint_spread_cell_style_feature,
    )

    return dewpoint_spread_cell_style_feature(*args, **kwargs)


def _temperature_f_from_display_value(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        _temperature_f_from_display_value as _temperature_f_from_display_value_feature,
    )

    return _temperature_f_from_display_value_feature(*args, **kwargs)


def temperature_cell_style(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        temperature_cell_style as temperature_cell_style_feature,
    )

    return temperature_cell_style_feature(*args, **kwargs)


def format_visibility_value(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        format_visibility_value as format_visibility_value_feature,
    )

    return format_visibility_value_feature(*args, **kwargs)


def format_visibility_condition(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        format_visibility_condition as format_visibility_condition_feature,
    )

    return format_visibility_condition_feature(*args, **kwargs)


def _dewpoint_spread_celsius(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        _dewpoint_spread_celsius as _dewpoint_spread_celsius_feature,
    )

    return _dewpoint_spread_celsius_feature(*args, **kwargs)


def format_weather_matrix_value(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        format_weather_matrix_value as format_weather_matrix_value_feature,
    )

    return format_weather_matrix_value_feature(*args, **kwargs)


def _positive_float(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        _positive_float as _positive_float_feature,
    )

    return _positive_float_feature(*args, **kwargs)


def _nonnegative_float(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        _nonnegative_float as _nonnegative_float_feature,
    )

    return _nonnegative_float_feature(*args, **kwargs)


def resolve_weather_alert_indicator(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        resolve_weather_alert_indicator as resolve_weather_alert_indicator_feature,
    )

    return resolve_weather_alert_indicator_feature(*args, **kwargs)


def build_weather_alert_indicator_html(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        build_weather_alert_indicator_html as build_weather_alert_indicator_html_feature,
    )

    return build_weather_alert_indicator_html_feature(*args, **kwargs)


def collect_night_weather_alert_emojis(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        collect_night_weather_alert_emojis as collect_night_weather_alert_emojis_feature,
    )

    return collect_night_weather_alert_emojis_feature(*args, **kwargs)


def normalize_hour_key(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        normalize_hour_key as normalize_hour_key_feature,
    )

    return normalize_hour_key_feature(*args, **kwargs)


def build_hourly_weather_maps(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        build_hourly_weather_maps as build_hourly_weather_maps_feature,
    )

    return build_hourly_weather_maps_feature(*args, **kwargs)


def build_hourly_weather_matrix(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        build_hourly_weather_matrix as build_hourly_weather_matrix_feature,
    )

    return build_hourly_weather_matrix_feature(*args, **kwargs)


def render_hourly_weather_matrix(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        render_hourly_weather_matrix as render_hourly_weather_matrix_feature,
    )

    return render_hourly_weather_matrix_feature(*args, **kwargs)


def _extract_finite_weather_values(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import (
        _extract_finite_weather_values as _extract_finite_weather_values_feature,
    )

    return _extract_finite_weather_values_feature(*args, **kwargs)


def build_astronomy_forecast_summary(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import build_astronomy_forecast_summary as build_astronomy_forecast_summary_feature

    return build_astronomy_forecast_summary_feature(*args, **kwargs)

def render_astronomy_forecast_summary(*args: Any, **kwargs: Any):
    from features.explorer.forecast_panels import render_astronomy_forecast_summary as render_astronomy_forecast_summary_feature

    return render_astronomy_forecast_summary_feature(*args, **kwargs)

def build_path_hovertext(*args: Any, **kwargs: Any):
    from features.explorer.plots import build_path_hovertext as build_path_hovertext_feature

    return build_path_hovertext_feature(*args, **kwargs)


def split_path_on_az_wrap(*args: Any, **kwargs: Any):
    from features.explorer.plots import split_path_on_az_wrap as split_path_on_az_wrap_feature

    return split_path_on_az_wrap_feature(*args, **kwargs)


def iter_labeled_events(*args: Any, **kwargs: Any):
    from features.explorer.plots import iter_labeled_events as iter_labeled_events_feature

    return iter_labeled_events_feature(*args, **kwargs)


def sample_direction_indices(*args: Any, **kwargs: Any):
    from features.explorer.plots import sample_direction_indices as sample_direction_indices_feature

    return sample_direction_indices_feature(*args, **kwargs)


def direction_marker_segments_cartesian(*args: Any, **kwargs: Any):
    from features.explorer.plots import (
        direction_marker_segments_cartesian as direction_marker_segments_cartesian_feature,
    )

    return direction_marker_segments_cartesian_feature(*args, **kwargs)


def direction_marker_segments_radial(*args: Any, **kwargs: Any):
    from features.explorer.plots import (
        direction_marker_segments_radial as direction_marker_segments_radial_feature,
    )

    return direction_marker_segments_radial_feature(*args, **kwargs)


def track_event_index(*args: Any, **kwargs: Any):
    from features.explorer.plots import track_event_index as track_event_index_feature

    return track_event_index_feature(*args, **kwargs)


def endpoint_marker_segments_cartesian(*args: Any, **kwargs: Any):
    from features.explorer.plots import (
        endpoint_marker_segments_cartesian as endpoint_marker_segments_cartesian_feature,
    )

    return endpoint_marker_segments_cartesian_feature(*args, **kwargs)


def endpoint_marker_segments_radial(*args: Any, **kwargs: Any):
    from features.explorer.plots import (
        endpoint_marker_segments_radial as endpoint_marker_segments_radial_feature,
    )

    return endpoint_marker_segments_radial_feature(*args, **kwargs)


def terminal_segment_from_path_arrays(*args: Any, **kwargs: Any):
    from features.explorer.plots import terminal_segment_from_path_arrays as terminal_segment_from_path_arrays_feature

    return terminal_segment_from_path_arrays_feature(*args, **kwargs)


def obstruction_step_profile(*args: Any, **kwargs: Any):
    from features.explorer.plots import obstruction_step_profile as obstruction_step_profile_feature

    return obstruction_step_profile_feature(*args, **kwargs)


def visible_track_segments(*args: Any, **kwargs: Any):
    from features.explorer.plots import visible_track_segments as visible_track_segments_feature

    return visible_track_segments_feature(*args, **kwargs)


def distribute_non_overlapping_values(*args: Any, **kwargs: Any):
    from features.explorer.plots import (
        distribute_non_overlapping_values as distribute_non_overlapping_values_feature,
    )

    return distribute_non_overlapping_values_feature(*args, **kwargs)


def build_unobstructed_altitude_area_plot(*args: Any, **kwargs: Any):
    from features.explorer.plots import build_unobstructed_altitude_area_plot as build_unobstructed_altitude_area_plot_feature

    return build_unobstructed_altitude_area_plot_feature(*args, **kwargs)

def build_path_plot(*args: Any, **kwargs: Any):
    from features.explorer.plots import build_path_plot as build_path_plot_feature

    return build_path_plot_feature(*args, **kwargs)

def build_path_plot_radial(*args: Any, **kwargs: Any):
    from features.explorer.plots import build_path_plot_radial as build_path_plot_radial_feature

    return build_path_plot_radial_feature(*args, **kwargs)

def build_night_plot(*args: Any, **kwargs: Any):
    from features.explorer.plots import build_night_plot as build_night_plot_feature

    return build_night_plot_feature(*args, **kwargs)

def resolve_manual_location(query: str) -> dict[str, Any] | None:
    from runtime.location_resolution import (
        resolve_manual_location as resolve_manual_location_feature,
    )

    return resolve_manual_location_feature(query)


def reverse_geocode_label(lat: float, lon: float) -> str:
    from runtime.location_resolution import (
        reverse_geocode_label as reverse_geocode_label_feature,
    )

    return reverse_geocode_label_feature(lat, lon)


def apply_browser_geolocation_payload(prefs: dict[str, Any], payload: Any) -> None:
    from features.sites.location_actions import (
        apply_browser_geolocation_payload as apply_browser_geolocation_payload_feature,
    )

    return apply_browser_geolocation_payload_feature(prefs, payload)


def approximate_location_from_ip() -> dict[str, Any] | None:
    from runtime.location_resolution import (
        approximate_location_from_ip as approximate_location_from_ip_feature,
    )

    return approximate_location_from_ip_feature()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
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


def build_legacy_survey_cutout_urls(
    *,
    ra_deg: float,
    dec_deg: float,
    fov_width_deg: float,
    fov_height_deg: float,
    layer: str = "dr8",
    max_pixels: int = 512,
) -> dict[str, Any] | None:
    try:
        ra = float(ra_deg)
        dec = float(dec_deg)
        fov_width = float(fov_width_deg)
        fov_height = float(fov_height_deg)
    except (TypeError, ValueError):
        return None

    if not (np.isfinite(ra) and np.isfinite(dec) and np.isfinite(fov_width) and np.isfinite(fov_height)):
        return None
    if fov_width <= 0.0 or fov_height <= 0.0:
        return None

    pixel_limit = int(max(64, min(int(max_pixels), 3000)))
    aspect_ratio = fov_width / fov_height if fov_height > 0.0 else 1.0
    if not np.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
        aspect_ratio = 1.0

    if aspect_ratio >= 1.0:
        width_px = pixel_limit
        height_px = max(64, min(pixel_limit, int(round(pixel_limit / aspect_ratio))))
    else:
        height_px = pixel_limit
        width_px = max(64, min(pixel_limit, int(round(pixel_limit * aspect_ratio))))

    pixscale_arcsec = max(
        (fov_width * 3600.0) / float(width_px),
        (fov_height * 3600.0) / float(height_px),
    )
    if not np.isfinite(pixscale_arcsec) or pixscale_arcsec <= 0.0:
        return None

    base_cutout_url = "https://www.legacysurvey.org/viewer/jpeg-cutout"
    base_viewer_url = "https://www.legacysurvey.org/viewer"
    cutout_url = (
        f"{base_cutout_url}?ra={ra:.6f}&dec={dec:.6f}&layer={layer}"
        f"&width={width_px}&height={height_px}&pixscale={pixscale_arcsec:.6f}"
    )
    viewer_url = f"{base_viewer_url}?ra={ra:.6f}&dec={dec:.6f}&layer={layer}"
    return {
        "image_url": cutout_url,
        "viewer_url": viewer_url,
        "width_px": width_px,
        "height_px": height_px,
        "pixscale_arcsec": pixscale_arcsec,
        "fov_width_deg": fov_width,
        "fov_height_deg": fov_height,
    }


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
    from features.sites.settings_sections import (
        render_location_settings_section as render_location_settings_section_feature,
    )

    return render_location_settings_section_feature(prefs)


def render_obstructions_settings_section(prefs: dict[str, Any]) -> None:
    from features.sites.settings_sections import (
        render_obstructions_settings_section as render_obstructions_settings_section_feature,
    )

    return render_obstructions_settings_section_feature(prefs)


def render_sites_page(prefs: dict[str, Any]) -> None:
    render_sites_page_feature(
        prefs,
        deps=SitesPageDeps(
            default_site_id=DEFAULT_SITE_ID,
            site_ids_in_order=site_ids_in_order,
            get_active_site_id=get_active_site_id,
            set_active_site=set_active_site,
            get_site_definition=get_site_definition,
            resolve_location_source_badge=resolve_location_source_badge,
            get_site_name=get_site_name,
            create_site=create_site,
            persist_and_rerun=persist_and_rerun,
            duplicate_site=duplicate_site,
            delete_site=delete_site,
            sync_active_site_to_legacy_fields=sync_active_site_to_legacy_fields,
            render_location_settings_section=render_location_settings_section,
            render_obstructions_settings_section=render_obstructions_settings_section,
        ),
    )


def render_settings_page(catalog_meta: dict[str, Any], prefs: dict[str, Any], browser_locale: str | None) -> None:
    render_settings_page_feature(
        catalog_meta,
        prefs,
        browser_locale,
        deps=SettingsPageDeps(
            temperature_unit_options=TEMPERATURE_UNIT_OPTIONS,
            persist_and_rerun=persist_and_rerun,
            resolve_temperature_unit=resolve_temperature_unit,
            is_user_logged_in=_is_user_logged_in,
            authlib_available=AUTHLIB_AVAILABLE,
            get_user_claim=_get_user_claim,
            cloud_sync_provider_google=CLOUD_SYNC_PROVIDER_GOOGLE,
            cloud_sync_provider_none=CLOUD_SYNC_PROVIDER_NONE,
            get_google_access_token=_get_google_access_token,
            google_drive_sync_pending_state_key=GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY,
            google_drive_sync_bootstrap_state_key=GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY,
            google_drive_sync_last_remote_file_modified_state_key=GOOGLE_DRIVE_SYNC_LAST_REMOTE_FILE_MODIFIED_STATE_KEY,
            google_drive_sync_last_remote_payload_updated_state_key=GOOGLE_DRIVE_SYNC_LAST_REMOTE_PAYLOAD_UPDATED_STATE_KEY,
            google_drive_sync_last_action_state_key=GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY,
            google_drive_sync_last_compare_summary_state_key=GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY,
            google_drive_sync_manual_action_state_key=GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY,
            google_drive_sync_state_state_key=GOOGLE_DRIVE_SYNC_STATE_STATE_KEY,
            google_drive_sync_deferred_action_state_key=GOOGLE_DRIVE_SYNC_DEFERRED_ACTION_STATE_KEY,
            google_drive_sync_merge_candidate_state_key=GOOGLE_DRIVE_SYNC_MERGE_CANDIDATE_STATE_KEY,
            google_drive_sync_merge_resolution_action_state_key=GOOGLE_DRIVE_SYNC_MERGE_RESOLUTION_ACTION_STATE_KEY,
            google_drive_sync_merge_selection_state_key=GOOGLE_DRIVE_SYNC_MERGE_SELECTION_STATE_KEY,
            catalog_cache_path_display=str(CATALOG_CACHE_PATH),
            build_settings_export_payload=build_settings_export_payload,
            parse_settings_import_payload=parse_settings_import_payload,
            site_ids_in_order=site_ids_in_order,
            list_ids_in_order=list_ids_in_order,
        ),
    )


def render_detail_panel(
    selected: pd.Series | None,
    catalog: pd.DataFrame,
    prefs: dict[str, Any],
    temperature_unit: str,
    use_12_hour: bool,
    detail_stack_vertical: bool,
    weather_forecast_day_offset: int = 0,
) -> None:
    from features.explorer.detail_panel import render_detail_panel as render_detail_panel_feature

    return render_detail_panel_feature(
        selected=selected,
        catalog=catalog,
        prefs=prefs,
        temperature_unit=temperature_unit,
        use_12_hour=use_12_hour,
        detail_stack_vertical=detail_stack_vertical,
        weather_forecast_day_offset=weather_forecast_day_offset,
    )


def _render_explorer_page_impl(*args: Any, **kwargs: Any) -> None:
    from features.explorer.page_impl import _render_explorer_page_impl as _render_explorer_page_impl_feature

    return _render_explorer_page_impl_feature(*args, **kwargs)


def render_sidebar_active_settings(*args: Any, **kwargs: Any):
    from features.explorer.detail_panel import (
        render_sidebar_active_settings as render_sidebar_active_settings_feature,
    )

    return render_sidebar_active_settings_feature(*args, **kwargs)


def main() -> None:
    from ui.app_main import main as app_main

    return app_main()


if __name__ == "__main__":
    main()
