from __future__ import annotations

import copy
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
from streamlit_js_eval import get_browser_language, get_geolocation
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

STATE_PATH = Path(".state/preferences.json")
CATALOG_SEED_PATH = Path("data/dso_catalog_seed.csv")
CATALOG_CACHE_PATH = Path("data/dso_catalog_cache.parquet")
CATALOG_META_PATH = Path("data/dso_catalog_cache_meta.json")

TEMPERATURE_UNIT_OPTIONS = {
    "Auto (browser)": "auto",
    "Fahrenheit": "f",
    "Celsius": "c",
}
FAHRENHEIT_COUNTRY_CODES = {"US", "BS", "BZ", "KY", "PW", "FM", "MH", "LR"}


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


def load_preferences() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return default_preferences()

    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return ensure_preferences_shape(payload)
    except (json.JSONDecodeError, OSError):
        return default_preferences()


def save_preferences(prefs: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(prefs, indent=2), encoding="utf-8")


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


def build_path_plot(track: pd.DataFrame, events: dict[str, pd.Series | None], obstructions: dict[str, float]) -> go.Figure:
    path_x, path_y, path_custom = split_path_on_az_wrap(track)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=path_x,
            y=path_y,
            mode="lines",
            name="Altitude path",
            line={"width": 3},
            customdata=path_custom,
            hovertemplate="Az %{x:.1f} deg<br>Alt %{y:.1f} deg<br>Time %{customdata[0]}<br>Dir %{customdata[1]}<br>Obs %{customdata[2]:.0f} deg<extra></extra>",
        )
    )

    obstruction_line_x = [wind_bin_center(direction) for direction in WIND16] + [360.0]
    obstruction_line_y = [obstructions.get(direction, 20.0) for direction in WIND16]
    obstruction_line_y.append(obstruction_line_y[0])

    fig.add_trace(
        go.Scatter(
            x=obstruction_line_x,
            y=obstruction_line_y,
            mode="lines",
            name="Obstructed region",
            line={"width": 0},
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.20)",
            hoverinfo="skip",
        )
    )

    for event_name, event in events.items():
        if event is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=[event["az"]],
                y=[event["alt"]],
                mode="markers+text",
                name=event_name.replace("_", " ").title(),
                text=[event_name.replace("_", " ").title()],
                textposition="top center",
                marker={"size": 9},
                showlegend=False,
                hovertemplate=f"{event_name.replace('_', ' ').title()} at {pd.Timestamp(event['time_local']).strftime('%H:%M')}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Sky Position",
        height=330,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
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
) -> go.Figure:
    fig = go.Figure()

    obstruction_theta = [wind_bin_center(direction) for direction in WIND16] + [360.0]
    obstruction_alt = [float(obstructions.get(direction, 20.0)) for direction in WIND16]
    obstruction_alt.append(obstruction_alt[0])

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
            line={"width": 0},
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
            fillcolor="rgba(239, 68, 68, 0.20)",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            theta=track["az"],
            r=track_r,
            mode="lines",
            name="Altitude path",
            line={"width": 3},
            customdata=np.stack(
                [track["time_local"].dt.strftime("%H:%M"), track["wind16"], track["min_alt_required"], track["alt"]],
                axis=-1,
            ),
            hovertemplate="Az %{theta:.1f} deg<br>Alt %{customdata[3]:.1f} deg<br>Time %{customdata[0]}<br>Dir %{customdata[1]}<br>Obs %{customdata[2]:.0f} deg<extra></extra>",
        )
    )

    for event_name, event in events.items():
        if event is None:
            continue
        fig.add_trace(
            go.Scatterpolar(
                theta=[event["az"]],
                r=[
                    max(0.0, min(90.0, 90.0 - float(event["alt"])))
                    if dome_view
                    else max(0.0, min(90.0, float(event["alt"])))
                ],
                mode="markers+text",
                text=[event_name.replace("_", " ").title()],
                textposition="top center",
                marker={"size": 9},
                showlegend=False,
                hovertemplate=f"{event_name.replace('_', ' ').title()} at {pd.Timestamp(event['time_local']).strftime('%H:%M')}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Sky Position",
        height=330,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
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


def build_night_plot(track: pd.DataFrame, temperature_by_hour: dict[str, float], temperature_unit: str) -> go.Figure:
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
    fig.update_layout(
        title="Hourly forecast",
        height=300,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        barmode="stack",
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


def render_main_tabs(results: pd.DataFrame, favorites: pd.DataFrame, set_list: pd.DataFrame, prefs: dict[str, Any]) -> None:
    with st.container(border=True):
        st.subheader("Targets")
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


def render_detail_panel(selected: pd.Series | None, prefs: dict[str, Any], temperature_unit: str) -> None:
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
            st.markdown(
                "\n".join(
                    [
                        f"**RA / Dec:** {float(selected['ra_deg']):.4f} deg / {float(selected['dec_deg']):.4f} deg",
                        f"**Constellation:** {selected.get('constellation') or '-'}",
                        f"**Alt / Az (now):** {float(selected['alt_now']):.1f} deg / {float(selected['az_now']):.1f} deg",
                        f"**16-wind:** {selected['wind16']}",
                    ]
                )
            )

        location = prefs["location"]
        window_start, window_end, tzinfo = tonight_window(location["lat"], location["lon"])
        track = compute_track(
            ra_deg=float(selected["ra_deg"]),
            dec_deg=float(selected["dec_deg"]),
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
            key=f"path_style_{target_id}",
        )
        if path_style == "Radial":
            dome_view = st.toggle("Dome View", value=True, key="dome_view_enabled")
            path_figure = build_path_plot_radial(
                track=track,
                events=events,
                obstructions=prefs["obstructions"],
                dome_view=dome_view,
            )
        else:
            path_figure = build_path_plot(track=track, events=events, obstructions=prefs["obstructions"])

        st.plotly_chart(
            path_figure,
            use_container_width=True,
        )

        temperatures = fetch_hourly_temperatures(
            lat=float(location["lat"]),
            lon=float(location["lon"]),
            tz_name=tzinfo.key,
            start_local_iso=window_start.isoformat(),
            end_local_iso=window_end.isoformat(),
        )
        st.plotly_chart(
            build_night_plot(track=track, temperature_by_hour=temperatures, temperature_unit=temperature_unit),
            use_container_width=True,
        )


def main() -> None:
    st_autorefresh(interval=60_000, key="altaz_refresh")

    if "prefs" not in st.session_state:
        st.session_state["prefs"] = load_preferences()

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

    query = st.text_input(
        "Search (M / NGC / IC / Sh2 + common names)",
        placeholder="M31, NGC 7000, IC1805, Sh2-132, Orion Nebula",
    )

    filtered = search_catalog(catalog, query)
    location = prefs["location"]
    results_enriched = compute_altaz_now(filtered, lat=float(location["lat"]), lon=float(location["lon"]))
    favorites_enriched = compute_altaz_now(
        subset_by_id_list(catalog, prefs["favorites"]),
        lat=float(location["lat"]),
        lon=float(location["lon"]),
    )
    set_list_enriched = compute_altaz_now(
        subset_by_id_list(catalog, prefs["set_list"]),
        lat=float(location["lat"]),
        lon=float(location["lon"]),
    )

    use_phone_layout = st.toggle("Phone layout preview", value=False)

    if use_phone_layout:
        render_main_tabs(results_enriched, favorites_enriched, set_list_enriched, prefs)
        selected_row = resolve_selected_row(catalog, prefs)
        with st.container(border=True):
            st.markdown("### Detail bottom sheet")
            render_detail_panel(
                selected=selected_row,
                prefs=prefs,
                temperature_unit=effective_temperature_unit,
            )
    else:
        result_col, detail_col = st.columns([1.1, 1])
        with result_col:
            render_main_tabs(results_enriched, favorites_enriched, set_list_enriched, prefs)

        selected_row = resolve_selected_row(catalog, prefs)
        with detail_col:
            render_detail_panel(
                selected=selected_row,
                prefs=prefs,
                temperature_unit=effective_temperature_unit,
            )


if __name__ == "__main__":
    main()
