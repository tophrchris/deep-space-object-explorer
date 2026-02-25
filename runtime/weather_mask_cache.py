from __future__ import annotations

# Transitional bridge during runtime/service split: this module still relies on shared
# helpers/constants from `ui.streamlit_app` until they are extracted.
from ui import streamlit_app as _legacy_ui

def _refresh_legacy_globals() -> None:
    for _name, _value in vars(_legacy_ui).items():
        if _name == "__builtins__":
            continue
        globals().setdefault(_name, _value)


_refresh_legacy_globals()

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
    _refresh_legacy_globals()
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
