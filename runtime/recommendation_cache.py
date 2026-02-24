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

@st.cache_resource(show_spinner=False)
def _load_catalog_recommendation_features_cached(
    cache_path_str: str,
    cache_mtime_ns: int,
    cache_size_bytes: int,
) -> pd.DataFrame:
    _refresh_legacy_globals()
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
    _refresh_legacy_globals()
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
    _refresh_legacy_globals()
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
    _refresh_legacy_globals()
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

