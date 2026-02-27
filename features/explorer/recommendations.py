from __future__ import annotations

# Transitional bridge during Explorer split: this module still relies on shared
# helpers/constants from `ui.streamlit_app` until they are extracted.
from ui import streamlit_app as _legacy_ui
from runtime.brightness_policy import (
    STATUS_BROADBAND_BORDERLINE,
    STATUS_IN_RANGE,
    STATUS_NARROWBAND_BOOSTED,
    STATUS_UNKNOWN,
    classify_target_magnitude,
    resolve_magnitude_thresholds,
    resolve_telescope_max_magnitude,
    slider_upper_bound,
)

def _refresh_legacy_globals() -> None:
    for _name, _value in vars(_legacy_ui).items():
        if _name == "__builtins__":
            continue
        globals().setdefault(_name, _value)


_refresh_legacy_globals()

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
    _refresh_legacy_globals()
    del active_preview_list_ids
    recommendation_night_title = format_recommendation_night_title(weather_forecast_day_offset, window_start)
    st.markdown(f"### Recommended Targets for {recommendation_night_title}")

    if catalog.empty:
        st.info("Catalog is empty.")
        return

    def _parse_emission_band_set(value: Any) -> set[str]:
        return parse_emission_band_set(value)

    def _build_keyword_priority_tokens(query_text: str) -> list[str]:
        cleaned = str(query_text or "").strip()
        if not cleaned:
            return []
        query_norm = normalize_text(cleaned)
        canonical_norm = normalize_text(canonicalize_designation(cleaned))
        tokens = {query_norm, canonical_norm}

        # Keep keyword-priority matching aligned with search_catalog token variants.
        if re.fullmatch(r"o[l1]{3}", query_norm):
            tokens.add("oiii")
        if re.fullmatch(r"n[l1]{2}", query_norm):
            tokens.add("nii")
        if re.fullmatch(r"s[l1]{2}", query_norm):
            tokens.add("sii")
        if query_norm == "o3":
            tokens.add("oiii")
        if query_norm == "n2":
            tokens.add("nii")
        if query_norm == "s2":
            tokens.add("sii")

        return [token for token in tokens if token]

    def _build_keyword_exact_match_patterns(query_text: str) -> list[re.Pattern[str]]:
        cleaned = str(query_text or "").strip()
        if not cleaned:
            return []

        raw_terms = {cleaned}
        canonical = str(canonicalize_designation(cleaned) or "").strip()
        if canonical:
            raw_terms.add(canonical)

        patterns: list[re.Pattern[str]] = []
        for term in raw_terms:
            lowered = term.lower()
            parts = [re.escape(part) for part in re.split(r"\s+", lowered) if part]
            if not parts:
                continue
            term_pattern = r"\s+".join(parts)
            patterns.append(re.compile(rf"(?<![a-z0-9]){term_pattern}(?![a-z0-9])"))
        return patterns

    def _keyword_match_priority_series(
        frame: pd.DataFrame,
        keyword_tokens: list[str],
        exact_match_patterns: list[re.Pattern[str]],
    ) -> pd.Series:
        if frame.empty:
            return pd.Series(index=frame.index, dtype=int)
        if not keyword_tokens and not exact_match_patterns:
            return pd.Series(np.zeros(len(frame), dtype=np.int8), index=frame.index, dtype=np.int8)

        def _norm_text_series(column_name: str) -> pd.Series:
            if column_name not in frame.columns:
                return pd.Series([""] * len(frame), index=frame.index, dtype=object)
            return frame[column_name].fillna("").astype(str).map(normalize_text)

        def _raw_text_series(column_name: str) -> pd.Series:
            if column_name not in frame.columns:
                return pd.Series([""] * len(frame), index=frame.index, dtype=object)
            return frame[column_name].fillna("").astype(str).str.lower()

        primary_id_raw = _raw_text_series("primary_id")
        common_name_raw = _raw_text_series("common_name")
        aliases_raw = _raw_text_series("aliases")
        description_raw = _raw_text_series("description")
        primary_id_norm = (
            frame["primary_id_norm"].fillna("").astype(str)
            if "primary_id_norm" in frame.columns
            else _norm_text_series("primary_id")
        )
        common_name_norm = _norm_text_series("common_name")
        aliases_norm = (
            frame["aliases_norm"].fillna("").astype(str)
            if "aliases_norm" in frame.columns
            else _norm_text_series("aliases")
        )
        description_norm = _norm_text_series("description")

        primary_match = pd.Series(False, index=frame.index, dtype=bool)
        common_match = pd.Series(False, index=frame.index, dtype=bool)
        alias_match = pd.Series(False, index=frame.index, dtype=bool)
        description_match = pd.Series(False, index=frame.index, dtype=bool)
        for token in keyword_tokens:
            primary_match = primary_match | primary_id_norm.str.contains(token, regex=False)
            common_match = common_match | common_name_norm.str.contains(token, regex=False)
            alias_match = alias_match | aliases_norm.str.contains(token, regex=False)
            description_match = description_match | description_norm.str.contains(token, regex=False)

        primary_exact = pd.Series(False, index=frame.index, dtype=bool)
        common_exact = pd.Series(False, index=frame.index, dtype=bool)
        alias_exact = pd.Series(False, index=frame.index, dtype=bool)
        description_exact = pd.Series(False, index=frame.index, dtype=bool)
        for pattern in exact_match_patterns:
            primary_exact = primary_exact | primary_id_raw.str.contains(pattern, regex=True)
            common_exact = common_exact | common_name_raw.str.contains(pattern, regex=True)
            alias_exact = alias_exact | aliases_raw.str.contains(pattern, regex=True)
            description_exact = description_exact | description_raw.str.contains(pattern, regex=True)

        priorities = np.zeros(len(frame), dtype=np.int8)
        priorities = np.where(description_match.to_numpy(dtype=bool), 1, priorities)
        priorities = np.where(alias_match.to_numpy(dtype=bool), 2, priorities)
        priorities = np.where(common_match.to_numpy(dtype=bool), 3, priorities)
        priorities = np.where(primary_match.to_numpy(dtype=bool), 4, priorities)
        priorities = np.where(description_exact.to_numpy(dtype=bool), 5, priorities)
        priorities = np.where(alias_exact.to_numpy(dtype=bool), 6, priorities)
        priorities = np.where(common_exact.to_numpy(dtype=bool), 7, priorities)
        priorities = np.where(primary_exact.to_numpy(dtype=bool), 8, priorities)
        return pd.Series(priorities, index=frame.index, dtype=np.int8)

    def _safe_positive_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric <= 0.0:
            return None
        return float(numeric)

    def _safe_finite_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return float(numeric)

    def _format_threshold_text(value: float) -> str:
        return f"{float(value):.1f}".rstrip("0").rstrip(".")

    def _target_emission_band_set(value: Any) -> set[str]:
        if isinstance(value, set):
            return {str(token).strip() for token in value if str(token).strip()}
        if isinstance(value, (list, tuple)):
            return {str(token).strip() for token in value if str(token).strip()}
        return _parse_emission_band_set(value)

    def _target_has_emissions_data(value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        return text.lower() not in {"-", "none", "nan"}

    def _is_hii_object_type(value: Any) -> bool:
        compact = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
        return compact in {"hii", "hiiregion"}

    def _is_bright_nebula_group(value: Any) -> bool:
        compact = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
        return compact == "brightnebula"

    def _target_is_narrowband_target(
        *,
        object_type_group_value: Any,
        object_type_value: Any,
        emission_tokens_value: Any,
        emission_lines_value: Any,
    ) -> bool:
        if _is_bright_nebula_group(object_type_group_value):
            return True
        if _is_hii_object_type(object_type_value):
            return True
        if _target_emission_band_set(emission_tokens_value):
            return True
        return _target_has_emissions_data(emission_lines_value)

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

    active_telescope_id = str(active_equipment.get("active_telescope_id", "")).strip()
    active_telescope = active_equipment.get("active_telescope")
    if not isinstance(active_telescope, dict):
        active_telescope = telescope_lookup.get(active_telescope_id) if active_telescope_id else None

    active_filter_id = str(active_equipment.get("active_filter_id", "__none__")).strip() or "__none__"
    active_filter = active_equipment.get("active_filter")
    if not isinstance(active_filter, dict):
        active_filter = filter_lookup.get(active_filter_id) if active_filter_id != "__none__" else None
    active_filter_bands = (
        _parse_emission_band_set(active_filter.get("emission_bands", []))
        if isinstance(active_filter, dict)
        else set()
    )
    narrowband_filter_active = bool(active_filter_bands)

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
    include_mount_key = "recommended_targets_include_mount_adaptation"
    min_size_enabled_key = "recommended_targets_min_size_enabled"
    min_size_value_key = "recommended_targets_min_size_pct"
    max_magnitude_enabled_key = "recommended_targets_max_magnitude_enabled"
    max_magnitude_value_key = "recommended_targets_max_magnitude_value"
    max_magnitude_default_telescope_key = "recommended_targets_max_magnitude_default_telescope_id"
    max_magnitude_last_recommended_key = "recommended_targets_max_magnitude_last_recommended"
    page_size_key = "recommended_targets_page_size"
    page_number_key = "recommended_targets_page_number"
    sort_field_key = "recommended_targets_sort_field"
    sort_direction_key = "recommended_targets_sort_direction"
    sort_signature_key = "recommended_targets_sort_signature"
    signature_key = "recommended_targets_criteria_signature"
    selection_token_key = "recommended_targets_selection_token"
    query_cache_key = "recommended_targets_query_cache"
    table_instance_key = "recommended_targets_table_instance"
    pending_hour_key_state_key = "recommended_targets_pending_hour_key"

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
    if include_mount_key not in st.session_state:
        st.session_state[include_mount_key] = True
    if min_size_enabled_key not in st.session_state:
        st.session_state[min_size_enabled_key] = False
    if min_size_value_key not in st.session_state:
        st.session_state[min_size_value_key] = 0
    if max_magnitude_enabled_key not in st.session_state:
        st.session_state[max_magnitude_enabled_key] = False
    if page_size_key not in st.session_state:
        st.session_state[page_size_key] = 100
    elif int(st.session_state.get(page_size_key, 100)) not in {10, 100, 200}:
        st.session_state[page_size_key] = 100
    if not isinstance(st.session_state.get(page_number_key), int):
        st.session_state[page_number_key] = 1
    if not isinstance(st.session_state.get(table_instance_key), int):
        st.session_state[table_instance_key] = 0
    if str(st.session_state.get(sort_direction_key, "Descending")).strip() not in {"Descending", "Ascending"}:
        st.session_state[sort_direction_key] = "Descending"

    pending_hour_key = str(st.session_state.pop(pending_hour_key_state_key, "")).strip()
    if pending_hour_key:
        matching_hour_options = [
            hour_option
            for hour_option, hour_key in hour_option_to_key.items()
            if str(hour_key or "").strip() == pending_hour_key
        ]
        if matching_hour_options:
            st.session_state[visible_hour_key] = [matching_hour_options[0]]
            st.session_state[page_number_key] = 1
            st.session_state[selection_token_key] = ""
            st.session_state["selected_id"] = ""
            st.session_state.pop(TARGET_DETAIL_MODAL_OPEN_REQUEST_KEY, None)
            st.session_state[table_instance_key] = int(st.session_state.get(table_instance_key, 0)) + 1

    active_telescope_default_marker = (
        str((active_telescope or {}).get("id", "")).strip()
        if isinstance(active_telescope, dict)
        else "__none__"
    )
    recommended_max_magnitude_default = resolve_telescope_max_magnitude(
        active_telescope if isinstance(active_telescope, dict) else None,
        fallback=10.5,
    )
    max_magnitude_slider_max = slider_upper_bound(recommended_max_magnitude_default)
    legacy_min_brightness_value = _safe_finite_float(
        st.session_state.get("recommended_targets_min_brightness_value")
    )
    seed_max_magnitude_value = _safe_finite_float(st.session_state.get(max_magnitude_value_key))
    if seed_max_magnitude_value is None and legacy_min_brightness_value is not None:
        seed_max_magnitude_value = float(legacy_min_brightness_value)
    if seed_max_magnitude_value is None:
        seed_max_magnitude_value = float(recommended_max_magnitude_default)
    previous_default_telescope_marker = str(
        st.session_state.get(max_magnitude_default_telescope_key, "")
    ).strip()
    previous_recommended_value = _safe_finite_float(
        st.session_state.get(max_magnitude_last_recommended_key)
    )
    if (
        previous_default_telescope_marker != active_telescope_default_marker
        and previous_recommended_value is not None
        and abs(float(seed_max_magnitude_value) - float(previous_recommended_value)) < 1e-6
    ):
        seed_max_magnitude_value = float(recommended_max_magnitude_default)
    seed_max_magnitude_value = min(
        max(0.0, float(seed_max_magnitude_value)),
        float(max_magnitude_slider_max),
    )
    seed_max_magnitude_value = round(float(seed_max_magnitude_value), 1)
    if max_magnitude_value_key not in st.session_state:
        st.session_state[max_magnitude_value_key] = seed_max_magnitude_value
    current_max_magnitude_value = _safe_finite_float(
        st.session_state.get(max_magnitude_value_key)
    )
    if current_max_magnitude_value is None:
        current_max_magnitude_value = seed_max_magnitude_value
    current_max_magnitude_value = min(
        max(0.0, float(current_max_magnitude_value)),
        float(max_magnitude_slider_max),
    )
    current_max_magnitude_value = round(float(current_max_magnitude_value), 1)
    st.session_state[max_magnitude_default_telescope_key] = active_telescope_default_marker
    st.session_state[max_magnitude_last_recommended_key] = float(recommended_max_magnitude_default)
    recommended_max_magnitude_default = round(float(recommended_max_magnitude_default), 1)
    recommended_max_magnitude_text = _format_threshold_text(recommended_max_magnitude_default)
    estimated_magnitude_telescope_name = (
        str(active_telescope.get("name", "selected telescope")).strip()
        if isinstance(active_telescope, dict)
        else "selected telescope"
    )
    if not estimated_magnitude_telescope_name:
        estimated_magnitude_telescope_name = "selected telescope"

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
    include_mount_adaptation = False
    use_minimum_size = False
    minimum_size_pct: float | None = None
    maximum_magnitude_mag = float(current_max_magnitude_value)
    use_maximum_magnitude = False
    magnitude_filter_available = False
    magnitude_filter_active = False
    telescope_fov_maj: float | None = None
    telescope_fov_min: float | None = None
    telescope_fov_area: float | None = None
    recommendation_min_framing_pct = 0.0
    recommendation_max_framing_pct = 500.0

    with criteria_col_3:
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
        if isinstance(active_telescope, dict):
            active_telescope_name = str(active_telescope.get("name", "Selected telescope")).strip() or "Selected telescope"
            include_telescope_details = st.checkbox(
                "Include telescope details in recommendations",
                key=include_telescope_key,
                help=f"Equipped telescope: {active_telescope_name}",
            )
            if not include_telescope_details:
                st.caption("including Telescope details enables filtering by target size and magnitude")
        if include_telescope_details and isinstance(active_telescope, dict):
            selected_telescope = active_telescope

        magnitude_filter_available = selected_telescope is not None
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

        if magnitude_filter_available:
            use_maximum_magnitude = st.checkbox("Enable Maximum Magnitude", key=max_magnitude_enabled_key)
            magnitude_filter_active = bool(use_maximum_magnitude)
            max_magnitude_slider_col, _ = st.columns([1, 1], gap="small")
            with max_magnitude_slider_col:
                max_magnitude_slider_value = st.slider(
                    "Maximum Magnitude",
                    min_value=0.0,
                    max_value=float(max_magnitude_slider_max),
                    value=float(current_max_magnitude_value),
                    step=0.1,
                    key=max_magnitude_value_key,
                    disabled=not use_maximum_magnitude,
                    help=(
                        f"The estimated max magnitude for {estimated_magnitude_telescope_name} is "
                        f"{recommended_max_magnitude_text}. "
                        "Targets are filtered by catalog magnitude (lower number = brighter). "
                        "Targets with unknown magnitude are excluded."
                    ),
                )
            if use_maximum_magnitude:
                maximum_magnitude_mag = round(float(max_magnitude_slider_value), 1)
            slider_thresholds = resolve_magnitude_thresholds(maximum_magnitude_mag)
            if use_maximum_magnitude and narrowband_filter_active:
                st.markdown(
                    (
                        "<div style=\"margin:0.1rem 0 0 0; color:#6b7280; font-size:0.875rem; line-height:1.35;\">"
                        "<div>Narrowband filter adjustment:</div>"
                        "<div>broadband effective &lt;= "
                        f"<span style=\"color:#991b1b; font-weight:700;\">{slider_thresholds.broadband_effective_max:.1f}</span>"
                        "</div>"
                        "<div>narrowband effective &lt;= "
                        f"<span style=\"color:#1d4ed8; font-weight:700;\">{slider_thresholds.narrowband_effective_max:.1f}</span>."
                        "</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            elif use_maximum_magnitude:
                st.caption(f"Maximum magnitude applied: <= {slider_thresholds.selected_max:.1f}.")

    mount_mode = active_mount_choice if include_mount_adaptation else "none"
    magnitude_thresholds = resolve_magnitude_thresholds(maximum_magnitude_mag)

    if search_notes_placeholder is not None:
        with search_notes_placeholder.container():
            if magnitude_filter_active:
                st.caption(
                    "Maximum Magnitude applied: "
                    f"selected <= {magnitude_thresholds.selected_max:.1f} "
                    "(targets with unknown magnitude are excluded)."
                )
                if narrowband_filter_active:
                    st.caption(
                        "Narrowband adjustment active: "
                        f"broadband effective <= {magnitude_thresholds.broadband_effective_max:.1f}, "
                        f"narrowband effective <= {magnitude_thresholds.narrowband_effective_max:.1f}."
                    )

    has_selected_hour_window = bool(selected_visible_hours)
    altitude_sort_label = "Max Alt in Window" if has_selected_hour_window else "Altitude at Peak"

    sort_options: list[tuple[str, str]] = [
        ("ranking", "Recommended"),
        ("target_name", "Target Name"),
        ("visible_minutes", "Duration of visibility"),
        ("object_type", "Object Type"),
        ("emissions", "Emissions"),
        ("apparent_size_sort_arcmin", "Apparent size"),
        ("peak_time_local", "Peak"),
        ("peak_altitude", altitude_sort_label),
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
            sort_field_col, sort_direction_col, _sort_control_spacer_col = st.columns([2, 2, 1], gap="small")
            sort_field = sort_field_col.selectbox(
                "Sort results by",
                options=sort_option_values,
                key=sort_field_key,
                format_func=lambda option_value: sort_option_labels.get(option_value, option_value),
            )
            sort_direction = sort_direction_col.segmented_control(
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
            "include_mount": bool(include_mount_adaptation),
            "active_filter_id_for_brightness": active_filter_id,
            "narrowband_filter_active": bool(narrowband_filter_active),
            "mount_mode": mount_mode,
            "telescope_id": str(selected_telescope.get("id", "")) if isinstance(selected_telescope, dict) else "",
            "min_size_enabled": bool(use_minimum_size),
            "min_size_pct": minimum_size_pct if minimum_size_pct is not None else "",
            "max_magnitude_enabled": bool(magnitude_filter_active),
            "max_magnitude_selected": float(maximum_magnitude_mag) if magnitude_filter_active else None,
            "max_magnitude_broadband_effective": (
                float(magnitude_thresholds.broadband_effective_max) if magnitude_filter_active else None
            ),
            "max_magnitude_narrowband_effective": (
                float(magnitude_thresholds.narrowband_effective_max) if magnitude_filter_active else None
            ),
            "telescope_fov_filter_min_pct": recommendation_min_framing_pct,
            "telescope_fov_filter_max_pct": recommendation_max_framing_pct,
            "catalog_mtime_ns": int(catalog_fingerprint[0]),
            "catalog_size_bytes": int(catalog_fingerprint[1]),
            "obstructions": {direction: float(obstructions.get(direction, 20.0)) for direction in WIND16},
            "cloud_cover_threshold": RECOMMENDATION_CLOUD_COVER_THRESHOLD,
            "sample_minutes": RECOMMENDATION_CACHE_SAMPLE_MINUTES,
            "result_limit_mode": "unlimited",
            "visibility_fallback_mode": "issue93_v1",
            "recommended_ranking_mode": "keyword_field_priority_v3",
            "window_start": pd.Timestamp(window_start).isoformat(),
            "window_end": pd.Timestamp(window_end).isoformat(),
        },
        sort_keys=True,
    )
    previous_criteria_signature = str(st.session_state.get(signature_key, ""))
    if previous_criteria_signature != criteria_signature:
        st.session_state[signature_key] = criteria_signature
        st.session_state[page_number_key] = 1
        st.session_state[selection_token_key] = ""
        if previous_criteria_signature:
            st.session_state["selected_id"] = ""
            st.session_state.pop(TARGET_DETAIL_MODAL_OPEN_REQUEST_KEY, None)
            st.session_state[table_instance_key] = int(st.session_state.get(table_instance_key, 0)) + 1

    query_cache_store = st.session_state.get(query_cache_key)
    if not isinstance(query_cache_store, dict):
        query_cache_store = {}
    query_cache_payload = query_cache_store.get(criteria_signature)
    if isinstance(query_cache_payload, dict):
        cached_status = str(query_cache_payload.get("status", "ok")).strip().lower()
        cached_recommended = query_cache_payload.get("recommended")
        required_cached_columns = {
            "primary_id",
            "target_name",
            "visible_minutes",
            "selected_visible_minutes",
            "visibility_reason",
            "object_type",
            "emissions",
            "apparent_size",
            "apparent_size_sort_arcmin",
            "peak_time_local",
            "peak_altitude",
            "peak_direction",
        }
        if selected_telescope is not None:
            required_cached_columns.add("framing_percent")
            required_cached_columns.add("framing_constraint_status")
        if sort_field != "ranking":
            required_cached_columns.add(str(sort_field))
        if cached_status == "ok" and (
            not isinstance(cached_recommended, pd.DataFrame)
            or not required_cached_columns.issubset(set(cached_recommended.columns))
        ):
            trace_cache_event(
                f"Discarding malformed session recommendation cache payload ({criteria_signature[:12]}...)"
            )
            query_cache_store.pop(criteria_signature, None)
            query_cache_payload = None
    recommended = pd.DataFrame()
    total_results_uncapped = 0
    empty_query_message: str | None = None
    size_framing_fallback_active = False
    size_framing_fallback_message: str | None = None

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
            size_framing_fallback_active = bool(query_cache_payload.get("size_framing_fallback_active", False))
            fallback_message_raw = str(query_cache_payload.get("size_framing_fallback_message", "")).strip()
            size_framing_fallback_message = fallback_message_raw or None
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
        keyword_priority_tokens = _build_keyword_priority_tokens(cleaned_keyword)
        keyword_exact_match_patterns = _build_keyword_exact_match_patterns(cleaned_keyword)
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

        if "magnitude_numeric" not in working_catalog.columns:
            working_catalog = working_catalog.copy()
            if "magnitude" in working_catalog.columns:
                working_catalog["magnitude_numeric"] = pd.to_numeric(working_catalog["magnitude"], errors="coerce")
            else:
                working_catalog["magnitude_numeric"] = np.nan
        if "emission_band_tokens" not in working_catalog.columns:
            working_catalog = working_catalog.copy()
            if "emission_lines" in working_catalog.columns:
                working_catalog["emission_band_tokens"] = working_catalog["emission_lines"].apply(
                    lambda value: tuple(sorted(_parse_emission_band_set(value)))
                )
            else:
                working_catalog["emission_band_tokens"] = [tuple() for _ in range(len(working_catalog))]

        magnitude_keep_mask: list[bool] = []
        magnitude_statuses: list[str] = []
        target_is_narrowband_flags: list[bool] = []
        object_type_group_values = (
            working_catalog["object_type_group"].tolist()
            if "object_type_group" in working_catalog.columns
            else ["" for _ in range(len(working_catalog))]
        )
        object_type_values = (
            working_catalog["object_type"].tolist()
            if "object_type" in working_catalog.columns
            else ["" for _ in range(len(working_catalog))]
        )
        emission_lines_values = (
            working_catalog["emission_lines"].tolist()
            if "emission_lines" in working_catalog.columns
            else ["" for _ in range(len(working_catalog))]
        )
        for magnitude_value, emission_tokens, object_type_group_value, object_type_value, emission_lines_value in zip(
            working_catalog["magnitude_numeric"].tolist(),
            working_catalog["emission_band_tokens"].tolist(),
            object_type_group_values,
            object_type_values,
            emission_lines_values,
        ):
            target_is_narrowband = _target_is_narrowband_target(
                object_type_group_value=object_type_group_value,
                object_type_value=object_type_value,
                emission_tokens_value=emission_tokens,
                emission_lines_value=emission_lines_value,
            )
            if magnitude_filter_active:
                include_target, status, _ = classify_target_magnitude(
                    magnitude_value,
                    selected_max=maximum_magnitude_mag,
                    narrowband_filter_active=narrowband_filter_active,
                    target_is_narrowband=target_is_narrowband,
                )
            else:
                include_target = True
                status = STATUS_IN_RANGE if _safe_finite_float(magnitude_value) is not None else STATUS_UNKNOWN
            magnitude_keep_mask.append(bool(include_target))
            magnitude_statuses.append(str(status))
            target_is_narrowband_flags.append(target_is_narrowband)

        working_catalog = working_catalog.copy()
        working_catalog["magnitude_policy_status"] = magnitude_statuses
        working_catalog["target_is_narrowband"] = target_is_narrowband_flags
        working_catalog = working_catalog[np.asarray(magnitude_keep_mask, dtype=bool)]

        if working_catalog.empty:
            if magnitude_filter_active:
                empty_query_message = "No targets match the current Maximum Magnitude threshold."
            else:
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
                    sample_times_local = np.asarray(altaz_bundle.get("sample_times_local_iso", ()), dtype=object)
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

                    display_peak_altitude = np.asarray(peak_altitude, dtype=float)
                    display_peak_time_local = np.asarray(peak_time_local, dtype=object)
                    display_peak_direction = np.asarray(peak_direction, dtype=object)
                    if selected_visible_hour_keys and bool(np.any(selected_hours_mask)):
                        selected_time_indices = np.where(selected_hours_mask)[0].astype(int)
                        if selected_time_indices.size > 0:
                            window_altitude_matrix = altitude_matrix[selected_time_indices, :]
                            window_wind_index_matrix = wind_index_matrix[selected_time_indices, :]
                            if (
                                window_altitude_matrix.ndim == 2
                                and window_altitude_matrix.shape[0] > 0
                                and window_altitude_matrix.shape[1] == len(candidate_col_indices)
                            ):
                                window_peak_idx = np.argmax(window_altitude_matrix, axis=0)
                                display_peak_altitude = window_altitude_matrix[
                                    window_peak_idx,
                                    np.arange(len(candidate_col_indices)),
                                ]
                                if sample_times_local.size > int(selected_time_indices.max()):
                                    window_sample_times_local = sample_times_local[selected_time_indices]
                                    display_peak_time_local = np.array(
                                        [window_sample_times_local[int(index)] for index in window_peak_idx],
                                        dtype=object,
                                    )
                                display_peak_direction = np.array(
                                    [
                                        WIND16[int(window_wind_index_matrix[int(index), target_idx])]
                                        for target_idx, index in enumerate(window_peak_idx)
                                    ],
                                    dtype=object,
                                )

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
                    if not np.any(eligible_mask):
                        if no_visible_targets:
                            empty_query_message = "No targets above the horizon meet the current visibility/weather/mount constraints."
                        else:
                            empty_query_message = "No targets meet the current visibility/weather/mount constraints."
                    else:
                        eligible_indices = np.where(eligible_mask)[0]
                        recommended = candidates.iloc[eligible_indices].copy()
                        recommended["keyword_match_priority"] = _keyword_match_priority_series(
                            recommended,
                            keyword_priority_tokens,
                            keyword_exact_match_patterns,
                        )
                        recommended["visible_minutes"] = full_night_visible_minutes[eligible_indices]
                        recommended["selected_visible_minutes"] = selected_visible_minutes[eligible_indices]
                        recommended["filter_match_tier"] = filter_match_tier[eligible_indices]
                        recommended["visibility_reason"] = visibility_reason[eligible_indices]
                        recommended["peak_altitude"] = np.round(display_peak_altitude[eligible_indices], 1)
                        recommended["peak_time_local"] = display_peak_time_local[eligible_indices]
                        recommended["peak_direction"] = display_peak_direction[eligible_indices]
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
                        recommended["framing_constraint_status"] = ""
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
                            effective_min_framing_pct = max(
                                float(recommendation_min_framing_pct),
                                float(minimum_size_pct) if minimum_size_pct is not None else float(recommendation_min_framing_pct),
                            )

                            def _framing_constraint_status(value: Any) -> str:
                                if value is None or pd.isna(value):
                                    return "Unknown"
                                try:
                                    numeric = float(value)
                                except (TypeError, ValueError):
                                    return "Unknown"
                                if not np.isfinite(numeric):
                                    return "Unknown"
                                if numeric < effective_min_framing_pct:
                                    return "Too small"
                                if numeric > float(recommendation_max_framing_pct):
                                    return "Too large"
                                return "OK"

                            recommended["framing_constraint_status"] = recommended["framing_percent"].apply(
                                _framing_constraint_status
                            )
                            recommended_before_size_filters = recommended.copy()

                            practical_framing_mask = recommended["framing_percent"].apply(
                                lambda value: (
                                    value is not None
                                    and not pd.isna(value)
                                    and np.isfinite(float(value))
                                    and float(recommendation_min_framing_pct) <= float(value) <= float(recommendation_max_framing_pct)
                                )
                            )
                            size_filtered_recommended = recommended[practical_framing_mask].copy()

                            if minimum_size_pct is not None:
                                size_filtered_recommended = size_filtered_recommended[
                                    size_filtered_recommended["framing_percent"].apply(
                                        lambda value: (
                                            value is not None
                                            and not pd.isna(value)
                                            and np.isfinite(float(value))
                                            and float(value) >= float(minimum_size_pct)
                                        )
                                    )
                                ].copy()

                            if (
                                size_filtered_recommended.empty
                                and not recommended_before_size_filters.empty
                                and keyword_search_active
                            ):
                                recommended = recommended_before_size_filters
                                size_framing_fallback_active = True
                                size_framing_fallback_message = (
                                    "No targets met the current telescope size/framing constraints. "
                                    "Showing matches anyway and highlighting Framing values that fall outside the limits."
                                )
                            else:
                                recommended = size_filtered_recommended

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
                                    "keyword_match_priority",
                                    "filter_match_tier",
                                    "visibility_bin_15",
                                    "peak_alt_band_10",
                                    "sort_size_metric",
                                    "selected_visible_minutes",
                                    "peak_altitude",
                                    "primary_id",
                                ],
                                ascending=[False, False, False, False, False, False, False, True],
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
                "size_framing_fallback_active": bool(size_framing_fallback_active),
                "size_framing_fallback_message": size_framing_fallback_message or "",
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

    page_size = int(st.session_state.get(page_size_key, 100))
    if page_size not in {10, 100, 200}:
        page_size = 100
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
    if size_framing_fallback_active:
        st.info(
            size_framing_fallback_message
            or (
                "No targets met the current telescope size/framing constraints. "
                "Showing matches anyway and highlighting out-of-range Framing values."
            )
        )
        st.caption("Framing highlight legend: red = too small, amber = too large, gray = unknown size.")
    start_index = (page_number - 1) * int(page_size)
    end_index = min(total_results, start_index + int(page_size))
    page_frame = recommended.iloc[start_index:end_index].copy().reset_index(drop=True)

    page_frame["Peak"] = page_frame["peak_time_local"].apply(
        lambda value: format_display_time(pd.Timestamp(value), use_12_hour=use_12_hour)
        if value is not None and not pd.isna(value)
        else "--"
    )
    altitude_display_column_label = "Max Alt in Window" if has_selected_hour_window else "Altitude at Peak"
    page_frame[altitude_display_column_label] = page_frame["peak_altitude"].apply(
        lambda value: f"{float(value):.1f} deg" if value is not None and not pd.isna(value) else "--"
    )
    page_frame["Direction"] = page_frame["peak_direction"].fillna("--").astype(str)
    magnitude_source_column = (
        "magnitude_numeric"
        if "magnitude_numeric" in page_frame.columns
        else ("magnitude" if "magnitude" in page_frame.columns else "")
    )
    if magnitude_source_column:
        page_frame["magnitude_display"] = page_frame[magnitude_source_column].apply(
            lambda value: (
                f"{float(value):.1f}"
                if value is not None and not pd.isna(value) and np.isfinite(float(value))
                else "--"
            )
        )
    else:
        page_frame["magnitude_display"] = "--"
    magnitude_policy_statuses: list[str] = []
    for row_idx in range(len(page_frame)):
        emission_tokens = page_frame.iloc[row_idx].get("emission_band_tokens")
        emission_lines_value = page_frame.iloc[row_idx].get("emission_lines")
        object_type_group_value = page_frame.iloc[row_idx].get("object_type_group")
        target_is_narrowband = _target_is_narrowband_target(
            object_type_group_value=object_type_group_value,
            object_type_value=page_frame.iloc[row_idx].get("object_type"),
            emission_tokens_value=emission_tokens,
            emission_lines_value=emission_lines_value,
        )
        raw_magnitude = page_frame.iloc[row_idx].get(magnitude_source_column) if magnitude_source_column else None
        if magnitude_filter_active:
            _, magnitude_status, _ = classify_target_magnitude(
                raw_magnitude,
                selected_max=maximum_magnitude_mag,
                narrowband_filter_active=narrowband_filter_active,
                target_is_narrowband=target_is_narrowband,
            )
        else:
            magnitude_status = STATUS_IN_RANGE if _safe_finite_float(raw_magnitude) is not None else STATUS_UNKNOWN
        magnitude_policy_statuses.append(str(magnitude_status))
    page_frame["magnitude_policy_status"] = magnitude_policy_statuses

    def _thumbnail_numeric(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(parsed):
            return None
        return float(parsed)

    def _thumbnail_cutout_fov_deg(row: pd.Series) -> tuple[float, float]:
        major_arcmin = _thumbnail_numeric(row.get("ang_size_maj_arcmin"))
        minor_arcmin = _thumbnail_numeric(row.get("ang_size_min_arcmin"))
        span_deg_candidates = [
            float(value) / 60.0
            for value in (major_arcmin, minor_arcmin)
            if value is not None and value > 0.0
        ]
        if span_deg_candidates:
            estimated_span = max(span_deg_candidates) * 3.0
            framed_span = float(max(0.5, min(8.0, estimated_span)))
            return framed_span, framed_span
        return 1.5, 1.5

    def _resolve_thumbnail_url(row: pd.Series) -> str | None:
        for key in ("image_url", "hero_image_url"):
            raw_value = str(row.get(key, "") or "").strip()
            if raw_value.lower().startswith(("https://", "http://")):
                return raw_value

        wikimedia_search_phrase = (
            str(row.get("common_name") or "").strip()
            or str(row.get("primary_id") or "").strip()
        )
        if wikimedia_search_phrase:
            image_data = fetch_free_use_image(wikimedia_search_phrase)
            wiki_image_url = str((image_data or {}).get("image_url", "") or "").strip()
            if wiki_image_url.lower().startswith(("https://", "http://")):
                return wiki_image_url

        ra_deg = _thumbnail_numeric(row.get("ra_deg"))
        dec_deg = _thumbnail_numeric(row.get("dec_deg"))
        if ra_deg is None or dec_deg is None:
            return None

        fov_width_deg, fov_height_deg = _thumbnail_cutout_fov_deg(row)
        for layer in ("unwise-neo4", "sfd", "sdss2"):
            legacy_cutout = build_legacy_survey_cutout_urls(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                fov_width_deg=fov_width_deg,
                fov_height_deg=fov_height_deg,
                layer=layer,
                max_pixels=256,
            )
            legacy_image_url = str((legacy_cutout or {}).get("image_url", "") or "").strip()
            if legacy_image_url.lower().startswith(("https://", "http://")):
                return legacy_image_url
        return None

    page_frame["thumbnail_url"] = page_frame.apply(_resolve_thumbnail_url, axis=1)

    display_columns = [
        "thumbnail_url",
        "target_name",
        "visibility_duration",
        "object_type",
        "magnitude_display",
        "emissions",
        "apparent_size",
    ]
    if selected_telescope is not None:
        display_columns.append("framing_percent")
    display_columns.extend(["Peak", altitude_display_column_label, "Direction"])

    rename_columns = {
        "thumbnail_url": "Thumbnail",
        "target_name": "Target Name",
        "visibility_duration": "Duration of visibility",
        "object_type": "Object Type",
        "magnitude_display": "Magnitude",
        "emissions": "Emissions",
        "apparent_size": "Apparent size",
        "framing_percent": "Framing",
    }
    display_table = page_frame[display_columns].rename(columns=rename_columns)

    column_config: dict[str, Any] = {
        "Thumbnail": (
            st.column_config.ImageColumn(label="", width=100, pinned=True)
            if hasattr(st.column_config, "ImageColumn")
            else st.column_config.TextColumn(label="", width=100, pinned=True)
        ),
        "Target Name": st.column_config.TextColumn(width="large", pinned=True),
        "Duration of visibility": st.column_config.TextColumn(width="small"),
        "Object Type": st.column_config.TextColumn(width="small"),
        "Magnitude": st.column_config.TextColumn(width="small"),
        "Emissions": st.column_config.TextColumn(width="small"),
        "Apparent size": st.column_config.TextColumn(width="small"),
        "Peak": st.column_config.TextColumn(width="small"),
        altitude_display_column_label: st.column_config.TextColumn(width="small"),
        "Direction": st.column_config.TextColumn(width="small"),
    }
    if "Framing" in display_table.columns:
        column_config["Framing"] = st.column_config.NumberColumn(width="small", format="%.0f%%")

    recommendation_styler = display_table.style.set_properties(
        subset=["Thumbnail"],
        **{
            "text-align": "left !important",
            "justify-content": "flex-start !important",
            "padding-left": "0px !important",
        },
    )
    if (
        magnitude_filter_active
        and "Magnitude" in display_table.columns
        and "magnitude_policy_status" in page_frame.columns
    ):
        def _style_magnitude_cells(series: pd.Series) -> list[str]:
            styles: list[str] = []
            for row_idx in series.index:
                status = str(page_frame.at[int(row_idx), "magnitude_policy_status"]).strip().lower()
                if status == STATUS_BROADBAND_BORDERLINE:
                    styles.append(
                        "color: #991b1b; font-weight: 700;"
                    )
                elif status == STATUS_NARROWBAND_BOOSTED:
                    styles.append(
                        "color: #1d4ed8; font-weight: 700;"
                    )
                elif status == STATUS_IN_RANGE:
                    styles.append("")
                else:
                    styles.append("")
            return styles

        recommendation_styler = recommendation_styler.apply(_style_magnitude_cells, subset=["Magnitude"])
    if "Framing" in display_table.columns and "framing_constraint_status" in page_frame.columns:
        def _style_framing_cells(series: pd.Series) -> list[str]:
            styles: list[str] = []
            for row_idx in series.index:
                status = str(page_frame.at[int(row_idx), "framing_constraint_status"]).strip().lower()
                if not size_framing_fallback_active:
                    styles.append("")
                    continue
                if status == "too small":
                    styles.append(
                        "background-color: rgba(220, 38, 38, 0.16); color: #991b1b; font-weight: 700;"
                    )
                elif status == "too large":
                    styles.append(
                        "background-color: rgba(245, 158, 11, 0.18); color: #92400e; font-weight: 700;"
                    )
                elif status == "unknown":
                    styles.append(
                        "background-color: rgba(100, 116, 139, 0.14); color: #475569;"
                    )
                else:
                    styles.append("")
            return styles

        recommendation_styler = recommendation_styler.apply(_style_framing_cells, subset=["Framing"])

    st.markdown(
        """
        <style>
        [data-testid="stDataFrame"] [data-testid="stDataFrameGlideDataEditor"] [role="gridcell"]:nth-child(1) {
            justify-content: flex-start !important;
            text-align: left !important;
            padding-left: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    recommendation_event = st.dataframe(
        apply_dataframe_styler_theme(recommendation_styler),
        hide_index=True,
        use_container_width=True,
        row_height=70,
        on_select="rerun",
        selection_mode="single-row",
        key=f"recommended_targets_table_{int(st.session_state.get(table_instance_key, 0))}",
        column_config=column_config,
    )

    per_page_col, page_col = st.columns([1, 1], gap="small")
    per_page_col.selectbox(
        "Results per page",
        options=[10, 100, 200],
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
