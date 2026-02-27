from __future__ import annotations

from runtime.lunar_ephemeris import (
    compute_lunar_eclipse_visibility_for_night,
    compute_lunar_phase_for_night,
    compute_moon_track,
)

# Transitional bridge during Explorer split: this module still relies on shared
# helpers/constants from `ui.streamlit_app` until they are extracted.
from ui import streamlit_app as _legacy_ui

def _refresh_legacy_globals() -> None:
    for _name, _value in vars(_legacy_ui).items():
        if _name == "__builtins__":
            continue
        globals().setdefault(_name, _value)


_refresh_legacy_globals()

def render_detail_panel(
    selected: pd.Series | None,
    catalog: pd.DataFrame,
    prefs: dict[str, Any],
    temperature_unit: str,
    use_12_hour: bool,
    detail_stack_vertical: bool,
    weather_forecast_day_offset: int = 0,
) -> None:
    _refresh_legacy_globals()
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

    def positive_numeric(value: Any) -> float | None:
        numeric = parse_numeric(value)
        if numeric is None or numeric <= 0.0:
            return None
        return float(numeric)

    def _compute_moon_track_for_window(
        *,
        start_local: datetime,
        end_local: datetime,
        tz_name: str,
    ) -> pd.DataFrame:
        try:
            moon_track = compute_moon_track(
                lat=location_lat,
                lon=location_lon,
                tz_name=tz_name,
                start_local_iso=pd.Timestamp(start_local).isoformat(),
                end_local_iso=pd.Timestamp(end_local).isoformat(),
                sample_minutes=10,
            )
            if isinstance(moon_track, pd.DataFrame):
                return moon_track
        except Exception:
            pass
        return pd.DataFrame()

    def _compute_moon_phase_key_for_window(
        *,
        start_local: datetime,
        end_local: datetime,
        tz_name: str,
    ) -> str | None:
        try:
            phase_payload = compute_lunar_phase_for_night(
                tz_name=tz_name,
                start_local_iso=pd.Timestamp(start_local).isoformat(),
                end_local_iso=pd.Timestamp(end_local).isoformat(),
            )
            phase_key = str((phase_payload or {}).get("phase_key", "")).strip().lower()
            return phase_key or None
        except Exception:
            return None

    def _compute_lunar_eclipse_visibility_for_window(
        *,
        start_local: datetime,
        end_local: datetime,
        tz_name: str,
    ) -> dict[str, Any]:
        try:
            payload = compute_lunar_eclipse_visibility_for_night(
                lat=location_lat,
                lon=location_lon,
                tz_name=tz_name,
                start_local_iso=pd.Timestamp(start_local).isoformat(),
                end_local_iso=pd.Timestamp(end_local).isoformat(),
                sample_minutes=1,
                obstructions=(prefs.get("obstructions") if isinstance(prefs.get("obstructions"), dict) else None),
            )
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {}

    def resolve_active_telescope_for_framing() -> dict[str, Any] | None:
        equipment_context = build_owned_equipment_context(prefs)
        telescope_lookup = equipment_context.get("telescope_lookup", {})
        if not isinstance(telescope_lookup, dict):
            return None
        owned_telescope_ids = list(equipment_context.get("owned_telescope_ids", []))
        active_telescope_id = str(prefs.get("active_telescope_id", "")).strip()
        if active_telescope_id not in owned_telescope_ids and owned_telescope_ids:
            active_telescope_id = str(owned_telescope_ids[0]).strip()
        active_telescope = telescope_lookup.get(active_telescope_id) if active_telescope_id else None
        return active_telescope if isinstance(active_telescope, dict) else None

    try:
        normalized_forecast_day_offset = int(weather_forecast_day_offset)
    except (TypeError, ValueError):
        normalized_forecast_day_offset = 0
    if normalized_forecast_day_offset < 0:
        normalized_forecast_day_offset = 0
    if normalized_forecast_day_offset > (ASTRONOMY_FORECAST_NIGHTS - 1):
        normalized_forecast_day_offset = ASTRONOMY_FORECAST_NIGHTS - 1
    plot_mount_choice = _normalize_mount_choice(
        prefs.get("active_mount_choice", "altaz"),
        default_choice="altaz",
    )

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
        moon_track = _compute_moon_track_for_window(
            start_local=window_start,
            end_local=window_end,
            tz_name=tzinfo.key,
        )
        moon_phase_key = _compute_moon_phase_key_for_window(
            start_local=window_start,
            end_local=window_end,
            tz_name=tzinfo.key,
        )
        lunar_eclipse_visibility = _compute_lunar_eclipse_visibility_for_window(
            start_local=window_start,
            end_local=window_end,
            tz_name=tzinfo.key,
        )

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
                        "image_url": clean_text(preview_target.get("image_url")),
                        "hero_image_url": clean_text(preview_target.get("hero_image_url")),
                        "ra_deg": parse_numeric(preview_target.get("ra_deg")),
                        "dec_deg": parse_numeric(preview_target.get("dec_deg")),
                        "ang_size_maj_arcmin": parse_numeric(preview_target.get("ang_size_maj_arcmin")),
                        "ang_size_min_arcmin": parse_numeric(preview_target.get("ang_size_min_arcmin")),
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
                            mount_choice=plot_mount_choice,
                            moon_track=moon_track,
                            moon_phase_key=moon_phase_key,
                            eclipse_visibility=lunar_eclipse_visibility,
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
                            mount_choice=plot_mount_choice,
                            moon_track=moon_track,
                            moon_phase_key=moon_phase_key,
                            eclipse_visibility=lunar_eclipse_visibility,
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
                            mount_choice=plot_mount_choice,
                            moon_track=moon_track,
                            moon_phase_key=moon_phase_key,
                            eclipse_visibility=lunar_eclipse_visibility,
                        ),
                        use_container_width=True,
                        key="preview_unobstructed_area_plot",
                    )
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
    catalog_image_source_url = clean_text(selected.get("image_attribution_url"))
    catalog_image_license = clean_text(selected.get("license_label"))
    info_url = clean_text(selected.get("info_url")) or catalog_image_source_url
    if not catalog_image_source_url:
        catalog_image_source_url = info_url

    image_candidates: list[dict[str, str]] = []
    seen_image_urls: set[str] = set()

    def append_image_candidate(
        *,
        label: str,
        image_url: Any,
        source_url: Any = "",
        license_label: Any = "",
    ) -> None:
        candidate_url = clean_text(image_url)
        if not candidate_url or candidate_url in seen_image_urls:
            return
        seen_image_urls.add(candidate_url)
        image_candidates.append(
            {
                "label": clean_text(label) or "Image",
                "image_url": candidate_url,
                "source_url": clean_text(source_url),
                "license_label": clean_text(license_label),
            }
        )

    append_image_candidate(
        label="Catalog image",
        image_url=catalog_image_url,
        source_url=catalog_image_source_url,
        license_label=catalog_image_license,
    )

    ra_deg_for_cutout = parse_numeric(selected.get("ra_deg"))
    dec_deg_for_cutout = parse_numeric(selected.get("dec_deg"))
    telescope_for_framing = resolve_active_telescope_for_framing()
    telescope_fov_maj_deg = positive_numeric(telescope_for_framing.get("fov_maj_deg")) if telescope_for_framing else None
    telescope_fov_min_deg = positive_numeric(telescope_for_framing.get("fov_min_deg")) if telescope_for_framing else None

    ang_size_maj_arcmin_for_cutout = positive_numeric(selected.get("ang_size_maj_arcmin"))
    ang_size_min_arcmin_for_cutout = positive_numeric(selected.get("ang_size_min_arcmin"))

    legacy_cutout_credit = ""
    if telescope_fov_maj_deg is not None and telescope_fov_min_deg is not None:
        cutout_fov_width_deg = telescope_fov_maj_deg
        cutout_fov_height_deg = telescope_fov_min_deg
        legacy_cutout_credit = "Legacy Survey DR8 sky cutout (framed to active telescope FOV)"
    else:
        object_span_deg_candidates = [
            value / 60.0
            for value in (ang_size_maj_arcmin_for_cutout, ang_size_min_arcmin_for_cutout)
            if value is not None and value > 0.0
        ]
        if object_span_deg_candidates:
            estimated_span_deg = max(object_span_deg_candidates) * 3.0
            cutout_fov_width_deg = float(max(0.5, min(8.0, estimated_span_deg)))
            cutout_fov_height_deg = cutout_fov_width_deg
        else:
            cutout_fov_width_deg = 1.5
            cutout_fov_height_deg = 1.5
        legacy_cutout_credit = "Legacy Survey DR8 sky cutout (auto-framed from target coordinates)"

    wikimedia_search_phrase = clean_text(selected.get("common_name")) or target_id
    if wikimedia_search_phrase:
        image_data = fetch_free_use_image(wikimedia_search_phrase)
        if image_data and image_data.get("image_url"):
            append_image_candidate(
                label="Wikimedia Commons",
                image_url=image_data.get("image_url"),
                source_url=image_data.get("source_url"),
                license_label=image_data.get("license_label"),
            )

    if ra_deg_for_cutout is not None and dec_deg_for_cutout is not None:
        legacy_cutout_layers = [
            ("sdss2", "SDSS2", "Legacy Survey SDSS2"),
            ("unwise-neo4", "unwise-neo4", "Legacy Survey unWISE neo4"),
            ("halpha", "Halpha", "Legacy Survey Halpha"),
            ("sfd", "SFD", "Legacy Survey SFD"),
        ]
        for legacy_layer_id, legacy_layer_display, legacy_layer_label in legacy_cutout_layers:
            legacy_cutout = build_legacy_survey_cutout_urls(
                ra_deg=ra_deg_for_cutout,
                dec_deg=dec_deg_for_cutout,
                fov_width_deg=cutout_fov_width_deg,
                fov_height_deg=cutout_fov_height_deg,
                layer=legacy_layer_id,
                max_pixels=512,
            )
            if legacy_cutout is None:
                continue
            append_image_candidate(
                label=legacy_layer_label,
                image_url=legacy_cutout.get("image_url"),
                source_url=legacy_cutout.get("viewer_url"),
                license_label=f"{legacy_cutout_credit} | Layer: {legacy_layer_display}",
            )

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
                carousel_index_key = "detail_image_carousel_index"
                carousel_target_key = "detail_image_carousel_target_id"
                if str(st.session_state.get(carousel_target_key, "")).strip() != target_id:
                    st.session_state[carousel_target_key] = target_id
                    st.session_state[carousel_index_key] = 0

                active_image: dict[str, str] | None = None
                if image_candidates:
                    image_count = len(image_candidates)
                    try:
                        current_image_index = int(st.session_state.get(carousel_index_key, 0))
                    except (TypeError, ValueError):
                        current_image_index = 0
                    if current_image_index < 0 or current_image_index >= image_count:
                        current_image_index = 0
                        st.session_state[carousel_index_key] = 0

                    if image_count > 1:
                        prev_col, status_col, next_col = st.columns([1, 2, 1], gap="small")
                        prev_clicked = prev_col.button(
                            "Prev",
                            key="detail_image_carousel_prev",
                            use_container_width=True,
                        )
                        next_clicked = next_col.button(
                            "Next",
                            key="detail_image_carousel_next",
                            use_container_width=True,
                        )
                        if prev_clicked and not next_clicked:
                            current_image_index = (current_image_index - 1) % image_count
                            st.session_state[carousel_index_key] = current_image_index
                        elif next_clicked and not prev_clicked:
                            current_image_index = (current_image_index + 1) % image_count
                            st.session_state[carousel_index_key] = current_image_index
                        status_col.caption(
                            f"Image {current_image_index + 1}/{image_count}: "
                            f"{image_candidates[current_image_index].get('label', 'Image')}"
                        )
                    else:
                        st.caption(f"Image 1/1: {image_candidates[0].get('label', 'Image')}")

                    active_image = image_candidates[current_image_index]
                    active_image_url = clean_text(active_image.get("image_url"))
                    image_url_html = html.escape(active_image_url, quote=True)
                    image_tag = (
                        '<div style="width:200px; height:200px; max-width:100%; display:flex; align-items:center; justify-content:center;">'
                        f'<img src="{image_url_html}" '
                        'style="max-width:200px; max-height:200px; width:auto; height:auto; object-fit:contain; object-position:center;" />'
                        "</div>"
                    )
                    st.markdown(image_tag, unsafe_allow_html=True)
                else:
                    st.info("No image URL available for this target.")
                active_image_source_url = clean_text(active_image.get("source_url")) if active_image else ""
                active_image_license = clean_text(active_image.get("license_label")) if active_image else ""
                if active_image_source_url:
                    st.caption(f"Image source: [Open link]({active_image_source_url})")
                if info_url:
                    st.caption(f"Background: [Open object page]({info_url})")
                if active_image_license:
                    st.caption(f"License/Credit: {active_image_license}")

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

                selected_object_type_group = normalize_object_type_group(selected.get("object_type_group"))
                selected_object_type_value = clean_text(selected.get("object_type"))
                aliases_value = clean_text(selected.get("aliases"))
                frame_occupancy_value = "-"
                if telescope_fov_maj_deg is not None and telescope_fov_min_deg is not None:
                    frame_area_deg2 = float(telescope_fov_maj_deg * telescope_fov_min_deg)
                    object_maj_deg = (
                        float(ang_size_maj_arcmin_value) / 60.0
                        if ang_size_maj_arcmin_value is not None and float(ang_size_maj_arcmin_value) > 0.0
                        else None
                    )
                    object_min_deg = (
                        float(ang_size_min_arcmin_value) / 60.0
                        if ang_size_min_arcmin_value is not None and float(ang_size_min_arcmin_value) > 0.0
                        else None
                    )
                    if object_maj_deg is None and object_min_deg is not None:
                        object_maj_deg = object_min_deg
                    if object_min_deg is None and object_maj_deg is not None:
                        object_min_deg = object_maj_deg
                    if (
                        frame_area_deg2 > 0.0
                        and object_maj_deg is not None
                        and object_min_deg is not None
                        and object_maj_deg > 0.0
                        and object_min_deg > 0.0
                    ):
                        object_area_deg2 = float(object_maj_deg * object_min_deg)
                        occupancy_pct = (object_area_deg2 / frame_area_deg2) * 100.0
                        frame_occupancy_value = f"{occupancy_pct:.0f}% (Approx.)"
                combined_distance = "-"
                if dist_value and dist_unit:
                    combined_distance = f"{dist_value}/{dist_unit}"
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
                    {"Property": "Aliases", "Value": aliases_value or "-"},
                    {"Property": "Size in Frame", "Value": frame_occupancy_value},
                    {"Property": "Distance", "Value": combined_distance},
                    {"Property": "Redshift", "Value": redshift or "-"},
                    {"Property": "Angular Size", "Value": ang_size_display or "-", "Tooltip": ang_size_tooltip},
                    {"Property": "Morphology", "Value": morphology or "-"},
                    {"Property": "Emissions Details", "Value": emission_details_display or "-"},
                ]
                if selected_object_type_group == "other" and selected_object_type_value:
                    property_items.insert(
                        3,
                        {"Property": "Object Type", "Value": selected_object_type_value},
                    )
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
                "image_url": clean_text(preview_target.get("image_url")),
                "hero_image_url": clean_text(preview_target.get("hero_image_url")),
                "ra_deg": parse_numeric(preview_target.get("ra_deg")),
                "dec_deg": parse_numeric(preview_target.get("dec_deg")),
                "ang_size_maj_arcmin": parse_numeric(preview_target.get("ang_size_maj_arcmin")),
                "ang_size_min_arcmin": parse_numeric(preview_target.get("ang_size_min_arcmin")),
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
    moon_track = _compute_moon_track_for_window(
        start_local=window_start,
        end_local=window_end,
        tz_name=tzinfo.key,
    )
    moon_phase_key = _compute_moon_phase_key_for_window(
        start_local=window_start,
        end_local=window_end,
        tz_name=tzinfo.key,
    )
    lunar_eclipse_visibility = _compute_lunar_eclipse_visibility_for_window(
        start_local=window_start,
        end_local=window_end,
        tz_name=tzinfo.key,
    )
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
                mount_choice=plot_mount_choice,
                moon_track=moon_track,
                moon_phase_key=moon_phase_key,
                eclipse_visibility=lunar_eclipse_visibility,
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
                mount_choice=plot_mount_choice,
                moon_track=moon_track,
                moon_phase_key=moon_phase_key,
                eclipse_visibility=lunar_eclipse_visibility,
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
            selected_metadata={
                "common_name": clean_text(selected.get("common_name")),
                "image_url": clean_text(selected.get("image_url")),
                "hero_image_url": clean_text(selected.get("hero_image_url")),
                "ra_deg": parse_numeric(selected.get("ra_deg")),
                "dec_deg": parse_numeric(selected.get("dec_deg")),
                "ang_size_maj_arcmin": parse_numeric(selected.get("ang_size_maj_arcmin")),
                "ang_size_min_arcmin": parse_numeric(selected.get("ang_size_min_arcmin")),
            },
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
                    mount_choice=plot_mount_choice,
                    moon_track=moon_track,
                    moon_phase_key=moon_phase_key,
                    eclipse_visibility=lunar_eclipse_visibility,
                ),
                use_container_width=True,
                key="detail_unobstructed_area_plot",
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
    _refresh_legacy_globals()
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
            sync_runtime_state = str(
                st.session_state.get(globals().get("GOOGLE_DRIVE_SYNC_STATE_STATE_KEY", ""), "")
            ).strip().lower()
            deferred_sync_action = str(
                st.session_state.get(globals().get("GOOGLE_DRIVE_SYNC_DEFERRED_ACTION_STATE_KEY", ""), "")
            ).strip().lower()
            show_sidebar_reauth = sync_runtime_state == "reauth_required" or bool(deferred_sync_action)
            if show_sidebar_reauth:
                st.caption(
                    "Cloud sync needs Google reauth (this starts with a Google sign-out)."
                    if sync_runtime_state == "reauth_required"
                    else "Reconnect Google to resume deferred cloud sync (sign-out/sign-in)."
                )
                authlib_available = bool(globals().get("AUTHLIB_AVAILABLE", True))
                is_logged_in = False
                try:
                    is_logged_in = bool(_is_user_logged_in())
                except Exception:
                    is_logged_in = False
                button_label = (
                    "Reconnect Google (Logs Out First)"
                    if is_logged_in
                    else "Sign In to Resume Sync"
                )
                if st.button(
                    button_label,
                    key="sidebar_google_reauth_resume",
                    use_container_width=True,
                    disabled=not authlib_available,
                    help=(
                        "Signs you out of Google first; after that, sign back in to resume cloud sync."
                        if is_logged_in
                        else "Sign in with Google and resume deferred cloud sync."
                    ),
                ):
                    try:
                        if is_logged_in:
                            st.logout()
                        else:
                            st.login("google")
                    except Exception as exc:
                        st.warning(f"Google auth action failed: {str(exc).strip()}")
    selected_ui_theme = theme_label_to_id[selected_theme_label]
    if selected_ui_theme != current_theme:
        prefs["ui_theme"] = selected_ui_theme
        persist_and_rerun(prefs)
    return selected_ui_theme
