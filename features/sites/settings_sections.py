from __future__ import annotations

# Transitional bridge during Sites split: this module still relies on shared
# helpers/constants from `ui.streamlit_app` until they are extracted.
from ui import streamlit_app as _legacy_ui

def _refresh_legacy_globals() -> None:
    for _name, _value in vars(_legacy_ui).items():
        if _name == "__builtins__":
            continue
        globals().setdefault(_name, _value)


_refresh_legacy_globals()

def render_location_settings_section(prefs: dict[str, Any]) -> None:
    _refresh_legacy_globals()
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
    _refresh_legacy_globals()
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


