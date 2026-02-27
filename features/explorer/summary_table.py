from __future__ import annotations

# Transitional bridge during Explorer split: this module still relies on shared
# helpers/constants from `ui.streamlit_app` until they are extracted.
from ui import streamlit_app as _legacy_ui

def _refresh_legacy_globals() -> None:
    for _name, _value in vars(_legacy_ui).items():
        if _name == "__builtins__":
            continue
        globals().setdefault(_name, _value)


_refresh_legacy_globals()

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
    _refresh_legacy_globals()
    if not rows:
        st.session_state["sky_summary_highlight_primary_id"] = ""
        return

    st.markdown("#### Targets")
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        st.session_state["sky_summary_highlight_primary_id"] = ""
        return

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

    def _safe_positive_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric <= 0.0:
            return None
        return float(numeric)

    summary_df["thumbnail_url"] = summary_df.apply(_resolve_thumbnail_url, axis=1)
    summary_df["apparent_size"] = summary_df.apply(
        lambda row: format_apparent_size_display(
            row.get("ang_size_maj_arcmin"),
            row.get("ang_size_min_arcmin"),
        ),
        axis=1,
    )

    selected_telescope: dict[str, Any] | None = None
    telescope_fov_area: float | None = None
    try:
        equipment_context = build_owned_equipment_context(prefs)
        active_equipment = sync_active_equipment_settings(prefs, equipment_context)
        active_telescope = active_equipment.get("active_telescope")
        if not isinstance(active_telescope, dict):
            telescope_lookup = dict(active_equipment.get("telescope_lookup", {}))
            active_telescope_id = str(active_equipment.get("active_telescope_id", "")).strip()
            active_telescope = telescope_lookup.get(active_telescope_id) if active_telescope_id else None
        if isinstance(active_telescope, dict):
            telescope_fov_maj = _safe_positive_float(active_telescope.get("fov_maj_deg"))
            telescope_fov_min = _safe_positive_float(active_telescope.get("fov_min_deg"))
            if telescope_fov_maj is not None and telescope_fov_min is not None:
                selected_telescope = active_telescope
                telescope_fov_area = float(telescope_fov_maj * telescope_fov_min)
    except Exception:
        selected_telescope = None
        telescope_fov_area = None

    summary_df["framing_percent"] = np.nan
    show_framing_column = (
        selected_telescope is not None
        and telescope_fov_area is not None
        and telescope_fov_area > 0.0
    )
    if show_framing_column:
        target_maj_deg = pd.to_numeric(summary_df["ang_size_maj_arcmin"], errors="coerce") / 60.0
        target_min_deg = pd.to_numeric(summary_df["ang_size_min_arcmin"], errors="coerce") / 60.0
        target_maj_deg = target_maj_deg.where(target_maj_deg > 0.0)
        target_min_deg = target_min_deg.where(target_min_deg > 0.0)
        target_maj_deg = target_maj_deg.fillna(target_min_deg)
        target_min_deg = target_min_deg.fillna(target_maj_deg)
        target_area_deg2 = target_maj_deg * target_min_deg
        summary_df["framing_percent"] = (target_area_deg2 / float(telescope_fov_area)) * 100.0

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
        "thumbnail_url",
        "target",
        "object_type_group",
        "first_visible",
        "culmination",
        "last_visible",
        "visible_total",
    ]
    if show_framing_column:
        display_columns.append("framing_percent")
    else:
        display_columns.append("apparent_size")
    if show_remaining:
        display_columns.append("visible_remaining_display")
    display_columns.extend(["culmination_alt", "culmination_dir", "list_action"])

    display = summary_df[display_columns].rename(
        columns={
            "thumbnail_url": "Thumbnail",
            "line_swatch": "Line",
            "target": "Target",
            "object_type_group": "Type",
            "apparent_size": "Apparent size",
            "framing_percent": "Framing",
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
        "Thumbnail": (
            st.column_config.ImageColumn(label="", width=100)
            if hasattr(st.column_config, "ImageColumn")
            else st.column_config.TextColumn(label="", width=100)
        ),
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
    if "Apparent size" in display.columns:
        column_config["Apparent size"] = st.column_config.TextColumn(width="small")
    if "Framing" in display.columns:
        column_config["Framing"] = st.column_config.NumberColumn(width="small", format="%.0f%%")
    if show_remaining:
        column_config["Remaining"] = st.column_config.TextColumn(width="small")

    styled = styled.set_properties(
        subset=["Thumbnail"],
        **{
            "text-align": "left !important",
            "justify-content": "flex-start !important",
            "padding-left": "0px !important",
        },
    )

    table_event = st.dataframe(
        styled,
        hide_index=True,
        use_container_width=True,
        row_height=70,
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
