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

