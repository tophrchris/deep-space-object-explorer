from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from lists.list_subsystem import (
    create_list,
    delete_list,
    editable_list_ids_in_order,
    get_list_ids,
    get_list_name,
    is_system_list,
    list_ids_in_order,
    move_list,
    rename_list,
)


def render_lists_settings_section(
    prefs: dict[str, Any],
    *,
    persist_and_rerun_fn: Callable[[dict[str, Any]], None],
    show_subheader: bool = True,
) -> None:
    if show_subheader:
        st.subheader("Lists")
    ordered_list_ids = list_ids_in_order(prefs, include_auto_recent=True)
    if ordered_list_ids:
        list_rows: list[dict[str, Any]] = []
        for index, list_id in enumerate(ordered_list_ids, start=1):
            list_rows.append(
                {
                    "Order": index,
                    "List": get_list_name(prefs, list_id),
                    "Items": len(get_list_ids(prefs, list_id)),
                    "Type": "System" if is_system_list(prefs, list_id) else "Editable",
                    "ID": list_id,
                }
            )
        list_frame = pd.DataFrame(list_rows)
        list_table_height = max(72, min(320, 36 * (len(list_frame) + 1)))
        st.dataframe(
            list_frame,
            hide_index=True,
            use_container_width=True,
            height=list_table_height,
        )
    st.caption("`Auto (Recent)` is system-managed and tracks the last 10 search selections.")

    reset_new_name_key = "lists_new_name_reset"
    if bool(st.session_state.pop(reset_new_name_key, False)):
        st.session_state["lists_new_name"] = ""

    create_cols = st.columns([3, 1], gap="small")
    with create_cols[0]:
        new_list_name = st.text_input("New list name", key="lists_new_name")
    with create_cols[1]:
        st.write("")
        if st.button("Create list", key="lists_create_button", use_container_width=True):
            created_list_id = create_list(prefs, new_list_name)
            if created_list_id:
                st.session_state["lists_manage_selected_id"] = created_list_id
                st.session_state[reset_new_name_key] = True
                persist_and_rerun_fn(prefs)
            else:
                st.warning("Enter a list name to create a new list.")

    editable_list_ids = editable_list_ids_in_order(prefs)
    if editable_list_ids:
        manage_select_key = "lists_manage_selected_id"
        current_manage_selection = str(st.session_state.get(manage_select_key, "")).strip()
        if current_manage_selection not in editable_list_ids:
            st.session_state[manage_select_key] = editable_list_ids[0]
            current_manage_selection = editable_list_ids[0]

        selected_manage_list_id = st.selectbox(
            "Manage editable list",
            options=editable_list_ids,
            index=editable_list_ids.index(current_manage_selection),
            key=manage_select_key,
            format_func=lambda list_id: get_list_name(prefs, list_id),
        )

        manage_name_key = "lists_manage_name_input"
        manage_name_for_id_key = "lists_manage_name_for_id"
        if str(st.session_state.get(manage_name_for_id_key, "")).strip() != selected_manage_list_id:
            st.session_state[manage_name_key] = get_list_name(prefs, selected_manage_list_id)
            st.session_state[manage_name_for_id_key] = selected_manage_list_id

        rename_value = st.text_input("Selected list name", key=manage_name_key)
        selected_manage_idx = editable_list_ids.index(selected_manage_list_id)
        control_cols = st.columns(4, gap="small")

        if control_cols[0].button("Rename list", key="lists_rename_button", use_container_width=True):
            if rename_list(prefs, selected_manage_list_id, rename_value):
                persist_and_rerun_fn(prefs)
            else:
                st.warning("List name unchanged or invalid.")

        if control_cols[1].button(
            "Move up",
            key="lists_move_up_button",
            use_container_width=True,
            disabled=(selected_manage_idx == 0),
        ):
            if move_list(prefs, selected_manage_list_id, -1):
                persist_and_rerun_fn(prefs)

        if control_cols[2].button(
            "Move down",
            key="lists_move_down_button",
            use_container_width=True,
            disabled=(selected_manage_idx >= (len(editable_list_ids) - 1)),
        ):
            if move_list(prefs, selected_manage_list_id, 1):
                persist_and_rerun_fn(prefs)

        if control_cols[3].button("Delete list", key="lists_delete_button", use_container_width=True):
            if delete_list(prefs, selected_manage_list_id):
                st.session_state.pop(manage_name_for_id_key, None)
                st.session_state.pop(manage_name_key, None)
                st.session_state.pop(manage_select_key, None)
                persist_and_rerun_fn(prefs)
            else:
                st.warning("Could not delete that list.")
    else:
        st.caption("Create your first editable list to enable Add to list actions.")
