from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class SitesPageDeps:
    default_site_id: str
    site_ids_in_order: Callable[[dict[str, Any]], list[str]]
    get_active_site_id: Callable[[dict[str, Any]], str]
    set_active_site: Callable[[dict[str, Any], str], bool]
    get_site_definition: Callable[[dict[str, Any], str], dict[str, Any]]
    resolve_location_source_badge: Callable[[Any], Any]
    get_site_name: Callable[[dict[str, Any], str], str]
    create_site: Callable[[dict[str, Any]], str | None]
    persist_and_rerun: Callable[[dict[str, Any]], None]
    duplicate_site: Callable[[dict[str, Any], str], str | None]
    delete_site: Callable[[dict[str, Any], str], bool]
    sync_active_site_to_legacy_fields: Callable[[dict[str, Any]], None]
    render_location_settings_section: Callable[[dict[str, Any]], None]
    render_obstructions_settings_section: Callable[[dict[str, Any]], None]


def render_sites_page(prefs: dict[str, Any], *, deps: SitesPageDeps) -> None:
    st.title("Observation Sites")
    st.caption("Manage sites. One site is active at a time and used across Explorer weather/visibility calculations.")

    persistence_notice = st.session_state.get("prefs_persistence_notice", "")
    if persistence_notice:
        st.warning(persistence_notice)

    location_notice = st.session_state.pop("location_notice", "")
    if location_notice:
        st.info(location_notice)

    site_ids = deps.site_ids_in_order(prefs)
    if not site_ids:
        site_ids = [deps.default_site_id]
    active_site_id = deps.get_active_site_id(prefs)
    if active_site_id not in site_ids and site_ids:
        active_site_id = site_ids[0]
        deps.set_active_site(prefs, active_site_id)

    with st.container(border=True):
        st.subheader("Sites")
        rows: list[dict[str, str]] = []
        for site_id in site_ids:
            site = deps.get_site_definition(prefs, site_id)
            location = site.get("location", {})
            lat_text = "-"
            lon_text = "-"
            try:
                lat_text = f"{float(location.get('lat')):.4f}"
                lon_text = f"{float(location.get('lon')):.4f}"
            except (TypeError, ValueError):
                pass
            source_badge = deps.resolve_location_source_badge(location.get("source"))
            source_label = source_badge[0] if source_badge is not None else "-"
            rows.append(
                {
                    "Active": "●" if site_id == active_site_id else "",
                    "Name": deps.get_site_name(prefs, site_id),
                    "Latitude": lat_text,
                    "Longitude": lon_text,
                    "Source": source_label,
                }
            )
        site_frame = pd.DataFrame(rows)
        table_height = max(72, min(280, 36 * (len(site_frame) + 1)))
        st.dataframe(site_frame, hide_index=True, use_container_width=True, height=table_height)

        select_key = "sites_selected_site_id"
        pending_select_key = "sites_selected_site_id_pending"
        pending_selected = str(st.session_state.pop(pending_select_key, "")).strip()
        if pending_selected in site_ids:
            st.session_state[select_key] = pending_selected
        selected_default = str(st.session_state.get(select_key, "")).strip()
        if selected_default not in site_ids and site_ids:
            selected_default = active_site_id
            st.session_state[select_key] = selected_default
        selected_site_id = st.selectbox(
            "Selected site",
            options=site_ids,
            index=site_ids.index(selected_default) if selected_default in site_ids else 0,
            key=select_key,
            format_func=lambda site_id: deps.get_site_name(prefs, site_id),
        )

        add_col, edit_col, duplicate_col, delete_col = st.columns(4, gap="small")
        if add_col.button("Add new site", use_container_width=True, key="sites_add_button"):
            created_site_id = deps.create_site(prefs)
            if created_site_id:
                deps.set_active_site(prefs, created_site_id)
                st.session_state[pending_select_key] = created_site_id
                st.session_state["location_notice"] = f"Created new site: {deps.get_site_name(prefs, created_site_id)}."
                deps.persist_and_rerun(prefs)

        if edit_col.button("Edit", use_container_width=True, key="sites_edit_button"):
            changed = deps.set_active_site(prefs, selected_site_id)
            if changed:
                st.session_state["location_notice"] = f"Active site set to: {deps.get_site_name(prefs, selected_site_id)}."
                deps.persist_and_rerun(prefs)

        if duplicate_col.button("Duplicate", use_container_width=True, key="sites_duplicate_button"):
            duplicated_site_id = deps.duplicate_site(prefs, selected_site_id)
            if duplicated_site_id:
                deps.set_active_site(prefs, duplicated_site_id)
                st.session_state[pending_select_key] = duplicated_site_id
                st.session_state["location_notice"] = (
                    f"Created duplicate site: {deps.get_site_name(prefs, duplicated_site_id)}."
                )
                deps.persist_and_rerun(prefs)

        can_delete = len(site_ids) > 1
        if delete_col.button(
            "Delete",
            use_container_width=True,
            key="sites_delete_button",
            disabled=not can_delete,
        ):
            st.session_state["sites_delete_pending_id"] = selected_site_id

        pending_delete_id = str(st.session_state.get("sites_delete_pending_id", "")).strip()
        if pending_delete_id and pending_delete_id in site_ids:
            st.warning(f"Delete site '{deps.get_site_name(prefs, pending_delete_id)}'? This cannot be undone.")
            confirm_col, cancel_col = st.columns(2, gap="small")
            if confirm_col.button("Confirm Delete", use_container_width=True, key="sites_delete_confirm_button"):
                deleted_name = deps.get_site_name(prefs, pending_delete_id)
                if deps.delete_site(prefs, pending_delete_id):
                    st.session_state.pop("sites_delete_pending_id", None)
                    st.session_state["location_notice"] = f"Deleted site: {deleted_name}."
                    deps.persist_and_rerun(prefs)
            if cancel_col.button("Cancel", use_container_width=True, key="sites_delete_cancel_button"):
                st.session_state.pop("sites_delete_pending_id", None)

    deps.sync_active_site_to_legacy_fields(prefs)
    active_site_name = deps.get_site_name(prefs, deps.get_active_site_id(prefs))
    st.caption(f"Editing active site: {active_site_name}")

    with st.container(border=True):
        deps.render_location_settings_section(prefs)

    with st.container(border=True):
        deps.render_obstructions_settings_section(prefs)
