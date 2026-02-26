from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import streamlit as st


@dataclass(frozen=True)
class SettingsPageDeps:
    temperature_unit_options: dict[str, str]
    persist_and_rerun: Callable[[dict[str, Any]], None]
    resolve_temperature_unit: Callable[[str, str | None], str]
    is_user_logged_in: Callable[[], bool]
    authlib_available: bool
    get_user_claim: Callable[[str], str]
    cloud_sync_provider_google: str
    cloud_sync_provider_none: str
    get_google_access_token: Callable[[], str]
    google_drive_sync_pending_state_key: str
    google_drive_sync_bootstrap_state_key: str
    google_drive_sync_last_remote_file_modified_state_key: str
    google_drive_sync_last_remote_payload_updated_state_key: str
    google_drive_sync_last_action_state_key: str
    google_drive_sync_last_compare_summary_state_key: str
    google_drive_sync_manual_action_state_key: str
    google_drive_sync_state_state_key: str
    google_drive_sync_deferred_action_state_key: str
    google_drive_sync_merge_candidate_state_key: str
    google_drive_sync_merge_resolution_action_state_key: str
    google_drive_sync_merge_selection_state_key: str
    catalog_cache_path_display: str
    build_settings_export_payload: Callable[[dict[str, Any]], dict[str, Any]]
    parse_settings_import_payload: Callable[[str], dict[str, Any] | None]
    site_ids_in_order: Callable[[dict[str, Any]], list[str]]
    list_ids_in_order: Callable[..., list[str]]


def _normalize_merge_candidate(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    additions = value.get("additions")
    if not isinstance(additions, dict):
        return None
    return value


def _merge_candidate_has_items(candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    additions = candidate.get("additions", {})
    if not isinstance(additions, dict):
        return False
    if additions.get("sites"):
        return True
    if additions.get("lists"):
        return True
    equipment = additions.get("equipment", {})
    if isinstance(equipment, dict):
        for values in equipment.values():
            if isinstance(values, list) and values:
                return True
    return False


def render_settings_page(
    catalog_meta: dict[str, Any],
    prefs: dict[str, Any],
    browser_locale: str | None,
    *,
    deps: SettingsPageDeps,
) -> None:
    st.title("Settings")
    st.caption("Manage display preferences, catalog metadata, and settings backup.")

    persistence_notice = st.session_state.get("prefs_persistence_notice", "")
    if persistence_notice:
        st.warning(persistence_notice)
    import_notice = str(st.session_state.pop("settings_import_notice", "")).strip()
    if import_notice:
        st.success(import_notice)
    st.subheader("Display")
    labels = list(deps.temperature_unit_options.keys())
    reverse_options = {value: label for label, value in deps.temperature_unit_options.items()}
    current_pref = str(prefs.get("temperature_unit", "auto")).lower()
    if current_pref not in reverse_options:
        current_pref = "auto"
    selected_label = st.selectbox(
        "Temperature units",
        options=labels,
        index=labels.index(reverse_options[current_pref]),
        key="temperature_unit_preference",
    )
    selected_pref = deps.temperature_unit_options[selected_label]
    if selected_pref != current_pref:
        prefs["temperature_unit"] = selected_pref
        deps.persist_and_rerun(prefs)

    effective_unit = deps.resolve_temperature_unit(selected_pref, browser_locale)
    source_note = "browser locale" if selected_pref == "auto" else "manual setting"
    st.caption(f"Active temperature unit: {effective_unit.upper()} ({source_note})")

    st.divider()
    st.subheader("Cloud Sync (Google Drive)")
    cloud_sync_notice = str(st.session_state.pop("cloud_sync_notice", "")).strip()
    if cloud_sync_notice:
        st.success(cloud_sync_notice)
    is_logged_in = deps.is_user_logged_in()
    deferred_sync_action = str(st.session_state.get(deps.google_drive_sync_deferred_action_state_key, "")).strip().lower()
    if not is_logged_in:
        st.caption("Sign in with Google to sync preferences and session state across devices.")
        if deferred_sync_action:
            st.caption(f"Deferred sync action will resume after sign-in: {deferred_sync_action}")
        if not deps.authlib_available:
            st.warning(
                "Google sign-in is unavailable because Authlib is not installed in this environment. "
                "Install dependencies from requirements.txt or run `pip install Authlib>=1.3.2`."
            )
        if st.button(
            "Sign in with Google",
            key="settings_google_sign_in",
            use_container_width=True,
            disabled=not deps.authlib_available,
        ):
            try:
                st.login("google")
            except Exception as exc:
                st.warning(f"Google sign-in could not start: {str(exc).strip()}")
    else:
        account_email = deps.get_user_claim("email")
        if account_email:
            st.caption(f"Signed in as {account_email}")
        else:
            st.caption("Signed in with Google")
        if st.button("Sign out of Google", key="settings_google_sign_out", use_container_width=True):
            st.logout()

        cloud_sync_enabled = bool(prefs.get("cloud_sync_enabled", False))
        next_cloud_sync_enabled = st.toggle(
            "Save settings to Google Drive",
            value=cloud_sync_enabled,
            key="settings_cloud_sync_enabled",
        )
        if next_cloud_sync_enabled != cloud_sync_enabled:
            prefs["cloud_sync_provider"] = (
                deps.cloud_sync_provider_google if next_cloud_sync_enabled else deps.cloud_sync_provider_none
            )
            prefs["cloud_sync_enabled"] = next_cloud_sync_enabled
            prefs["cloud_sync_last_error"] = ""
            deps.persist_and_rerun(prefs)

        cloud_file_id = str(prefs.get("cloud_sync_file_id", "")).strip()
        local_last_saved = str(prefs.get("last_updated_utc", "")).strip()
        cloud_last_ok = str(prefs.get("cloud_sync_last_ok_utc", "")).strip()
        cloud_last_error = str(prefs.get("cloud_sync_last_error", "")).strip()
        pending_cloud_sync = bool(st.session_state.get(deps.google_drive_sync_pending_state_key, False))
        cloud_bootstrapped = bool(st.session_state.get(deps.google_drive_sync_bootstrap_state_key, False))
        sync_runtime_state = str(st.session_state.get(deps.google_drive_sync_state_state_key, "")).strip().lower()
        remote_file_modified_seen = str(
            st.session_state.get(deps.google_drive_sync_last_remote_file_modified_state_key, "")
        ).strip()
        remote_payload_updated_seen = str(
            st.session_state.get(deps.google_drive_sync_last_remote_payload_updated_state_key, "")
        ).strip()
        last_sync_action = str(st.session_state.get(deps.google_drive_sync_last_action_state_key, "")).strip()
        last_compare_summary = str(st.session_state.get(deps.google_drive_sync_last_compare_summary_state_key, "")).strip()
        merge_candidate = _normalize_merge_candidate(
            st.session_state.get(deps.google_drive_sync_merge_candidate_state_key)
        )
        merge_decision_required = _merge_candidate_has_items(merge_candidate)
        if local_last_saved:
            st.caption(f"Local settings last saved: {local_last_saved}")
        if cloud_last_ok:
            st.caption(f"Last successful cloud sync (app timestamp): {cloud_last_ok}")
        if remote_payload_updated_seen:
            st.caption(f"Latest cloud payload timestamp seen: {remote_payload_updated_seen}")
        if remote_file_modified_seen:
            st.caption(f"Latest Google Drive file modified time seen: {remote_file_modified_seen}")
        st.caption(
            "Cloud sync state: "
            f"pending_upload={'yes' if pending_cloud_sync else 'no'} | "
            f"bootstrapped_this_session={'yes' if cloud_bootstrapped else 'no'}"
        )
        if sync_runtime_state:
            st.caption(f"Sync runtime status: {sync_runtime_state}")
        if deferred_sync_action:
            st.caption(f"Deferred sync action after reconnect: {deferred_sync_action}")
        if cloud_file_id:
            st.caption(f"Cloud settings file: {cloud_file_id}")
        if last_compare_summary:
            st.caption(f"Last compare result: {last_compare_summary}")
        if last_sync_action:
            st.caption(f"Last sync action: {last_sync_action}")
        if cloud_last_error:
            st.warning(f"Last cloud sync error: {cloud_last_error}")

        token_available = bool(deps.get_google_access_token())
        if not token_available:
            st.warning(
                "Google access token unavailable. Configure auth with expose_tokens=[\"access\"] "
                "and include Drive appData scope."
            )
        if sync_runtime_state == "reauth_required":
            st.warning(
                "Google session/token appears stale. Reconnect Google auth (sign out/in) to resume cloud sync."
            )
            reconnect_col, reconnect_note_col = st.columns([1, 2], gap="small")
            if reconnect_col.button(
                "Reconnect Google (Logs Out First)",
                key="settings_google_reconnect_now",
                use_container_width=True,
                help="This signs you out of Google first; then sign back in to resume cloud sync.",
            ):
                st.logout()
            reconnect_note_col.caption(
                "This button signs you out first. Then sign back in; pending cloud sync will resume after sign-in."
            )
        if merge_decision_required and isinstance(merge_candidate, dict):
            additions = merge_candidate.get("additions", {})
            summary = merge_candidate.get("summary", {})
            remote_account = str(merge_candidate.get("account_email") or merge_candidate.get("account_sub") or "").strip()
            summary_bits: list[str] = []
            for label, key in (
                ("sites", "site_count"),
                ("equipment", "equipment_count"),
                ("lists", "list_count"),
            ):
                try:
                    count = int(summary.get(key, 0))
                except Exception:
                    count = 0
                if count > 0:
                    summary_bits.append(f"{count} {label}")
            summary_text = ", ".join(summary_bits) if summary_bits else "local additions"
            st.info(
                "Cloud sync needs a merge decision before upload. "
                f"Local session includes {summary_text} not present in the cloud profile."
            )
            if remote_account:
                st.caption(f"Cloud profile account: {remote_account}")

            selected_sites: list[str] = []
            selected_lists: list[str] = []
            selected_equipment: dict[str, list[str]] = {}
            with st.container(border=True):
                st.caption("Choose which local additions to add to your cloud profile.")
                site_items = additions.get("sites", [])
                if isinstance(site_items, list) and site_items:
                    st.markdown("**Sites**")
                    for item in site_items:
                        if not isinstance(item, dict):
                            continue
                        site_id = str(item.get("id", "")).strip()
                        if not site_id:
                            continue
                        label = str(item.get("label", "")).strip() or site_id
                        if st.checkbox(label, value=True, key=f"settings_sync_merge_site_{site_id}"):
                            selected_sites.append(site_id)

                list_items = additions.get("lists", [])
                if isinstance(list_items, list) and list_items:
                    st.markdown("**Lists**")
                    for item in list_items:
                        if not isinstance(item, dict):
                            continue
                        list_id = str(item.get("id", "")).strip()
                        if not list_id:
                            continue
                        label = str(item.get("label", "")).strip() or list_id
                        if st.checkbox(label, value=True, key=f"settings_sync_merge_list_{list_id}"):
                            selected_lists.append(list_id)

                equipment_items = additions.get("equipment", {})
                if isinstance(equipment_items, dict) and equipment_items:
                    st.markdown("**Equipment**")
                    for raw_category_id, raw_values in sorted(equipment_items.items()):
                        category_id = str(raw_category_id).strip()
                        if not category_id:
                            continue
                        values = raw_values if isinstance(raw_values, list) else []
                        if not values:
                            continue
                        category_label = category_id.replace("_", " ").title()
                        st.caption(category_label)
                        for raw_item in values:
                            item = raw_item if isinstance(raw_item, dict) else {}
                            item_id = str(item.get("id", "")).strip()
                            if not item_id:
                                continue
                            label = str(item.get("label", "")).strip() or item_id
                            if st.checkbox(
                                label,
                                value=True,
                                key=f"settings_sync_merge_equipment_{category_id}_{item_id}",
                            ):
                                selected_equipment.setdefault(category_id, []).append(item_id)

                merge_use_cloud_col, merge_selected_col = st.columns([1, 1], gap="small")
                use_cloud_clicked = merge_use_cloud_col.button(
                    "Use Cloud Profile Only",
                    key="settings_sync_merge_use_cloud_only",
                    use_container_width=True,
                    help="Apply the cloud profile locally and discard local additions from this browser session.",
                )
                merge_selected_clicked = merge_selected_col.button(
                    "Merge Selected Into Cloud",
                    key="settings_sync_merge_apply_selected",
                    use_container_width=True,
                    help="Add selected local sites/equipment/lists to the cloud profile, then upload the merged profile.",
                )

            if use_cloud_clicked:
                st.session_state[deps.google_drive_sync_merge_selection_state_key] = {}
                st.session_state[deps.google_drive_sync_merge_resolution_action_state_key] = "use_cloud"
                st.session_state[deps.google_drive_sync_bootstrap_state_key] = False
                st.rerun()

            if merge_selected_clicked:
                selected_any = bool(selected_sites or selected_lists or any(selected_equipment.values()))
                if not selected_any:
                    st.warning("Select at least one local item to merge, or choose 'Use Cloud Profile Only'.")
                else:
                    st.session_state[deps.google_drive_sync_merge_selection_state_key] = {
                        "sites": selected_sites,
                        "lists": selected_lists,
                        "equipment": selected_equipment,
                    }
                    st.session_state[deps.google_drive_sync_merge_resolution_action_state_key] = "merge_selected"
                    st.session_state[deps.google_drive_sync_bootstrap_state_key] = False
                    st.rerun()

        sync_controls_disabled = (
            (not token_available)
            or (not bool(next_cloud_sync_enabled))
            or merge_decision_required
            or (sync_runtime_state == "reauth_required")
        )
        pull_col, push_col = st.columns([1, 1], gap="small")
        pull_clicked = pull_col.button(
            "Pull from Google Drive",
            key="settings_cloud_sync_pull_now",
            use_container_width=True,
            disabled=sync_controls_disabled,
            help="Fetch cloud settings and apply them locally (latest cloud snapshot wins).",
        )
        push_clicked = push_col.button(
            "Push to Google Drive",
            key="settings_cloud_sync_push_now",
            use_container_width=True,
            disabled=sync_controls_disabled,
            help="Upload the current local settings and session snapshot to Google Drive.",
        )
        if pull_clicked:
            st.session_state[deps.google_drive_sync_manual_action_state_key] = "pull"
            st.session_state[deps.google_drive_sync_bootstrap_state_key] = False
            st.session_state[deps.google_drive_sync_pending_state_key] = False
            st.rerun()
        if push_clicked:
            st.session_state[deps.google_drive_sync_manual_action_state_key] = "push"
            st.session_state[deps.google_drive_sync_bootstrap_state_key] = True
            st.session_state[deps.google_drive_sync_pending_state_key] = True
            st.rerun()

    st.divider()
    st.subheader("Catalog")
    st.caption(
        f"Rows: {int(catalog_meta.get('row_count', 0))} | "
        f"Source: {catalog_meta.get('source', deps.catalog_cache_path_display)}"
    )
    catalog_counts = catalog_meta.get("catalog_counts", {})
    if isinstance(catalog_counts, dict) and catalog_counts:
        counts_line = " | ".join(f"{catalog_name}: {count}" for catalog_name, count in sorted(catalog_counts.items()))
        st.caption(counts_line)
    catalog_filters = catalog_meta.get("filters", {})
    if isinstance(catalog_filters, dict):
        catalogs_count = len(catalog_filters.get("catalogs", []))
        object_types_count = len(catalog_filters.get("object_types", []))
        constellations_count = len(catalog_filters.get("constellations", []))
        st.caption(
            "Filter options: "
            f"catalogs={catalogs_count} | object types={object_types_count} | constellations={constellations_count}"
        )
    loaded_at = str(catalog_meta.get("loaded_at_utc", "")).strip()
    if loaded_at:
        st.caption(f"Loaded: {loaded_at}")
    validation = catalog_meta.get("validation", {})
    if isinstance(validation, dict):
        row_count = int(validation.get("row_count", 0))
        unique_ids = int(validation.get("unique_primary_id_count", 0))
        duplicate_ids = int(validation.get("duplicate_primary_id_count", 0))
        blank_ids = int(validation.get("blank_primary_id_count", 0))
        st.caption(
            "Validation: "
            f"rows={row_count} | unique_ids={unique_ids} | duplicate_ids={duplicate_ids} | blank_ids={blank_ids}"
        )
        warnings = validation.get("warnings", [])
        if isinstance(warnings, list):
            for warning in warnings:
                warning_text = str(warning).strip()
                if warning_text:
                    st.warning(warning_text)

    st.divider()
    st.subheader("Settings Backup / Restore")
    export_payload = deps.build_settings_export_payload(prefs)
    export_text = json.dumps(export_payload, indent=2)
    export_filename = f"dso-explorer-settings-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    st.download_button(
        "Export settings JSON",
        data=export_text,
        file_name=export_filename,
        mime="application/json",
        use_container_width=True,
    )

    uploaded_settings = st.file_uploader(
        "Import settings JSON",
        type=["json"],
        key="settings_import_file",
        help="Imports sites, obstructions, equipment, lists, and display preferences.",
    )
    if st.button("Import settings", use_container_width=True, key="settings_import_apply"):
        if uploaded_settings is None:
            st.warning("Choose a JSON file first.")
        else:
            raw_bytes = uploaded_settings.getvalue()
            try:
                raw_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                st.warning("Could not read that file as UTF-8 JSON.")
            else:
                imported_prefs = deps.parse_settings_import_payload(raw_text)
                if imported_prefs is None:
                    st.warning("Invalid settings file format.")
                else:
                    imported_site_count = len(deps.site_ids_in_order(imported_prefs))
                    imported_list_count = len(deps.list_ids_in_order(imported_prefs, include_auto_recent=True))
                    imported_equipment_payload = imported_prefs.get("equipment", {})
                    imported_equipment_count = 0
                    if isinstance(imported_equipment_payload, dict):
                        imported_equipment_count = sum(
                            len(items)
                            for items in imported_equipment_payload.values()
                            if isinstance(items, list)
                        )
                    st.session_state["location_notice"] = "Settings imported."
                    st.session_state["settings_import_notice"] = (
                        "Settings imported: "
                        f"{imported_site_count} site(s), "
                        f"{imported_list_count} list(s), "
                        f"{imported_equipment_count} equipment item(s)."
                    )
                    deps.persist_and_rerun(imported_prefs)
