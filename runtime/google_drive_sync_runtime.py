from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from runtime.google_drive_sync import GoogleDriveAPIError

# Transitional bridge during runtime/service split: this module still relies on shared
# helpers/constants from `ui.streamlit_app` until they are extracted.
from ui import streamlit_app as _legacy_ui


def _refresh_legacy_globals() -> None:
    for _name, _value in vars(_legacy_ui).items():
        if _name == "__builtins__":
            continue
        globals().setdefault(_name, _value)


_refresh_legacy_globals()

_SYNC_STATE_UNAUTHENTICATED = "unauthenticated"
_SYNC_STATE_AUTH_PRESENT_NO_TOKEN = "auth_present_no_token"
_SYNC_STATE_AUTH_READY_UNVERIFIED = "auth_ready_unverified"
_SYNC_STATE_COMPARE_REMOTE = "compare_remote"
_SYNC_STATE_MERGE_DECISION_REQUIRED = "merge_decision_required"
_SYNC_STATE_READY = "ready"
_SYNC_STATE_REAUTH_REQUIRED = "reauth_required"
_SYNC_STATE_ERROR = "error"
_SYNC_STATE_DISABLED = "disabled"

_MERGE_ACTION_USE_CLOUD = "use_cloud"
_MERGE_ACTION_MERGE_SELECTED = "merge_selected"

_SYSTEM_LIST_IDS = {"auto_recent"}


def _set_sync_state(state: str) -> None:
    st.session_state[GOOGLE_DRIVE_SYNC_STATE_STATE_KEY] = str(state).strip().lower()


def _clear_merge_prompt() -> None:
    st.session_state.pop(GOOGLE_DRIVE_SYNC_MERGE_CANDIDATE_STATE_KEY, None)
    st.session_state.pop(GOOGLE_DRIVE_SYNC_MERGE_RESOLUTION_ACTION_STATE_KEY, None)
    st.session_state.pop(GOOGLE_DRIVE_SYNC_MERGE_SELECTION_STATE_KEY, None)


def _clear_deferred_sync_action() -> None:
    st.session_state.pop(GOOGLE_DRIVE_SYNC_DEFERRED_ACTION_STATE_KEY, None)


def _defer_sync_action(action: str) -> None:
    normalized = str(action or "").strip().lower()
    if normalized not in {"pull", "push", "auto"}:
        st.session_state.pop(GOOGLE_DRIVE_SYNC_DEFERRED_ACTION_STATE_KEY, None)
        return
    st.session_state[GOOGLE_DRIVE_SYNC_DEFERRED_ACTION_STATE_KEY] = normalized


def _normalized_manual_sync_action() -> str:
    raw = str(st.session_state.get(GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY, "")).strip().lower()
    return raw if raw in {"pull", "push"} else ""


def _set_manual_sync_action(action: str) -> None:
    normalized = str(action or "").strip().lower()
    if normalized in {"pull", "push"}:
        st.session_state[GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY] = normalized
    else:
        st.session_state[GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY] = ""


def _consume_merge_resolution_action() -> str:
    raw = str(st.session_state.get(GOOGLE_DRIVE_SYNC_MERGE_RESOLUTION_ACTION_STATE_KEY, "")).strip().lower()
    st.session_state[GOOGLE_DRIVE_SYNC_MERGE_RESOLUTION_ACTION_STATE_KEY] = ""
    if raw in {_MERGE_ACTION_USE_CLOUD, _MERGE_ACTION_MERGE_SELECTED}:
        return raw
    return ""


def _merge_selection_payload() -> dict[str, Any]:
    raw = st.session_state.get(GOOGLE_DRIVE_SYNC_MERGE_SELECTION_STATE_KEY, {})
    return raw if isinstance(raw, dict) else {}


def _json_fingerprint(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        encoded = json.dumps(str(value), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _site_label_from_prefs(prefs: dict[str, Any], site_id: str) -> str:
    site = prefs.get("sites", {}).get(site_id, {})
    if isinstance(site, dict):
        name = str(site.get("name") or "").strip()
        if name:
            return name
        location = site.get("location", {})
        if isinstance(location, dict):
            loc_label = str(location.get("label") or "").strip()
            if loc_label:
                return loc_label
    return site_id


def _list_label_from_prefs(prefs: dict[str, Any], list_id: str) -> str:
    meta = prefs.get("list_meta", {})
    if isinstance(meta, dict):
        entry = meta.get(list_id, {})
        if isinstance(entry, dict):
            name = str(entry.get("name") or "").strip()
            if name:
                return name
    return list_id


def _equipment_additions(local_prefs: dict[str, Any], remote_prefs: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    local_equipment = local_prefs.get("equipment", {})
    remote_equipment = remote_prefs.get("equipment", {})
    if not isinstance(local_equipment, dict):
        return result
    if not isinstance(remote_equipment, dict):
        remote_equipment = {}

    for raw_category_id, raw_values in local_equipment.items():
        category_id = str(raw_category_id).strip()
        if not category_id:
            continue
        local_ids = [str(item).strip() for item in raw_values if str(item).strip()] if isinstance(raw_values, list) else []
        remote_values = remote_equipment.get(category_id, [])
        remote_ids = {
            str(item).strip() for item in remote_values if str(item).strip()
        } if isinstance(remote_values, list) else set()
        additions = [item_id for item_id in local_ids if item_id not in remote_ids]
        if additions:
            result[category_id] = [{"id": item_id, "label": item_id} for item_id in additions]
    return result


def _build_login_merge_candidate(
    local_prefs: dict[str, Any],
    remote_prefs: dict[str, Any],
    *,
    account_sub: str,
    account_email: str,
    remote_file_id: str,
    remote_updated_at_iso: str,
) -> dict[str, Any] | None:
    local = ensure_preferences_shape(local_prefs)
    remote = ensure_preferences_shape(remote_prefs)

    local_sites = local.get("sites", {})
    remote_sites = remote.get("sites", {})
    local_site_ids = [site_id for site_id in site_ids_in_order(local) if site_id in local_sites]
    remote_site_ids = set(site_ids_in_order(remote))
    site_additions = [
        {"id": site_id, "label": _site_label_from_prefs(local, site_id)}
        for site_id in local_site_ids
        if site_id not in remote_site_ids
    ]

    local_list_ids = [
        list_id
        for list_id in list_ids_in_order(local)
        if list_id not in _SYSTEM_LIST_IDS
    ]
    remote_list_ids = {
        list_id
        for list_id in list_ids_in_order(remote)
        if list_id not in _SYSTEM_LIST_IDS
    }
    list_additions = [
        {"id": list_id, "label": _list_label_from_prefs(local, list_id)}
        for list_id in local_list_ids
        if list_id not in remote_list_ids
    ]

    equipment_additions = _equipment_additions(local, remote)
    equipment_count = sum(len(items) for items in equipment_additions.values())

    if not site_additions and not list_additions and equipment_count <= 0:
        return None

    candidate = {
        "account_sub": str(account_sub).strip(),
        "account_email": str(account_email).strip(),
        "remote_file_id": str(remote_file_id).strip(),
        "remote_updated_at_utc": str(remote_updated_at_iso).strip(),
        "remote_prefs": remote,
        "additions": {
            "sites": site_additions,
            "lists": list_additions,
            "equipment": equipment_additions,
        },
        "summary": {
            "site_count": len(site_additions),
            "list_count": len(list_additions),
            "equipment_count": equipment_count,
        },
        "local_prefs_fingerprint": _json_fingerprint(local),
        "remote_prefs_fingerprint": _json_fingerprint(remote),
    }
    return candidate


def _merge_selected_local_additions_into_remote(
    local_prefs: dict[str, Any],
    remote_prefs: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    local = ensure_preferences_shape(local_prefs)
    merged = ensure_preferences_shape(copy.deepcopy(remote_prefs))

    selected_site_ids = {
        str(item).strip() for item in selection.get("sites", []) if str(item).strip()
    }
    if selected_site_ids:
        local_sites = local.get("sites", {}) if isinstance(local.get("sites"), dict) else {}
        merged_sites = merged.get("sites", {}) if isinstance(merged.get("sites"), dict) else {}
        merged_site_order = list(merged.get("site_order", [])) if isinstance(merged.get("site_order"), list) else []
        merged_site_order_set = {str(item).strip() for item in merged_site_order if str(item).strip()}
        for site_id in site_ids_in_order(local):
            if site_id not in selected_site_ids:
                continue
            site_payload = local_sites.get(site_id)
            if isinstance(site_payload, dict):
                merged_sites[site_id] = copy.deepcopy(site_payload)
                if site_id not in merged_site_order_set:
                    merged_site_order.append(site_id)
                    merged_site_order_set.add(site_id)
        merged["sites"] = merged_sites
        merged["site_order"] = merged_site_order

    raw_selected_equipment = selection.get("equipment", {})
    if isinstance(raw_selected_equipment, dict):
        local_equipment = local.get("equipment", {}) if isinstance(local.get("equipment"), dict) else {}
        merged_equipment = merged.get("equipment", {}) if isinstance(merged.get("equipment"), dict) else {}
        for raw_category_id, raw_values in raw_selected_equipment.items():
            category_id = str(raw_category_id).strip()
            if not category_id:
                continue
            selected_ids = [str(item).strip() for item in raw_values if str(item).strip()] if isinstance(raw_values, list) else []
            if not selected_ids:
                continue
            local_ids = [
                str(item).strip()
                for item in (local_equipment.get(category_id, []) if isinstance(local_equipment.get(category_id, []), list) else [])
                if str(item).strip()
            ]
            merged_ids = [
                str(item).strip()
                for item in (merged_equipment.get(category_id, []) if isinstance(merged_equipment.get(category_id, []), list) else [])
                if str(item).strip()
            ]
            merged_seen = set(merged_ids)
            selected_set = set(selected_ids)
            for item_id in local_ids:
                if item_id in selected_set and item_id not in merged_seen:
                    merged_ids.append(item_id)
                    merged_seen.add(item_id)
            merged_equipment[category_id] = merged_ids
        merged["equipment"] = merged_equipment

    selected_list_ids = {
        str(item).strip() for item in selection.get("lists", []) if str(item).strip()
    }
    if selected_list_ids:
        local_lists = local.get("lists", {}) if isinstance(local.get("lists"), dict) else {}
        local_list_meta = local.get("list_meta", {}) if isinstance(local.get("list_meta"), dict) else {}
        merged_lists = merged.get("lists", {}) if isinstance(merged.get("lists"), dict) else {}
        merged_list_meta = merged.get("list_meta", {}) if isinstance(merged.get("list_meta"), dict) else {}
        merged_list_order = list(merged.get("list_order", [])) if isinstance(merged.get("list_order"), list) else []
        merged_list_order_set = {str(item).strip() for item in merged_list_order if str(item).strip()}
        for list_id in list_ids_in_order(local):
            if list_id in _SYSTEM_LIST_IDS or list_id not in selected_list_ids:
                continue
            if list_id in local_lists:
                merged_lists[list_id] = copy.deepcopy(local_lists.get(list_id, []))
                if isinstance(local_list_meta.get(list_id), dict):
                    merged_list_meta[list_id] = copy.deepcopy(local_list_meta[list_id])
                if list_id not in merged_list_order_set:
                    merged_list_order.append(list_id)
                    merged_list_order_set.add(list_id)
        merged["lists"] = merged_lists
        merged["list_meta"] = merged_list_meta
        merged["list_order"] = merged_list_order

    return ensure_preferences_shape(merged)


def _apply_restored_preferences(
    prefs: dict[str, Any],
    restored_prefs: dict[str, Any],
    *,
    remote_file_id: str,
    keep_cloud_enabled: bool,
    apply_session_state_payload: Any = None,
    set_notice: bool = True,
    action_message: str = "Pulled settings from Google Drive and applied them locally.",
    compare_summary: str = "Applied cloud settings snapshot.",
) -> None:
    restored = ensure_preferences_shape(restored_prefs)
    restored["cloud_sync_provider"] = CLOUD_SYNC_PROVIDER_GOOGLE
    restored["cloud_sync_enabled"] = bool(keep_cloud_enabled)
    restored["cloud_sync_initialized"] = True
    restored["cloud_sync_file_id"] = str(remote_file_id).strip()
    restored["cloud_sync_last_ok_utc"] = _utc_now_iso()
    restored["cloud_sync_last_error"] = ""
    prefs.clear()
    prefs.update(restored)
    st.session_state["prefs"] = prefs
    save_preferences(prefs, mark_cloud_pending=False, touch_last_updated_utc=False)
    st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = False
    st.session_state[GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY] = True
    _set_manual_sync_action("")
    _set_sync_state(_SYNC_STATE_READY)
    st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = str(action_message).strip()
    st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = str(compare_summary).strip()
    _clear_merge_prompt()
    _clear_deferred_sync_action()

    if apply_session_state_payload is not None:
        restored_session_keys = apply_session_snapshot(apply_session_state_payload)
        if set_notice:
            st.session_state["cloud_sync_notice"] = (
                "Google Drive settings restored "
                f"({restored_session_keys} session key(s) applied)."
            )
    st.rerun()


def _persist_cloud_sync_error(
    prefs: dict[str, Any],
    message: str,
    *,
    preserve_pending: bool,
    sync_state: str,
    clear_manual_action: bool = True,
    last_action_prefix: str = "Error while syncing with Google Drive:",
) -> None:
    error_message = str(message).strip()
    if clear_manual_action:
        _set_manual_sync_action("")
    st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = f"{last_action_prefix} {error_message}".strip()
    _set_sync_state(sync_state)
    if str(prefs.get("cloud_sync_last_error", "")).strip() != error_message:
        prefs["cloud_sync_last_error"] = error_message
        st.session_state["prefs"] = prefs
        save_preferences(prefs, mark_cloud_pending=False, touch_last_updated_utc=False)
    if not preserve_pending:
        st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = False


def _auth_failure_deferred_action(manual_sync_action: str) -> str:
    normalized_manual = str(manual_sync_action or "").strip().lower()
    if normalized_manual in {"pull", "push"}:
        return normalized_manual
    if bool(st.session_state.get(GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY, False)):
        return "push"
    return "auto"


def _is_auth_error(exc: Exception) -> bool:
    return isinstance(exc, GoogleDriveAPIError) and bool(getattr(exc, "is_auth_error", False))


def _mark_reauth_required(prefs: dict[str, Any], message: str, *, manual_sync_action: str) -> None:
    _defer_sync_action(_auth_failure_deferred_action(manual_sync_action))
    _persist_cloud_sync_error(
        prefs,
        message,
        preserve_pending=True,
        sync_state=_SYNC_STATE_REAUTH_REQUIRED,
        clear_manual_action=True,
        last_action_prefix="Google authentication required for cloud sync:",
    )
    st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
        "Cloud sync paused until Google auth is reconnected."
    )


def _merge_candidate_account_matches(candidate: dict[str, Any], account_sub: str) -> bool:
    candidate_sub = str(candidate.get("account_sub", "")).strip()
    if not candidate_sub:
        return True
    return candidate_sub == str(account_sub).strip()


def _process_merge_resolution_if_requested(
    prefs: dict[str, Any],
    *,
    account_sub: str,
    manual_sync_action: str,
) -> bool:
    resolution_action = _consume_merge_resolution_action()
    if not resolution_action:
        return False

    raw_candidate = st.session_state.get(GOOGLE_DRIVE_SYNC_MERGE_CANDIDATE_STATE_KEY)
    if not isinstance(raw_candidate, dict):
        return False
    candidate = raw_candidate

    if not _merge_candidate_account_matches(candidate, account_sub):
        _clear_merge_prompt()
        _persist_cloud_sync_error(
            prefs,
            "Cloud merge prompt no longer matches the signed-in Google account. Please retry cloud sync.",
            preserve_pending=True,
            sync_state=_SYNC_STATE_ERROR,
        )
        return True

    remote_file_id = str(candidate.get("remote_file_id", "")).strip()
    remote_prefs_raw = candidate.get("remote_prefs")
    if not isinstance(remote_prefs_raw, dict):
        _clear_merge_prompt()
        _persist_cloud_sync_error(
            prefs,
            "Cloud merge prompt is missing the remote profile snapshot. Please retry cloud sync.",
            preserve_pending=True,
            sync_state=_SYNC_STATE_ERROR,
        )
        return True

    keep_cloud_enabled = bool(prefs.get("cloud_sync_enabled", True))
    if resolution_action == _MERGE_ACTION_USE_CLOUD:
        _apply_restored_preferences(
            prefs,
            remote_prefs_raw,
            remote_file_id=remote_file_id,
            keep_cloud_enabled=keep_cloud_enabled,
            apply_session_state_payload=None,
            set_notice=False,
            action_message="Applied cloud profile; local additions were not merged.",
            compare_summary="Login-time cloud compare completed (cloud profile kept).",
        )
        return True

    if resolution_action == _MERGE_ACTION_MERGE_SELECTED:
        selection = _merge_selection_payload()
        merged_prefs = _merge_selected_local_additions_into_remote(prefs, remote_prefs_raw, selection)
        merged_prefs["cloud_sync_provider"] = CLOUD_SYNC_PROVIDER_GOOGLE
        merged_prefs["cloud_sync_enabled"] = keep_cloud_enabled
        merged_prefs["cloud_sync_initialized"] = True
        if remote_file_id:
            merged_prefs["cloud_sync_file_id"] = remote_file_id
        merged_prefs["cloud_sync_last_error"] = ""
        prefs.clear()
        prefs.update(merged_prefs)
        st.session_state["prefs"] = prefs
        save_preferences(prefs, mark_cloud_pending=False, touch_last_updated_utc=True)
        st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = True
        st.session_state[GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY] = True
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
            "Login-time cloud compare completed; selected local additions merged into cloud profile."
        )
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
            "Selected local additions merged into the cloud profile; upload queued."
        )
        _set_sync_state(_SYNC_STATE_READY)
        _clear_merge_prompt()
        _clear_deferred_sync_action()
        # Let the normal push path below upload immediately if possible.
        if manual_sync_action != "push":
            _set_manual_sync_action("push")
        return False

    return False


def _sync_runtime_state_reset_for_logout() -> None:
    _set_sync_state(_SYNC_STATE_UNAUTHENTICATED)
    st.session_state[GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY] = False
    st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACCOUNT_STATE_KEY] = ""
    _set_manual_sync_action("")
    st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = "Not signed in; cloud sync idle."
    _clear_merge_prompt()


def _compare_first_bootstrap(
    prefs: dict[str, Any],
    *,
    access_token: str,
    account_sub: str,
    account_email: str,
    manual_pull_requested: bool,
    manual_sync_action: str,
) -> None:
    local_updated_at = _parse_utc_timestamp(prefs.get("last_updated_utc", ""))
    _set_sync_state(_SYNC_STATE_COMPARE_REMOTE)

    try:
        remote_file = find_settings_file(access_token, filename=DEFAULT_SETTINGS_FILENAME)
    except Exception as exc:
        if _is_auth_error(exc):
            _mark_reauth_required(prefs, str(exc), manual_sync_action=manual_sync_action)
        else:
            _persist_cloud_sync_error(
                prefs,
                str(exc),
                preserve_pending=True,
                sync_state=_SYNC_STATE_ERROR,
            )
        return

    remote_file_id = ""
    remote_payload: dict[str, Any] | None = None
    remote_modified_time = ""
    if isinstance(remote_file, dict):
        remote_file_id = str(remote_file.get("id", "")).strip()
        remote_modified_time = str(remote_file.get("modifiedTime", "")).strip()
    st.session_state[GOOGLE_DRIVE_SYNC_LAST_REMOTE_FILE_MODIFIED_STATE_KEY] = remote_modified_time
    if not remote_file_id:
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_REMOTE_PAYLOAD_UPDATED_STATE_KEY] = ""
    if remote_file_id:
        try:
            remote_payload = read_settings_payload(access_token, remote_file_id)
        except Exception as exc:
            if _is_auth_error(exc):
                _mark_reauth_required(prefs, str(exc), manual_sync_action=manual_sync_action)
            else:
                _persist_cloud_sync_error(
                    prefs,
                    str(exc),
                    preserve_pending=True,
                    sync_state=_SYNC_STATE_ERROR,
                )
            return

    remote_settings_present = False
    remote_updated_at = datetime.fromtimestamp(0, tz=timezone.utc)
    if remote_payload:
        remote_owner_sub = str(remote_payload.get("owner_sub", "")).strip()
        if remote_owner_sub and remote_owner_sub != str(account_sub).strip():
            _persist_cloud_sync_error(
                prefs,
                "Cloud settings payload owner does not match the signed-in Google account.",
                preserve_pending=True,
                sync_state=_SYNC_STATE_ERROR,
            )
            st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
                "Cloud compare halted because the cloud payload owner did not match the signed-in account."
            )
            return

        remote_prefs_raw = remote_payload.get("preferences")
        if isinstance(remote_prefs_raw, dict):
            remote_settings_present = True
            remote_updated_at = _payload_updated_at_utc(
                remote_payload,
                fallback_modified_time=remote_modified_time,
            )
            st.session_state[GOOGLE_DRIVE_SYNC_LAST_REMOTE_PAYLOAD_UPDATED_STATE_KEY] = remote_updated_at.isoformat()

            if manual_pull_requested:
                _apply_restored_preferences(
                    prefs,
                    remote_prefs_raw,
                    remote_file_id=remote_file_id,
                    keep_cloud_enabled=bool(prefs.get("cloud_sync_enabled", True)),
                    apply_session_state_payload=remote_payload.get("session_state", {}),
                    set_notice=True,
                    action_message="Pulled settings from Google Drive and applied them locally.",
                    compare_summary="Manual pull requested; applying cloud settings snapshot.",
                )
                return

            candidate = _build_login_merge_candidate(
                prefs,
                remote_prefs_raw,
                account_sub=account_sub,
                account_email=account_email,
                remote_file_id=remote_file_id,
                remote_updated_at_iso=remote_updated_at.isoformat(),
            )
            if candidate is not None:
                st.session_state[GOOGLE_DRIVE_SYNC_MERGE_CANDIDATE_STATE_KEY] = candidate
                st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
                    "Cloud profile found. Local session has additions that require a merge decision before upload."
                )
                st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
                    "Merge decision required before cloud sync can continue."
                )
                st.session_state[GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY] = False
                _set_sync_state(_SYNC_STATE_MERGE_DECISION_REQUIRED)
                _set_manual_sync_action("")
                return

            compare_message = (
                "Manual pull requested; applying cloud settings snapshot."
                if manual_pull_requested
                else "Applying cloud settings snapshot after login-time compare (no local additions detected)."
            )
            action_message = (
                "Pulled settings from Google Drive and applied them locally."
                if manual_pull_requested
                else "Applied cloud profile after login-time compare."
            )
            _apply_restored_preferences(
                prefs,
                remote_prefs_raw,
                remote_file_id=remote_file_id,
                keep_cloud_enabled=bool(prefs.get("cloud_sync_enabled", True)),
                apply_session_state_payload=remote_payload.get("session_state", {}),
                set_notice=True,
                action_message=action_message,
                compare_summary=compare_message,
            )
            return

        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
            "Found Google Drive settings file, but it does not contain a valid preferences payload."
        )
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
            "Cloud compare failed: cloud settings file payload is invalid."
        )
        _set_sync_state(_SYNC_STATE_ERROR)
        return

    # No remote payload / no file: compare complete. Allow seeding cloud from local profile.
    if remote_file_id:
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
            "Found Google Drive settings file, but it could not be read as a settings payload."
        )
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
            "Cloud compare failed: settings file content is not a valid payload."
        )
        _set_sync_state(_SYNC_STATE_ERROR)
        return

    st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = "No Google Drive settings file found yet."
    if manual_pull_requested:
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
            "Manual pull requested, but no cloud settings file exists yet."
        )
        _set_manual_sync_action("")
        st.session_state[GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY] = True
        _set_sync_state(_SYNC_STATE_READY)
        return

    # First compare completed and no remote exists; seed cloud from local if enabled.
    if bool(prefs.get("cloud_sync_enabled", False)):
        st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = True
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
            "No cloud settings file found; local settings will be uploaded to seed Google Drive."
        )
    else:
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
            "No cloud settings file found; cloud sync is disabled."
        )
    st.session_state[GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY] = True
    _set_sync_state(_SYNC_STATE_READY)
    _clear_merge_prompt()


def maybe_sync_prefs_with_google_drive(prefs: dict[str, Any]) -> None:
    _refresh_legacy_globals()

    is_logged_in = _is_user_logged_in()
    if not is_logged_in:
        _sync_runtime_state_reset_for_logout()
        return

    manual_sync_action = _normalized_manual_sync_action()
    if manual_sync_action:
        _defer_sync_action(manual_sync_action)

    access_token = _get_google_access_token()
    if not access_token:
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
            "Signed in, but Google access token is unavailable."
        )
        _mark_reauth_required(
            prefs,
            "Google access token not available. Configure auth expose_tokens=['access'] and include Drive appData scope.",
            manual_sync_action=manual_sync_action,
        )
        return

    account_sub = _get_user_claim("sub") or _get_user_claim("email")
    account_email = _get_user_claim("email")
    manual_pull_requested = manual_sync_action == "pull"
    manual_push_requested = manual_sync_action == "push"
    was_cloud_sync_initialized = bool(prefs.get("cloud_sync_initialized", False))
    last_account_sub = str(st.session_state.get(GOOGLE_DRIVE_SYNC_LAST_ACCOUNT_STATE_KEY, "")).strip()
    first_sign_in_for_session = bool(account_sub) and account_sub != last_account_sub
    registration_sign_in = first_sign_in_for_session and (not was_cloud_sync_initialized)

    if first_sign_in_for_session:
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACCOUNT_STATE_KEY] = str(account_sub).strip()
        st.session_state[GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY] = False
        _clear_merge_prompt()
        _set_sync_state(_SYNC_STATE_AUTH_READY_UNVERIFIED)
        prefs_changed = False
        if str(prefs.get("cloud_sync_provider", "")).strip().lower() != CLOUD_SYNC_PROVIDER_GOOGLE:
            prefs["cloud_sync_provider"] = CLOUD_SYNC_PROVIDER_GOOGLE
            prefs_changed = True
        if registration_sign_in:
            prefs["cloud_sync_initialized"] = True
            prefs_changed = True
            if not bool(prefs.get("cloud_sync_enabled", False)):
                prefs["cloud_sync_enabled"] = True
                prefs_changed = True
        if str(prefs.get("cloud_sync_last_error", "")).strip():
            prefs["cloud_sync_last_error"] = ""
            prefs_changed = True
        if prefs_changed:
            st.session_state["prefs"] = prefs
            save_preferences(prefs, mark_cloud_pending=False, touch_last_updated_utc=False)

    # If we have a deferred sync action after successful auth, restore it once.
    if not manual_sync_action:
        deferred_action = str(st.session_state.get(GOOGLE_DRIVE_SYNC_DEFERRED_ACTION_STATE_KEY, "")).strip().lower()
        if deferred_action in {"pull", "push"}:
            manual_sync_action = deferred_action
            manual_pull_requested = manual_sync_action == "pull"
            manual_push_requested = manual_sync_action == "push"
            _set_manual_sync_action(manual_sync_action)
        _clear_deferred_sync_action()

    # Process pending merge resolution before any other sync behavior.
    merge_processing_consumed = _process_merge_resolution_if_requested(
        prefs,
        account_sub=account_sub,
        manual_sync_action=manual_sync_action,
    )
    if merge_processing_consumed:
        return
    manual_sync_action = _normalized_manual_sync_action()
    manual_pull_requested = manual_sync_action == "pull"
    manual_push_requested = manual_sync_action == "push"

    if not bool(prefs.get("cloud_sync_enabled", False)):
        _set_manual_sync_action("")
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = "Signed in; cloud sync is disabled."
        _set_sync_state(_SYNC_STATE_DISABLED)
        return

    # Compare-first bootstrap: do this for any unverified authenticated session, including manual push.
    bootstrap_complete = bool(st.session_state.get(GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY, False))
    if (not bootstrap_complete) or manual_pull_requested:
        _compare_first_bootstrap(
            prefs,
            access_token=access_token,
            account_sub=str(account_sub).strip(),
            account_email=str(account_email).strip(),
            manual_pull_requested=manual_pull_requested,
            manual_sync_action=manual_sync_action,
        )
        # compare may rerun/return after setting merge prompt/error/ready
        bootstrap_complete = bool(st.session_state.get(GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY, False))
        # Never continue into push after manual pull handling or a failed/incomplete compare.
        if manual_pull_requested:
            return
        if not bootstrap_complete:
            return

    if bool(st.session_state.get(GOOGLE_DRIVE_SYNC_MERGE_CANDIDATE_STATE_KEY)):
        _set_sync_state(_SYNC_STATE_MERGE_DECISION_REQUIRED)
        return

    if not bool(st.session_state.get(GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY, False)):
        if manual_push_requested:
            _set_manual_sync_action("")
        if str(st.session_state.get(GOOGLE_DRIVE_SYNC_STATE_STATE_KEY, "")).strip().lower() not in {
            _SYNC_STATE_REAUTH_REQUIRED,
            _SYNC_STATE_ERROR,
            _SYNC_STATE_DISABLED,
        }:
            _set_sync_state(_SYNC_STATE_READY)
        return

    try:
        if manual_push_requested:
            st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
                "Manual push requested; uploading current local settings to Google Drive."
            )
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = "Uploading local settings to Google Drive..."
        payload = _build_cloud_settings_payload(prefs, str(account_sub).strip())
        updated_file = upsert_settings_file(
            access_token,
            payload,
            preferred_file_id=str(prefs.get("cloud_sync_file_id", "")).strip(),
            filename=DEFAULT_SETTINGS_FILENAME,
        )
        next_file_id = str(updated_file.get("id", "")).strip()
        if next_file_id:
            prefs["cloud_sync_file_id"] = next_file_id
        remote_modified_time = str(updated_file.get("modifiedTime", "")).strip()
        if remote_modified_time:
            st.session_state[GOOGLE_DRIVE_SYNC_LAST_REMOTE_FILE_MODIFIED_STATE_KEY] = remote_modified_time
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_REMOTE_PAYLOAD_UPDATED_STATE_KEY] = (
            str(payload.get("updated_at_utc", "")).strip()
        )
        prefs["cloud_sync_last_ok_utc"] = _utc_now_iso()
        prefs["cloud_sync_last_error"] = ""
        st.session_state["prefs"] = prefs
        save_preferences(prefs, mark_cloud_pending=False, touch_last_updated_utc=False)
        st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = False
        _set_manual_sync_action("")
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
            "Manual push completed successfully."
            if manual_push_requested
            else "Local settings uploaded to Google Drive successfully."
        )
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
            "Uploaded local settings to Google Drive (manual push)."
            if manual_push_requested
            else "Uploaded local settings to Google Drive."
        )
        _set_sync_state(_SYNC_STATE_READY)
    except Exception as exc:
        if _is_auth_error(exc):
            _mark_reauth_required(prefs, str(exc), manual_sync_action=manual_sync_action)
            return
        _persist_cloud_sync_error(
            prefs,
            str(exc),
            preserve_pending=True,
            sync_state=_SYNC_STATE_ERROR,
        )
