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

def maybe_sync_prefs_with_google_drive(prefs: dict[str, Any]) -> None:
    _refresh_legacy_globals()
    def _persist_cloud_sync_error(message: str) -> None:
        error_message = str(message).strip()
        st.session_state[GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY] = ""
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
            f"Error while syncing with Google Drive: {error_message}"
        )
        if str(prefs.get("cloud_sync_last_error", "")).strip() == error_message:
            return
        prefs["cloud_sync_last_error"] = error_message
        st.session_state["prefs"] = prefs
        save_preferences(prefs, mark_cloud_pending=False, touch_last_updated_utc=False)
        st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = False

    is_logged_in = _is_user_logged_in()
    if not is_logged_in:
        st.session_state[GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY] = False
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACCOUNT_STATE_KEY] = ""
        st.session_state[GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY] = ""
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = "Not signed in; cloud sync idle."
        return

    access_token = _get_google_access_token()
    if not access_token:
        st.session_state[GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY] = ""
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
            "Signed in, but Google access token is unavailable."
        )
        _persist_cloud_sync_error(
            "Google access token not available. Configure auth expose_tokens=[\"access\"]."
        )
        st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = False
        return

    account_sub = _get_user_claim("sub") or _get_user_claim("email")
    manual_sync_action = str(st.session_state.get(GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY, "")).strip().lower()
    if manual_sync_action not in {"pull", "push"}:
        manual_sync_action = ""
    manual_pull_requested = manual_sync_action == "pull"
    manual_push_requested = manual_sync_action == "push"
    was_cloud_sync_initialized = bool(prefs.get("cloud_sync_initialized", False))
    last_account_sub = str(st.session_state.get(GOOGLE_DRIVE_SYNC_LAST_ACCOUNT_STATE_KEY, "")).strip()
    first_sign_in_for_session = bool(account_sub) and account_sub != last_account_sub
    registration_sign_in = first_sign_in_for_session and (not was_cloud_sync_initialized)
    if first_sign_in_for_session:
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACCOUNT_STATE_KEY] = account_sub
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

    if not bool(prefs.get("cloud_sync_enabled", False)):
        st.session_state[GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY] = ""
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = "Signed in; cloud sync is disabled."
        return

    cloud_file_id = str(prefs.get("cloud_sync_file_id", "")).strip()
    bootstrap_complete = bool(st.session_state.get(GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY, False))
    if (not bootstrap_complete and not manual_push_requested) or manual_pull_requested:
        local_updated_at = _parse_utc_timestamp(prefs.get("last_updated_utc", ""))
        try:
            remote_file = find_settings_file(access_token, filename=DEFAULT_SETTINGS_FILENAME)
        except Exception as exc:
            _persist_cloud_sync_error(str(exc))
            st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = False
            st.session_state[GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY] = True
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
                _persist_cloud_sync_error(str(exc))
                st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = False
                st.session_state[GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY] = True
                return

        remote_settings_present = False
        remote_updated_at = datetime.fromtimestamp(0, tz=timezone.utc)
        if remote_payload:
            remote_prefs_raw = remote_payload.get("preferences")
            if isinstance(remote_prefs_raw, dict):
                remote_settings_present = True
                remote_updated_at = _payload_updated_at_utc(
                    remote_payload,
                    fallback_modified_time=remote_modified_time,
                )
                st.session_state[GOOGLE_DRIVE_SYNC_LAST_REMOTE_PAYLOAD_UPDATED_STATE_KEY] = (
                    remote_updated_at.isoformat()
                )
                should_apply_remote_snapshot = manual_pull_requested or (remote_updated_at > local_updated_at)
                if should_apply_remote_snapshot:
                    st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
                        "Manual pull requested; applying cloud settings snapshot."
                        if manual_pull_requested
                        else "Remote cloud settings are newer than local settings; applying remote snapshot."
                    )
                    restored_prefs = ensure_preferences_shape(remote_prefs_raw)
                    restored_prefs["cloud_sync_provider"] = CLOUD_SYNC_PROVIDER_GOOGLE
                    restored_prefs["cloud_sync_enabled"] = bool(prefs.get("cloud_sync_enabled", True))
                    restored_prefs["cloud_sync_initialized"] = True
                    restored_prefs["cloud_sync_file_id"] = remote_file_id
                    restored_prefs["cloud_sync_last_ok_utc"] = _utc_now_iso()
                    restored_prefs["cloud_sync_last_error"] = ""
                    st.session_state["prefs"] = restored_prefs
                    save_preferences(
                        restored_prefs,
                        mark_cloud_pending=False,
                        touch_last_updated_utc=False,
                    )
                    st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = False
                    restored_session_keys = apply_session_snapshot(remote_payload.get("session_state", {}))
                    st.session_state[GOOGLE_DRIVE_SYNC_LAST_APPLIED_TOKEN_STATE_KEY] = (
                        f"{remote_file_id}:{remote_updated_at.isoformat()}"
                    )
                    st.session_state[GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY] = True
                    st.session_state[GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY] = ""
                    st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
                        "Pulled settings from Google Drive and applied them locally."
                        if manual_pull_requested
                        else "Pulled newer settings from Google Drive and applied them locally."
                    )
                    st.session_state["cloud_sync_notice"] = (
                        "Google Drive settings restored "
                        f"({restored_session_keys} session key(s) applied)."
                    )
                    st.rerun()
                if (not manual_pull_requested) and (local_updated_at > remote_updated_at):
                    st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
                        "Local settings are newer than cloud settings; upload queued."
                    )
                    st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = True
                elif (not manual_pull_requested) and (local_updated_at == remote_updated_at):
                    st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
                        "Local and cloud settings timestamps match; no sync required."
                    )
                    st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
                        "Cloud settings already match local settings."
                    )
                elif manual_pull_requested:
                    st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
                        "Manual pull requested, but cloud payload was not newer; no changes were applied."
                    )
                    st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
                        "Manual pull completed; local settings already match or exceed cloud timestamp."
                    )
                    st.session_state[GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY] = ""
        elif remote_file_id:
            st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
                "Found Google Drive settings file, but it does not contain a valid preferences payload."
            )
            if manual_pull_requested:
                st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
                    "Manual pull requested, but the cloud settings file payload is invalid."
                )
                st.session_state[GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY] = ""
        else:
            st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
                "No Google Drive settings file found yet."
            )
            if manual_pull_requested:
                st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
                    "Manual pull requested, but no cloud settings file exists yet."
                )
                st.session_state[GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY] = ""

        should_push_after_bootstrap = bool(st.session_state.get(GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY, False))

        if remote_file_id and remote_file_id != cloud_file_id:
            prefs["cloud_sync_file_id"] = remote_file_id
            prefs["cloud_sync_last_error"] = ""
            st.session_state["prefs"] = prefs
            save_preferences(prefs, mark_cloud_pending=False, touch_last_updated_utc=False)
            st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = should_push_after_bootstrap

        if (
            registration_sign_in
            and not remote_file_id
            and not remote_settings_present
            and bool(prefs.get("cloud_sync_enabled", False))
        ):
            # First-time cloud registration with no remote settings file yet:
            # seed local settings/session state to Drive.
            st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = (
                "First sign-in detected with no cloud settings file; seeding Google Drive from local settings."
            )
            st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = True

        st.session_state[GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY] = True

    if not bool(st.session_state.get(GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY, False)):
        if manual_push_requested:
            st.session_state[GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY] = ""
        return

    try:
        if manual_push_requested:
            st.session_state[GOOGLE_DRIVE_SYNC_LAST_COMPARE_SUMMARY_STATE_KEY] = (
                "Manual push requested; uploading current local settings to Google Drive."
            )
        st.session_state[GOOGLE_DRIVE_SYNC_LAST_ACTION_STATE_KEY] = "Uploading local settings to Google Drive..."
        payload = _build_cloud_settings_payload(prefs, account_sub)
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
        st.session_state[GOOGLE_DRIVE_SYNC_MANUAL_ACTION_STATE_KEY] = ""
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
    except Exception as exc:
        _persist_cloud_sync_error(str(exc))
        st.session_state[GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY] = False


