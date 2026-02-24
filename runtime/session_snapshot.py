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

_SESSION_STATE_UNSERIALIZABLE = object()


def _session_value_to_jsonable(value: Any, *, depth: int = 0) -> Any:
    _refresh_legacy_globals()
    if depth > 6:
        return _SESSION_STATE_UNSERIALIZABLE
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, list):
        items = []
        for item in value:
            parsed = _session_value_to_jsonable(item, depth=depth + 1)
            if parsed is not _SESSION_STATE_UNSERIALIZABLE:
                items.append(parsed)
        return items
    if isinstance(value, tuple):
        items = []
        for item in value:
            parsed = _session_value_to_jsonable(item, depth=depth + 1)
            if parsed is not _SESSION_STATE_UNSERIALIZABLE:
                items.append(parsed)
        return items
    if isinstance(value, set):
        items = []
        for item in sorted(value, key=lambda current: str(current)):
            parsed = _session_value_to_jsonable(item, depth=depth + 1)
            if parsed is not _SESSION_STATE_UNSERIALIZABLE:
                items.append(parsed)
        return items
    if isinstance(value, dict):
        parsed_dict: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            parsed_key = str(raw_key).strip()
            if not parsed_key:
                continue
            parsed_value = _session_value_to_jsonable(raw_value, depth=depth + 1)
            if parsed_value is _SESSION_STATE_UNSERIALIZABLE:
                continue
            parsed_dict[parsed_key] = parsed_value
        return parsed_dict
    return _SESSION_STATE_UNSERIALIZABLE


def build_syncable_session_snapshot() -> dict[str, Any]:
    _refresh_legacy_globals()
    excluded_exact = {
        "prefs",
        "prefs_bootstrap_runs",
        "prefs_bootstrapped",
        "altaz_refresh",
        "recommended_targets_query_cache",
        "browser_language",
        "browser_hour_cycle",
        "browser_month_day_pattern",
        "browser_viewport_width",
        "prefs_persistence_notice",
        "settings_import_notice",
        GOOGLE_DRIVE_SYNC_BOOTSTRAP_STATE_KEY,
        GOOGLE_DRIVE_SYNC_PENDING_STATE_KEY,
        GOOGLE_DRIVE_SYNC_LAST_ACCOUNT_STATE_KEY,
        GOOGLE_DRIVE_SYNC_LAST_APPLIED_TOKEN_STATE_KEY,
    }
    excluded_prefixes = (
        "FormSubmitter:",
        "prefs_bootstrap_wait_",
        "browser_prefs_",
        "cloud_sync_",
        "google_drive_sync_",
        "settings_import_",
    )

    snapshot: dict[str, Any] = {}
    for raw_key, raw_value in st.session_state.items():
        key = str(raw_key).strip()
        if not key:
            continue
        if key in excluded_exact:
            continue
        if any(key.startswith(prefix) for prefix in excluded_prefixes):
            continue
        parsed_value = _session_value_to_jsonable(raw_value)
        if parsed_value is _SESSION_STATE_UNSERIALIZABLE:
            continue
        snapshot[key] = parsed_value
    return snapshot


def apply_session_snapshot(snapshot: Any) -> int:
    _refresh_legacy_globals()
    if not isinstance(snapshot, dict):
        return 0
    restore_excluded_exact = {
        "altaz_refresh",
        "recommended_targets_query_cache",
    }
    restore_excluded_prefixes = (
        "prefs_bootstrap_wait_",
        "google_drive_sync_",
        "cloud_sync_",
    )
    applied = 0
    for raw_key, raw_value in snapshot.items():
        key = str(raw_key).strip()
        if not key or key == "prefs":
            continue
        if key in restore_excluded_exact:
            continue
        if any(key.startswith(prefix) for prefix in restore_excluded_prefixes):
            continue
        parsed_value = _session_value_to_jsonable(raw_value)
        if parsed_value is _SESSION_STATE_UNSERIALIZABLE:
            continue
        try:
            st.session_state[key] = parsed_value
        except Exception as exc:
            # Some widget-managed keys (e.g. auto-refresh) may already be instantiated
            # before cloud session restore runs. Skip them instead of crashing.
            if "cannot be modified after the widget" in str(exc):
                continue
            raise
        applied += 1
    return applied


def _build_cloud_settings_payload(prefs: dict[str, Any], owner_sub: str) -> dict[str, Any]:
    _refresh_legacy_globals()
    return {
        "format": "dso_explorer_cloud_settings",
        "version": GOOGLE_DRIVE_SYNC_SESSION_SNAPSHOT_VERSION,
        "updated_at_utc": _utc_now_iso(),
        "owner_sub": str(owner_sub).strip(),
        "preferences": ensure_preferences_shape(prefs),
        "session_state": build_syncable_session_snapshot(),
    }

