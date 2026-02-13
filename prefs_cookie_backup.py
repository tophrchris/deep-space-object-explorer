from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st

try:
    from extra_streamlit_components import CookieManager
except Exception:
    CookieManager = None

COOKIE_BACKUP_KEY = "dso_explorer_prefs_backup_v1"
COOKIE_BACKUP_TTL_DAYS = 365
COOKIE_BACKUP_MAX_VALUE_CHARS = 3500
COOKIE_BACKUP_DISABLE_AFTER_MISSES = 2

_STATE_ENABLED = "_prefs_cookie_backup_enabled"
_STATE_RUNTIME_ENABLED = "_prefs_cookie_backup_runtime_enabled"
_STATE_MANAGER = "_prefs_cookie_backup_manager"
_STATE_PENDING_HASH = "_prefs_cookie_backup_pending_hash"
_STATE_UNOBSERVED_MISSES = "_prefs_cookie_backup_unobserved_misses"
_STATE_NOTICE = "prefs_cookie_backup_notice"


def set_cookie_backup_runtime_enabled(enabled: bool) -> None:
    st.session_state[_STATE_RUNTIME_ENABLED] = bool(enabled)
    if enabled:
        return

    st.session_state[_STATE_ENABLED] = False
    st.session_state.pop(_STATE_MANAGER, None)
    st.session_state.pop(_STATE_PENDING_HASH, None)
    st.session_state.pop(_STATE_UNOBSERVED_MISSES, None)
    st.session_state.pop(_STATE_NOTICE, None)


def is_cookie_backup_enabled() -> bool:
    if not bool(st.session_state.get(_STATE_RUNTIME_ENABLED, True)):
        return False
    if CookieManager is None:
        return False
    return bool(st.session_state.get(_STATE_ENABLED, True))


def get_cookie_backup_notice() -> str:
    if not bool(st.session_state.get(_STATE_RUNTIME_ENABLED, True)):
        return ""
    notice = st.session_state.get(_STATE_NOTICE, "")
    return str(notice).strip() if isinstance(notice, str) else ""


def bootstrap_cookie_backup() -> None:
    if not is_cookie_backup_enabled():
        return
    _ = _get_cookie_manager()


def read_preferences_cookie_backup() -> str:
    if not is_cookie_backup_enabled():
        return ""

    manager = _get_cookie_manager()
    if manager is None:
        return ""

    try:
        all_cookies = manager.get_all(key="browser_prefs_cookie_get_all")
        value = all_cookies.get(COOKIE_BACKUP_KEY, "") if isinstance(all_cookies, dict) else ""
        _track_pending_write_visibility(value)
        return str(value).strip() if isinstance(value, str) else ""
    except Exception:
        return ""


def write_preferences_cookie_backup(encoded_value: str) -> bool:
    if not is_cookie_backup_enabled():
        return False

    payload_hash = hashlib.sha1(encoded_value.encode("ascii")).hexdigest()[:12]
    try:
        if len(encoded_value) > COOKIE_BACKUP_MAX_VALUE_CHARS:
            _set_cookie_value("", -1, component_key=f"browser_prefs_cookie_clear_{payload_hash}")
            st.session_state[_STATE_NOTICE] = (
                "Cookie backup skipped: preferences are too large for a browser cookie."
            )
            return False

        cookie_saved = _set_cookie_value(
            encoded_value,
            COOKIE_BACKUP_TTL_DAYS,
            component_key=f"browser_prefs_cookie_write_{payload_hash}",
        )
        if not cookie_saved:
            _disable_cookie_backup("Cookie backup is unavailable in this browser session. Using local storage only.")
            return False

        st.session_state.pop(_STATE_NOTICE, None)
        return True
    except Exception:
        _disable_cookie_backup("Cookie backup is unavailable in this browser session. Using local storage only.")
        return False


def _disable_cookie_backup(notice: str) -> None:
    st.session_state[_STATE_ENABLED] = False
    st.session_state.pop(_STATE_PENDING_HASH, None)
    st.session_state.pop(_STATE_UNOBSERVED_MISSES, None)
    st.session_state[_STATE_NOTICE] = notice


def _get_cookie_manager() -> Any | None:
    manager = st.session_state.get(_STATE_MANAGER)
    if CookieManager is not None and isinstance(manager, CookieManager):
        return manager

    try:
        manager = CookieManager(key="browser_cookie_manager_init")
        st.session_state[_STATE_MANAGER] = manager
        return manager
    except Exception:
        st.session_state[_STATE_MANAGER] = None
        return None


def _track_pending_write_visibility(observed_value: Any) -> None:
    pending_hash = str(st.session_state.get(_STATE_PENDING_HASH, "")).strip()
    if not pending_hash:
        return

    observed_hash = (
        hashlib.sha1(observed_value.encode("ascii")).hexdigest()[:12]
        if isinstance(observed_value, str) and observed_value
        else ""
    )
    if observed_hash and observed_hash == pending_hash:
        st.session_state.pop(_STATE_PENDING_HASH, None)
        st.session_state.pop(_STATE_UNOBSERVED_MISSES, None)
        return

    misses = int(st.session_state.get(_STATE_UNOBSERVED_MISSES, 0)) + 1
    st.session_state[_STATE_UNOBSERVED_MISSES] = misses
    if misses >= COOKIE_BACKUP_DISABLE_AFTER_MISSES:
        _disable_cookie_backup("Cookie backup appears blocked in this browser session. Using local storage only.")


def _set_cookie_value(value: str, duration_days: int, component_key: str) -> bool:
    manager = _get_cookie_manager()
    if manager is None:
        return False

    try:
        if duration_days < 0:
            manager.delete(COOKIE_BACKUP_KEY, key=f"{component_key}_delete")
            st.session_state.pop(_STATE_PENDING_HASH, None)
            return True

        ttl_days = max(1, int(duration_days))
        max_age_seconds = int(ttl_days * 24 * 60 * 60)
        manager.set(
            cookie=COOKIE_BACKUP_KEY,
            val=value,
            key=component_key,
            path="/",
            max_age=max_age_seconds,
            same_site="lax",
        )
        payload_hash = hashlib.sha1(value.encode("ascii")).hexdigest()[:12]
        st.session_state[_STATE_PENDING_HASH] = payload_hash
        return True
    except Exception:
        return False
