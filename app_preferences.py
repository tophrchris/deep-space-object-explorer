from __future__ import annotations

import base64
import copy
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from app_constants import DEFAULT_LOCATION, UI_THEME_LIGHT, UI_THEME_OPTIONS, WIND16
from lists.list_subsystem import (
    AUTO_RECENT_LIST_ID,
    default_list_meta,
    default_list_order,
    default_lists_payload,
    normalize_list_preferences,
)
from streamlit_js_eval import get_local_storage, set_local_storage, streamlit_js_eval

BROWSER_PREFS_STORAGE_KEY = "dso_explorer_prefs_v2"
PREFS_BOOTSTRAP_MAX_RUNS = 6
PREFS_BOOTSTRAP_RETRY_INTERVAL_MS = 250
SETTINGS_EXPORT_FORMAT_VERSION = 2


def default_preferences() -> dict[str, Any]:
    return {
        "lists": default_lists_payload(),
        "list_order": default_list_order(),
        "list_meta": default_list_meta(),
        "active_preview_list_id": AUTO_RECENT_LIST_ID,
        "obstructions": {direction: 20.0 for direction in WIND16},
        "location": copy.deepcopy(DEFAULT_LOCATION),
        "temperature_unit": "auto",
        "ui_theme": UI_THEME_LIGHT,
    }


def ensure_preferences_shape(raw: dict[str, Any]) -> dict[str, Any]:
    prefs = default_preferences()
    if isinstance(raw, dict):
        prefs.update(normalize_list_preferences(raw))

        temp_unit = str(raw.get("temperature_unit", "auto")).strip().lower()
        prefs["temperature_unit"] = temp_unit if temp_unit in {"auto", "f", "c"} else "auto"

        ui_theme = str(raw.get("ui_theme", UI_THEME_LIGHT)).strip().lower()
        prefs["ui_theme"] = ui_theme if ui_theme in UI_THEME_OPTIONS else UI_THEME_LIGHT

        obs = raw.get("obstructions", {})
        if isinstance(obs, dict):
            for key in WIND16:
                value = obs.get(key, 20.0)
                try:
                    prefs["obstructions"][key] = float(value)
                except (TypeError, ValueError):
                    prefs["obstructions"][key] = 20.0

        loc = raw.get("location", {})
        if isinstance(loc, dict):
            merged = copy.deepcopy(DEFAULT_LOCATION)
            merged.update({k: loc.get(k, merged[k]) for k in merged})
            # Migrate legacy built-in default location payloads (source=default)
            # to the new explicit "unset" state.
            if str(merged.get("source", "")).strip().lower() == "default":
                merged = copy.deepcopy(DEFAULT_LOCATION)
            prefs["location"] = merged

    return prefs


def encode_preferences_for_storage(prefs: dict[str, Any]) -> str:
    compact_json = json.dumps(ensure_preferences_shape(prefs), separators=(",", ":"), ensure_ascii=True)
    return base64.urlsafe_b64encode(compact_json.encode("utf-8")).decode("ascii")


def decode_preferences_from_storage(raw_value: str) -> dict[str, Any] | None:
    try:
        decoded_json = base64.urlsafe_b64decode(str(raw_value).encode("ascii")).decode("utf-8")
        payload = json.loads(decoded_json)
        if not isinstance(payload, dict):
            return None
        return ensure_preferences_shape(payload)
    except Exception:
        return None


def _eval_js_hidden(js_expression: str, *, key: str, want_output: bool = True) -> Any:
    # Keep streamlit_js_eval utility probes from reserving visible layout height.
    wrapped_expression = "(setFrameHeight(0), (" + str(js_expression) + "))"
    return streamlit_js_eval(js_expressions=wrapped_expression, key=key, want_output=want_output)


def load_preferences() -> tuple[dict[str, Any], bool]:
    retry_needed = False
    raw_local = get_local_storage(BROWSER_PREFS_STORAGE_KEY, component_key="browser_prefs_read")
    local_exists_probe = _eval_js_hidden(
        (
            "Object.prototype.hasOwnProperty.call(window.localStorage, "
            + json.dumps(BROWSER_PREFS_STORAGE_KEY)
            + ")"
        ),
        key="browser_prefs_local_exists_probe",
    )

    if raw_local is None and local_exists_probe is None:
        retry_needed = True

    if isinstance(raw_local, str) and raw_local.strip():
        decoded = decode_preferences_from_storage(raw_local)
        if decoded is not None:
            st.session_state.pop("prefs_persistence_notice", None)
            return decoded, False
    elif local_exists_probe is True:
        # Local key appears to exist but returned value is unavailable in this pass.
        retry_needed = True

    return default_preferences(), retry_needed


def save_preferences(prefs: dict[str, Any]) -> bool:
    try:
        encoded = encode_preferences_for_storage(prefs)
    except Exception:
        st.session_state["prefs_persistence_notice"] = (
            "Browser-local preference storage is unavailable. Using session-only preferences."
        )
        return False

    payload_hash = hashlib.sha1(encoded.encode("ascii")).hexdigest()[:12]
    local_saved = False
    try:
        set_local_storage(
            BROWSER_PREFS_STORAGE_KEY,
            encoded,
            component_key=f"browser_prefs_write_{payload_hash}",
        )
        local_saved = True
    except Exception:
        local_saved = False

    if local_saved:
        st.session_state.pop("prefs_persistence_notice", None)
        return True

    st.session_state["prefs_persistence_notice"] = (
        "Browser-local preference storage is unavailable. Using session-only preferences."
    )
    return False


def build_settings_export_payload(prefs: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "dso_explorer_settings",
        "version": SETTINGS_EXPORT_FORMAT_VERSION,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "preferences": ensure_preferences_shape(prefs),
    }


def parse_settings_import_payload(raw_text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    candidate = payload
    if isinstance(payload.get("preferences"), dict):
        candidate = payload["preferences"]

    if not isinstance(candidate, dict):
        return None
    return ensure_preferences_shape(candidate)


def persist_and_rerun(prefs: dict[str, Any]) -> None:
    st.session_state["prefs"] = prefs
    save_preferences(prefs)
    st.rerun()
