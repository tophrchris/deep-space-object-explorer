from __future__ import annotations

import base64
import copy
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from app_constants import (
    DEFAULT_LOCATION,
    DEFAULT_SITE_ID,
    DEFAULT_SITE_NAME,
    UI_THEME_LIGHT,
    UI_THEME_OPTIONS,
    WIND16,
)
from features.lists.list_subsystem import (
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
SETTINGS_EXPORT_FORMAT_VERSION = 3

CLOUD_SYNC_PROVIDER_GOOGLE = "google"
CLOUD_SYNC_PROVIDER_NONE = "none"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_obstructions() -> dict[str, float]:
    return {direction: 20.0 for direction in WIND16}


def default_site_payload(name: str = DEFAULT_SITE_NAME) -> dict[str, Any]:
    site_name = str(name or "").strip() or DEFAULT_SITE_NAME
    location = copy.deepcopy(DEFAULT_LOCATION)
    location["label"] = site_name
    return {
        "name": site_name,
        "location": location,
        "obstructions": _default_obstructions(),
    }


def default_sites_payload() -> dict[str, dict[str, Any]]:
    return {DEFAULT_SITE_ID: default_site_payload()}


def default_site_order() -> list[str]:
    return [DEFAULT_SITE_ID]


def _normalize_site_payload(raw_site: Any) -> dict[str, Any] | None:
    if not isinstance(raw_site, dict):
        return None

    site_name = str(raw_site.get("name") or "").strip() or DEFAULT_SITE_NAME
    location = copy.deepcopy(DEFAULT_LOCATION)
    raw_location = raw_site.get("location", {})
    if isinstance(raw_location, dict):
        for key in location:
            if key in raw_location:
                location[key] = raw_location.get(key, location[key])
    if str(location.get("source", "")).strip().lower() == "default":
        location = copy.deepcopy(DEFAULT_LOCATION)

    location_label = str(location.get("label") or "").strip()
    if location_label:
        site_name = site_name or location_label
    else:
        location["label"] = site_name

    obstructions = _default_obstructions()
    raw_obstructions = raw_site.get("obstructions", {})
    if isinstance(raw_obstructions, dict):
        for key in WIND16:
            value = raw_obstructions.get(key, 20.0)
            try:
                obstructions[key] = float(value)
            except (TypeError, ValueError):
                obstructions[key] = 20.0

    return {
        "name": site_name,
        "location": location,
        "obstructions": obstructions,
    }


def _normalize_equipment(raw_equipment: Any) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    if not isinstance(raw_equipment, dict):
        return normalized

    for raw_category_id, raw_values in raw_equipment.items():
        category_id = str(raw_category_id).strip()
        if not category_id:
            continue
        if not isinstance(raw_values, (list, tuple, set)):
            continue
        items: list[str] = []
        for raw_value in raw_values:
            value = str(raw_value).strip()
            if value and value not in items:
                items.append(value)
        normalized[category_id] = items
    return normalized


def default_preferences() -> dict[str, Any]:
    sites = default_sites_payload()
    active_site_id = default_site_order()[0]
    active_site = sites[active_site_id]
    return {
        "lists": default_lists_payload(),
        "list_order": default_list_order(),
        "list_meta": default_list_meta(),
        "active_preview_list_id": AUTO_RECENT_LIST_ID,
        "sites": sites,
        "site_order": default_site_order(),
        "active_site_id": active_site_id,
        "equipment": {},
        "active_telescope_id": "",
        "active_filter_id": "__none__",
        "active_mount_choice": "altaz",
        # Legacy convenience fields mirror the active site.
        "obstructions": copy.deepcopy(active_site["obstructions"]),
        "location": copy.deepcopy(active_site["location"]),
        "temperature_unit": "auto",
        "ui_theme": UI_THEME_LIGHT,
        "last_updated_utc": _utc_now_iso(),
        "cloud_sync_provider": CLOUD_SYNC_PROVIDER_GOOGLE,
        "cloud_sync_enabled": False,
        "cloud_sync_initialized": False,
        "cloud_sync_file_id": "",
        "cloud_sync_last_ok_utc": "",
        "cloud_sync_last_error": "",
    }


def ensure_preferences_shape(raw: dict[str, Any]) -> dict[str, Any]:
    prefs = default_preferences()
    if isinstance(raw, dict):
        prefs.update(normalize_list_preferences(raw))

        temp_unit = str(raw.get("temperature_unit", "auto")).strip().lower()
        prefs["temperature_unit"] = temp_unit if temp_unit in {"auto", "f", "c"} else "auto"

        ui_theme = str(raw.get("ui_theme", UI_THEME_LIGHT)).strip().lower()
        prefs["ui_theme"] = ui_theme if ui_theme in UI_THEME_OPTIONS else UI_THEME_LIGHT

        raw_last_updated_utc = str(raw.get("last_updated_utc", "")).strip()
        prefs["last_updated_utc"] = raw_last_updated_utc

        raw_cloud_provider = str(raw.get("cloud_sync_provider", CLOUD_SYNC_PROVIDER_GOOGLE)).strip().lower()
        if raw_cloud_provider not in {CLOUD_SYNC_PROVIDER_GOOGLE, CLOUD_SYNC_PROVIDER_NONE}:
            raw_cloud_provider = CLOUD_SYNC_PROVIDER_GOOGLE
        prefs["cloud_sync_provider"] = raw_cloud_provider
        prefs["cloud_sync_enabled"] = bool(raw.get("cloud_sync_enabled", False))
        prefs["cloud_sync_initialized"] = bool(raw.get("cloud_sync_initialized", False))
        prefs["cloud_sync_file_id"] = str(raw.get("cloud_sync_file_id", "")).strip()
        prefs["cloud_sync_last_ok_utc"] = str(raw.get("cloud_sync_last_ok_utc", "")).strip()
        prefs["cloud_sync_last_error"] = str(raw.get("cloud_sync_last_error", "")).strip()

        normalized_sites: dict[str, dict[str, Any]] = {}
        raw_sites = raw.get("sites", {})
        if isinstance(raw_sites, dict):
            for raw_site_id, raw_site in raw_sites.items():
                site_id = str(raw_site_id).strip()
                if not site_id:
                    continue
                site_payload = _normalize_site_payload(raw_site)
                if site_payload is not None:
                    normalized_sites[site_id] = site_payload

        if not normalized_sites:
            # Fallback from legacy single-site payloads when sites are absent.
            legacy_site = default_site_payload()
            loc = raw.get("location", {})
            if isinstance(loc, dict):
                merged = copy.deepcopy(DEFAULT_LOCATION)
                merged.update({k: loc.get(k, merged[k]) for k in merged})
                if str(merged.get("source", "")).strip().lower() == "default":
                    merged = copy.deepcopy(DEFAULT_LOCATION)
                legacy_site["location"] = merged
                legacy_label = str(merged.get("label") or "").strip()
                if legacy_label:
                    legacy_site["name"] = legacy_label
            obs = raw.get("obstructions", {})
            if isinstance(obs, dict):
                for key in WIND16:
                    value = obs.get(key, 20.0)
                    try:
                        legacy_site["obstructions"][key] = float(value)
                    except (TypeError, ValueError):
                        legacy_site["obstructions"][key] = 20.0
            normalized_sites[DEFAULT_SITE_ID] = legacy_site

        raw_site_order = raw.get("site_order", [])
        site_order: list[str] = []
        if isinstance(raw_site_order, (list, tuple)):
            for raw_site_id in raw_site_order:
                site_id = str(raw_site_id).strip()
                if site_id and site_id in normalized_sites and site_id not in site_order:
                    site_order.append(site_id)
        for site_id in normalized_sites:
            if site_id not in site_order:
                site_order.append(site_id)
        if not site_order:
            site_order = list(normalized_sites.keys()) or [DEFAULT_SITE_ID]

        active_site_id = str(raw.get("active_site_id", "")).strip()
        if active_site_id not in normalized_sites:
            active_site_id = site_order[0]

        prefs["sites"] = normalized_sites
        prefs["site_order"] = site_order
        prefs["active_site_id"] = active_site_id
        prefs["equipment"] = _normalize_equipment(raw.get("equipment", {}))
        prefs["active_telescope_id"] = str(raw.get("active_telescope_id", "")).strip()
        raw_filter_id = str(raw.get("active_filter_id", "__none__")).strip()
        prefs["active_filter_id"] = raw_filter_id if raw_filter_id else "__none__"
        raw_mount_choice = str(raw.get("active_mount_choice", "altaz")).strip().lower()
        if raw_mount_choice not in {"eq", "altaz"}:
            raw_mount_choice = "altaz"
        prefs["active_mount_choice"] = raw_mount_choice

        active_site = normalized_sites.get(active_site_id) or default_site_payload()
        prefs["location"] = copy.deepcopy(active_site.get("location", DEFAULT_LOCATION))
        prefs["obstructions"] = copy.deepcopy(active_site.get("obstructions", _default_obstructions()))

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


def save_preferences(
    prefs: dict[str, Any],
    *,
    mark_cloud_pending: bool = True,
    touch_last_updated_utc: bool = True,
) -> bool:
    normalized_prefs = ensure_preferences_shape(prefs)
    if touch_last_updated_utc:
        normalized_prefs["last_updated_utc"] = _utc_now_iso()
    prefs.clear()
    prefs.update(normalized_prefs)

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
        if mark_cloud_pending:
            st.session_state["cloud_sync_pending"] = True
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
