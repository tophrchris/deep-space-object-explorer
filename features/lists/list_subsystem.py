from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

AUTO_RECENT_LIST_ID = "auto_recent"
AUTO_RECENT_LIST_NAME = "Auto (Recent)"
AUTO_RECENT_LIST_MAX_ITEMS = 10
DEFAULT_EDITABLE_LIST_ID = "targets"
DEFAULT_EDITABLE_LIST_NAME = "Targets"


def clean_primary_id_list(raw_values: Any) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    if not isinstance(raw_values, (list, tuple, set)):
        return cleaned
    for value in raw_values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        cleaned.append(item)
        seen.add(item)
    return cleaned


def default_lists_payload() -> dict[str, list[str]]:
    return {
        AUTO_RECENT_LIST_ID: [],
        DEFAULT_EDITABLE_LIST_ID: [],
    }


def default_list_order() -> list[str]:
    return [
        AUTO_RECENT_LIST_ID,
        DEFAULT_EDITABLE_LIST_ID,
    ]


def default_list_meta() -> dict[str, dict[str, Any]]:
    return {
        AUTO_RECENT_LIST_ID: {
            "name": AUTO_RECENT_LIST_NAME,
            "system": True,
            "max_items": AUTO_RECENT_LIST_MAX_ITEMS,
        },
        DEFAULT_EDITABLE_LIST_ID: {
            "name": DEFAULT_EDITABLE_LIST_NAME,
            "system": False,
        },
    }


def _merge_list_state_into_preferences(
    prefs: dict[str, Any],
    list_state: Mapping[str, Any],
) -> None:
    merged = dict(prefs)
    for key, value in list_state.items():
        merged[key] = value
    prefs.clear()
    prefs.update(merged)


def _sanitize_list_name(value: Any) -> str:
    return str(value or "").strip()


def _slugify_list_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return slug


def _build_unique_list_id(existing_ids: set[str], desired: str) -> str:
    base = _slugify_list_id(desired) or "list"
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def normalize_list_preferences(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    lists_payload = default_lists_payload()
    if isinstance(raw, Mapping):
        raw_lists = raw.get("lists")
        if isinstance(raw_lists, Mapping):
            for raw_list_id, raw_ids in raw_lists.items():
                list_id = str(raw_list_id).strip()
                if not list_id:
                    continue
                lists_payload[list_id] = clean_primary_id_list(raw_ids)

    lists_payload[AUTO_RECENT_LIST_ID] = lists_payload.get(AUTO_RECENT_LIST_ID, [])[-AUTO_RECENT_LIST_MAX_ITEMS:]

    list_meta = default_list_meta()
    for list_id in lists_payload:
        if list_id not in list_meta:
            list_meta[list_id] = {"name": list_id, "system": False}

    raw_meta = raw.get("list_meta") if isinstance(raw, Mapping) else None
    if isinstance(raw_meta, Mapping):
        for raw_list_id, raw_entry in raw_meta.items():
            list_id = str(raw_list_id).strip()
            if not list_id or list_id not in lists_payload:
                continue
            if not isinstance(raw_entry, Mapping):
                continue
            name = str(raw_entry.get("name", "")).strip()
            if name:
                list_meta[list_id]["name"] = name
            if list_id != AUTO_RECENT_LIST_ID:
                list_meta[list_id]["system"] = bool(raw_entry.get("system", list_meta[list_id].get("system", False)))

    list_meta[AUTO_RECENT_LIST_ID] = {
        "name": AUTO_RECENT_LIST_NAME,
        "system": True,
        "max_items": AUTO_RECENT_LIST_MAX_ITEMS,
    }

    list_order: list[str] = []
    seen_order: set[str] = set()
    raw_order = raw.get("list_order") if isinstance(raw, Mapping) else None
    if isinstance(raw_order, list):
        for value in raw_order:
            list_id = str(value).strip()
            if not list_id or list_id in seen_order or list_id not in lists_payload:
                continue
            list_order.append(list_id)
            seen_order.add(list_id)
    for list_id in default_list_order():
        if list_id in lists_payload and list_id not in seen_order:
            list_order.append(list_id)
            seen_order.add(list_id)
    for list_id in lists_payload:
        if list_id not in seen_order:
            list_order.append(list_id)
            seen_order.add(list_id)

    raw_preview = raw.get("active_preview_list_id", AUTO_RECENT_LIST_ID) if isinstance(raw, Mapping) else AUTO_RECENT_LIST_ID
    preview_list_id = str(raw_preview).strip()
    if preview_list_id not in lists_payload:
        preview_list_id = AUTO_RECENT_LIST_ID

    return {
        "lists": lists_payload,
        "list_order": list_order,
        "list_meta": list_meta,
        "active_preview_list_id": preview_list_id,
    }


def all_listed_ids_in_order(prefs: Mapping[str, Any], *, include_auto_recent: bool = True) -> list[str]:
    normalized = normalize_list_preferences(prefs)
    lists_payload = normalized.get("lists", {})
    if not isinstance(lists_payload, Mapping):
        return []

    seen: set[str] = set()
    ordered_ids: list[str] = []
    list_order = normalized.get("list_order", [])
    if not isinstance(list_order, list):
        list_order = []

    def _append_list_items(list_id: str) -> None:
        if not include_auto_recent and list_id == AUTO_RECENT_LIST_ID:
            return
        values = clean_primary_id_list(lists_payload.get(list_id, []))
        for value in values:
            if value in seen:
                continue
            ordered_ids.append(value)
            seen.add(value)

    for value in list_order:
        list_id = str(value).strip()
        if list_id:
            _append_list_items(list_id)

    for raw_list_id in lists_payload.keys():
        list_id = str(raw_list_id).strip()
        if list_id:
            _append_list_items(list_id)

    return ordered_ids


def list_ids_in_order(prefs: Mapping[str, Any], *, include_auto_recent: bool = True) -> list[str]:
    normalized = normalize_list_preferences(prefs)
    lists_payload = normalized.get("lists", {})
    if not isinstance(lists_payload, Mapping):
        return []

    ordered: list[str] = []
    seen: set[str] = set()
    for value in normalized.get("list_order", []):
        list_id = str(value).strip()
        if not list_id or list_id in seen or list_id not in lists_payload:
            continue
        if not include_auto_recent and list_id == AUTO_RECENT_LIST_ID:
            continue
        ordered.append(list_id)
        seen.add(list_id)

    for raw_list_id in lists_payload:
        list_id = str(raw_list_id).strip()
        if not list_id or list_id in seen:
            continue
        if not include_auto_recent and list_id == AUTO_RECENT_LIST_ID:
            continue
        ordered.append(list_id)
        seen.add(list_id)
    return ordered


def editable_list_ids_in_order(prefs: Mapping[str, Any]) -> list[str]:
    return [
        list_id
        for list_id in list_ids_in_order(prefs, include_auto_recent=True)
        if not is_system_list(prefs, list_id)
    ]


def get_list_ids(prefs: Mapping[str, Any], list_id: str) -> list[str]:
    normalized = normalize_list_preferences(prefs)
    cleaned_list_id = str(list_id).strip()
    if not cleaned_list_id:
        return []
    return clean_primary_id_list(normalized.get("lists", {}).get(cleaned_list_id, []))


def get_list_name(prefs: Mapping[str, Any], list_id: str) -> str:
    normalized = normalize_list_preferences(prefs)
    cleaned_list_id = str(list_id).strip()
    if not cleaned_list_id:
        return ""
    raw_name = normalized.get("list_meta", {}).get(cleaned_list_id, {}).get("name", "")
    resolved = str(raw_name).strip()
    return resolved or cleaned_list_id


def is_system_list(prefs: Mapping[str, Any], list_id: str) -> bool:
    normalized = normalize_list_preferences(prefs)
    cleaned_list_id = str(list_id).strip()
    if not cleaned_list_id:
        return False
    return bool(normalized.get("list_meta", {}).get(cleaned_list_id, {}).get("system", False))


def get_active_preview_list_id(prefs: Mapping[str, Any]) -> str:
    normalized = normalize_list_preferences(prefs)
    preview_list_id = str(normalized.get("active_preview_list_id", AUTO_RECENT_LIST_ID)).strip()
    if preview_list_id in normalized.get("lists", {}):
        return preview_list_id
    return AUTO_RECENT_LIST_ID


def set_active_preview_list_id(prefs: dict[str, Any], list_id: str) -> bool:
    normalized = normalize_list_preferences(prefs)
    next_list_id = str(list_id).strip()
    if next_list_id not in normalized.get("lists", {}):
        next_list_id = AUTO_RECENT_LIST_ID
    if next_list_id == str(normalized.get("active_preview_list_id", AUTO_RECENT_LIST_ID)).strip():
        return False
    updated = dict(normalized)
    updated["active_preview_list_id"] = next_list_id
    reshaped = normalize_list_preferences(updated)
    _merge_list_state_into_preferences(prefs, reshaped)
    return True


def create_list(prefs: dict[str, Any], name: str) -> str | None:
    normalized = normalize_list_preferences(prefs)
    cleaned_name = _sanitize_list_name(name)
    if not cleaned_name:
        return None

    lists_payload = dict(normalized.get("lists", {}))
    list_meta = dict(normalized.get("list_meta", {}))
    list_order = list(normalized.get("list_order", []))
    existing_ids = {str(list_id).strip() for list_id in lists_payload if str(list_id).strip()}
    next_list_id = _build_unique_list_id(existing_ids, cleaned_name)
    if not next_list_id:
        return None

    lists_payload[next_list_id] = []
    list_meta[next_list_id] = {"name": cleaned_name, "system": False}
    if next_list_id not in list_order:
        list_order.append(next_list_id)

    updated = dict(normalized)
    updated["lists"] = lists_payload
    updated["list_meta"] = list_meta
    updated["list_order"] = list_order
    reshaped = normalize_list_preferences(updated)
    _merge_list_state_into_preferences(prefs, reshaped)
    return next_list_id


def rename_list(prefs: dict[str, Any], list_id: str, name: str) -> bool:
    normalized = normalize_list_preferences(prefs)
    cleaned_list_id = str(list_id).strip()
    cleaned_name = _sanitize_list_name(name)
    if not cleaned_list_id or not cleaned_name:
        return False
    if cleaned_list_id not in normalized.get("lists", {}):
        return False
    if is_system_list(normalized, cleaned_list_id):
        return False

    list_meta = dict(normalized.get("list_meta", {}))
    current_entry = dict(list_meta.get(cleaned_list_id, {}))
    current_name = str(current_entry.get("name", "")).strip()
    if cleaned_name == current_name:
        return False

    current_entry["name"] = cleaned_name
    current_entry["system"] = False
    list_meta[cleaned_list_id] = current_entry

    updated = dict(normalized)
    updated["list_meta"] = list_meta
    reshaped = normalize_list_preferences(updated)
    _merge_list_state_into_preferences(prefs, reshaped)
    return True


def delete_list(prefs: dict[str, Any], list_id: str) -> bool:
    normalized = normalize_list_preferences(prefs)
    cleaned_list_id = str(list_id).strip()
    if not cleaned_list_id:
        return False
    if cleaned_list_id not in normalized.get("lists", {}):
        return False
    if is_system_list(normalized, cleaned_list_id):
        return False

    lists_payload = dict(normalized.get("lists", {}))
    list_meta = dict(normalized.get("list_meta", {}))
    list_order = list(normalized.get("list_order", []))

    lists_payload.pop(cleaned_list_id, None)
    list_meta.pop(cleaned_list_id, None)
    list_order = [value for value in list_order if str(value).strip() != cleaned_list_id]

    # Remove any persisted per-list target schedules for this list.
    list_target_schedules = prefs.get("list_target_schedules")
    if isinstance(list_target_schedules, Mapping) and cleaned_list_id in list_target_schedules:
        next_schedules = dict(list_target_schedules)
        next_schedules.pop(cleaned_list_id, None)
        prefs["list_target_schedules"] = next_schedules

    updated = dict(normalized)
    updated["lists"] = lists_payload
    updated["list_meta"] = list_meta
    updated["list_order"] = list_order
    if str(updated.get("active_preview_list_id", "")).strip() == cleaned_list_id:
        updated["active_preview_list_id"] = AUTO_RECENT_LIST_ID
    reshaped = normalize_list_preferences(updated)
    _merge_list_state_into_preferences(prefs, reshaped)
    return True


def move_list(prefs: dict[str, Any], list_id: str, direction: int) -> bool:
    if direction not in {-1, 1}:
        return False

    normalized = normalize_list_preferences(prefs)
    cleaned_list_id = str(list_id).strip()
    if not cleaned_list_id:
        return False
    if cleaned_list_id not in normalized.get("lists", {}):
        return False
    if is_system_list(normalized, cleaned_list_id):
        return False

    all_order = list_ids_in_order(normalized, include_auto_recent=True)
    user_order = [value for value in all_order if not is_system_list(normalized, value)]
    if cleaned_list_id not in user_order:
        return False

    current_idx = user_order.index(cleaned_list_id)
    next_idx = current_idx + direction
    if next_idx < 0 or next_idx >= len(user_order):
        return False

    user_order[current_idx], user_order[next_idx] = user_order[next_idx], user_order[current_idx]
    system_order = [value for value in all_order if is_system_list(normalized, value)]

    updated = dict(normalized)
    updated["list_order"] = system_order + user_order
    reshaped = normalize_list_preferences(updated)
    _merge_list_state_into_preferences(prefs, reshaped)
    return True


def toggle_target_in_list(prefs: dict[str, Any], list_id: str, primary_id: str) -> bool:
    normalized = normalize_list_preferences(prefs)
    cleaned_list_id = str(list_id).strip()
    target_id = str(primary_id).strip()
    if not cleaned_list_id or not target_id:
        return False
    lists_payload = dict(normalized.get("lists", {}))
    if cleaned_list_id not in lists_payload:
        return False

    current_ids = clean_primary_id_list(lists_payload.get(cleaned_list_id, []))
    if target_id in current_ids:
        next_ids = [value for value in current_ids if value != target_id]
    else:
        next_ids = list(current_ids) + [target_id]
    if cleaned_list_id == AUTO_RECENT_LIST_ID:
        next_ids = next_ids[-AUTO_RECENT_LIST_MAX_ITEMS:]
    if next_ids == current_ids:
        return False

    lists_payload[cleaned_list_id] = next_ids
    updated = dict(normalized)
    updated["lists"] = lists_payload
    reshaped = normalize_list_preferences(updated)
    _merge_list_state_into_preferences(prefs, reshaped)
    return True


def push_target_to_auto_recent_list(prefs: dict[str, Any], primary_id: str) -> bool:
    target_id = str(primary_id).strip()
    if not target_id:
        return False

    normalized = normalize_list_preferences(prefs)
    lists_payload = dict(normalized.get("lists", {}))
    current_auto_ids = clean_primary_id_list(lists_payload.get(AUTO_RECENT_LIST_ID, []))
    next_auto_ids = [value for value in current_auto_ids if value != target_id]
    next_auto_ids.append(target_id)
    next_auto_ids = next_auto_ids[-AUTO_RECENT_LIST_MAX_ITEMS:]
    if next_auto_ids == current_auto_ids:
        return False

    lists_payload[AUTO_RECENT_LIST_ID] = next_auto_ids
    updated = dict(normalized)
    updated["lists"] = lists_payload
    reshaped = normalize_list_preferences(updated)
    _merge_list_state_into_preferences(prefs, reshaped)
    return True
