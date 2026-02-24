from __future__ import annotations

from typing import Any

from features.equipment.catalog import load_equipment_catalog


def _normalize_mount_choice(value: Any, *, default_choice: str = "altaz") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"eq", "equatorial", "equatorial-mount"}:
        return "eq"
    if normalized in {"altaz", "alt/az", "alt-az", "alt az"}:
        return "altaz"
    return default_choice


def mount_choice_label(choice: str) -> str:
    return "EQ" if str(choice).strip().lower() == "eq" else "Alt/Az"


def build_owned_equipment_context(prefs: dict[str, Any]) -> dict[str, Any]:
    equipment_catalog = load_equipment_catalog()
    categories = equipment_catalog.get("categories", [])
    category_items_by_id: dict[str, dict[str, dict[str, Any]]] = {}
    if isinstance(categories, list):
        for category in categories:
            if not isinstance(category, dict):
                continue
            category_id = str(category.get("id", "")).strip()
            if not category_id:
                continue
            items = category.get("items", [])
            if not isinstance(items, list):
                continue
            item_lookup = {
                str(item.get("id", "")).strip(): item
                for item in items
                if isinstance(item, dict) and str(item.get("id", "")).strip()
            }
            category_items_by_id[category_id] = item_lookup

    owned_equipment = prefs.get("equipment", {})
    if not isinstance(owned_equipment, dict):
        owned_equipment = {}

    telescope_lookup = category_items_by_id.get("telescopes", {})
    filter_lookup = category_items_by_id.get("filters", {})

    owned_telescope_ids = [
        item_id
        for item_id in [str(item).strip() for item in owned_equipment.get("telescopes", []) if str(item).strip()]
        if item_id in telescope_lookup
    ]
    owned_filter_ids = [
        item_id
        for item_id in [str(item).strip() for item in owned_equipment.get("filters", []) if str(item).strip()]
        if item_id in filter_lookup
    ]
    owned_accessory_ids = {
        str(item).strip()
        for item in owned_equipment.get("accessories", [])
        if str(item).strip()
    }
    owned_telescopes = [telescope_lookup[item_id] for item_id in owned_telescope_ids]
    owned_filters = [filter_lookup[item_id] for item_id in owned_filter_ids]

    return {
        "category_items_by_id": category_items_by_id,
        "telescope_lookup": telescope_lookup,
        "filter_lookup": filter_lookup,
        "owned_telescope_ids": owned_telescope_ids,
        "owned_filter_ids": owned_filter_ids,
        "owned_accessory_ids": owned_accessory_ids,
        "owned_telescopes": owned_telescopes,
        "owned_filters": owned_filters,
        "eq_owned": "equatorial-mount" in owned_accessory_ids,
    }


def sync_active_equipment_settings(
    prefs: dict[str, Any],
    equipment_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = equipment_context if isinstance(equipment_context, dict) else build_owned_equipment_context(prefs)

    owned_telescope_ids = list(context.get("owned_telescope_ids", []))
    owned_filter_ids = list(context.get("owned_filter_ids", []))
    telescope_lookup = context.get("telescope_lookup", {})
    filter_lookup = context.get("filter_lookup", {})

    default_mount_choice = "eq" if bool(context.get("eq_owned", False)) else "altaz"
    current_mount_choice = _normalize_mount_choice(
        prefs.get("active_mount_choice", default_mount_choice),
        default_choice=default_mount_choice,
    )

    current_telescope_id = str(prefs.get("active_telescope_id", "")).strip()
    if owned_telescope_ids:
        active_telescope_id = (
            current_telescope_id if current_telescope_id in owned_telescope_ids else owned_telescope_ids[0]
        )
    else:
        active_telescope_id = ""

    current_filter_id = str(prefs.get("active_filter_id", "__none__")).strip()
    valid_filter_ids = {"__none__"} | set(owned_filter_ids)
    active_filter_id = current_filter_id if current_filter_id in valid_filter_ids else "__none__"

    changed = False
    if str(prefs.get("active_telescope_id", "")).strip() != active_telescope_id:
        prefs["active_telescope_id"] = active_telescope_id
        changed = True
    if str(prefs.get("active_filter_id", "__none__")).strip() != active_filter_id:
        prefs["active_filter_id"] = active_filter_id
        changed = True
    if _normalize_mount_choice(
        prefs.get("active_mount_choice", default_mount_choice),
        default_choice=default_mount_choice,
    ) != current_mount_choice:
        prefs["active_mount_choice"] = current_mount_choice
        changed = True

    active_telescope = telescope_lookup.get(active_telescope_id) if active_telescope_id else None
    active_filter = filter_lookup.get(active_filter_id) if active_filter_id != "__none__" else None

    return {
        "changed": changed,
        "active_telescope_id": active_telescope_id,
        "active_filter_id": active_filter_id,
        "active_mount_choice": current_mount_choice,
        "active_telescope": active_telescope,
        "active_filter": active_filter,
        **context,
    }
