from __future__ import annotations

# Transitional bridge during Sites split: this module still relies on shared
# helpers/constants from `ui.streamlit_app` until they are extracted.
from ui import streamlit_app as _legacy_ui

def _refresh_legacy_globals() -> None:
    for _name, _value in vars(_legacy_ui).items():
        if _name == "__builtins__":
            continue
        globals().setdefault(_name, _value)


_refresh_legacy_globals()

def persist_legacy_fields_to_active_site(prefs: dict[str, Any]) -> None:
    _refresh_legacy_globals()
    sites = prefs.get("sites", {})
    if not isinstance(sites, dict):
        sites = {}
    active_site_id = get_active_site_id(prefs)
    current_site = get_site_definition(prefs, active_site_id)

    location = prefs.get("location", {})
    merged_location = copy.deepcopy(DEFAULT_LOCATION)
    if isinstance(location, dict):
        for key in merged_location:
            if key in location:
                merged_location[key] = location.get(key, merged_location[key])
    location_label = str(merged_location.get("label") or "").strip()

    obstructions = {
        direction: clamp_obstruction_altitude(
            prefs.get("obstructions", {}).get(direction, 20.0) if isinstance(prefs.get("obstructions"), dict) else 20.0,
            default=20.0,
        )
        for direction in WIND16
    }

    site_name = str(current_site.get("name") or "").strip()
    if location_label:
        site_name = location_label
    if not site_name:
        site_name = DEFAULT_SITE_NAME
        merged_location["label"] = site_name

    sites[active_site_id] = {
        "name": site_name,
        "location": merged_location,
        "obstructions": obstructions,
    }
    prefs["sites"] = sites

    ordered = site_ids_in_order(prefs)
    if active_site_id not in ordered:
        ordered.append(active_site_id)
    prefs["site_order"] = ordered
    prefs["active_site_id"] = active_site_id
    prefs["location"] = copy.deepcopy(merged_location)
    prefs["obstructions"] = copy.deepcopy(obstructions)



def default_site_definition(name: str = DEFAULT_SITE_NAME) -> dict[str, Any]:
    _refresh_legacy_globals()
    site_name = str(name or "").strip() or DEFAULT_SITE_NAME
    location = copy.deepcopy(DEFAULT_LOCATION)
    location["label"] = site_name
    return {
        "name": site_name,
        "location": location,
        "obstructions": {direction: 20.0 for direction in WIND16},
    }


def site_ids_in_order(prefs: dict[str, Any]) -> list[str]:
    _refresh_legacy_globals()
    sites = prefs.get("sites", {})
    if not isinstance(sites, dict):
        return []

    ordered: list[str] = []
    raw_order = prefs.get("site_order", [])
    if isinstance(raw_order, (list, tuple)):
        for raw_site_id in raw_order:
            site_id = str(raw_site_id).strip()
            if site_id and site_id in sites and site_id not in ordered:
                ordered.append(site_id)

    for raw_site_id in sites.keys():
        site_id = str(raw_site_id).strip()
        if site_id and site_id not in ordered:
            ordered.append(site_id)
    return ordered


def get_active_site_id(prefs: dict[str, Any]) -> str:
    _refresh_legacy_globals()
    ordered = site_ids_in_order(prefs)
    if not ordered:
        return DEFAULT_SITE_ID
    candidate = str(prefs.get("active_site_id", "")).strip()
    return candidate if candidate in ordered else ordered[0]


def get_site_definition(prefs: dict[str, Any], site_id: str) -> dict[str, Any]:
    _refresh_legacy_globals()
    sites = prefs.get("sites", {})
    if not isinstance(sites, dict):
        return default_site_definition()
    site = sites.get(site_id)
    if isinstance(site, dict):
        return site
    return default_site_definition()


def sync_active_site_to_legacy_fields(prefs: dict[str, Any]) -> None:
    _refresh_legacy_globals()
    ordered = site_ids_in_order(prefs)
    if not ordered:
        default_site = default_site_definition()
        prefs["sites"] = {DEFAULT_SITE_ID: default_site}
        prefs["site_order"] = [DEFAULT_SITE_ID]
        prefs["active_site_id"] = DEFAULT_SITE_ID
        ordered = [DEFAULT_SITE_ID]

    active_site_id = get_active_site_id(prefs)
    active_site = get_site_definition(prefs, active_site_id)
    site_name = str(active_site.get("name") or "").strip() or DEFAULT_SITE_NAME
    location = copy.deepcopy(active_site.get("location", DEFAULT_LOCATION))
    if not str(location.get("label") or "").strip():
        location["label"] = site_name
    obstructions_raw = active_site.get("obstructions", {})
    obstructions = {
        direction: clamp_obstruction_altitude(
            obstructions_raw.get(direction, 20.0) if isinstance(obstructions_raw, dict) else 20.0,
            default=20.0,
        )
        for direction in WIND16
    }

    prefs["active_site_id"] = active_site_id
    prefs["location"] = location
    prefs["obstructions"] = obstructions



def set_active_site(prefs: dict[str, Any], site_id: str) -> bool:
    _refresh_legacy_globals()
    ordered = site_ids_in_order(prefs)
    if site_id not in ordered:
        return False
    changed = str(prefs.get("active_site_id", "")).strip() != site_id
    prefs["active_site_id"] = site_id
    sync_active_site_to_legacy_fields(prefs)
    return changed


def duplicate_site(prefs: dict[str, Any], site_id: str) -> str | None:
    _refresh_legacy_globals()
    source_site = get_site_definition(prefs, site_id)
    source_name = str(source_site.get("name") or "").strip() or DEFAULT_SITE_NAME
    sites = prefs.get("sites", {})
    if not isinstance(sites, dict):
        sites = {}

    existing_names = {
        str(site.get("name") or "").strip()
        for site in sites.values()
        if isinstance(site, dict) and str(site.get("name") or "").strip()
    }
    copy_index = 1
    candidate_name = f"{source_name} - copy {copy_index}"
    while candidate_name in existing_names:
        copy_index += 1
        candidate_name = f"{source_name} - copy {copy_index}"

    duplicated = copy.deepcopy(source_site)
    duplicated["name"] = candidate_name
    duplicated_location = duplicated.get("location", {})
    if isinstance(duplicated_location, dict):
        duplicated_location["label"] = candidate_name
        duplicated["location"] = duplicated_location

    new_site_id = f"site_{uuid.uuid4().hex[:8]}"
    sites[new_site_id] = duplicated
    prefs["sites"] = sites

    ordered = site_ids_in_order(prefs)
    if site_id in ordered:
        insert_idx = ordered.index(site_id) + 1
        ordered.insert(insert_idx, new_site_id)
    else:
        ordered.append(new_site_id)
    prefs["site_order"] = ordered
    return new_site_id


def create_site(prefs: dict[str, Any], name: str | None = None) -> str | None:
    _refresh_legacy_globals()
    sites = prefs.get("sites", {})
    if not isinstance(sites, dict):
        sites = {}

    base_name = str(name or "").strip() or DEFAULT_SITE_NAME
    existing_names = {
        str(site.get("name") or "").strip().lower()
        for site in sites.values()
        if isinstance(site, dict) and str(site.get("name") or "").strip()
    }
    candidate_name = base_name
    suffix = 2
    while candidate_name.strip().lower() in existing_names:
        candidate_name = f"{base_name} {suffix}"
        suffix += 1

    new_site_id = f"site_{uuid.uuid4().hex[:8]}"
    sites[new_site_id] = default_site_definition(candidate_name)
    prefs["sites"] = sites

    ordered = site_ids_in_order(prefs)
    if new_site_id not in ordered:
        ordered.append(new_site_id)
    prefs["site_order"] = ordered
    return new_site_id


def delete_site(prefs: dict[str, Any], site_id: str) -> bool:
    _refresh_legacy_globals()
    ordered = site_ids_in_order(prefs)
    if site_id not in ordered or len(ordered) <= 1:
        return False

    sites = prefs.get("sites", {})
    if not isinstance(sites, dict):
        return False
    sites.pop(site_id, None)
    prefs["sites"] = sites

    ordered = [item for item in ordered if item != site_id]
    prefs["site_order"] = ordered
    if str(prefs.get("active_site_id", "")).strip() == site_id:
        prefs["active_site_id"] = ordered[0]
    sync_active_site_to_legacy_fields(prefs)
    return True

