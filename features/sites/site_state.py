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

