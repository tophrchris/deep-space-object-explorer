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

def apply_resolved_location(prefs: dict[str, Any], resolved_location: dict[str, Any]) -> tuple[str, bool]:
    _refresh_legacy_globals()
    active_site = get_site_definition(prefs, get_active_site_id(prefs))
    current_site_name = str(active_site.get("name") or "").strip()
    resolved_label = str(resolved_location.get("label") or "").strip()

    keep_existing_site_name = bool(current_site_name) and not is_default_site_name(current_site_name)
    next_label = current_site_name if keep_existing_site_name else resolved_label
    if not next_label:
        next_label = current_site_name or resolved_label or DEFAULT_SITE_NAME

    lat = float(resolved_location["lat"])
    lon = float(resolved_location["lon"])
    source = str(resolved_location.get("source") or "manual").strip() or "manual"
    resolved_at = str(resolved_location.get("resolved_at") or datetime.now(timezone.utc).isoformat()).strip()

    prefs["location"] = {
        "lat": lat,
        "lon": lon,
        "label": next_label,
        "source": source,
        "resolved_at": resolved_at,
    }
    persist_legacy_fields_to_active_site(prefs)

    result_label = resolved_label or f"{lat:.4f}, {lon:.4f}"
    return result_label, keep_existing_site_name



def apply_browser_geolocation_payload(prefs: dict[str, Any], payload: Any) -> None:
    _refresh_legacy_globals()
    try:
        if isinstance(payload, dict):
            coords = payload.get("coords")
            if isinstance(coords, dict):
                lat = float(coords.get("latitude"))
                lon = float(coords.get("longitude"))
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    raise ValueError("Coordinates out of range")

                resolved_label, kept_site_name = apply_resolved_location(
                    prefs,
                    {
                        "lat": lat,
                        "lon": lon,
                        "label": reverse_geocode_label(lat, lon),
                        "source": "browser",
                        "resolved_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                st.session_state["location_notice"] = (
                    f"Browser geolocation applied: {resolved_label}. Site name unchanged."
                    if kept_site_name
                    else f"Browser geolocation applied: {resolved_label}."
                )
                st.session_state["prefs"] = prefs
                save_preferences(prefs)
                return

            error = payload.get("error")
            if isinstance(error, dict):
                code = str(error.get("code", "")).strip()
                if code == "1":
                    st.session_state["location_notice"] = (
                        "Location permission denied - keeping previous location."
                    )
                elif code == "2":
                    st.session_state["location_notice"] = (
                        "Location unavailable - keeping previous location."
                    )
                elif code == "3":
                    st.session_state["location_notice"] = (
                        "Location request timed out - keeping previous location."
                    )
                else:
                    message = str(error.get("message", "")).strip()
                    detail = f": {message}" if message else "."
                    st.session_state["location_notice"] = (
                        f"Could not resolve browser geolocation{detail} Keeping previous location."
                    )
                return
    except Exception:
        st.session_state["location_notice"] = (
            "Could not parse browser geolocation - keeping previous location."
        )
        return

    st.session_state["location_notice"] = (
        "Could not read browser geolocation response - keeping previous location."
    )
