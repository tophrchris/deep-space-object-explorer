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

def resolve_manual_location(query: str) -> dict[str, Any] | None:
    _refresh_legacy_globals()
    cleaned = query.strip()
    if not cleaned:
        return None

    def _valid_lat_lon(lat_value: Any, lon_value: Any) -> tuple[float, float] | None:
        try:
            lat = float(lat_value)
            lon = float(lon_value)
        except (TypeError, ValueError):
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        return lat, lon

    def _payload(lat: float, lon: float, label: str) -> dict[str, Any]:
        clean_label = str(label or "").strip() or f"{lat:.4f}, {lon:.4f}"
        return {
            "lat": lat,
            "lon": lon,
            "label": clean_label,
            "source": "search",
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }

    coord_match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", cleaned)
    if coord_match:
        coords = _valid_lat_lon(coord_match.group(1), coord_match.group(2))
        if coords is not None:
            lat, lon = coords
            return _payload(lat, lon, reverse_geocode_label(lat, lon))

    zip_match = re.match(r"^\s*(\d{5})(?:-\d{4})?\s*$", cleaned)
    if zip_match:
        zip_code = zip_match.group(1)
        try:
            response = requests.get(f"https://api.zippopotam.us/us/{zip_code}", timeout=8)
            if response.ok:
                payload = response.json()
                places = payload.get("places") or []
                if places:
                    first = places[0]
                    coords = _valid_lat_lon(first.get("latitude"), first.get("longitude"))
                    if coords is not None:
                        lat, lon = coords
                        place_name = str(first.get("place name") or "").strip()
                        state = str(first.get("state abbreviation") or first.get("state") or "").strip()
                        label_parts = [part for part in [place_name, state] if part]
                        return _payload(lat, lon, ", ".join(label_parts) if label_parts else f"ZIP {zip_code}")
        except Exception:
            pass

    attempts = [cleaned]
    if "," in cleaned:
        attempts.append(re.sub(r"\s+", " ", cleaned.replace(",", " ")).strip())
    attempts = list(dict.fromkeys([candidate for candidate in attempts if candidate]))

    provider_factories: list[tuple[str, Any]] = [
        ("nominatim", lambda: Nominatim(user_agent="dso-explorer-prototype")),
        ("arcgis", lambda: ArcGIS(timeout=10)),
        ("photon", lambda: Photon(user_agent="dso-explorer-prototype")),
    ]
    for candidate in attempts:
        for _, provider_factory in provider_factories:
            try:
                geocoder = provider_factory()
                match = geocoder.geocode(candidate, exactly_one=True, timeout=10)
                if not match:
                    continue
                coords = _valid_lat_lon(getattr(match, "latitude", None), getattr(match, "longitude", None))
                if coords is None:
                    continue
                lat, lon = coords
                raw_label = str(getattr(match, "address", "") or "").split(",")[0].strip()
                return _payload(lat, lon, raw_label or candidate)
            except Exception:
                continue

    return None


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def reverse_geocode_label(lat: float, lon: float) -> str:
    _refresh_legacy_globals()
    geocoder = Nominatim(user_agent="dso-explorer-prototype")

    try:
        match = geocoder.reverse((lat, lon), exactly_one=True, timeout=10)
        if not match:
            return f"{lat:.4f}, {lon:.4f}"

        address = match.raw.get("address", {})
        locality = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("hamlet")
            or address.get("county")
        )
        region = address.get("state")

        parts = [part for part in [locality, region] if part]
        if parts:
            return ", ".join(parts)

        title = str(match.address).split(",")[0].strip()
        return title or f"{lat:.4f}, {lon:.4f}"
    except Exception:
        return f"{lat:.4f}, {lon:.4f}"

@st.cache_data(show_spinner=False, ttl=15 * 60)
def approximate_location_from_ip() -> dict[str, Any] | None:
    _refresh_legacy_globals()
    def _valid_lat_lon(lat_value: Any, lon_value: Any) -> tuple[float, float] | None:
        try:
            lat = float(lat_value)
            lon = float(lon_value)
        except (TypeError, ValueError):
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        return lat, lon

    def _location_payload(lat: float, lon: float, city: Any, region: Any, country: Any) -> dict[str, Any]:
        city_str = str(city or "").strip()
        region_str = str(region or "").strip()
        country_str = str(country or "").strip()
        label_parts = [part for part in [city_str, region_str, country_str] if part]
        label = ", ".join(label_parts) if label_parts else "IP-based estimate"
        return {
            "lat": lat,
            "lon": lon,
            "label": label,
            "source": "ip",
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }

    try:
        response = requests.get("https://ipapi.co/json/", timeout=8)
        if response.ok:
            payload = response.json()
            if not bool(payload.get("error")):
                coords = _valid_lat_lon(payload.get("latitude"), payload.get("longitude"))
                if coords is not None:
                    lat, lon = coords
                    return _location_payload(
                        lat,
                        lon,
                        payload.get("city"),
                        payload.get("region"),
                        payload.get("country_name") or payload.get("country"),
                    )
    except Exception:
        pass

    try:
        response = requests.get("https://ipwho.is/", timeout=8)
        if response.ok:
            payload = response.json()
            if bool(payload.get("success", True)):
                coords = _valid_lat_lon(payload.get("latitude"), payload.get("longitude"))
                if coords is not None:
                    lat, lon = coords
                    return _location_payload(lat, lon, payload.get("city"), payload.get("region"), payload.get("country"))
    except Exception:
        pass

    try:
        response = requests.get("https://ipinfo.io/json", timeout=8)
        if response.ok:
            payload = response.json()
            loc = str(payload.get("loc") or "").strip()
            if "," in loc:
                lat_raw, lon_raw = loc.split(",", 1)
                coords = _valid_lat_lon(lat_raw, lon_raw)
                if coords is not None:
                    lat, lon = coords
                    return _location_payload(lat, lon, payload.get("city"), payload.get("region"), payload.get("country"))
    except Exception:
        pass

    return None
