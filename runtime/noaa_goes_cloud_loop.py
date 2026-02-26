from __future__ import annotations

import math
import re
from typing import Any

import requests
import streamlit as st

_NOAA_CONUS_PAGE_URL = "https://www.star.nesdis.noaa.gov/goes/conus.php"
_NOAA_SECTOR_PAGE_URL = "https://www.star.nesdis.noaa.gov/goes/sector_band.php"
_NOAA_TIMEOUT_SECONDS = 20.0
_NOAA_IMAGE_SIZE = "625x375"
_GOES_EARTH_EQ_RADIUS_M = 6378137.0
_GOES_EARTH_POLAR_RADIUS_M = 6356752.31414
_GOES_PERSPECTIVE_HEIGHT_M = 35786023.0
_GOES_PLATFORM_HEIGHT_M = _GOES_EARTH_EQ_RADIUS_M + _GOES_PERSPECTIVE_HEIGHT_M
_NOAA_SECTOR_APPROX_BOUNDS: dict[str, dict[str, float]] = {
    # Approximate sector extents for broad "good enough" site pin placement.
    # NOAA sector images use a geostationary projection, so this is intentionally
    # a coarse equirectangular approximation for UI context, not measurement.
    # EUS includes a large Atlantic slice; using a farther-east bound prevents
    # East Coast sites from plotting too far right.
    "eus": {"west": -108.0, "east": -38.0, "south": 20.0, "north": 55.0},
    "wus": {"west": -138.0, "east": -100.0, "south": 20.0, "north": 56.0},
    "ak": {"west": -171.0, "east": -130.0, "south": 49.0, "north": 73.0},
    "hi": {"west": -162.0, "east": -153.0, "south": 17.5, "north": 23.5},
    "pr": {"west": -68.8, "east": -63.2, "south": 17.2, "north": 19.8},
}


def _normalize_lat_lon(lat: Any, lon: Any) -> tuple[float, float] | None:
    try:
        lat_value = float(lat)
        lon_value = float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= lat_value <= 90.0 and -180.0 <= lon_value <= 180.0):
        return None
    return (lat_value, lon_value)


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _goes_subsatellite_longitude_deg(sat: str) -> float:
    sat_token = str(sat).strip().upper()
    if sat_token == "G18":
        return -137.2
    return -75.2


def _goes_fixed_grid_xy(lat: float, lon: float, *, sat: str) -> tuple[float, float] | None:
    lat_rad = math.radians(float(lat))
    lon_rad = math.radians(float(lon))
    lon0_rad = math.radians(_goes_subsatellite_longitude_deg(sat))
    req = _GOES_EARTH_EQ_RADIUS_M
    rpol = _GOES_EARTH_POLAR_RADIUS_M
    H = _GOES_PLATFORM_HEIGHT_M

    e2 = (req * req - rpol * rpol) / (req * req)
    phi_c = math.atan((rpol * rpol) / (req * req) * math.tan(lat_rad))
    cos_phi_c = math.cos(phi_c)
    sin_phi_c = math.sin(phi_c)
    rc = rpol / math.sqrt(1.0 - (e2 * cos_phi_c * cos_phi_c))
    dlon = lon_rad - lon0_rad

    sx = H - rc * cos_phi_c * math.cos(dlon)
    sy = -rc * cos_phi_c * math.sin(dlon)
    sz = rc * sin_phi_c

    visibility_lhs = H * (H - sx)
    visibility_rhs = (sy * sy) + ((req * req) / (rpol * rpol) * (sz * sz))
    if visibility_lhs < visibility_rhs:
        return None

    denom = math.sqrt(sx * sx + sy * sy + sz * sz)
    if not denom:
        return None
    x_arg = -sy / denom
    x_arg = max(-1.0, min(1.0, x_arg))
    x = math.asin(x_arg)
    y = math.atan2(sz, sx)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


def _sector_projected_extent_from_bbox(sector: str, *, sat: str) -> tuple[float, float, float, float] | None:
    sector_token = str(sector).strip().lower()
    bounds = _NOAA_SECTOR_APPROX_BOUNDS.get(sector_token)
    if not isinstance(bounds, dict):
        return None
    west = float(bounds.get("west", 0.0))
    east = float(bounds.get("east", 0.0))
    south = float(bounds.get("south", 0.0))
    north = float(bounds.get("north", 0.0))
    if not (east > west and north > south):
        return None

    xs: list[float] = []
    ys: list[float] = []
    steps = 16
    for i in range(steps + 1):
        lat_value = south + (north - south) * (i / steps)
        for j in range(steps + 1):
            lon_value = west + (east - west) * (j / steps)
            projected = _goes_fixed_grid_xy(lat_value, lon_value, sat=sat)
            if projected is None:
                continue
            x, y = projected
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        return None
    return (min(xs), max(xs), min(ys), max(ys))


def _approximate_sector_site_pin(lat: float, lon: float, sector: str, *, sat: str) -> dict[str, Any] | None:
    sector_token = str(sector).strip().lower()
    sat_token = str(sat).strip().upper()
    projected = _goes_fixed_grid_xy(lat, lon, sat=sat_token)
    if projected is None:
        return None
    extent = _sector_projected_extent_from_bbox(
        sector_token,
        sat=sat_token,
    )
    if extent is None:
        return None
    x, y = projected
    x_min, x_max, y_min, y_max = extent
    if not (x_max > x_min and y_max > y_min):
        return None

    x_raw = (x - x_min) / (x_max - x_min)
    y_raw = (y_max - y) / (y_max - y_min)

    # Hide the pin if the site is clearly outside the chosen sector bbox; keep
    # a small tolerance because sector selection itself is broad.
    if x_raw < -0.20 or x_raw > 1.20 or y_raw < -0.20 or y_raw > 1.20:
        return None

    return {
        "kind": "site_pin_approx",
        "method": "sector_bbox_geos_projected",
        "sector": sector_token,
        "x_frac": round(_clamp01(x_raw), 6),
        "y_frac": round(_clamp01(y_raw), 6),
        "label": "Site (approx)",
    }


def _pick_goes_conus_satellite_for_site(lat: float, lon: float) -> tuple[str, str]:
    # Pick the broader NOAA CONUS/PACUS panel from the closer GOES geostationary longitude.
    east_ref_lon = -75.0
    west_ref_lon = -137.0
    use_east = abs(lon - east_ref_lon) <= abs(lon - west_ref_lon)
    if use_east:
        return ("G19", "GOES-East CONUS")
    return ("G18", "GOES-West PACUS")


def _pick_goes_sector_panel_for_site(lat: float, lon: float) -> tuple[str, str, str]:
    # Broad sector-first coverage for zoomed-in observed cloud context while keeping wide regional view.
    if lat >= 50.0 and lon <= -130.0:
        return ("G18", "ak", "GOES-West Alaska")
    if 18.0 <= lat <= 25.5 and -162.5 <= lon <= -152.0:
        return ("G18", "hi", "GOES-West Hawaii")
    if 15.0 <= lat <= 21.5 and -70.5 <= lon <= -63.0:
        return ("G19", "pr", "GOES-East Puerto Rico")
    if lon <= -103.0:
        return ("G18", "wus", "GOES-West Western U.S.")
    return ("G19", "eus", "GOES-East Eastern U.S.")


def _extract_conus_product_gif_url(html: str, *, sat: str, product: str) -> str:
    sat_num = str(sat).strip().upper().lstrip("G")
    product_token = re.escape(str(product).strip())
    size_token = re.escape(_NOAA_IMAGE_SIZE)
    pattern = (
        r"https://cdn\.star\.nesdis\.noaa\.gov/GOES"
        + re.escape(sat_num)
        + r"/ABI/CONUS/"
        + product_token
        + r"/[^\"' ]+-"
        + size_token
        + r"\.gif"
    )
    match = re.search(pattern, html)
    return match.group(0) if match else ""


def _extract_sector_animation_frames(html: str, *, sat: str, sector: str, product: str) -> list[str]:
    block_match = re.search(r"animationImages\s*=\s*\[(.*?)\];", html, flags=re.DOTALL)
    if not block_match:
        return []
    block = block_match.group(1)
    sat_num = re.escape(str(sat).strip().upper().lstrip("G"))
    sector_token = re.escape(str(sector).strip())
    product_token = re.escape(str(product).strip())
    pattern = (
        r"https://cdn\.star\.nesdis\.noaa\.gov/GOES"
        + sat_num
        + r"/ABI/SECTOR/"
        + sector_token
        + r"/"
        + product_token
        + r"/[^\"' ]+-1000x1000\.jpg"
    )
    frame_urls: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(pattern, block):
        url = match.group(0)
        if url in seen:
            continue
        frame_urls.append(url)
        seen.add(url)
    return frame_urls


def _extract_sector_og_image(html: str, *, sat: str, sector: str, product: str) -> str:
    sat_num = re.escape(str(sat).strip().upper().lstrip("G"))
    sector_token = re.escape(str(sector).strip())
    product_token = re.escape(str(product).strip())
    pattern = (
        r'<meta\s+property=\"og:image\"\s+content=\"('
        r"https://cdn\.star\.nesdis\.noaa\.gov/GOES"
        + sat_num
        + r"/ABI/SECTOR/"
        + sector_token
        + r"/"
        + product_token
        + r"/2000x2000\.jpg"
        r')\"'
    )
    match = re.search(pattern, html)
    return match.group(1) if match else ""


@st.cache_data(show_spinner=False, ttl=5 * 60)
def _fetch_noaa_conus_cloud_loop(sat: str) -> dict[str, str]:
    sat_token = str(sat).strip().upper()
    if sat_token not in {"G18", "G19"}:
        sat_token = "G19"
    response = requests.get(
        _NOAA_CONUS_PAGE_URL,
        params={"sat": sat_token},
        timeout=_NOAA_TIMEOUT_SECONDS,
        headers={"User-Agent": "DeepSpaceObjectExplorer/1.0"},
    )
    response.raise_for_status()
    html = response.text

    gif_url = _extract_conus_product_gif_url(html, sat=sat_token, product="DayNightCloudMicroCombo")
    product_name = "Day/Night Cloud Microphysics"
    if not gif_url:
        gif_url = _extract_conus_product_gif_url(html, sat=sat_token, product="13")
        product_name = "Band 13 (IR Clean Longwave)"

    return {
        "sat": sat_token,
        "gif_url": gif_url,
        "product_name": product_name,
        "source_page_url": f"{_NOAA_CONUS_PAGE_URL}?sat={sat_token}",
    }


@st.cache_data(show_spinner=False, ttl=5 * 60)
def _fetch_noaa_sector_cloud_panel(sat: str, sector: str, *, length: int = 12) -> dict[str, Any]:
    sat_token = str(sat).strip().upper()
    sector_token = str(sector).strip().lower()
    if sat_token not in {"G18", "G19"}:
        sat_token = "G19"
    if not sector_token:
        return {}
    params_base = {
        "sat": sat_token,
        "sector": sector_token,
        "length": max(1, int(length)),
    }

    for band, product_name in (
        ("DayNightCloudMicroCombo", "Day/Night Cloud Microphysics"),
        ("13", "Band 13 (IR Clean Longwave)"),
    ):
        response = requests.get(
            _NOAA_SECTOR_PAGE_URL,
            params={**params_base, "band": band},
            timeout=_NOAA_TIMEOUT_SECONDS,
            headers={"User-Agent": "DeepSpaceObjectExplorer/1.0"},
        )
        response.raise_for_status()
        html = response.text
        frame_urls = _extract_sector_animation_frames(html, sat=sat_token, sector=sector_token, product=band)
        image_url = ""
        if frame_urls:
            image_url = frame_urls[-1]
        if not image_url:
            image_url = _extract_sector_og_image(html, sat=sat_token, sector=sector_token, product=band)
        if frame_urls or image_url:
            return {
                "sat": sat_token,
                "sector": sector_token,
                "band": band,
                "product_name": product_name,
                "frame_urls": frame_urls,
                "image_url": image_url,
                "source_page_url": (
                    f"{_NOAA_SECTOR_PAGE_URL}?sat={sat_token}&sector={sector_token}&band={band}&length={int(params_base['length'])}"
                ),
            }

    return {
        "sat": sat_token,
        "sector": sector_token,
        "frame_urls": [],
        "image_url": "",
        "source_page_url": (
            f"{_NOAA_SECTOR_PAGE_URL}?sat={sat_token}&sector={sector_token}&band=DayNightCloudMicroCombo&length={int(params_base['length'])}"
        ),
    }


def resolve_site_cloud_loop(lat: Any, lon: Any) -> dict[str, Any]:
    normalized = _normalize_lat_lon(lat, lon)
    if normalized is None:
        return {"available": False, "reason": "invalid_lat_lon"}

    lat_value, lon_value = normalized
    sat, sector_code, panel_label = _pick_goes_sector_panel_for_site(lat_value, lon_value)
    try:
        sector_payload = _fetch_noaa_sector_cloud_panel(sat, sector_code, length=12)
    except Exception as exc:
        sector_payload = {
            "frame_urls": [],
            "image_url": "",
            "source_page_url": "",
            "error": str(exc).strip(),
        }

    frame_urls = sector_payload.get("frame_urls", [])
    if not isinstance(frame_urls, list):
        frame_urls = []
    frame_urls = [str(url).strip() for url in frame_urls if str(url).strip()]
    latest_sector_image = str(sector_payload.get("image_url", "")).strip()

    if frame_urls or latest_sector_image:
        site_overlay_pin = _approximate_sector_site_pin(lat_value, lon_value, sector_code, sat=sat)
        return {
            "available": True,
            "image_url": latest_sector_image or (frame_urls[-1] if frame_urls else ""),
            "frame_urls": frame_urls,
            "panel_label": panel_label,
            "sat": sat,
            "sector": sector_code,
            "product_name": str(sector_payload.get("product_name", "")).strip() or "Cloud Loop",
            "source_page_url": str(sector_payload.get("source_page_url", "")).strip(),
            "site_overlay_pin": site_overlay_pin,
            "site_lat": lat_value,
            "site_lon": lon_value,
            "kind": "observed_satellite_sector_loop" if frame_urls else "observed_satellite_sector_image",
        }

    # Fallback to broad CONUS/PACUS animated GIF if sector lookup fails.
    fallback_sat, fallback_panel_label = _pick_goes_conus_satellite_for_site(lat_value, lon_value)
    try:
        payload = _fetch_noaa_conus_cloud_loop(fallback_sat)
    except Exception as exc:
        return {
            "available": False,
            "reason": "fetch_failed",
            "panel_label": panel_label,
            "sat": sat,
            "error": str(exc).strip(),
        }
    gif_url = str(payload.get("gif_url", "")).strip()
    if not gif_url:
        return {
            "available": False,
            "reason": "gif_not_found",
            "panel_label": fallback_panel_label,
            "sat": fallback_sat,
            "source_page_url": str(payload.get("source_page_url", "")).strip(),
        }

    return {
        "available": True,
        "image_url": gif_url,
        "frame_urls": [],
        "panel_label": fallback_panel_label,
        "sat": fallback_sat,
        "product_name": str(payload.get("product_name", "")).strip() or "Cloud Loop",
        "source_page_url": str(payload.get("source_page_url", "")).strip(),
        "site_overlay_pin": None,
        "site_lat": lat_value,
        "site_lon": lon_value,
        "image_size": _NOAA_IMAGE_SIZE,
        "kind": "observed_satellite_loop",
    }
