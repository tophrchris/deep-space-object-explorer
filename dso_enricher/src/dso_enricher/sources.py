from __future__ import annotations

import json
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pick_mapping_value(mapping: dict[str, Any], candidates: list[str]) -> Any:
    lower_lookup = {key.lower(): key for key in mapping}
    for candidate in candidates:
        key = lower_lookup.get(candidate.lower())
        if key is None:
            continue
        value = mapping[key]
        if value is None:
            continue
        text = str(value).strip()
        if text not in ("", "--"):
            return value
    return None


def _scan_nested_value(payload: Any, key_fragments: tuple[str, ...]) -> Any:
    stack = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                lowered = key.lower()
                if any(fragment in lowered for fragment in key_fragments):
                    if isinstance(value, (str, int, float)):
                        return value
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return None


@dataclass
class SourceResponse:
    data: dict[str, Any] = field(default_factory=dict)
    url: str | None = None
    ambiguous: bool = False
    candidates: list[str] = field(default_factory=list)
    error: str | None = None


class JsonCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}

    def get(self, source: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._data.get(source, {}).get(key)

    def set(self, source: str, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._data.setdefault(source, {})[key] = value

    def flush(self) -> None:
        with self._lock:
            payload = json.dumps(self._data, ensure_ascii=True, sort_keys=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(payload, encoding="utf-8")


class BaseClient:
    cache_key = "base"

    def __init__(
        self,
        timeout_s: float,
        requests_per_second: float,
        cache: JsonCache | None = None,
        retries: int = 2,
        retry_backoff_s: float = 0.4,
    ) -> None:
        self.timeout_s = timeout_s
        self.min_interval_s = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._last_call = 0.0
        self._rate_lock = threading.Lock()
        self._cache = cache
        self._retries = max(0, retries)
        self._retry_backoff_s = max(0.0, retry_backoff_s)
        self._headers = {
            "User-Agent": "dso-enricher/0.1 (+https://example.local)",
            "Accept": "*/*",
        }

    def _rate_limit(self) -> None:
        with self._rate_lock:
            if self.min_interval_s <= 0:
                return
            now = time.monotonic()
            wait = self.min_interval_s - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _cached(self, key: str) -> dict[str, Any] | None:
        if self._cache is None:
            return None
        return self._cache.get(self.cache_key, key)

    def _store_cache(self, key: str, value: dict[str, Any]) -> None:
        if self._cache is None:
            return
        self._cache.set(self.cache_key, key, value)

    def _fetch(self, url: str) -> tuple[bytes, str]:
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                self._rate_limit()
                request = urllib.request.Request(url, headers=self._headers)
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = response.read()
                    final_url = response.geturl()
                return body, final_url
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
            ) as exc:
                last_error = exc
                if attempt >= self._retries:
                    break
                sleep_s = self._retry_backoff_s * (2**attempt)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        if last_error is not None:
            raise last_error
        raise RuntimeError("request failed without an explicit error")

    def _fetch_json(self, url: str) -> tuple[Any, str]:
        body, final_url = self._fetch(url)
        payload = json.loads(body.decode("utf-8", errors="replace"))
        return payload, final_url

    def _parallel_lookup(
        self,
        identifiers: list[str],
        fn: Any,
        max_workers: int = 8,
    ) -> dict[str, SourceResponse]:
        unique: list[str] = []
        seen: set[str] = set()
        for identifier in identifiers:
            if not identifier:
                continue
            if identifier in seen:
                continue
            seen.add(identifier)
            unique.append(identifier)

        results: dict[str, SourceResponse] = {}
        if not unique:
            return results

        workers = max(1, min(max_workers, len(unique)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fn, ident): ident for ident in unique}
            for future in as_completed(futures):
                ident = futures[future]
                try:
                    results[ident] = future.result()
                except Exception as exc:  # noqa: BLE001
                    results[ident] = SourceResponse(error=str(exc))
        return results


class SesameClient(BaseClient):
    cache_key = "sesame"
    _endpoint_candidates = [
        "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV?",
        "https://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-oxp/SNV?",
    ]

    def resolve(self, identifier: str) -> SourceResponse:
        cached = self._cached(identifier)
        if cached is not None:
            return SourceResponse(**cached)

        encoded = urllib.parse.quote_plus(identifier)
        last_error = "unknown error"
        for base_url in self._endpoint_candidates:
            url = f"{base_url}{encoded}"
            try:
                body, final_url = self._fetch(url)
                parsed = self._parse_xml_response(body.decode("utf-8", errors="replace"))
                parsed.url = final_url
                self._store_cache(identifier, parsed.__dict__)
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        fallback = SourceResponse(error=f"Sesame lookup failed for '{identifier}': {last_error}")
        self._store_cache(identifier, fallback.__dict__)
        return fallback

    def _parse_xml_response(self, body: str) -> SourceResponse:
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            return SourceResponse(error=f"Sesame XML parse error: {exc}")

        oname: str | None = None
        aliases: list[str] = []
        ra_deg: float | None = None
        dec_deg: float | None = None

        for element in root.iter():
            tag = element.tag.split("}")[-1].lower()
            text = (element.text or "").strip()
            if not text:
                continue
            if tag == "oname" and oname is None:
                oname = text
            elif tag == "alias":
                aliases.append(text)
            elif tag == "jradeg" and ra_deg is None:
                ra_deg = _safe_float(text)
            elif tag == "jdedeg" and dec_deg is None:
                dec_deg = _safe_float(text)

        if oname is None and ra_deg is None and dec_deg is None and not aliases:
            return SourceResponse(error="Sesame returned no parseable match")

        return SourceResponse(
            data={
                "main_id": oname,
                "aliases": aliases,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
            }
        )

    def lookup_many(self, identifiers: list[str], max_workers: int = 8) -> dict[str, SourceResponse]:
        return self._parallel_lookup(identifiers, self.resolve, max_workers=max_workers)


class SimbadClient(BaseClient):
    cache_key = "simbad"
    _endpoints = [
        "https://simbad.cds.unistra.fr/simbad/sim-id",
        "https://simbad.u-strasbg.fr/simbad/sim-id",
    ]

    def lookup(self, identifier: str) -> SourceResponse:
        cached = self._cached(identifier)
        if cached is not None:
            return SourceResponse(**cached)

        last_error = "unknown error"
        last_url: str | None = None
        params = urllib.parse.urlencode({"Ident": identifier, "output.format": "VOTABLE"})
        for endpoint in self._endpoints:
            url = f"{endpoint}?{params}"
            last_url = url
            try:
                body, final_url = self._fetch(url)
                parsed = self._parse_votable(body)
                parsed.url = final_url
                self._store_cache(identifier, parsed.__dict__)
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                continue
        fallback = SourceResponse(
            error=f"SIMBAD lookup failed for '{identifier}': {last_error}",
            url=last_url,
        )
        self._store_cache(identifier, fallback.__dict__)
        return fallback

    def _parse_votable(self, content: bytes) -> SourceResponse:
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            return SourceResponse(error=f"SIMBAD VOTable parse failed: {exc}")

        fields: list[str] = []
        first_tr: ET.Element | None = None
        for element in root.iter():
            tag = element.tag.split("}")[-1].upper()
            if tag == "FIELD":
                fields.append(element.attrib.get("name") or element.attrib.get("ID") or "")
            elif tag == "TR" and first_tr is None:
                first_tr = element

        if first_tr is None or not fields:
            return SourceResponse(error="SIMBAD returned no VOTable rows")

        values: list[str] = []
        for td in first_tr:
            if td.tag.split("}")[-1].upper() == "TD":
                values.append((td.text or "").strip())
        if not values:
            return SourceResponse(error="SIMBAD returned empty first VOTable row")

        mapping: dict[str, Any] = {}
        for index, field_name in enumerate(fields):
            if not field_name:
                continue
            if index < len(values):
                mapping[field_name] = values[index]

        main_id = _safe_text(_pick_mapping_value(mapping, ["MAIN_ID", "main_id"]))
        object_type = _safe_text(_pick_mapping_value(mapping, ["OTYPE", "OTYPE_S", "OTYPE_V", "OTYPE_TXT"]))
        ra_deg = _safe_float(_pick_mapping_value(mapping, ["RA_d", "RA_DEG", "RA"]))
        dec_deg = _safe_float(_pick_mapping_value(mapping, ["DEC_d", "DEC_DEG", "DEC"]))
        cross_ids_raw = _safe_text(_pick_mapping_value(mapping, ["IDS", "IDLIST"]))
        cross_ids = [piece.strip() for piece in cross_ids_raw.split("|")] if cross_ids_raw else []

        return SourceResponse(
            data={
                "main_id": main_id,
                "object_type": object_type,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "coord_ref_bibcode": _safe_text(_pick_mapping_value(mapping, ["COO_BIBCODE"])),
                "coord_quality": _safe_text(_pick_mapping_value(mapping, ["COO_QUAL"])),
                "coord_waveband": _safe_text(_pick_mapping_value(mapping, ["COO_WAVELENGTH"])),
                "redshift": _safe_float(_pick_mapping_value(mapping, ["RVZ_REDSHIFT", "Z_VALUE"])),
                "radial_velocity_kms": _safe_float(_pick_mapping_value(mapping, ["RVZ_RADVEL", "RV_VALUE"])),
                "spectral_type": _safe_text(_pick_mapping_value(mapping, ["SP_TYPE"])),
                "ang_size_maj_arcmin": _safe_float(_pick_mapping_value(mapping, ["GALDIM_MAJAXIS", "DIM_MAJAXIS"])),
                "ang_size_min_arcmin": _safe_float(_pick_mapping_value(mapping, ["GALDIM_MINAXIS", "DIM_MINAXIS"])),
                "pa_deg": _safe_float(_pick_mapping_value(mapping, ["GALDIM_ANGLE", "DIM_ANGLE"])),
                "mag_v": _safe_float(_pick_mapping_value(mapping, ["FLUX_V", "V", "V_MAG"])),
                "cross_ids": cross_ids,
            }
        )

    def lookup_many(self, identifiers: list[str], max_workers: int = 8) -> dict[str, SourceResponse]:
        return self._parallel_lookup(identifiers, self.lookup, max_workers=max_workers)


class NedClient(BaseClient):
    cache_key = "ned"
    _lookup_endpoint = "https://ned.ipac.caltech.edu/srs/ObjectLookup"

    def lookup(self, identifier: str) -> SourceResponse:
        cached = self._cached(identifier)
        if cached is not None:
            return SourceResponse(**cached)

        params = urllib.parse.urlencode(
            {
                "objname": identifier,
                "extend": "no",
                "out_csys": "Equatorial",
                "out_equinox": "J2000.0",
            }
        )
        url = f"{self._lookup_endpoint}?{params}"
        result_url = f"https://ned.ipac.caltech.edu/byname?objname={urllib.parse.quote_plus(identifier)}"
        try:
            payload, _final_url = self._fetch_json(url)
            parsed = self._parse_ned_payload(payload)
            parsed.url = result_url
            self._store_cache(identifier, parsed.__dict__)
            return parsed
        except Exception as exc:  # noqa: BLE001
            fallback = SourceResponse(error=f"NED lookup failed for '{identifier}': {exc}", url=result_url)
            self._store_cache(identifier, fallback.__dict__)
            return fallback

    def _parse_ned_payload(self, payload: Any) -> SourceResponse:
        serialized = json.dumps(payload, ensure_ascii=True).lower()
        ambiguous = "ambiguous name" in serialized
        data: dict[str, Any] = {}
        candidates: list[str] = []

        pref_name = _scan_nested_value(payload, ("preferred", "objectname", "name"))
        redshift = _scan_nested_value(payload, ("redshift",))
        rv = _scan_nested_value(payload, ("radial", "velocity"))
        dist = _scan_nested_value(payload, ("distance",))
        dist_err = _scan_nested_value(payload, ("dist_err", "distance_err", "distanceerror", "err_distance"))
        dist_err_plus = _scan_nested_value(payload, ("dist_err_plus", "distance_err_plus", "err_plus"))
        dist_err_minus = _scan_nested_value(payload, ("dist_err_minus", "distance_err_minus", "err_minus"))
        morphology = _scan_nested_value(payload, ("morph",))
        if pref_name:
            data["main_id"] = _safe_text(pref_name)
        data["redshift"] = _safe_float(redshift)
        data["radial_velocity_kms"] = _safe_float(rv)
        data["dist_value"] = _safe_float(dist)
        dist_err_value = _safe_float(dist_err)
        data["dist_err_plus"] = _safe_float(dist_err_plus)
        data["dist_err_minus"] = _safe_float(dist_err_minus)
        if dist_err_value is not None:
            data["dist_err_plus"] = data["dist_err_plus"] or dist_err_value
            data["dist_err_minus"] = data["dist_err_minus"] or dist_err_value
        if data["dist_value"] is not None:
            data["dist_unit"] = "Mpc"
            data["dist_method"] = "ned_lookup"
        data["morphology"] = _safe_text(morphology)

        stack = [payload]
        while stack and len(candidates) < 20:
            current = stack.pop()
            if isinstance(current, dict):
                for key, value in current.items():
                    if "candidate" in key.lower() and isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                maybe_name = _scan_nested_value(item, ("name", "object"))
                                if maybe_name:
                                    candidates.append(str(maybe_name))
                            elif isinstance(item, (str, int)):
                                candidates.append(str(item))
                    stack.append(value)
            elif isinstance(current, list):
                stack.extend(current)

        return SourceResponse(data=data, ambiguous=ambiguous, candidates=candidates)

    def lookup_many(self, identifiers: list[str], max_workers: int = 8) -> dict[str, SourceResponse]:
        return self._parallel_lookup(identifiers, self.lookup, max_workers=max_workers)


class VizierClient(BaseClient):
    cache_key = "vizier"
    _json_endpoint = "https://vizier.cds.unistra.fr/viz-bin/asu-json"
    _tsv_endpoint = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"

    def lookup(self, identifier: str) -> SourceResponse:
        cached = self._cached(identifier)
        if cached is not None:
            return SourceResponse(**cached)

        params = urllib.parse.urlencode(
            {
                "-source": "VII/239/rc3",
                "-c": identifier,
                "-out.max": "1",
                "-out": "Type,logD25,logR25,BT,PA",
            }
        )
        json_url = f"{self._json_endpoint}?{params}"
        result_url = (
            "https://vizier.cds.unistra.fr/viz-bin/VizieR-4?"
            f"-source=VII/239/rc3&-c={urllib.parse.quote_plus(identifier)}"
        )
        last_error = "unknown error"
        try:
            payload, _final_url = self._fetch_json(json_url)
            parsed = self._parse_vizier_payload(payload)
            parsed.url = result_url
            self._store_cache(identifier, parsed.__dict__)
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

        try:
            tsv_url = f"{self._tsv_endpoint}?{params}"
            body, _final_url = self._fetch(tsv_url)
            parsed = self._parse_vizier_tsv(body.decode("utf-8", errors="replace"))
            parsed.url = result_url
            self._store_cache(identifier, parsed.__dict__)
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_error = f"{last_error}; {exc}"

        fallback = SourceResponse(error=f"VizieR lookup failed for '{identifier}': {last_error}", url=result_url)
        self._store_cache(identifier, fallback.__dict__)
        return fallback

    def _parse_vizier_payload(self, payload: Any) -> SourceResponse:
        morphology = _safe_text(_scan_nested_value(payload, ("morph", "hubble", "type")))
        mag_v = _safe_float(_scan_nested_value(payload, ("vmag", "v_mag", "vt")))
        log_d25 = _safe_float(_scan_nested_value(payload, ("logd25",)))
        log_r25 = _safe_float(_scan_nested_value(payload, ("logr25",)))
        pa = _safe_float(_scan_nested_value(payload, ("pa", "angle")))

        maj = None
        min_axis = None
        if log_d25 is not None:
            maj = (10 ** log_d25) / 10.0
            if log_r25 is not None:
                min_axis = maj / (10 ** log_r25)

        return SourceResponse(
            data={
                "morphology": morphology,
                "mag_v": mag_v,
                "ang_size_maj_arcmin": maj,
                "ang_size_min_arcmin": min_axis,
                "pa_deg": pa,
            }
        )

    def _parse_vizier_tsv(self, body: str) -> SourceResponse:
        rows: list[list[str]] = []
        for raw_line in body.splitlines():
            line = raw_line.strip("\n")
            if not line:
                continue
            if line.startswith("#"):
                continue
            rows.append(line.split("\t"))
        if len(rows) < 2:
            return SourceResponse(error="VizieR TSV returned no data rows")

        header = [cell.strip() for cell in rows[0]]
        values = rows[1]
        mapping: dict[str, str] = {}
        for idx, key in enumerate(header):
            if idx < len(values):
                mapping[key] = values[idx].strip()

        log_d25 = _safe_float(_pick_mapping_value(mapping, ["logD25", "logd25"]))
        log_r25 = _safe_float(_pick_mapping_value(mapping, ["logR25", "logr25"]))
        mag_v = _safe_float(_pick_mapping_value(mapping, ["BT", "Vmag", "VT"]))
        pa = _safe_float(_pick_mapping_value(mapping, ["PA", "Angle"]))

        maj = None
        min_axis = None
        if log_d25 is not None:
            maj = (10 ** log_d25) / 10.0
            if log_r25 is not None:
                min_axis = maj / (10 ** log_r25)

        return SourceResponse(
            data={
                "morphology": _safe_text(_pick_mapping_value(mapping, ["Type", "Morph"])),
                "mag_v": mag_v,
                "ang_size_maj_arcmin": maj,
                "ang_size_min_arcmin": min_axis,
                "pa_deg": pa,
            }
        )

    def lookup_many(self, identifiers: list[str], max_workers: int = 8) -> dict[str, SourceResponse]:
        return self._parallel_lookup(identifiers, self.lookup, max_workers=max_workers)
