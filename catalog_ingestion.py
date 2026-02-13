from __future__ import annotations

import io
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import astropy.units as u
import pandas as pd
import requests
from astropy.coordinates import SkyCoord

REQUIRED_COLUMNS = [
    "primary_id",
    "catalog",
    "common_name",
    "object_type",
    "ra_deg",
    "dec_deg",
]

OPTIONAL_COLUMNS = [
    "constellation",
    "aliases",
    "image_url",
    "image_attribution_url",
    "license_label",
]

ENRICHED_OPTIONAL_COLUMNS = [
    "description",
    "info_url",
    "dist_value",
    "dist_unit",
    "redshift",
    "morphology",
    "emission_lines",
]

CATALOG_CACHE_MAX_AGE_HOURS = 24
INGESTION_VERSION = 5
OPENNGC_SOURCE_URL = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/NGC.csv"
OPENNGC_TIMEOUT_SECONDS = 45
SIMBAD_TAP_SYNC_ENDPOINTS = [
    "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
    "https://simbad.u-strasbg.fr/simbad/sim-tap/sync",
]
SIMBAD_TIMEOUT_SECONDS = 90
SIMBAD_NAMED_OBJECTS_QUERY = """
    SELECT
      b.oid,
      b.main_id,
      b.ra,
      b.dec,
      b.otype,
      i.id AS name_identifier
    FROM basic AS b
    JOIN ident AS i
      ON b.oid = i.oidref
    WHERE
      i.id LIKE 'NAME %'
      AND b.oid IN (
        SELECT DISTINCT b2.oid
        FROM basic AS b2
        JOIN ident AS i2
          ON b2.oid = i2.oidref
      )
"""

OPENNGC_TYPE_MAP = {
    "*": "Star",
    "**": "Double Star",
    "*Ass": "Asterism",
    "Cl+N": "Cluster + Nebula",
    "Dup": "Duplicate",
    "EmN": "Emission Nebula",
    "G": "Galaxy",
    "GCl": "Globular Cluster",
    "GGroup": "Galaxy Group",
    "GPair": "Galaxy Pair",
    "GTrpl": "Galaxy Triplet",
    "HII": "HII Region",
    "Neb": "Nebula",
    "NonEx": "Nonexistent",
    "Nova": "Nova",
    "OCl": "Open Cluster",
    "Other": "Other",
    "PN": "Planetary Nebula",
    "RfN": "Reflection Nebula",
    "SNR": "Supernova Remnant",
}


def _merge_alias_values(existing: str, additions: list[str]) -> str:
    merged: list[str] = []
    seen: set[str] = set()

    for value in str(existing).split(";"):
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            merged.append(cleaned)

    for value in additions:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            merged.append(cleaned)

    return ";".join(merged)


def _error_summary(error: Exception) -> str:
    message = str(error).strip()
    if message:
        return f"{error.__class__.__name__}: {message}"
    return error.__class__.__name__


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Catalog source is missing required columns: {missing_list}")

    normalized = frame.copy()

    for column in OPTIONAL_COLUMNS + ENRICHED_OPTIONAL_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""

    normalized = normalized[REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ENRICHED_OPTIONAL_COLUMNS]

    normalized["primary_id"] = normalized["primary_id"].fillna("").astype(str).str.strip()
    normalized["catalog"] = normalized["catalog"].fillna("").astype(str).str.upper().str.strip()
    normalized["common_name"] = normalized["common_name"].fillna("").astype(str).str.strip()
    normalized["object_type"] = normalized["object_type"].fillna("").astype(str).str.strip()
    normalized["constellation"] = normalized["constellation"].fillna("").astype(str).str.strip()
    normalized["aliases"] = normalized["aliases"].fillna("").astype(str).str.strip()
    normalized["description"] = normalized["description"].fillna("").astype(str).str.strip()
    normalized["info_url"] = normalized["info_url"].fillna("").astype(str).str.strip()
    normalized["dist_unit"] = normalized["dist_unit"].fillna("").astype(str).str.strip()
    normalized["morphology"] = normalized["morphology"].fillna("").astype(str).str.strip()
    normalized["emission_lines"] = normalized["emission_lines"].fillna("").astype(str).str.strip()

    normalized["ra_deg"] = pd.to_numeric(normalized["ra_deg"], errors="coerce")
    normalized["dec_deg"] = pd.to_numeric(normalized["dec_deg"], errors="coerce")
    normalized["dist_value"] = pd.to_numeric(normalized["dist_value"], errors="coerce")
    normalized["redshift"] = pd.to_numeric(normalized["redshift"], errors="coerce")

    normalized = normalized.dropna(subset=["ra_deg", "dec_deg"])
    normalized = normalized[normalized["primary_id"] != ""]
    normalized = normalized.drop_duplicates(subset=["primary_id"], keep="first")
    normalized = normalized.sort_values(by=["catalog", "primary_id"], ascending=[True, True]).reset_index(drop=True)

    return normalized


def _cache_is_fresh(cache_path: Path, max_age_hours: int) -> bool:
    if not cache_path.exists():
        return False

    modified_at = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(timezone.utc) - modified_at
    return age <= timedelta(hours=max_age_hours)


def _cache_requires_rebuild(metadata: dict[str, Any]) -> bool:
    version_raw = metadata.get("ingestion_version", 0)
    row_count_raw = metadata.get("row_count", 0)

    try:
        version = int(version_raw)
    except (TypeError, ValueError):
        version = 0

    try:
        row_count = int(row_count_raw)
    except (TypeError, ValueError):
        row_count = 0

    if version < INGESTION_VERSION:
        return True

    if row_count < 1000:
        return True

    return False


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_metadata(metadata_path: Path, payload: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _to_catalog_number(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    match = re.search(r"\d+", text)
    if not match:
        return ""

    return str(int(match.group(0)))


def _parse_name_designation(name_raw: str) -> tuple[str, str, str] | None:
    compact = " ".join(str(name_raw).strip().split())
    if not compact:
        return None

    match = re.match(r"^(NGC|IC)\s*0*([0-9]+)(.*)$", compact, flags=re.IGNORECASE)
    if not match:
        return None

    catalog = match.group(1).upper()
    number = str(int(match.group(2)))
    suffix = str(match.group(3)).strip()

    primary_id = f"{catalog} {number}" if not suffix else f"{catalog} {number} {suffix}"
    return primary_id, catalog, number


def _normalize_primary_id_for_app(primary_id_raw: str) -> str:
    compact = " ".join(str(primary_id_raw).strip().split())
    if not compact:
        return ""

    messier_match = re.match(r"^M\s*0*([0-9]+)$", compact, flags=re.IGNORECASE)
    if messier_match:
        return f"M{int(messier_match.group(1))}"

    ngc_match = re.match(r"^NGC\s*0*([0-9]+)(.*)$", compact, flags=re.IGNORECASE)
    if ngc_match:
        number = str(int(ngc_match.group(1)))
        suffix = str(ngc_match.group(2)).strip()
        return f"NGC {number}" if not suffix else f"NGC {number} {suffix}"

    ic_match = re.match(r"^IC\s*0*([0-9]+)(.*)$", compact, flags=re.IGNORECASE)
    if ic_match:
        number = str(int(ic_match.group(1)))
        suffix = str(ic_match.group(2)).strip()
        return f"IC {number}" if not suffix else f"IC {number} {suffix}"

    sh2_match = re.match(r"^(SH2|SH2-|SH\s*2-?)\s*0*([0-9]+)(.*)$", compact, flags=re.IGNORECASE)
    if sh2_match:
        number = str(int(sh2_match.group(2)))
        suffix = str(sh2_match.group(3)).strip()
        return f"Sh2-{number}" if not suffix else f"Sh2-{number}{suffix}"

    return compact


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _parse_common_names(raw_common_names: str) -> list[str]:
    return [name.strip() for name in str(raw_common_names).split(",") if name and name.strip()]


def _build_aliases(
    *,
    primary_id: str,
    name_designation: str,
    m_number: str,
    ngc_number: str,
    ic_number: str,
    common_names: list[str],
    identifiers_raw: str,
) -> str:
    candidates: list[str] = []

    if name_designation:
        candidates.append(name_designation)

    if m_number:
        candidates.extend([f"M{m_number}", f"Messier {m_number}"])
    if ngc_number:
        candidates.append(f"NGC {ngc_number}")
    if ic_number:
        candidates.append(f"IC {ic_number}")

    candidates.extend(common_names[:5])

    if identifiers_raw and identifiers_raw.lower() != "nan":
        identifiers = [token.strip() for token in str(identifiers_raw).split(",") if token and token.strip()]
        candidates.extend(identifiers[:8])

    aliases = [item for item in _dedupe_preserve_order(candidates) if item and item != primary_id]
    return ";".join(aliases)


def _parse_radec(ra_raw: str, dec_raw: str) -> tuple[float, float] | None:
    ra_text = str(ra_raw).strip()
    dec_text = str(dec_raw).strip()
    if not ra_text or not dec_text:
        return None

    try:
        coords = SkyCoord(ra=ra_text, dec=dec_text, unit=(u.hourangle, u.deg), frame="icrs")
        return float(coords.ra.deg), float(coords.dec.deg)
    except Exception:
        return None


def _query_simbad_tap(query: str) -> pd.DataFrame:
    payload = {
        "request": "doQuery",
        "lang": "adql",
        "format": "csv",
        "query": query,
    }

    last_error: Exception | None = None
    for endpoint in SIMBAD_TAP_SYNC_ENDPOINTS:
        try:
            response = requests.post(endpoint, data=payload, timeout=SIMBAD_TIMEOUT_SECONDS)
            response.raise_for_status()

            content_type = str(response.headers.get("content-type", "")).lower()
            body = response.text
            if "text/xml" in content_type or body.lstrip().startswith("<?xml"):
                raise ValueError("SIMBAD TAP returned XML instead of CSV")

            frame = pd.read_csv(io.StringIO(body))
            frame.columns = [str(column).strip().lower() for column in frame.columns]
            return frame.fillna("")
        except Exception as error:
            last_error = error
            continue

    if last_error is None:
        raise RuntimeError("SIMBAD TAP query failed without explicit error")
    raise RuntimeError(f"SIMBAD TAP query failed: {_error_summary(last_error)}") from last_error


def _canonical_primary_id_from_simbad_identifier(raw_identifier: str) -> str | None:
    compact = " ".join(str(raw_identifier).strip().split()).upper()
    if not compact:
        return None

    if compact.startswith("NAME "):
        compact = compact[5:].strip()
        if not compact:
            return None

    messier_match = re.match(r"^M\s*0*([0-9]+)$", compact)
    if messier_match:
        return f"M{int(messier_match.group(1))}"

    ngc_match = re.match(r"^NGC\s*0*([0-9]+)\s*([A-Z]*)$", compact)
    if ngc_match:
        ngc_number = str(int(ngc_match.group(1)))
        suffix = str(ngc_match.group(2)).strip()
        if suffix:
            return f"NGC {ngc_number} {suffix}"
        return f"NGC {ngc_number}"

    sh2_match = re.match(r"^SH\.?\s*2[-\s]*0*([0-9]+)\s*([A-Z]*)$", compact)
    if sh2_match:
        sh2_number = str(int(sh2_match.group(1)))
        suffix = str(sh2_match.group(2)).strip()
        if suffix:
            return f"Sh2-{sh2_number}{suffix}"
        return f"Sh2-{sh2_number}"

    return None


def _strip_name_prefix(raw_identifier: str) -> str:
    compact = " ".join(str(raw_identifier).strip().split())
    if compact.upper().startswith("NAME "):
        return compact[5:].strip()
    return compact


def _catalog_from_primary_id(primary_id: str) -> str:
    cleaned = str(primary_id).strip()
    if re.fullmatch(r"M\s*[0-9]+", cleaned, flags=re.IGNORECASE):
        return "M"
    if cleaned.startswith("NGC "):
        return "NGC"
    if cleaned.startswith("IC "):
        return "IC"
    if cleaned.startswith("Sh2-"):
        return "SH2"
    return "SIMBAD"


def _primary_id_rank(primary_id: str) -> tuple[int, int]:
    order = {"M": 0, "NGC": 1, "IC": 2, "SH2": 3, "SIMBAD": 4}
    catalog = _catalog_from_primary_id(primary_id)
    return order.get(catalog, 5), len(str(primary_id))


def _parse_json_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]

    text = str(raw or "").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
    except json.JSONDecodeError:
        pass

    if ";" in text:
        return [item.strip() for item in text.split(";") if item.strip()]
    return [text]


def _catalog_from_enriched_family(id_family_guess: str, primary_id: str) -> str:
    family = str(id_family_guess or "").strip().lower()
    if family == "messier":
        return "M"
    if family == "ngc":
        return "NGC"
    if family == "ic":
        return "IC"
    if family == "caldwell":
        return "C"
    if family == "survey":
        if str(primary_id).upper().startswith("SH2"):
            return "SH2"
        return "SURVEY"
    if family == "common_name":
        return _catalog_from_primary_id(primary_id)
    return _catalog_from_primary_id(primary_id)


def _prefer_primary_id(current: str, candidate: str) -> str:
    current_cleaned = str(current).strip()
    candidate_cleaned = str(candidate).strip()
    if not candidate_cleaned:
        return current_cleaned
    if not current_cleaned:
        return candidate_cleaned
    if _primary_id_rank(candidate_cleaned) < _primary_id_rank(current_cleaned):
        return candidate_cleaned
    return current_cleaned


def fetch_simbad_named_objects() -> pd.DataFrame:
    source = _query_simbad_tap(SIMBAD_NAMED_OBJECTS_QUERY)
    if source.empty:
        raise ValueError("SIMBAD named-object query returned zero rows")

    required = {"oid", "main_id", "ra", "dec", "otype", "name_identifier"}
    missing_columns = required - set(source.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"SIMBAD named-object query missing columns: {missing_list}")

    return source


def ingest_all_from_simbad_named_objects(named_objects: pd.DataFrame) -> pd.DataFrame:
    if named_objects.empty:
        raise ValueError("SIMBAD named-object source is empty for full ingest")

    records_by_oid: dict[str, dict[str, Any]] = {}

    for _, row in named_objects.iterrows():
        oid = str(row.get("oid", "")).strip()
        if not oid:
            continue

        if oid not in records_by_oid:
            records_by_oid[oid] = {
                "_oid": oid,
                "primary_id": f"SIMBAD OID {oid}",
                "catalog": "SIMBAD",
                "common_name": "",
                "object_type": str(row.get("otype", "")).strip(),
                "ra_deg": row.get("ra", ""),
                "dec_deg": row.get("dec", ""),
                "constellation": "",
                "aliases": "",
                "image_url": "",
                "image_attribution_url": "",
                "license_label": "",
            }

        entry = records_by_oid[oid]
        name_identifier = str(row.get("name_identifier", "")).strip()
        main_id = str(row.get("main_id", "")).strip()
        name_value = _strip_name_prefix(name_identifier)
        main_value = " ".join(main_id.split()).strip()

        primary_candidates: list[str] = []
        for raw_candidate in [name_identifier, main_id, name_value, main_value]:
            if not raw_candidate:
                continue
            canonical = _canonical_primary_id_from_simbad_identifier(raw_candidate)
            primary_candidates.append(canonical if canonical else raw_candidate)

        preferred_primary = str(entry.get("primary_id", "")).strip()
        for candidate in primary_candidates:
            preferred_primary = _prefer_primary_id(preferred_primary, candidate)

        entry["primary_id"] = preferred_primary if preferred_primary else f"SIMBAD OID {oid}"
        entry["catalog"] = _catalog_from_primary_id(entry["primary_id"])
        entry["aliases"] = _merge_alias_values(
            str(entry.get("aliases", "")),
            [name_identifier, name_value, main_id, main_value],
        )

        if not str(entry.get("common_name", "")).strip():
            for candidate in [name_value, main_value]:
                if not candidate:
                    continue
                if _canonical_primary_id_from_simbad_identifier(candidate):
                    continue
                if candidate != entry["primary_id"]:
                    entry["common_name"] = candidate
                    break

        if not str(entry.get("object_type", "")).strip():
            entry["object_type"] = str(row.get("otype", "")).strip()

    if not records_by_oid:
        raise ValueError("SIMBAD named-object query could not map any rows")

    records = list(records_by_oid.values())
    used_primary_ids: set[str] = set()
    for record in records:
        oid = str(record.get("_oid", "")).strip()
        primary_id = str(record.get("primary_id", "")).strip() or f"SIMBAD OID {oid}"
        if primary_id in used_primary_ids:
            primary_id = f"SIMBAD OID {oid}"
            record["catalog"] = "SIMBAD"
        used_primary_ids.add(primary_id)
        record["primary_id"] = primary_id
        record.pop("_oid", None)

    frame = pd.DataFrame.from_records(records)
    return _normalize_frame(frame)


def ingest_sh2_from_simbad(named_objects: pd.DataFrame) -> pd.DataFrame:
    if named_objects.empty:
        raise ValueError("SIMBAD named-object source is empty for SH2 ingest")

    records: dict[str, dict[str, Any]] = {}
    for _, row in named_objects.iterrows():
        candidates = [
            str(row.get("name_identifier", "")).strip(),
            str(row.get("main_id", "")).strip(),
        ]

        primary_id: str | None = None
        for candidate in candidates:
            resolved = _canonical_primary_id_from_simbad_identifier(candidate)
            if resolved and resolved.startswith("Sh2-"):
                primary_id = resolved
                break

        if primary_id is None:
            continue

        if primary_id not in records:
            records[primary_id] = {
                "primary_id": primary_id,
                "catalog": "SH2",
                "common_name": "",
                "object_type": str(row.get("otype", "")).strip(),
                "ra_deg": row.get("ra", ""),
                "dec_deg": row.get("dec", ""),
                "constellation": "",
                "aliases": "",
                "image_url": "",
                "image_attribution_url": "",
                "license_label": "",
            }

        name_identifier = str(row.get("name_identifier", "")).strip()
        main_id = str(row.get("main_id", "")).strip()
        alias_candidates = [name_identifier]
        if main_id and main_id != primary_id:
            alias_candidates.append(main_id)

        entry = records[primary_id]
        entry["aliases"] = _merge_alias_values(str(entry.get("aliases", "")), alias_candidates)
        if not entry["common_name"] and main_id and main_id != primary_id:
            entry["common_name"] = main_id

    if not records:
        raise ValueError("SIMBAD named-object query could not map any SH2 rows")

    frame = pd.DataFrame.from_records(list(records.values()))
    return _normalize_frame(frame)


def build_simbad_m_ngc_reference(named_objects: pd.DataFrame) -> pd.DataFrame:
    if named_objects.empty:
        raise ValueError("SIMBAD named-object source is empty for M/NGC enrichment")

    records: dict[str, dict[str, Any]] = {}
    for _, row in named_objects.iterrows():
        candidates = [
            str(row.get("name_identifier", "")).strip(),
            str(row.get("main_id", "")).strip(),
        ]

        primary_id: str | None = None
        for candidate in candidates:
            resolved = _canonical_primary_id_from_simbad_identifier(candidate)
            if resolved and (resolved.startswith("M") or resolved.startswith("NGC ")):
                primary_id = resolved
                break

        if primary_id is None:
            continue

        if primary_id not in records:
            records[primary_id] = {
                "primary_id": primary_id,
                "main_id": str(row.get("main_id", "")).strip(),
                "object_type": str(row.get("otype", "")).strip(),
                "aliases": "",
            }

        entry = records[primary_id]
        main_id = str(row.get("main_id", "")).strip()
        name_identifier = str(row.get("name_identifier", "")).strip()
        entry["aliases"] = _merge_alias_values(
            str(entry.get("aliases", "")),
            [name_identifier, main_id],
        )
        if not entry["main_id"] and main_id:
            entry["main_id"] = main_id
        if not entry["object_type"] and str(row.get("otype", "")).strip():
            entry["object_type"] = str(row.get("otype", "")).strip()

    if not records:
        raise ValueError("SIMBAD named-object query did not yield mappable M/NGC identifiers")

    return pd.DataFrame.from_records(list(records.values()))


def enrich_with_simbad_m_ngc(frame: pd.DataFrame, simbad_reference: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty or simbad_reference.empty:
        return frame, 0

    indexed = simbad_reference.set_index("primary_id")
    enriched = frame.copy()
    updated_rows = 0

    for idx, row in enriched.iterrows():
        primary_id = str(row.get("primary_id", ""))
        catalog = str(row.get("catalog", "")).upper()
        if catalog not in {"M", "NGC"} or primary_id not in indexed.index:
            continue

        simbad_row = indexed.loc[primary_id]
        changed = False

        main_id = str(simbad_row.get("main_id", "")).strip()
        if not str(row.get("common_name", "")).strip() and main_id and main_id != primary_id:
            enriched.at[idx, "common_name"] = main_id
            changed = True

        simbad_object_type = str(simbad_row.get("object_type", "")).strip()
        existing_object_type = str(row.get("object_type", "")).strip()
        if (not existing_object_type or existing_object_type.lower() in {"other", "-"}) and simbad_object_type:
            enriched.at[idx, "object_type"] = simbad_object_type
            changed = True

        merged_aliases = _merge_alias_values(
            str(row.get("aliases", "")),
            [alias for alias in str(simbad_row.get("aliases", "")).split(";") if str(alias).strip()],
        )
        if merged_aliases != str(row.get("aliases", "")):
            enriched.at[idx, "aliases"] = merged_aliases
            changed = True

        if changed:
            updated_rows += 1

    return _normalize_frame(enriched), int(updated_rows)


def ingest_from_seed(seed_path: Path) -> pd.DataFrame:
    if not seed_path.exists():
        raise FileNotFoundError(f"Catalog seed file not found: {seed_path}")

    source = pd.read_csv(seed_path)
    return _normalize_frame(source)


def ingest_from_enriched_csv(enriched_path: Path) -> pd.DataFrame:
    if not enriched_path.exists():
        raise FileNotFoundError(f"Enriched catalog CSV not found: {enriched_path}")

    source = pd.read_csv(enriched_path, keep_default_na=False)
    if source.empty:
        raise ValueError("Enriched catalog CSV is empty")

    records: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        primary_candidate = (
            str(row.get("id_norm", "")).strip()
            or str(row.get("simbad_main_id", "")).strip()
            or str(row.get("id_raw", "")).strip()
        )
        primary_id = _normalize_primary_id_for_app(primary_candidate)
        if not primary_id:
            continue

        cross_ids = _parse_json_list(row.get("cross_ids", ""))
        aliases = [item for item in _dedupe_preserve_order(cross_ids) if item and item != primary_id]
        links = _parse_json_list(row.get("links", ""))
        info_url = str(links[0]).strip() if links else str(row.get("simbad_object_url", "")).strip()
        emission_lines = _parse_json_list(row.get("emission_lines", ""))

        catalog_guess = str(row.get("catalog", "")).strip().upper()
        catalog = catalog_guess or _catalog_from_enriched_family(str(row.get("id_family_guess", "")), primary_id)

        object_type_norm = str(row.get("object_type_norm", "")).strip()
        object_type_simbad = str(row.get("object_type_simbad", "")).strip()
        object_type = object_type_norm or object_type_simbad

        records.append(
            {
                "primary_id": primary_id,
                "catalog": catalog,
                "common_name": str(row.get("common_name", "")).strip(),
                "object_type": object_type,
                "ra_deg": row.get("ra_j2000_deg", ""),
                "dec_deg": row.get("dec_j2000_deg", ""),
                "constellation": str(row.get("constellation", "")).strip(),
                "aliases": ";".join(aliases),
                "image_url": str(row.get("hero_image_url", "")).strip(),
                "image_attribution_url": info_url,
                "license_label": str(row.get("hero_image_credit", "")).strip(),
                "description": str(row.get("description", "")).strip(),
                "info_url": info_url,
                "dist_value": row.get("dist_value", ""),
                "dist_unit": str(row.get("dist_unit", "")).strip(),
                "redshift": row.get("redshift", ""),
                "morphology": str(row.get("morphology", "")).strip(),
                "emission_lines": "; ".join(emission_lines),
            }
        )

    if not records:
        raise ValueError("Enriched catalog CSV produced zero valid rows")

    return _normalize_frame(pd.DataFrame.from_records(records))


def ingest_sh2_from_seed(seed_path: Path) -> pd.DataFrame:
    if not seed_path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ENRICHED_OPTIONAL_COLUMNS)

    source = pd.read_csv(seed_path)
    if "catalog" not in source.columns:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ENRICHED_OPTIONAL_COLUMNS)

    sh2 = source[source["catalog"].fillna("").astype(str).str.upper() == "SH2"].copy()
    if sh2.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ENRICHED_OPTIONAL_COLUMNS)

    return _normalize_frame(sh2)


def ingest_from_openngc() -> pd.DataFrame:
    response = requests.get(OPENNGC_SOURCE_URL, timeout=OPENNGC_TIMEOUT_SECONDS)
    response.raise_for_status()

    source = pd.read_csv(io.StringIO(response.text), sep=";", dtype=str).fillna("")

    records: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        name_designation = _parse_name_designation(str(row.get("Name", "")))
        m_number = _to_catalog_number(row.get("M", ""))
        ngc_number = _to_catalog_number(row.get("NGC", ""))
        ic_number = _to_catalog_number(row.get("IC", ""))

        if m_number:
            primary_id = f"M{m_number}"
            catalog = "M"
        elif name_designation is not None:
            primary_id, catalog, _ = name_designation
        elif ngc_number:
            primary_id = f"NGC {ngc_number}"
            catalog = "NGC"
        elif ic_number:
            primary_id = f"IC {ic_number}"
            catalog = "IC"
        else:
            continue

        coords = _parse_radec(row.get("RA", ""), row.get("Dec", ""))
        if coords is None:
            continue

        common_names = _parse_common_names(str(row.get("Common names", "")))
        common_name = common_names[0] if common_names else ""

        object_type_code = str(row.get("Type", "")).strip()
        object_type = OPENNGC_TYPE_MAP.get(object_type_code, object_type_code)

        aliases = _build_aliases(
            primary_id=primary_id,
            name_designation=name_designation[0] if name_designation else "",
            m_number=m_number,
            ngc_number=ngc_number,
            ic_number=ic_number,
            common_names=common_names,
            identifiers_raw=str(row.get("Identifiers", "")),
        )

        records.append(
            {
                "primary_id": primary_id,
                "catalog": catalog,
                "common_name": common_name,
                "object_type": object_type,
                "ra_deg": coords[0],
                "dec_deg": coords[1],
                "constellation": str(row.get("Const", "")).strip(),
                "aliases": aliases,
                "image_url": "",
                "image_attribution_url": "",
                "license_label": "",
            }
        )

    if not records:
        raise ValueError("OpenNGC ingestion produced zero rows")

    return _normalize_frame(pd.DataFrame.from_records(records))


def merge_catalogs(primary: pd.DataFrame, additional: pd.DataFrame) -> pd.DataFrame:
    if additional.empty:
        return _normalize_frame(primary)

    combined = pd.concat([primary, additional], ignore_index=True)
    return _normalize_frame(combined)


def merge_catalog_with_cache_only_additions(primary: pd.DataFrame, cached: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if cached.empty:
        return _normalize_frame(primary), 0

    primary_ids = set(primary["primary_id"].astype(str).tolist())
    additions = cached[~cached["primary_id"].astype(str).isin(primary_ids)].copy()
    if additions.empty:
        return _normalize_frame(primary), 0

    merged = merge_catalogs(primary, additions)
    return merged, int(len(additions))


def load_unified_catalog(
    *,
    seed_path: Path,
    cache_path: Path,
    metadata_path: Path,
    enriched_path: Path | None = None,
    force_refresh: bool = False,
    max_cache_age_hours: int = CATALOG_CACHE_MAX_AGE_HOURS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = _read_metadata(metadata_path)
    prefer_enriched = enriched_path is not None and enriched_path.exists()

    can_use_cache = (
        not force_refresh
        and _cache_is_fresh(cache_path, max_cache_age_hours)
        and not _cache_requires_rebuild(metadata)
        and (
            not prefer_enriched
            or str(metadata.get("load_mode", "")) in {"enriched_csv", "enriched_csv_plus_cache"}
        )
    )

    if can_use_cache:
        try:
            cached = pd.read_parquet(cache_path)
            frame = _normalize_frame(cached)
            metadata.setdefault("load_mode", "cache")
            metadata.setdefault("row_count", len(frame))
            metadata.setdefault("catalog_counts", frame["catalog"].value_counts().to_dict())
            return frame, metadata
        except Exception:
            pass

    notes: list[str] = []
    source_parts: list[str] = []
    simbad_metadata: dict[str, Any] = {}
    simbad_named_objects: pd.DataFrame | None = None

    if prefer_enriched:
        try:
            frame = ingest_from_enriched_csv(enriched_path)
            load_mode = "enriched_csv"
            source_parts.append(str(enriched_path))
            simbad_metadata["named_object_rows"] = 0
            simbad_metadata["named_catalog_rows"] = 0
            simbad_metadata["named_new_rows_added"] = 0
            simbad_metadata["sh2_source"] = "enriched_csv"
            simbad_metadata["sh2_row_count"] = int((frame["catalog"] == "SH2").sum())
            simbad_metadata["m_ngc_enriched_rows"] = 0

            simbad_metadata["parquet_row_count"] = 0
            simbad_metadata["parquet_rows_added"] = 0
            if cache_path.exists():
                try:
                    cached = pd.read_parquet(cache_path)
                    cached_frame = _normalize_frame(cached)
                    simbad_metadata["parquet_row_count"] = int(len(cached_frame))
                    frame, parquet_rows_added = merge_catalog_with_cache_only_additions(frame, cached_frame)
                    simbad_metadata["parquet_rows_added"] = int(parquet_rows_added)
                    if parquet_rows_added > 0:
                        load_mode = "enriched_csv_plus_cache"
                        source_parts.append(str(cache_path))
                except Exception as error:
                    notes.append(f"Parquet supplement ingest failed: {_error_summary(error)}")
        except Exception as error:
            notes.append(f"Enriched CSV ingest failed: {_error_summary(error)}")
            prefer_enriched = False

    if not prefer_enriched:
        try:
            frame = ingest_from_openngc()
            load_mode = "openngc_ingest"
            source_parts.append(OPENNGC_SOURCE_URL)
        except Exception as error:
            frame = ingest_from_seed(seed_path)
            load_mode = "seed_fallback"
            source_parts.append(str(seed_path))
            notes.append(f"OpenNGC ingest failed: {_error_summary(error)}")

        try:
            simbad_named_objects = fetch_simbad_named_objects()
            source_parts.append("SIMBAD NAME objects (sim-tap)")
            simbad_metadata["named_object_rows"] = int(len(simbad_named_objects))
        except Exception as error:
            simbad_metadata["named_object_rows"] = 0
            notes.append(f"SIMBAD named-object query failed: {_error_summary(error)}")

    if not prefer_enriched and simbad_named_objects is not None:
        try:
            simbad_named_frame = ingest_all_from_simbad_named_objects(simbad_named_objects)
            simbad_metadata["named_catalog_rows"] = int(len(simbad_named_frame))

            pre_merge_count = int(len(frame))
            frame = merge_catalogs(frame, simbad_named_frame)
            simbad_metadata["named_new_rows_added"] = int(max(0, len(frame) - pre_merge_count))

            simbad_sh2_count = int((simbad_named_frame["catalog"] == "SH2").sum())
            if simbad_sh2_count > 0:
                simbad_metadata["sh2_source"] = "simbad"
                simbad_metadata["sh2_row_count"] = simbad_sh2_count
            else:
                sh2_frame = ingest_sh2_from_seed(seed_path)
                frame = merge_catalogs(frame, sh2_frame)
                simbad_metadata["sh2_source"] = "seed_fallback"
                simbad_metadata["sh2_row_count"] = int(len(sh2_frame))
                notes.append("SIMBAD SH2 ingest fallback: no SH2 rows mapped from named-object source")
        except Exception as error:
            sh2_frame = ingest_sh2_from_seed(seed_path)
            frame = merge_catalogs(frame, sh2_frame)
            simbad_metadata["named_catalog_rows"] = 0
            simbad_metadata["named_new_rows_added"] = 0
            simbad_metadata["sh2_source"] = "seed_fallback"
            simbad_metadata["sh2_row_count"] = int(len(sh2_frame))
            notes.append(f"SIMBAD full ingest failed: {_error_summary(error)}")
    elif not prefer_enriched:
        sh2_frame = ingest_sh2_from_seed(seed_path)
        frame = merge_catalogs(frame, sh2_frame)
        simbad_metadata["named_catalog_rows"] = 0
        simbad_metadata["named_new_rows_added"] = 0
        simbad_metadata["sh2_source"] = "seed_fallback"
        simbad_metadata["sh2_row_count"] = int(len(sh2_frame))
        notes.append("SIMBAD SH2 ingest skipped: named-object source unavailable")

    if not prefer_enriched and simbad_named_objects is not None:
        try:
            simbad_reference = build_simbad_m_ngc_reference(simbad_named_objects)
            frame, enriched_count = enrich_with_simbad_m_ngc(frame, simbad_reference)
            simbad_metadata["m_ngc_enriched_rows"] = int(enriched_count)
        except Exception as error:
            simbad_metadata["m_ngc_enriched_rows"] = 0
            notes.append(f"SIMBAD M/NGC enrichment failed: {_error_summary(error)}")
    elif not prefer_enriched:
        simbad_metadata["m_ngc_enriched_rows"] = 0
        notes.append("SIMBAD M/NGC enrichment skipped: named-object source unavailable")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)

    metadata = {
        "ingestion_version": INGESTION_VERSION,
        "load_mode": load_mode,
        "source": " | ".join(source_parts),
        "cache": str(cache_path),
        "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(frame)),
        "catalog_counts": {str(key): int(value) for key, value in frame["catalog"].value_counts().to_dict().items()},
        "cache_max_age_hours": int(max_cache_age_hours),
        "simbad": simbad_metadata,
    }

    if notes:
        metadata["notes"] = notes

    _write_metadata(metadata_path, metadata)

    return frame, metadata
