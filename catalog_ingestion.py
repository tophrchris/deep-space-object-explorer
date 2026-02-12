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

CATALOG_CACHE_MAX_AGE_HOURS = 24
INGESTION_VERSION = 3
OPENNGC_SOURCE_URL = "https://raw.githubusercontent.com/mattiaverga/OpenNGC/master/database_files/NGC.csv"
OPENNGC_TIMEOUT_SECONDS = 45
SIMBAD_TAP_SYNC_ENDPOINTS = [
    "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
    "https://simbad.u-strasbg.fr/simbad/sim-tap/sync",
]
SIMBAD_TIMEOUT_SECONDS = 90

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


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Catalog source is missing required columns: {missing_list}")

    normalized = frame.copy()

    for column in OPTIONAL_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""

    normalized = normalized[REQUIRED_COLUMNS + OPTIONAL_COLUMNS]

    normalized["primary_id"] = normalized["primary_id"].fillna("").astype(str).str.strip()
    normalized["catalog"] = normalized["catalog"].fillna("").astype(str).str.upper().str.strip()
    normalized["common_name"] = normalized["common_name"].fillna("").astype(str).str.strip()
    normalized["object_type"] = normalized["object_type"].fillna("").astype(str).str.strip()
    normalized["constellation"] = normalized["constellation"].fillna("").astype(str).str.strip()
    normalized["aliases"] = normalized["aliases"].fillna("").astype(str).str.strip()

    normalized["ra_deg"] = pd.to_numeric(normalized["ra_deg"], errors="coerce")
    normalized["dec_deg"] = pd.to_numeric(normalized["dec_deg"], errors="coerce")

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
    raise RuntimeError(f"SIMBAD TAP query failed: {last_error.__class__.__name__}") from last_error


def _canonical_primary_id_from_simbad_identifier(raw_identifier: str) -> str | None:
    compact = " ".join(str(raw_identifier).strip().split()).upper()
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

    sh2_match = re.match(r"^SH\s*2[-\s]*0*([0-9]+)\s*([A-Z]*)$", compact)
    if sh2_match:
        sh2_number = str(int(sh2_match.group(1)))
        suffix = str(sh2_match.group(2)).strip()
        if suffix:
            return f"Sh2-{sh2_number}{suffix}"
        return f"Sh2-{sh2_number}"

    return None


def ingest_sh2_from_simbad() -> pd.DataFrame:
    query = """
        SELECT
            id AS identifier,
            basic.main_id AS main_id,
            basic.ra AS ra_deg,
            basic.dec AS dec_deg,
            basic.otype AS object_type
        FROM ident
        JOIN basic ON ident.oidref = basic.oid
        WHERE
               id LIKE 'SH2-%'
            OR id LIKE 'SH 2-%'
            OR id LIKE 'SH  2-%'
            OR id LIKE 'Sh2-%'
            OR id LIKE 'Sh 2-%'
            OR id LIKE 'Sh  2-%'
            OR id LIKE 'sh2-%'
            OR id LIKE 'sh 2-%'
            OR id LIKE 'sh  2-%'
            OR id LIKE 'SH2 %'
            OR id LIKE 'SH 2 %'
            OR id LIKE 'SH  2 %'
            OR id LIKE 'Sh2 %'
            OR id LIKE 'Sh 2 %'
            OR id LIKE 'Sh  2 %'
            OR id LIKE 'sh2 %'
            OR id LIKE 'sh 2 %'
            OR id LIKE 'sh  2 %'
    """
    source = _query_simbad_tap(query)
    if source.empty:
        raise ValueError("SIMBAD SH2 query returned zero rows")

    required = {"identifier", "main_id", "ra_deg", "dec_deg", "object_type"}
    missing_columns = required - set(source.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"SIMBAD SH2 query missing columns: {missing_list}")

    records: dict[str, dict[str, Any]] = {}
    for _, row in source.iterrows():
        primary_id = _canonical_primary_id_from_simbad_identifier(str(row["identifier"]))
        if primary_id is None or not primary_id.startswith("Sh2-"):
            continue

        if primary_id not in records:
            records[primary_id] = {
                "primary_id": primary_id,
                "catalog": "SH2",
                "common_name": "",
                "object_type": str(row.get("object_type", "")).strip(),
                "ra_deg": row.get("ra_deg", ""),
                "dec_deg": row.get("dec_deg", ""),
                "constellation": "",
                "aliases": "",
                "image_url": "",
                "image_attribution_url": "",
                "license_label": "",
            }

        identifier = str(row.get("identifier", "")).strip()
        main_id = str(row.get("main_id", "")).strip()
        alias_candidates = [identifier]
        if main_id and main_id != primary_id:
            alias_candidates.append(main_id)

        entry = records[primary_id]
        entry["aliases"] = _merge_alias_values(str(entry.get("aliases", "")), alias_candidates)
        if not entry["common_name"] and main_id and main_id != primary_id:
            entry["common_name"] = main_id

    if not records:
        raise ValueError("SIMBAD SH2 query could not map any rows to Sh2 identifiers")

    frame = pd.DataFrame.from_records(records.values())
    return _normalize_frame(frame)


def build_simbad_m_ngc_reference() -> pd.DataFrame:
    query = """
        SELECT
            id AS identifier,
            basic.main_id AS main_id,
            basic.otype AS object_type
        FROM ident
        JOIN basic ON ident.oidref = basic.oid
        WHERE
               id LIKE 'M %'
            OR id LIKE 'M%'
            OR id LIKE 'm %'
            OR id LIKE 'm%'
            OR id LIKE 'NGC %'
            OR id LIKE 'NGC%'
            OR id LIKE 'ngc %'
            OR id LIKE 'ngc%'
    """
    source = _query_simbad_tap(query)
    if source.empty:
        raise ValueError("SIMBAD M/NGC enrichment query returned zero rows")

    required = {"identifier", "main_id", "object_type"}
    missing_columns = required - set(source.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"SIMBAD M/NGC query missing columns: {missing_list}")

    records: dict[str, dict[str, Any]] = {}
    for _, row in source.iterrows():
        identifier = str(row.get("identifier", "")).strip()
        primary_id = _canonical_primary_id_from_simbad_identifier(identifier)
        if primary_id is None or not (primary_id.startswith("M") or primary_id.startswith("NGC ")):
            continue

        if primary_id not in records:
            records[primary_id] = {
                "primary_id": primary_id,
                "main_id": str(row.get("main_id", "")).strip(),
                "object_type": str(row.get("object_type", "")).strip(),
                "aliases": "",
            }

        entry = records[primary_id]
        main_id = str(row.get("main_id", "")).strip()
        entry["aliases"] = _merge_alias_values(
            str(entry.get("aliases", "")),
            [identifier, main_id],
        )
        if not entry["main_id"] and main_id:
            entry["main_id"] = main_id
        if not entry["object_type"] and str(row.get("object_type", "")).strip():
            entry["object_type"] = str(row.get("object_type", "")).strip()

    if not records:
        raise ValueError("SIMBAD M/NGC enrichment query did not yield mappable identifiers")

    return pd.DataFrame.from_records(records.values())


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


def ingest_sh2_from_seed(seed_path: Path) -> pd.DataFrame:
    if not seed_path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

    source = pd.read_csv(seed_path)
    if "catalog" not in source.columns:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

    sh2 = source[source["catalog"].fillna("").astype(str).str.upper() == "SH2"].copy()
    if sh2.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

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


def load_unified_catalog(
    *,
    seed_path: Path,
    cache_path: Path,
    metadata_path: Path,
    force_refresh: bool = False,
    max_cache_age_hours: int = CATALOG_CACHE_MAX_AGE_HOURS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = _read_metadata(metadata_path)

    can_use_cache = (
        not force_refresh
        and _cache_is_fresh(cache_path, max_cache_age_hours)
        and not _cache_requires_rebuild(metadata)
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

    try:
        frame = ingest_from_openngc()
        load_mode = "openngc_ingest"
        source_parts.append(OPENNGC_SOURCE_URL)
    except Exception as error:
        frame = ingest_from_seed(seed_path)
        load_mode = "seed_fallback"
        source_parts.append(str(seed_path))
        notes.append(f"OpenNGC ingest failed: {error.__class__.__name__}")

    try:
        sh2_frame = ingest_sh2_from_simbad()
        source_parts.append("SIMBAD SH2 (sim-tap)")
        simbad_metadata["sh2_source"] = "simbad"
        simbad_metadata["sh2_row_count"] = int(len(sh2_frame))
    except Exception as error:
        sh2_frame = ingest_sh2_from_seed(seed_path)
        simbad_metadata["sh2_source"] = "seed_fallback"
        simbad_metadata["sh2_row_count"] = int(len(sh2_frame))
        notes.append(f"SIMBAD SH2 ingest failed: {error.__class__.__name__}")

    frame = merge_catalogs(frame, sh2_frame)

    try:
        simbad_reference = build_simbad_m_ngc_reference()
        frame, enriched_count = enrich_with_simbad_m_ngc(frame, simbad_reference)
        source_parts.append("SIMBAD M/NGC enrichment (sim-tap)")
        simbad_metadata["m_ngc_enriched_rows"] = int(enriched_count)
    except Exception as error:
        simbad_metadata["m_ngc_enriched_rows"] = 0
        notes.append(f"SIMBAD M/NGC enrichment failed: {error.__class__.__name__}")

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
