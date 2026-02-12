from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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


def ingest_from_seed(seed_path: Path) -> pd.DataFrame:
    if not seed_path.exists():
        raise FileNotFoundError(f"Catalog seed file not found: {seed_path}")

    source = pd.read_csv(seed_path)
    return _normalize_frame(source)


def load_unified_catalog(
    *,
    seed_path: Path,
    cache_path: Path,
    metadata_path: Path,
    force_refresh: bool = False,
    max_cache_age_hours: int = CATALOG_CACHE_MAX_AGE_HOURS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    use_cache = not force_refresh and _cache_is_fresh(cache_path, max_cache_age_hours)

    if use_cache:
        try:
            cached = pd.read_parquet(cache_path)
            frame = _normalize_frame(cached)
            metadata = _read_metadata(metadata_path)
            metadata.setdefault("load_mode", "cache")
            metadata.setdefault("row_count", len(frame))
            metadata.setdefault("catalog_counts", frame["catalog"].value_counts().to_dict())
            return frame, metadata
        except Exception:
            pass

    frame = ingest_from_seed(seed_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)

    metadata = {
        "load_mode": "seed_ingest",
        "source": str(seed_path),
        "cache": str(cache_path),
        "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(frame)),
        "catalog_counts": {str(key): int(value) for key, value in frame["catalog"].value_counts().to_dict().items()},
        "cache_max_age_hours": int(max_cache_age_hours),
    }
    _write_metadata(metadata_path, metadata)

    return frame, metadata
