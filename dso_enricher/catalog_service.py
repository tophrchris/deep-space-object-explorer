from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dso_enricher.catalog_ingestion import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, load_unified_catalog

CATALOG_MODE_LEGACY = "legacy"
CATALOG_MODE_CURATED_PARQUET = "curated_parquet"

SEARCH_INDEX_COLUMNS = ["primary_id_norm", "aliases_norm", "search_blob_norm"]
_VALID_CATALOG_MODES = {CATALOG_MODE_LEGACY, CATALOG_MODE_CURATED_PARQUET}


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def canonicalize_designation(query: str) -> str:
    compact = normalize_text(query)

    match = re.match(r"^(messier|m)(\d+)$", compact)
    if match:
        return f"M{int(match.group(2))}"

    match = re.match(r"^(ngc)(\d+)$", compact)
    if match:
        return f"NGC {int(match.group(2))}"

    match = re.match(r"^(ic)(\d+)$", compact)
    if match:
        return f"IC {int(match.group(2))}"

    match = re.match(r"^(sh2|sharpless)(\d+)$", compact)
    if match:
        return f"Sh2-{int(match.group(2))}"

    return query.strip()


def list_catalog_filters(catalog: pd.DataFrame) -> dict[str, list[str]]:
    def _sorted_unique(column: str) -> list[str]:
        if column not in catalog.columns:
            return []
        values = (
            catalog[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        values = values[values != ""]
        return sorted(values.unique().tolist())

    return {
        "catalogs": _sorted_unique("catalog"),
        "object_types": _sorted_unique("object_type"),
        "object_type_groups": _sorted_unique("object_type_group"),
        "constellations": _sorted_unique("constellation"),
    }


def get_object_by_id(catalog: pd.DataFrame, primary_id: str | None) -> pd.Series | None:
    target_id = str(primary_id or "").strip()
    if not target_id or "primary_id" not in catalog.columns:
        return None

    matches = catalog[catalog["primary_id"] == target_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def search_catalog(catalog: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query.strip():
        return catalog.copy()

    indexed = _ensure_search_index(catalog)
    canonical = canonicalize_designation(query)
    query_tokens = _build_search_tokens(query=query, canonical=canonical)

    exact_mask = pd.Series(False, index=indexed.index)
    alias_exact = pd.Series(False, index=indexed.index)
    partial_mask = pd.Series(False, index=indexed.index)
    for token in query_tokens:
        exact_mask = exact_mask | (indexed["primary_id_norm"] == token)
        alias_exact = alias_exact | indexed["aliases_norm"].str.contains(token, regex=False)
        partial_mask = partial_mask | indexed["search_blob_norm"].str.contains(token, regex=False)

    results = indexed[exact_mask | alias_exact | partial_mask].copy()
    results["_rank"] = np.where(exact_mask.loc[results.index] | alias_exact.loc[results.index], 0, 1)
    results = results.sort_values(by=["_rank", "catalog", "primary_id"], ascending=[True, True, True])
    return results.drop(columns=["_rank"])


def _build_search_tokens(query: str, canonical: str) -> list[str]:
    query_norm = normalize_text(query)
    canonical_norm = normalize_text(canonical)
    tokens = {query_norm, canonical_norm}

    # Tolerate common emission-line query variants:
    # - `Olll` / `Nll` / `Sll` where `l` is used instead of `I`
    # - shorthand `O3`, `N2`, `S2`
    if re.fullmatch(r"o[l1]{3}", query_norm):
        tokens.add("oiii")
    if re.fullmatch(r"n[l1]{2}", query_norm):
        tokens.add("nii")
    if re.fullmatch(r"s[l1]{2}", query_norm):
        tokens.add("sii")
    if query_norm == "o3":
        tokens.add("oiii")
    if query_norm == "n2":
        tokens.add("nii")
    if query_norm == "s2":
        tokens.add("sii")

    return [token for token in tokens if token]


def load_catalog_from_cache(*, cache_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    indexed = _build_search_index(pd.read_parquet(cache_path))
    validation = validate_catalog(indexed)
    catalog_counts = indexed["catalog"].value_counts().to_dict() if "catalog" in indexed.columns else {}
    metadata = {
        "load_mode": "cache_parquet",
        "source": str(cache_path),
        "cache": str(cache_path),
        "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(indexed)),
        "catalog_counts": catalog_counts,
        "validation": validation,
        "filters": list_catalog_filters(indexed),
    }
    return indexed, metadata


def load_catalog_data(
    *,
    seed_path: Path,
    cache_path: Path,
    metadata_path: Path,
    enriched_path: Path | None = None,
    force_refresh: bool = False,
    mode: str = CATALOG_MODE_LEGACY,
    curated_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    requested_mode = str(mode).strip().lower()
    notes: list[str] = []

    if requested_mode not in _VALID_CATALOG_MODES:
        notes.append(f"Unknown catalog mode '{requested_mode}', falling back to '{CATALOG_MODE_LEGACY}'.")
        requested_mode = CATALOG_MODE_LEGACY

    if requested_mode == CATALOG_MODE_CURATED_PARQUET:
        curated_catalog_path = curated_path or cache_path
        try:
            curated_frame = pd.read_parquet(curated_catalog_path)
            indexed_curated = _build_search_index(curated_frame)
            validation = validate_catalog(indexed_curated)
            if validation["missing_required_columns"]:
                missing = ", ".join(validation["missing_required_columns"])
                raise ValueError(f"missing required columns: {missing}")
            if validation["duplicate_primary_id_count"] > 0:
                raise ValueError(f"duplicate primary_id rows: {validation['duplicate_primary_id_count']}")
            if validation["blank_primary_id_count"] > 0:
                raise ValueError(f"blank primary_id rows: {validation['blank_primary_id_count']}")
            if validation["row_count"] <= 0:
                raise ValueError("catalog has zero rows")

            catalog_counts = (
                indexed_curated["catalog"].value_counts().to_dict()
                if "catalog" in indexed_curated.columns
                else {}
            )
            metadata = {
                "load_mode": CATALOG_MODE_CURATED_PARQUET,
                "source": str(curated_catalog_path),
                "cache": str(curated_catalog_path),
                "row_count": int(len(indexed_curated)),
                "catalog_counts": catalog_counts,
                "validation": validation,
                "feature_mode_requested": mode,
                "feature_mode_active": CATALOG_MODE_CURATED_PARQUET,
                "filters": list_catalog_filters(indexed_curated),
            }
            return indexed_curated, metadata
        except Exception as error:
            notes.append(
                "Curated catalog mode fallback to legacy: "
                f"{error.__class__.__name__}: {str(error).strip() or 'unknown error'}"
            )

    frame, metadata = load_unified_catalog(
        seed_path=seed_path,
        cache_path=cache_path,
        metadata_path=metadata_path,
        enriched_path=enriched_path,
        force_refresh=force_refresh,
    )
    indexed = _build_search_index(frame)
    validation = validate_catalog(indexed)

    meta = dict(metadata or {})
    existing_notes = meta.get("notes", [])
    merged_notes: list[str] = []
    if isinstance(existing_notes, list):
        merged_notes.extend(str(note) for note in existing_notes if str(note).strip())
    elif isinstance(existing_notes, str) and existing_notes.strip():
        merged_notes.append(existing_notes.strip())
    merged_notes.extend(notes)
    if merged_notes:
        meta["notes"] = merged_notes

    if "row_count" not in meta:
        meta["row_count"] = int(len(indexed))
    if "catalog_counts" not in meta and "catalog" in indexed.columns:
        meta["catalog_counts"] = indexed["catalog"].value_counts().to_dict()

    meta["validation"] = validation
    meta["feature_mode_requested"] = mode
    meta["feature_mode_active"] = CATALOG_MODE_LEGACY
    meta["filters"] = list_catalog_filters(indexed)

    return indexed, meta


def validate_catalog(catalog: pd.DataFrame) -> dict[str, Any]:
    required_columns = list(REQUIRED_COLUMNS)
    missing_required_columns = [column for column in required_columns if column not in catalog.columns]

    row_count = int(len(catalog))
    if "primary_id" in catalog.columns:
        primary_ids = catalog["primary_id"].fillna("").astype(str).str.strip()
        blank_primary_id_count = int((primary_ids == "").sum())
        unique_primary_id_count = int(primary_ids[primary_ids != ""].nunique())
        duplicate_primary_id_count = int(primary_ids[primary_ids != ""].duplicated().sum())
    else:
        blank_primary_id_count = row_count
        unique_primary_id_count = 0
        duplicate_primary_id_count = 0

    warnings: list[str] = []
    if missing_required_columns:
        warnings.append(f"Missing required columns: {', '.join(missing_required_columns)}.")
    if row_count == 0:
        warnings.append("Catalog contains zero rows.")
    if blank_primary_id_count > 0:
        warnings.append(f"Catalog contains {blank_primary_id_count} blank primary_id values.")
    if duplicate_primary_id_count > 0:
        warnings.append(f"Catalog contains {duplicate_primary_id_count} duplicate primary_id values.")

    return {
        "required_columns": required_columns,
        "optional_columns": list(OPTIONAL_COLUMNS),
        "row_count": row_count,
        "unique_primary_id_count": unique_primary_id_count,
        "blank_primary_id_count": blank_primary_id_count,
        "duplicate_primary_id_count": duplicate_primary_id_count,
        "missing_required_columns": missing_required_columns,
        "warnings": warnings,
    }


def _ensure_search_index(catalog: pd.DataFrame) -> pd.DataFrame:
    if all(column in catalog.columns for column in SEARCH_INDEX_COLUMNS):
        return catalog
    return _build_search_index(catalog)


def _build_search_index(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.copy()
    for column in [
        "common_name",
        "object_type",
        "object_type_group",
        "constellation",
        "aliases",
        "catalog",
        "description",
        "emission_lines",
    ]:
        if column in indexed.columns:
            indexed[column] = indexed[column].fillna("")

    object_type_text = indexed["object_type"].fillna("") if "object_type" in indexed.columns else ""
    object_type_group_text = indexed["object_type_group"].fillna("") if "object_type_group" in indexed.columns else ""
    emission_lines_text = indexed["emission_lines"].fillna("") if "emission_lines" in indexed.columns else ""

    indexed["primary_id_norm"] = indexed["primary_id"].map(normalize_text)
    indexed["aliases_norm"] = indexed["aliases"].map(normalize_text)
    indexed["search_blob_norm"] = (
        indexed["primary_id"].fillna("")
        + " "
        + indexed["common_name"].fillna("")
        + " "
        + indexed["aliases"].fillna("")
        + " "
        + indexed["catalog"].fillna("")
        + " "
        + indexed["description"].fillna("")
        + " "
        + object_type_text
        + " "
        + object_type_group_text
        + " "
        + emission_lines_text
    ).map(normalize_text)

    return indexed
