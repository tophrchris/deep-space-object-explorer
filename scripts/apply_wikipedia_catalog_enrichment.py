#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog_ingestion import ENRICHED_OPTIONAL_COLUMNS, OPTIONAL_COLUMNS, REQUIRED_COLUMNS

CACHE_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS + ENRICHED_OPTIONAL_COLUMNS
NUMERIC_COLUMNS = {"ra_deg", "dec_deg", "dist_value", "redshift"}
STRING_COLUMNS = [column for column in CACHE_COLUMNS if column not in NUMERIC_COLUMNS]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_text(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _to_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        values = [str(item).strip() for item in raw]
        return [value for value in values if value]

    text = str(raw or "").strip()
    if not text:
        return []

    pieces: list[str] = []
    if ";" in text:
        pieces = [part.strip() for part in text.split(";")]
    elif "," in text:
        pieces = [part.strip() for part in text.split(",")]
    else:
        pieces = [text]

    return [piece for piece in pieces if piece]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isnan(number):
            return None
        return number
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def _infer_catalog(primary_id: str) -> str:
    text = str(primary_id or "").strip().upper()
    if text.startswith("NGC "):
        return "NGC"
    if text.startswith("IC "):
        return "IC"
    if text.startswith("SH2-"):
        return "SH2"
    if text.startswith("M") and text[1:].strip().replace(" ", "").isdigit():
        return "M"
    return "WIKI"


def _parse_enrichment_row(row: dict[str, Any]) -> dict[str, Any]:
    catalog_enrichment = row.get("catalog_enrichment")
    if not isinstance(catalog_enrichment, dict):
        catalog_enrichment = {}

    aliases = _to_list(catalog_enrichment.get("aliases", []))
    designations = _to_list(row.get("designations", []))
    aliases = _dedupe_preserve_order(aliases + designations)

    emission_lines = _dedupe_preserve_order(_to_list(catalog_enrichment.get("emission_lines", [])))

    info_url = str(catalog_enrichment.get("info_url", "")).strip()
    if not info_url:
        info_url = str(row.get("wikipedia_url", "")).strip()

    description = str(catalog_enrichment.get("description", "")).strip()
    if not description:
        description = str(row.get("description", "")).strip()

    image_url = str(catalog_enrichment.get("image_url", "")).strip()
    if not image_url:
        image_url = str(row.get("image_url", "")).strip()

    return {
        "common_name": str(catalog_enrichment.get("common_name", "")).strip(),
        "object_type": str(catalog_enrichment.get("object_type", "")).strip(),
        "ra_deg": _coerce_float(catalog_enrichment.get("ra_deg")),
        "dec_deg": _coerce_float(catalog_enrichment.get("dec_deg")),
        "constellation": str(catalog_enrichment.get("constellation", "")).strip(),
        "aliases": aliases,
        "image_url": image_url,
        "image_attribution_url": str(catalog_enrichment.get("image_attribution_url", "")).strip(),
        "description": description,
        "info_url": info_url,
        "emission_lines": emission_lines,
        "wikipedia_primary_id": str(catalog_enrichment.get("wikipedia_primary_id", "")).strip(),
        "wikipedia_catalog": str(catalog_enrichment.get("wikipedia_catalog", "")).strip().upper(),
    }


def _has_enrichment_payload(enrichment: dict[str, Any]) -> bool:
    return any(
        [
            bool(enrichment.get("common_name")),
            bool(enrichment.get("object_type")),
            enrichment.get("ra_deg") is not None,
            enrichment.get("dec_deg") is not None,
            bool(enrichment.get("constellation")),
            bool(enrichment.get("aliases")),
            bool(enrichment.get("image_url")),
            bool(enrichment.get("image_attribution_url")),
            bool(enrichment.get("description")),
            bool(enrichment.get("info_url")),
            bool(enrichment.get("emission_lines")),
        ]
    )


def _make_empty_row() -> dict[str, Any]:
    row: dict[str, Any] = {}
    for column in STRING_COLUMNS:
        row[column] = ""
    for column in NUMERIC_COLUMNS:
        row[column] = float("nan")
    return row


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in CACHE_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = "" if column in STRING_COLUMNS else float("nan")

    prepared = prepared[CACHE_COLUMNS]
    for column in STRING_COLUMNS:
        prepared[column] = prepared[column].fillna("").astype(str).str.strip()
    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge wikipedia_catalog_enrichment.json into dso_catalog_cache.parquet "
            "by updating matching primary_id rows and inserting missing rows."
        ),
    )
    parser.add_argument(
        "--catalog-cache-path",
        type=Path,
        default=Path("data/dso_catalog_cache.parquet"),
        help="Input parquet cache path.",
    )
    parser.add_argument(
        "--enrichment-path",
        type=Path,
        default=Path("data/wikipedia_catalog_enrichment.json"),
        help="Wikipedia enrichment JSON path.",
    )
    parser.add_argument(
        "--output-cache-path",
        type=Path,
        default=None,
        help="Output parquet path (default: overwrite --catalog-cache-path).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/dso_catalog_cache_meta.json"),
        help="Metadata JSON path to refresh with row counts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print merge stats without writing parquet or metadata.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    cache_path = args.catalog_cache_path
    enrichment_path = args.enrichment_path
    output_cache_path = args.output_cache_path or cache_path
    metadata_path = args.metadata_path

    if not cache_path.exists():
        raise FileNotFoundError(f"Catalog cache not found: {cache_path}")
    if not enrichment_path.exists():
        raise FileNotFoundError(f"Wikipedia enrichment JSON not found: {enrichment_path}")

    cache_frame = _prepare_frame(pd.read_parquet(cache_path))
    payload = json.loads(enrichment_path.read_text(encoding="utf-8"))
    targets = payload.get("targets", [])
    if not isinstance(targets, list):
        raise ValueError("Enrichment payload is missing a valid 'targets' array.")

    cache_frame["primary_id"] = cache_frame["primary_id"].fillna("").astype(str).str.strip()
    id_to_index = {pid: idx for idx, pid in cache_frame["primary_id"].items() if pid}

    updates_by_field: dict[str, int] = {}
    update_rows = 0
    inserted_rows = 0
    skipped_rows = 0
    skipped_inserts_missing_coords = 0
    insert_candidates: list[dict[str, Any]] = []

    for item in targets:
        if not isinstance(item, dict):
            skipped_rows += 1
            continue

        primary_id = str(item.get("primary_id", "")).strip()
        if not primary_id:
            skipped_rows += 1
            continue

        enrichment = _parse_enrichment_row(item)
        if not _has_enrichment_payload(enrichment):
            skipped_rows += 1
            continue

        if primary_id in id_to_index:
            idx = id_to_index[primary_id]
            row_changed = False

            for field in ("common_name", "object_type", "constellation", "image_url", "image_attribution_url", "description", "info_url"):
                new_value = str(enrichment.get(field, "")).strip()
                if not new_value:
                    continue
                old_value = str(cache_frame.at[idx, field]).strip()
                if old_value == new_value:
                    continue
                cache_frame.at[idx, field] = new_value
                updates_by_field[field] = updates_by_field.get(field, 0) + 1
                row_changed = True

            for field in ("ra_deg", "dec_deg"):
                new_number = _coerce_float(enrichment.get(field))
                if new_number is None:
                    continue
                old_number = _coerce_float(cache_frame.at[idx, field])
                if old_number is not None and abs(old_number - new_number) < 1e-12:
                    continue
                cache_frame.at[idx, field] = new_number
                updates_by_field[field] = updates_by_field.get(field, 0) + 1
                row_changed = True

            alias_additions = _to_list(enrichment.get("aliases", []))
            if alias_additions:
                existing_aliases = _to_list(cache_frame.at[idx, "aliases"])
                merged_aliases = _dedupe_preserve_order(existing_aliases + alias_additions)
                merged_alias_text = ";".join(merged_aliases)
                if merged_alias_text != str(cache_frame.at[idx, "aliases"]).strip():
                    cache_frame.at[idx, "aliases"] = merged_alias_text
                    updates_by_field["aliases"] = updates_by_field.get("aliases", 0) + 1
                    row_changed = True

            emission_additions = _to_list(enrichment.get("emission_lines", []))
            if emission_additions:
                existing_emission = _to_list(cache_frame.at[idx, "emission_lines"])
                merged_emission = _dedupe_preserve_order(existing_emission + emission_additions)
                merged_emission_text = "; ".join(merged_emission)
                if merged_emission_text != str(cache_frame.at[idx, "emission_lines"]).strip():
                    cache_frame.at[idx, "emission_lines"] = merged_emission_text
                    updates_by_field["emission_lines"] = updates_by_field.get("emission_lines", 0) + 1
                    row_changed = True

            if row_changed:
                update_rows += 1
            continue

        ra_deg = _coerce_float(enrichment.get("ra_deg"))
        dec_deg = _coerce_float(enrichment.get("dec_deg"))
        if ra_deg is None or dec_deg is None:
            skipped_inserts_missing_coords += 1
            continue

        new_row = _make_empty_row()
        new_row["primary_id"] = primary_id
        new_row["catalog"] = (
            str(item.get("catalog", "")).strip().upper()
            or str(enrichment.get("wikipedia_catalog", "")).strip().upper()
            or _infer_catalog(primary_id)
        )
        new_row["common_name"] = str(enrichment.get("common_name", "")).strip()
        new_row["object_type"] = str(enrichment.get("object_type", "")).strip()
        new_row["ra_deg"] = ra_deg
        new_row["dec_deg"] = dec_deg
        new_row["constellation"] = str(enrichment.get("constellation", "")).strip()
        new_row["aliases"] = ";".join(_to_list(enrichment.get("aliases", [])))
        new_row["object_type_group"] = str(item.get("object_type_group", "")).strip()
        new_row["image_url"] = str(enrichment.get("image_url", "")).strip()
        new_row["image_attribution_url"] = str(enrichment.get("image_attribution_url", "")).strip()
        new_row["description"] = str(enrichment.get("description", "")).strip()
        new_row["info_url"] = str(enrichment.get("info_url", "")).strip()
        new_row["emission_lines"] = "; ".join(_to_list(enrichment.get("emission_lines", [])))
        insert_candidates.append(new_row)
        inserted_rows += 1

    if insert_candidates:
        cache_frame = pd.concat([cache_frame, pd.DataFrame.from_records(insert_candidates)], ignore_index=True)

    cache_frame = _prepare_frame(cache_frame)
    cache_frame = cache_frame[cache_frame["primary_id"] != ""]
    cache_frame = cache_frame.drop_duplicates(subset=["primary_id"], keep="first")
    cache_frame = cache_frame.sort_values(by=["catalog", "primary_id"], ascending=[True, True]).reset_index(drop=True)

    summary = {
        "cache_path": str(cache_path),
        "enrichment_path": str(enrichment_path),
        "output_cache_path": str(output_cache_path),
        "targets_seen": int(len(targets)),
        "rows_updated": int(update_rows),
        "rows_inserted": int(inserted_rows),
        "skipped_rows_without_payload": int(skipped_rows),
        "skipped_inserts_missing_coordinates": int(skipped_inserts_missing_coords),
        "field_updates": {key: int(value) for key, value in sorted(updates_by_field.items())},
        "result_row_count": int(len(cache_frame)),
    }

    print("[wikipedia-merge] summary")
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        print("[wikipedia-merge] dry-run enabled; no files written")
        return

    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_frame.to_parquet(output_cache_path, index=False)
    print(f"[wikipedia-merge] wrote parquet: {output_cache_path}")

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
        except (OSError, json.JSONDecodeError):
            metadata = {}

    metadata["cache"] = str(output_cache_path)
    metadata["loaded_at_utc"] = _utc_now_iso()
    metadata["row_count"] = int(len(cache_frame))
    metadata["catalog_counts"] = {
        str(key): int(value) for key, value in cache_frame["catalog"].value_counts().to_dict().items()
    }
    metadata["wikipedia_enrichment_merge"] = {
        "applied_at_utc": _utc_now_iso(),
        "enrichment_path": str(enrichment_path),
        "targets_seen": int(len(targets)),
        "rows_updated": int(update_rows),
        "rows_inserted": int(inserted_rows),
        "skipped_rows_without_payload": int(skipped_rows),
        "skipped_inserts_missing_coordinates": int(skipped_inserts_missing_coords),
        "field_updates": {key: int(value) for key, value in sorted(updates_by_field.items())},
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[wikipedia-merge] wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
