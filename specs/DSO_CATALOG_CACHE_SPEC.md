# `dso_catalog_cache.parquet` Specification

Status: normative for cache creation and maintenance.

This document defines the expected structure and field behavior for `data/dso_catalog_cache.parquet`.
It is derived from the active normalization and write paths in `catalog_ingestion.py` and `scripts/apply_wikipedia_catalog_enrichment.py`.

## 1. Artifact Contract

- File path: `data/dso_catalog_cache.parquet`.
- Format: a single Parquet table written with `index=False`.
- Grain: one row per unique deep-sky object (`primary_id`).
- Canonical writer behavior:
1. Coerce to canonical column set and order.
2. Normalize strings and numerics.
3. Drop invalid rows (missing primary key or missing coordinates).
4. De-duplicate by `primary_id` (keep first).
5. Sort by `catalog`, then `primary_id`.

## 2. Canonical Column Set and Order

The Parquet file must contain exactly these columns in this order:

1. `primary_id`
2. `catalog`
3. `common_name`
4. `object_type`
5. `ra_deg`
6. `dec_deg`
7. `constellation`
8. `aliases`
9. `object_type_group`
10. `image_url`
11. `image_attribution_url`
12. `license_label`
13. `description`
14. `info_url`
15. `dist_value`
16. `dist_unit`
17. `redshift`
18. `ang_size_maj_arcmin`
19. `ang_size_min_arcmin`
20. `morphology`
21. `emission_lines`

## 3. Type Contract

String-like columns:

- `primary_id`
- `catalog`
- `common_name`
- `object_type`
- `constellation`
- `aliases`
- `object_type_group`
- `image_url`
- `image_attribution_url`
- `license_label`
- `description`
- `info_url`
- `dist_unit`
- `morphology`
- `emission_lines`

Numeric columns (float-compatible):

- `ra_deg`
- `dec_deg`
- `dist_value`
- `redshift`
- `ang_size_maj_arcmin`
- `ang_size_min_arcmin`

Numeric parse failures must become null/NaN during normalization.

## 4. Global Hard Requirements

These are hard constraints enforced by the current ingestion/normalization code:

1. Required columns must exist: `primary_id`, `catalog`, `common_name`, `object_type`, `ra_deg`, `dec_deg`.
2. Missing optional/enriched columns are created with empty-string defaults before normalization.
3. All string fields listed in Section 3 are trimmed.
4. `catalog` is uppercased and trimmed.
5. `ra_deg` and `dec_deg` must be numeric and non-null; rows failing this are dropped.
6. `primary_id` must be non-empty after trim; empty rows are dropped.
7. Duplicate `primary_id` rows are removed, keeping the first encountered row.
8. Final rows are sorted by `catalog`, then `primary_id`, ascending.

## 5. Field-by-Field Specification

### 5.1 `primary_id`

- Type: string.
- Hard rules:
1. Must be present as a column.
2. Trimmed.
3. Must be non-empty per row (rows with blank values are dropped).
4. Must be unique in final output (first occurrence wins).
- Semantic contract:
1. Stable row identifier and primary key for downstream consumers.
2. Canonicalized when possible, especially for designation-like IDs.
- Canonical forms expected:
1. `M<number>` (for Messier).
2. `NGC <number>` or `NGC <number> <suffix>`.
3. `IC <number>` or `IC <number> <suffix>`.
4. `Sh2-<number>` or `Sh2-<number><suffix>`.
5. `SIMBAD OID <oid>` as fallback for unresolved SIMBAD objects.

### 5.2 `catalog`

- Type: string.
- Hard rules:
1. Must be present as a column.
2. Uppercased and trimmed.
- Semantic contract:
1. Top-level catalog family label used for sorting/filtering.
2. Typical values: `M`, `NGC`, `IC`, `SH2`, `SIMBAD`.
3. Extended values can appear from enrichment workflows (`C`, `SURVEY`, `WIKI`) and should remain uppercase.

### 5.3 `common_name`

- Type: string.
- Hard rules:
1. Required column.
2. Trimmed.
- Semantic contract:
1. Human-friendly name.
2. Empty string allowed.

### 5.4 `object_type`

- Type: string.
- Hard rules:
1. Required column.
2. Trimmed.
- Semantic contract:
1. Object classification label.
2. Can be remapped from SIMBAD otype mappings.
3. Can be canonicalized via object-type grouping mapping rules.
4. Empty string allowed.

### 5.5 `ra_deg`

- Type: numeric float.
- Hard rules:
1. Required column.
2. Coerced with numeric parsing.
3. Non-numeric/null after coercion causes row drop.
- Semantic contract:
1. Right Ascension in decimal degrees, expected ICRS/J2000.
2. Quality expectation (recommended): value in `[0, 360)`.

### 5.6 `dec_deg`

- Type: numeric float.
- Hard rules:
1. Required column.
2. Coerced with numeric parsing.
3. Non-numeric/null after coercion causes row drop.
- Semantic contract:
1. Declination in decimal degrees, expected ICRS/J2000.
2. Quality expectation (recommended): value in `[-90, 90]`.

### 5.7 `constellation`

- Type: string.
- Hard rules:
1. Optional column, created if absent.
2. Trimmed.
- Semantic contract:
1. Constellation association.
2. Empty string allowed.

### 5.8 `aliases`

- Type: string.
- Hard rules:
1. Optional column, created if absent.
2. Trimmed.
- Semantic contract:
1. Semicolon-delimited alias list.
2. Expected delimiter is `;` between entries.
3. Entries should not repeat `primary_id`.
4. Merge operations should preserve order and de-duplicate.
5. Empty string allowed.

### 5.9 `object_type_group`

- Type: string.
- Hard rules:
1. Optional column, created if absent.
2. Trimmed.
- Semantic contract:
1. Coarser grouping used by recommendations/search filtering.
2. Derived via mapping rules keyed by normalized `object_type`.
3. Default fallback group is `other` when no mapping matches.
4. Current mapping file group labels include `Galaxies`, `Clusters`, `Bright Nebula`, `Dark Nebula`, `Stars`, plus fallback `other`.

### 5.10 `image_url`

- Type: string.
- Hard rules:
1. Optional column, created if absent.
2. Trimmed.
- Semantic contract:
1. Direct image URL for object thumbnail/detail.
2. Empty string allowed.

### 5.11 `image_attribution_url`

- Type: string.
- Hard rules:
1. Optional column, created if absent.
2. Trimmed.
- Semantic contract:
1. URL for image source/attribution/license context.
2. Empty string allowed.

### 5.12 `license_label`

- Type: string.
- Hard rules:
1. Optional column, created if absent.
2. Trimmed.
- Semantic contract:
1. Human-readable license or credit label for `image_url`.
2. Empty string allowed.

### 5.13 `description`

- Type: string.
- Hard rules:
1. Enriched optional column, created if absent.
2. Trimmed.
- Semantic contract:
1. Free-text object description.
2. Can have appended SIMBAD type detail text.
3. Empty string allowed.

### 5.14 `info_url`

- Type: string.
- Hard rules:
1. Enriched optional column, created if absent.
2. Trimmed.
- Semantic contract:
1. Primary background/reference URL for object metadata.
2. Empty string allowed.

### 5.15 `dist_value`

- Type: numeric float.
- Hard rules:
1. Enriched optional column, created if absent.
2. Numeric coercion with invalids set to null/NaN.
- Semantic contract:
1. Distance numeric component paired with `dist_unit`.
2. Null/NaN allowed.

### 5.16 `dist_unit`

- Type: string.
- Hard rules:
1. Enriched optional column, created if absent.
2. Trimmed.
- Semantic contract:
1. Unit corresponding to `dist_value`.
2. Expected examples: `ly`, `pc`, `kpc`, `Mpc`.
3. Empty string allowed.

### 5.17 `redshift`

- Type: numeric float.
- Hard rules:
1. Enriched optional column, created if absent.
2. Numeric coercion with invalids set to null/NaN.
- Semantic contract:
1. Cosmological redshift value (`z`) when known.
2. Null/NaN allowed.

### 5.18 `ang_size_maj_arcmin`

- Type: numeric float.
- Hard rules:
1. Enriched optional column, created if absent.
2. Numeric coercion with invalids set to null/NaN.
- Semantic contract:
1. Apparent major-axis angular size in arcminutes.
2. Null/NaN allowed.

### 5.19 `ang_size_min_arcmin`

- Type: numeric float.
- Hard rules:
1. Enriched optional column, created if absent.
2. Numeric coercion with invalids set to null/NaN.
- Semantic contract:
1. Apparent minor-axis angular size in arcminutes.
2. Null/NaN allowed.

### 5.20 `morphology`

- Type: string.
- Hard rules:
1. Enriched optional column, created if absent.
2. Trimmed.
- Semantic contract:
1. Morphological classification text.
2. Empty string allowed.

### 5.21 `emission_lines`

- Type: string.
- Hard rules:
1. Enriched optional column, created if absent.
2. Trimmed.
- Semantic contract:
1. Emission feature list, stored as delimiter-separated text.
2. Expected delimiter from enrichment merge is `; `.
3. Merge operations should preserve order and de-duplicate.
4. Empty string allowed.

## 6. Source and Merge Precedence Expectations

When rebuilding from source data, expected precedence is:

1. Base frame from enriched CSV when available; otherwise OpenNGC; otherwise seed fallback.
2. Add SIMBAD named-object ingest rows.
3. Enrich M/NGC rows from SIMBAD reference where beneficial.
4. Merge popular-DSO supplement rows.
5. Apply object-type label mapping and object-type grouping.
6. Normalize and write Parquet.
7. Optionally apply Wikipedia enrichment merge, which updates existing rows by `primary_id` and can insert new rows only when both coordinates are present.

When merging with an existing cache as supplement:

1. Existing rows in the base frame are authoritative by `primary_id`.
2. Cache rows are appended only for IDs absent from the base frame.

## 7. Standalone Project Acceptance Checks

A standalone cache-builder project should fail the build if any of the following checks fail:

1. Column list exactly matches Section 2 (name + order).
2. All required columns exist.
3. `primary_id` has no blanks.
4. `primary_id` has no duplicates.
5. `ra_deg` and `dec_deg` are numeric and non-null for all rows.
6. `catalog` is uppercase and trimmed for all rows.
7. Output is sorted by `catalog`, `primary_id`.
8. Parquet is written without index.

Recommended quality warnings (non-fatal but should be reported):


1. `ra_deg` outside `[0, 360)` or `dec_deg` outside `[-90, 90]`.
2. URL fields present but not absolute HTTP(S) URLs.
3. `object_type_group` not in known mapped groups and not `other`.

## 8. Code References (Normative)

- Schema lists and ingestion constants: `catalog_ingestion.py:15`, `catalog_ingestion.py:24`, `catalog_ingestion.py:33`.
- Normalization and hard row filters: `catalog_ingestion.py:311`.
- Object type grouping and default fallback: `catalog_ingestion.py:186`, `catalog_ingestion.py:262`.
- ID canonicalization behavior: `catalog_ingestion.py:397`, `catalog_ingestion.py:521`.
- Ingestion mappings for enriched/OpenNGC/popular supplement: `catalog_ingestion.py:906`, `catalog_ingestion.py:989`, `catalog_ingestion.py:1073`.
- Merge precedence with cache-only additions: `catalog_ingestion.py:1150`.
- Unified write path (`to_parquet`): `catalog_ingestion.py:1367`.
- Wikipedia merge constraints and insert/update behavior: `scripts/apply_wikipedia_catalog_enrichment.py:20`, `scripts/apply_wikipedia_catalog_enrichment.py:244`, `scripts/apply_wikipedia_catalog_enrichment.py:279`, `scripts/apply_wikipedia_catalog_enrichment.py:325`, `scripts/apply_wikipedia_catalog_enrichment.py:384`.
