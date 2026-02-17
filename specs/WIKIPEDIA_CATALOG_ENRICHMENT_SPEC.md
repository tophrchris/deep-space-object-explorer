# `build_wikipedia_catalog_enrichment.py` Specification

Status: normative for `scripts/build_wikipedia_catalog_enrichment.py`.

This document specifies behavior for building `wikipedia_catalog_enrichment.json` from catalog targets in
`dso_catalog_cache.parquet`.

## 1. Purpose

Generate a standalone enrichment JSON payload that can later be merged into the Parquet catalog by
`scripts/apply_wikipedia_catalog_enrichment.py`.

Primary responsibilities:

1. Select target rows from the catalog.
2. Resolve best Wikipedia page match per target.
3. Extract structured enrichment fields from summary/wikitext/image metadata.
4. Persist deterministic JSON output and HTTP cache.
5. Support resumable incremental runs.

## 2. Script and Runtime

- Script: `scripts/build_wikipedia_catalog_enrichment.py`
- Entrypoint: `main()`
- Language/runtime: Python 3
- Core dependency: `pandas` (for reading Parquet and frame filtering)
- HTTP transport dependency: `curl` CLI via `subprocess.run(...)` (not `requests`)

If `curl` is unavailable or returns non-zero, HTTP calls fail and are counted in `http_failures`.

## 3. External Endpoints

The script calls:

1. Wikipedia REST summary endpoint:
   - `https://en.wikipedia.org/api/rest_v1/page/summary/{title}`
2. Wikipedia API endpoint:
   - `https://en.wikipedia.org/w/api.php`

API methods used:

1. `list=search`
2. `prop=extracts` (`exintro`, plaintext)
3. `prop=pageimages` (`name|original`)
4. `prop=imageinfo` (`iiprop=url`)
5. `action=parse` (`prop=wikitext`)

## 4. CLI Contract

Arguments and defaults:

1. `--catalog-path` (default `data/dso_catalog_cache.parquet`)
2. `--output-path` (default `data/wikipedia_catalog_enrichment.json`)
3. `--cache-path` (default `data/wikipedia_api_cache.json`)
4. `--include-catalogs` (default `M,SH2`)
5. `--include-groups` (default `bright nebula,dark nebula`)
6. `--include-primary-ids` (default empty)
7. `--limit` (default `0`, means no limit)
8. `--requests-per-second` (default `4.0`)
9. `--timeout-s` (default `10.0`)
10. `--resume` (default enabled)
11. `--no-resume` (disables resume)
12. `--retry-no-match` (reprocess rows previously `no_match` when resuming)
13. `--checkpoint-every` (default `25`, `0` disables checkpoint writes)
14. `--progress-every` (default `10`)
15. `--verbosity` (default `2`, levels `0-4`)

## 5. Inputs

### 5.1 Catalog Parquet Input

`--catalog-path` must be a readable Parquet file.

Hard requirement:

1. Must contain column `primary_id`; otherwise the run raises `ValueError`.

Optional-but-used columns:

1. `catalog`
2. `object_type_group`
3. `common_name`
4. `aliases`

Missing optional columns are treated as empty strings during selection/matching logic.

### 5.2 Existing Output (Resume)

If resume is enabled and output exists, script attempts to load `targets[]` rows keyed by `primary_id`.

Resume loader behavior:

1. Invalid JSON/object shape is logged and ignored.
2. Non-dict target rows are discarded.
3. Rows without `primary_id` are discarded.
4. Duplicate `primary_id` rows keep first occurrence.

## 6. HTTP Cache Contract

Cache file (`--cache-path`) stores response entries keyed by synthesized request keys, for example:

1. `summary::<title_lower>`
2. `search::<query_lower>::<limit>`
3. `extract::<title_lower>`
4. `pageimage_details::<title_lower>`
5. `imageinfo::<file_title_lower>`
6. `wikitext::<title_lower>`

Cache entry shape:

1. Success: `{"ok": true, "payload": {...}}`
2. Failure: failures are returned at runtime (`{"ok": false, "error": ...}`) but are not persisted into cache.

Client metrics tracked:

1. `cache_hits`
2. `http_requests`
3. `http_successes`
4. `http_failures`
5. `rate_limit_sleeps`

## 7. Target Selection Rules

Targets are selected from the catalog where:

1. `catalog` (uppercased) is in `--include-catalogs`, OR
2. `object_type_group` (lowercased) is in `--include-groups`, OR
3. `primary_id` normalized key is in `--include-primary-ids`.

Primary-ID includes are canonicalized before matching:

1. Messier -> `M<number><suffix?>`
2. NGC -> `NGC <number> <suffix?>`
3. IC -> `IC <number> <suffix?>`
4. Sharpless -> `Sh2-<number><suffix?>`

If requested primary IDs are not present in selected rows, synthetic rows are created:

1. `primary_id` set to requested canonical ID
2. `catalog` inferred from ID family (fallback `WIKI`)
3. numeric columns set `NaN`, string columns set `""`

Final target frame is:

1. `primary_id` trimmed
2. blank IDs removed
3. deduped by `primary_id` (keep first)
4. sorted by `catalog`, `primary_id`

## 8. Resolver Pipeline

Per target, `_resolve_wikipedia_page(...)` performs:

1. Build identity tokens from `primary_id`, `common_name`, and aliases.
2. Build direct title candidates.
3. Build search queries.
4. Classify target preference (`nebula` preference heuristic).

### 8.1 Direct Title Pass

Tries first 6 title candidates via summary endpoint.

Direct score formula:

1. `score = _score_summary(...) + 15 - (candidate_index * 2)`

Early accept when all true:

1. `score >= 80`
2. not disambiguation
3. if target prefers nebula, topic label is `nebula`

### 8.2 Search Pass

Triggered when direct pass insufficient (`best < 80`, with nebula preference override).

For each of first 5 search queries:

1. fetch up to 8 search results
2. rank by `_score_search_result(...)`
3. evaluate top result summary
4. compute combined score:
   - `_score_summary(...) + _score_search_result(...)`

Messier shorthand fallback:

1. If query is `M <n>` style and top result does not look like a Messier object page,
2. retry query `Messier <n>` and re-score.

Search early accept when all true:

1. `score >= 85`
2. not disambiguation
3. if target prefers nebula, topic label is `nebula`

### 8.3 Final Candidate Selection

Selection logic:

1. default selected = overall best
2. if target prefers nebula and best nebula score >= 30, select nebula-best instead
3. if selected score < 40 or no title, return `no_match`

## 9. Content Extraction Rules

After a page is selected:

1. Description:
   - Start with summary extract.
   - If summary extract has fewer than 3 sentences, supplement with intro extract API.
   - Final text capped to at most 10 sentences.
2. Info URL:
   - Prefer summary content URL, fallback wiki URL by title.
3. Image URL:
   - Prefer summary image (`originalimage` then `thumbnail`).
   - Fallback to `pageimages` original source.
4. Image attribution URL:
   - Prefer `pageimages` file attribution (`imageinfo`).
   - Fallback from derived file title.
   - Final fallback to info URL.
5. Wikitext parsing:
   - Extract infobox block by brace-depth parse.
   - Parse infobox fields into normalized key/value entries.
6. Structured fields from infobox:
   - `common_name`
   - `object_type`
   - `ra_deg`
   - `dec_deg`
   - `constellation`
   - `emission_lines`
   - `ang_size_maj_arcmin`
   - `ang_size_min_arcmin`
7. Designations:
   - Extract from infobox keys matching designation hints, excluding known non-designation keys.
8. Aliases:
   - de-duplicated union of designations, extracted common name, stripped page title
   - remove aliases equal to catalog `primary_id` after normalization.
9. Wikipedia primary reference:
   - infer `wikipedia_primary_id` + `wikipedia_catalog` from title/aliases/designations by first catalog-ID match.

Special redirect note:

When target prefers nebula but resolved page looks star-like and was redirected from a different search term,
append relation sentence to description and include selected title in designations.

## 10. Numeric Parsing Heuristics

### 10.1 RA

Accepted patterns (from raw and cleaned infobox values):

1. degrees (contains `deg`/`°`)
2. HMS triplet (h, m, s -> converted to degrees *15)
3. 2-part hour/minute
4. single value interpreted as hours if `<=24`, else degrees if `<=360`

Normalized into `[0, 360)` when accepted.

### 10.2 DEC

Accepted patterns:

1. degrees with sign
2. DMS triplet
3. 2-part degree/minute
4. signed single value

Accepted range is `[-90, 90]`.

### 10.3 Apparent Size

Sources: infobox keys matching angular-size hints with exclusion filters.

Unit handling:

1. arcsec -> multiply by `1/60`
2. arcmin -> multiply by `1`
3. deg -> multiply by `60`
4. unitless only allowed for specific key classes (`angular`, `apparent`, `dimensions`, etc.)

Sanitized accepted range: `[0.01, 5000.0]` arcmin.
If both major/minor exist and `minor > major`, values are swapped.

## 11. Output JSON Specification

Top-level object:

1. `generated_at_utc` (ISO-8601 UTC string)
2. `catalog_path` (string path)
3. `selection` (object)
4. `run` (object)
5. `stats` (object)
6. `http` (object)
7. `targets` (array)

`selection`:

1. `catalogs`: `string[]`
2. `object_type_groups`: `string[]`
3. `primary_ids`: `string[]`
4. `target_count`: `int`

`run`:

1. `resume_enabled`: `bool`
2. `resume_retry_no_match`: `bool`
3. `resume_rows_reused`: `int`
4. `rows_processed_this_run`: `int`
5. `duration_seconds`: `float` (rounded to 3 decimals)
6. `verbosity`: `int`

`stats`:

1. `matched`
2. `no_match`
3. `unknown_status`
4. `with_image`
5. `with_designations`
6. `with_3plus_sentence_description`

All `stats` values are integers.

`http`:

1. `cache_hits`
2. `http_requests`
3. `http_successes`
4. `http_failures`
5. `rate_limit_sleeps`

All `http` values are integers.

Per `targets[]` row:

1. `primary_id`: `string`
2. `catalog`: `string`
3. `object_type_group`: `string`
4. `common_name`: `string`
5. `status`: `"matched" | "no_match"`
6. `match_method`: `string` (`direct_title`, `search`, `no_match`, `none`)
7. `match_score`: `int`
8. `query_used`: `string`
9. `wikipedia_title`: `string`
10. `wikipedia_url`: `string`
11. `description`: `string`
12. `description_sentence_count`: `int`
13. `image_url`: `string`
14. `designations`: `string[]`
15. `catalog_enrichment`: object (see below)

`catalog_enrichment` required keys:

1. `common_name`: `string`
2. `object_type`: `string`
3. `ra_deg`: `number | null`
4. `dec_deg`: `number | null`
5. `ang_size_maj_arcmin`: `number | null`
6. `ang_size_min_arcmin`: `number | null`
7. `constellation`: `string`
8. `aliases`: `string[]`
9. `image_url`: `string`
10. `image_attribution_url`: `string`
11. `description`: `string`
12. `info_url`: `string`
13. `emission_lines`: `string[]`
14. `wikipedia_primary_id`: `string`
15. `wikipedia_catalog`: `string`

## 12. Resume and Reuse Rules

For each target row in a resume run:

1. If prior row exists and has complete `catalog_enrichment` keys:
   - reuse without reprocessing.
2. If `--retry-no-match` and prior `status == no_match`:
   - reprocess.
3. If prior row missing required `catalog_enrichment` keys:
   - reprocess.

Rows reused are counted in `resume_rows_reused`.
Rows newly processed this invocation are counted in `rows_processed_this_run`.

## 13. Checkpoint and Finalization

Checkpoint behavior:

1. Triggered every `N` newly processed rows (`--checkpoint-every`).
2. Writes interim output JSON to `--output-path`.
3. Flushes HTTP cache JSON to `--cache-path`.

Finalization:

1. Recompute final stats.
2. Write final payload JSON.
3. Flush HTTP cache.
4. Emit completion logs with summary and HTTP stats.

Even with zero selected targets, script writes a valid empty payload.

## 14. Downstream Merge Contract

This script is the producer for `scripts/apply_wikipedia_catalog_enrichment.py`.

Consumer expectations:

1. Payload must have `targets` array.
2. Per-row `catalog_enrichment` should include all keys in Section 11.
3. Existing catalog rows are updated by `primary_id`.
4. New rows may be inserted only when both `ra_deg` and `dec_deg` are present.
5. Alias and emission lists are merged with de-duplication.

## 15. Failure Modes

Hard failures (raise):

1. Invalid/missing catalog file unreadable by `pandas.read_parquet`.
2. Missing required input column `primary_id`.

Soft failures (continue with degraded result):

1. HTTP call failures -> resolver may return `no_match`.
2. Missing summary/search/wikitext/image content -> partial enrichment values.
3. Invalid existing output JSON in resume mode -> ignored with log.

## 16. Operational Notes

1. Defaults are intentionally narrow (`M`, `SH2`, nebula groups) to focus enrichment volume.
2. Resolver scoring is heuristic and may change page choice with minor token changes.
3. Output JSON is written with `ensure_ascii=True`, stable key ordering only in cache file (not payload).

## 17. Normative Code References

- Constants/endpoints/default selectors: `scripts/build_wikipedia_catalog_enrichment.py:19`
- Wikitext cleaning and infobox parsing: `scripts/build_wikipedia_catalog_enrichment.py:325`
- RA/DEC and angular-size extraction: `scripts/build_wikipedia_catalog_enrichment.py:507`, `scripts/build_wikipedia_catalog_enrichment.py:540`, `scripts/build_wikipedia_catalog_enrichment.py:800`
- Identifier canonicalization and query construction: `scripts/build_wikipedia_catalog_enrichment.py:865`, `scripts/build_wikipedia_catalog_enrichment.py:987`, `scripts/build_wikipedia_catalog_enrichment.py:1026`
- Scoring functions: `scripts/build_wikipedia_catalog_enrichment.py:1089`, `scripts/build_wikipedia_catalog_enrichment.py:1118`
- Lookup/result data model: `scripts/build_wikipedia_catalog_enrichment.py:1148`
- HTTP client caching/rate-limit/retry: `scripts/build_wikipedia_catalog_enrichment.py:1185`
- Resolver control flow and thresholds: `scripts/build_wikipedia_catalog_enrichment.py:1516`
- Target selection: `scripts/build_wikipedia_catalog_enrichment.py:1838`
- Output payload construction: `scripts/build_wikipedia_catalog_enrichment.py:2065`
- CLI arguments: `scripts/build_wikipedia_catalog_enrichment.py:2106`
- Main run loop (resume/checkpoint/final write): `scripts/build_wikipedia_catalog_enrichment.py:2209`
- Downstream consumer contract: `scripts/apply_wikipedia_catalog_enrichment.py:107`, `scripts/apply_wikipedia_catalog_enrichment.py:240`
