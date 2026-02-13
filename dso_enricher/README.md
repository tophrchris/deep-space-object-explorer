# DSO Enricher

`dso_enricher` is a standalone Python pipeline for enriching Deep Sky Object CSV rows.

It is intentionally self-contained so extraction is simple: copy the `dso_enricher/`
directory to a new repository and update input/output paths.

## Scope

- Identifier normalization and catalog family inference.
- Resolver chain support: `Sesame -> SIMBAD -> NED -> VizieR`.
- Near-full enriched schema from the DSO enrichment report.
- Per-field provenance tracking and QC flags.
- Ambiguous rows written to a dedicated review queue CSV.
- URL-only media handling (no image/PDF downloads).

## Project Layout

```text
dso_enricher/
  requirements.txt
  src/dso_enricher/
    cli.py
    enricher.py
    normalization.py
    schema.py
    sources.py
  tests/
```

## Setup

From repository root:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r dso_enricher/requirements.txt
```

`requirements.txt` is intentionally minimal; baseline pipeline uses only stdlib.

## Run Milestone (100 rows from each sample file)

From repository root:

```bash
PYTHONPATH=dso_enricher/src python3 -m dso_enricher \
  --input "data/dso_catalog_cache_ngc_ic_filtered.csv" \
  --input "data/dso_catalog_cache_messier_sharpless.csv" \
  --max-rows-per-file 100 \
  --output-dir dso_enricher/output \
  --prefetch-workers 8
```

Outputs:

- `dso_enricher/output/enriched.csv`
- `dso_enricher/output/ambiguous_review_queue.csv`

Shortcut script:

```bash
./dso_enricher/scripts/run_milestone.sh --disable-remote
```

Note: if an input file has fewer than `--max-rows-per-file`, all available rows are processed.

Performance knobs:

- `--prefetch-workers` controls parallel source prefetch for unique identifiers.
- `--requests-per-second` controls per-source rate limiting.
- `--timeout-s` sets per-request timeout.

## Notes

- Remote queries are best-effort and fail open; rows are still emitted.
- Use `--disable-remote` for fully offline processing.
- `hero_image_url` is normalized to a direct image URL (`jpg/png`) instead of a webpage URL.
- `links` is curated to object info pages (e.g., SIMBAD/NED/NASA Messier), not direct image assets.
