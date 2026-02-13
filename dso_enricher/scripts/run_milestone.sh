#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH=dso_enricher/src python3 -m dso_enricher \
  --input "data/dso_catalog_cache_ngc_ic_filtered.csv" \
  --input "data/dso_catalog_cache_messier_sharpless.csv" \
  --max-rows-per-file 100 \
  --output-dir dso_enricher/output \
  --cache-path dso_enricher/cache/source_cache.json \
  "$@"
