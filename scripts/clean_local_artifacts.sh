#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name ".DS_Store" -delete

rm -rf .pycache_tmp
rm -f data/dso_catalog_cache_messier_sharpless.csv
rm -f data/dso_catalog_cache_ngc_ic.csv
rm -f data/dso_catalog_cache_ngc_ic_filtered.csv
rm -f data/wikipedia_api_cache.json
rm -f data/wikipedia_catalog_enrichment.json
rm -f data/wikipedia_catalog_enrichment.json.bak2.json

echo "Local artifacts cleaned."
