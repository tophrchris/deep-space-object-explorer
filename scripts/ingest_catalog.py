#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catalog_ingestion import load_unified_catalog


def main() -> None:
    frame, metadata = load_unified_catalog(
        seed_path=Path("data/dso_catalog_seed.csv"),
        cache_path=Path("data/dso_catalog_cache.parquet"),
        metadata_path=Path("data/dso_catalog_cache_meta.json"),
        force_refresh=True,
    )

    counts = metadata.get("catalog_counts", {})
    counts_str = " | ".join(f"{key}: {value}" for key, value in sorted(counts.items()))

    print(f"Ingested {len(frame)} targets")
    if counts_str:
        print(f"Catalog counts: {counts_str}")
    print(f"Cache written: {metadata.get('cache', 'data/dso_catalog_cache.parquet')}")


if __name__ == "__main__":
    main()
