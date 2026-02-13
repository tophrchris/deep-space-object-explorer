from __future__ import annotations

import argparse
import json
from pathlib import Path

from dso_enricher.enricher import EnrichmentPipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dso-enricher",
        description="Enrich Deep Sky Object CSV rows with cross-catalog metadata.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input CSV path (use multiple times for multiple files).",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=100,
        help="Max rows to process from each input file (default: 100).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for enriched and review CSVs.",
    )
    parser.add_argument(
        "--enriched-filename",
        default="enriched.csv",
        help="Output filename for enriched rows.",
    )
    parser.add_argument(
        "--review-queue-filename",
        default="ambiguous_review_queue.csv",
        help="Output filename for ambiguous-review queue rows.",
    )
    parser.add_argument(
        "--cache-path",
        default="cache/source_cache.json",
        help="Cache path for remote lookups.",
    )
    parser.add_argument(
        "--disable-remote",
        action="store_true",
        help="Disable Sesame/SIMBAD/NED/VizieR lookups (offline mode).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=8.0,
        help="HTTP timeout seconds per remote request.",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=5.0,
        help="Rate limit for each remote source client.",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=8,
        help="Worker count for parallel prefetch lookups.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_paths = [Path(value) for value in args.input]
    for input_path in input_paths:
        if not input_path.exists():
            parser.error(f"Input file does not exist: {input_path}")

    config = PipelineConfig(
        max_rows_per_file=args.max_rows_per_file,
        output_dir=Path(args.output_dir),
        enriched_filename=args.enriched_filename,
        review_queue_filename=args.review_queue_filename,
        disable_remote=args.disable_remote,
        timeout_s=args.timeout_s,
        requests_per_second=args.requests_per_second,
        prefetch_workers=args.prefetch_workers,
        cache_path=Path(args.cache_path),
    )

    pipeline = EnrichmentPipeline(config)
    summary = pipeline.run(input_paths)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0
