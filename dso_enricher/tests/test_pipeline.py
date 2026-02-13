from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from dso_enricher.enricher import EnrichmentPipeline, PipelineConfig
from dso_enricher.schema import new_blank_row


class PipelineTests(unittest.TestCase):
    def test_disable_remote_derivations_populate_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "input.csv"
            output_dir = root / "output"
            cache_path = root / "cache.json"
            self._write_input(
                input_path,
                [
                    {
                        "primary_id": "M1",
                        "catalog": "M",
                        "common_name": "Crab Nebula",
                        "object_type": "Supernova Remnant",
                        "ra_deg": "83.6331",
                        "dec_deg": "22.0145",
                        "constellation": "Taurus",
                        "aliases": "Messier 1;NGC 1952;NAME Crab",
                        "image_url": "",
                        "image_attribution_url": "",
                        "license_label": "",
                    }
                ],
            )

            pipeline = EnrichmentPipeline(
                PipelineConfig(
                    max_rows_per_file=100,
                    output_dir=output_dir,
                    cache_path=cache_path,
                    disable_remote=True,
                )
            )
            summary = pipeline.run([input_path])
            self.assertEqual(summary["rows_processed"], 1)

            with (output_dir / "enriched.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))

            self.assertEqual(row["messier_id"], "M 1")
            self.assertEqual(row["ngc_id"], "NGC 1952")
            self.assertNotEqual(row["hero_image_url"], "")
            self.assertTrue(
                row["hero_image_url"].lower().endswith((".jpg", ".jpeg", ".png"))
                or "format=jpg" in row["hero_image_url"].lower()
                or "format=png" in row["hero_image_url"].lower()
            )
            self.assertIn("simbad", row["links"].lower())
            self.assertNotIn("hips2fits", row["links"].lower())
            self.assertIn("H-alpha", row["emission_lines"])
            self.assertIn("cross-catalog", row["notable_features"])

    def test_redshift_derives_distance_when_missing(self) -> None:
        pipeline = EnrichmentPipeline(PipelineConfig(disable_remote=True))
        row = new_blank_row()
        row["redshift"] = 0.01
        provenance = row["field_provenance"]

        pipeline._derive_distance_and_velocity(row, provenance)

        self.assertIsNotNone(row["radial_velocity_kms"])
        self.assertIsNotNone(row["dist_value"])
        self.assertEqual(row["dist_unit"], "Mpc")
        self.assertEqual(row["dist_method"], "hubble_law_approx")

    def _write_input(self, path: Path, rows: list[dict[str, str]]) -> None:
        columns = [
            "primary_id",
            "catalog",
            "common_name",
            "object_type",
            "ra_deg",
            "dec_deg",
            "constellation",
            "aliases",
            "image_url",
            "image_attribution_url",
            "license_label",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


if __name__ == "__main__":
    unittest.main()
