import json
import unittest

from dso_enricher.schema import ALL_COLUMNS, CORE_SCHEMA_COLUMNS, new_blank_row, serialize_row


class SchemaTests(unittest.TestCase):
    def test_core_schema_contains_required_fields(self) -> None:
        required = {
            "row_id",
            "id_raw",
            "id_norm",
            "match_status",
            "simbad_main_id",
            "hero_image_url",
            "links",
            "field_provenance",
            "qc_flags",
        }
        self.assertTrue(required.issubset(set(CORE_SCHEMA_COLUMNS)))

    def test_blank_row_has_all_columns(self) -> None:
        row = new_blank_row()
        self.assertEqual(set(row.keys()), set(ALL_COLUMNS))
        self.assertIsInstance(row["field_provenance"], dict)
        self.assertIsInstance(row["links"], list)
        self.assertIsInstance(row["qc_flags"], list)

    def test_serialize_json_fields(self) -> None:
        row = new_blank_row()
        row["cross_ids"] = ["M 1", "NGC 1952"]
        row["field_provenance"] = {"cross_ids": {"src": "input_csv", "ts": "2026-01-01T00:00:00+00:00"}}
        serialized = serialize_row(row)
        self.assertIsInstance(serialized["cross_ids"], str)
        self.assertEqual(json.loads(serialized["cross_ids"]), ["M 1", "NGC 1952"])
        self.assertIsInstance(serialized["field_provenance"], str)
        self.assertEqual(
            json.loads(serialized["field_provenance"])["cross_ids"]["src"],
            "input_csv",
        )


if __name__ == "__main__":
    unittest.main()
