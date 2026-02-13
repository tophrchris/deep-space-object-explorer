import unittest

from dso_enricher.normalization import (
    catalog_ids_from_values,
    clean_common_name,
    normalize_identifier,
    normalize_object_type,
    split_aliases,
)


class NormalizationTests(unittest.TestCase):
    def test_identifier_normalization(self) -> None:
        self.assertEqual(normalize_identifier("M101"), ("M 101", "messier"))
        self.assertEqual(normalize_identifier("NGC1976"), ("NGC 1976", "ngc"))
        self.assertEqual(normalize_identifier("IC5146"), ("IC 5146", "ic"))
        self.assertEqual(normalize_identifier("C38"), ("C 38", "caldwell"))
        self.assertEqual(normalize_identifier("Sh 2-155"), ("SH2-155", "survey"))

    def test_common_name_cleanup(self) -> None:
        self.assertEqual(clean_common_name("NAME Crab Nebula"), "Crab Nebula")
        self.assertEqual(clean_common_name("  Orion Nebula "), "Orion Nebula")

    def test_alias_split(self) -> None:
        aliases = split_aliases("Messier 1;NGC 1952;NAME Crab")
        self.assertEqual(aliases, ["Messier 1", "NGC 1952", "Crab"])

    def test_catalog_id_extraction(self) -> None:
        ids = catalog_ids_from_values(["M1", "NGC 1952", "IC 0"])
        self.assertEqual(ids["messier_id"], "M 1")
        self.assertEqual(ids["ngc_id"], "NGC 1952")

    def test_object_type_mapping(self) -> None:
        self.assertEqual(normalize_object_type("Spiral Galaxy"), "galaxy")
        self.assertEqual(normalize_object_type("Open Cluster"), "open_cluster")
        self.assertEqual(normalize_object_type("Planetary Nebula"), "planetary_nebula")


if __name__ == "__main__":
    unittest.main()
