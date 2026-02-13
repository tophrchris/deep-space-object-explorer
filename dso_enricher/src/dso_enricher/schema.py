from __future__ import annotations

import json
from typing import Any


CORE_SCHEMA_COLUMNS = [
    "row_id",
    "id_raw",
    "id_norm",
    "id_family_guess",
    "match_status",
    "match_method",
    "match_score",
    "simbad_main_id",
    "simbad_object_url",
    "cross_ids",
    "messier_id",
    "ngc_id",
    "ic_id",
    "caldwell_id",
    "common_name",
    "object_type_simbad",
    "object_type_norm",
    "ra_j2000_deg",
    "dec_j2000_deg",
    "ra_j2000_hms",
    "dec_j2000_dms",
    "coord_ref_bibcode",
    "coord_quality",
    "coord_waveband",
    "constellation",
    "dist_value",
    "dist_unit",
    "dist_err_minus",
    "dist_err_plus",
    "dist_method",
    "redshift",
    "radial_velocity_kms",
    "mag_v",
    "mag_band",
    "mag_system",
    "ang_size_maj_arcmin",
    "ang_size_min_arcmin",
    "pa_deg",
    "surface_brightness_mag_arcsec2",
    "sb_band",
    "emission_lines",
    "multiwave_flags",
    "spectral_type",
    "morphology",
    "discovery_by",
    "discovery_year",
    "notable_features",
    "description",
    "hero_image_url",
    "hero_image_credit",
    "hero_image_filters",
    "links",
    "field_provenance",
    "qc_flags",
]

# Added for traceability to simplify queue triage and later extraction.
EXTRA_COLUMNS = ["source_file", "source_row_index"]

ALL_COLUMNS = EXTRA_COLUMNS + CORE_SCHEMA_COLUMNS

JSON_COLUMNS = {
    "cross_ids",
    "emission_lines",
    "multiwave_flags",
    "hero_image_filters",
    "links",
    "field_provenance",
    "qc_flags",
}


def new_blank_row() -> dict[str, Any]:
    row: dict[str, Any] = {column: None for column in ALL_COLUMNS}
    for column in JSON_COLUMNS:
        row[column] = [] if column != "field_provenance" else {}
    return row


def serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for column in ALL_COLUMNS:
        value = row.get(column)
        if column in JSON_COLUMNS:
            if value in (None, "", []):
                serialized[column] = "" if column != "field_provenance" else "{}"
            elif column == "field_provenance" and isinstance(value, dict):
                serialized[column] = json.dumps(value, ensure_ascii=True, sort_keys=True)
            else:
                serialized[column] = json.dumps(value, ensure_ascii=True)
        else:
            serialized[column] = value
    return serialized
