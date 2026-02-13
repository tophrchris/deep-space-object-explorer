from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse

from dso_enricher.normalization import (
    catalog_ids_from_values,
    clean_common_name,
    dedupe_str_list,
    normalize_identifier,
    normalize_object_type,
    split_aliases,
)
from dso_enricher.schema import ALL_COLUMNS, new_blank_row, serialize_row
from dso_enricher.sources import JsonCache, NedClient, SesameClient, SimbadClient, VizierClient

C_KM_S = 299792.458
H0_KM_S_MPC = 70.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {})


def _composite_name_hint(identifier: str) -> bool:
    upper = identifier.upper()
    hints = (" GROUP", " TAIL", " PATCH", "A+B", " PAIR", " COMPLEX", " REGION")
    return any(token in upper for token in hints)


def _ra_deg_to_hms(ra_deg: float) -> str:
    total_seconds = (ra_deg / 15.0) * 3600.0
    hours = int(total_seconds // 3600.0)
    minutes = int((total_seconds % 3600.0) // 60.0)
    seconds = total_seconds % 60.0
    return f"{hours:02d} {minutes:02d} {seconds:06.3f}"


def _dec_deg_to_dms(dec_deg: float) -> str:
    sign = "+" if dec_deg >= 0 else "-"
    abs_deg = abs(dec_deg)
    degrees = int(abs_deg)
    minutes_full = (abs_deg - degrees) * 60.0
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60.0
    return f"{sign}{degrees:02d} {minutes:02d} {seconds:05.2f}"


def _is_direct_image_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    path = (parsed.path or "").lower()
    if path.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return True
    query = parse_qs(parsed.query)
    format_param = "".join(query.get("format", [])).lower()
    if format_param in {"jpg", "jpeg", "png", "webp"}:
        return True
    return False


def _maybe_convert_page_to_image_url(value: str | None) -> str | None:
    if not value:
        return None
    if _is_direct_image_url(value):
        return value

    parsed = urlparse(value)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").strip("/")
    parts = [part for part in path.split("/") if part]

    # ESA/Hubble page format: /images/<asset_id>/ -> CDN direct image
    if "esahubble.org" in host and len(parts) >= 2 and parts[0] == "images":
        asset_id = parts[1]
        return f"https://cdn.esahubble.org/archives/images/large/{asset_id}.jpg"

    return None


def _build_dss_cutout_url(ra_deg: float, dec_deg: float, fov_deg: float) -> str:
    return (
        "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
        f"?hips=CDS/P/DSS2/color&format=jpg&width=1200&height=1200"
        f"&fov={fov_deg:.4f}&projection=TAN&coordsys=icrs"
        f"&ra={ra_deg:.6f}&dec={dec_deg:.6f}"
    )


def _build_simbad_info_url(identifier: str) -> str:
    return f"https://simbad.u-strasbg.fr/simbad/sim-id?Ident={quote_plus(identifier)}"


def _build_ned_info_url(identifier: str) -> str:
    return f"https://ned.ipac.caltech.edu/byname?objname={quote_plus(identifier)}"


def _build_nasa_messier_page(messier_num: int) -> str:
    return (
        "https://science.nasa.gov/mission/hubble/science/explore-the-night-sky/"
        f"hubble-messier-catalog/messier-{messier_num}/"
    )


@dataclass
class PipelineConfig:
    max_rows_per_file: int = 100
    output_dir: Path = Path("output")
    enriched_filename: str = "enriched.csv"
    review_queue_filename: str = "ambiguous_review_queue.csv"
    disable_remote: bool = False
    timeout_s: float = 8.0
    requests_per_second: float = 5.0
    prefetch_workers: int = 8
    cache_path: Path = Path("cache/source_cache.json")


class EnrichmentPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.cache = JsonCache(config.cache_path)
        if not config.disable_remote:
            self.sesame = SesameClient(config.timeout_s, config.requests_per_second, self.cache)
            self.simbad = SimbadClient(config.timeout_s, config.requests_per_second, self.cache)
            self.ned = NedClient(config.timeout_s, config.requests_per_second, self.cache)
            self.vizier = VizierClient(config.timeout_s, config.requests_per_second, self.cache)
        else:
            self.sesame = None
            self.simbad = None
            self.ned = None
            self.vizier = None

    def run(self, input_paths: list[Path]) -> dict[str, Any]:
        enriched_rows: list[dict[str, Any]] = []
        review_rows: list[dict[str, Any]] = []
        pending_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
        per_file_rows: dict[str, int] = {}
        row_id = 1

        for input_path in input_paths:
            file_rows = self._read_limited_csv(input_path)
            per_file_rows[input_path.name] = len(file_rows)
            for source_row_index, source_row in file_rows:
                row = self._build_base_row(
                    row_id=row_id,
                    source_file=input_path.name,
                    source_row_index=source_row_index,
                    source_row=source_row,
                )
                pending_rows.append((row, {}))
                row_id += 1

        if not self.config.disable_remote:
            self._prefetch_remote([row for row, _context in pending_rows])
        else:
            for row, _context in pending_rows:
                self._append_qc_flag(row, "remote_enrichment_disabled")

        for row, source_context in pending_rows:
            if not self.config.disable_remote:
                self._enrich_with_remote_sources(row, source_context)
            self._apply_derived_fields(row)
            self._finalize_quality_flags(row)
            enriched_rows.append(serialize_row(row))

            if row.get("match_status") == "ambiguous":
                review_rows.append(
                    {
                        "row_id": row["row_id"],
                        "id_raw": row["id_raw"],
                        "id_norm": row["id_norm"],
                        "source_file": row["source_file"],
                        "source_row_index": row["source_row_index"],
                        "reason": ",".join(row.get("qc_flags", [])) if row.get("qc_flags") else "ambiguous",
                        "candidates": json.dumps(source_context.get("candidates", []), ensure_ascii=True),
                        "source_context": json.dumps(source_context, ensure_ascii=True, sort_keys=True),
                    }
                )

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        enriched_path = output_dir / self.config.enriched_filename
        review_path = output_dir / self.config.review_queue_filename

        self._write_csv(enriched_path, ALL_COLUMNS, enriched_rows)
        review_columns = [
            "row_id",
            "id_raw",
            "id_norm",
            "source_file",
            "source_row_index",
            "reason",
            "candidates",
            "source_context",
        ]
        self._write_csv(review_path, review_columns, review_rows)

        self.cache.flush()

        summary = {
            "rows_processed": len(enriched_rows),
            "ambiguous_rows": len(review_rows),
            "rows_processed_per_file": per_file_rows,
            "output_path": str(enriched_path),
            "review_queue_path": str(review_path),
        }
        return summary

    def _prefetch_remote(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        id_norms = dedupe_str_list([str(row.get("id_norm") or "") for row in rows if row.get("id_norm")])
        if self.sesame and id_norms:
            self.sesame.lookup_many(id_norms, max_workers=self.config.prefetch_workers)

        simbad_targets: list[str] = []
        for row in rows:
            identifier = str(row.get("id_norm") or "")
            if not identifier:
                continue
            target = identifier
            if self.sesame:
                sesame = self.sesame.resolve(identifier)
                if sesame.data and sesame.data.get("main_id"):
                    target = str(sesame.data.get("main_id"))
            simbad_targets.append(target)
        simbad_targets = dedupe_str_list(simbad_targets)
        if self.simbad and simbad_targets:
            self.simbad.lookup_many(simbad_targets, max_workers=self.config.prefetch_workers)

        ned_targets: list[str] = []
        vizier_targets: list[str] = []
        for row in rows:
            identifier = str(row.get("id_norm") or "")
            if not identifier:
                continue
            simbad_target = identifier
            if self.sesame:
                sesame = self.sesame.resolve(identifier)
                if sesame.data and sesame.data.get("main_id"):
                    simbad_target = str(sesame.data.get("main_id"))
            simbad_data: dict[str, Any] | None = None
            if self.simbad:
                simbad = self.simbad.lookup(simbad_target)
                if simbad.data:
                    simbad_data = simbad.data
                    simbad_target = str(simbad.data.get("main_id") or simbad_target)
            if self._is_extragalactic_candidate(row, simbad_data):
                ned_targets.append(simbad_target)
                vizier_targets.append(simbad_target)

        if self.ned and ned_targets:
            self.ned.lookup_many(dedupe_str_list(ned_targets), max_workers=self.config.prefetch_workers)
        if self.vizier and vizier_targets:
            self.vizier.lookup_many(dedupe_str_list(vizier_targets), max_workers=self.config.prefetch_workers)

    def _read_limited_csv(self, path: Path) -> list[tuple[int, dict[str, Any]]]:
        rows: list[tuple[int, dict[str, Any]]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                if index > self.config.max_rows_per_file:
                    break
                rows.append((index, row))
        return rows

    def _write_csv(self, path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column) for column in columns})

    def _build_base_row(
        self,
        row_id: int,
        source_file: str,
        source_row_index: int,
        source_row: dict[str, Any],
    ) -> dict[str, Any]:
        row = new_blank_row()
        provenance: dict[str, Any] = {}

        primary_id = str(source_row.get("primary_id", "") or "").strip()
        fallback_name = str(source_row.get("common_name", "") or "").strip()
        id_raw = primary_id or fallback_name
        id_norm, id_family = normalize_identifier(id_raw)

        row["source_file"] = source_file
        row["source_row_index"] = source_row_index
        row["row_id"] = row_id
        row["id_raw"] = id_raw
        row["id_norm"] = id_norm
        row["id_family_guess"] = id_family
        row["match_status"] = "matched" if id_norm else "unmatched"
        row["match_method"] = "input_seed"
        row["match_score"] = 0.35 if id_norm else 0.0

        if _composite_name_hint(id_norm):
            row["match_status"] = "ambiguous"
            row["match_score"] = 0.2
            self._append_qc_flag(row, "composite_identifier")

        self._set_field(row, provenance, "common_name", clean_common_name(fallback_name), "input_csv")
        self._set_field(row, provenance, "object_type_simbad", source_row.get("object_type"), "input_csv")
        self._set_field(
            row,
            provenance,
            "object_type_norm",
            normalize_object_type(source_row.get("object_type")),
            "derived",
        )
        self._set_field(row, provenance, "ra_j2000_deg", _safe_float(source_row.get("ra_deg")), "input_csv")
        self._set_field(row, provenance, "dec_j2000_deg", _safe_float(source_row.get("dec_deg")), "input_csv")
        self._set_field(row, provenance, "constellation", source_row.get("constellation"), "input_csv")
        self._set_field(row, provenance, "hero_image_url", source_row.get("image_url"), "input_csv")

        hero_credit_parts = [
            str(source_row.get("image_attribution_url", "") or "").strip(),
            str(source_row.get("license_label", "") or "").strip(),
        ]
        hero_credit = " | ".join([part for part in hero_credit_parts if part])
        self._set_field(row, provenance, "hero_image_credit", hero_credit, "input_csv")

        aliases = split_aliases(str(source_row.get("aliases", "") or ""))
        cross_ids = dedupe_str_list([id_norm, *aliases])
        row["cross_ids"] = cross_ids
        provenance["cross_ids"] = {"src": "input_csv", "ts": _utc_now_iso()}

        catalog_ids = catalog_ids_from_values(cross_ids)
        for field, value in catalog_ids.items():
            self._set_field(row, provenance, field, value, "derived")

        if _is_empty(row.get("hero_image_url")):
            row["hero_image_url"] = None
        row["links"] = []
        row["field_provenance"] = provenance
        return row

    def _enrich_with_remote_sources(self, row: dict[str, Any], source_context: dict[str, Any]) -> None:
        provenance: dict[str, Any] = row["field_provenance"]
        id_norm = str(row.get("id_norm") or "")
        source_context["candidates"] = []

        sesame = self.sesame.resolve(id_norm) if self.sesame else None
        if sesame:
            source_context["sesame"] = {"error": sesame.error, "ambiguous": sesame.ambiguous}
            if sesame.data:
                self._set_field(row, provenance, "simbad_main_id", sesame.data.get("main_id"), "sesame")
                self._set_field(row, provenance, "ra_j2000_deg", sesame.data.get("ra_deg"), "sesame")
                self._set_field(row, provenance, "dec_j2000_deg", sesame.data.get("dec_deg"), "sesame")
                row["cross_ids"] = dedupe_str_list(row.get("cross_ids", []) + sesame.data.get("aliases", []))
                provenance["cross_ids"] = {"src": "sesame", "ts": _utc_now_iso()}
                row["match_status"] = "matched" if row["match_status"] != "ambiguous" else "ambiguous"
                row["match_method"] = "sesame_name"
                row["match_score"] = max(float(row.get("match_score") or 0.0), 0.7)
            elif sesame.error:
                self._append_qc_flag(row, "sesame_lookup_failed")

        simbad_target = row.get("simbad_main_id") or id_norm
        simbad = self.simbad.lookup(str(simbad_target)) if self.simbad else None
        simbad_data: dict[str, Any] | None = None
        if simbad:
            source_context["simbad"] = {"error": simbad.error}
            if simbad.data:
                simbad_data = simbad.data
                self._set_field(row, provenance, "simbad_main_id", simbad.data.get("main_id"), "simbad", overwrite=True)
                self._set_field(
                    row,
                    provenance,
                    "object_type_simbad",
                    simbad.data.get("object_type"),
                    "simbad",
                    overwrite=True,
                )
                self._set_field(row, provenance, "ra_j2000_deg", simbad.data.get("ra_deg"), "simbad", overwrite=True)
                self._set_field(row, provenance, "dec_j2000_deg", simbad.data.get("dec_deg"), "simbad", overwrite=True)
                self._set_field(
                    row,
                    provenance,
                    "coord_ref_bibcode",
                    simbad.data.get("coord_ref_bibcode"),
                    "simbad",
                    overwrite=True,
                )
                self._set_field(
                    row,
                    provenance,
                    "coord_quality",
                    simbad.data.get("coord_quality"),
                    "simbad",
                    overwrite=True,
                )
                self._set_field(
                    row,
                    provenance,
                    "coord_waveband",
                    simbad.data.get("coord_waveband"),
                    "simbad",
                    overwrite=True,
                )
                self._set_field(row, provenance, "redshift", simbad.data.get("redshift"), "simbad")
                self._set_field(
                    row,
                    provenance,
                    "radial_velocity_kms",
                    simbad.data.get("radial_velocity_kms"),
                    "simbad",
                )
                self._set_field(row, provenance, "spectral_type", simbad.data.get("spectral_type"), "simbad")
                self._set_field(row, provenance, "ang_size_maj_arcmin", simbad.data.get("ang_size_maj_arcmin"), "simbad")
                self._set_field(row, provenance, "ang_size_min_arcmin", simbad.data.get("ang_size_min_arcmin"), "simbad")
                self._set_field(row, provenance, "pa_deg", simbad.data.get("pa_deg"), "simbad")
                self._set_field(row, provenance, "mag_v", simbad.data.get("mag_v"), "simbad")
                if row.get("mag_v") is not None and _is_empty(row.get("mag_band")):
                    self._set_field(row, provenance, "mag_band", "V", "derived")
                if row.get("mag_v") is not None and _is_empty(row.get("mag_system")):
                    self._set_field(row, provenance, "mag_system", "unknown", "derived")

                row["cross_ids"] = dedupe_str_list(row.get("cross_ids", []) + simbad.data.get("cross_ids", []))
                provenance["cross_ids"] = {"src": "simbad", "ts": _utc_now_iso()}
                row["match_status"] = "matched" if row["match_status"] != "ambiguous" else "ambiguous"
                row["match_method"] = "simbad_id"
                row["match_score"] = max(float(row.get("match_score") or 0.0), 0.95)
            elif simbad.error:
                self._append_qc_flag(row, "simbad_lookup_failed")

            simbad_link = _build_simbad_info_url(str(row.get("simbad_main_id") or id_norm))
            self._set_field(row, provenance, "simbad_object_url", simbad_link, "simbad", overwrite=True)
            self._append_link(row, simbad_link)

        if self._is_extragalactic_candidate(row, simbad_data):
            ned_target = row.get("simbad_main_id") or id_norm
            ned = self.ned.lookup(str(ned_target)) if self.ned else None
            if ned:
                source_context["ned"] = {"error": ned.error, "ambiguous": ned.ambiguous}
                if ned.ambiguous:
                    row["match_status"] = "ambiguous"
                    self._append_qc_flag(row, "ned_ambiguous_name")
                    source_context["candidates"] = dedupe_str_list(source_context.get("candidates", []) + ned.candidates)
                if ned.data:
                    self._set_field(row, provenance, "redshift", ned.data.get("redshift"), "ned", overwrite=False)
                    self._set_field(
                        row,
                        provenance,
                        "radial_velocity_kms",
                        ned.data.get("radial_velocity_kms"),
                        "ned",
                        overwrite=False,
                    )
                    self._set_field(row, provenance, "dist_value", ned.data.get("dist_value"), "ned")
                    self._set_field(row, provenance, "dist_unit", ned.data.get("dist_unit"), "ned")
                    self._set_field(row, provenance, "dist_err_minus", ned.data.get("dist_err_minus"), "ned")
                    self._set_field(row, provenance, "dist_err_plus", ned.data.get("dist_err_plus"), "ned")
                    self._set_field(row, provenance, "dist_method", ned.data.get("dist_method"), "ned")
                    self._set_field(row, provenance, "morphology", ned.data.get("morphology"), "ned")
                    if _is_empty(row.get("simbad_main_id")):
                        self._set_field(row, provenance, "simbad_main_id", ned.data.get("main_id"), "ned")
                elif ned.error:
                    self._append_qc_flag(row, "ned_lookup_failed")
                if ned.url:
                    self._append_link(row, _build_ned_info_url(str(row.get("simbad_main_id") or id_norm)))

            vizier_target = row.get("simbad_main_id") or id_norm
            vizier = self.vizier.lookup(str(vizier_target)) if self.vizier else None
            if vizier:
                source_context["vizier"] = {"error": vizier.error}
                if vizier.data:
                    self._set_field(row, provenance, "morphology", vizier.data.get("morphology"), "vizier_tap")
                    self._set_field(row, provenance, "mag_v", vizier.data.get("mag_v"), "vizier_tap")
                    if row.get("mag_v") is not None and _is_empty(row.get("mag_band")):
                        self._set_field(row, provenance, "mag_band", "V", "derived")
                    self._set_field(
                        row,
                        provenance,
                        "ang_size_maj_arcmin",
                        vizier.data.get("ang_size_maj_arcmin"),
                        "vizier_tap",
                    )
                    self._set_field(
                        row,
                        provenance,
                        "ang_size_min_arcmin",
                        vizier.data.get("ang_size_min_arcmin"),
                        "vizier_tap",
                    )
                    self._set_field(row, provenance, "pa_deg", vizier.data.get("pa_deg"), "vizier_tap")
                elif vizier.error:
                    self._append_qc_flag(row, "vizier_lookup_failed")
                if vizier.url:
                    self._append_link(row, vizier.url)

        catalog_ids = catalog_ids_from_values([row.get("id_norm", "")] + row.get("cross_ids", []))
        for field, value in catalog_ids.items():
            self._set_field(row, provenance, field, value, "derived")

    def _apply_derived_fields(self, row: dict[str, Any]) -> None:
        provenance: dict[str, Any] = row["field_provenance"]

        object_type_norm = normalize_object_type(row.get("object_type_simbad"))
        if object_type_norm and _is_empty(row.get("object_type_norm")):
            self._set_field(row, provenance, "object_type_norm", object_type_norm, "derived")

        ra = _safe_float(row.get("ra_j2000_deg"))
        dec = _safe_float(row.get("dec_j2000_deg"))
        if ra is not None and dec is not None:
            self._set_field(row, provenance, "ra_j2000_hms", _ra_deg_to_hms(ra), "derived")
            self._set_field(row, provenance, "dec_j2000_dms", _dec_deg_to_dms(dec), "derived")

        mag_v = _safe_float(row.get("mag_v"))
        maj = _safe_float(row.get("ang_size_maj_arcmin"))
        min_axis = _safe_float(row.get("ang_size_min_arcmin"))
        if mag_v is not None and maj and min_axis and maj > 0 and min_axis > 0:
            a_arcsec = maj * 60.0
            b_arcsec = min_axis * 60.0
            area = math.pi * (a_arcsec / 2.0) * (b_arcsec / 2.0)
            if area > 0:
                sb = mag_v + 2.5 * math.log10(area)
                self._set_field(
                    row,
                    provenance,
                    "surface_brightness_mag_arcsec2",
                    round(sb, 3),
                    "derived",
                    overwrite=True,
                )
                self._set_field(row, provenance, "sb_band", row.get("mag_band") or "V", "derived")

        self._derive_distance_and_velocity(row, provenance)
        self._derive_emission_lines(row, provenance)
        self._derive_multiwave_flags(row, provenance)
        self._derive_archive_urls(row, provenance)
        self._derive_info_links(row, provenance)
        self._derive_notable_features(row, provenance)

        if _is_empty(row.get("description")):
            common_name = row.get("common_name")
            object_type = row.get("object_type_norm") or "deep_sky_object"
            display_name = common_name or row.get("id_norm")
            pieces = [f"{display_name} is classified as {str(object_type).replace('_', ' ')}."]
            if row.get("morphology"):
                pieces.append(f"Morphology: {row.get('morphology')}.")
            if row.get("dist_value") and row.get("dist_unit"):
                pieces.append(f"Distance estimate: {row.get('dist_value')} {row.get('dist_unit')}.")
            if row.get("redshift") is not None:
                pieces.append(f"Redshift z={row.get('redshift')}.")
            description = " ".join(pieces)
            self._set_field(row, provenance, "description", description, "derived")

        row["links"] = dedupe_str_list(row.get("links", []))
        row["cross_ids"] = dedupe_str_list(row.get("cross_ids", []))

    def _finalize_quality_flags(self, row: dict[str, Any]) -> None:
        ra = _safe_float(row.get("ra_j2000_deg"))
        dec = _safe_float(row.get("dec_j2000_deg"))
        if ra is None or dec is None:
            self._append_qc_flag(row, "missing_coordinates")

        if row.get("match_status") == "unmatched":
            self._append_qc_flag(row, "unmatched_identifier")
        if row.get("match_status") == "ambiguous":
            self._append_qc_flag(row, "requires_manual_review")

        coord_quality = str(row.get("coord_quality") or "").upper()
        if coord_quality in {"D", "E"}:
            self._append_qc_flag(row, "coord_low_quality")

        row["qc_flags"] = dedupe_str_list(row.get("qc_flags", []))

    def _set_field(
        self,
        row: dict[str, Any],
        provenance: dict[str, Any],
        field: str,
        value: Any,
        source: str,
        overwrite: bool = False,
    ) -> None:
        if _is_empty(value):
            return
        if not overwrite and not _is_empty(row.get(field)):
            return
        row[field] = value
        provenance[field] = {"src": source, "ts": _utc_now_iso()}

    def _append_qc_flag(self, row: dict[str, Any], flag: str) -> None:
        flags = row.get("qc_flags")
        if not isinstance(flags, list):
            flags = []
            row["qc_flags"] = flags
        flags.append(flag)

    def _append_link(self, row: dict[str, Any], link: str | None) -> None:
        if _is_empty(link):
            return
        links = row.get("links")
        if not isinstance(links, list):
            links = []
            row["links"] = links
        links.append(str(link))

    def _is_extragalactic_candidate(
        self,
        row: dict[str, Any],
        simbad_data: dict[str, Any] | None = None,
    ) -> bool:
        checks: list[str] = []
        for value in (
            row.get("object_type_simbad"),
            row.get("object_type_norm"),
            (simbad_data or {}).get("object_type") if simbad_data else None,
        ):
            if value:
                checks.append(str(value).lower())

        if any(
            token in " ".join(checks)
            for token in ("galaxy", "seyfert", "qso", "quasar", "bl lac", "agn")
        ):
            return True
        if row.get("redshift") is not None:
            return True
        id_norm = str(row.get("id_norm") or "").upper()
        return id_norm.startswith(("PGC ", "UGC "))

    def _derive_distance_and_velocity(self, row: dict[str, Any], provenance: dict[str, Any]) -> None:
        redshift = _safe_float(row.get("redshift"))
        radial_velocity = _safe_float(row.get("radial_velocity_kms"))
        if radial_velocity is None and redshift is not None:
            self._set_field(
                row,
                provenance,
                "radial_velocity_kms",
                round(redshift * C_KM_S, 3),
                "derived",
            )
        if _is_empty(row.get("dist_value")) and redshift is not None and redshift > 0:
            approx_mpc = redshift * C_KM_S / H0_KM_S_MPC
            self._set_field(row, provenance, "dist_value", round(approx_mpc, 3), "derived")
            self._set_field(row, provenance, "dist_unit", "Mpc", "derived")
            self._set_field(row, provenance, "dist_method", "hubble_law_approx", "derived")

    def _derive_emission_lines(self, row: dict[str, Any], provenance: dict[str, Any]) -> None:
        text = " ".join(
            [
                str(row.get("object_type_simbad") or ""),
                str(row.get("object_type_norm") or ""),
            ]
        ).lower()
        lines: list[str] = []
        if any(token in text for token in ("emission nebula", "hii", "h ii", "planetary nebula")):
            lines.extend(["H-alpha", "[OIII]", "[NII]"])
        if "supernova remnant" in text:
            lines.extend(["H-alpha", "[SII]", "[OIII]"])
        if lines:
            row["emission_lines"] = dedupe_str_list(row.get("emission_lines", []) + lines)
            provenance["emission_lines"] = {"src": "derived", "ts": _utc_now_iso()}
            if _is_empty(row.get("hero_image_filters")):
                row["hero_image_filters"] = row["emission_lines"]
                provenance["hero_image_filters"] = {"src": "derived", "ts": _utc_now_iso()}

    def _derive_multiwave_flags(self, row: dict[str, Any], provenance: dict[str, Any]) -> None:
        cross_ids = [str(item).upper() for item in row.get("cross_ids", []) if item]
        flags: list[str] = []
        if any(token in cid for cid in cross_ids for token in ("IRAS", "2MASS", "2MASX", "WISE", "AKARI")):
            flags.append("IR")
        if any(token in cid for cid in cross_ids for token in ("GALEX",)):
            flags.append("UV")
        if any(token in cid for cid in cross_ids for token in ("XMM", "CHANDRA", "CXO", "ROSAT", "1RXS", "2RXS")):
            flags.append("X-ray")
        if any(token in cid for cid in cross_ids for token in ("NVSS", "PKS", "3C", "4C", "FIRST", "SUMSS")):
            flags.append("Radio")
        if row.get("object_type_norm") in ("galaxy", "nebula", "emission_nebula", "planetary_nebula"):
            flags.append("Optical")
        if flags:
            row["multiwave_flags"] = dedupe_str_list(row.get("multiwave_flags", []) + flags)
            provenance["multiwave_flags"] = {"src": "derived", "ts": _utc_now_iso()}

    def _derive_archive_urls(self, row: dict[str, Any], provenance: dict[str, Any]) -> None:
        existing_url = str(row.get("hero_image_url") or "").strip()
        converted = _maybe_convert_page_to_image_url(existing_url)
        if converted and converted != existing_url:
            self._set_field(row, provenance, "hero_image_url", converted, "archive_curated", overwrite=True)
            if _is_empty(row.get("hero_image_credit")):
                self._set_field(row, provenance, "hero_image_credit", "ESA/Hubble CDN", "archive_curated")
            if existing_url and not _is_direct_image_url(existing_url):
                self._append_link(row, existing_url)
            return

        if _is_direct_image_url(existing_url):
            return

        ra_deg = _safe_float(row.get("ra_j2000_deg"))
        dec_deg = _safe_float(row.get("dec_j2000_deg"))
        if ra_deg is None or dec_deg is None:
            return

        maj_arcmin = _safe_float(row.get("ang_size_maj_arcmin"))
        fov_deg = 0.35
        if maj_arcmin and maj_arcmin > 0:
            # Framing at ~2x major axis with practical clamp for tiny/huge targets.
            fov_deg = max(0.12, min(2.0, (maj_arcmin * 2.0) / 60.0))

        cutout_url = _build_dss_cutout_url(ra_deg=ra_deg, dec_deg=dec_deg, fov_deg=fov_deg)
        self._set_field(row, provenance, "hero_image_url", cutout_url, "archive_curated", overwrite=True)
        if _is_empty(row.get("hero_image_credit")):
            self._set_field(
                row,
                provenance,
                "hero_image_credit",
                "CDS DSS2 cutout via HiPS2FITS",
                "archive_curated",
            )

    def _derive_info_links(self, row: dict[str, Any], provenance: dict[str, Any]) -> None:
        existing_links = row.get("links")
        if not isinstance(existing_links, list):
            existing_links = []

        # Keep links focused on human-readable info pages.
        filtered_links = [link for link in existing_links if not _is_direct_image_url(link)]
        candidates: list[str] = []

        identifier = str(row.get("simbad_main_id") or row.get("id_norm") or "").strip()
        if identifier:
            candidates.append(_build_simbad_info_url(identifier))
            if self._is_extragalactic_candidate(row):
                candidates.append(_build_ned_info_url(identifier))

        messier_id = str(row.get("messier_id") or "").strip()
        if messier_id:
            parts = messier_id.split()
            if len(parts) == 2:
                try:
                    messier_num = int(parts[1])
                    candidates.append(_build_nasa_messier_page(messier_num))
                except ValueError:
                    pass

        merged = dedupe_str_list(filtered_links + candidates)
        row["links"] = merged
        if merged:
            provenance["links"] = {"src": "derived", "ts": _utc_now_iso()}

    def _derive_notable_features(self, row: dict[str, Any], provenance: dict[str, Any]) -> None:
        features: list[str] = []
        if row.get("match_status") == "ambiguous":
            features.append("identifier requires manual review")
        if row.get("messier_id") and row.get("ngc_id"):
            features.append("cross-catalog Messier/NGC linkage")
        if row.get("redshift") is not None:
            features.append("extragalactic redshift available")
        if row.get("morphology"):
            features.append(f"morphology {row.get('morphology')}")
        if features:
            self._set_field(
                row,
                provenance,
                "notable_features",
                "; ".join(dedupe_str_list(features)),
                "derived",
                overwrite=True,
            )
