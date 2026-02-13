from __future__ import annotations

import re
from typing import Iterable


_SPACE_RE = re.compile(r"\s+")
_MESSIER_RE = re.compile(r"^M\s*0*(\d+)$", re.IGNORECASE)
_NGC_RE = re.compile(r"^NGC\s*0*(\d+)(.*)$", re.IGNORECASE)
_IC_RE = re.compile(r"^IC\s*0*(\d+)(.*)$", re.IGNORECASE)
_CALDWELL_RE = re.compile(r"^C\s*0*(\d+)$", re.IGNORECASE)
_SHARPLESS_RE = re.compile(r"^(SH\s*2|SH2|SH-?2)\s*[- ]?\s*0*(\d+)$", re.IGNORECASE)


def _compact_spaces(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip())


def _clean_common_name(value: str) -> str:
    cleaned = _compact_spaces(value)
    if cleaned.upper().startswith("NAME "):
        cleaned = cleaned[5:].strip()
    return cleaned


def normalize_identifier(raw_identifier: str) -> tuple[str, str]:
    """Return canonical identifier and family guess."""
    if not raw_identifier:
        return "", "unknown"

    value = _compact_spaces(raw_identifier)
    messier = _MESSIER_RE.match(value)
    if messier:
        return f"M {int(messier.group(1))}", "messier"

    ngc = _NGC_RE.match(value)
    if ngc:
        suffix = _compact_spaces(ngc.group(2))
        base = f"NGC {int(ngc.group(1))}"
        return (f"{base} {suffix}".strip(), "ngc")

    ic = _IC_RE.match(value)
    if ic:
        suffix = _compact_spaces(ic.group(2))
        base = f"IC {int(ic.group(1))}"
        return (f"{base} {suffix}".strip(), "ic")

    caldwell = _CALDWELL_RE.match(value)
    if caldwell:
        return f"C {int(caldwell.group(1))}", "caldwell"

    sharpless = _SHARPLESS_RE.match(value)
    if sharpless:
        return f"SH2-{int(sharpless.group(2))}", "survey"

    upper = value.upper()
    if upper.startswith(("BARNARD", "LBN", "LDN", "UGC", "PGC", "MEL", "CR", "RCW")):
        return upper, "survey"

    return value, "common_name"


def split_aliases(aliases_text: str | None) -> list[str]:
    if not aliases_text:
        return []
    values = [_clean_common_name(v) for v in aliases_text.split(";")]
    return [v for v in values if v]


def dedupe_str_list(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def catalog_ids_from_values(values: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "messier_id": None,
        "ngc_id": None,
        "ic_id": None,
        "caldwell_id": None,
    }
    for raw in values:
        norm, _ = normalize_identifier(raw)
        if norm.startswith("M "):
            result["messier_id"] = result["messier_id"] or norm
        elif norm.startswith("NGC "):
            result["ngc_id"] = result["ngc_id"] or norm
        elif norm.startswith("IC "):
            result["ic_id"] = result["ic_id"] or norm
        elif norm.startswith("C "):
            result["caldwell_id"] = result["caldwell_id"] or norm
    return result


def normalize_object_type(value: str | None) -> str | None:
    if not value:
        return None
    text = value.lower()
    if "galaxy" in text or "seyfert" in text:
        return "galaxy"
    if "planetary nebula" in text:
        return "planetary_nebula"
    if "emission nebula" in text or "hii" in text or "h ii" in text:
        return "emission_nebula"
    if "reflection nebula" in text:
        return "reflection_nebula"
    if "supernova remnant" in text:
        return "supernova_remnant"
    if "globular" in text:
        return "globular_cluster"
    if "open cluster" in text:
        return "open_cluster"
    if "cluster" in text:
        return "cluster"
    if "nebula" in text:
        return "nebula"
    return "other"


def clean_common_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = _clean_common_name(value)
    return cleaned or None
