#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Callable
from urllib.parse import quote, unquote, urlencode, urlparse

import pandas as pd

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

DEFAULT_INCLUDE_CATALOGS = ("M", "SH2")
DEFAULT_INCLUDE_GROUPS = ("bright nebula", "dark nebula")

ASTRO_HINT_TOKENS = (
    "astronomy",
    "messier",
    "sharpless",
    "nebula",
    "galaxy",
    "cluster",
    "supernova",
    "stellar",
    "h ii",
    "hii",
    "interstellar",
    "deep sky",
)
DISAMBIGUATION_HINTS = (
    "disambiguation",
    "may refer to",
    "can refer to",
)
NEBULA_TOPIC_TOKENS = (
    "nebula",
    "nebulous",
    "h ii region",
    "emission region",
    "molecular cloud",
    "interstellar cloud",
    "dark cloud",
    "reflection nebula",
    "planetary nebula",
    "supernova remnant",
)
STAR_TOPIC_TOKENS = (
    "star",
    "binary star",
    "multiple star",
    "variable star",
    "stellar",
    "spectral type",
    "main sequence",
    "giant star",
    "white dwarf",
)
MESSIER_OBJECT_HINT_TOKENS = (
    "galaxy",
    "nebula",
    "star",
    "cluster",
    "globular",
    "open cluster",
    "planetary nebula",
    "supernova remnant",
)

DESIGNATION_KEY_HINTS = (
    "designation",
    "designations",
    "othername",
    "othernames",
    "alias",
    "aliases",
    "altname",
    "alternatename",
    "otherid",
    "otherids",
    "catalog",
    "catalogue",
    "name",
    "names",
)
DESIGNATION_KEY_EXCLUDES = (
    "image",
    "caption",
    "discover",
    "distance",
    "magnitude",
    "constellation",
    "epoch",
    "ra",
    "dec",
    "radius",
    "mass",
    "temperature",
)

BR_TAG_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
REF_TAG_RE = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", flags=re.IGNORECASE | re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
WIKILINK_PIPED_RE = re.compile(r"\[\[[^\]|]+\|([^\]]+)\]\]")
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
NAME_PREFIX_RE = re.compile(r"^\s*NAME\s+(.+)$", flags=re.IGNORECASE)
DESIGNATION_INVALID_CHAR_RE = re.compile(r"[^A-Za-z0-9+\-./() ]+")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_text(value: str | None) -> str:
    return NON_ALNUM_RE.sub("", str(value or "").strip().lower())


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def _split_aliases(raw_aliases: Any) -> list[str]:
    text = str(raw_aliases or "").strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split(";")]
    return [part for part in parts if part]


def _strip_name_prefix(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = NAME_PREFIX_RE.match(text)
    if not match:
        return text
    stripped = str(match.group(1) or "").strip()
    return stripped or text


def _name_variants(value: Any, *, include_name_prefixed_original: bool = False) -> list[str]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return []
    stripped = _strip_name_prefix(cleaned)
    if stripped and stripped != cleaned:
        if include_name_prefixed_original:
            return [cleaned, stripped]
        return [stripped]
    return [cleaned]


def _with_name_prefix_variants(
    values: list[str],
    *,
    include_name_prefixed_original: bool = False,
) -> list[str]:
    expanded: list[str] = []
    for value in values:
        expanded.extend(
            _name_variants(
                value,
                include_name_prefixed_original=include_name_prefixed_original,
            )
        )
    return _dedupe_preserve_order(expanded)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _sentence_list(text: str) -> list[str]:
    compact = " ".join(str(text or "").split())
    if not compact:
        return []
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(compact) if segment.strip()]
    return sentences


def _truncate(text: str, max_len: int = 120) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= max_len:
        return value
    return f"{value[: max_len - 3]}..."


def _description_3_to_4_sentences(*texts: str) -> str:
    best: list[str] = []
    for text in texts:
        sentences = _sentence_list(text)
        if len(sentences) >= 4:
            return " ".join(sentences[:4])
        if len(sentences) >= 3:
            return " ".join(sentences[:3])
        if len(sentences) > len(best):
            best = sentences
    if not best:
        return ""
    return " ".join(best[:4])


def _append_sentence_if_missing(base_text: str, sentence: str) -> str:
    base = str(base_text or "").strip()
    extra = str(sentence or "").strip()
    if not extra:
        return base
    if not extra.endswith("."):
        extra = f"{extra}."
    if not base:
        return extra

    normalized_base = _normalize_text(base)
    normalized_extra = _normalize_text(extra)
    if normalized_extra and normalized_extra in normalized_base:
        return base
    return f"{base} {extra}".strip()


def _looks_disambiguation(title: str, extract: str) -> bool:
    content = f"{title} {extract}".lower()
    return any(token in content for token in DISAMBIGUATION_HINTS)


def _looks_astronomy(summary_text: str) -> bool:
    lowered = summary_text.lower()
    return any(token in lowered for token in ASTRO_HINT_TOKENS)


def _page_topic_label(title: str, extract: str, short_description: str) -> str:
    lowered = f"{title} {extract} {short_description}".lower()
    nebula_hits = sum(1 for token in NEBULA_TOPIC_TOKENS if token in lowered)
    star_hits = sum(1 for token in STAR_TOPIC_TOKENS if token in lowered)

    if nebula_hits > star_hits and nebula_hits > 0:
        return "nebula"
    if star_hits > nebula_hits and star_hits > 0:
        return "star"
    return "other"


def _looks_messier_object_page(title: str, extract: str, short_description: str) -> bool:
    lowered = f"{title} {extract} {short_description}".lower()
    return any(token in lowered for token in MESSIER_OBJECT_HINT_TOKENS)


def _is_m_shorthand_query(query: str, messier_number: int | None) -> bool:
    if messier_number is None:
        return False
    match = re.fullmatch(r"m\s*0*([0-9]+)", str(query or "").strip(), flags=re.IGNORECASE)
    if not match:
        return False
    try:
        return int(match.group(1)) == int(messier_number)
    except ValueError:
        return False


def _row_prefers_nebula(row: dict[str, Any]) -> bool:
    primary_id = _strip_name_prefix(str(row.get("primary_id", "")).strip())
    object_type_group = str(row.get("object_type_group", "")).strip().lower()
    object_type = str(row.get("object_type", "")).strip().lower()
    common_name = str(row.get("common_name", "")).strip().lower()

    if "nebula" in object_type_group:
        return True
    if "nebula" in object_type:
        return True
    if "nebula" in common_name:
        return True
    if primary_id.lower().startswith("sh2-"):
        return True
    return False


def _clean_wikitext_value(raw_value: str) -> str:
    text = str(raw_value or "")
    text = COMMENT_RE.sub(" ", text)
    text = REF_TAG_RE.sub(" ", text)
    text = BR_TAG_RE.sub(";", text)

    # Collapse trivial templates; run repeatedly so nested simple templates reduce.
    for _ in range(6):
        reduced = TEMPLATE_RE.sub(" ", text)
        if reduced == text:
            break
        text = reduced

    text = WIKILINK_PIPED_RE.sub(r"\1", text)
    text = WIKILINK_RE.sub(r"\1", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_designation_token(token: str) -> str:
    cleaned = str(token or "").strip().strip(" ,;:.")
    cleaned = cleaned.replace("''", "").replace("'''", "")
    cleaned = cleaned.replace("{{", " ").replace("}}", " ")
    cleaned = cleaned.replace("[[", " ").replace("]]", " ")
    cleaned = cleaned.replace("|", " ")
    cleaned = DESIGNATION_INVALID_CHAR_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(" ,;:.\"'")
    return cleaned


def _extract_infobox_block(wikitext: str) -> str:
    source = str(wikitext or "")
    start = source.lower().find("{{infobox")
    if start < 0:
        return ""

    depth = 0
    index = start
    while index < len(source) - 1:
        pair = source[index:index + 2]
        if pair == "{{":
            depth += 1
            index += 2
            continue
        if pair == "}}":
            depth -= 1
            index += 2
            if depth <= 0:
                return source[start:index]
            continue
        index += 1
    return ""


def _parse_infobox_fields(infobox_block: str) -> list[tuple[str, str]]:
    if not infobox_block.strip():
        return []

    fields: list[tuple[str, str]] = []
    current_key: str | None = None
    current_value_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_key, current_value_lines
        if current_key is None:
            return
        joined = "\n".join(current_value_lines).strip()
        fields.append((current_key, joined))
        current_key = None
        current_value_lines = []

    for raw_line in infobox_block.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped.startswith("|"):
            _flush()
            body = stripped[1:]
            if "=" not in body:
                continue
            key, value = body.split("=", 1)
            key = key.strip()
            if not key:
                continue
            current_key = key
            current_value_lines = [value.strip()]
        else:
            if current_key is not None:
                current_value_lines.append(stripped)

    _flush()
    return fields


def _designation_key_match(key: str) -> bool:
    normalized = NON_ALNUM_RE.sub("", str(key or "").lower())
    if not normalized:
        return False
    if any(bad in normalized for bad in DESIGNATION_KEY_EXCLUDES):
        return False
    return any(hint in normalized for hint in DESIGNATION_KEY_HINTS)


def _extract_designations_from_wikitext(wikitext: str) -> list[str]:
    infobox = _extract_infobox_block(wikitext)
    if not infobox:
        return []

    values: list[str] = []
    for key, raw_value in _parse_infobox_fields(infobox):
        if not _designation_key_match(key):
            continue
        cleaned_value = _clean_wikitext_value(raw_value)
        if not cleaned_value:
            continue
        parts = re.split(r"[;,•·\n]+", cleaned_value)
        if not parts:
            continue
        for part in parts:
            token = _clean_designation_token(part)
            if not token:
                continue
            lowered = token.lower()
            if lowered.startswith("http") or "://" in lowered:
                continue
            if lowered in {"none", "n/a", "unknown", "various", "multiple"}:
                continue
            if len(token) > 120:
                continue
            if not re.search(r"[A-Za-z0-9]", token):
                continue
            values.append(token)

    return _dedupe_preserve_order(values)


def _strip_parenthetical_suffix(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(text or "").strip()).strip()


def _extract_infobox_entries(wikitext: str) -> list[tuple[str, str, str]]:
    infobox = _extract_infobox_block(wikitext)
    if not infobox:
        return []

    entries: list[tuple[str, str, str]] = []
    for key, raw_value in _parse_infobox_fields(infobox):
        normalized_key = NON_ALNUM_RE.sub("", str(key or "").lower())
        if not normalized_key:
            continue
        cleaned_value = _clean_wikitext_value(raw_value)
        entries.append((normalized_key, cleaned_value, str(raw_value or "")))
    return entries


def _select_infobox_value(
    entries: list[tuple[str, str, str]],
    *,
    includes: tuple[str, ...],
    excludes: tuple[str, ...] = (),
) -> str:
    for normalized_key, cleaned_value, _raw_value in entries:
        if any(bad in normalized_key for bad in excludes):
            continue
        if any(good in normalized_key for good in includes):
            if cleaned_value:
                return cleaned_value
    return ""


def _to_float_tokens(text: str) -> list[float]:
    values: list[float] = []
    for token in NUMBER_RE.findall(str(text or "")):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _parse_ra_candidate(raw_value: str) -> float | None:
    text = str(raw_value or "").replace("−", "-").replace("–", "-").strip()
    if not text:
        return None

    lowered = text.lower()
    values = _to_float_tokens(text)
    if not values:
        return None

    if ("deg" in lowered or "°" in text or "degree" in lowered) and 0.0 <= values[0] <= 360.0:
        return float(values[0] % 360.0)

    if len(values) >= 3:
        hours, minutes, seconds = abs(values[0]), abs(values[1]), abs(values[2])
        deg = (hours + (minutes / 60.0) + (seconds / 3600.0)) * 15.0
        if 0.0 <= deg <= 360.0:
            return float(deg % 360.0)

    if len(values) == 2:
        hours, minutes = abs(values[0]), abs(values[1])
        deg = (hours + (minutes / 60.0)) * 15.0
        if 0.0 <= deg <= 360.0:
            return float(deg % 360.0)

    value = values[0]
    if 0.0 <= value <= 24.0:
        return float((value * 15.0) % 360.0)
    if 0.0 <= value <= 360.0:
        return float(value % 360.0)
    return None


def _parse_dec_candidate(raw_value: str) -> float | None:
    text = str(raw_value or "").replace("−", "-").replace("–", "-").strip()
    if not text:
        return None

    lowered = text.lower()
    values = _to_float_tokens(text)
    if not values:
        return None

    sign = -1.0 if text.lstrip().startswith("-") else 1.0
    if values[0] < 0:
        sign = -1.0

    if ("deg" in lowered or "°" in text or "degree" in lowered) and len(values) >= 1:
        degree = float(values[0])
        if text.lstrip().startswith("+"):
            degree = abs(degree)
        if text.lstrip().startswith("-"):
            degree = -abs(degree)
        if -90.0 <= degree <= 90.0:
            return degree

    if len(values) >= 3:
        degrees, minutes, seconds = abs(values[0]), abs(values[1]), abs(values[2])
        dec = sign * (degrees + (minutes / 60.0) + (seconds / 3600.0))
        if -90.0 <= dec <= 90.0:
            return float(dec)

    if len(values) == 2:
        degrees, minutes = abs(values[0]), abs(values[1])
        dec = sign * (degrees + (minutes / 60.0))
        if -90.0 <= dec <= 90.0:
            return float(dec)

    dec = float(values[0])
    if text.lstrip().startswith("+"):
        dec = abs(dec)
    elif text.lstrip().startswith("-"):
        dec = -abs(dec)
    if -90.0 <= dec <= 90.0:
        return dec
    return None


def _is_ra_key(normalized_key: str) -> bool:
    key = str(normalized_key or "")
    if not key:
        return False
    if "radius" in key:
        return False
    if "image" in key or "caption" in key:
        return False
    return key == "ra" or key.startswith("ra") or "rightascension" in key


def _is_dec_key(normalized_key: str) -> bool:
    key = str(normalized_key or "")
    if not key:
        return False
    if "image" in key or "caption" in key:
        return False
    return key == "dec" or key.startswith("dec") or "declination" in key


def _extract_ra_dec_from_infobox(entries: list[tuple[str, str, str]]) -> tuple[float | None, float | None]:
    ra_value: float | None = None
    dec_value: float | None = None

    for normalized_key, cleaned_value, raw_value in entries:
        if ra_value is None and _is_ra_key(normalized_key):
            ra_value = _parse_ra_candidate(raw_value) or _parse_ra_candidate(cleaned_value)
        if dec_value is None and _is_dec_key(normalized_key):
            dec_value = _parse_dec_candidate(raw_value) or _parse_dec_candidate(cleaned_value)
        if ra_value is not None and dec_value is not None:
            break

    return ra_value, dec_value


def _extract_common_name(
    *,
    entries: list[tuple[str, str, str]],
    title: str,
    primary_id: str,
) -> str:
    infobox_name = _select_infobox_value(
        entries,
        includes=("commonname", "propername", "name"),
        excludes=(
            "image",
            "caption",
            "catalog",
            "designation",
            "alias",
            "othername",
            "othernames",
        ),
    )
    first_name = re.split(r"[;,•·\n]+", infobox_name)[0].strip() if infobox_name else ""

    for candidate in [first_name, _strip_parenthetical_suffix(title)]:
        text = str(candidate or "").strip()
        if not text:
            continue
        if len(text) > 120:
            continue
        if _normalize_text(text) == _normalize_text(primary_id):
            continue
        return text
    return ""


def _extract_object_type(
    *,
    entries: list[tuple[str, str, str]],
    short_description: str,
    topic_label: str,
) -> str:
    from_infobox = _select_infobox_value(
        entries,
        includes=("objecttype", "classification", "type", "class"),
        excludes=(
            "spectral",
            "variable",
            "image",
            "caption",
            "name",
            "distance",
            "radius",
            "magnitude",
            "temperature",
            "mass",
        ),
    )
    if from_infobox:
        token = re.split(r"[.;\n]+", from_infobox)[0].strip()
        if token:
            return token

    fallback = str(short_description or "").strip()
    if fallback:
        token = re.split(r"[.;\n]+", fallback)[0].strip()
        if token:
            return token

    if topic_label == "nebula":
        return "Nebula"
    if topic_label == "star":
        return "Star"
    return ""


def _extract_constellation(entries: list[tuple[str, str, str]]) -> str:
    value = _select_infobox_value(
        entries,
        includes=("constellation", "constell"),
        excludes=("image", "caption"),
    )
    if not value:
        return ""
    token = re.split(r"[;,•·\n]+", value)[0].strip()
    return token


def _extract_emission_lines(entries: list[tuple[str, str, str]]) -> list[str]:
    value = _select_infobox_value(
        entries,
        includes=("emissionline", "emissionlines", "spectralline", "spectrallines"),
        excludes=("image", "caption"),
    )
    if not value:
        return []

    lines: list[str] = []
    for part in re.split(r"[;,•·\n]+", value):
        token = _clean_designation_token(part)
        if not token:
            continue
        if len(token) > 40:
            continue
        if not re.search(r"[A-Za-z0-9]", token):
            continue
        lines.append(token)
    return _dedupe_preserve_order(lines)


def _identifier_catalog(primary_id: str) -> str:
    cleaned = str(primary_id or "").strip()
    if re.fullmatch(r"M\s*[0-9]+[A-Za-z]*", cleaned, flags=re.IGNORECASE):
        return "M"
    if re.fullmatch(r"NGC\s*[0-9]+(?:\s*[A-Za-z]+)?", cleaned, flags=re.IGNORECASE):
        return "NGC"
    if re.fullmatch(r"IC\s*[0-9]+(?:\s*[A-Za-z]+)?", cleaned, flags=re.IGNORECASE):
        return "IC"
    if re.fullmatch(r"Sh2-[0-9]+[A-Za-z]*", cleaned, flags=re.IGNORECASE):
        return "SH2"
    return ""


def _canonicalize_requested_primary_id(raw_value: Any) -> str:
    text = " ".join(str(raw_value or "").strip().split())
    if not text:
        return ""

    text = _strip_name_prefix(text)

    messier_match = re.fullmatch(r"(?:M|Messier)\s*0*([0-9]+)\s*([A-Za-z]*)", text, flags=re.IGNORECASE)
    if messier_match:
        number = str(int(messier_match.group(1)))
        suffix = str(messier_match.group(2) or "").strip().upper()
        return f"M{number}{suffix}" if suffix else f"M{number}"

    ngc_match = re.fullmatch(r"NGC\s*0*([0-9]+)\s*([A-Za-z]*)", text, flags=re.IGNORECASE)
    if ngc_match:
        number = str(int(ngc_match.group(1)))
        suffix = str(ngc_match.group(2) or "").strip().upper()
        return f"NGC {number}" if not suffix else f"NGC {number} {suffix}"

    ic_match = re.fullmatch(r"IC\s*0*([0-9]+)\s*([A-Za-z]*)", text, flags=re.IGNORECASE)
    if ic_match:
        number = str(int(ic_match.group(1)))
        suffix = str(ic_match.group(2) or "").strip().upper()
        return f"IC {number}" if not suffix else f"IC {number} {suffix}"

    sh2_match = re.fullmatch(
        r"(?:Sh2|Sh\s*2|Sharpless\s*2)[-\s]*0*([0-9]+)\s*([A-Za-z]*)",
        text,
        flags=re.IGNORECASE,
    )
    if sh2_match:
        number = str(int(sh2_match.group(1)))
        suffix = str(sh2_match.group(2) or "").strip().upper()
        return f"Sh2-{number}{suffix}" if suffix else f"Sh2-{number}"

    return text


def _extract_catalog_id_candidates(text: str) -> list[tuple[int, str, str]]:
    source = str(text or "")
    matches: list[tuple[int, str, str]] = []

    for match in re.finditer(r"\b(?:Messier|M)\s*0*([0-9]+)\s*([A-Za-z]*)\b", source, flags=re.IGNORECASE):
        number = str(int(match.group(1)))
        suffix = str(match.group(2) or "").strip()
        ident = f"M{number}{suffix}" if suffix else f"M{number}"
        matches.append((int(match.start()), ident, "M"))

    for match in re.finditer(r"\bNGC\s*0*([0-9]+)\s*([A-Za-z]*)\b", source, flags=re.IGNORECASE):
        number = str(int(match.group(1)))
        suffix = str(match.group(2) or "").strip()
        ident = f"NGC {number}" if not suffix else f"NGC {number} {suffix}"
        matches.append((int(match.start()), ident, "NGC"))

    for match in re.finditer(r"\bIC\s*0*([0-9]+)\s*([A-Za-z]*)\b", source, flags=re.IGNORECASE):
        number = str(int(match.group(1)))
        suffix = str(match.group(2) or "").strip()
        ident = f"IC {number}" if not suffix else f"IC {number} {suffix}"
        matches.append((int(match.start()), ident, "IC"))

    for match in re.finditer(
        r"\b(?:Sh2|Sh\s*2|Sharpless\s*2)[-\s]*0*([0-9]+)\s*([A-Za-z]*)\b",
        source,
        flags=re.IGNORECASE,
    ):
        number = str(int(match.group(1)))
        suffix = str(match.group(2) or "").strip()
        ident = f"Sh2-{number}{suffix}" if suffix else f"Sh2-{number}"
        matches.append((int(match.start()), ident, "SH2"))

    matches.sort(key=lambda item: item[0])
    return matches


def _extract_wikipedia_primary_catalog_reference(
    *,
    title: str,
    aliases: list[str],
    designations: list[str],
) -> tuple[str, str]:
    for candidate in [title, *aliases, *designations]:
        parsed = _extract_catalog_id_candidates(candidate)
        if parsed:
            _position, ident, catalog = parsed[0]
            return ident, catalog

    fallback = _strip_parenthetical_suffix(title)
    return fallback, _identifier_catalog(fallback)


def _file_title_from_image_url(image_url: str) -> str:
    parsed = urlparse(str(image_url or "").strip())
    path = str(parsed.path or "")
    if not path:
        return ""
    filename = unquote(path.rsplit("/", 1)[-1]).strip()
    if not filename:
        return ""
    filename = re.sub(r"^[0-9]+px-", "", filename)
    if not filename:
        return ""
    if filename.lower().startswith("file:"):
        return filename
    return f"File:{filename}"


def _messier_num(primary_id: str) -> int | None:
    match = re.fullmatch(r"M\s*([0-9]+)", str(primary_id).strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _sharpless_parts(primary_id: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"Sh2-([0-9]+)([A-Za-z]*)", str(primary_id).strip(), flags=re.IGNORECASE)
    if not match:
        return None
    number = str(int(match.group(1)))
    suffix = str(match.group(2) or "").strip()
    return number, suffix


def _build_title_candidates(row: dict[str, Any]) -> list[str]:
    primary_id = str(row.get("primary_id", "")).strip()
    common_name = str(row.get("common_name", "")).strip()
    aliases = _split_aliases(row.get("aliases", ""))
    primary_id_clean = _strip_name_prefix(primary_id)
    primary_id_variants = _name_variants(primary_id, include_name_prefixed_original=False)
    common_name_variants = _name_variants(common_name, include_name_prefixed_original=False)
    alias_variants = _with_name_prefix_variants(aliases, include_name_prefixed_original=False)

    candidates: list[str] = []
    messier_number = _messier_num(primary_id_clean)
    sharpless = _sharpless_parts(primary_id_clean)

    if messier_number is not None:
        candidates.extend(
            [
                f"M {messier_number}",
                f"Messier {messier_number}",
                f"M{messier_number}",
            ]
        )

    if sharpless is not None:
        number, suffix = sharpless
        suffix_text = suffix if suffix else ""
        candidates.extend(
            [
                f"Sh2-{number}{suffix_text}",
                f"Sh 2-{number}{suffix_text}",
                f"Sharpless 2-{number}{suffix_text}",
            ]
        )

    candidates.extend(primary_id_variants)
    candidates.extend(common_name_variants)
    candidates.extend(alias_variants[:10])
    return _dedupe_preserve_order(candidates)


def _build_search_queries(row: dict[str, Any], title_candidates: list[str]) -> list[str]:
    primary_id = str(row.get("primary_id", "")).strip()
    common_name = str(row.get("common_name", "")).strip()
    primary_id_clean = _strip_name_prefix(primary_id)
    primary_id_variants = _name_variants(primary_id, include_name_prefixed_original=False)
    common_name_variants = _name_variants(common_name, include_name_prefixed_original=False)
    queries: list[str] = []

    messier_number = _messier_num(primary_id_clean)
    sharpless = _sharpless_parts(primary_id_clean)
    if messier_number is not None:
        queries.extend(
            [
                f"M {messier_number}",
                f"M{messier_number}",
                f"Messier {messier_number}",
            ]
        )
    if sharpless is not None:
        number, suffix = sharpless
        suffix_text = suffix if suffix else ""
        queries.append(f"Sharpless 2-{number}{suffix_text}")
        queries.append(f"Sh2-{number}{suffix_text}")

    queries.extend(primary_id_variants)
    queries.extend(common_name_variants)
    queries.extend(title_candidates[:4])
    return _dedupe_preserve_order(queries)


def _build_identity_tokens(row: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    primary_id = str(row.get("primary_id", "")).strip()
    common_name = str(row.get("common_name", "")).strip()
    primary_id_clean = _strip_name_prefix(primary_id)
    aliases = _split_aliases(row.get("aliases", ""))
    primary_id_variants = _name_variants(primary_id, include_name_prefixed_original=False)
    common_name_variants = _name_variants(common_name, include_name_prefixed_original=False)
    alias_variants = _with_name_prefix_variants(aliases, include_name_prefixed_original=False)
    tokens.extend(primary_id_variants)
    tokens.extend(common_name_variants)
    tokens.extend(alias_variants[:8])

    messier_number = _messier_num(primary_id_clean)
    if messier_number is not None:
        tokens.extend([f"M{messier_number}", f"M {messier_number}", f"Messier {messier_number}"])

    sharpless = _sharpless_parts(primary_id_clean)
    if sharpless is not None:
        number, suffix = sharpless
        suffix_text = suffix if suffix else ""
        tokens.extend(
            [
                f"Sh2-{number}{suffix_text}",
                f"Sh 2-{number}{suffix_text}",
                f"Sharpless 2-{number}{suffix_text}",
            ]
        )

    normalized = [_normalize_text(token) for token in tokens]
    return [token for token in _dedupe_preserve_order(normalized) if len(token) >= 2]


def _score_search_result(result: dict[str, Any], identity_tokens: list[str]) -> int:
    title = str(result.get("title", "")).strip()
    snippet = _strip_html(str(result.get("snippet", "")))
    content_norm = _normalize_text(f"{title} {snippet}")
    title_norm = _normalize_text(title)

    score = 0
    if "disambiguation" in title.lower():
        score -= 90

    token_hits = 0
    for token in identity_tokens:
        if not token:
            continue
        if token in title_norm:
            score += 35
            token_hits += 1
            continue
        if token in content_norm:
            score += 12
            token_hits += 1
    if token_hits == 0:
        score -= 20

    if _looks_astronomy(f"{title} {snippet}"):
        score += 18
    return score


def _score_summary(title: str, extract: str, description: str, identity_tokens: list[str]) -> int:
    title_norm = _normalize_text(title)
    content_norm = _normalize_text(f"{title} {extract} {description}")
    score = 0

    if _looks_disambiguation(title, extract):
        score -= 120

    token_hits = 0
    for token in identity_tokens:
        if not token:
            continue
        if token in title_norm:
            score += 40
            token_hits += 1
            continue
        if token in content_norm:
            score += 14
            token_hits += 1
    if token_hits == 0:
        score -= 25

    if _looks_astronomy(f"{title} {extract} {description}"):
        score += 22

    if extract.strip():
        score += 10
    return score


@dataclass
class WikipediaLookup:
    title: str = ""
    url: str = ""
    description: str = ""
    image_url: str = ""
    image_attribution_url: str = ""
    info_url: str = ""
    designations: list[str] = None  # type: ignore[assignment]
    aliases: list[str] = None  # type: ignore[assignment]
    common_name: str = ""
    object_type: str = ""
    ra_deg: float | None = None
    dec_deg: float | None = None
    constellation: str = ""
    emission_lines: list[str] = None  # type: ignore[assignment]
    wikipedia_primary_id: str = ""
    wikipedia_catalog: str = ""
    match_method: str = "none"
    match_score: int = -999
    query_used: str = ""
    requested_title: str = ""
    topic_label: str = ""
    redirected_from_query: bool = False
    sentence_count: int = 0

    def __post_init__(self) -> None:
        if self.designations is None:
            self.designations = []
        if self.aliases is None:
            self.aliases = []
        if self.emission_lines is None:
            self.emission_lines = []


class WikipediaClient:
    def __init__(
        self,
        *,
        timeout_s: float,
        requests_per_second: float,
        cache_path: Path,
        log: Callable[[int, str], None] | None = None,
        user_agent: str = "deep-space-object-explorer-wiki-enrichment/1.0",
    ) -> None:
        self.timeout_s = timeout_s
        self.min_interval_s = (1.0 / requests_per_second) if requests_per_second > 0 else 0.0
        self._last_request_monotonic = 0.0
        self._cache_path = cache_path
        self._cache: dict[str, Any] = {}
        self._user_agent = user_agent
        self._log = log
        self.cache_hits = 0
        self.http_requests = 0
        self.http_successes = 0
        self.http_failures = 0
        self.rate_limit_sleeps = 0
        self._load_cache()

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            self._cache = {}
            return
        try:
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._cache = payload
            else:
                self._cache = {}
        except (OSError, json.JSONDecodeError):
            self._cache = {}

    def flush_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache, ensure_ascii=True, sort_keys=True), encoding="utf-8")

    def _rate_limit(self) -> None:
        if self.min_interval_s <= 0:
            return
        now = time.monotonic()
        wait_s = self.min_interval_s - (now - self._last_request_monotonic)
        if wait_s > 0:
            self.rate_limit_sleeps += 1
            time.sleep(wait_s)
        self._last_request_monotonic = time.monotonic()

    def _request_json(
        self,
        *,
        cache_key: str,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if isinstance(cached, dict):
                self.cache_hits += 1
                if self._log is not None:
                    self._log(4, f"[http/cache-hit] {cache_key}")
                return cached

        full_url = url
        if params:
            query = urlencode(params, doseq=True)
            separator = "&" if "?" in full_url else "?"
            full_url = f"{full_url}{separator}{query}"

        last_error: str | None = None
        for attempt in range(3):
            self._rate_limit()
            try:
                self.http_requests += 1
                command = [
                    "curl",
                    "-sS",
                    "-L",
                    "--max-time",
                    str(max(1, int(round(self.timeout_s)))),
                    "-H",
                    "Accept: application/json",
                    "-H",
                    f"User-Agent: {self._user_agent}",
                    full_url,
                ]
                if self._log is not None:
                    self._log(4, f"[http/request] attempt={attempt + 1} url={full_url}")
                process = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if process.returncode != 0:
                    raise RuntimeError(process.stderr.strip() or f"curl exit {process.returncode}")

                payload = json.loads(process.stdout)
                entry = {"ok": True, "payload": payload}
                self._cache[cache_key] = entry
                self.http_successes += 1
                return entry
            except Exception as error:
                last_error = f"{error.__class__.__name__}: {str(error).strip()}"
                self.http_failures += 1
                if self._log is not None:
                    self._log(3, f"[http/error] attempt={attempt + 1} cache_key={cache_key} error={last_error}")
                if attempt < 2:
                    time.sleep(0.4 * (2**attempt))
                    continue

        return {"ok": False, "error": str(last_error or "request failed")}

    def request_stats(self) -> dict[str, int]:
        return {
            "cache_hits": int(self.cache_hits),
            "http_requests": int(self.http_requests),
            "http_successes": int(self.http_successes),
            "http_failures": int(self.http_failures),
            "rate_limit_sleeps": int(self.rate_limit_sleeps),
        }

    def summary(self, title: str) -> dict[str, Any] | None:
        normalized_title = str(title or "").strip()
        if not normalized_title:
            return None
        encoded = quote(normalized_title.replace(" ", "_"), safe="")
        url = WIKIPEDIA_SUMMARY_URL.format(title=encoded)
        cache_key = f"summary::{normalized_title.lower()}"
        response = self._request_json(cache_key=cache_key, url=url)
        if not response.get("ok"):
            return None
        payload = response.get("payload")
        if not isinstance(payload, dict):
            return None
        return payload

    def search(self, query: str, limit: int = 7) -> list[dict[str, Any]]:
        query_text = str(query or "").strip()
        if not query_text:
            return []

        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query_text,
            "srlimit": int(limit),
            "utf8": "1",
            "srprop": "snippet",
        }
        cache_key = f"search::{query_text.lower()}::{limit}"
        response = self._request_json(cache_key=cache_key, url=WIKIPEDIA_API_URL, params=params)
        if not response.get("ok"):
            return []
        payload = response.get("payload")
        if not isinstance(payload, dict):
            return []
        search_results = payload.get("query", {}).get("search", [])
        if not isinstance(search_results, list):
            return []
        return [item for item in search_results if isinstance(item, dict)]

    def intro_extract(self, title: str) -> str:
        normalized_title = str(title or "").strip()
        if not normalized_title:
            return ""

        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "titles": normalized_title,
        }
        cache_key = f"extract::{normalized_title.lower()}"
        response = self._request_json(cache_key=cache_key, url=WIKIPEDIA_API_URL, params=params)
        if not response.get("ok"):
            return ""
        payload = response.get("payload")
        if not isinstance(payload, dict):
            return ""
        pages = payload.get("query", {}).get("pages", [])
        if not isinstance(pages, list) or not pages:
            return ""
        page = pages[0]
        if not isinstance(page, dict):
            return ""
        return str(page.get("extract", "")).strip()

    def page_image(self, title: str) -> str:
        details = self.page_image_details(title)
        return str(details.get("image_url", "")).strip()

    def image_file_description_url(self, file_title: str) -> str:
        normalized_file = str(file_title or "").strip()
        if not normalized_file:
            return ""
        if not normalized_file.lower().startswith("file:"):
            normalized_file = f"File:{normalized_file}"

        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "imageinfo",
            "iiprop": "url",
            "titles": normalized_file,
        }
        cache_key = f"imageinfo::{normalized_file.lower()}"
        response = self._request_json(cache_key=cache_key, url=WIKIPEDIA_API_URL, params=params)
        if not response.get("ok"):
            return ""
        payload = response.get("payload")
        if not isinstance(payload, dict):
            return ""
        pages = payload.get("query", {}).get("pages", [])
        if not isinstance(pages, list) or not pages:
            return ""
        page = pages[0]
        if not isinstance(page, dict):
            return ""
        imageinfo = page.get("imageinfo", [])
        if not isinstance(imageinfo, list) or not imageinfo:
            return ""
        info = imageinfo[0]
        if not isinstance(info, dict):
            return ""
        description_url = str(info.get("descriptionurl", "")).strip()
        if description_url:
            return description_url
        return str(info.get("url", "")).strip()

    def page_image_details(self, title: str) -> dict[str, str]:
        normalized_title = str(title or "").strip()
        if not normalized_title:
            return {"image_url": "", "file_title": "", "attribution_url": ""}
        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
            "prop": "pageimages",
            "piprop": "name|original",
            "titles": normalized_title,
        }
        cache_key = f"pageimage_details::{normalized_title.lower()}"
        response = self._request_json(cache_key=cache_key, url=WIKIPEDIA_API_URL, params=params)
        if not response.get("ok"):
            return {"image_url": "", "file_title": "", "attribution_url": ""}
        payload = response.get("payload")
        if not isinstance(payload, dict):
            return {"image_url": "", "file_title": "", "attribution_url": ""}
        pages = payload.get("query", {}).get("pages", [])
        if not isinstance(pages, list) or not pages:
            return {"image_url": "", "file_title": "", "attribution_url": ""}
        page = pages[0]
        if not isinstance(page, dict):
            return {"image_url": "", "file_title": "", "attribution_url": ""}
        pageimage = str(page.get("pageimage", "")).strip()
        file_title = f"File:{pageimage}" if pageimage else ""
        original = page.get("original")
        image_url = ""
        if not isinstance(original, dict):
            image_url = ""
        else:
            image_url = str(original.get("source", "")).strip()
        attribution_url = self.image_file_description_url(file_title) if file_title else ""
        return {
            "image_url": image_url,
            "file_title": file_title,
            "attribution_url": attribution_url,
        }

    def wikitext(self, title: str) -> str:
        normalized_title = str(title or "").strip()
        if not normalized_title:
            return ""
        params = {
            "action": "parse",
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
            "prop": "wikitext",
            "page": normalized_title,
        }
        cache_key = f"wikitext::{normalized_title.lower()}"
        response = self._request_json(cache_key=cache_key, url=WIKIPEDIA_API_URL, params=params)
        if not response.get("ok"):
            return ""
        payload = response.get("payload")
        if not isinstance(payload, dict):
            return ""
        parsed = payload.get("parse")
        if not isinstance(parsed, dict):
            return ""
        return str(parsed.get("wikitext", "")).strip()


def _summary_to_url(summary_payload: dict[str, Any], fallback_title: str) -> str:
    content_urls = summary_payload.get("content_urls")
    if isinstance(content_urls, dict):
        desktop = content_urls.get("desktop")
        if isinstance(desktop, dict):
            page_url = str(desktop.get("page", "")).strip()
            if page_url:
                return page_url
    safe_title = quote(str(fallback_title).replace(" ", "_"), safe="")
    return f"https://en.wikipedia.org/wiki/{safe_title}"


def _summary_to_image(summary_payload: dict[str, Any]) -> str:
    original = summary_payload.get("originalimage")
    if isinstance(original, dict):
        source = str(original.get("source", "")).strip()
        if source:
            return source
    thumbnail = summary_payload.get("thumbnail")
    if isinstance(thumbnail, dict):
        source = str(thumbnail.get("source", "")).strip()
        if source:
            return source
    return ""


def _resolve_wikipedia_page(
    row: dict[str, Any],
    client: WikipediaClient,
    *,
    log: Callable[[int, str], None] | None = None,
) -> WikipediaLookup:
    def _log(level: int, message: str) -> None:
        if log is not None:
            log(level, message)

    identity_tokens = _build_identity_tokens(row)
    title_candidates = _build_title_candidates(row)
    search_queries = _build_search_queries(row, title_candidates)
    target_prefers_nebula = _row_prefers_nebula(row)
    messier_number = _messier_num(_strip_name_prefix(str(row.get("primary_id", "")).strip()))
    _log(3, f"[resolver] identity_tokens={identity_tokens}")
    _log(3, f"[resolver] title_candidates={title_candidates}")
    _log(3, f"[resolver] search_queries={search_queries}")
    _log(3, f"[resolver] target_prefers_nebula={target_prefers_nebula}")

    best = WikipediaLookup(match_method="none", match_score=-999)
    best_nebula = WikipediaLookup(match_method="none", match_score=-999)
    checked_titles: set[str] = set()

    # First pass: direct title attempts for high-precision matches.
    for idx, candidate_title in enumerate(title_candidates[:6]):
        candidate_title = str(candidate_title).strip()
        if not candidate_title:
            continue
        normalized = candidate_title.lower()
        if normalized in checked_titles:
            continue
        checked_titles.add(normalized)
        _log(3, f"[resolver/direct] candidate='{candidate_title}'")

        summary = client.summary(candidate_title)
        if not summary:
            _log(3, "[resolver/direct] no summary payload")
            continue

        title = str(summary.get("title", "")).strip() or candidate_title
        extract = str(summary.get("extract", "")).strip()
        short_description = str(summary.get("description", "")).strip()
        topic_label = _page_topic_label(title, extract, short_description)
        redirected = _normalize_text(title) != _normalize_text(candidate_title)
        score = _score_summary(title, extract, short_description, identity_tokens) + 15 - (idx * 2)
        _log(
            3,
            f"[resolver/direct] resolved_title='{title}' score={score} topic={topic_label} "
            f"redirected={redirected} desc='{_truncate(short_description, 80)}'",
        )
        candidate_lookup = WikipediaLookup(
            title=title,
            url=_summary_to_url(summary, title),
            description="",
            image_url="",
            designations=[],
            match_method="direct_title",
            match_score=score,
            query_used=candidate_title,
            requested_title=candidate_title,
            topic_label=topic_label,
            redirected_from_query=redirected,
            sentence_count=0,
        )

        if score > best.match_score:
            best = candidate_lookup
            _log(3, f"[resolver/direct] best_updated title='{title}' score={score}")
        if topic_label == "nebula" and score > best_nebula.match_score:
            best_nebula = candidate_lookup
            _log(3, f"[resolver/direct] nebula_best_updated title='{title}' score={score}")

        if (
            score >= 80
            and not _looks_disambiguation(title, extract)
            and (not target_prefers_nebula or topic_label == "nebula")
        ):
            _log(3, f"[resolver/direct] early_accept title='{title}' score={score}")
            break

    # Second pass: search fallback.
    needs_search = best.match_score < 80
    if target_prefers_nebula and best_nebula.match_score < 80:
        needs_search = True

    if needs_search:
        for query in search_queries[:5]:
            _log(3, f"[resolver/search] query='{query}'")
            results = client.search(query, limit=8)
            if not results:
                _log(3, "[resolver/search] no results")
                continue

            ranked = sorted(
                results,
                key=lambda result: _score_search_result(result, identity_tokens),
                reverse=True,
            )
            top_result = ranked[0]
            top_title = str(top_result.get("title", "")).strip()
            if not top_title:
                _log(3, "[resolver/search] top result missing title")
                continue
            normalized_top = top_title.lower()
            if normalized_top in checked_titles:
                _log(3, f"[resolver/search] top title already checked '{top_title}'")
                continue
            checked_titles.add(normalized_top)
            _log(3, f"[resolver/search] top_title='{top_title}'")

            summary = client.summary(top_title)
            if not summary:
                _log(3, "[resolver/search] summary missing for top title")
                continue

            query_used_for_candidate = query
            title = str(summary.get("title", "")).strip() or top_title
            extract = str(summary.get("extract", "")).strip()
            short_description = str(summary.get("description", "")).strip()
            topic_label = _page_topic_label(title, extract, short_description)
            redirected = _normalize_text(title) != _normalize_text(top_title)

            # For Messier shorthand queries like "M 31", retry with "Messier 31"
            # when the shorthand result does not appear to be a DSO page.
            if _is_m_shorthand_query(query, messier_number) and not _looks_messier_object_page(
                title,
                extract,
                short_description,
            ):
                fallback_query = f"Messier {messier_number}"
                _log(
                    3,
                    f"[resolver/search] shorthand_query='{query}' produced non-dso page; "
                    f"trying fallback='{fallback_query}'",
                )
                fallback_results = client.search(fallback_query, limit=8)
                if fallback_results:
                    fallback_ranked = sorted(
                        fallback_results,
                        key=lambda result: _score_search_result(result, identity_tokens),
                        reverse=True,
                    )
                    fallback_top = fallback_ranked[0]
                    fallback_title = str(fallback_top.get("title", "")).strip()
                    if fallback_title:
                        fallback_norm = fallback_title.lower()
                        if fallback_norm not in checked_titles:
                            checked_titles.add(fallback_norm)
                        fallback_summary = client.summary(fallback_title)
                        if fallback_summary:
                            top_result = fallback_top
                            top_title = fallback_title
                            summary = fallback_summary
                            query_used_for_candidate = fallback_query
                            title = str(summary.get("title", "")).strip() or top_title
                            extract = str(summary.get("extract", "")).strip()
                            short_description = str(summary.get("description", "")).strip()
                            topic_label = _page_topic_label(title, extract, short_description)
                            redirected = _normalize_text(title) != _normalize_text(top_title)
                            _log(
                                3,
                                f"[resolver/search] fallback resolved_title='{title}' topic={topic_label}",
                            )

            score = _score_summary(title, extract, short_description, identity_tokens)
            score += _score_search_result(top_result, identity_tokens)
            _log(
                3,
                f"[resolver/search] resolved_title='{title}' score={score} topic={topic_label} "
                f"redirected={redirected} desc='{_truncate(short_description, 80)}'",
            )
            candidate_lookup = WikipediaLookup(
                title=title,
                url=_summary_to_url(summary, title),
                description="",
                image_url="",
                designations=[],
                match_method="search",
                match_score=score,
                query_used=query_used_for_candidate,
                requested_title=top_title,
                topic_label=topic_label,
                redirected_from_query=redirected,
                sentence_count=0,
            )

            if score > best.match_score:
                best = candidate_lookup
                _log(3, f"[resolver/search] best_updated title='{title}' score={score}")
            if topic_label == "nebula" and score > best_nebula.match_score:
                best_nebula = candidate_lookup
                _log(3, f"[resolver/search] nebula_best_updated title='{title}' score={score}")

            if (
                score >= 85
                and not _looks_disambiguation(title, extract)
                and (not target_prefers_nebula or topic_label == "nebula")
            ):
                _log(3, f"[resolver/search] early_accept title='{title}' score={score}")
                break

    selected = best
    if target_prefers_nebula and best_nebula.match_score >= 30:
        selected = best_nebula
        if _normalize_text(selected.title) != _normalize_text(best.title):
            _log(
                2,
                f"[resolver/select] preferring nebula page title='{selected.title}' "
                f"over title='{best.title}'",
            )

    if selected.match_score < 40 or not selected.title:
        _log(2, f"[resolver/final] no match score={selected.match_score}")
        return WikipediaLookup(match_method="no_match", match_score=selected.match_score)

    chosen_summary = client.summary(selected.title) or {}
    chosen_extract = str(chosen_summary.get("extract", "")).strip()
    chosen_short_desc = str(chosen_summary.get("description", "")).strip()
    info_url = selected.url or _summary_to_url(chosen_summary, selected.title)
    intro_extract = client.intro_extract(selected.title) if len(_sentence_list(chosen_extract)) < 3 else ""
    description = _description_3_to_4_sentences(chosen_extract, intro_extract)

    image_url = _summary_to_image(chosen_summary)
    image_details = client.page_image_details(selected.title)
    if not image_url:
        image_url = str(image_details.get("image_url", "")).strip()
    image_file_title = str(image_details.get("file_title", "")).strip()
    image_attribution_url = str(image_details.get("attribution_url", "")).strip()
    if not image_file_title and image_url:
        image_file_title = _file_title_from_image_url(image_url)
    if not image_attribution_url and image_file_title:
        image_attribution_url = client.image_file_description_url(image_file_title)
    if not image_attribution_url:
        image_attribution_url = info_url

    wikitext = client.wikitext(selected.title)
    designations = _extract_designations_from_wikitext(wikitext)
    infobox_entries = _extract_infobox_entries(wikitext)
    searched_name = _strip_name_prefix(
        selected.requested_title or selected.query_used or str(row.get("primary_id", "")).strip()
    )
    if (
        target_prefers_nebula
        and selected.topic_label == "star"
        and selected.redirected_from_query
        and searched_name
        and _normalize_text(searched_name) != _normalize_text(selected.title)
    ):
        description = _append_sentence_if_missing(
            description,
            f"{selected.title} is an object related to {searched_name}",
        )
        designations = _dedupe_preserve_order([selected.title, *designations])
        _log(
            2,
            f"[resolver/final] applied star-redirect relation note redirected='{selected.title}' "
            f"searched='{searched_name}'",
        )

    catalog_primary_id = str(row.get("primary_id", "")).strip()
    common_name = _extract_common_name(
        entries=infobox_entries,
        title=selected.title,
        primary_id=catalog_primary_id,
    )
    object_type = _extract_object_type(
        entries=infobox_entries,
        short_description=chosen_short_desc,
        topic_label=selected.topic_label,
    )
    ra_deg, dec_deg = _extract_ra_dec_from_infobox(infobox_entries)
    constellation = _extract_constellation(infobox_entries)
    emission_lines = _extract_emission_lines(infobox_entries)
    aliases = _dedupe_preserve_order([*designations, common_name, _strip_parenthetical_suffix(selected.title)])
    aliases = [alias for alias in aliases if _normalize_text(alias) != _normalize_text(catalog_primary_id)]
    wikipedia_primary_id, wikipedia_catalog = _extract_wikipedia_primary_catalog_reference(
        title=selected.title,
        aliases=aliases,
        designations=designations,
    )

    sentence_count = len(_sentence_list(description))
    _log(
        2,
        f"[resolver/final] title='{selected.title}' method={selected.match_method} score={selected.match_score} "
        f"topic={selected.topic_label} redirected={selected.redirected_from_query} "
        f"sentences={sentence_count} image={'yes' if bool(image_url) else 'no'} "
        f"designations={len(designations)} aliases={len(aliases)} "
        f"short_desc='{_truncate(chosen_short_desc, 80)}'",
    )

    return WikipediaLookup(
        title=selected.title,
        url=info_url,
        info_url=info_url,
        description=description,
        image_url=image_url,
        image_attribution_url=image_attribution_url,
        designations=designations,
        aliases=aliases,
        common_name=common_name,
        object_type=object_type,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        constellation=constellation,
        emission_lines=emission_lines,
        wikipedia_primary_id=wikipedia_primary_id,
        wikipedia_catalog=wikipedia_catalog,
        match_method=selected.match_method,
        match_score=selected.match_score,
        query_used=selected.query_used,
        requested_title=selected.requested_title,
        topic_label=selected.topic_label,
        redirected_from_query=selected.redirected_from_query,
        sentence_count=sentence_count,
    )


def _load_targets(
    *,
    catalog_path: Path,
    include_catalogs: tuple[str, ...],
    include_groups: tuple[str, ...],
    include_primary_ids: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.read_parquet(catalog_path)
    if "primary_id" not in frame.columns:
        raise ValueError("Catalog is missing required column: primary_id")

    catalogs_upper = {value.strip().upper() for value in include_catalogs if value.strip()}
    groups_lower = {value.strip().lower() for value in include_groups if value.strip()}
    requested_primary_ids = _dedupe_preserve_order(
        [
            _canonicalize_requested_primary_id(value)
            for value in include_primary_ids
            if str(value or "").strip()
        ]
    )
    requested_primary_keys = {_normalize_text(value) for value in requested_primary_ids if value}

    catalog_col = frame.get("catalog", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    group_col = (
        frame.get("object_type_group", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    base_mask = catalog_col.isin(catalogs_upper) | group_col.isin(groups_lower)
    if requested_primary_keys:
        primary_key_col = frame.get("primary_id", pd.Series("", index=frame.index)).fillna("").astype(str).map(
            lambda value: _normalize_text(" ".join(str(value or "").strip().split()))
        )
        base_mask = base_mask | primary_key_col.isin(requested_primary_keys)

    selected = frame.loc[base_mask].copy()

    if requested_primary_ids:
        selected_key_set = {
            _normalize_text(" ".join(str(value or "").strip().split()))
            for value in selected.get("primary_id", pd.Series("", index=selected.index)).tolist()
            if str(value or "").strip()
        }

        synthetic_records: list[dict[str, Any]] = []
        for requested_primary_id in requested_primary_ids:
            normalized_requested = _normalize_text(requested_primary_id)
            if not normalized_requested:
                continue
            if normalized_requested in selected_key_set:
                continue
            selected_key_set.add(normalized_requested)

            synthetic: dict[str, Any] = {}
            for column in frame.columns:
                if pd.api.types.is_numeric_dtype(frame[column].dtype):
                    synthetic[column] = float("nan")
                else:
                    synthetic[column] = ""

            synthetic["primary_id"] = requested_primary_id
            synthetic["catalog"] = _identifier_catalog(requested_primary_id) or "WIKI"
            synthetic["common_name"] = ""
            synthetic["object_type_group"] = ""
            synthetic["aliases"] = ""
            synthetic_records.append(synthetic)

        if synthetic_records:
            selected = pd.concat(
                [selected, pd.DataFrame.from_records(synthetic_records, columns=list(frame.columns))],
                ignore_index=True,
                sort=False,
            )

    selected["primary_id"] = selected["primary_id"].fillna("").astype(str).str.strip()
    selected = selected[selected["primary_id"] != ""]
    selected = selected.drop_duplicates(subset=["primary_id"], keep="first")
    selected = selected.sort_values(by=["catalog", "primary_id"], ascending=[True, True]).reset_index(drop=True)
    return selected


def _parse_csv_list(raw: str) -> tuple[str, ...]:
    parts = [item.strip() for item in str(raw or "").split(",")]
    return tuple([item for item in parts if item])


EXPECTED_CATALOG_ENRICHMENT_KEYS = (
    "common_name",
    "object_type",
    "ra_deg",
    "dec_deg",
    "constellation",
    "aliases",
    "image_url",
    "image_attribution_url",
    "description",
    "info_url",
    "emission_lines",
    "wikipedia_primary_id",
    "wikipedia_catalog",
)


def _has_catalog_enrichment(row: dict[str, Any]) -> bool:
    payload = row.get("catalog_enrichment")
    if not isinstance(payload, dict):
        return False
    return all(key in payload for key in EXPECTED_CATALOG_ENRICHMENT_KEYS)


def _make_logger(verbosity: int) -> Callable[[int, str], None]:
    def _log(level: int, message: str) -> None:
        if verbosity >= level:
            print(message, flush=True)

    return _log


def _load_existing_rows(
    output_path: Path,
    *,
    log: Callable[[int, str], None],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    if not output_path.exists():
        return {}, {
            "loaded_rows": 0,
            "invalid_rows": 0,
            "duplicate_primary_ids": 0,
        }

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        log(1, f"[resume] unable to parse existing output '{output_path}': {error}")
        return {}, {
            "loaded_rows": 0,
            "invalid_rows": 0,
            "duplicate_primary_ids": 0,
        }

    if not isinstance(payload, dict):
        log(1, f"[resume] existing output is not a JSON object: {output_path}")
        return {}, {
            "loaded_rows": 0,
            "invalid_rows": 0,
            "duplicate_primary_ids": 0,
        }

    source_rows = payload.get("targets", [])
    if not isinstance(source_rows, list):
        log(1, f"[resume] existing output has no 'targets' array: {output_path}")
        return {}, {
            "loaded_rows": 0,
            "invalid_rows": 0,
            "duplicate_primary_ids": 0,
        }

    loaded: dict[str, dict[str, Any]] = {}
    invalid_rows = 0
    duplicate_primary_ids = 0
    for row in source_rows:
        if not isinstance(row, dict):
            invalid_rows += 1
            continue
        primary_id = str(row.get("primary_id", "")).strip()
        if not primary_id:
            invalid_rows += 1
            continue
        if primary_id in loaded:
            duplicate_primary_ids += 1
            continue
        loaded[primary_id] = row

    return loaded, {
        "loaded_rows": int(len(loaded)),
        "invalid_rows": int(invalid_rows),
        "duplicate_primary_ids": int(duplicate_primary_ids),
    }


def _compute_row_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    matched = 0
    no_match = 0
    with_image = 0
    with_designations = 0
    with_3plus_sentences = 0
    unknown_status = 0

    for row in rows:
        status = str(row.get("status", "")).strip().lower()
        if status == "matched":
            matched += 1
        elif status == "no_match":
            no_match += 1
        else:
            unknown_status += 1

        if str(row.get("image_url", "")).strip():
            with_image += 1

        designations = row.get("designations", [])
        if isinstance(designations, list) and len(designations) > 0:
            with_designations += 1

        sentence_count_raw = row.get("description_sentence_count", 0)
        try:
            sentence_count = int(sentence_count_raw)
        except (TypeError, ValueError):
            sentence_count = 0
        if sentence_count >= 3:
            with_3plus_sentences += 1

    return {
        "matched": int(matched),
        "no_match": int(no_match),
        "unknown_status": int(unknown_status),
        "with_image": int(with_image),
        "with_designations": int(with_designations),
        "with_3plus_sentence_description": int(with_3plus_sentences),
    }


def _build_payload(
    *,
    generated_at_utc: str,
    catalog_path: Path,
    include_catalogs: tuple[str, ...],
    include_groups: tuple[str, ...],
    include_primary_ids: tuple[str, ...],
    target_count: int,
    rows: list[dict[str, Any]],
    stats: dict[str, int],
    resume_enabled: bool,
    resume_retry_no_match: bool,
    resume_rows_reused: int,
    rows_processed_this_run: int,
    duration_seconds: float,
    http_stats: dict[str, int],
    verbosity: int,
) -> dict[str, Any]:
    return {
        "generated_at_utc": generated_at_utc,
        "catalog_path": str(catalog_path),
        "selection": {
            "catalogs": list(include_catalogs),
            "object_type_groups": list(include_groups),
            "primary_ids": list(include_primary_ids),
            "target_count": int(target_count),
        },
        "run": {
            "resume_enabled": bool(resume_enabled),
            "resume_retry_no_match": bool(resume_retry_no_match),
            "resume_rows_reused": int(resume_rows_reused),
            "rows_processed_this_run": int(rows_processed_this_run),
            "duration_seconds": round(float(duration_seconds), 3),
            "verbosity": int(verbosity),
        },
        "stats": stats,
        "http": http_stats,
        "targets": rows,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build standalone Wikipedia enrichment JSON for selected catalog targets.",
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=Path("data/dso_catalog_cache.parquet"),
        help="Input catalog parquet path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/wikipedia_catalog_enrichment.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/wikipedia_api_cache.json"),
        help="HTTP response cache JSON path.",
    )
    parser.add_argument(
        "--include-catalogs",
        type=str,
        default=",".join(DEFAULT_INCLUDE_CATALOGS),
        help="Comma-separated catalog filters (upper/lower insensitive).",
    )
    parser.add_argument(
        "--include-groups",
        type=str,
        default=",".join(DEFAULT_INCLUDE_GROUPS),
        help="Comma-separated object_type_group filters (upper/lower insensitive).",
    )
    parser.add_argument(
        "--include-primary-ids",
        type=str,
        default="",
        help=(
            "Comma-separated primary IDs to include in addition to catalog/group "
            "selection (case-insensitive)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max target count (0 means no limit).",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=4.0,
        help="Wikipedia API request rate cap.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing output JSON when present (default: enabled).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume mode and rebuild output from scratch.",
    )
    parser.add_argument(
        "--retry-no-match",
        action="store_true",
        help="When resuming, reprocess rows previously marked as no_match.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write intermediate output and flush API cache every N newly processed rows (0 disables).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print rollup progress every N targets.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        help=(
            "Logging verbosity (0=quiet, 1=milestones, 2=per-target, "
            "3=resolver internals, 4=HTTP/cache internals)."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_started = time.monotonic()
    verbosity = max(0, int(args.verbosity))
    log = _make_logger(verbosity)

    include_catalogs = _parse_csv_list(args.include_catalogs)
    include_groups = _parse_csv_list(args.include_groups)
    include_primary_ids = _parse_csv_list(args.include_primary_ids)

    log(1, "[startup] wikipedia enrichment run")
    log(
        1,
        "[startup] config "
        f"catalog_path={args.catalog_path} output_path={args.output_path} cache_path={args.cache_path} "
        f"resume={args.resume} retry_no_match={args.retry_no_match} "
        f"limit={args.limit} rps={args.requests_per_second} timeout_s={args.timeout_s} "
        f"checkpoint_every={args.checkpoint_every} progress_every={args.progress_every} verbosity={verbosity} "
        f"include_primary_ids={len(include_primary_ids)}",
    )

    targets = _load_targets(
        catalog_path=args.catalog_path,
        include_catalogs=include_catalogs,
        include_groups=include_groups,
        include_primary_ids=include_primary_ids,
    )
    if args.limit and args.limit > 0:
        targets = targets.head(int(args.limit)).copy()
        log(1, f"[startup] applied limit: {args.limit}")

    resume_rows_by_id: dict[str, dict[str, Any]] = {}
    resume_meta: dict[str, int] = {
        "loaded_rows": 0,
        "invalid_rows": 0,
        "duplicate_primary_ids": 0,
    }
    if args.resume:
        resume_rows_by_id, resume_meta = _load_existing_rows(args.output_path, log=log)
        log(
            1,
            "[resume] loaded "
            f"rows={resume_meta['loaded_rows']} invalid={resume_meta['invalid_rows']} "
            f"duplicates={resume_meta['duplicate_primary_ids']}",
        )
    else:
        log(1, "[resume] disabled")

    client = WikipediaClient(
        timeout_s=float(args.timeout_s),
        requests_per_second=float(args.requests_per_second),
        cache_path=args.cache_path,
        log=log,
    )

    rows: list[dict[str, Any]] = []
    resume_rows_reused = 0
    rows_processed_this_run = 0

    total = len(targets)
    log(1, f"[startup] selected targets: {total}")
    if total == 0:
        log(1, "[startup] no targets selected; writing empty payload")

    for idx, (_, target_row) in enumerate(targets.iterrows(), start=1):
        row = target_row.to_dict()
        primary_id = str(row.get("primary_id", "")).strip()
        catalog = str(row.get("catalog", "")).strip()
        object_type_group = str(row.get("object_type_group", "")).strip()
        common_name = str(row.get("common_name", "")).strip()

        existing = resume_rows_by_id.get(primary_id)
        if existing is not None:
            existing_status = str(existing.get("status", "")).strip().lower()
            should_retry_no_match = args.retry_no_match and existing_status == "no_match"
            missing_catalog_enrichment = not _has_catalog_enrichment(existing)
            if not should_retry_no_match and not missing_catalog_enrichment:
                rows.append(existing)
                resume_rows_reused += 1
                log(
                    2,
                    f"[{idx}/{total}] reuse primary_id={primary_id} status={existing_status or 'unknown'} "
                    f"catalog={catalog}",
                )

                if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == total):
                    snapshot = _compute_row_stats(rows)
                    log(
                        1,
                        f"[progress {idx}/{total}] matched={snapshot['matched']} no_match={snapshot['no_match']} "
                        f"with_image={snapshot['with_image']} with_designations={snapshot['with_designations']} "
                        f"reused={resume_rows_reused} new={rows_processed_this_run}",
                        )
                continue
            if missing_catalog_enrichment:
                log(2, f"[{idx}/{total}] retry-missing-enrichment primary_id={primary_id} catalog={catalog}")
            elif should_retry_no_match:
                log(2, f"[{idx}/{total}] retry-no-match primary_id={primary_id} catalog={catalog}")

        log(
            2,
            f"[{idx}/{total}] process primary_id={primary_id} catalog={catalog} "
            f"group='{object_type_group}' common_name='{_truncate(common_name, 60)}'",
        )

        lookup = _resolve_wikipedia_page(row, client, log=log)
        status = "matched" if lookup.match_method not in {"no_match", "none"} and lookup.title else "no_match"

        rows.append(
            {
                "primary_id": primary_id,
                "catalog": catalog,
                "object_type_group": object_type_group,
                "common_name": common_name,
                "status": status,
                "match_method": lookup.match_method,
                "match_score": int(lookup.match_score),
                "query_used": lookup.query_used,
                "wikipedia_title": lookup.title,
                "wikipedia_url": lookup.url,
                "description": lookup.description,
                "description_sentence_count": int(lookup.sentence_count),
                "image_url": lookup.image_url,
                "designations": lookup.designations,
                "catalog_enrichment": {
                    "common_name": lookup.common_name,
                    "object_type": lookup.object_type,
                    "ra_deg": lookup.ra_deg,
                    "dec_deg": lookup.dec_deg,
                    "constellation": lookup.constellation,
                    "aliases": lookup.aliases,
                    "image_url": lookup.image_url,
                    "image_attribution_url": lookup.image_attribution_url,
                    "description": lookup.description,
                    "info_url": lookup.info_url or lookup.url,
                    "emission_lines": lookup.emission_lines,
                    "wikipedia_primary_id": lookup.wikipedia_primary_id,
                    "wikipedia_catalog": lookup.wikipedia_catalog,
                },
            }
        )
        rows_processed_this_run += 1

        log(
            2,
            f"[{idx}/{total}] result status={status} method={lookup.match_method} score={lookup.match_score} "
            f"title='{_truncate(lookup.title, 70)}' image={'yes' if bool(lookup.image_url) else 'no'} "
            f"designations={len(lookup.designations)} sentences={lookup.sentence_count}",
        )

        if args.checkpoint_every > 0 and (rows_processed_this_run % args.checkpoint_every == 0):
            interim_stats = _compute_row_stats(rows)
            interim_payload = _build_payload(
                generated_at_utc=_utc_now_iso(),
                catalog_path=args.catalog_path,
                include_catalogs=include_catalogs,
                include_groups=include_groups,
                include_primary_ids=include_primary_ids,
                target_count=total,
                rows=rows,
                stats=interim_stats,
                resume_enabled=bool(args.resume),
                resume_retry_no_match=bool(args.retry_no_match),
                resume_rows_reused=resume_rows_reused,
                rows_processed_this_run=rows_processed_this_run,
                duration_seconds=(time.monotonic() - run_started),
                http_stats=client.request_stats(),
                verbosity=verbosity,
            )
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            args.output_path.write_text(json.dumps(interim_payload, indent=2, ensure_ascii=True), encoding="utf-8")
            client.flush_cache()
            log(
                1,
                f"[checkpoint] wrote rows={len(rows)}/{total} reused={resume_rows_reused} new={rows_processed_this_run}",
            )

        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == total):
            snapshot = _compute_row_stats(rows)
            log(
                1,
                f"[progress {idx}/{total}] matched={snapshot['matched']} no_match={snapshot['no_match']} "
                f"with_image={snapshot['with_image']} with_designations={snapshot['with_designations']} "
                f"reused={resume_rows_reused} new={rows_processed_this_run}",
            )

    final_stats = _compute_row_stats(rows)
    payload = _build_payload(
        generated_at_utc=_utc_now_iso(),
        catalog_path=args.catalog_path,
        include_catalogs=include_catalogs,
        include_groups=include_groups,
        include_primary_ids=include_primary_ids,
        target_count=total,
        rows=rows,
        stats=final_stats,
        resume_enabled=bool(args.resume),
        resume_retry_no_match=bool(args.retry_no_match),
        resume_rows_reused=resume_rows_reused,
        rows_processed_this_run=rows_processed_this_run,
        duration_seconds=(time.monotonic() - run_started),
        http_stats=client.request_stats(),
        verbosity=verbosity,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    client.flush_cache()

    log(
        1,
        "[done] "
        f"rows_total={len(rows)} reused={resume_rows_reused} new={rows_processed_this_run} "
        f"matched={final_stats['matched']} no_match={final_stats['no_match']} "
        f"with_image={final_stats['with_image']} with_designations={final_stats['with_designations']}",
    )
    log(1, f"[done] wrote enrichment JSON: {args.output_path}")
    log(1, f"[done] wrote API cache JSON: {args.cache_path}")
    log(1, f"[done] http stats: {client.request_stats()}")


if __name__ == "__main__":
    main()
