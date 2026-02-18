from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable, Mapping

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def subset_by_id_list(frame: pd.DataFrame, ordered_ids: list[str]) -> pd.DataFrame:
    if not ordered_ids:
        return frame.iloc[0:0].copy()

    id_rank = {identifier: idx for idx, identifier in enumerate(ordered_ids)}
    subset = frame[frame["primary_id"].isin(id_rank)].copy()
    if subset.empty:
        return subset

    subset["_rank"] = subset["primary_id"].map(id_rank)
    subset = subset.sort_values(by="_rank", ascending=True).drop(columns=["_rank"]).reset_index(drop=True)
    return subset


def format_search_suggestion(
    row: pd.Series,
    *,
    is_listed: bool,
    wind16_arrows: Mapping[str, str],
) -> str:
    primary_id = str(row.get("primary_id") or "").strip() or "Unknown ID"
    common_name = str(row.get("common_name") or "").strip()
    object_type = str(row.get("object_type") or "").strip() or "Unknown type"
    wind = str(row.get("wind16") or "").strip() or "--"
    arrow = str(wind16_arrows.get(wind, " ")) or " "
    alt_now = row.get("alt_now")
    try:
        alt_text = f"{float(alt_now):.0f} deg"
    except (TypeError, ValueError):
        alt_text = "--"

    id_and_name = f"{primary_id} - {common_name}" if common_name else primary_id
    listed_suffix = " [list]" if is_listed else ""
    return f"{id_and_name} • {object_type} • {arrow} {wind} {alt_text}{listed_suffix}"


def catalog_search_rank(catalog_value: str | None) -> int:
    norm = normalize_text(catalog_value)
    if norm in {"m", "messier"} or norm.startswith("messier"):
        return 0
    if norm.startswith("sha") or norm.startswith("sh2") or norm.startswith("sh") or norm.startswith("sharpless"):
        return 1
    if norm.startswith("ic"):
        return 2
    if norm.startswith("ngc"):
        return 3
    return 4


def object_type_group_search_rank(object_type_group_value: str | None) -> int:
    norm = normalize_text(object_type_group_value)
    if norm in {"galaxy", "galaxies"}:
        return 0
    if norm == "brightnebula":
        return 1
    if norm == "darknebula":
        return 2
    if norm == "clusters":
        return 3
    if norm == "stars":
        return 4
    if norm == "other":
        return 5
    return 6


def compute_abs_minutes_to_culmination(targets: pd.DataFrame, lon: float) -> pd.Series:
    if targets.empty or "ra_deg" not in targets.columns:
        return pd.Series(np.inf, index=targets.index, dtype=float)

    try:
        now_utc = Time(datetime.now(timezone.utc))
        lst_deg = float(now_utc.sidereal_time("apparent", longitude=float(lon) * u.deg).degree) % 360.0
    except Exception:
        return pd.Series(np.inf, index=targets.index, dtype=float)

    ra_values = pd.to_numeric(targets["ra_deg"], errors="coerce").to_numpy(dtype=float)
    abs_minutes = np.full(len(targets), np.inf, dtype=float)
    valid = np.isfinite(ra_values)
    if np.any(valid):
        hour_angle_deg = ((lst_deg - ra_values[valid] + 540.0) % 360.0) - 180.0
        abs_minutes[valid] = np.abs(hour_angle_deg) * 4.0

    return pd.Series(abs_minutes, index=targets.index, dtype=float)


def searchbox_target_options(
    search_term: str,
    *,
    catalog: pd.DataFrame,
    lat: float,
    lon: float,
    listed_ids: list[str] | set[str] | None,
    max_options: int,
    wind16_arrows: Mapping[str, str],
    search_catalog_fn: Callable[[pd.DataFrame, str], pd.DataFrame],
    compute_altaz_now_fn: Callable[[pd.DataFrame, float, float], pd.DataFrame],
) -> list[tuple[str, str]]:
    query = str(search_term or "").strip()
    if not query:
        return []

    listed_list = [str(value) for value in (listed_ids or [])]
    listed_set = set(listed_list)
    query_norm = normalize_text(query)
    list_query_terms = {"list", "lists", "listed"}
    if query_norm in list_query_terms:
        matches = subset_by_id_list(catalog, listed_list).copy()
    else:
        matches = search_catalog_fn(catalog, query).copy()
        if "object_type_group" in catalog.columns and query_norm:
            group_norm = catalog["object_type_group"].fillna("").astype(str).map(normalize_text)
            group_matches = catalog[group_norm.str.contains(query_norm, regex=False)].copy()
            if not group_matches.empty:
                if matches.empty:
                    matches = group_matches
                else:
                    matches = (
                        pd.concat([matches, group_matches], ignore_index=True)
                        .drop_duplicates(subset=["primary_id"], keep="first")
                        .reset_index(drop=True)
                    )
    if matches.empty:
        return []

    matches = compute_altaz_now_fn(matches, lat=lat, lon=lon)
    if query_norm not in list_query_terms:
        matches["_listed_rank"] = np.where(matches["primary_id"].astype(str).isin(listed_set), 0, 1)
        matches["_horizon_rank"] = np.where(
            pd.to_numeric(matches["alt_now"], errors="coerce").fillna(-9999.0) >= 0.0,
            0,
            1,
        )
        matches["_catalog_rank"] = matches["catalog"].map(catalog_search_rank)
        matches["_type_group_rank"] = matches["object_type_group"].map(object_type_group_search_rank)
        matches["_altitude_sort"] = pd.to_numeric(matches["alt_now"], errors="coerce").fillna(-9999.0)
        matches["_culm_abs_minutes"] = compute_abs_minutes_to_culmination(matches, lon=lon)
        matches = matches.sort_values(
            by=[
                "_listed_rank",
                "_horizon_rank",
                "_catalog_rank",
                "_type_group_rank",
                "_altitude_sort",
                "_culm_abs_minutes",
                "primary_id",
            ],
            ascending=[True, True, True, True, False, True, True],
            kind="stable",
        ).drop(
            columns=[
                "_listed_rank",
                "_horizon_rank",
                "_catalog_rank",
                "_type_group_rank",
                "_altitude_sort",
                "_culm_abs_minutes",
            ]
        )
    matches = matches.head(max_options)

    options: list[tuple[str, str]] = []
    for _, row in matches.iterrows():
        primary_id = str(row.get("primary_id") or "")
        options.append(
            (
                format_search_suggestion(row, is_listed=primary_id in listed_set, wind16_arrows=wind16_arrows),
                primary_id,
            )
        )
    return options
