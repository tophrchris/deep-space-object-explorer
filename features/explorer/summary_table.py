from __future__ import annotations

from datetime import datetime, time as dt_time
import html
import re

from runtime.brightness_policy import (
    STATUS_BROADBAND_BORDERLINE,
    STATUS_IN_RANGE,
    STATUS_NARROWBAND_BOOSTED,
    STATUS_OUT_OF_RANGE,
    STATUS_UNKNOWN,
    classify_target_magnitude,
    resolve_telescope_max_magnitude,
)

# Transitional bridge during Explorer split: this module still relies on shared
# helpers/constants from `ui.streamlit_app` until they are extracted.
from ui import streamlit_app as _legacy_ui

def _refresh_legacy_globals() -> None:
    for _name, _value in vars(_legacy_ui).items():
        if _name == "__builtins__":
            continue
        globals().setdefault(_name, _value)


_refresh_legacy_globals()

SKY_SUMMARY_TABLE_MODE_STATE_KEY = "sky_summary_table_mode"
TARGET_TIPS_SCHEDULES_STATE_KEY = "target_tips_schedule_by_target_id"
TARGET_TIPS_SCHEDULE_ACTIVE_LIST_STATE_KEY = "target_tips_schedule_active_list_id"


def _parse_schedule_clock(value: Any) -> dt_time | None:
    if isinstance(value, datetime):
        return value.time().replace(second=0, microsecond=0)
    if isinstance(value, dt_time):
        return value.replace(second=0, microsecond=0)
    text = str(value or "").strip()
    if not text:
        return None
    for pattern in ("%H:%M", "%H:%M:%S"):
        try:
            parsed = datetime.strptime(text, pattern).time()
            return parsed.replace(second=0, microsecond=0)
        except ValueError:
            continue
    return None


def _format_schedule_duration_hhmm(total_minutes: int) -> str:
    bounded_minutes = max(0, int(total_minutes))
    hours, minutes = divmod(bounded_minutes, 60)
    if hours > 0 and minutes > 0:
        return f"{hours} {'hour' if hours == 1 else 'hours'}, {minutes} {'min' if minutes == 1 else 'mins'}"
    if hours > 0:
        return f"{hours} {'hour' if hours == 1 else 'hours'}"
    return f"{minutes} {'min' if minutes == 1 else 'mins'}"


def _parse_duration_minutes(value: Any) -> float | None:
    text = str(value or "").strip().lower()
    if not text or text == "--":
        return None
    if ":" in text:
        parts = [part.strip() for part in text.split(":", maxsplit=1)]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return float((int(parts[0]) * 60) + int(parts[1]))

    match = re.match(r"^\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*$", text)
    if match is None:
        verbose_match = re.match(
            r"^\s*(?:(\d+)\s*hours?)?\s*(?:,\s*)?(?:(\d+)\s*mins?)?\s*$",
            text,
        )
        if verbose_match is None:
            return None
        hours = int(verbose_match.group(1) or 0)
        minutes = int(verbose_match.group(2) or 0)
        if hours == 0 and minutes == 0 and text not in {"0 min", "0 mins", "0 hour", "0 hours"}:
            return None
        return float((hours * 60) + minutes)
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    if hours == 0 and minutes == 0 and text not in {"0m", "0h", "0h 0m", "0h0m"}:
        # Reject non-duration strings that regex-collapsed to zero.
        return None
    return float((hours * 60) + minutes)


def _remaining_minutes_for_sort(row: pd.Series) -> float | None:
    remaining_minutes = _parse_duration_minutes(row.get("visible_remaining"))
    if remaining_minutes is not None:
        return remaining_minutes
    return _parse_duration_minutes(row.get("visible_total"))


def _thumbnail_numeric(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _thumbnail_cutout_fov_deg(major_arcmin: Any, minor_arcmin: Any) -> tuple[float, float]:
    major_value = _thumbnail_numeric(major_arcmin)
    minor_value = _thumbnail_numeric(minor_arcmin)
    span_deg_candidates = [
        float(value) / 60.0
        for value in (major_value, minor_value)
        if value is not None and value > 0.0
    ]
    if span_deg_candidates:
        estimated_span = max(span_deg_candidates) * 3.0
        framed_span = float(max(0.5, min(8.0, estimated_span)))
        return framed_span, framed_span
    return 1.5, 1.5


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def _resolve_summary_thumbnail_url_cached(
    primary_id: str,
    common_name: str,
    image_url: str,
    hero_image_url: str,
    ra_deg: float | None,
    dec_deg: float | None,
    ang_size_maj_arcmin: float | None,
    ang_size_min_arcmin: float | None,
) -> str | None:
    _refresh_legacy_globals()
    for raw_value in (str(image_url or "").strip(), str(hero_image_url or "").strip()):
        if raw_value.lower().startswith(("https://", "http://")):
            return raw_value

    wikimedia_search_phrase = str(common_name or "").strip() or str(primary_id or "").strip()
    if wikimedia_search_phrase:
        image_data = fetch_free_use_image(wikimedia_search_phrase)
        wiki_image_url = str((image_data or {}).get("image_url", "") or "").strip()
        if wiki_image_url.lower().startswith(("https://", "http://")):
            return wiki_image_url

    ra_value = _thumbnail_numeric(ra_deg)
    dec_value = _thumbnail_numeric(dec_deg)
    if ra_value is None or dec_value is None:
        return None

    fov_width_deg, fov_height_deg = _thumbnail_cutout_fov_deg(ang_size_maj_arcmin, ang_size_min_arcmin)
    for layer in ("unwise-neo4", "sfd", "sdss2"):
        legacy_cutout = build_legacy_survey_cutout_urls(
            ra_deg=ra_value,
            dec_deg=dec_value,
            fov_width_deg=fov_width_deg,
            fov_height_deg=fov_height_deg,
            layer=layer,
            max_pixels=256,
        )
        legacy_image_url = str((legacy_cutout or {}).get("image_url", "") or "").strip()
        if legacy_image_url.lower().startswith(("https://", "http://")):
            return legacy_image_url
    return None


def render_sky_position_summary_table(
    rows: list[dict[str, Any]],
    prefs: dict[str, Any],
    use_12_hour: bool,
    *,
    preview_list_id: str,
    preview_list_name: str,
    allow_list_membership_toggle: bool,
    show_remaining: bool = False,
    now_local: pd.Timestamp | datetime | None = None,
) -> None:
    _refresh_legacy_globals()
    if not rows:
        st.session_state["sky_summary_highlight_primary_id"] = ""
        return

    title_col, mode_col = st.columns([3, 1], gap="small")
    title_col.markdown("#### Targets")
    selected_mode = mode_col.segmented_control(
        "Targets Mode",
        options=["Info", "Schedule"],
        default="Info",
        key=SKY_SUMMARY_TABLE_MODE_STATE_KEY,
        label_visibility="collapsed",
    )
    schedule_mode_active = str(selected_mode or "Info").strip().lower() == "schedule"
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        st.session_state["sky_summary_highlight_primary_id"] = ""
        return

    def _safe_positive_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric <= 0.0:
            return None
        return float(numeric)

    def _safe_finite_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return float(numeric)

    def _parse_emission_band_set_local(value: Any) -> set[str]:
        parser = globals().get("parse_emission_band_set")
        if not callable(parser):
            return set()
        try:
            parsed = parser(value)
        except Exception:
            return set()
        if isinstance(parsed, set):
            return {str(token).strip() for token in parsed if str(token).strip()}
        if isinstance(parsed, (list, tuple)):
            return {str(token).strip() for token in parsed if str(token).strip()}
        return set()

    def _target_emission_band_set(value: Any) -> set[str]:
        if isinstance(value, set):
            return {str(token).strip() for token in value if str(token).strip()}
        if isinstance(value, (list, tuple)):
            return {str(token).strip() for token in value if str(token).strip()}
        return _parse_emission_band_set_local(value)

    def _target_has_emissions_data(value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        return text.lower() not in {"-", "none", "nan"}

    def _is_hii_object_type(value: Any) -> bool:
        compact = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
        return compact in {"hii", "hiiregion"}

    def _is_bright_nebula_group(value: Any) -> bool:
        compact = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
        return compact == "brightnebula"

    def _target_is_narrowband_target(
        *,
        object_type_group_value: Any,
        object_type_value: Any,
        emission_tokens_value: Any,
        emission_lines_value: Any,
    ) -> bool:
        if _is_bright_nebula_group(object_type_group_value):
            return True
        if _is_hii_object_type(object_type_value):
            return True
        if _target_emission_band_set(emission_tokens_value):
            return True
        return _target_has_emissions_data(emission_lines_value)

    def _coerce_bool_flag(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        normalized = str(value or "").strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
        return bool(value)

    def _active_filter_emission_band_set(filter_item: Any) -> set[str]:
        if not isinstance(filter_item, dict):
            return set()

        explicit_band_set = _parse_emission_band_set_local(filter_item.get("emission_bands", []))
        if explicit_band_set:
            return explicit_band_set

        for emission_field in ("emission_lines", "Emission_lines"):
            parsed = _parse_emission_band_set_local(filter_item.get(emission_field))
            if parsed:
                return parsed

        inferred_bands: set[str] = set()
        if _coerce_bool_flag(filter_item.get("has_HA")):
            inferred_bands.add("HA")
        if _coerce_bool_flag(filter_item.get("has_OIII")):
            inferred_bands.add("OIII")
        if _coerce_bool_flag(filter_item.get("has_SII")):
            inferred_bands.add("SII")
        return inferred_bands

    summary_df["thumbnail_url"] = summary_df.apply(
        lambda row: _resolve_summary_thumbnail_url_cached(
            primary_id=str(row.get("primary_id") or "").strip(),
            common_name=str(row.get("common_name") or "").strip(),
            image_url=str(row.get("image_url") or "").strip(),
            hero_image_url=str(row.get("hero_image_url") or "").strip(),
            ra_deg=_thumbnail_numeric(row.get("ra_deg")),
            dec_deg=_thumbnail_numeric(row.get("dec_deg")),
            ang_size_maj_arcmin=_thumbnail_numeric(row.get("ang_size_maj_arcmin")),
            ang_size_min_arcmin=_thumbnail_numeric(row.get("ang_size_min_arcmin")),
        ),
        axis=1,
    )
    summary_df["apparent_size"] = summary_df.apply(
        lambda row: format_apparent_size_display(
            row.get("ang_size_maj_arcmin"),
            row.get("ang_size_min_arcmin"),
        ),
        axis=1,
    )

    active_equipment: dict[str, Any] = {}
    selected_telescope: dict[str, Any] | None = None
    telescope_fov_area: float | None = None
    try:
        equipment_context = build_owned_equipment_context(prefs)
        active_equipment = sync_active_equipment_settings(prefs, equipment_context)
        active_telescope = active_equipment.get("active_telescope")
        if not isinstance(active_telescope, dict):
            telescope_lookup = dict(active_equipment.get("telescope_lookup", {}))
            active_telescope_id = str(active_equipment.get("active_telescope_id", "")).strip()
            active_telescope = telescope_lookup.get(active_telescope_id) if active_telescope_id else None
        if isinstance(active_telescope, dict):
            telescope_fov_maj = _safe_positive_float(active_telescope.get("fov_maj_deg"))
            telescope_fov_min = _safe_positive_float(active_telescope.get("fov_min_deg"))
            if telescope_fov_maj is not None and telescope_fov_min is not None:
                selected_telescope = active_telescope
                telescope_fov_area = float(telescope_fov_maj * telescope_fov_min)
    except Exception:
        active_equipment = {}
        selected_telescope = None
        telescope_fov_area = None

    include_telescope_details = bool(st.session_state.get("recommended_targets_include_telescope_details", False))
    magnitude_filter_mode = str(
        st.session_state.get("recommended_targets_magnitude_filter_mode", "None")
    ).strip().title()
    if magnitude_filter_mode not in {"None", "Maximum", "Range"}:
        magnitude_filter_mode = "None"
    magnitude_filter_active = bool(
        include_telescope_details
        and selected_telescope is not None
        and magnitude_filter_mode in {"Maximum", "Range"}
    )
    magnitude_selected_max = _safe_finite_float(
        st.session_state.get("recommended_targets_max_magnitude_value")
    )
    if magnitude_selected_max is None:
        magnitude_selected_max = float(resolve_telescope_max_magnitude(selected_telescope, fallback=10.5))
    magnitude_filter_min_mag: float | None = None
    if magnitude_filter_mode == "Range":
        raw_range = st.session_state.get("recommended_targets_magnitude_filter_range_mag")
        if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
            low_value = _safe_finite_float(raw_range[0])
            high_value = _safe_finite_float(raw_range[1])
            if low_value is not None and high_value is not None:
                magnitude_filter_min_mag = float(min(low_value, high_value))

    active_filter = active_equipment.get("active_filter")
    if not isinstance(active_filter, dict):
        filter_lookup = dict(active_equipment.get("filter_lookup", {}))
        active_filter_id = str(active_equipment.get("active_filter_id", "__none__")).strip() or "__none__"
        active_filter = filter_lookup.get(active_filter_id) if active_filter_id != "__none__" else None
    narrowband_filter_active = bool(_active_filter_emission_band_set(active_filter))

    magnitude_source_column = (
        "magnitude_numeric"
        if "magnitude_numeric" in summary_df.columns
        else ("magnitude" if "magnitude" in summary_df.columns else "")
    )
    if magnitude_source_column:
        summary_df["magnitude_display"] = summary_df[magnitude_source_column].apply(
            lambda value: (
                f"{float(_safe_finite_float(value)):.1f}"
                if _safe_finite_float(value) is not None
                else "--"
            )
        )
    else:
        summary_df["magnitude_display"] = "--"

    magnitude_policy_statuses: list[str] = []
    for row_idx in range(len(summary_df)):
        raw_magnitude = summary_df.iloc[row_idx].get(magnitude_source_column) if magnitude_source_column else None
        target_is_narrowband = _target_is_narrowband_target(
            object_type_group_value=summary_df.iloc[row_idx].get("object_type_group"),
            object_type_value=summary_df.iloc[row_idx].get("object_type"),
            emission_tokens_value=summary_df.iloc[row_idx].get("emission_band_tokens"),
            emission_lines_value=summary_df.iloc[row_idx].get("emission_lines"),
        )
        if magnitude_filter_active:
            _, magnitude_status, _ = classify_target_magnitude(
                raw_magnitude,
                selected_max=float(magnitude_selected_max),
                narrowband_filter_active=bool(narrowband_filter_active),
                target_is_narrowband=bool(target_is_narrowband),
            )
            if magnitude_filter_mode == "Range" and magnitude_filter_min_mag is not None:
                parsed_row_magnitude = _safe_finite_float(raw_magnitude)
                if parsed_row_magnitude is None or float(parsed_row_magnitude) < float(magnitude_filter_min_mag):
                    magnitude_status = STATUS_OUT_OF_RANGE
        else:
            magnitude_status = STATUS_IN_RANGE if _safe_finite_float(raw_magnitude) is not None else STATUS_UNKNOWN
        magnitude_policy_statuses.append(str(magnitude_status))
    summary_df["magnitude_policy_status"] = magnitude_policy_statuses

    summary_df["framing_percent"] = np.nan
    show_framing_column = (
        selected_telescope is not None
        and telescope_fov_area is not None
        and telescope_fov_area > 0.0
    )
    if show_framing_column:
        target_maj_deg = pd.to_numeric(summary_df["ang_size_maj_arcmin"], errors="coerce") / 60.0
        target_min_deg = pd.to_numeric(summary_df["ang_size_min_arcmin"], errors="coerce") / 60.0
        target_maj_deg = target_maj_deg.where(target_maj_deg > 0.0)
        target_min_deg = target_min_deg.where(target_min_deg > 0.0)
        target_maj_deg = target_maj_deg.fillna(target_min_deg)
        target_min_deg = target_min_deg.fillna(target_maj_deg)
        target_area_deg2 = target_maj_deg * target_min_deg
        summary_df["framing_percent"] = (target_area_deg2 / float(telescope_fov_area)) * 100.0

    summary_df["line_swatch"] = "■"
    summary_df["visible_remaining_display"] = "--"
    if show_remaining:
        for row_index, row in summary_df.iterrows():
            first_visible = row.get("first_visible")
            total_duration = str(row.get("visible_total") or "").strip()
            remaining = str(row.get("visible_remaining") or "").strip()
            if (
                (not total_duration or total_duration == "--")
                or (not remaining or remaining == "--")
                or pd.isna(first_visible)
            ):
                continue
            try:
                first_visible_ts = pd.Timestamp(first_visible)
                if now_local is None:
                    now_ts = (
                        pd.Timestamp.now(tz=first_visible_ts.tzinfo)
                        if first_visible_ts.tzinfo is not None
                        else pd.Timestamp.now()
                    )
                else:
                    now_ts = pd.Timestamp(now_local)
                    if first_visible_ts.tzinfo is not None and now_ts.tzinfo is None:
                        now_ts = now_ts.tz_localize(first_visible_ts.tzinfo)
                    elif first_visible_ts.tzinfo is None and now_ts.tzinfo is not None:
                        now_ts = now_ts.tz_localize(None)
                    elif first_visible_ts.tzinfo is not None and now_ts.tzinfo is not None:
                        now_ts = now_ts.tz_convert(first_visible_ts.tzinfo)

                if now_ts > first_visible_ts:
                    summary_df.at[row_index, "visible_remaining_display"] = remaining
                else:
                    summary_df.at[row_index, "visible_remaining_display"] = total_duration
            except Exception:
                continue
    normalized_preview_list_id = str(preview_list_id or "").strip()
    schedules_state = st.session_state.get(TARGET_TIPS_SCHEDULES_STATE_KEY)
    session_schedule_scope_id = str(st.session_state.get(TARGET_TIPS_SCHEDULE_ACTIVE_LIST_STATE_KEY, "")).strip()
    if not isinstance(schedules_state, dict) or session_schedule_scope_id != normalized_preview_list_id:
        schedules_state = {}
        raw_all_schedules = prefs.get("list_target_schedules")
        raw_scoped_schedules = (
            raw_all_schedules.get(normalized_preview_list_id, {})
            if isinstance(raw_all_schedules, dict)
            else {}
        )
        if isinstance(raw_scoped_schedules, dict):
            for raw_target_id, raw_schedule_payload in raw_scoped_schedules.items():
                target_id = str(raw_target_id).strip()
                if not target_id or not isinstance(raw_schedule_payload, dict):
                    continue
                start_clock = _parse_schedule_clock(raw_schedule_payload.get("start_time"))
                end_clock = _parse_schedule_clock(raw_schedule_payload.get("end_time"))
                if start_clock is None or end_clock is None:
                    continue
                schedules_state[target_id] = {
                    "start_time": f"{int(start_clock.hour):02d}:{int(start_clock.minute):02d}",
                    "end_time": f"{int(end_clock.hour):02d}:{int(end_clock.minute):02d}",
                }
        st.session_state[TARGET_TIPS_SCHEDULES_STATE_KEY] = schedules_state
        st.session_state[TARGET_TIPS_SCHEDULE_ACTIVE_LIST_STATE_KEY] = normalized_preview_list_id
    summary_df["scheduled_start_display"] = "--"
    summary_df["scheduled_end_display"] = "--"
    summary_df["scheduled_duration_display"] = "--"
    summary_df["scheduled_display"] = "--"
    summary_df["scheduled_start_raw_minutes"] = np.nan
    for row_index, row in summary_df.iterrows():
        primary_id = str(row.get("primary_id", "")).strip()
        if not primary_id:
            continue
        schedule_payload = schedules_state.get(primary_id)
        if not isinstance(schedule_payload, dict):
            continue
        start_clock = _parse_schedule_clock(schedule_payload.get("start_time"))
        end_clock = _parse_schedule_clock(schedule_payload.get("end_time"))
        if start_clock is None or end_clock is None:
            continue
        start_display = format_display_time(
            pd.Timestamp.combine(pd.Timestamp.today(), start_clock),
            use_12_hour=use_12_hour,
        )
        end_display = format_display_time(
            pd.Timestamp.combine(pd.Timestamp.today(), end_clock),
            use_12_hour=use_12_hour,
        )
        start_minutes = int(start_clock.hour * 60 + start_clock.minute)
        end_minutes = int(end_clock.hour * 60 + end_clock.minute)
        if end_minutes <= start_minutes:
            end_minutes += 24 * 60
            end_display = f"{end_display} (+1d)"
        duration_display = _format_schedule_duration_hhmm(end_minutes - start_minutes)
        summary_df.at[row_index, "scheduled_start_display"] = start_display
        summary_df.at[row_index, "scheduled_end_display"] = end_display
        summary_df.at[row_index, "scheduled_duration_display"] = duration_display
        summary_df.at[row_index, "scheduled_display"] = f"{start_display}-{end_display} ({duration_display})"
        summary_df.at[row_index, "scheduled_start_raw_minutes"] = float(start_minutes)

    summary_df["remaining_sort_minutes"] = summary_df.apply(_remaining_minutes_for_sort, axis=1)

    if schedule_mode_active:
        # Sort in "night order" (evening -> overnight), not raw clock order.
        # Example: 11:59 PM should come before 12:01 AM.
        night_pivot_minutes = 12 * 60
        first_visible_times = pd.to_datetime(summary_df.get("first_visible"), errors="coerce")
        if isinstance(first_visible_times, pd.Series):
            valid_first_visible = first_visible_times.dropna()
            if not valid_first_visible.empty:
                earliest_first_visible = pd.Timestamp(valid_first_visible.min())
                night_pivot_minutes = int(earliest_first_visible.hour * 60 + earliest_first_visible.minute)

        def _night_order_minutes(raw_minutes: Any) -> float:
            try:
                value = float(raw_minutes)
            except (TypeError, ValueError):
                return float("nan")
            if not np.isfinite(value):
                return float("nan")
            return value + 1440.0 if value < float(night_pivot_minutes) else value

        summary_df["scheduled_start_sort_minutes"] = summary_df["scheduled_start_raw_minutes"].apply(_night_order_minutes)
        summary_df["__sort_has_schedule"] = np.where(summary_df["scheduled_start_sort_minutes"].notna(), 0, 1)
        summary_df["__sort_start_minutes"] = summary_df["scheduled_start_sort_minutes"].fillna(np.inf)
        summary_df["__sort_remaining_minutes"] = summary_df["remaining_sort_minutes"].fillna(np.inf)
        summary_df["__sort_original_order"] = np.arange(len(summary_df), dtype=int)
        summary_df = summary_df.sort_values(
            by=[
                "__sort_has_schedule",
                "__sort_start_minutes",
                "__sort_remaining_minutes",
                "__sort_original_order",
            ],
            ascending=[True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        first_visible_sort = pd.to_datetime(summary_df.get("first_visible"), errors="coerce")
        peak_sort = pd.to_datetime(summary_df.get("culmination"), errors="coerce")
        if not isinstance(first_visible_sort, pd.Series):
            first_visible_sort = pd.Series(
                [pd.NaT] * len(summary_df),
                index=summary_df.index,
                dtype="datetime64[ns]",
            )
        if not isinstance(peak_sort, pd.Series):
            peak_sort = pd.Series(
                [pd.NaT] * len(summary_df),
                index=summary_df.index,
                dtype="datetime64[ns]",
            )
        summary_df["__sort_first_visible"] = first_visible_sort
        summary_df["__sort_peak"] = peak_sort
        summary_df["__sort_original_order"] = np.arange(len(summary_df), dtype=int)
        summary_df = summary_df.sort_values(
            by=[
                "__sort_first_visible",
                "__sort_peak",
                "__sort_original_order",
            ],
            ascending=[True, True, True],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
    display_columns = [
        "line_swatch",
        "thumbnail_url",
        "target",
    ]
    if not schedule_mode_active:
        display_columns.append("object_type_group")
    if schedule_mode_active:
        display_columns.extend(
            [
                "scheduled_display",
            ]
        )
    else:
        display_columns.extend(
            [
                "first_visible",
                "culmination",
                "last_visible",
                "visible_total",
            ]
        )
    display_columns.append("magnitude_display")
    if show_framing_column:
        display_columns.append("framing_percent")
    else:
        display_columns.append("apparent_size")
    if show_remaining and not schedule_mode_active:
        display_columns.append("visible_remaining_display")
    display_columns.extend(["culmination_alt", "culmination_dir"])

    display = summary_df[display_columns].rename(
        columns={
            "thumbnail_url": "Thumbnail",
            "line_swatch": "Line",
            "target": "Target",
            "object_type_group": "Type",
            "magnitude_display": "Magnitude",
            "apparent_size": "Apparent size",
            "framing_percent": "Framing",
            "first_visible": "First Visible",
            "culmination": "Peak",
            "last_visible": "Last Visible",
            "visible_total": "Duration",
            "scheduled_display": "Scheduled",
            "visible_remaining_display": "Remaining",
            "culmination_alt": "Max Alt",
            "culmination_dir": "Direction",
        }
    )
    is_dark_theme = is_dark_ui_theme()
    theme_palette = resolve_theme_palette()
    dark_table_styles = theme_palette.get("dataframe_styler", {})
    dark_td_bg = str(dark_table_styles.get("td_bg", "#0F172A"))
    dark_td_text = str(dark_table_styles.get("td_text", "#E5E7EB"))

    def _style_summary_row(row: pd.Series) -> list[str]:
        base_cell_style = f"background-color: {dark_td_bg}; color: {dark_td_text};" if is_dark_theme else ""
        styles = [base_cell_style for _ in row]
        color = str(summary_df.loc[row.name, "line_color"]).strip()
        row_primary_id = str(summary_df.loc[row.name, "primary_id"]).strip()
        selected_detail_id = str(st.session_state.get("selected_id") or "").strip()
        highlighted_summary_id = str(st.session_state.get("sky_summary_highlight_primary_id", "")).strip()
        row_is_selected = (
            row_primary_id
            and (
                (selected_detail_id and row_primary_id == selected_detail_id)
                or (highlighted_summary_id and row_primary_id == highlighted_summary_id)
            )
        )
        if row_is_selected:
            selected_bg = _muted_rgba_from_hex(color, alpha=0.16)
            for idx in range(len(styles)):
                base_style = styles[idx]
                if base_style and not base_style.endswith(";"):
                    base_style = f"{base_style};"
                styles[idx] = f"{base_style} background-color: {selected_bg};"
        if color:
            line_idx = row.index.get_loc("Line")
            base_style = styles[line_idx]
            if base_style and not base_style.endswith(";"):
                base_style = f"{base_style};"
            styles[line_idx] = f"{base_style} color: {color}; font-weight: 700;"
        if magnitude_filter_active and "Magnitude" in row.index:
            magnitude_idx = row.index.get_loc("Magnitude")
            magnitude_base_style = styles[magnitude_idx]
            if magnitude_base_style and not magnitude_base_style.endswith(";"):
                magnitude_base_style = f"{magnitude_base_style};"
            magnitude_status = str(summary_df.loc[row.name, "magnitude_policy_status"]).strip().lower()
            if magnitude_status == STATUS_BROADBAND_BORDERLINE:
                styles[magnitude_idx] = f"{magnitude_base_style} color: #1d4ed8; font-weight: 700;"
            elif magnitude_status == STATUS_NARROWBAND_BOOSTED:
                styles[magnitude_idx] = f"{magnitude_base_style} color: #991b1b; font-weight: 700;"
        return styles

    styled = apply_dataframe_styler_theme(display.style.apply(_style_summary_row, axis=1))

    column_config: dict[str, Any] = {
        "Line": st.column_config.TextColumn(width="small"),
        "Thumbnail": (
            st.column_config.ImageColumn(label="", width=100)
            if hasattr(st.column_config, "ImageColumn")
            else st.column_config.TextColumn(label="", width=100)
        ),
        "Target": st.column_config.TextColumn(width="large"),
        "Type": st.column_config.TextColumn(width="small"),
        "Magnitude": st.column_config.TextColumn(width="small"),
        "First Visible": st.column_config.DatetimeColumn(
            width="small",
            format=("h:mm a" if use_12_hour else "HH:mm"),
        ),
        "Peak": st.column_config.DatetimeColumn(
            width="small",
            format=("h:mm a" if use_12_hour else "HH:mm"),
        ),
        "Max Alt": st.column_config.TextColumn(width="small"),
        "Last Visible": st.column_config.DatetimeColumn(
            width="small",
            format=("h:mm a" if use_12_hour else "HH:mm"),
        ),
        "Duration": st.column_config.TextColumn(width="small"),
        "Direction": st.column_config.TextColumn(width="small"),
    }
    if "Apparent size" in display.columns:
        column_config["Apparent size"] = st.column_config.TextColumn(width="small")
    if "Framing" in display.columns:
        column_config["Framing"] = st.column_config.NumberColumn(width="small", format="%.0f%%")
    if "Scheduled" in display.columns:
        column_config["Scheduled"] = st.column_config.TextColumn(width="medium")
    if show_remaining and not schedule_mode_active:
        column_config["Remaining"] = st.column_config.TextColumn(width="small")

    styled = styled.set_properties(
        subset=["Thumbnail"],
        **{
            "text-align": "left !important",
            "justify-content": "flex-start !important",
            "padding-left": "0px !important",
        },
    )

    selected_rows: list[int] = []
    selected_columns: list[Any] = []
    selected_cells: list[Any] = []
    mui_table = globals().get("st_mui_table")
    if callable(mui_table):
        frozen_line_col_width_px = 44
        frozen_thumbnail_col_width_px = 108
        frozen_target_col_width_px = 320
        frozen_thumbnail_left_px = frozen_line_col_width_px
        frozen_target_left_px = frozen_line_col_width_px + frozen_thumbnail_col_width_px
        frozen_header_bg = dark_td_bg if is_dark_theme else "#FFFFFF"
        summary_table_custom_css = f"""
.MuiTableCell-root {{ padding: 0 !important; }}
.MuiTablePagination-root {{ display: none !important; }}

/* Freeze leading columns: Line, Thumbnail, Target */
.MuiTableHead-root .MuiTableCell-root:nth-of-type(1),
.MuiTableBody-root .MuiTableCell-root:nth-of-type(1) {{
  position: sticky;
  left: 0px;
}}
.MuiTableHead-root .MuiTableCell-root:nth-of-type(2),
.MuiTableBody-root .MuiTableCell-root:nth-of-type(2) {{
  position: sticky;
  left: {frozen_thumbnail_left_px}px;
}}
.MuiTableHead-root .MuiTableCell-root:nth-of-type(3),
.MuiTableBody-root .MuiTableCell-root:nth-of-type(3) {{
  position: sticky;
  left: {frozen_target_left_px}px;
}}

.MuiTableHead-root .MuiTableCell-root:nth-of-type(1),
.MuiTableHead-root .MuiTableCell-root:nth-of-type(2),
.MuiTableHead-root .MuiTableCell-root:nth-of-type(3) {{
  z-index: 12 !important;
  background-color: {frozen_header_bg} !important;
}}
.MuiTableBody-root .MuiTableCell-root:nth-of-type(1),
.MuiTableBody-root .MuiTableCell-root:nth-of-type(2),
.MuiTableBody-root .MuiTableCell-root:nth-of-type(3) {{
  z-index: 10 !important;
  background-color: inherit !important;
}}

/* Keep frozen column widths stable so sticky offsets remain aligned. */
.MuiTableHead-root .MuiTableCell-root:nth-of-type(1),
.MuiTableBody-root .MuiTableCell-root:nth-of-type(1) {{
  min-width: {frozen_line_col_width_px}px !important;
  width: {frozen_line_col_width_px}px !important;
  max-width: {frozen_line_col_width_px}px !important;
}}
.MuiTableHead-root .MuiTableCell-root:nth-of-type(2),
.MuiTableBody-root .MuiTableCell-root:nth-of-type(2) {{
  min-width: {frozen_thumbnail_col_width_px}px !important;
  width: {frozen_thumbnail_col_width_px}px !important;
  max-width: {frozen_thumbnail_col_width_px}px !important;
}}
.MuiTableHead-root .MuiTableCell-root:nth-of-type(3),
.MuiTableBody-root .MuiTableCell-root:nth-of-type(3) {{
  min-width: {frozen_target_col_width_px}px !important;
}}
"""
        selected_detail_id = str(st.session_state.get("selected_id") or "").strip()
        highlighted_summary_id = str(st.session_state.get("sky_summary_highlight_primary_id", "")).strip()

        def _format_table_time(raw_value: Any) -> str:
            if raw_value is None or pd.isna(raw_value):
                return "--"
            try:
                timestamp = pd.Timestamp(raw_value)
            except Exception:
                return str(raw_value).strip() or "--"
            return format_display_time(timestamp, use_12_hour=use_12_hour)

        def _mui_cell_html(text: str, style_parts: list[str], *, raw_html: str = "") -> str:
            content_html = raw_html if raw_html else (html.escape(text) if text else "&nbsp;")
            style = " ".join([part for part in style_parts if str(part).strip()])
            return f"<div style='{style}'>{content_html}</div>"

        mui_frame = display.copy()
        for row_idx, row in display.iterrows():
            row_color = str(summary_df.loc[row_idx, "line_color"]).strip()
            row_primary_id = str(summary_df.loc[row_idx, "primary_id"]).strip()
            row_is_selected = (
                row_primary_id
                and (
                    (selected_detail_id and row_primary_id == selected_detail_id)
                    or (highlighted_summary_id and row_primary_id == highlighted_summary_id)
                )
            )
            row_bg = _muted_rgba_from_hex(row_color, alpha=0.16) if row_is_selected else (dark_td_bg if is_dark_theme else "")
            row_fg = dark_td_text if is_dark_theme else ""
            for column_name in display.columns:
                raw_value = row.get(column_name)
                style_parts = [
                    "display: flex;",
                    "align-items: center;",
                    "justify-content: center;",
                    "width: 100%;",
                    "height: 56px;",
                    "box-sizing: border-box;",
                    "padding: 6px 8px;",
                    "margin: 0;",
                    "border-radius: 0;",
                    "white-space: nowrap;",
                    "text-align: center;",
                ]
                if row_bg:
                    style_parts.append(f"background-color: {row_bg};")
                if row_fg:
                    style_parts.append(f"color: {row_fg};")

                cell_text = ""
                cell_html = ""
                if column_name == "Line":
                    cell_text = "■"
                    style_parts.append("font-weight: 700;")
                    if row_color:
                        style_parts.append(f"color: {row_color};")
                elif column_name == "Thumbnail":
                    thumbnail_url = str(raw_value or "").strip()
                    style_parts.extend(["text-align: left;", "justify-content: flex-start;", "padding-left: 0px;"])
                    if thumbnail_url.lower().startswith(("https://", "http://")):
                        escaped_url = html.escape(thumbnail_url, quote=True)
                        cell_html = (
                            f"<img src=\"{escaped_url}\" "
                            "style=\"width:84px; height:56px; object-fit:cover; border-radius:6px; display:block;\" />"
                        )
                elif column_name in {"First Visible", "Peak", "Last Visible"}:
                    cell_text = _format_table_time(raw_value)
                elif column_name == "Framing":
                    try:
                        cell_text = f"{float(raw_value):.0f}%"
                    except (TypeError, ValueError):
                        cell_text = "--"
                elif column_name == "Magnitude":
                    cell_text = "" if raw_value is None or pd.isna(raw_value) else str(raw_value).strip()
                    if not cell_text:
                        cell_text = "--"
                    if magnitude_filter_active:
                        magnitude_status = str(summary_df.at[int(row_idx), "magnitude_policy_status"]).strip().lower()
                        if magnitude_status == STATUS_BROADBAND_BORDERLINE:
                            style_parts.append("color: #1d4ed8;")
                            style_parts.append("font-weight: 700;")
                        elif magnitude_status == STATUS_NARROWBAND_BOOSTED:
                            style_parts.append("color: #991b1b;")
                            style_parts.append("font-weight: 700;")
                else:
                    cell_text = "" if raw_value is None or pd.isna(raw_value) else str(raw_value).strip()
                    if not cell_text:
                        cell_text = "--"
                    if column_name in {
                        "Target",
                        "Type",
                        "Duration",
                        "Scheduled",
                        "Apparent size",
                        "Remaining",
                        "Direction",
                        "Max Alt",
                    }:
                        style_parts.append("text-align: left;")
                        style_parts.append("justify-content: flex-start;")

                mui_frame.at[row_idx, column_name] = _mui_cell_html(cell_text, style_parts, raw_html=cell_html)

        clicked_cell = mui_table(
            mui_frame,
            enablePagination=True,
            paginationSizes=[24],
            customCss=summary_table_custom_css,
            showHeaders=True,
            key="sky_summary_mui_table",
            stickyHeader=False,
            showIndex=False,
            enable_sorting=False,
            return_clicked_cell=True,
            paperStyle={
                "width": "100%",
                "overflow": "visible",
                "paddingBottom": "0px",
                "border": "1px solid rgba(148, 163, 184, 0.35)",
            },
        )
        if isinstance(clicked_cell, dict):
            raw_row = clicked_cell.get("row")
            try:
                parsed_row_index = int(raw_row)
            except (TypeError, ValueError):
                parsed_row_index = -1
            if 0 <= parsed_row_index < len(summary_df):
                selected_rows = [parsed_row_index]

            raw_column = (
                clicked_cell.get("column")
                if "column" in clicked_cell
                else clicked_cell.get("col", clicked_cell.get("field"))
            )
            parsed_column_index: int | None = None
            if isinstance(raw_column, str) and raw_column in mui_frame.columns:
                parsed_column_index = int(mui_frame.columns.get_loc(raw_column))
            else:
                try:
                    parsed_column_value = int(raw_column)
                except (TypeError, ValueError):
                    parsed_column_value = -1
                if 0 <= parsed_column_value < len(mui_frame.columns):
                    parsed_column_index = parsed_column_value
            if parsed_column_index is not None:
                selected_columns = [parsed_column_index]
            if selected_rows and parsed_column_index is not None:
                selected_cells = [(selected_rows[0], parsed_column_index)]
    else:
        table_event = st.dataframe(
            styled,
            hide_index=True,
            use_container_width=True,
            row_height=70,
            on_select="rerun",
            selection_mode="single-cell",
            key="sky_summary_table",
            column_config=column_config,
        )

        if table_event is not None:
            try:
                selected_rows = list(table_event.selection.rows)
                selected_columns = list(table_event.selection.columns)
                selected_cells = list(table_event.selection.cells)
            except Exception:
                if isinstance(table_event, dict):
                    selection_payload = table_event.get("selection", {})
                    selected_rows = list(selection_payload.get("rows", []))
                    selected_columns = list(selection_payload.get("columns", []))
                    selected_cells = list(selection_payload.get("cells", []))

    selected_index: int | None = None
    selected_column_index: int | None = None
    if selected_cells:
        first_cell = selected_cells[0]
        raw_row: Any = None
        raw_column: Any = None
        if isinstance(first_cell, (list, tuple)) and len(first_cell) >= 2:
            raw_row = first_cell[0]
            raw_column = first_cell[1]
        elif isinstance(first_cell, dict):
            raw_row = first_cell.get("row")
            raw_column = first_cell.get("column")

        if raw_row is not None:
            try:
                parsed_row_index = int(raw_row)
                if 0 <= parsed_row_index < len(summary_df):
                    selected_index = parsed_row_index
            except (TypeError, ValueError):
                selected_index = None

        if raw_column is not None:
            try:
                selected_column_index = int(raw_column)
            except (TypeError, ValueError):
                raw_column_name = str(raw_column)
                if raw_column_name in display.columns:
                    selected_column_index = int(display.columns.get_loc(raw_column_name))

    if selected_rows:
        try:
            parsed_row_index = int(selected_rows[0])
            if 0 <= parsed_row_index < len(summary_df):
                selected_index = parsed_row_index
        except (TypeError, ValueError):
            selected_index = None

    if selected_column_index is None and selected_columns:
        raw_column = selected_columns[0]
        try:
            selected_column_index = int(raw_column)
        except (TypeError, ValueError):
            raw_column_name = str(raw_column)
            if raw_column_name in display.columns:
                selected_column_index = int(display.columns.get_loc(raw_column_name))

    selected_primary_id = ""
    if selected_index is not None:
        selected_primary_id = str(summary_df.iloc[selected_index].get("primary_id", ""))

    selection_token = ""
    if selected_index is not None:
        selection_token = f"{selected_index}:{selected_column_index if selected_column_index is not None else '*'}"
    last_selection_token = str(st.session_state.get("sky_summary_last_selection_token", ""))
    selection_changed = bool(selection_token) and selection_token != last_selection_token
    # Keep the last non-empty selection token so unrelated reruns (for example,
    # style toggles) do not turn a stale row/cell selection into a "new" click.
    if selection_token:
        st.session_state["sky_summary_last_selection_token"] = selection_token

    current_highlight_id = str(st.session_state.get("sky_summary_highlight_primary_id", ""))
    # Apply row focus whenever a row is selected; this keeps Tips/plots aligned even
    # when Streamlit emits selection payloads with unstable token shapes.
    if selected_primary_id and selected_primary_id != current_highlight_id:
        st.session_state["sky_summary_highlight_primary_id"] = selected_primary_id

    st.caption("Recommended targets choose the detail target. Use this table to highlight rows.")
