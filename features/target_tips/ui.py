from __future__ import annotations

from datetime import datetime, time as dt_time
import html
import re
from typing import Any

import pandas as pd
import streamlit as st

from app_preferences import save_preferences
from features.lists.list_subsystem import AUTO_RECENT_LIST_ID

from .engine import TargetTip, TargetTipsContext, collect_target_tips

TARGET_TIPS_SCHEDULES_STATE_KEY = "target_tips_schedule_by_target_id"
TARGET_TIPS_SCHEDULE_ACTIVE_LIST_STATE_KEY = "target_tips_schedule_active_list_id"
TARGET_TIPS_OPEN_DETAILS_REQUEST_KEY = "target_tips_open_details_request_target_id"


def _tip_text_html(text: str) -> str:
    escaped = html.escape(str(text or ""))
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def _coerce_time_value(value: Any) -> dt_time | None:
    if isinstance(value, dt_time):
        return value.replace(second=0, microsecond=0)
    if isinstance(value, datetime):
        return value.time().replace(second=0, microsecond=0)
    if value is None:
        return None
    if isinstance(value, str):
        raw = str(value).strip()
        if not raw:
            return None
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                parsed = datetime.strptime(raw, fmt).time()
                return parsed.replace(second=0, microsecond=0)
            except ValueError:
                continue
    return None


def _format_display_clock(value: dt_time, *, use_12_hour: bool) -> str:
    if use_12_hour:
        return value.strftime("%I:%M %p").lstrip("0")
    return value.strftime("%H:%M")


def _format_duration_label(total_minutes: int) -> str:
    bounded_minutes = max(0, int(total_minutes))
    hours, minutes = divmod(bounded_minutes, 60)
    if hours > 0 and minutes > 0:
        return f"{hours} {'hour' if hours == 1 else 'hours'}, {minutes} {'min' if minutes == 1 else 'mins'}"
    if hours > 0:
        return f"{hours} {'hour' if hours == 1 else 'hours'}"
    return f"{minutes} {'min' if minutes == 1 else 'mins'}"


def split_target_tip_lines(tips: list[TargetTip]) -> tuple[list[str], list[str], list[str], list[str]]:
    opportunity_lines: list[str] = []
    warning_lines: list[str] = []
    bullet_lines: list[str] = []
    muted_lines: list[str] = []

    for tip in tips:
        text = str(tip.get("text", "")).strip()
        if not text:
            continue

        display = str(tip.get("display", "bullet")).strip().lower() or "bullet"
        if display == "muted":
            muted_lines.append(text)
            continue
        if display == "opportunity":
            opportunity_lines.append(text)
            continue
        if display == "strong":
            bullet_lines.append(f"- **{text}**")
            continue
        if display == "warning":
            warning_lines.append(text)
            continue
        bullet_lines.append(f"- {text}")

    return opportunity_lines, warning_lines, bullet_lines, muted_lines


def render_target_tips_panel(
    selected_id: str,
    selected_label: str,
    selected_target_data: pd.Series | dict[str, Any] | None,
    selected_track: pd.DataFrame | None,
    summary_rows: list[dict[str, Any]],
    nightly_weather_alert_emojis: list[str],
    hourly_weather_rows: list[dict[str, Any]],
    prefs: dict[str, Any],
    *,
    temperature_unit: str,
    use_12_hour: bool,
    local_now: datetime,
    window_start: datetime,
    window_end: datetime,
) -> None:
    selected_primary_id = str(selected_id or "").strip()
    selected_display_label = str(selected_label or "").strip()
    st.markdown("##### Target Tips")

    def _render_schedule_popover(row_payload: dict[str, Any] | None = None) -> None:
        if not selected_primary_id or not hasattr(st, "popover"):
            return

        def _active_schedule_list_id() -> str:
            list_id = str(st.session_state.get(TARGET_TIPS_SCHEDULE_ACTIVE_LIST_STATE_KEY, "")).strip()
            if list_id:
                return list_id
            fallback = str(prefs.get("active_preview_list_id", AUTO_RECENT_LIST_ID)).strip()
            return fallback or AUTO_RECENT_LIST_ID

        def _normalize_schedule_payload(raw_payload: Any) -> dict[str, str] | None:
            if not isinstance(raw_payload, dict):
                return None
            start_time = _coerce_time_value(raw_payload.get("start_time"))
            end_time = _coerce_time_value(raw_payload.get("end_time"))
            if start_time is None or end_time is None:
                return None
            return {
                "start_time": start_time.strftime("%H:%M"),
                "end_time": end_time.strftime("%H:%M"),
            }

        def _load_scoped_schedules_from_prefs(list_id: str) -> dict[str, dict[str, str]]:
            raw_all_schedules = prefs.get("list_target_schedules")
            if not isinstance(raw_all_schedules, dict):
                return {}
            raw_list_schedules = raw_all_schedules.get(list_id)
            if not isinstance(raw_list_schedules, dict):
                return {}
            normalized: dict[str, dict[str, str]] = {}
            for raw_target_id, raw_payload in raw_list_schedules.items():
                target_id = str(raw_target_id).strip()
                if not target_id:
                    continue
                normalized_payload = _normalize_schedule_payload(raw_payload)
                if normalized_payload is None:
                    continue
                normalized[target_id] = normalized_payload
            return normalized

        def _persist_scoped_schedules(list_id: str, schedules_payload: dict[str, dict[str, str]]) -> None:
            normalized_payload: dict[str, dict[str, str]] = {}
            for raw_target_id, raw_payload in schedules_payload.items():
                target_id = str(raw_target_id).strip()
                if not target_id:
                    continue
                normalized_schedule = _normalize_schedule_payload(raw_payload)
                if normalized_schedule is None:
                    continue
                normalized_payload[target_id] = normalized_schedule

            raw_all_schedules = prefs.get("list_target_schedules")
            all_schedules = dict(raw_all_schedules) if isinstance(raw_all_schedules, dict) else {}
            if normalized_payload:
                all_schedules[list_id] = normalized_payload
            else:
                all_schedules.pop(list_id, None)
            prefs["list_target_schedules"] = all_schedules
            st.session_state[TARGET_TIPS_SCHEDULES_STATE_KEY] = normalized_payload
            st.session_state[TARGET_TIPS_SCHEDULE_ACTIVE_LIST_STATE_KEY] = list_id
            st.session_state["prefs"] = prefs
            save_preferences(prefs)

        active_list_id = _active_schedule_list_id()
        schedules_state = st.session_state.get(TARGET_TIPS_SCHEDULES_STATE_KEY)
        session_scope_id = str(st.session_state.get(TARGET_TIPS_SCHEDULE_ACTIVE_LIST_STATE_KEY, "")).strip()
        if not isinstance(schedules_state, dict) or session_scope_id != active_list_id:
            schedules_state = _load_scoped_schedules_from_prefs(active_list_id)
            st.session_state[TARGET_TIPS_SCHEDULES_STATE_KEY] = schedules_state
            st.session_state[TARGET_TIPS_SCHEDULE_ACTIVE_LIST_STATE_KEY] = active_list_id
        existing_schedule = schedules_state.get(selected_primary_id, {})
        if not isinstance(existing_schedule, dict):
            existing_schedule = {}

        window_start_ts = pd.Timestamp(window_start)
        window_end_ts = pd.Timestamp(window_end)
        visible_start_ts = pd.NaT
        visible_end_ts = pd.NaT
        if isinstance(row_payload, dict):
            visible_start_ts = pd.to_datetime(row_payload.get("first_visible"), errors="coerce")
            visible_end_ts = pd.to_datetime(row_payload.get("last_visible"), errors="coerce")
        if pd.isna(visible_start_ts):
            visible_start_ts = window_start_ts
        if pd.isna(visible_end_ts):
            visible_end_ts = window_end_ts

        # Keep bounds on clean 10-minute boundaries inside the visible window.
        # Example: first-visible 5:48 -> earliest slider start 5:50.
        visible_start_ts = pd.Timestamp(visible_start_ts).ceil("10min")
        visible_end_ts = pd.Timestamp(visible_end_ts).floor("10min")

        if visible_end_ts <= visible_start_ts:
            visible_start_ts = pd.Timestamp(window_start_ts).ceil("10min")
            visible_end_ts = pd.Timestamp(window_end_ts).floor("10min")
        if visible_end_ts <= visible_start_ts:
            st.caption("Schedule unavailable for the current target visibility window.")
            return

        slot_step_minutes = 10
        total_visible_minutes = int((visible_end_ts - visible_start_ts).total_seconds() // 60)
        schedule_slot_offsets = list(range(0, total_visible_minutes + 1, slot_step_minutes))
        if not schedule_slot_offsets:
            schedule_slot_offsets = [0]
        if schedule_slot_offsets[-1] != total_visible_minutes:
            schedule_slot_offsets.append(total_visible_minutes)
        if len(schedule_slot_offsets) < 2:
            st.caption("Schedule unavailable for the current target visibility window.")
            return

        def _offset_to_timestamp(offset_minutes: int) -> pd.Timestamp:
            return visible_start_ts + pd.Timedelta(minutes=int(offset_minutes))

        def _offset_to_clock(offset_minutes: int) -> dt_time:
            return _offset_to_timestamp(offset_minutes).time().replace(second=0, microsecond=0)

        def _offset_label(offset_minutes: int) -> str:
            slot_timestamp = _offset_to_timestamp(offset_minutes)
            label = _format_display_clock(slot_timestamp.time().replace(second=0, microsecond=0), use_12_hour=use_12_hour)
            if slot_timestamp.date() > visible_start_ts.date():
                return f"{label} (+1d)"
            return label

        def _clock_to_offset(value: dt_time) -> int:
            candidate = visible_start_ts.normalize() + pd.Timedelta(hours=int(value.hour), minutes=int(value.minute))
            if candidate < visible_start_ts:
                candidate += pd.Timedelta(days=1)
            if candidate > visible_end_ts:
                candidate = visible_end_ts
            raw_offset = int(round((candidate - visible_start_ts).total_seconds() / 60.0))
            return min(schedule_slot_offsets, key=lambda option: abs(option - raw_offset))

        default_start_time = _coerce_time_value(existing_schedule.get("start_time")) or _offset_to_clock(0)
        default_end_time = _coerce_time_value(existing_schedule.get("end_time")) or _offset_to_clock(total_visible_minutes)
        default_start_offset = _clock_to_offset(default_start_time)
        default_end_offset = _clock_to_offset(default_end_time)
        if default_end_offset <= default_start_offset:
            default_end_offset = min(total_visible_minutes, default_start_offset + slot_step_minutes)
        default_range = (default_start_offset, default_end_offset)
        if default_range[0] == default_range[1]:
            default_range = (0, schedule_slot_offsets[-1])
        has_saved_schedule = selected_primary_id in schedules_state

        if has_saved_schedule:
            st.caption(
                "Current schedule: "
                f"{_format_display_clock(default_start_time, use_12_hour=use_12_hour)} -> "
                f"{_format_display_clock(default_end_time, use_12_hour=use_12_hour)}"
            )
        else:
            st.caption("No schedule set.")
        with st.popover(
            "Schedule",
            help="Schedule this target for the current night window.",
            icon=":material/event:",
            width="stretch",
        ):
            selected_range_key = f"target_tips_schedule_range_{selected_primary_id}"

            def _persist_schedule(start_offset_value: int, end_offset_value: int) -> bool:
                start_time_value = _offset_to_clock(start_offset_value)
                end_time_value = _offset_to_clock(end_offset_value)
                next_payload = {
                    "start_time": start_time_value.strftime("%H:%M"),
                    "end_time": end_time_value.strftime("%H:%M"),
                }
                current_schedules = st.session_state.get(TARGET_TIPS_SCHEDULES_STATE_KEY)
                if not isinstance(current_schedules, dict):
                    current_schedules = {}
                existing_payload = _normalize_schedule_payload(current_schedules.get(selected_primary_id))
                if existing_payload == next_payload:
                    return False
                next_schedules = dict(current_schedules)
                next_schedules[selected_primary_id] = next_payload
                _persist_scoped_schedules(active_list_id, next_schedules)
                return True

            def _apply_live_schedule_from_slider() -> None:
                if not has_saved_schedule:
                    return
                slider_value = st.session_state.get(selected_range_key, default_range)
                if isinstance(slider_value, tuple) and len(slider_value) == 2:
                    live_start_offset = int(slider_value[0])
                    live_end_offset = int(slider_value[1])
                else:
                    live_start_offset = int(default_range[0])
                    live_end_offset = int(default_range[1])
                if live_end_offset <= live_start_offset:
                    live_end_offset = min(total_visible_minutes, live_start_offset + slot_step_minutes)
                _persist_schedule(live_start_offset, live_end_offset)

            selected_range = st.select_slider(
                "Start / End",
                options=schedule_slot_offsets,
                value=default_range,
                format_func=_offset_label,
                key=selected_range_key,
                on_change=_apply_live_schedule_from_slider,
            )
            if isinstance(selected_range, tuple) and len(selected_range) == 2:
                start_offset = int(selected_range[0])
                end_offset = int(selected_range[1])
            else:
                start_offset = int(default_range[0])
                end_offset = int(default_range[1])
            if end_offset <= start_offset:
                end_offset = min(total_visible_minutes, start_offset + slot_step_minutes)

            selected_duration_minutes = max(0, int(end_offset - start_offset))
            duration_label = _format_duration_label(selected_duration_minutes)
            st.caption(
                f"Selected duration: {duration_label} "
                f"({_offset_label(start_offset)} -> {_offset_label(end_offset)})."
            )

            timezone_label = str(pd.Timestamp(window_start).tzinfo or "").strip()
            if timezone_label:
                st.caption(f"Times are interpreted in local site time ({timezone_label}).")
            if not has_saved_schedule:
                if st.button("Apply", key=f"target_tips_schedule_apply_{selected_primary_id}", use_container_width=True):
                    if _persist_schedule(start_offset, end_offset):
                        st.rerun()
            if st.button("Delete", key=f"target_tips_schedule_delete_{selected_primary_id}", use_container_width=True):
                if selected_primary_id in schedules_state:
                    next_schedules = dict(schedules_state)
                    next_schedules.pop(selected_primary_id, None)
                    _persist_scoped_schedules(active_list_id, next_schedules)
                    st.rerun()

    def _render_view_target_details_button() -> None:
        if not selected_primary_id:
            return
        if st.button(
            "View Target Details",
            key=f"target_tips_view_details_{selected_primary_id}",
            use_container_width=True,
        ):
            st.session_state[TARGET_TIPS_OPEN_DETAILS_REQUEST_KEY] = selected_primary_id
            st.rerun()
    if not selected_primary_id:
        st.caption(selected_display_label or "No target selected")
        st.caption("No target tips available.")
        return

    if not summary_rows:
        st.caption("No target tips available.")
        _render_schedule_popover(None)
        _render_view_target_details_button()
        return

    selected_row = next(
        (row for row in summary_rows if str(row.get("primary_id", "")).strip() == selected_primary_id),
        summary_rows[0],
    )
    context = TargetTipsContext(
        selected_id=selected_primary_id,
        selected_label=selected_display_label,
        selected_target_data=selected_target_data,
        selected_track=selected_track,
        selected_row=selected_row,
        nightly_weather_alert_emojis=nightly_weather_alert_emojis,
        hourly_weather_rows=hourly_weather_rows,
        temperature_unit=temperature_unit,
        use_12_hour=use_12_hour,
        local_now=local_now,
        window_start=window_start,
        window_end=window_end,
    )
    tips = collect_target_tips(context)

    st.caption(selected_display_label or selected_primary_id)
    if not tips:
        st.caption("No target tips available.")
        _render_schedule_popover(selected_row)
        _render_view_target_details_button()
        return
    opportunity_lines, warning_lines, bullet_lines, muted_lines = split_target_tip_lines(tips)
    for opportunity_line in opportunity_lines:
        st.markdown(
            (
                "<div style=\"background:#1E3A8A; border:1px solid #1D4ED8; color:#FFFFFF; "
                "padding:0.35rem 0.55rem; border-radius:0.35rem; margin:0 0 0.35rem 0;\">"
                f"<strong>Opportunity:</strong> {_tip_text_html(opportunity_line)}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    for warning_line in warning_lines:
        st.markdown(
            (
                "<div style=\"background:#FEF9C3; border:1px solid #FDE68A; color:#713F12; "
                "padding:0.35rem 0.55rem; border-radius:0.35rem; margin:0 0 0.35rem 0;\">"
                f"<strong>Warning:</strong> {_tip_text_html(warning_line)}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    if bullet_lines:
        st.markdown("\n".join(bullet_lines))
    for muted_line in muted_lines:
        st.caption(muted_line)
    _render_schedule_popover(selected_row)
    _render_view_target_details_button()
