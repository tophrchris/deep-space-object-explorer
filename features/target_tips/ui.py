from __future__ import annotations

from datetime import datetime
import html
import re
from typing import Any

import pandas as pd
import streamlit as st

from .engine import TargetTip, TargetTipsContext, collect_target_tips


def _tip_text_html(text: str) -> str:
    escaped = html.escape(str(text or ""))
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


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
    *,
    temperature_unit: str,
    use_12_hour: bool,
    local_now: datetime,
    window_start: datetime,
    window_end: datetime,
) -> None:
    st.markdown("##### Target Tips")
    selected_primary_id = str(selected_id or "").strip()
    selected_display_label = str(selected_label or "").strip()
    if not selected_primary_id:
        st.caption(selected_display_label or "No target selected")
        st.caption("No target tips available.")
        return

    if not summary_rows:
        st.caption("No target tips available.")
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
