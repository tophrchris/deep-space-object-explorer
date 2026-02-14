from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from .engine import TargetTip, TargetTipsContext, collect_target_tips


def render_target_tip(tip: TargetTip) -> None:
    text = str(tip.get("text", "")).strip()
    if not text:
        return

    display = str(tip.get("display", "bullet")).strip().lower() or "bullet"
    if display == "strong":
        st.markdown(f"- **{text}**")
        return
    if display == "muted":
        st.caption(text)
        return
    if display == "warning":
        st.warning(text)
        return
    st.markdown(f"- {text}")


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
    st.markdown("#### Target Tips")
    if not summary_rows:
        st.caption("No target tips available.")
        return

    selected_row = next(
        (row for row in summary_rows if str(row.get("primary_id", "")).strip() == selected_id),
        summary_rows[0],
    )
    context = TargetTipsContext(
        selected_id=selected_id,
        selected_label=selected_label,
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

    st.caption(selected_label)
    if not tips:
        st.caption("No target tips available.")
        return
    for tip in tips:
        render_target_tip(tip)
