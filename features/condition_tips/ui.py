from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st

from .engine import ConditionTip, ConditionTipsContext, collect_condition_tips


def _tip_text_html(text: str) -> str:
    escaped = html.escape(str(text or ""))
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def split_condition_tip_lines(tips: list[ConditionTip]) -> tuple[list[str], list[str], list[str], list[str]]:
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


def render_condition_tips_panel(
    *,
    title: str,
    title_tooltip: str | None = None,
    period_label: str,
    forecast_date_text: str,
    hourly_weather_rows: list[dict[str, Any]],
    summary_row: dict[str, Any] | None,
    temperature_unit: str,
    use_12_hour: bool,
    prepended_muted_lines: list[str] | None = None,
) -> None:
    rendered_title = html.escape(str(title or "").strip() or "Conditions")
    tooltip_text = str(title_tooltip or "").strip()
    if tooltip_text:
        tooltip_attr = html.escape(tooltip_text, quote=True).replace("\n", "&#10;")
        st.markdown(
            (
                f"##### {rendered_title} "
                f"<span title=\"{tooltip_attr}\" style=\"opacity:0.72; cursor:help;\">ⓘ</span>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"##### {rendered_title}")

    context = ConditionTipsContext(
        period_label=period_label,
        forecast_date_text=forecast_date_text,
        hourly_weather_rows=hourly_weather_rows,
        summary_row=summary_row,
        temperature_unit=temperature_unit,
        use_12_hour=use_12_hour,
    )
    tips = collect_condition_tips(context)
    if not tips:
        st.caption("No condition tips available.")
        return

    opportunity_lines, warning_lines, bullet_lines, muted_lines = split_condition_tip_lines(tips)
    if prepended_muted_lines:
        prefixed_muted_lines = [str(line).strip() for line in prepended_muted_lines if str(line).strip()]
        muted_lines = [*prefixed_muted_lines, *muted_lines]
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
