from __future__ import annotations

import re
from typing import Any

import streamlit as st

from app_constants import (
    UI_THEME_AURA_DRACULA,
    UI_THEME_BLUE_LIGHT,
    UI_THEME_DARK,
    UI_THEME_LIGHT,
    UI_THEME_MONOKAI_ST3,
    UI_THEME_OPTIONS,
)

PATH_PLOT_BACKGROUND_COLOR = "#E2F0FB"
PATH_PLOT_HORIZONTAL_GRID_COLOR = "rgba(255, 255, 255, 0.95)"
OBSTRUCTION_FILL_COLOR = "rgba(181, 186, 192, 0.40)"
OBSTRUCTION_LINE_COLOR = "rgba(148, 163, 184, 0.95)"
CARDINAL_GRIDLINE_COLOR = "rgba(100, 116, 139, 0.45)"


THEME_PALETTES: dict[str, dict[str, Any]] = {
    UI_THEME_LIGHT: {
        "is_dark": False,
        "small_note_color": "#666666",
        "location_meta_color": "#64748b",
        "location_badges": {
            "manual": {"bg": "#dcfce7", "text": "#166534", "border": "#86efac"},
            "browser": {"bg": "#dbeafe", "text": "#1e40af", "border": "#93c5fd"},
            "ip": {"bg": "#fef3c7", "text": "#92400e", "border": "#fcd34d"},
        },
        "plot": {
            "text": "#111111",
            "muted_text": "#334155",
            "paper_bg": "rgba(0,0,0,0)",
            "plot_bg": PATH_PLOT_BACKGROUND_COLOR,
            "grid": PATH_PLOT_HORIZONTAL_GRID_COLOR,
            "annotation_bg": "rgba(255, 255, 255, 0.45)",
            "annotation_border": "rgba(148, 163, 184, 0.35)",
            "obstruction_fill": OBSTRUCTION_FILL_COLOR,
            "obstruction_line": OBSTRUCTION_LINE_COLOR,
            "cardinal_grid": CARDINAL_GRIDLINE_COLOR,
        },
        "dataframe_styler": {},
        "dataframe_tokens": {},
    },
    UI_THEME_BLUE_LIGHT: {
        "is_dark": False,
        "custom_light_chrome": True,
        "text_color": "#1a1a1a",
        "muted_text_color": "#5b7685",
        "small_note_color": "#5b7685",
        "location_meta_color": "#5b7685",
        "app_bg": "#ffffff",
        "panel_bg": "#ffffff",
        "header_bg": "#007acc",
        "sidebar_bg": "#ffffff",
        "border_color": "#e5e5e5",
        "input_bg": "#ffffff",
        "input_border": "#007acc3a",
        "location_badges": {
            "manual": {"bg": "#e2eeff", "text": "#3778b7", "border": "#c2d0e5"},
            "browser": {"bg": "#d8efff", "text": "#007acc", "border": "#b9ddf6"},
            "ip": {"bg": "#f1f1f1", "text": "#5b7685", "border": "#e5e5e5"},
        },
        "dataframe_tokens": {
            "gdg-accent-color": "#3778b7",
            "gdg-accent-fg": "#e2eeff",
            "gdg-text-dark": "#1a1a1a",
            "gdg-text-medium": "#5b7685",
            "gdg-text-light": "#7d95a1",
            "gdg-text-header": "#007acc",
            "gdg-text-group-header": "#5b7685",
            "gdg-bg-icon-header": "#ffffff",
            "gdg-fg-icon-header": "#3778b7",
            "gdg-bg-cell": "#ffffff",
            "gdg-bg-cell-medium": "#f7f7f7",
            "gdg-bg-header": "#ffffff",
            "gdg-bg-header-has-focus": "#e2eefa",
            "gdg-bg-header-hovered": "#e8e8e8",
            "gdg-bg-bubble": "#e2eeff",
            "gdg-bg-bubble-selected": "#d8efff",
            "gdg-bg-search-result": "rgba(0, 122, 204, 0.14)",
            "gdg-border-color": "#e5e5e5",
            "gdg-horizontal-border-color": "rgba(125, 149, 161, 0.35)",
            "gdg-drilldown-border": "rgba(0, 122, 204, 0.35)",
            "gdg-link-color": "#3778b7",
        },
        "dataframe_styler": {
            "th_bg": "#ffffff",
            "th_text": "#007acc",
            "th_border": "#e5e5e5",
            "td_bg": "#ffffff",
            "td_text": "#1a1a1a",
            "td_border": "#e5e5e5",
        },
        "plot": {
            "text": "#1a1a1a",
            "muted_text": "#5b7685",
            "paper_bg": "rgba(0,0,0,0)",
            "plot_bg": "#ffffff",
            "grid": "rgba(0, 122, 204, 0.16)",
            "annotation_bg": "rgba(226, 238, 251, 0.55)",
            "annotation_border": "#007acc3a",
            "obstruction_fill": "rgba(91, 118, 133, 0.22)",
            "obstruction_line": "rgba(55, 120, 183, 0.85)",
            "cardinal_grid": "rgba(0, 122, 204, 0.25)",
        },
    },
    UI_THEME_DARK: {
        "is_dark": True,
        "text_color": "#e5e7eb",
        "muted_text_color": "#94a3b8",
        "app_bg": "#0b1220",
        "panel_bg": "#111827",
        "header_bg": "rgba(11, 18, 32, 0.92)",
        "sidebar_bg": "#0f172a",
        "border_color": "rgba(148, 163, 184, 0.35)",
        "input_bg": "#0f172a",
        "location_badges": {
            "manual": {"bg": "rgba(34, 197, 94, 0.22)", "text": "#bbf7d0", "border": "rgba(34, 197, 94, 0.45)"},
            "browser": {"bg": "rgba(59, 130, 246, 0.22)", "text": "#bfdbfe", "border": "rgba(59, 130, 246, 0.45)"},
            "ip": {"bg": "rgba(245, 158, 11, 0.22)", "text": "#fde68a", "border": "rgba(245, 158, 11, 0.45)"},
        },
        "dataframe_tokens": {
            "gdg-accent-color": "#38bdf8",
            "gdg-accent-fg": "#0b1220",
            "gdg-text-dark": "#e5e7eb",
            "gdg-text-medium": "#cbd5e1",
            "gdg-text-light": "#94a3b8",
            "gdg-text-header": "#e2e8f0",
            "gdg-text-group-header": "#cbd5e1",
            "gdg-bg-icon-header": "#111827",
            "gdg-fg-icon-header": "#e5e7eb",
            "gdg-bg-cell": "#0f172a",
            "gdg-bg-cell-medium": "#111827",
            "gdg-bg-header": "#111827",
            "gdg-bg-header-has-focus": "#1e293b",
            "gdg-bg-header-hovered": "#1e293b",
            "gdg-bg-bubble": "#1e293b",
            "gdg-bg-bubble-selected": "#334155",
            "gdg-bg-search-result": "rgba(56, 189, 248, 0.20)",
            "gdg-border-color": "rgba(148, 163, 184, 0.35)",
            "gdg-horizontal-border-color": "rgba(148, 163, 184, 0.24)",
            "gdg-drilldown-border": "rgba(148, 163, 184, 0.45)",
            "gdg-link-color": "#7dd3fc",
        },
        "dataframe_styler": {
            "th_bg": "#111827",
            "th_text": "#E2E8F0",
            "th_border": "rgba(148, 163, 184, 0.35)",
            "td_bg": "#0F172A",
            "td_text": "#E5E7EB",
            "td_border": "rgba(148, 163, 184, 0.24)",
        },
        "plot": {
            "text": "#E5E7EB",
            "muted_text": "#CBD5E1",
            "paper_bg": "rgba(0,0,0,0)",
            "plot_bg": "#0F172A",
            "grid": "rgba(148, 163, 184, 0.22)",
            "annotation_bg": "rgba(15, 23, 42, 0.86)",
            "annotation_border": "rgba(148, 163, 184, 0.55)",
            "obstruction_fill": "rgba(71, 85, 105, 0.40)",
            "obstruction_line": "rgba(148, 163, 184, 0.95)",
            "cardinal_grid": "rgba(148, 163, 184, 0.35)",
        },
    },
    UI_THEME_AURA_DRACULA: {
        "is_dark": True,
        "text_color": "#edecee",
        "muted_text_color": "#adacae",
        "app_bg": "#140e1a",
        "panel_bg": "#191120",
        "header_bg": "#100b15",
        "sidebar_bg": "#191120",
        "border_color": "#f694ff81",
        "input_bg": "#140e1a",
        "location_badges": {
            "manual": {"bg": "rgba(97, 255, 202, 0.20)", "text": "#9afee2", "border": "rgba(97, 255, 202, 0.45)"},
            "browser": {"bg": "rgba(162, 119, 255, 0.20)", "text": "#d3bcff", "border": "rgba(162, 119, 255, 0.45)"},
            "ip": {"bg": "rgba(244, 119, 255, 0.20)", "text": "#f8c3ff", "border": "rgba(244, 119, 255, 0.45)"},
        },
        "dataframe_tokens": {
            "gdg-accent-color": "#61ffca",
            "gdg-accent-fg": "#140e1a",
            "gdg-text-dark": "#edecee",
            "gdg-text-medium": "#cdccce",
            "gdg-text-light": "#adacae",
            "gdg-text-header": "#edecee",
            "gdg-text-group-header": "#cdccce",
            "gdg-bg-icon-header": "#100b15",
            "gdg-fg-icon-header": "#61ffca",
            "gdg-bg-cell": "#140e1a",
            "gdg-bg-cell-medium": "#191120",
            "gdg-bg-header": "#100b15",
            "gdg-bg-header-has-focus": "#23192d",
            "gdg-bg-header-hovered": "#23192d",
            "gdg-bg-bubble": "#2e2b38",
            "gdg-bg-bubble-selected": "#3b334b",
            "gdg-bg-search-result": "rgba(162, 119, 255, 0.18)",
            "gdg-border-color": "#f694ff81",
            "gdg-horizontal-border-color": "rgba(246, 148, 255, 0.30)",
            "gdg-drilldown-border": "rgba(246, 148, 255, 0.52)",
            "gdg-link-color": "#61ffca",
        },
        "dataframe_styler": {
            "th_bg": "#100b15",
            "th_text": "#edecee",
            "th_border": "#f694ff81",
            "td_bg": "#140e1a",
            "td_text": "#edecee",
            "td_border": "rgba(246, 148, 255, 0.30)",
        },
        "plot": {
            "text": "#edecee",
            "muted_text": "#cdccce",
            "paper_bg": "rgba(0,0,0,0)",
            "plot_bg": "#140e1a",
            "grid": "rgba(246, 148, 255, 0.18)",
            "annotation_bg": "rgba(20, 14, 26, 0.88)",
            "annotation_border": "rgba(246, 148, 255, 0.52)",
            "obstruction_fill": "rgba(59, 51, 75, 0.42)",
            "obstruction_line": "rgba(162, 119, 255, 0.90)",
            "cardinal_grid": "rgba(97, 255, 202, 0.24)",
        },
    },
    UI_THEME_MONOKAI_ST3: {
        "is_dark": True,
        "text_color": "#f8f8f2",
        "muted_text_color": "#75715e",
        "app_bg": "#272822",
        "panel_bg": "#1e1f1c",
        "header_bg": "#1e1f1c",
        "sidebar_bg": "#1e1f1c",
        "border_color": "#414339",
        "input_bg": "#414339",
        "location_badges": {
            "manual": {"bg": "rgba(166, 226, 46, 0.20)", "text": "#d8f7a8", "border": "rgba(166, 226, 46, 0.45)"},
            "browser": {"bg": "rgba(102, 217, 239, 0.20)", "text": "#bdeef8", "border": "rgba(102, 217, 239, 0.45)"},
            "ip": {"bg": "rgba(253, 151, 31, 0.20)", "text": "#ffd39a", "border": "rgba(253, 151, 31, 0.45)"},
        },
        "dataframe_tokens": {
            "gdg-accent-color": "#75715E",
            "gdg-accent-fg": "#f8f8f2",
            "gdg-text-dark": "#f8f8f2",
            "gdg-text-medium": "#ccccc7",
            "gdg-text-light": "#75715e",
            "gdg-text-header": "#f8f8f2",
            "gdg-text-group-header": "#ccccc7",
            "gdg-bg-icon-header": "#1e1f1c",
            "gdg-fg-icon-header": "#f8f8f2",
            "gdg-bg-cell": "#272822",
            "gdg-bg-cell-medium": "#1e1f1c",
            "gdg-bg-header": "#1e1f1c",
            "gdg-bg-header-has-focus": "#414339",
            "gdg-bg-header-hovered": "#414339",
            "gdg-bg-bubble": "#414339",
            "gdg-bg-bubble-selected": "#49483e",
            "gdg-bg-search-result": "rgba(117, 113, 94, 0.25)",
            "gdg-border-color": "#414339",
            "gdg-horizontal-border-color": "rgba(117, 113, 94, 0.45)",
            "gdg-drilldown-border": "#75715E",
            "gdg-link-color": "#66D9EF",
        },
        "dataframe_styler": {
            "th_bg": "#1e1f1c",
            "th_text": "#f8f8f2",
            "th_border": "#414339",
            "td_bg": "#272822",
            "td_text": "#f8f8f2",
            "td_border": "rgba(117, 113, 94, 0.45)",
        },
        "plot": {
            "text": "#f8f8f2",
            "muted_text": "#ccccc7",
            "paper_bg": "rgba(0,0,0,0)",
            "plot_bg": "#272822",
            "grid": "rgba(117, 113, 94, 0.28)",
            "annotation_bg": "rgba(30, 31, 28, 0.88)",
            "annotation_border": "#75715E",
            "obstruction_fill": "rgba(65, 67, 57, 0.45)",
            "obstruction_line": "rgba(117, 113, 94, 0.95)",
            "cardinal_grid": "rgba(166, 226, 46, 0.26)",
        },
    },
}


def resolve_ui_theme(theme_name: str | None = None) -> str:
    candidate = str(theme_name or "").strip().lower()
    if candidate in UI_THEME_OPTIONS:
        return candidate

    prefs = st.session_state.get("prefs")
    if isinstance(prefs, dict):
        candidate = str(prefs.get("ui_theme", UI_THEME_LIGHT)).strip().lower()
        if candidate in UI_THEME_OPTIONS:
            return candidate
    return UI_THEME_LIGHT


def resolve_theme_palette(theme_name: str | None = None) -> dict[str, Any]:
    theme = resolve_ui_theme(theme_name)
    return THEME_PALETTES.get(theme, THEME_PALETTES[UI_THEME_LIGHT])


def is_dark_ui_theme(theme_name: str | None = None) -> bool:
    return bool(resolve_theme_palette(theme_name).get("is_dark", False))


def normalize_plotly_color(value: Any, fallback: str) -> str:
    raw = str(value if value is not None else fallback).strip()
    if not raw:
        raw = fallback

    eight_digit_hex_match = re.fullmatch(r"#([0-9a-fA-F]{8})", raw)
    if eight_digit_hex_match:
        payload = eight_digit_hex_match.group(1)
        red = int(payload[0:2], 16)
        green = int(payload[2:4], 16)
        blue = int(payload[4:6], 16)
        alpha = int(payload[6:8], 16) / 255.0
        return f"rgba({red},{green},{blue},{alpha:.3f})"

    four_digit_hex_match = re.fullmatch(r"#([0-9a-fA-F]{4})", raw)
    if four_digit_hex_match:
        payload = four_digit_hex_match.group(1)
        red = int(payload[0] * 2, 16)
        green = int(payload[1] * 2, 16)
        blue = int(payload[2] * 2, 16)
        alpha = int(payload[3] * 2, 16) / 255.0
        return f"rgba({red},{green},{blue},{alpha:.3f})"

    return raw


def apply_ui_theme_css(theme_name: str) -> None:
    palette = resolve_theme_palette(theme_name)
    badges = palette.get("location_badges", {})
    manual_badge = badges.get("manual", {})
    browser_badge = badges.get("browser", {})
    ip_badge = badges.get("ip", {})
    collapsed_sidebar_css = """
                section[data-testid="stSidebar"][aria-expanded="false"],
                [data-testid="stSidebar"][aria-expanded="false"] {{
                    min-width: 4.5rem !important;
                    max-width: 4.5rem !important;
                    width: 4.5rem !important;
                    margin-left: 0 !important;
                    left: 0 !important;
                    transform: translateX(0) !important;
                    visibility: visible !important;
                    display: block !important;
                }}
                section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child,
                [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
                    width: 4.5rem !important;
                    min-width: 4.5rem !important;
                    max-width: 4.5rem !important;
                    margin-left: 0 !important;
                    transform: translateX(0) !important;
                }}
                section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNav"] a,
                [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNav"] a {{
                    justify-content: center;
                    padding-left: 0.25rem !important;
                    padding-right: 0.25rem !important;
                }}
                section[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNav"] p,
                [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarNav"] p {{
                    width: 1.35rem !important;
                    max-width: 1.35rem !important;
                    overflow: hidden !important;
                    white-space: nowrap !important;
                    text-overflow: clip !important;
                    margin-left: auto !important;
                    margin-right: auto !important;
                }}
    """

    if palette.get("is_dark", False):
        dataframe_tokens = palette.get("dataframe_tokens", {})
        token_css = "\n".join([f"                    --{key}: {value};" for key, value in dataframe_tokens.items()])
        st.markdown(
            f"""
            <style>
                :root {{
                    --dso-text-color: {palette.get("text_color", "#e5e7eb")};
                    --dso-muted-text-color: {palette.get("muted_text_color", "#94a3b8")};
                    --dso-app-bg: {palette.get("app_bg", "#0b1220")};
                    --dso-panel-bg: {palette.get("panel_bg", "#111827")};
                    --dso-border-color: {palette.get("border_color", "rgba(148, 163, 184, 0.35)")};
                }}
                iframe[title^="streamlit_js_eval"] {{
                    height: 0 !important;
                    min-height: 0 !important;
                    border: 0 !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    display: block !important;
                }}
                [data-testid="stAppViewContainer"] {{
                    background: var(--dso-app-bg);
                }}
                [data-testid="stHeader"] {{
                    background: {palette.get("header_bg", "rgba(11, 18, 32, 0.92)")};
                }}
                [data-testid="stSidebar"] {{
                    background: {palette.get("sidebar_bg", "#0f172a")};
                    border-right: 1px solid var(--dso-border-color);
                }}
                [data-testid="stSidebar"] * {{
                    color: var(--dso-text-color);
                }}
{collapsed_sidebar_css}
                [data-testid="stMainBlockContainer"],
                [data-testid="stAppViewBlockContainer"] {{
                    padding-top: 1.15rem !important;
                }}
                [data-testid="stMainBlockContainer"] * {{
                    color: var(--dso-text-color);
                }}
                [data-testid="stCaptionContainer"], .small-note {{
                    color: var(--dso-muted-text-color) !important;
                }}
                .small-note {{
                    font-size: 0.9rem;
                }}
                .dso-location-meta {{
                    display: flex;
                    align-items: center;
                    gap: 0.4rem;
                    flex-wrap: wrap;
                    margin: 0.125rem 0 0.35rem 0;
                    color: var(--dso-muted-text-color);
                    font-size: 0.88rem;
                    line-height: 1.2rem;
                }}
                .dso-location-source-badge {{
                    display: inline-flex;
                    align-items: center;
                    padding: 0.05rem 0.5rem;
                    border-radius: 999px;
                    border: 1px solid transparent;
                    font-size: 0.72rem;
                    font-weight: 600;
                    letter-spacing: 0.01em;
                    line-height: 1.1rem;
                }}
                .dso-location-source-badge--manual {{
                    background: {manual_badge.get("bg", "rgba(34, 197, 94, 0.22)")};
                    color: {manual_badge.get("text", "#bbf7d0")};
                    border-color: {manual_badge.get("border", "rgba(34, 197, 94, 0.45)")};
                }}
                .dso-location-source-badge--browser {{
                    background: {browser_badge.get("bg", "rgba(59, 130, 246, 0.22)")};
                    color: {browser_badge.get("text", "#bfdbfe")};
                    border-color: {browser_badge.get("border", "rgba(59, 130, 246, 0.45)")};
                }}
                .dso-location-source-badge--ip {{
                    background: {ip_badge.get("bg", "rgba(245, 158, 11, 0.22)")};
                    color: {ip_badge.get("text", "#fde68a")};
                    border-color: {ip_badge.get("border", "rgba(245, 158, 11, 0.45)")};
                }}
                [data-testid="stDataFrame"] {{
                    border: 1px solid var(--dso-border-color);
                    border-radius: 0.5rem;
{token_css}
                }}
                [data-testid="stDataFrame"] div[role="grid"] {{
                    background: {palette.get("input_bg", "#0f172a")};
                }}
                [data-testid="stDataFrame"] [data-baseweb="input"] > div {{
                    background: {palette.get("input_bg", "#0f172a")};
                    color: var(--dso-text-color);
                    border-color: var(--dso-border-color);
                }}
                [data-testid="stDataFrame"] button {{
                    color: var(--dso-text-color);
                }}
                [data-testid="stVerticalBlockBorderWrapper"] > div {{
                    background: var(--dso-panel-bg);
                    border: 1px solid var(--dso-border-color);
                }}
                [data-testid="stTextInputRootElement"] > div,
                [data-testid="stSelectbox"] > div,
                [data-testid="stNumberInput"] > div {{
                    background: {palette.get("input_bg", "#0f172a")};
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        return

    if palette.get("custom_light_chrome", False):
        dataframe_tokens = palette.get("dataframe_tokens", {})
        token_css = "\n".join([f"                    --{key}: {value};" for key, value in dataframe_tokens.items()])
        st.markdown(
            f"""
            <style>
                :root {{
                    --dso-text-color: {palette.get("text_color", "#1a1a1a")};
                    --dso-muted-text-color: {palette.get("muted_text_color", "#64748b")};
                    --dso-app-bg: {palette.get("app_bg", "#ffffff")};
                    --dso-panel-bg: {palette.get("panel_bg", "#ffffff")};
                    --dso-border-color: {palette.get("border_color", "#e5e5e5")};
                }}
                iframe[title^="streamlit_js_eval"] {{
                    height: 0 !important;
                    min-height: 0 !important;
                    border: 0 !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    display: block !important;
                }}
                [data-testid="stAppViewContainer"] {{
                    background: var(--dso-app-bg);
                }}
                [data-testid="stHeader"] {{
                    background: {palette.get("header_bg", "#ffffff")};
                }}
                [data-testid="stSidebar"] {{
                    background: {palette.get("sidebar_bg", "#ffffff")};
                    border-right: 1px solid var(--dso-border-color);
                }}
                [data-testid="stSidebar"] * {{
                    color: var(--dso-text-color);
                }}
{collapsed_sidebar_css}
                [data-testid="stMainBlockContainer"],
                [data-testid="stAppViewBlockContainer"] {{
                    padding-top: 1.15rem !important;
                }}
                [data-testid="stMainBlockContainer"] * {{
                    color: var(--dso-text-color);
                }}
                [data-testid="stCaptionContainer"], .small-note {{
                    color: var(--dso-muted-text-color) !important;
                }}
                .small-note {{
                    font-size: 0.9rem;
                }}
                .dso-location-meta {{
                    display: flex;
                    align-items: center;
                    gap: 0.4rem;
                    flex-wrap: wrap;
                    margin: 0.125rem 0 0.35rem 0;
                    color: var(--dso-muted-text-color);
                    font-size: 0.88rem;
                    line-height: 1.2rem;
                }}
                .dso-location-source-badge {{
                    display: inline-flex;
                    align-items: center;
                    padding: 0.05rem 0.5rem;
                    border-radius: 999px;
                    border: 1px solid transparent;
                    font-size: 0.72rem;
                    font-weight: 600;
                    letter-spacing: 0.01em;
                    line-height: 1.1rem;
                }}
                .dso-location-source-badge--manual {{
                    background: {manual_badge.get("bg", "#dcfce7")};
                    color: {manual_badge.get("text", "#166534")};
                    border-color: {manual_badge.get("border", "#86efac")};
                }}
                .dso-location-source-badge--browser {{
                    background: {browser_badge.get("bg", "#dbeafe")};
                    color: {browser_badge.get("text", "#1e40af")};
                    border-color: {browser_badge.get("border", "#93c5fd")};
                }}
                .dso-location-source-badge--ip {{
                    background: {ip_badge.get("bg", "#fef3c7")};
                    color: {ip_badge.get("text", "#92400e")};
                    border-color: {ip_badge.get("border", "#fcd34d")};
                }}
                [data-testid="stDataFrame"] {{
                    border: 1px solid var(--dso-border-color);
                    border-radius: 0.5rem;
{token_css}
                }}
                [data-testid="stDataFrame"] div[role="grid"] {{
                    background: {palette.get("input_bg", "#ffffff")};
                }}
                [data-testid="stDataFrame"] [data-baseweb="input"] > div {{
                    background: {palette.get("input_bg", "#ffffff")};
                    color: var(--dso-text-color);
                    border-color: {palette.get("input_border", "#e5e5e5")};
                }}
                [data-testid="stDataFrame"] button {{
                    color: var(--dso-text-color);
                }}
                [data-testid="stVerticalBlockBorderWrapper"] > div {{
                    background: var(--dso-panel-bg);
                    border: 1px solid var(--dso-border-color);
                }}
                [data-testid="stTextInputRootElement"] > div,
                [data-testid="stSelectbox"] > div,
                [data-testid="stNumberInput"] > div {{
                    background: {palette.get("input_bg", "#ffffff")};
                    border-color: {palette.get("input_border", "#e5e5e5")};
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <style>
            .small-note {{
                font-size: 0.9rem;
                color: {palette.get("small_note_color", "#666666")};
            }}
{collapsed_sidebar_css}
            [data-testid="stMainBlockContainer"],
            [data-testid="stAppViewBlockContainer"] {{
                padding-top: 1.15rem !important;
            }}
            .dso-location-meta {{
                display: flex;
                align-items: center;
                gap: 0.4rem;
                flex-wrap: wrap;
                margin: 0.125rem 0 0.35rem 0;
                color: {palette.get("location_meta_color", "#64748b")};
                font-size: 0.88rem;
                line-height: 1.2rem;
            }}
            .dso-location-source-badge {{
                display: inline-flex;
                align-items: center;
                padding: 0.05rem 0.5rem;
                border-radius: 999px;
                border: 1px solid transparent;
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.01em;
                line-height: 1.1rem;
            }}
            .dso-location-source-badge--manual {{
                background: {manual_badge.get("bg", "#dcfce7")};
                color: {manual_badge.get("text", "#166534")};
                border-color: {manual_badge.get("border", "#86efac")};
            }}
            .dso-location-source-badge--browser {{
                background: {browser_badge.get("bg", "#dbeafe")};
                color: {browser_badge.get("text", "#1e40af")};
                border-color: {browser_badge.get("border", "#93c5fd")};
            }}
            .dso-location-source-badge--ip {{
                background: {ip_badge.get("bg", "#fef3c7")};
                color: {ip_badge.get("text", "#92400e")};
                border-color: {ip_badge.get("border", "#fcd34d")};
            }}
            iframe[title^="streamlit_js_eval"] {{
                height: 0 !important;
                min-height: 0 !important;
                border: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
                display: block !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def resolve_plot_theme_colors(theme_name: str | None = None) -> dict[str, str]:
    palette = resolve_theme_palette(theme_name)
    plot_colors = palette.get("plot", {})
    return {
        "text": normalize_plotly_color(plot_colors.get("text"), "#111111"),
        "muted_text": normalize_plotly_color(plot_colors.get("muted_text"), "#334155"),
        "paper_bg": normalize_plotly_color(plot_colors.get("paper_bg"), "rgba(0,0,0,0)"),
        "plot_bg": normalize_plotly_color(plot_colors.get("plot_bg"), PATH_PLOT_BACKGROUND_COLOR),
        "grid": normalize_plotly_color(plot_colors.get("grid"), PATH_PLOT_HORIZONTAL_GRID_COLOR),
        "annotation_bg": normalize_plotly_color(plot_colors.get("annotation_bg"), "rgba(255, 255, 255, 0.45)"),
        "annotation_border": normalize_plotly_color(plot_colors.get("annotation_border"), "rgba(148, 163, 184, 0.35)"),
        "obstruction_fill": normalize_plotly_color(plot_colors.get("obstruction_fill"), OBSTRUCTION_FILL_COLOR),
        "obstruction_line": normalize_plotly_color(plot_colors.get("obstruction_line"), OBSTRUCTION_LINE_COLOR),
        "cardinal_grid": normalize_plotly_color(plot_colors.get("cardinal_grid"), CARDINAL_GRIDLINE_COLOR),
    }


def apply_dataframe_styler_theme(styler: Any, *, theme_name: str | None = None) -> Any:
    palette = resolve_theme_palette(theme_name)
    styles = palette.get("dataframe_styler", {})
    if not styles:
        return styler
    return styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", str(styles.get("th_bg", "#111827"))),
                    ("color", str(styles.get("th_text", "#E2E8F0"))),
                    ("border-color", str(styles.get("th_border", "rgba(148, 163, 184, 0.35)"))),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", str(styles.get("td_bg", "#0F172A"))),
                    ("color", str(styles.get("td_text", "#E5E7EB"))),
                    ("border-color", str(styles.get("td_border", "rgba(148, 163, 184, 0.24)"))),
                ],
            },
        ],
        overwrite=False,
    )
