from __future__ import annotations

from typing import Any

import streamlit as st

from app_constants import UI_THEME_DARK, UI_THEME_LIGHT, UI_THEME_OPTIONS

PATH_PLOT_BACKGROUND_COLOR = "#E2F0FB"
PATH_PLOT_HORIZONTAL_GRID_COLOR = "rgba(255, 255, 255, 0.95)"
OBSTRUCTION_FILL_COLOR = "rgba(181, 186, 192, 0.40)"
OBSTRUCTION_LINE_COLOR = "rgba(148, 163, 184, 0.95)"
CARDINAL_GRIDLINE_COLOR = "rgba(100, 116, 139, 0.45)"


def apply_ui_theme_css(theme_name: str) -> None:
    theme = str(theme_name or "").strip().lower()
    is_dark = theme == UI_THEME_DARK
    if is_dark:
        st.markdown(
            """
            <style>
                :root {
                    --dso-text-color: #e5e7eb;
                    --dso-muted-text-color: #94a3b8;
                    --dso-app-bg: #0b1220;
                    --dso-panel-bg: #111827;
                    --dso-border-color: rgba(148, 163, 184, 0.35);
                }
                iframe[title^="streamlit_js_eval"] {
                    height: 0 !important;
                    min-height: 0 !important;
                    border: 0 !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    display: block !important;
                }
                [data-testid="stAppViewContainer"] {
                    background: var(--dso-app-bg);
                }
                [data-testid="stHeader"] {
                    background: rgba(11, 18, 32, 0.92);
                }
                [data-testid="stSidebar"] {
                    background: #0f172a;
                    border-right: 1px solid var(--dso-border-color);
                }
                [data-testid="stSidebar"] * {
                    color: var(--dso-text-color);
                }
                [data-testid="stMainBlockContainer"] * {
                    color: var(--dso-text-color);
                }
                [data-testid="stCaptionContainer"], .small-note {
                    color: var(--dso-muted-text-color) !important;
                }
                .small-note {
                    font-size: 0.9rem;
                }
                [data-testid="stDataFrame"] {
                    border: 1px solid var(--dso-border-color);
                    border-radius: 0.5rem;
                    /* Glide Data Grid theme tokens used by st.dataframe. */
                    --gdg-accent-color: #38bdf8;
                    --gdg-accent-fg: #0b1220;
                    --gdg-text-dark: #e5e7eb;
                    --gdg-text-medium: #cbd5e1;
                    --gdg-text-light: #94a3b8;
                    --gdg-text-header: #e2e8f0;
                    --gdg-text-group-header: #cbd5e1;
                    --gdg-bg-icon-header: #111827;
                    --gdg-fg-icon-header: #e5e7eb;
                    --gdg-bg-cell: #0f172a;
                    --gdg-bg-cell-medium: #111827;
                    --gdg-bg-header: #111827;
                    --gdg-bg-header-has-focus: #1e293b;
                    --gdg-bg-header-hovered: #1e293b;
                    --gdg-bg-bubble: #1e293b;
                    --gdg-bg-bubble-selected: #334155;
                    --gdg-bg-search-result: rgba(56, 189, 248, 0.20);
                    --gdg-border-color: rgba(148, 163, 184, 0.35);
                    --gdg-horizontal-border-color: rgba(148, 163, 184, 0.24);
                    --gdg-drilldown-border: rgba(148, 163, 184, 0.45);
                    --gdg-link-color: #7dd3fc;
                }
                [data-testid="stDataFrame"] div[role="grid"] {
                    background: #0f172a;
                }
                [data-testid="stDataFrame"] [data-baseweb="input"] > div {
                    background: #0f172a;
                    color: var(--dso-text-color);
                    border-color: var(--dso-border-color);
                }
                [data-testid="stDataFrame"] button {
                    color: var(--dso-text-color);
                }
                [data-testid="stVerticalBlockBorderWrapper"] > div {
                    background: var(--dso-panel-bg);
                    border: 1px solid var(--dso-border-color);
                }
                [data-testid="stTextInputRootElement"] > div,
                [data-testid="stSelectbox"] > div,
                [data-testid="stNumberInput"] > div {
                    background: #0f172a;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <style>
            .small-note {
                font-size: 0.9rem;
                color: #666;
            }
            iframe[title^="streamlit_js_eval"] {
                height: 0 !important;
                min-height: 0 !important;
                border: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
                display: block !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def resolve_plot_theme_colors(theme_name: str | None = None) -> dict[str, str]:
    theme = resolve_ui_theme(theme_name)
    if theme == UI_THEME_DARK:
        return {
            "text": "#E5E7EB",
            "muted_text": "#CBD5E1",
            # Keep only the plotting area tinted; leave figure chrome transparent.
            "paper_bg": "rgba(0,0,0,0)",
            "plot_bg": "#0F172A",
            "grid": "rgba(148, 163, 184, 0.22)",
            "annotation_bg": "rgba(15, 23, 42, 0.86)",
            "annotation_border": "rgba(148, 163, 184, 0.55)",
            "obstruction_fill": "rgba(71, 85, 105, 0.40)",
            "obstruction_line": "rgba(148, 163, 184, 0.95)",
            "cardinal_grid": "rgba(148, 163, 184, 0.35)",
        }
    return {
        "text": "#111111",
        "muted_text": "#334155",
        # Keep only the plotting area tinted; leave figure chrome transparent.
        "paper_bg": "rgba(0,0,0,0)",
        "plot_bg": PATH_PLOT_BACKGROUND_COLOR,
        "grid": PATH_PLOT_HORIZONTAL_GRID_COLOR,
        "annotation_bg": "rgba(255, 255, 255, 0.45)",
        "annotation_border": "rgba(148, 163, 184, 0.35)",
        "obstruction_fill": OBSTRUCTION_FILL_COLOR,
        "obstruction_line": OBSTRUCTION_LINE_COLOR,
        "cardinal_grid": CARDINAL_GRIDLINE_COLOR,
    }


def apply_dataframe_styler_theme(styler: Any, *, theme_name: str | None = None) -> Any:
    if resolve_ui_theme(theme_name) != UI_THEME_DARK:
        return styler

    return styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#111827"),
                    ("color", "#E2E8F0"),
                    ("border-color", "rgba(148, 163, 184, 0.35)"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#0F172A"),
                    ("color", "#E5E7EB"),
                    ("border-color", "rgba(148, 163, 184, 0.24)"),
                ],
            },
        ],
        overwrite=False,
    )
