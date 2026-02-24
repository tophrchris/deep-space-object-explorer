from __future__ import annotations

from typing import Any

import streamlit as st

from app_preferences import persist_and_rerun
from features.lists.list_settings_ui import render_lists_settings_section


def render_lists_page(prefs: dict[str, Any]) -> None:
    st.title("Lists")
    st.caption("Manage target lists used across explorer search, overlays, and detail actions.")

    persistence_notice = st.session_state.get("prefs_persistence_notice", "")
    if persistence_notice:
        st.warning(persistence_notice)

    with st.container(border=True):
        render_lists_settings_section(
            prefs,
            persist_and_rerun_fn=persist_and_rerun,
            show_subheader=False,
        )
