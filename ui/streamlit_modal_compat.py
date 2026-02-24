from __future__ import annotations

from contextlib import contextmanager

import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit import rerun as st_rerun  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from streamlit import experimental_rerun as st_rerun  # type: ignore[no-redef]


class Modal:
    """Compatibility fallback for streamlit_modal.Modal."""

    def __init__(self, title: str, key: str, padding: int = 20, max_width: int = 744) -> None:
        self.title = title
        self.padding = int(padding)
        self.max_width = f"{int(max_width)}px"
        self.key = key

    def is_open(self) -> bool:
        return bool(st.session_state.get(f"{self.key}-opened", False))

    def open(self) -> None:
        st.session_state[f"{self.key}-opened"] = True
        st_rerun()

    def close(self, rerun_condition: bool = True) -> None:
        st.session_state[f"{self.key}-opened"] = False
        if rerun_condition:
            st_rerun()

    @contextmanager
    def container(self):
        st.markdown(
            f"""
            <style>
            div[data-modal-container='true'][key='{self.key}'] {{
                position: fixed;
                width: 100vw !important;
                left: 0;
                z-index: 999992;
            }}

            div[data-modal-container='true'][key='{self.key}'] > div:first-child {{
                margin: auto;
            }}

            div[data-modal-container='true'][key='{self.key}'] h1 a {{
                display: none;
            }}

            div[data-modal-container='true'][key='{self.key}']::before {{
                position: fixed;
                content: ' ';
                left: 0;
                right: 0;
                top: 0;
                bottom: 0;
                z-index: 1000;
                background-color: rgba(50, 50, 50, 0.8);
            }}

            div[data-modal-container='true'][key='{self.key}'] > div:first-child {{
                max-width: {self.max_width};
            }}

            div[data-modal-container='true'][key='{self.key}'] > div:first-child > div:first-child {{
                width: unset !important;
                background-color: #fff;
                padding: {self.padding}px;
                margin-top: {2 * self.padding}px;
                margin-left: -{self.padding}px;
                margin-right: -{self.padding}px;
                margin-bottom: -{2 * self.padding}px;
                z-index: 1001;
                border-radius: 5px;
            }}

            div[data-modal-container='true'][key='{self.key}'] > div:first-child > div:first-child > div:first-child {{
                overflow-y: scroll;
                max-height: 80vh;
                overflow-x: hidden;
                max-width: {self.max_width};
            }}

            div[data-modal-container='true'][key='{self.key}'] > div > div:nth-child(2) {{
                z-index: 1003;
                position: absolute;
            }}

            div[data-modal-container='true'][key='{self.key}'] > div > div:nth-child(2) > div {{
                text-align: right;
                padding-right: {self.padding}px;
                max-width: {self.max_width};
            }}

            div[data-modal-container='true'][key='{self.key}'] > div > div:nth-child(2) > div > button {{
                right: 0;
                margin-top: {2 * self.padding + 14}px;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            container = st.container()
            title_col, close_col = container.columns([0.9, 0.1])
            if self.title:
                with title_col:
                    st.header(self.title)
            with close_col:
                close_clicked = st.button("X", key=f"{self.key}-close")
                if close_clicked:
                    self.close()
            container.divider()

        components.html(
            f"""
            <script>
            const iframes = parent.document.body.getElementsByTagName('iframe');
            let container;
            for (const iframe of iframes) {{
                if ((iframe.srcdoc || '').indexOf('STREAMLIT-MODAL-IFRAME-{self.key}') !== -1) {{
                    container = iframe.parentNode.previousElementSibling || iframe.parentNode.previousSibling;
                    if (!container) continue;
                    container.setAttribute('data-modal-container', 'true');
                    container.setAttribute('key', '{self.key}');
                    const contentDiv = container.querySelector('div:first-child > div:first-child');
                    if (contentDiv) {{
                        contentDiv.style.backgroundColor = getComputedStyle(parent.document.body).backgroundColor;
                    }}
                    break;
                }}
            }}
            </script>
            <!-- STREAMLIT-MODAL-IFRAME-{self.key} -->
            """,
            height=0,
            width=0,
        )

        with container:
            yield container

