from __future__ import annotations

# Transitional bridge during UI split: this module still relies on shared
# helpers/constants from `ui.streamlit_app` until they are extracted.
from ui import streamlit_app as _legacy_ui

def _refresh_legacy_globals() -> None:
    for _name, _value in vars(_legacy_ui).items():
        if _name == "__builtins__":
            continue
        globals().setdefault(_name, _value)


_refresh_legacy_globals()

def main() -> None:
    _refresh_legacy_globals()
    st_autorefresh(interval=900_000, key="altaz_refresh")

    if "prefs_bootstrap_runs" not in st.session_state:
        st.session_state["prefs_bootstrap_runs"] = 0
    if "prefs_bootstrapped" not in st.session_state:
        st.session_state["prefs_bootstrapped"] = False
    if "prefs" not in st.session_state:
        st.session_state["prefs"] = default_preferences()

    if not bool(st.session_state.get("prefs_bootstrapped", False)):
        runs = int(st.session_state.get("prefs_bootstrap_runs", 0))
        loaded_prefs, retry_needed = load_preferences()
        st.session_state["prefs"] = loaded_prefs
        runs += 1
        st.session_state["prefs_bootstrap_runs"] = runs

        if retry_needed and runs < PREFS_BOOTSTRAP_MAX_RUNS:
            st_autorefresh(
                interval=PREFS_BOOTSTRAP_RETRY_INTERVAL_MS,
                limit=1,
                key=f"prefs_bootstrap_wait_{runs}",
            )
            st.stop()

        st.session_state["prefs_bootstrapped"] = True

    prefs = ensure_preferences_shape(st.session_state["prefs"])
    sync_active_site_to_legacy_fields(prefs)
    st.session_state["prefs"] = prefs
    maybe_sync_prefs_with_google_drive(prefs)

    if not is_location_configured(prefs.get("location")):
        if not bool(st.session_state.get("initial_location_fallback_attempted", False)):
            browser_applied = False
            initial_geo_payload = get_geolocation(component_key="initial_site_browser_geo")
            if isinstance(initial_geo_payload, dict):
                coords = initial_geo_payload.get("coords")
                if isinstance(coords, dict):
                    apply_browser_geolocation_payload(prefs, initial_geo_payload)
                    browser_applied = is_location_configured(prefs.get("location"))

            st.session_state["initial_location_fallback_attempted"] = True
            if not browser_applied:
                resolved = approximate_location_from_ip()
                if resolved:
                    resolved_label, kept_site_name = apply_resolved_location(prefs, resolved)
                    st.session_state["prefs"] = prefs
                    save_preferences(prefs)
                    st.session_state["location_notice"] = (
                        f"Approximate location applied: {resolved_label}. Site name unchanged."
                        if kept_site_name
                        else f"Approximate location applied: {resolved_label}."
                    )
                else:
                    st.session_state["location_notice"] = (
                        "Location not set. Open the Sites page to set it manually or via browser geolocation (IP estimate is fallback)."
                    )

    theme_label_to_id = {
        "Light": UI_THEME_LIGHT,
        "Blue Light": UI_THEME_BLUE_LIGHT,
        "Dark": UI_THEME_DARK,
        "Aura Dracula": UI_THEME_AURA_DRACULA,
        "Monokai ST3": UI_THEME_MONOKAI_ST3,
    }
    current_theme = str(prefs.get("ui_theme", UI_THEME_LIGHT)).strip().lower()
    if current_theme not in set(theme_label_to_id.values()):
        current_theme = UI_THEME_LIGHT
    apply_ui_theme_css(current_theme)
    # Safety override: keep the app in wide layout even if Streamlit page config
    # is ignored during a hot-reload/circular-import edge case.
    st.markdown(
        """
        <style>
        div.block-container {
          max-width: min(1800px, 98vw);
          padding-left: 1.25rem;
          padding-right: 1.25rem;
        }
        @media (max-width: 900px) {
          div.block-container {
            max-width: 100vw;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    browser_language_raw = eval_js_hidden("window.navigator.language", key="browser_language_pref")
    if isinstance(browser_language_raw, str) and browser_language_raw.strip():
        st.session_state["browser_language"] = browser_language_raw.strip()
    browser_language = st.session_state.get("browser_language")
    browser_hour_cycle_raw = eval_js_hidden(
        (
            "new Intl.DateTimeFormat(window.navigator.language, "
            "{hour: 'numeric', minute: '2-digit'}).resolvedOptions().hourCycle"
        ),
        key="browser_hour_cycle_pref",
    )
    if isinstance(browser_hour_cycle_raw, str) and browser_hour_cycle_raw.strip():
        st.session_state["browser_hour_cycle"] = browser_hour_cycle_raw.strip()
    browser_hour_cycle = st.session_state.get("browser_hour_cycle")
    browser_month_day_pattern_raw = eval_js_hidden(
        (
            "new Intl.DateTimeFormat(window.navigator.language, {month: 'numeric', day: 'numeric'})"
            ".formatToParts(new Date(2024, 1, 15))"
            ".map((part) => `${part.type}:${part.value}`)"
            ".join('|')"
        ),
        key="browser_month_day_pattern_pref",
    )
    if isinstance(browser_month_day_pattern_raw, str) and browser_month_day_pattern_raw.strip():
        st.session_state["browser_month_day_pattern"] = browser_month_day_pattern_raw.strip()
    browser_month_day_pattern = st.session_state.get("browser_month_day_pattern")
    use_12_hour = resolve_12_hour_clock(browser_language, browser_hour_cycle)
    viewport_width_raw = eval_js_hidden(
        (
            "(() => {"
            "  try {"
            "    const values = ["
            "      Number(window.innerWidth || 0),"
            "      Number(document?.documentElement?.clientWidth || 0),"
            "      Number(window.parent?.innerWidth || 0),"
            "      Number(window.top?.innerWidth || 0)"
            "    ].filter((v) => Number.isFinite(v) && v > 0);"
            "    return values.length ? Math.max(...values) : 0;"
            "  } catch (e) {"
            "    return Number(window.innerWidth || 0);"
            "  }"
            "})()"
        ),
        key="browser_viewport_width_probe",
    )
    if isinstance(viewport_width_raw, (int, float)) and float(viewport_width_raw) > 0:
        st.session_state["browser_viewport_width"] = int(float(viewport_width_raw))
    elif isinstance(viewport_width_raw, str):
        try:
            parsed_width = float(viewport_width_raw.strip())
            if parsed_width > 0:
                st.session_state["browser_viewport_width"] = int(parsed_width)
        except (TypeError, ValueError):
            pass
    browser_viewport_width = int(st.session_state.get("browser_viewport_width", 1920))
    detail_stack_vertical = browser_viewport_width < DETAIL_PANE_STACK_BREAKPOINT_PX

    effective_temperature_unit = resolve_temperature_unit(
        str(prefs.get("temperature_unit", "auto")),
        browser_language,
    )

    catalog, catalog_meta = load_catalog_app_cached(CATALOG_CACHE_PATH)
    sanitize_saved_lists(catalog=catalog, prefs=prefs)

    def explorer_page() -> None:
        render_explorer_page(
            catalog=catalog,
            catalog_meta=catalog_meta,
            prefs=prefs,
            temperature_unit=effective_temperature_unit,
            use_12_hour=use_12_hour,
            detail_stack_vertical=detail_stack_vertical,
            browser_locale=browser_language,
            browser_month_day_pattern=browser_month_day_pattern,
        )

    def sites_page() -> None:
        render_sites_page(prefs=prefs)

    def equipment_page() -> None:
        render_equipment_page(prefs=prefs)

    def lists_page() -> None:
        render_lists_page(prefs=prefs)

    def settings_page() -> None:
        render_settings_page(
            catalog_meta=catalog_meta,
            prefs=prefs,
            browser_locale=browser_language,
        )

    if hasattr(st, "navigation") and hasattr(st, "Page"):
        navigation = st.navigation(
            [
                st.Page(explorer_page, title="Explorer", icon="✨", default=True),
                st.Page(sites_page, title="Sites", icon="📍"),
                st.Page(equipment_page, title="Equipment", icon="🧰"),
                st.Page(lists_page, title="Lists", icon="📚"),
                st.Page(settings_page, title="Settings", icon="⚙️"),
            ]
        )
        render_sidebar_active_settings(
            prefs=prefs,
            theme_label_to_id=theme_label_to_id,
        )
        navigation.run()
        return

    selected_page = st.sidebar.radio(
        "Page",
        ["Explorer", "Sites", "Equipment", "Lists", "Settings"],
        key="app_page_selector",
    )
    render_sidebar_active_settings(
        prefs=prefs,
        theme_label_to_id=theme_label_to_id,
    )
    if selected_page == "Sites":
        sites_page()
        return
    if selected_page == "Equipment":
        equipment_page()
        return
    if selected_page == "Lists":
        lists_page()
        return
    if selected_page == "Settings":
        settings_page()
        return
    explorer_page()

