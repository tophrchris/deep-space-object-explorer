from __future__ import annotations

# Transitional bridge during Explorer split: this module still relies on shared
# helpers/constants from `ui.streamlit_app` until they are extracted.
from ui import streamlit_app as _legacy_ui

def _refresh_legacy_globals() -> None:
    for _name, _value in vars(_legacy_ui).items():
        if _name == "__builtins__":
            continue
        globals().setdefault(_name, _value)


_refresh_legacy_globals()

def _render_explorer_page_impl(
    catalog: pd.DataFrame,
    catalog_meta: dict[str, Any],
    prefs: dict[str, Any],
    temperature_unit: str,
    use_12_hour: bool,
    detail_stack_vertical: bool,
    browser_locale: str | None = None,
    browser_month_day_pattern: str | None = None,
) -> None:
    _refresh_legacy_globals()
    persistence_notice = st.session_state.get("prefs_persistence_notice", "")
    if persistence_notice:
        st.warning(persistence_notice)

    now_utc = datetime.now(timezone.utc)
    refresh_status_html = (
        "<p class='small-note' style='text-align: right;'>Alt/Az auto-refresh 15m<br>"
        f"Updated: {now_utc.strftime('%Y-%m-%d')} "
        f"{format_display_time(now_utc, use_12_hour=use_12_hour, include_seconds=True)} UTC"
        "</p>"
    )

    site_ids = site_ids_in_order(prefs)
    active_site_id = get_active_site_id(prefs)
    if active_site_id not in site_ids and site_ids:
        active_site_id = site_ids[0]
        set_active_site(prefs, active_site_id)
    st.title("Observation Planner")
    st.caption(f"Catalog rows loaded: {int(catalog_meta.get('row_count', len(catalog)))}")

    location = prefs["location"]
    if not is_location_configured(location):
        st.warning("Observation site location is not set. Open the Sites page to set location and obstructions.")
        return
    location_lat = float(location["lat"])
    location_lon = float(location["lon"])
    weather_forecast_day_offset = resolve_weather_forecast_day_offset(
        max_offset=ASTRONOMY_FORECAST_NIGHTS - 1
    )
    weather_window_start, weather_window_end, weather_tzinfo = weather_forecast_window(
        location_lat,
        location_lon,
        day_offset=weather_forecast_day_offset,
    )
    forecast_title_label = describe_weather_forecast_period(weather_forecast_day_offset)
    forecast_date_text = format_weather_forecast_date(
        weather_window_start,
        browser_locale=browser_locale,
        browser_month_day_pattern=browser_month_day_pattern,
    )
    hourly_title_period = format_hourly_weather_title_period(
        weather_forecast_day_offset,
        weather_window_start,
        browser_locale=browser_locale,
        browser_month_day_pattern=browser_month_day_pattern,
    )
    astronomy_summary = build_astronomy_forecast_summary(
        location_lat,
        location_lon,
        tz_name=weather_tzinfo.key,
        temperature_unit=temperature_unit,
        browser_locale=browser_locale,
        browser_month_day_pattern=browser_month_day_pattern,
        nights=ASTRONOMY_FORECAST_NIGHTS,
    )
    hourly_weather_rows = fetch_hourly_weather(
        lat=location_lat,
        lon=location_lon,
        tz_name=weather_tzinfo.key,
        start_local_iso=weather_window_start.isoformat(),
        end_local_iso=weather_window_end.isoformat(),
        hourly_fields=EXTENDED_FORECAST_HOURLY_FIELDS,
    )
    night_rating_details = compute_night_rating_details(
        hourly_weather_rows,
        temperature_unit=temperature_unit,
    )
    condition_tips_title = format_condition_tips_title(
        weather_forecast_day_offset,
        weather_window_start,
        rating_emoji=(str(night_rating_details.get("emoji", "")).strip() if night_rating_details else None),
    )
    condition_tips_tooltip = format_night_rating_tooltip(night_rating_details)
    weather_matrix, weather_tooltips, weather_indicators = build_hourly_weather_matrix(
        hourly_weather_rows,
        use_12_hour=use_12_hour,
        temperature_unit=temperature_unit,
    )
    selected_summary_row: dict[str, Any] | None = None
    if not astronomy_summary.empty:
        for _, summary_row in astronomy_summary.iterrows():
            row_day_offset = normalize_weather_forecast_day_offset(
                summary_row.get("day_offset", 0),
                max_offset=ASTRONOMY_FORECAST_NIGHTS - 1,
            )
            if row_day_offset == weather_forecast_day_offset:
                selected_summary_row = dict(summary_row)
                break

    with st.container(border=True):
        st.markdown(f"### Site Conditions for {resolve_location_display_name(location)}")

        if detail_stack_vertical:
            five_day_container = st.container()
            hourly_container = st.container()
            conditions_container = st.container()
        else:
            top_cols = st.columns([3, 4, 3], gap="medium")
            five_day_container = top_cols[0]
            hourly_container = top_cols[1]
            conditions_container = top_cols[2]

        with five_day_container:
            st.markdown("5-night astronomy forecast.")
            render_astronomy_forecast_summary(
                astronomy_summary,
                temperature_unit=temperature_unit,
                selected_day_offset=weather_forecast_day_offset,
            )
            st.caption("Click any row to set the active weather night across the page.")
            st.caption(WEATHER_ALERT_INDICATOR_LEGEND_CAPTION)
            st.markdown(night_time_clarity_scale_legend_html(), unsafe_allow_html=True)
            st.markdown(dew_risk_scale_legend_html(), unsafe_allow_html=True)

        with hourly_container:
            st.markdown(
                f"Hourly weather for {hourly_title_period}."
            )
            weather_display = weather_matrix.reset_index().rename(columns={"index": "Element"})
            weather_tooltip_display = weather_tooltips.reset_index().rename(columns={"index": "Element"})
            weather_indicator_display = weather_indicators.reset_index().rename(columns={"index": "Element"})
            render_hourly_weather_matrix(
                weather_display,
                temperature_unit=temperature_unit,
                tooltip_frame=weather_tooltip_display,
                indicator_frame=weather_indicator_display,
            )

        with conditions_container:
            with st.container(border=True):
                render_condition_tips_panel(
                    title=condition_tips_title,
                    title_tooltip=condition_tips_tooltip,
                    period_label=forecast_title_label,
                    forecast_date_text=forecast_date_text,
                    hourly_weather_rows=hourly_weather_rows,
                    summary_row=selected_summary_row,
                    temperature_unit=temperature_unit,
                    use_12_hour=use_12_hour,
                )

    selected_row = resolve_selected_row(catalog)
    active_preview_list_ids = get_list_ids(prefs, get_active_preview_list_id(prefs))
    if selected_row is not None or bool(active_preview_list_ids):
        render_detail_panel(
            selected=selected_row,
            catalog=catalog,
            prefs=prefs,
            temperature_unit=temperature_unit,
            use_12_hour=use_12_hour,
            detail_stack_vertical=detail_stack_vertical,
            weather_forecast_day_offset=weather_forecast_day_offset,
        )
    st.markdown(refresh_status_html, unsafe_allow_html=True)

