from __future__ import annotations

import json
import math

from runtime.lunar_ephemeris import (
    build_hourly_lunar_altitude_map,
    compute_lunar_phase_for_night,
)
from runtime.noaa_goes_cloud_loop import resolve_site_cloud_loop

# Transitional bridge during Explorer split: this module still relies on shared
# helpers/constants from `ui.streamlit_app` until they are extracted.
from ui import streamlit_app as _legacy_ui

def _refresh_legacy_globals() -> None:
    for _name, _value in vars(_legacy_ui).items():
        if _name == "__builtins__":
            continue
        globals().setdefault(_name, _value)


_refresh_legacy_globals()


def _render_observed_cloud_panel(goes_cloud_loop: dict[str, Any] | None) -> None:
    payload = goes_cloud_loop if isinstance(goes_cloud_loop, dict) else {}
    if not bool(payload.get("available", False)):
        return

    panel_label = str(payload.get("panel_label", "")).strip()
    product_name = str(payload.get("product_name", "")).strip()
    source_page_url = str(payload.get("source_page_url", "")).strip()
    image_url = str(payload.get("image_url", "")).strip()
    raw_frame_urls = payload.get("frame_urls", [])
    frame_urls = [str(url).strip() for url in raw_frame_urls if str(url).strip()] if isinstance(raw_frame_urls, list) else []
    site_pin_payload: dict[str, Any] | None = None
    raw_site_pin = payload.get("site_overlay_pin", None)
    if isinstance(raw_site_pin, dict):
        try:
            pin_x = float(raw_site_pin.get("x_frac", "nan"))
            pin_y = float(raw_site_pin.get("y_frac", "nan"))
        except (TypeError, ValueError):
            pin_x = float("nan")
            pin_y = float("nan")
        if (
            math.isfinite(pin_x)
            and math.isfinite(pin_y)
            and 0.0 <= pin_x <= 1.0
            and 0.0 <= pin_y <= 1.0
        ):
            site_pin_payload = {
                "x_frac": pin_x,
                "y_frac": pin_y,
                "label": str(raw_site_pin.get("label", "")).strip() or "Site (approx)",
                "method": str(raw_site_pin.get("method", "")).strip(),
            }

    if panel_label or product_name:
        title_bits = [bit for bit in [panel_label, product_name] if bit]
        st.caption("Observed cloud loop: " + " | ".join(title_bits))

    if len(frame_urls) >= 2:
        sampled_frame_urls = frame_urls[-12:]
        frame_list_json = json.dumps(sampled_frame_urls)
        site_pin_json = json.dumps(site_pin_payload) if site_pin_payload else "null"
        first_frame = sampled_frame_urls[0]
        slideshow_html = f"""
        <html>
          <body style="margin:0;background:transparent;">
            <div id="dso-noaa-cloud-loop-wrap" style="position:relative; width:100%; height:300px; display:flex; justify-content:center; align-items:center; background:#0b0b0b; border-radius:6px; overflow:hidden;">
              <img id="dso-noaa-cloud-loop" src="{first_frame}" alt="Observed cloud loop" style="width:100%; height:100%; object-fit:contain; display:block;" />
              <div id="dso-noaa-cloud-site-pin" title="Approximate site location" style="position:absolute; width:18px; height:24px; display:none; pointer-events:none; transform:translate(-50%, -92%); z-index:3;">
                <svg viewBox="0 0 24 32" width="18" height="24" aria-hidden="true">
                  <path d="M12 31 C12 31 3 20 3 12.5 C3 7.25 7.25 3 12.5 3 C17.75 3 22 7.25 22 12.5 C22 20 12 31 12 31 Z" fill="rgba(255,255,255,0.10)" stroke="#ffffff" stroke-width="2" stroke-linejoin="round"/>
                  <circle cx="12.5" cy="12.5" r="3.5" fill="#ffffff" opacity="0.92"/>
                </svg>
              </div>
              <div id="dso-noaa-cloud-ts" style="position:absolute; right:10px; bottom:10px; background:rgba(0,0,0,0.65); color:#f1f1f1; font:12px/1.2 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; padding:5px 8px; border-radius:4px; pointer-events:none; white-space:nowrap;">
                Frame time (local)
              </div>
            </div>
            <script>
              const frames = {frame_list_json};
              const sitePin = {site_pin_json};
              let idx = 0;
              const wrap = document.getElementById("dso-noaa-cloud-loop-wrap");
              const img = document.getElementById("dso-noaa-cloud-loop");
              const pinEl = document.getElementById("dso-noaa-cloud-site-pin");
              const tsEl = document.getElementById("dso-noaa-cloud-ts");
              const localFormatter = new Intl.DateTimeFormat(undefined, {{
                month: "numeric",
                day: "numeric",
                hour: "numeric",
                minute: "2-digit",
                timeZoneName: "short",
              }});
              function parseNoaaFrameDate(url) {{
                if (typeof url !== "string") return null;
                const match = url.match(/\\/(\\d{{11}})_GOES\\d+/i);
                if (!match) return null;
                const token = match[1];
                const year = Number(token.slice(0, 4));
                const dayOfYear = Number(token.slice(4, 7));
                const hour = Number(token.slice(7, 9));
                const minute = Number(token.slice(9, 11));
                if (!Number.isFinite(year) || !Number.isFinite(dayOfYear) || !Number.isFinite(hour) || !Number.isFinite(minute)) {{
                  return null;
                }}
                const dt = new Date(Date.UTC(year, 0, 1, hour, minute));
                dt.setUTCDate(dayOfYear);
                return dt;
              }}
              function positionSitePin() {{
                if (!pinEl || !wrap || !img || !sitePin || typeof sitePin !== "object") {{
                  if (pinEl) pinEl.style.display = "none";
                  return;
                }}
                const xFrac = Number(sitePin.x_frac);
                const yFrac = Number(sitePin.y_frac);
                if (!Number.isFinite(xFrac) || !Number.isFinite(yFrac)) {{
                  pinEl.style.display = "none";
                  return;
                }}
                const wrapRect = wrap.getBoundingClientRect();
                const wrapW = Number(wrapRect.width) || 0;
                const wrapH = Number(wrapRect.height) || 0;
                const naturalW = Number(img.naturalWidth) || 1;
                const naturalH = Number(img.naturalHeight) || 1;
                if (!(wrapW > 0 && wrapH > 0 && naturalW > 0 && naturalH > 0)) {{
                  pinEl.style.display = "none";
                  return;
                }}
                const imageAspect = naturalW / naturalH;
                const wrapAspect = wrapW / wrapH;
                let drawW = wrapW;
                let drawH = wrapH;
                let offsetX = 0;
                let offsetY = 0;
                if (imageAspect > wrapAspect) {{
                  drawW = wrapW;
                  drawH = wrapW / imageAspect;
                  offsetY = (wrapH - drawH) / 2;
                }} else {{
                  drawH = wrapH;
                  drawW = wrapH * imageAspect;
                  offsetX = (wrapW - drawW) / 2;
                }}
                const clampedX = Math.max(0, Math.min(1, xFrac));
                const clampedY = Math.max(0, Math.min(1, yFrac));
                pinEl.style.left = (offsetX + (clampedX * drawW)) + "px";
                pinEl.style.top = (offsetY + (clampedY * drawH)) + "px";
                pinEl.style.display = "block";
              }}
              function setFrame(frameIndex) {{
                if (!img || !Array.isArray(frames) || frames.length < 1) return;
                idx = ((frameIndex % frames.length) + frames.length) % frames.length;
                const nextUrl = frames[idx];
                img.src = nextUrl;
                if (tsEl) {{
                  const parsedDate = parseNoaaFrameDate(nextUrl);
                  tsEl.textContent = parsedDate
                    ? "Frame (local): " + localFormatter.format(parsedDate)
                    : "Frame (local time unavailable)";
                }}
              }}
              if (img) {{
                img.addEventListener("load", () => {{
                  positionSitePin();
                }});
              }}
              if (typeof window !== "undefined") {{
                window.addEventListener("resize", () => {{
                  positionSitePin();
                }});
              }}
              if (img && Array.isArray(frames) && frames.length > 0) {{
                setFrame(0);
              }}
              if (img && Array.isArray(frames) && frames.length > 1) {{
                setInterval(() => {{
                  setFrame(idx + 1);
                }}, 700);
              }}
              setTimeout(positionSitePin, 0);
            </script>
          </body>
        </html>
        """
        try:
            st.components.v1.html(slideshow_html, height=310, scrolling=False)
        except Exception:
            if image_url:
                st.image(image_url, use_container_width=True)
    elif image_url:
        st.image(image_url, use_container_width=True)

    if source_page_url:
        st.caption(f"[NOAA GOES source page]({source_page_url})")

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
    lunar_hourly_altitude_by_hour = build_hourly_lunar_altitude_map(
        lat=location_lat,
        lon=location_lon,
        tz_name=weather_tzinfo.key,
        start_local_iso=weather_window_start.isoformat(),
        end_local_iso=weather_window_end.isoformat(),
        sample_minutes=10,
    )
    lunar_phase_payload = compute_lunar_phase_for_night(
        tz_name=weather_tzinfo.key,
        start_local_iso=weather_window_start.isoformat(),
        end_local_iso=weather_window_end.isoformat(),
    )
    lunar_phase_key = str((lunar_phase_payload or {}).get("phase_key", "")).strip() or None
    goes_cloud_loop = resolve_site_cloud_loop(location_lat, location_lon)
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
        lunar_hourly_altitude_by_hour=lunar_hourly_altitude_by_hour,
        lunar_phase_key=lunar_phase_key,
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
            left_col, right_col = st.columns([3, 7], gap="medium")
            five_day_container = left_col
            with right_col:
                hourly_container = st.container()
                conditions_container = st.container()

        with five_day_container:
            st.markdown("5-night astronomy forecast.")
            render_astronomy_forecast_summary(
                astronomy_summary,
                temperature_unit=temperature_unit,
                selected_day_offset=weather_forecast_day_offset,
            )
            st.caption("Click any row to set the active weather night across the page.")
            st.markdown(site_conditions_legends_table_html(), unsafe_allow_html=True)
            _render_observed_cloud_panel(goes_cloud_loop)

        with hourly_container:
            st.markdown(
                f"Hourly weather for {hourly_title_period}."
            )
            weather_display = weather_matrix.reset_index().rename(columns={"index": "Element"})
            weather_tooltip_display = weather_tooltips.reset_index().rename(columns={"index": "Element"})
            weather_indicator_display = weather_indicators.reset_index().rename(columns={"index": "Element"})

            hourly_header_to_hour_key: dict[str, str] = {}
            hourly_header_column_states: dict[str, str] = {}
            current_local_hour_floor: pd.Timestamp | None = None
            if weather_forecast_day_offset == 0:
                try:
                    current_local_hour_floor = pd.Timestamp(datetime.now(weather_tzinfo)).floor("h")
                except Exception:
                    current_local_hour_floor = None
            for hourly_row in hourly_weather_rows:
                time_iso = str(hourly_row.get("time_iso", "")).strip()
                if not time_iso:
                    continue
                try:
                    timestamp = pd.Timestamp(time_iso)
                except Exception:
                    continue
                header_label = format_hour_label(timestamp, use_12_hour=use_12_hour)
                header_hour_key = normalize_hour_key(timestamp)
                if header_label and header_hour_key and header_label not in hourly_header_to_hour_key:
                    hourly_header_to_hour_key[header_label] = header_hour_key
                if header_label and current_local_hour_floor is not None and header_label not in hourly_header_column_states:
                    hour_floor_local = pd.Timestamp(timestamp).floor("h")
                    if hour_floor_local < current_local_hour_floor:
                        hourly_header_column_states[header_label] = "past"
                    elif hour_floor_local == current_local_hour_floor:
                        hourly_header_column_states[header_label] = "current"

            clicked_hour_column_label = render_hourly_weather_matrix(
                weather_display,
                temperature_unit=temperature_unit,
                tooltip_frame=weather_tooltip_display,
                indicator_frame=weather_indicator_display,
                hour_column_states=(hourly_header_column_states or None),
            )
            st.caption("Click an hour column header to search Recommended Targets for that hour.")
            clicked_hour_key = (
                hourly_header_to_hour_key.get(str(clicked_hour_column_label).strip(), "")
                if clicked_hour_column_label is not None
                else ""
            )
            if clicked_hour_key:
                click_token = f"{weather_forecast_day_offset}:{clicked_hour_key}"
                if str(st.session_state.get("hourly_weather_selected_hour_click_token", "")).strip() != click_token:
                    st.session_state["hourly_weather_selected_hour_click_token"] = click_token
                    st.session_state["recommended_targets_pending_hour_key"] = clicked_hour_key

        with conditions_container:
            with st.container(border=True):
                selected_sunset = str((selected_summary_row or {}).get("Sunset", "")).strip()
                selected_sunrise = str((selected_summary_row or {}).get("Sunrise", "")).strip()
                prepended_muted_condition_lines: list[str] = []
                if selected_sunset or selected_sunrise:
                    sunset_text = selected_sunset or "--"
                    sunrise_text = selected_sunrise or "--"
                    prepended_muted_condition_lines.append(
                        f"Sunset/Sunrise (local): {sunset_text} / {sunrise_text}"
                    )
                render_condition_tips_panel(
                    title=condition_tips_title,
                    title_tooltip=condition_tips_tooltip,
                    period_label=forecast_title_label,
                    forecast_date_text=forecast_date_text,
                    hourly_weather_rows=hourly_weather_rows,
                    summary_row=selected_summary_row,
                    temperature_unit=temperature_unit,
                    use_12_hour=use_12_hour,
                    prepended_muted_lines=prepended_muted_condition_lines,
                )

    selected_row = resolve_selected_row(catalog)
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
