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

def event_time_value(series: pd.Series | None) -> pd.Timestamp | pd.NaTType:
    _refresh_legacy_globals()
    if series is None:
        return pd.NaT
    try:
        return pd.Timestamp(series["time_local"])
    except Exception:
        return pd.NaT


def compute_total_visible_time(track: pd.DataFrame) -> timedelta:
    _refresh_legacy_globals()
    if track.empty or "visible" not in track:
        return timedelta(0)

    visible_mask = track["visible"].fillna(False).astype(bool).to_numpy()
    if not visible_mask.any():
        return timedelta(0)

    times = pd.to_datetime(track["time_local"])
    diffs = times.diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    step = positive_diffs.median() if not positive_diffs.empty else pd.Timedelta(minutes=10)

    total = pd.Timedelta(0)
    idx = 0
    while idx < len(visible_mask):
        if not visible_mask[idx]:
            idx += 1
            continue
        start_idx = idx
        while idx + 1 < len(visible_mask) and visible_mask[idx + 1]:
            idx += 1
        end_idx = idx
        total += (times.iloc[end_idx] - times.iloc[start_idx]) + step
        idx += 1

    if total < pd.Timedelta(0):
        return timedelta(0)
    return total.to_pytimedelta()


def compute_remaining_visible_time(track: pd.DataFrame, now: pd.Timestamp | datetime | None = None) -> timedelta:
    _refresh_legacy_globals()
    if track.empty or "visible" not in track:
        return timedelta(0)

    visible_mask = track["visible"].fillna(False).astype(bool).to_numpy()
    if not visible_mask.any():
        return timedelta(0)

    times = pd.to_datetime(track["time_local"])
    if times.empty:
        return timedelta(0)

    diffs = times.diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    step = positive_diffs.median() if not positive_diffs.empty else pd.Timedelta(minutes=10)

    first_time = pd.Timestamp(times.iloc[0])
    if now is None:
        now_ts = pd.Timestamp.now(tz=first_time.tzinfo) if first_time.tzinfo is not None else pd.Timestamp.now()
    else:
        now_ts = pd.Timestamp(now)
        if first_time.tzinfo is not None and now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize(first_time.tzinfo)
        elif first_time.tzinfo is None and now_ts.tzinfo is not None:
            now_ts = now_ts.tz_localize(None)
        elif first_time.tzinfo is not None and now_ts.tzinfo is not None:
            now_ts = now_ts.tz_convert(first_time.tzinfo)

    total = pd.Timedelta(0)
    idx = 0
    while idx < len(visible_mask):
        if not visible_mask[idx]:
            idx += 1
            continue
        start_idx = idx
        while idx + 1 < len(visible_mask) and visible_mask[idx + 1]:
            idx += 1
        end_idx = idx
        interval_start = pd.Timestamp(times.iloc[start_idx])
        interval_end = pd.Timestamp(times.iloc[end_idx]) + step
        effective_start = max(interval_start, now_ts)
        if interval_end > effective_start:
            total += interval_end - effective_start
        idx += 1

    if total < pd.Timedelta(0):
        return timedelta(0)
    return total.to_pytimedelta()


def format_duration_hm(duration: timedelta) -> str:
    _refresh_legacy_globals()
    total_minutes = max(0, int(round(duration.total_seconds() / 60.0)))
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"



def build_sky_position_summary_rows(
    selected_id: str | None,
    selected_label: str | None,
    selected_type_group: str | None,
    selected_color: str | None,
    selected_events: dict[str, pd.Series | None] | None,
    selected_track: pd.DataFrame | None,
    overlay_tracks: list[dict[str, Any]],
    list_member_ids: set[str],
    selected_metadata: dict[str, Any] | None = None,
    now_local: pd.Timestamp | datetime | None = None,
    row_order_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    _refresh_legacy_globals()
    def _build_row(
        primary_id: str,
        label: str,
        type_group: str,
        color: str,
        events: dict[str, pd.Series | None],
        is_in_list: bool,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        culmination = events.get("culmination")
        culmination_dir = str(culmination["wind16"]) if culmination is not None else "--"
        culmination_alt = f"{float(culmination['alt']):.1f} deg" if culmination is not None else "--"
        metadata = metadata if isinstance(metadata, dict) else {}
        return {
            "primary_id": primary_id,
            "line_color": color,
            "target": label,
            "object_type_group": type_group or "other",
            "object_type": str(metadata.get("object_type", "") or "").strip(),
            "magnitude": metadata.get("magnitude"),
            "emission_lines": metadata.get("emission_lines"),
            "emission_band_tokens": metadata.get("emission_band_tokens"),
            "common_name": str(metadata.get("common_name", "") or "").strip(),
            "image_url": str(metadata.get("image_url", "") or "").strip(),
            "hero_image_url": str(metadata.get("hero_image_url", "") or "").strip(),
            "ra_deg": metadata.get("ra_deg"),
            "dec_deg": metadata.get("dec_deg"),
            "ang_size_maj_arcmin": metadata.get("ang_size_maj_arcmin"),
            "ang_size_min_arcmin": metadata.get("ang_size_min_arcmin"),
            "rise": event_time_value(events.get("rise")),
            "first_visible": event_time_value(events.get("first_visible")),
            "culmination": event_time_value(events.get("culmination")),
            "culmination_alt": culmination_alt,
            "last_visible": event_time_value(events.get("last_visible")),
            "set": event_time_value(events.get("set")),
            "visible_total": "--",
            "visible_remaining": "--",
            "culmination_dir": culmination_dir,
            "is_in_list": is_in_list,
        }

    rows: list[dict[str, Any]] = []
    selected_primary_id = str(selected_id or "").strip()
    if selected_primary_id and isinstance(selected_track, pd.DataFrame):
        selected_row = _build_row(
            selected_primary_id,
            str(selected_label or selected_primary_id),
            str(selected_type_group or "other"),
            str(selected_color or "#22c55e"),
            selected_events or {},
            selected_primary_id in list_member_ids,
            selected_metadata,
        )
        selected_row["visible_total"] = format_duration_hm(compute_total_visible_time(selected_track))
        selected_row["visible_remaining"] = format_duration_hm(
            compute_remaining_visible_time(selected_track, now=now_local)
        )
        rows.append(selected_row)
    for target_track in overlay_tracks:
        primary_id = str(target_track.get("primary_id", ""))
        target_row = _build_row(
            primary_id,
            str(target_track.get("label", "List target")),
            str(target_track.get("object_type_group", "other")),
            str(target_track.get("color", "#22c55e")),
            target_track.get("events", {}),
            primary_id in list_member_ids,
            {
                "common_name": target_track.get("common_name"),
                "object_type": target_track.get("object_type"),
                "magnitude": target_track.get("magnitude"),
                "emission_lines": target_track.get("emission_lines"),
                "emission_band_tokens": target_track.get("emission_band_tokens"),
                "image_url": target_track.get("image_url"),
                "hero_image_url": target_track.get("hero_image_url"),
                "ra_deg": target_track.get("ra_deg"),
                "dec_deg": target_track.get("dec_deg"),
                "ang_size_maj_arcmin": target_track.get("ang_size_maj_arcmin"),
                "ang_size_min_arcmin": target_track.get("ang_size_min_arcmin"),
            },
        )
        overlay_track = target_track.get("track")
        if isinstance(overlay_track, pd.DataFrame):
            target_row["visible_total"] = format_duration_hm(compute_total_visible_time(overlay_track))
            target_row["visible_remaining"] = format_duration_hm(
                compute_remaining_visible_time(overlay_track, now=now_local)
            )
        rows.append(target_row)

    if row_order_ids:
        order_map = {str(primary_id): idx for idx, primary_id in enumerate(row_order_ids)}
        default_rank = len(order_map)
        rows = sorted(rows, key=lambda row: order_map.get(str(row.get("primary_id", "")), default_rank))
    return rows
