from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app_preferences import save_preferences
from features.equipment.catalog import EQUIPMENT_CATALOG_PATH, format_equipment_value, load_equipment_catalog


def render_equipment_page(prefs: dict[str, Any]) -> None:
    st.title("Equipment")
    st.caption("Store your telescopes/accessories/filters. Recommendation integration will come in a later update.")

    persistence_notice = st.session_state.get("prefs_persistence_notice", "")
    if persistence_notice:
        st.warning(persistence_notice)

    catalog = load_equipment_catalog()
    categories = catalog.get("categories", [])
    if not isinstance(categories, list) or not categories:
        st.warning(f"No equipment catalog entries found at `{EQUIPMENT_CATALOG_PATH}`.")
        return

    current_equipment_raw = prefs.get("equipment", {})
    if not isinstance(current_equipment_raw, dict):
        current_equipment_raw = {}

    def _normalize_selected_ids(raw_values: Any, allowed_ids: set[str]) -> list[str]:
        normalized: list[str] = []
        if not isinstance(raw_values, (list, tuple, set)):
            return normalized
        for raw_value in raw_values:
            item_id = str(raw_value).strip()
            if item_id and item_id in allowed_ids and item_id not in normalized:
                normalized.append(item_id)
        return normalized

    next_equipment: dict[str, list[str]] = {}
    current_equipment_known: dict[str, list[str]] = {}

    with st.container(border=True):
        st.subheader("Owned Equipment")
        st.caption("Use the Owned column to select one or more items per table. Changes are saved automatically.")

        for category in categories:
            category_id = str(category.get("id", "")).strip()
            if not category_id:
                continue
            label = str(category.get("label", category_id)).strip() or category_id
            description = str(category.get("description", "")).strip()
            items = category.get("items", [])
            if not isinstance(items, list) or not items:
                continue

            item_by_id: dict[str, dict[str, Any]] = {}
            ordered_item_ids: list[str] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id", "")).strip()
                item_name = str(item.get("name", "")).strip()
                if not item_id or not item_name:
                    continue
                item_by_id[item_id] = item
                ordered_item_ids.append(item_id)
            if not ordered_item_ids:
                continue

            allowed_ids = set(ordered_item_ids)
            existing_selected = _normalize_selected_ids(current_equipment_raw.get(category_id, []), allowed_ids)
            current_equipment_known[category_id] = existing_selected

            st.markdown(f"#### {label}")
            if description:
                st.caption(description)

            display_columns = category.get("display_columns", [])
            parsed_columns = [
                column
                for column in display_columns
                if isinstance(column, dict) and str(column.get("field", "")).strip()
            ]
            existing_set = set(existing_selected)
            table_rows: list[dict[str, Any]] = []
            for item_id in ordered_item_ids:
                item = item_by_id[item_id]
                row: dict[str, Any] = {
                    "Owned": item_id in existing_set,
                    "Name": str(item.get("name", item_id)).strip() or item_id,
                }
                for column in parsed_columns:
                    field = str(column.get("field", "")).strip()
                    column_label = str(column.get("label", field)).strip() or field
                    row[column_label] = format_equipment_value(item.get(field))
                table_rows.append(row)

            table_frame = pd.DataFrame(table_rows)
            table_height = max(72, min(360, 36 * (len(table_frame) + 1)))
            editable_cols = {"Owned"}
            disabled_cols = [column for column in table_frame.columns if column not in editable_cols]

            editor_kwargs: dict[str, Any] = {
                "hide_index": True,
                "use_container_width": False,
                "disabled": disabled_cols,
                "num_rows": "fixed",
                "height": table_height,
                "key": f"equipment_table_{category_id}",
            }
            if hasattr(st, "column_config") and hasattr(st.column_config, "CheckboxColumn"):
                editor_kwargs["column_config"] = {
                    "Owned": st.column_config.CheckboxColumn("Owned", width="small"),
                }

            edited_frame = st.data_editor(table_frame, **editor_kwargs)
            owned_values = (
                edited_frame["Owned"].tolist()
                if isinstance(edited_frame, pd.DataFrame) and "Owned" in edited_frame.columns
                else [False] * len(ordered_item_ids)
            )
            next_equipment[category_id] = [
                ordered_item_ids[idx]
                for idx, owned in enumerate(owned_values[: len(ordered_item_ids)])
                if bool(owned)
            ]

    if next_equipment != current_equipment_known:
        prefs["equipment"] = next_equipment
        st.session_state["prefs"] = prefs
        save_preferences(prefs)
