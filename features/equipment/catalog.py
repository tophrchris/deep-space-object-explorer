from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

EQUIPMENT_CATALOG_PATH = Path("data/equipment/equipment_catalog.json")


def load_equipment_catalog(catalog_path: str = str(EQUIPMENT_CATALOG_PATH)) -> dict[str, Any]:
    try:
        raw_payload = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    except Exception:
        return {"categories": []}

    categories: list[dict[str, Any]] = []
    raw_categories = raw_payload.get("categories", [])
    if not isinstance(raw_categories, list):
        return {"categories": []}

    for raw_category in raw_categories:
        if not isinstance(raw_category, dict):
            continue
        category_id = str(raw_category.get("id") or "").strip()
        if not category_id:
            continue
        label = str(raw_category.get("label") or category_id).strip() or category_id
        description = str(raw_category.get("description") or "").strip()

        display_columns: list[dict[str, str]] = []
        raw_columns = raw_category.get("display_columns", [])
        if isinstance(raw_columns, list):
            for raw_column in raw_columns:
                if not isinstance(raw_column, dict):
                    continue
                field = str(raw_column.get("field") or "").strip()
                if not field:
                    continue
                column_label = str(raw_column.get("label") or field).strip() or field
                display_columns.append({"field": field, "label": column_label})

        items: list[dict[str, Any]] = []
        raw_items = raw_category.get("items", [])
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                item_id = str(raw_item.get("id") or "").strip()
                item_name = str(raw_item.get("name") or "").strip()
                if not item_id or not item_name:
                    continue
                item_payload = {"id": item_id, "name": item_name}
                for key, value in raw_item.items():
                    if key in {"id", "name"}:
                        continue
                    item_payload[str(key)] = value
                items.append(item_payload)

        categories.append(
            {
                "id": category_id,
                "label": label,
                "description": description,
                "display_columns": display_columns,
                "items": items,
            }
        )

    return {"categories": categories}


def format_equipment_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "; ".join(parts) if parts else "-"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "-"
        return f"{value:.6g}"
    text = str(value).strip()
    return text if text else "-"
