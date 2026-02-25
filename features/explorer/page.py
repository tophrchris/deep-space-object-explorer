from __future__ import annotations

from typing import Any

import pandas as pd
from features.explorer.page_impl import _render_explorer_page_impl


def render_explorer_page(
    *,
    catalog: pd.DataFrame,
    catalog_meta: dict[str, Any],
    prefs: dict[str, Any],
    temperature_unit: str,
    use_12_hour: bool,
    detail_stack_vertical: bool,
    browser_locale: str | None,
    browser_month_day_pattern: str | None,
) -> None:
    _render_explorer_page_impl(
        catalog=catalog,
        catalog_meta=catalog_meta,
        prefs=prefs,
        temperature_unit=temperature_unit,
        use_12_hour=use_12_hour,
        detail_stack_vertical=detail_stack_vertical,
        browser_locale=browser_locale,
        browser_month_day_pattern=browser_month_day_pattern,
    )
