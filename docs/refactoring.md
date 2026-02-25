# Refactoring Plan (Follow-Up Work)

## Purpose

This document is a concrete execution plan for the next refactoring passes after the `maintenance-code-cleanup` branch cleanup and the first `ui/features/runtime` split.

Primary goals:

- Reduce fragility (especially circular imports and hidden dependencies).
- Continue shrinking `ui/streamlit_app.py` into a compatibility facade only.
- Make feature code testable without importing the monolithic Streamlit module.
- Preserve app behavior while refactoring incrementally.

## Current State (Checkpoint)

Completed:

- `app.py` is a thin entrypoint.
- Major feature/runtime modules have been split out (`features/*`, `runtime/*`, `ui/app_main.py`).
- Explorer detail/recommendations/plots/forecast modules exist.
- Site state CRUD/sync is extracted to `features/sites/site_state.py`.
- Explorer weather/path helper clusters are extracted.
- Startup reliability issue fixed: prefs bootstrap retries are now non-blocking in `ui/app_main.py`.
- Browser JS probe calls in `ui/app_main.py` are cached in `st.session_state` (one-time per session).
- `_legacy_ui` bridge pattern removed from:
  - `features/explorer/plots.py`
  - `features/explorer/forecast_panels.py`
  (replaced by explicit lazy symbol lookups)

Still true:

- `ui/streamlit_app.py` remains a large compatibility module and still owns many shared helpers/constants.
- Many extracted modules still depend on `ui.streamlit_app` via transitional lazy lookups.
- There is still architectural backflow from `features/runtime -> ui`.

## Refactoring Principles (Do Not Regress)

- Keep behavior stable. Prefer extraction and delegation over redesign.
- Separate concerns by the current rule of thumb:
  - UI/page rendering -> `ui/` or `features/*/page.py`
  - Feature/domain transforms/ranking/formatting -> `features/*`
  - External I/O/services/caches -> `runtime/*`
- Avoid adding new hidden globals or `globals().setdefault(...)` bridges.
- Prefer explicit imports or explicit dependency objects over dynamic symbol injection.
- Keep changes small enough to smoke test after each slice.

## High-Priority Next Work

### 1. Remove Remaining Transitional Bridges (Highest Risk)

Target:

- Eliminate `_legacy_ui` + `_refresh_legacy_globals()` from remaining modules.

Likely remaining modules (check before starting):

- `features/explorer/detail_panel.py`
- `features/explorer/recommendations.py`
- `features/explorer/summary_table.py`
- `features/explorer/summary_rows.py`
- `features/explorer/page_impl.py`
- `features/explorer/night_rating.py`
- `features/sites/settings_sections.py`
- `features/sites/location_actions.py`
- `features/sites/site_state.py`
- `runtime/recommendation_cache.py`
- `runtime/weather_mask_cache.py`
- `runtime/location_resolution.py`
- `runtime/google_drive_sync_runtime.py`
- `runtime/session_snapshot.py`
- `ui/app_main.py`

Strategy:

1. Convert one module at a time.
2. Prefer direct imports from real source modules.
3. If circular imports block direct imports, use explicit lazy lookups (like `_ui_name(...)`) as a temporary step.
4. Do not reintroduce dynamic globals refresh.

Exit criteria:

- No `globals().setdefault(...)` bridge pattern left in app code.
- No module-local `_refresh_legacy_globals()` helpers left.

### 2. Extract Shared Helpers Out of `ui/streamlit_app.py`

Why:

- Bridge removal eventually stalls unless shared logic is moved to neutral modules.
- `ui/streamlit_app.py` currently still acts like a hidden dependency container.

Create explicit shared modules (suggested):

- `features/explorer/shared_formatting.py`
  - `format_display_time`
  - `format_hour_label`
  - `format_weather_forecast_date`
  - `format_emissions_display`
  - `format_description_preview`
  - `format_apparent_size_display`
  - `apparent_size_sort_key_arcmin`
  - `normalize_object_type_group`

- `features/explorer/shared_plotting.py`
  - `_interpolate_cloud_cover_color`
  - `_interpolate_temperature_color_f`
  - `_interpolate_color_stops`
  - `_ideal_text_color_for_hex`
  - `_muted_rgba_from_hex`
  - mount warning zone helpers
  - path/chart constants currently only in `ui/streamlit_app.py`

- `features/explorer/shared_forecast.py`
  - forecast state keys/constants
  - `WEATHER_MATRIX_ROWS`
  - `WEATHER_ALERT_RAIN_PRIORITY`
  - `ASTRONOMY_FORECAST_NIGHTS`
  - `weather_forecast_window`
  - `astronomical_night_window`

Notes:

- Start by moving pure functions/constants only.
- Avoid moving UI rendering code in the same commit as shared helper extraction.

Exit criteria:

- `features/explorer/plots.py` and `features/explorer/forecast_panels.py` no longer need any `ui.streamlit_app` lookups.
- Same eventually for `detail_panel.py`, `recommendations.py`, `page_impl.py`.

### 3. Continue Shrinking `ui/streamlit_app.py` to Compat Facade

Target outcome:

- `ui/streamlit_app.py` should mainly contain:
  - legacy wrappers/delegators
  - top-level constants only if not yet relocated
  - no substantial feature logic

Next extraction candidates (high value, low risk):

- Sites:
  - `build_location_selection_map(...)` -> `features/sites/map_ui.py` (UI helper)
- Explorer/runtime:
  - `fetch_free_use_image(...)` -> `runtime/media_lookup.py`
  - `build_legacy_survey_cutout_urls(...)` -> `features/explorer/survey_links.py`

Then medium-risk clusters:

- Remaining formatting helpers shared across Explorer modules
- Any lingering site action/state helper not yet moved

Exit criteria:

- `ui/streamlit_app.py` contains no major domain algorithms or data transformation logic.
- Line count trends downward without regressions.

## Secondary Work (After Bridge Removal)

### 4. Improve Dependency Boundaries

Introduce explicit dependency objects for page renderers where helpful:

- `ExplorerPageDeps`
- `SitesPageDeps`
- `SettingsPageDeps` (already present pattern)

Use dependency objects to avoid implicit imports and circular references.

### 5. Add Tests Around Extracted Pure Logic

Priority test targets:

- `features/sites/site_state.py`
  - active site switching
  - duplicate/create/delete site behavior
  - legacy field sync/persist behavior

- `features/explorer/plots.py`
  - `split_path_on_az_wrap`
  - endpoint marker helpers
  - `terminal_segment_from_path_arrays`
  - `distribute_non_overlapping_values`

- `features/explorer/forecast_panels.py`
  - weather alert indicator resolution
  - weather matrix formatting/styling value transforms
  - astronomy forecast table row formatting (pure pieces only)

Test style:

- Prefer pure function tests over Streamlit UI snapshot tests.
- Add small fixture dataframes for representative cases.

### 6. Performance/Reload Follow-Up

Keep an eye on:

- Cold start vs warm rerun behavior
- localStorage/bootstrap retry behavior in Safari
- cache churn caused by changing keys/inputs during refactors

Potential follow-ups:

- Add lightweight timing utility (disabled by default) for startup phases
- Audit `@st.cache_data` / `@st.cache_resource` key stability

## Execution Plan (Suggested Sequence)

### Phase A: Bridge Elimination Foundation

1. Inventory all remaining `_legacy_ui` / `_refresh_legacy_globals()` modules.
2. Convert easiest non-Explorer runtime modules first (`runtime/session_snapshot.py`, `runtime/weather_mask_cache.py`).
3. Extract Explorer shared pure helpers/constants into neutral modules.
4. Convert `features/explorer/detail_panel.py`, `recommendations.py`, `page_impl.py` to explicit imports.
5. Convert `ui/app_main.py` bridge last (it still depends on many symbols from `ui.streamlit_app`).

### Phase B: `ui/streamlit_app.py` Decomposition

1. Move remaining obvious helper clusters (map/media/survey links).
2. Move shared formatting helpers to `features/explorer/shared_*`.
3. Replace wrappers/imports to point at new modules.
4. Re-run smoke tests after each extraction slice.

### Phase C: Testing and Hardening

1. Add unit tests for moved pure helpers.
2. Add one import smoke test to catch circular imports early.
3. Add one startup smoke test path (non-interactive) if feasible.

## Safety Checklist (Use Every Refactor Slice)

- `python3 -m py_compile` on touched modules
- Import smoke:
  - `import app`
  - `import ui.streamlit_app`
  - import touched feature/runtime modules
- Run app manually and verify:
  - app loads
  - Explorer page renders
  - one chart renders
  - weather matrix renders
  - site switching works
- If a bridge change touches Explorer rendering, test both:
  - initial cold start
  - one warm reload

## Commit Strategy

Use small, reviewable commits:

- `Extract shared explorer formatting helpers`
- `Remove bridge from explorer detail panel`
- `Move site map helper to features.sites`
- `Add tests for plot geometry helpers`

Avoid combining:

- behavior changes + mass file moves + performance tweaks

## Definition of Done (Long-Term)

- `ui/streamlit_app.py` is a thin compatibility facade or retired entirely.
- Feature modules do not depend on `ui.streamlit_app`.
- Runtime modules do not depend on UI modules.
- Shared pure helpers live in explicit modules with tests.
- App startup/rerun is stable and debuggable without temporary instrumentation.
