# DSO Explorer — Web Prototype PRD (v1.2)

> This document supersedes `dso_explorer_web_prd_v0.md` and reflects the implemented product state through **February 13, 2026**.

- Updated: 2026-02-13
- Platforms: phone + tablet + laptop (responsive)
- Runtime target: Streamlit Community Cloud (private GitHub repo supported)

## 1. Product goals

- Unified discovery across Messier, NGC, IC, and Sharpless (Sh2).
- Fast loop: Search -> Select target -> Review detail + plots -> Save to Favorites / Set List.
- Reliable real-time sky context using current location and 60s Alt/Az refresh.
- Local/session persistence of user preferences (favorites, set list, location, obstructions, display choices).

## 2. Non-goals

- No user accounts or cloud profile sync.
- No collaborative list sharing.
- No native-mobile-only feature set parity.

## 3. Implemented scope (current)

### 3.1 Catalog ingestion and schema

- Unified schema in use:
- `primary_id`
- `catalog` (`M`, `NGC`, `IC`, `SH2`)
- `common_name`
- `object_type`
- `ra_deg`, `dec_deg`
- `constellation`
- `aliases`
- `image_url`, `image_attribution_url`, `license_label`
- `description`
- `info_url`
- `dist_value`, `dist_unit`
- `redshift`
- `morphology`
- `emission_lines`

- Catalog loading behavior:
- App loader is mediated through `catalog_service.load_catalog_data(...)` with feature modes:
- `legacy` (active default)
- `curated_parquet` (available fallback mode)
- Legacy path now passes `data/DSO_CATALOG_ENRICHED.CSV` into ingestion.
- Enriched rows load first and remain authoritative by `primary_id`.
- Parquet cache rows are appended only for IDs not present in enriched data.
- Legacy ingest route (OpenNGC + SIMBAD + seed fallback) remains available for rebuild/fallback.
- Disk cache + metadata include ingestion-version migration behavior.
- In-app `Refresh catalog cache` control remains available on the Settings page.
- Search index includes `description` in addition to ID/name/aliases/catalog tokens.

### 3.2 Location handling

- First run defaults to Princeton, NJ.
- Location controls include:
- Manual ZIP/place resolve.
- Browser geolocation via `streamlit-js-eval`.
- IP fallback location.

- Browser geolocation behavior:
- Handles success + explicit error states (permission denied, unavailable, timeout, parse/read failure).
- Keeps prior valid location on failure.

### 3.3 Alt/Az semantics and timing

- Alt/Az values in tables and detail fields represent "now" at active location.
- App auto-refreshes every 60 seconds.
- Night window computed as sunset (today) -> sunrise (tomorrow), with safe fallback window if astronomy calc fails.

### 3.4 Obstructions model

- 16-bin direction model (`N ... NNW`) with user-editable minimum altitude per direction.
- Obstructions feed:
- `visible` state in track calculations.
- first-visible / last-visible event extraction.
- visual shading in path plots.

### 3.5 Main UI structure

- App is now multi-page:
- `Explorer` page
- `Settings` page
- Navigation uses `st.navigation`/`st.Page` when available, with sidebar-radio fallback when unavailable.
- Settings page sections are implemented and grouped as:
- Location
- Display
- Catalog
- Obstructions
- Settings Backup / Restore
- Explorer page keeps search-first target selection flow plus detail/plot review.

### 3.6 Detail panel

- Header with ID/name + Favorite and Set List toggles.
- Three-column detail layout:
- Left: image
- Middle: description + links
- Right: property/value table
- Image rendering behavior:
- Prefer catalog `image_url` (direct image URL) first.
- Fallback to Wikimedia lookup only when catalog image URL is missing.
- Render at max 400x400 while preserving aspect ratio (scale-to-fit, no distortion).
- Middle-column links include:
- image source link
- background/info link
- Object attributes presented as key/value property table.
- Property rows with blank/`-` values are suppressed.
- Property table includes enrichment fields now surfaced in UI:
- `dist_value`, `dist_unit`
- `redshift`
- `morphology`
- `emission_lines` (displayed as emissions details)
- Tonight event summary line (rise, first-visible, culmination, last-visible).

### 3.7 Plot system

- Sky Position plot:
- Style control uses segmented selector (`Line` | `Radial`).
- `Dome View` toggle for radial axis direction:
- `true`: center = 90 deg altitude.
- `false`: center = 0 deg altitude.
- Style and dome preferences persist across target switches.
- Selected target always renders as full-night path.
- Set List targets render as additional full-night paths (same sunset->sunrise window).
- All plotted target paths are solid lines and color-coded by target.
- Event labels are rendered per plotted target:
- `Rise`, `Set`, `First Visible`, `Last Visible`, `Culmination`.
- Direction arrows are rendered along each plotted path to indicate motion over time.
- Obstruction region renders as a hard stepped 16-bin floor (no smoothing), with light-gray fill.
- Sky Position legend is disabled; path identity is provided in the summary table below the plot.
- Sky Position Summary table (under chart):
- Implemented as sortable Streamlit dataframe.
- Columns:
- `Line` (color swatch)
- `Target`
- `Rise`, `First Visible`, `Culmination`, `Last Visible`, `Set`
- `Visible` (total visible duration across tonight window)
- `Culm Dir` (compass direction at culmination)
- `Set List` (Pinned/Unpinned state)
- Multi-row selection supports bulk Set List toggling via `Toggle Selected` action.

- Hourly Forecast plot:
- Stacked hourly bars where total height is hourly max altitude.
- Two segments: `Obstructed` + `Clear`.
- 12-hour hour labels.
- Direction + temperature annotation line.
- Legend positioned above chart.

### 3.8 Temperature handling

- Hourly temperature fetch from free weather source (Open-Meteo).
- Unit preference supports:
- `Auto (browser locale)`
- `Fahrenheit`
- `Celsius`

- Failure is non-blocking; plot still renders.

### 3.9 Persistence model

- Preferences persisted locally when filesystem is writable:
- Favorites
- Set List order
- Last valid location
- 16-bin obstructions
- Temperature preference

- Cloud-safe fallback:
- If preference file write fails, app continues with session-only state and shows non-blocking warning.

### 3.10 Deployment readiness

- `requirements.txt` maintained for Streamlit Cloud dependency install.
- `runtime.txt` pins Python `3.11`.
- `.streamlit/config.toml` included.
- README includes Streamlit Community Cloud deployment steps.

## 4. Known gaps / open issues

- Manual ZIP/place resolution reliability still needs hardening in some cases.
- IP fallback quality/reliability needs further validation.
- These are tracked as open bug issues and are intentionally not marked complete in this PRD.

## 5. Acceptance criteria (v1)

- User can search across M/NGC/IC/SH2 and select a target from search suggestions.
- App provides dedicated `Explorer` and `Settings` pages with stable navigation.
- Settings page contains clearly defined sections for location, display, catalog, obstructions, and settings backup/restore.
- Catalog load path supports enriched-first ingest with cache-only ID supplementation, preserving enriched values for overlapping IDs.
- Detail panel shows required metadata, list actions, image/source/background attribution behavior, and two always-visible plots.
- Detail panel uses the implemented 3-column layout with image scaling at max 400x400 and preserved aspect ratio.
- Detail property table hides blank rows and surfaces enriched fields where available.
- Sky plot supports Line/Radial and radial Dome View axis inversion.
- Sky plot includes full-night paths for selected + Set List targets, per-target event labels, and directional movement arrows.
- Sky obstruction rendering uses hard stepped per-direction floors in light gray.
- Sky Position summary appears under the chart as a sortable table with event times, total visible time, culmination direction, and bulk Set List toggling.
- Hourly plot renders stacked obstructed/clear bars with direction and temperature labeling.
- Alt/Az refreshes at least every 60 seconds.
- Browser geolocation failure cases do not replace prior valid location.
- App runs locally and is deployable to Streamlit Community Cloud using `main` + `app.py`.

## 6. Change log (v0 -> current v1.2)

- Full catalog ingestion via OpenNGC with cache migration and SH2 merge (PR #13).
- Result row click opens detail directly (PR #14).
- Empty-search handling made safe (PR #15).
- Favorites and Set List moved into main-pane tabs with count labels (PR #16).
- Browser geolocation flow made reliable with explicit error handling (PR #18).
- Plot rendering bug fixes:
- obstruction shading
- azimuth wrap artifact removal
- hourly 12-hour labels (PR #20)

- Plot + weather enhancements:
- line/radial style selector
- plot titles
- temperature unit preference with auto locale mode (PR #21)

- Hourly forecast converted to stacked obstructed/clear bars (PR #24).
- Radial Dome View axis toggle added (PR #24).
- Sky style control changed from radio to segmented control (PR #24).
- UI/UX adjustments:
- persistent plot-style preferences across target changes
- legends moved above charts
- search moved into Targets pane
- desktop 35/65 layout
- detail key/value property table (PR #26)

- Streamlit Community Cloud deployment prep:
- runtime pin
- cloud-safe preference persistence fallback
- README deploy instructions (commit `69dd9cc`)

- Sky Position multi-target enhancement set (issue #32 branch):
- Set List entries plotted as full-night path overlays (not single-point markers).
- Paths standardized to solid lines, with per-target event labels and motion-direction arrows.
- Sky Position legend removed.
- Obstruction region changed to hard stepped floors with light-gray styling.
- Added sortable Sky Position summary dataframe with:
- color swatch line indicator
- rise/first visible/culmination/last visible/set times
- total visible duration
- culmination direction
- Set List pinned state
- Added bulk Set List toggling from summary table via multi-row selection.
- Multi-page app refactor with dedicated Settings page sections (issue #42, PR #43).
- Catalog service layer introduced for loader-mode routing (`legacy` / `curated_parquet`) and centralized search/index behavior.
- Enrichment pipeline integrated via standalone `dso_enricher/` project and app-side enriched catalog ingestion.
- App catalog source now includes `data/DSO_CATALOG_ENRICHED.CSV` with enriched schema fields wired through ingestion and UI.
- Enriched-first + parquet supplementation behavior added to preserve enriched rows while recovering cache-only targets (PR #45).
- Search indexing expanded to include `description`.
- Detail panel updated to 3-column layout with:
- scaled direct image rendering (max 400x400, aspect ratio preserved)
- description + source/background links column
- enriched property/value table
- Detail property table row rendering adjusted to suppress blank visual rows (commit `d991f41`).
