# DSO Explorer — Web Prototype PRD (v1)

> This document supersedes `dso_explorer_web_prd_v0.md` and reflects the implemented product state after merged changes through **February 12, 2026**.

- Updated: 2026-02-12
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
- `constellation` (when available)

- Catalog loading behavior:
- Full OpenNGC ingestion for M/NGC/IC.
- SH2 rows merged from local seed data.
- Disk cache + metadata with migration/version handling.
- Fallback to local seed data if live ingest fails.
- In-app `Refresh catalog cache` control.

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

- Desktop split: Targets pane 35%, Detail pane 65%.
- Phone preview mode: stacked layout with detail bottom-sheet style container.
- Targets pane includes:
- In-pane search input.
- Tabs with live counts in labels: Results, Favorites, Set List.
- Row-click selection behavior for all three tabs.
- Set List reorder/remove actions in main pane.

### 3.6 Detail panel

- Header with ID/name + Favorite and Set List toggles.
- Free-use image lookup via Wikimedia (with attribution/license when available).
- Object attributes presented as key/value property table.
- Tonight event summary line (rise, first-visible, culmination, last-visible).

### 3.7 Plot system

- Sky Position plot:
- Style control uses segmented selector (`Line` | `Radial`).
- `Dome View` toggle for radial axis direction:
- `true`: center = 90 deg altitude.
- `false`: center = 0 deg altitude.
- Style and dome preferences persist across target switches.
- Legend positioned above chart.

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

- User can search across M/NGC/IC/SH2 and select a target by row click.
- Results/Favorites/Set List appear as tabs in the Targets pane with counts.
- Detail panel shows required metadata, list actions, image attribution behavior, and two always-visible plots.
- Sky plot supports Line/Radial and radial Dome View axis inversion.
- Hourly plot renders stacked obstructed/clear bars with direction and temperature labeling.
- Alt/Az refreshes at least every 60 seconds.
- Browser geolocation failure cases do not replace prior valid location.
- App runs locally and is deployable to Streamlit Community Cloud using `main` + `app.py`.

## 6. Change log from v0 to v1 (implemented)

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
