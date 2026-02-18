# Deep Space Object Explorer (Streamlit Prototype)

A responsive Streamlit prototype for exploring deep sky objects across Messier, NGC, IC, and Sharpless catalogs.

## Current prototype coverage

- Unified search across M/NGC/IC/Sh2 IDs and common names
- Desktop split layout and phone-style stacked layout preview
- Location controls (default Princeton, manual geocode, browser geolocation permission flow, IP fallback)
- 16-bin obstruction editor (default 20 deg)
- Generic Lists (including `Auto (Recent)` + editable custom lists) with browser-local persistence
- Settings export/import via JSON for backup or migration to another machine
- Catalog ingestion module with normalized schema + disk cache + metadata
- Target detail panel with:
  - Object metadata
  - Current Alt/Az + 16-wind direction
  - Free-use image lookup (Wikimedia Commons, if available)
  - Path Plot (Alt vs Az, Line/Radial style, radial Dome View toggle)
  - Night Plot (hourly stacked max altitude: obstructed + clear, direction + optional temperature)
- Alt/Az auto-refresh every 60 seconds

## Project structure

- `app.py`: Streamlit app entry point
- `dso_enricher/catalog_ingestion.py`: catalog ingest + normalization + cache metadata
- `lists/`: list management/search/ui modules
- `data/dso_catalog_seed.csv`: seed normalized catalog for v0 prototype
- `specs/`: technical specs for catalog cache and Wikipedia enrichment scripts
- `docs/`: project notes, backlog, and product planning docs
- `docs/catalog_issue_backlog.md`: draft catalog issues to open later on GitHub
- `TODO.md`: prioritized build backlog

## Specifications

- [`specs/DSO_CATALOG_CACHE_SPEC.md`](specs/DSO_CATALOG_CACHE_SPEC.md): canonical schema and constraints for `dso_catalog_cache.parquet`
- [`specs/WIKIPEDIA_CATALOG_ENRICHMENT_SPEC.md`](specs/WIKIPEDIA_CATALOG_ENRICHMENT_SPEC.md): producer spec for `scripts/build_wikipedia_catalog_enrichment.py`
- [`specs/APPLY_WIKIPEDIA_CATALOG_ENRICHMENT_SPEC.md`](specs/APPLY_WIKIPEDIA_CATALOG_ENRICHMENT_SPEC.md): merge/apply spec for `scripts/apply_wikipedia_catalog_enrichment.py`

## Local setup

1. Create a virtual environment with Python 3.11:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Rebuild catalog cache from seed data (optional):

```bash
python scripts/ingest_catalog.py
```

5. Clean local generated artifacts (optional):

```bash
./scripts/clean_local_artifacts.sh
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (private is fine).
2. In Streamlit Community Cloud, click `Create app` and select this repo/branch.
3. Set `Main file path` to `app.py`.
4. Deploy.

Notes:
- The app uses `requirements.txt` automatically for Python dependencies.
- `runtime.txt` pins Python to 3.11 for cloud parity.
- Preferences persist in browser `localStorage` (not shared across users).
- For private repos, make sure Streamlit has GitHub access to this repository.

## Release milestones

- `v0.1.0` (Pre-release): complete
  - Tag: `v0.1.0`
  - Release: <https://github.com/tophrchris/deep-space-object-explorer/releases/tag/v0.1.0>
  - Milestone: <https://github.com/tophrchris/deep-space-object-explorer/milestone/1>
- `v0.2.0`: in progress
  - Milestone: <https://github.com/tophrchris/deep-space-object-explorer/milestone/2>

## v0.2.0 roadmap

- Layout refactor and interaction flow cleanup (`#27`)
- Location UX/robustness improvements (`#28`, `#17`, `#2`)
- Additional catalog quality and enrichment follow-ups
- Catalog issue drafts to file as GitHub issues: [`docs/catalog_issue_backlog.md`](docs/catalog_issue_backlog.md)
- Next-pass UI/UX polish after core behavior stabilizes
- main planning workflow ideas:
  - start with "Tonight at a glance"
    - key conditions (visibility, weather, etc) hour by hour
  - find a target (either recommended or search), see target details and hourly forecast for the target
  - add the target to the candidate list for a given hour
  - all targets show up on the Sky Position plot
  - use the "tonights plan" plot to make hour by hour decisions
- improved settings experience
- more instructional content
- "about this app" page with contact info

- expand list planning workflows (templates, quick actions, nicknames)
- ability to give a nickname to an object

- for each wind4, for each hour, find the best target of each type (emission? galaxy? cluster?) in terms of number of minutes in the sweet zone (unobstructed, not too high/low based on mount)

- object type based colors for paths

## Notes

- Catalog ingest loads full OpenNGC data and augments it with SIMBAD (`NAME` objects are retained, with unmatched rows kept as `SIMBAD` catalog entries).
- If live ingest fails, the app falls back to the local seed dataset.
- Catalog cache is reused by default and refreshed only on demand (sidebar `Refresh catalog cache` or `python scripts/ingest_catalog.py`).
- Weather and image lookups fail gracefully without breaking plots.
