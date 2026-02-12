# Deep Space Object Explorer (Streamlit Prototype)

A responsive Streamlit prototype for exploring deep sky objects across Messier, NGC, IC, and Sharpless catalogs.

## Current prototype coverage

- Unified search across M/NGC/IC/Sh2 IDs and common names
- Desktop split layout and phone-style stacked layout preview
- Location controls (default Princeton, manual geocode, browser geolocation permission flow, IP fallback)
- 16-bin obstruction editor (default 20 deg)
- Favorites and Set List with browser-local persistence (per user/device)
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
- `catalog_ingestion.py`: catalog ingest + normalization + cache metadata
- `data/dso_catalog_seed.csv`: seed normalized catalog for v0 prototype
- `TODO.md`: prioritized build backlog

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

## Notes

- Catalog ingest loads full OpenNGC data and augments it with SIMBAD (full SH2 ingest + M/NGC enrichment).
- If live ingest fails, the app falls back to the local seed dataset.
- Use the sidebar `Refresh catalog cache` button to force a re-ingest.
- Weather and image lookups fail gracefully without breaking plots.
