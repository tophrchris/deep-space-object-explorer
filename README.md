# Deep Space Object Explorer (Streamlit Prototype)

A responsive Streamlit prototype for exploring deep sky objects across Messier, NGC, IC, and Sharpless catalogs.

## Current prototype coverage

- Unified search across M/NGC/IC/Sh2 IDs and common names
- Desktop split layout and phone-style stacked layout preview
- Location controls (default Princeton, manual geocode, IP-based approximation)
- 16-bin obstruction editor (default 20 deg)
- Favorites and Set List with local persistence
- Target detail panel with:
  - Object metadata
  - Current Alt/Az + 16-wind direction
  - Free-use image lookup (Wikimedia Commons, if available)
  - Path Plot (Alt vs Az)
  - Night Plot (hourly max altitude + direction + optional temperature)
- Alt/Az auto-refresh every 60 seconds

## Project structure

- `app.py`: Streamlit app entry point
- `data/dso_catalog_seed.csv`: seed normalized catalog for v0 prototype
- `.state/preferences.json`: local persistence file created at runtime

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

## Notes

- This seed version includes a curated subset of catalog entries to prove the full app loop.
- Catalog loading is local right now; next step is replacing the seed dataset with full live catalog fetch + disk cache.
- Weather and image lookups fail gracefully without breaking plots.
