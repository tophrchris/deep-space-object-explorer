# Deep Space Object Explorer (Streamlit Prototype)

A responsive Streamlit prototype for exploring deep sky objects across Messier, NGC, IC, and Sharpless catalogs.

## Current prototype coverage

- Unified search across M/NGC/IC/Sh2 IDs and common names
- Desktop split layout and phone-style stacked layout preview
- Sites page with multi-site management (active site, duplicate, delete, edit)
- Location controls per site (search, browser geolocation, map right-click selection, IP fallback)
- Obstruction controls per site (N/E/S/W, WIND16 sliders, or `.hrz` upload reduced to WIND16)
- Generic Lists page (including `Auto (Recent)` + editable custom lists) with browser-local persistence
- Equipment page (store/display selections) driven by runtime JSON at `data/equipment/equipment_catalog.json`
- Settings export/import via JSON for backup or migration to another machine
- Runtime catalog loading from shipped `data/dso_catalog_cache.parquet`
- Target detail panel with:
  - Object metadata
  - Current Alt/Az + 16-wind direction
  - Free-use image lookup (Wikimedia Commons, if available)
  - Path Plot (Alt vs Az, Line/Radial style, radial Dome View toggle)
  - Night Plot (hourly stacked max altitude: obstructed + clear, direction + optional temperature)
- Alt/Az auto-refresh every 60 seconds

## Project structure

- `app.py`: Streamlit app entry point
- `catalog_runtime/catalog_service.py`: runtime catalog load/search helpers
- `lists/`: list management/search/ui modules
- `data/dso_catalog_cache.parquet`: shipped runtime catalog cache
- `data/equipment/equipment_catalog.json`: runtime-editable equipment catalog definitions
- `docs/`: project notes, backlog, and product planning docs
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

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (private is fine).
2. In Streamlit Community Cloud, click `Create app` and select this repo/branch.
3. Set `Main file path` to `app.py`.
4. Deploy.

Notes:
- The app uses `requirements.txt` automatically for Python dependencies.
- `runtime.txt` pins Python to 3.11 for cloud parity.
- Preferences persist in browser `localStorage` by default; optional Google Drive sync can mirror prefs/session state.
- For private repos, make sure Streamlit has GitHub access to this repository.

Google sign-in + Drive appData sync setup:
- Configure OIDC in `.streamlit/secrets.toml` with Google as provider.
- Include `expose_tokens = ["access"]` and Drive appData scope.
- Minimal example:

```toml
[auth]
redirect_uri = "https://<your-app-url>/oauth2callback"
cookie_secret = "<random-secret>"
expose_tokens = ["access"]

[auth.google]
client_id = "<google-client-id>"
client_secret = "<google-client-secret>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
client_kwargs = { scope = "openid profile email https://www.googleapis.com/auth/drive.appdata", prompt = "select_account" }
```

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
- seestarS 30 has 4.3x2.4
## Notes

- The app expects a shipped catalog parquet at `data/dso_catalog_cache.parquet`.
- Weather and image lookups fail gracefully without breaking plots.
