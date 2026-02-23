# TODO

## Phase 1: Catalog Ingestion (Current focus)
- [x] Define unified catalog schema and normalize seed data.
- [x] Add disk cache (`parquet`) + metadata for ingest runs.
- [x] Add in-app control to refresh catalog ingestion cache.
- [x] Expand seed dataset toward full catalog coverage for M/NGC/IC/SH2.
- [ ] Add source-specific ingest adapters (one module per catalog source).
- [x] Add ingest validation checks (required fields, RA/Dec ranges, duplicate IDs).
- [x] Add ingest tests and a CLI entry point for batch refresh.

## Phase 2: Location + Sky Semantics
- [x] Manual geocoding (ZIP/place) with fallback to last valid location.
- [x] Browser geolocation permission flow (navigator API via Streamlit component).
- [x] Improve reverse-geocode labels (city/state/country quality checks).
- [ ] Add explicit permission-state messaging and retry flow.
- [x] Add location source badges (`default`, `manual`, `browser`, `ip`).
- [x] Add multi-site data model support (no UI yet): each site should store `name`, `location`, and `obstructions`, plus an active site pointer.

## Phase 3: Core Product Loop
- [x] Search + results + target detail wired to live Alt/Az calculations.
- [x] Generic list persistence (including auto recent list).
- [x] Obstruction 16-bin editor applied to visibility calculations.
- [ ] List reorder UX improvements and quick actions.
- [x] Better empty/loading/error states in detail panel.

## Phase 4: Data Enrichment
- [x] Free-use image lookup with attribution fallback.
- [x] Optional hourly weather temperature row in night plot.
- [ ] Harden upstream API error handling and retries.
- [ ] Add cache TTL controls per external source.

## Phase 5: UI/UX Polish (Deferred by request)
- [x] Refine responsive desktop/tablet split behavior.
- [x] Add VS Code-inspired Aura Dracula theme preset (base variant).
- [x] Add VS Code-inspired Blue Light theme preset.
- [x] Add VS Code-inspired Monokai ST3 theme preset.
- [ ] Implement phone-style detail bottom sheet interactions.
- [ ] Improve typography, spacing, and hierarchy for scanability.
- [x] Render 5-night forecast with custom HTML table styling while preserving in-place row-click selection (no URL navigation).
- [ ] Add onboarding guidance for first-time users.

## Phase 6: Release Engineering
- [x] Finalize private GitHub repo settings and branch protections.
- [ ] Add CI checks (lint + type checks + smoke tests).
- [ ] Add deployment target (Streamlit Community Cloud or container).



# longer term planning horizon

## unpriorized backlog
- [x] add the notion of "sweet spot" (visible, unobstructed, not in danger zone of mounts, no lunar interference)
- [x] disconnect night sky preview from "pinning", introduce more general list concept
- [x] improved experience for entering obstructions
- [x] start reshaping "explorer" Page into "tonights plan" Page
- [x] target detail as modal? https://github.com/teamtv/streamlit_modal
- [ ] add lunar interference model
- [ ] month planning view?
- [ ] seasonal/annual planning concepts
- [ ] add monthly visibility chart to targets?
- [ ] list management as a first class app experience- show objects in list similar to how the wikipedia enrichment ui shows them


- [ ] add resilience for missing weather data
- [ ] look at things like https://en.wikipedia.org/wiki/Lists_of_nebulae for catalog enrichment
- [ ] integrate barnard objects, caldwell objects

## launch Phase
- [ ] build github pages based website
- [ ] create product usage documentation- wiki?
- [ ] create hero images of app 
- [ ] integrate some sort of uservoice system for bug tracking
- [ ] survey to gauge interest, locations, equipment 

## refactoring plans
- [ ] pull weather service out into its own folder
- [ ] break up main UI (search, forecast, details, night sky preview) into files/folders



