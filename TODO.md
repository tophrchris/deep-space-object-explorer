# TODO

## Phase 1: Catalog Ingestion (Current focus)
- [x] Define unified catalog schema and normalize seed data.
- [x] Add disk cache (`parquet`) + metadata for ingest runs.
- [x] Add in-app control to refresh catalog ingestion cache.
- [x] Expand seed dataset toward full catalog coverage for M/NGC/IC/SH2.
- [ ] Add source-specific ingest adapters (one module per catalog source).
- [ ] Add ingest validation checks (required fields, RA/Dec ranges, duplicate IDs).
- [x] Add ingest tests and a CLI entry point for batch refresh.

## Phase 2: Location + Sky Semantics
- [x] Manual geocoding (ZIP/place) with fallback to last valid location.
- [x] Browser geolocation permission flow (navigator API via Streamlit component).
- [x] Improve reverse-geocode labels (city/state/country quality checks).
- [ ] Add explicit permission-state messaging and retry flow.
- [ ] Add location source badges (`default`, `manual`, `browser`, `ip`).

## Phase 3: Core Product Loop
- [x] Search + results + target detail wired to live Alt/Az calculations.
- [x] Favorites + Set List persistence.
- [x] Obstruction 16-bin editor applied to visibility calculations.
- [ ] Set List reorder UX improvements and quick actions.
- [ ] Better empty/loading/error states in detail panel.

## Phase 4: Data Enrichment
- [x] Free-use image lookup with attribution fallback.
- [x] Optional hourly weather temperature row in night plot.
- [ ] Harden upstream API error handling and retries.
- [ ] Add cache TTL controls per external source.

## Phase 5: UI/UX Polish (Deferred by request)
- [x] Refine responsive desktop/tablet split behavior.
- [ ] Implement phone-style detail bottom sheet interactions.
- [ ] Improve typography, spacing, and hierarchy for scanability.
- [ ] Add onboarding guidance for first-time users.

## Phase 6: Release Engineering
- [ ] Finalize private GitHub repo settings and branch protections.
- [ ] Add CI checks (lint + type checks + smoke tests).
- [ ] Add deployment target (Streamlit Community Cloud or container).



# longer term planning horizon

## unpriorized backlog
- [ ] add resilience for missing weather data
- [ ] add lunar interference model
- [ ] add the notion of "sweet spot" (visible, unobstructed, not in danger zone of mounts, no lunar interference)
- [ ] disconnect night sky preview from "pinning", introduce more general list concept
- [ ] look at things like https://en.wikipedia.org/wiki/Lists_of_nebulae for catalog enrichment
- [ ] integrate barnard objects, caldwell objects
  
## launch Phase
- [ ] build github pages based website
- [ ] create product usage documentation- wiki?
- [ ] create hero images of app 
- [ ] integrate some sort of uservoice system for bug tracking


## refactoring plans
- [ ] pull weather service out into its own folder
- [ ] break up main UI (search, forecast, details, night sky preview) into files/folders
