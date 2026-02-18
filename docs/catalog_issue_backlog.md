# Catalog Issue Backlog

Prepared: 2026-02-13

Use these drafts to create GitHub issues later.

## Catalog: improve SIMBAD ID canonicalization coverage

- Labels: area:catalog

## Goal
Expand identifier parsing so SIMBAD objects map reliably across common DSO naming schemes beyond current coverage.

## Scope
- Extend canonicalization for additional catalog patterns (for example IC variants, Barnard-style forms, and other common SIMBAD name variants).
- Preserve deterministic primary-ID selection when multiple identifiers exist.
- Keep backward compatibility for current M/NGC/SH2 mappings.

## Acceptance Criteria
- New patterns are parsed into stable primary IDs.
- Existing mappings do not regress.
- Ingest metadata includes counts of rows mapped by identifier family.

---

## Catalog: implement field-level source-priority merge rules

- Labels: area:catalog

## Goal
Move from row-level merge preference to field-level source priority so the best value is kept per field.

## Scope
- Define source precedence by field (example: coordinates, object type, aliases, common name).
- Merge values per field instead of replacing full rows.
- Document merge policy in code + README.

## Acceptance Criteria
- Merge logic is explicit and deterministic.
- Existing data is preserved unless a higher-priority source provides a better value.
- Metadata reports which fields were updated during enrichment.

---

## Catalog: add ingest validation report

- Labels: area:catalog

## Goal
Add an ingest validation report to detect data quality issues early.

## Scope
- Validate required fields and RA/Dec ranges.
- Flag unknown/empty object types.
- Detect duplicate IDs and alias collisions.
- Emit summary counts + sample rows for each issue class.

## Acceptance Criteria
- Validation runs automatically after ingest.
- Validation results are persisted in metadata/artifact output.
- Ingest continues with warnings unless explicitly configured to fail.

---

## Catalog: generate catalog QA summary artifact after ingest

- Labels: area:catalog

## Goal
Produce a reproducible catalog QA summary artifact after each ingest run.

## Scope
- Write a machine-readable summary (JSON) with row counts, source coverage, and validation stats.
- Include deltas vs previous ingest (new/updated/dropped rows where feasible).
- Keep artifact path stable and documented.

## Acceptance Criteria
- Artifact is generated on every force-ingest and normal refresh that rebuilds cache.
- Artifact includes timestamp, ingestion version, and source breakdown.
- Artifact can be used for quick regression checks.

---

## Catalog: add parser/mapping tests for SIMBAD edge cases

- Labels: area:catalog

## Goal
Add automated tests for parser and mapping edge cases using real-world SIMBAD-style identifiers.

## Scope
- Unit tests for canonicalization and primary-ID selection.
- Fixture-based tests for tricky identifier formats and spacing/punctuation variants.
- Regression tests for previously observed failures.

## Acceptance Criteria
- Tests run in CI/local test workflow.
- Core parser/enrichment functions have edge-case coverage.
- Recent SIMBAD-related regressions are codified as tests.

---

