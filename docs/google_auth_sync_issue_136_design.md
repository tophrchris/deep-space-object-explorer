# Google Auth / Sync Hardening Plan (Issue #136)

## Problem Summary

Issue `#136` reports two classes of failures in Google auth/cloud sync:

1. Periodic sync failures mentioning auth tokens.
2. A fresh browser session can log in and immediately overwrite a user's cloud settings with a local "blank" profile.

The second issue is caused by the current sync bootstrap flow comparing timestamps and allowing a local upload before the app has safely compared local unauthenticated edits with the user's existing cloud profile.

## Goals

- Prevent first-login overwrite of an existing cloud profile.
- Pull cloud settings first, compare to local, and block auto-push until compare completes.
- If the unauthenticated local session contains additions (`sites`, `equipment`, `lists`), prompt the user with a checkbox-based merge UI.
- Add resiliency for expired/invalid access tokens by preserving pending sync intent and prompting re-authentication.
- Keep implementation aligned to the `runtime/` + `features/` + `ui/` split.

## State Machine (v1)

This is implemented primarily in `runtime/google_drive_sync_runtime.py`, with UI surfaces in `features/settings/page.py`.

### States

- `unauthenticated`
  - User not logged in to Google.
  - No Drive API calls.
  - Preserve local settings and pending sync intent.

- `auth_present_no_token`
  - User appears logged in, but no usable access token is available.
  - Sync is blocked.
  - UI prompts user to reconnect (sign out/in).

- `auth_ready_unverified`
  - Logged in with access token, but cloud compare has not been completed yet for the current account/session.
  - This is the required entry point after login or account switch.

- `compare_remote`
  - Fetch and inspect cloud settings file.
  - Validate payload and owner.
  - Do not auto-push yet.

- `merge_decision_required`
  - Remote cloud settings exist and local unauthenticated session contains additions.
  - Present checkbox-based merge UI for `sites`, `equipment`, and `lists`.

- `ready`
  - Compare/bootstrap completed for this session/account.
  - Normal push/pull behavior can resume.

- `reauth_required`
  - Drive call failed due to token/auth/scope issue (e.g., `401`, auth-related `403`).
  - Pending sync intent is preserved.
  - User must reconnect Google auth before sync resumes.

- `error`
  - Non-auth sync failure (network, payload, API error, etc.).
  - State carries an error message; some errors remain retryable.

## Compare-First Bootstrap Rules

On the first authenticated run for a session/account:

1. Attempt to read the cloud settings file.
2. If no cloud settings file exists:
   - seed cloud from local settings (same as today), but only after compare completes.
3. If a cloud settings file exists:
   - treat cloud settings as the base profile,
   - compare local unauthenticated session preferences against cloud,
   - if local has additions in `sites`, `equipment`, or `lists`, require user merge decision,
   - otherwise apply cloud settings locally and complete bootstrap.

Manual `Pull` still behaves as a cloud-wins action.

## Merge Scope (v1)

The login-time merge UI only handles additions in:

- `sites` (new site IDs not present in cloud)
- `equipment` (new owned equipment IDs by category not present in cloud)
- `lists` (new list IDs not present in cloud)

It intentionally does **not** attempt to merge:

- edits to existing items with the same ID
- session snapshot/widget state
- conflict resolution for renamed items or same-name different-ID items

Those remain future enhancements.

## Token Expiry / Reauth Resiliency

The app currently relies on Streamlit auth (`st.login`, `st.user`) and exposed access tokens.
This plan does not add a custom refresh-token flow.

Instead:

- classify Google Drive auth failures (`401`, auth-related `403`)
- mark sync state as `reauth_required`
- preserve `cloud_sync_pending`
- preserve a deferred sync action (`pull`, `push`, or `auto`)
- prompt the user to reconnect Google auth (sign out/in)
- on reconnect, re-enter `auth_ready_unverified` and run compare-first again

## Runtime / Features / UI Responsibilities

- `runtime/google_drive_sync.py`
  - Google Drive request helpers
  - typed API errors and auth-error classification

- `runtime/google_drive_sync_runtime.py`
  - sync orchestration/state machine
  - compare-first bootstrap
  - merge candidate generation and merge application

- `features/settings/page.py`
  - display sync status / reauth prompt
  - checkbox merge UI and user decision actions

- `ui/streamlit_app.py`
  - state key definitions
  - dependency wiring only (no new sync business logic)

## Known Edge Cases (Deferred / Explicitly Out of Scope in v1)

- merging edits to existing items (same ID)
- equal timestamps but differing payload content (fingerprint-based compare)
- multi-tab conflict resolution
- full refresh-token lifecycle management
- session snapshot merge semantics during login-time merge

## Acceptance Criteria (v1)

- Fresh browser + login does not auto-overwrite an existing cloud profile before compare.
- Existing cloud profile is pulled locally on login when no local additions exist.
- Local additions (`sites`, `equipment`, `lists`) trigger merge UI with per-item checkboxes.
- Auth/token failures preserve pending sync state and prompt reauthentication.
- Manual push/pull still work after bootstrap completes (subject to token availability).
