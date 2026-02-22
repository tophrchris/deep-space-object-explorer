from __future__ import annotations

import json
import uuid
from typing import Any

import requests

DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"
DEFAULT_SETTINGS_FILENAME = "dso_explorer_settings.json"


def _auth_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
    }


def _request_json(
    method: str,
    url: str,
    *,
    access_token: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    data: bytes | str | None = None,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    request_headers = _auth_headers(access_token)
    if headers:
        request_headers.update(headers)
    response = requests.request(
        method=method,
        url=url,
        params=params,
        headers=request_headers,
        data=data,
        timeout=timeout_seconds,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Google Drive API error ({response.status_code}): {response.text[:300]}"
        )
    if not response.text:
        return {}
    payload = response.json()
    if isinstance(payload, dict):
        return payload
    return {}


def find_settings_file(
    access_token: str,
    *,
    filename: str = DEFAULT_SETTINGS_FILENAME,
) -> dict[str, Any] | None:
    query = f"name = '{filename}' and 'appDataFolder' in parents and trashed = false"
    payload = _request_json(
        "GET",
        f"{DRIVE_API_BASE}/files",
        access_token=access_token,
        params={
            "q": query,
            "spaces": "appDataFolder",
            "pageSize": 1,
            "fields": "files(id,name,modifiedTime,size)",
            "orderBy": "modifiedTime desc",
        },
    )
    files = payload.get("files", [])
    if isinstance(files, list) and files:
        first = files[0]
        if isinstance(first, dict):
            return first
    return None


def read_settings_payload(access_token: str, file_id: str) -> dict[str, Any] | None:
    resolved_file_id = str(file_id).strip()
    if not resolved_file_id:
        return None
    response = requests.get(
        f"{DRIVE_API_BASE}/files/{resolved_file_id}",
        params={"alt": "media"},
        headers=_auth_headers(access_token),
        timeout=20.0,
    )
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        raise RuntimeError(
            f"Google Drive API error ({response.status_code}): {response.text[:300]}"
        )
    payload = response.json()
    if isinstance(payload, dict):
        return payload
    return None


def _build_multipart_body(
    metadata: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[bytes, str]:
    boundary = f"dso-explorer-{uuid.uuid4().hex}"
    metadata_json = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True)
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    body = (
        f"--{boundary}\r\n"
        "Content-Type: application/json; charset=UTF-8\r\n\r\n"
        f"{metadata_json}\r\n"
        f"--{boundary}\r\n"
        "Content-Type: application/json; charset=UTF-8\r\n\r\n"
        f"{payload_json}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    return body, boundary


def create_settings_file(
    access_token: str,
    payload: dict[str, Any],
    *,
    filename: str = DEFAULT_SETTINGS_FILENAME,
) -> dict[str, Any]:
    metadata = {
        "name": filename,
        "parents": ["appDataFolder"],
        "mimeType": "application/json",
    }
    body, boundary = _build_multipart_body(metadata, payload)
    return _request_json(
        "POST",
        f"{DRIVE_UPLOAD_BASE}/files",
        access_token=access_token,
        params={"uploadType": "multipart", "fields": "id,name,modifiedTime,size"},
        headers={"Content-Type": f"multipart/related; boundary={boundary}"},
        data=body,
    )


def update_settings_file(
    access_token: str,
    file_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    resolved_file_id = str(file_id).strip()
    if not resolved_file_id:
        raise ValueError("file_id is required")
    metadata: dict[str, Any] = {"mimeType": "application/json"}
    body, boundary = _build_multipart_body(metadata, payload)
    return _request_json(
        "PATCH",
        f"{DRIVE_UPLOAD_BASE}/files/{resolved_file_id}",
        access_token=access_token,
        params={"uploadType": "multipart", "fields": "id,name,modifiedTime,size"},
        headers={"Content-Type": f"multipart/related; boundary={boundary}"},
        data=body,
    )


def upsert_settings_file(
    access_token: str,
    payload: dict[str, Any],
    *,
    preferred_file_id: str = "",
    filename: str = DEFAULT_SETTINGS_FILENAME,
) -> dict[str, Any]:
    resolved_file_id = str(preferred_file_id).strip()
    if resolved_file_id:
        try:
            return update_settings_file(
                access_token,
                resolved_file_id,
                payload,
            )
        except Exception:
            # Fall through to discover/create in case the file id was deleted or stale.
            pass

    existing = find_settings_file(access_token, filename=filename)
    if existing and str(existing.get("id", "")).strip():
        return update_settings_file(
            access_token,
            str(existing.get("id", "")).strip(),
            payload,
        )

    return create_settings_file(
        access_token,
        payload,
        filename=filename,
    )
