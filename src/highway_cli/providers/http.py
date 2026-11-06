from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict


class ProviderHTTPError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _safe_http_detail(raw: str) -> str:
    """Keep useful provider diagnostics without echoing account metadata."""

    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw[:1000]
    error = value.get("error", value) if isinstance(value, dict) else value
    if isinstance(error, dict):
        safe = {
            key: error[key]
            for key in ("message", "code", "type", "provider_name")
            if key in error
        }
        return json.dumps(safe or {"message": "provider request failed"}, ensure_ascii=False)
    return str(error)[:1000]


def post_json(url: str, api_key: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderHTTPError(
            f"HTTP {exc.code}: {_safe_http_detail(detail)}", status_code=exc.code
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderHTTPError(str(exc.reason)) from exc

