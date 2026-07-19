from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "[redacted]"
_SENSITIVE_KEYS = frozenset(
    {
        "api-key",
        "api_key",
        "authorization",
        "cookie",
        "credentials",
        "discord_token",
        "id_token",
        "oauth_token",
        "password",
        "proxy_password",
        "refresh_token",
        "secret",
        "set-cookie",
        "token",
        "user_token",
        "x-api-key",
    }
)


def redact_sensitive(value: Any) -> Any:
    """Recursively redact credential-shaped fields without altering ordinary model content."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.casefold().replace(" ", "_")
            if normalized in _SENSITIVE_KEYS or normalized.endswith(("_password", "_secret")):
                result[key] = REDACTED
            elif normalized.endswith("_token") and normalized not in {"input_token", "output_token"}:
                result[key] = REDACTED
            else:
                result[key] = redact_sensitive(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_sensitive(item) for item in value]
    return value
