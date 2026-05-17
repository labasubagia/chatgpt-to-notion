"""Logging helpers."""

from __future__ import annotations

import logging
from urllib.parse import urlsplit, urlunsplit

SENSITIVE_HEADERS = {"authorization", "cookie", "x-api-key"}


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"chatgpt_to_notion.{name}")


def redact_headers(headers: dict[str, str] | None) -> dict[str, str]:
    if not headers:
        return {}
    return {
        k: ("<REDACTED>" if k.lower() in SENSITIVE_HEADERS else v)
        for k, v in headers.items()
    }


def safe_url(url: str) -> str:
    """Removes query params containing tokens/signatures for safe logging."""
    try:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:
        return "<INVALID_URL>"
