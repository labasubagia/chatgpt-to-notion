"""Tests for logging.py."""

import logging
from chatgpt_to_notion.shared.logging import get_logger, redact_headers, safe_url


def test_get_logger():
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "chatgpt_to_notion.test_module"


def test_redact_headers_empty_or_none():
    assert redact_headers(None) == {}
    assert redact_headers({}) == {}


def test_redact_headers_removes_sensitive_data():
    headers = {
        "authorization": "Bearer token123",
        "cookie": "session=abc",
        "x-api-key": "secret",
        "content-type": "application/json",
        "Authorization": "Bearer uppercase", 
    }
    redacted = redact_headers(headers)
    assert redacted["authorization"] == "<REDACTED>"
    assert redacted["cookie"] == "<REDACTED>"
    assert redacted["x-api-key"] == "<REDACTED>"
    assert redacted["Authorization"] == "<REDACTED>"
    assert redacted["content-type"] == "application/json"


def test_safe_url():
    url_with_query = "https://example.com/image.png?sig=123&token=abc"
    assert safe_url(url_with_query) == "https://example.com/image.png"

    clean_url = "https://example.com/page"
    assert safe_url(clean_url) == "https://example.com/page"

    invalid_url = "not_a_valid_url_at_all"
    # urlsplit will just parse it as path for invalid stuff, but it stays relatively safe
    assert safe_url(invalid_url) == "not_a_valid_url_at_all"
