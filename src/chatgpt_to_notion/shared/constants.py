"""Shared constants."""

import os
from urllib.parse import urlparse

MAX_RETRIES = 10
HTTP_TIMEOUT_SECONDS = 120
OUTPUT_PATH = "./output"
DEFAULT_CONFIG_PATH = "config.toml"

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def image_ext_from_url(url: str) -> str:
    """Return the image file extension from a URL, defaulting to .png."""
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return ext
    return ".png"
