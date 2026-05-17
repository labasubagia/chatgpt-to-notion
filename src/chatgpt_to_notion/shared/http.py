"""HTTP helper utilities."""

import asyncio

import aiohttp
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .constants import HTTP_TIMEOUT_SECONDS, MAX_RETRIES


def http_retryable(status_code: int | None) -> bool:
    if status_code is None:
        return False
    return status_code == 429 or status_code >= 500


def should_retry_http(exception: Exception) -> bool:
    if isinstance(exception, aiohttp.ClientResponseError):
        return http_retryable(exception.status)
    if isinstance(exception, aiohttp.ClientError):
        status = getattr(exception, "status", None)
        if status is not None:
            return http_retryable(status)
        return True
    if isinstance(
        exception,
        aiohttp.ClientConnectorError | aiohttp.ClientTimeout | asyncio.TimeoutError,
    ):
        return True
    return False


def retry_http():
    return retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(should_retry_http),
        reraise=True,
    )


def get_http_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
