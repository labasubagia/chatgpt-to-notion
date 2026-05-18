"""HTTP helper utilities."""

import asyncio

import aiohttp
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .constants import HTTP_TIMEOUT_SECONDS, MAX_RETRIES
from .logging import get_logger

logger = get_logger("http")


def http_retryable(status_code: int | None) -> bool:
    if status_code is None:
        return False
    return status_code == 429 or status_code >= 500


class DetailedHTTPError(Exception):
    def __init__(self, status: int, message: str, body: str):
        super().__init__(f"HTTP {status} {message}\nResponse Body: {body}")
        self.status = status
        self.message_text = message
        self.body = body


async def raise_for_status_with_detail(response: aiohttp.ClientResponse) -> None:
    if response.status >= 400:
        try:
            body = await response.text()
        except Exception:
            body = "<could not read body>"
        raise DetailedHTTPError(response.status, response.reason or "", body)


def should_retry_http(exception: BaseException) -> bool:
    if isinstance(exception, DetailedHTTPError):
        return http_retryable(exception.status)
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


def before_sleep_custom(retry_state) -> None:
    import os

    fn_name = retry_state.fn.__name__ if retry_state.fn else "unknown"
    wait_time = retry_state.idle_for
    exc = retry_state.outcome.exception() if retry_state.outcome else None

    ctx = ""
    args = retry_state.args or ()
    kwargs = retry_state.kwargs or {}

    file_path = kwargs.get("file_path")
    if not file_path:
        if len(args) > 2 and isinstance(args[2], str):
            file_path = args[2]
        else:
            for arg in args:
                if isinstance(arg, str) and (
                    arg.endswith(".png") or "/" in arg or "\\" in arg
                ):
                    file_path = arg
                    break

    if file_path:
        ctx = f" [{os.path.basename(file_path)}]"
    else:
        for key in ["conversation_id", "db_id", "query"]:
            if key in kwargs:
                ctx = f" [{kwargs[key]}]"
                break
        if not ctx:
            for arg in args:
                if isinstance(arg, str) and len(arg) > 5 and not arg.startswith("http"):
                    ctx = f" [{arg}]"
                    break

    logger.warning(
        "Retrying %s%s in %s seconds as it raised %s: %s",
        fn_name,
        ctx,
        wait_time,
        type(exc).__name__ if exc else "Exception",
        exc,
    )


def retry_http():
    return retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(should_retry_http),
        before_sleep=before_sleep_custom,
        reraise=True,
    )


def get_http_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
