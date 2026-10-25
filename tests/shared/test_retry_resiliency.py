"""Tests for retry/resiliency logic in http module."""

import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatgpt_to_notion.shared.http import (
    DetailedHTTPError,
    http_retryable,
    retry_http,
    should_retry_http,
    wait_with_retry_after,
)


def _make_detailed_error(status: int, retry_after: float | None = None):
    """Helper to create DetailedHTTPError with required args."""
    from aiohttp import RequestInfo
    return DetailedHTTPError(
        status=status,
        message="Test Error",
        body="{}",
        url="http://test.com",
        req=RequestInfo("GET", "http://test.com", {}, ()),
        retry_after=retry_after,
    )


# Import the mock helper from conftest
from tests.conftest import make_mock_response


def make_mock_response_with_content(data, status=200):
    """Create a mock response with content.iter_chunked support."""
    resp = make_mock_response(data, status=status)

    class MockContent:
        def __init__(self, data):
            self._data = data if isinstance(data, bytes) else str(data).encode()

        async def iter_chunked(self, chunk_size):
            for i in range(0, len(self._data), chunk_size):
                yield self._data[i:i + chunk_size]

    resp.content = MockContent(data)
    return resp


# Valid 1x1 PNG bytes for image tests
PNG_BYTES = bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
    0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
    0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
    0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
    0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
    0x42, 0x60, 0x82,
])


class TestHttpRetryable:
    """Tests for http_retryable function."""

    def test_none_status_not_retryable(self):
        assert http_retryable(None) is False

    def test_429_is_retryable(self):
        assert http_retryable(429) is True

    def test_500_is_retryable(self):
        assert http_retryable(500) is True

    def test_503_is_retryable(self):
        assert http_retryable(503) is True

    def test_400_not_retryable(self):
        assert http_retryable(400) is False

    def test_401_not_retryable(self):
        assert http_retryable(401) is False

    def test_404_not_retryable(self):
        assert http_retryable(404) is False


class TestShouldRetryHttp:
    """Tests for should_retry_http function."""

    def test_detailed_http_error_429_retries(self):
        exc = _make_detailed_error(429)
        assert should_retry_http(exc) is True

    def test_detailed_http_error_500_retries(self):
        exc = _make_detailed_error(500)
        assert should_retry_http(exc) is True

    def test_detailed_http_error_400_no_retry(self):
        exc = _make_detailed_error(400)
        assert should_retry_http(exc) is False

    def test_generic_exception_no_retry(self):
        exc = Exception("network error")
        assert should_retry_http(exc) is False

    def test_none_no_retry(self):
        assert should_retry_http(None) is False


class TestWaitWithRetryAfter:
    """Tests for wait_with_retry_after function."""

    def test_retry_after_header_respected(self):
        exc = _make_detailed_error(429, retry_after=30.0)
        retry_state = MagicMock()
        retry_state.outcome = MagicMock()
        retry_state.outcome.exception.return_value = exc
        retry_state.attempt_number = 1
        result = wait_with_retry_after(retry_state)
        assert result == 30.0

    def test_retry_after_capped_at_60(self):
        exc = _make_detailed_error(429, retry_after=120.0)
        retry_state = MagicMock()
        retry_state.outcome = MagicMock()
        retry_state.outcome.exception.return_value = exc
        retry_state.attempt_number = 1
        result = wait_with_retry_after(retry_state)
        assert result == 60.0

    def test_exponential_backoff_fallback(self):
        exc = _make_detailed_error(500)
        retry_state = MagicMock()
        retry_state.outcome = MagicMock()
        retry_state.outcome.exception.return_value = exc
        retry_state.attempt_number = 1
        result = wait_with_retry_after(retry_state)
        assert result == 2.0

    def test_exponential_backoff_attempt_2(self):
        exc = _make_detailed_error(500)
        retry_state = MagicMock()
        retry_state.outcome = MagicMock()
        retry_state.outcome.exception.return_value = exc
        retry_state.attempt_number = 2
        result = wait_with_retry_after(retry_state)
        assert result == 4.0

    def test_exponential_backoff_capped_at_60(self):
        exc = _make_detailed_error(500)
        retry_state = MagicMock()
        retry_state.outcome = MagicMock()
        retry_state.outcome.exception.return_value = exc
        retry_state.attempt_number = 6
        result = wait_with_retry_after(retry_state)
        assert result == 60.0

    def test_no_exception_fallback(self):
        retry_state = MagicMock()
        retry_state.outcome = None
        retry_state.attempt_number = 1
        result = wait_with_retry_after(retry_state)
        assert result == 2.0


class TestRetryHttpDecorator:
    """Tests for retry_http decorator behavior."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self):
        call_count = 0

        @retry_http()
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        call_count = 0

        @retry_http()
        async def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _make_detailed_error(429, retry_after=0.01)
            return "success"

        result = await fail_twice()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_500(self):
        call_count = 0

        @retry_http()
        async def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _make_detailed_error(500)
            return "success"

        result = await fail_twice()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        call_count = 0

        @retry_http()
        async def fail_once():
            nonlocal call_count
            call_count += 1
            raise _make_detailed_error(400)

        with pytest.raises(DetailedHTTPError):
            await fail_once()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        call_count = 0
        from tenacity import retry, stop_after_attempt, retry_if_exception_type

        @retry(
            stop=stop_after_attempt(10),
            wait=lambda retry_state: 0,
            retry=retry_if_exception_type(DetailedHTTPError),
            reraise=True,
        )
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise _make_detailed_error(500)

        with pytest.raises(DetailedHTTPError):
            await always_fail()
        assert call_count == 10


class TestDownloadImageRetry:
    """Tests for download_image retry behavior."""

    @pytest.mark.asyncio
    async def test_download_retries_on_5xx(self, mock_aiohttp_session):
        from chatgpt_to_notion.services.history_service import download_image

        call_count = 0

        def tracking_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                resp = make_mock_response({}, status=503)
            else:
                resp = make_mock_response_with_content(PNG_BYTES, status=200)

            class ContextManager:
                async def __aenter__(ctx_self):
                    return resp
                async def __aexit__(ctx_self, *args):
                    pass
            return ContextManager()

        mock_aiohttp_session.get = tracking_get

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            file_path = f.name

        original_wait = download_image.retry.wait
        download_image.retry.wait = lambda rs: 0
        try:
            await download_image(mock_aiohttp_session, "http://example.com/image.png", file_path)
        finally:
            download_image.retry.wait = original_wait

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_download_429_respects_retry_after(self, mock_aiohttp_session):
        from chatgpt_to_notion.services.history_service import download_image

        call_count = 0

        def tracking_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                resp = make_mock_response({}, status=429)
                resp.headers = {"Retry-After": "1"}
            else:
                resp = make_mock_response_with_content(PNG_BYTES, status=200)

            class ContextManager:
                async def __aenter__(ctx_self):
                    return resp
                async def __aexit__(ctx_self, *args):
                    pass
            return ContextManager()

        mock_aiohttp_session.get = tracking_get

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            file_path = f.name

        original_wait = download_image.retry.wait
        download_image.retry.wait = lambda rs: 0
        try:
            await download_image(mock_aiohttp_session, "http://example.com/image.png", file_path)
        finally:
            download_image.retry.wait = original_wait

        assert call_count == 2


class TestUploadRetry:
    """Tests for upload retry behavior."""

    @pytest.mark.asyncio
    async def test_create_upload_img_retries_on_429(self, mock_aiohttp_session, tmp_path):
        from chatgpt_to_notion.adapters.notion_api import create_upload_img

        call_count = 0

        def tracking_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                resp = make_mock_response({"id": "upload-1"}, status=429)
                resp.headers = {"Retry-After": "0"}
            else:
                resp = make_mock_response({"id": "upload-1"}, status=200)

            class ContextManager:
                async def __aenter__(ctx_self):
                    return resp
                async def __aexit__(ctx_self, *args):
                    pass
            return ContextManager()

        mock_aiohttp_session.post = tracking_post

        test_file = tmp_path / "test.png"
        test_file.write_bytes(PNG_BYTES)

        with patch("chatgpt_to_notion.adapters.notion_api.get_headers", return_value={"Authorization": "Bearer test"}):
            result = await create_upload_img(mock_aiohttp_session, str(test_file))

        assert result["id"] == "upload-1"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_send_upload_img_streaming(self, mock_aiohttp_session, tmp_path):
        from chatgpt_to_notion.adapters.notion_api import send_upload_img

        mock_response = make_mock_response({"id": "upload-1", "status": "complete"}, status=200)

        class ContextManager:
            async def __aenter__(ctx_self):
                return mock_response
            async def __aexit__(ctx_self, *args):
                pass

        cm_instance = ContextManager()
        mock_post = MagicMock(return_value=cm_instance)
        mock_aiohttp_session.post = mock_post

        test_file = tmp_path / "test.png"
        test_file.write_bytes(PNG_BYTES)

        with patch("chatgpt_to_notion.adapters.notion_api.get_headers", return_value={"Authorization": "Bearer test", "Notion-Version": "2025-09-03"}):
            result = await send_upload_img(mock_aiohttp_session, "upload-1", str(test_file))

        assert result["id"] == "upload-1"
        call_args = mock_post.call_args
        assert "data" in call_args.kwargs
        import aiohttp
        assert isinstance(call_args.kwargs["data"], aiohttp.FormData)


class TestExponentialBackoffCalculation:
    """Tests for exponential backoff calculation."""

    def test_backoff_formula(self):
        attempts = [1, 2, 3, 4, 5, 6]
        expected = [2, 4, 8, 16, 32, 60]

        for attempt, expect in zip(attempts, expected):
            retry_state = MagicMock()
            retry_state.outcome = MagicMock()
            retry_state.outcome.exception.return_value = _make_detailed_error(500)
            retry_state.attempt_number = attempt

            result = wait_with_retry_after(retry_state)
            assert result == expect, f"Attempt {attempt}: expected {expect}, got {result}"