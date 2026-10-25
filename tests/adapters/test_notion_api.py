"""Unit tests for Notion API adapter."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatgpt_to_notion.adapters.notion_api import (
    get_headers,
    _content_type_for_ext,
    get_db_data_sources,
    query_data_source,
    _query_data_source_for_page,
    is_page_exists_in_db,
    create_upload_img,
    send_upload_img,
    add_page_to_db,
)
from chatgpt_to_notion.shared.http import DetailedHTTPError


def _async_cm(return_value):
    """Create async context manager returning return_value on __aenter__."""
    mgr = AsyncMock()
    mgr.__aenter__.return_value = return_value
    mgr.__aexit__.return_value = None
    return mgr


def _response_with_status(status=200, json_data=None):
    """Create a mock response with status and json."""
    resp = AsyncMock()
    resp.status = status
    if json_data is not None:
        resp.json = AsyncMock(return_value=json_data)
    return resp


def _session_with_post(response):
    """Create a session mock where post() returns an async context manager."""
    session = MagicMock()
    session.post.return_value = _async_cm(response)
    session.get.return_value = _async_cm(response)
    return session


class TestGetHeaders:
    def test_returns_headers_from_context(self):
        with patch("chatgpt_to_notion.adapters.notion_api.get_notion_context") as mock_ctx:
            mock_ctx.return_value.headers = {
                "Authorization": "Bearer x",
                "Notion-Version": "2022-06-28",
            }
            assert get_headers() == {
                "Authorization": "Bearer x",
                "Notion-Version": "2022-06-28",
            }


class TestContentTypeForExt:
    @pytest.mark.parametrize(
        "ext,expected",
        [
            (".png", "image/png"),
            (".jpg", "image/jpeg"),
            (".jpeg", "image/jpeg"),
            (".webp", "image/webp"),
            (".unknown", "image/png"),
            ("", "image/png"),
        ],
    )
    def test_returns_correct_mime(self, ext, expected):
        assert _content_type_for_ext(ext) == expected


class TestGetDbDataSources:
    @pytest.mark.asyncio
    async def test_returns_cached(self):
        session = MagicMock()
        # Patch at the module where functions are used
        with patch("chatgpt_to_notion.adapters.notion_api.get_cached_data_sources") as mock_get, \
             patch("chatgpt_to_notion.adapters.sqlite_store.init_db") as mock_init, \
             patch("chatgpt_to_notion.adapters.notion_api.async_set_cached_data_sources") as mock_set:
            
            mock_get.return_value = [{"id": "cached"}]
            mock_init.return_value = None
            
            result = await get_db_data_sources(session, "db1")
            assert result == [{"id": "cached"}]
            session.get.assert_not_called()
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetches_and_caches_on_miss(self):
        response = _response_with_status(200, {"data_sources": [{"id": "ds1"}]})
        session = MagicMock()
        session.get.return_value = _async_cm(response)

        with patch("chatgpt_to_notion.adapters.notion_api.get_cached_data_sources") as mock_get, \
             patch("chatgpt_to_notion.adapters.notion_api.async_set_cached_data_sources") as mock_set, \
             patch("chatgpt_to_notion.adapters.sqlite_store.init_db") as mock_init:
            
            mock_get.return_value = None
            mock_set.return_value = None
            mock_init.return_value = None

            result = await get_db_data_sources(session, "db1")
            assert result == [{"id": "ds1"}]
            mock_set.assert_awaited_once_with("db1", [{"id": "ds1"}])


class TestQueryDataSource:
    @pytest.mark.asyncio
    async def test_posts_query(self):
        response = _response_with_status(200, {"results": []})
        session = _session_with_post(response)

        result = await query_data_source(session, "ds1", "query")
        assert result == {"results": []}
        session.post.assert_called_once()


class TestQueryDataSourceForPage:
    @pytest.mark.asyncio
    async def test_returns_true_on_match(self):
        session = MagicMock()
        with patch("chatgpt_to_notion.adapters.notion_api.query_data_source") as mock_q:
            mock_q.return_value = {
                "results": [
                    {"properties": {"Name": {"title": [{"text": {"content": "match"}}]}}}
                ]
            }
            result = await _query_data_source_for_page(session, "ds1", "match", {})
            assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_no_match(self):
        session = MagicMock()
        with patch("chatgpt_to_notion.adapters.notion_api.query_data_source") as mock_q:
            mock_q.return_value = {"results": []}
            result = await _query_data_source_for_page(session, "ds1", "nomatch", {})
            assert result is False


class TestIsPageExistsInDb:
    @pytest.mark.asyncio
    async def test_false_on_empty_data_sources(self):
        session = MagicMock()
        with patch("chatgpt_to_notion.adapters.notion_api.get_db_data_sources") as mock_gds:
            mock_gds.return_value = []
            result = await is_page_exists_in_db(session, "db1", "query")
            assert result is False

    @pytest.mark.asyncio
    async def test_true_on_any_match(self):
        session = MagicMock()
        with patch("chatgpt_to_notion.adapters.notion_api.get_db_data_sources") as mock_gds:
            mock_gds.return_value = [{"id": "ds1"}, {"id": "ds2"}]
            with patch("chatgpt_to_notion.adapters.notion_api._query_data_source_for_page") as mock_q:
                mock_q.side_effect = [False, True]
                result = await is_page_exists_in_db(session, "db1", "query")
                assert result is True


class TestCreateUploadImg:
    @pytest.mark.asyncio
    async def test_raises_on_missing_file(self):
        session = MagicMock()
        with pytest.raises(FileNotFoundError):
            await create_upload_img(session, "/nonexistent.png")

    @pytest.mark.asyncio
    async def test_creates_upload(self):
        response = _response_with_status(200, {"id": "upload1"})
        session = _session_with_post(response)

        with patch("os.path.exists", return_value=True):
            result = await create_upload_img(session, "test.png")
            assert result == {"id": "upload1"}


class TestSendUploadImg:
    @pytest.mark.asyncio
    async def test_raises_on_missing_file(self):
        session = MagicMock()
        with pytest.raises(FileNotFoundError):
            await send_upload_img(session, "upload1", "/nonexistent.png")

    @pytest.mark.asyncio
    async def test_handles_400_as_complete(self):
        # DetailedHTTPError(status, message, body, url, req)
        exc = DetailedHTTPError(400, "Bad Request", "", "http://test", MagicMock())
        mgr = AsyncMock()
        mgr.__aenter__.side_effect = exc
        mgr.__aexit__.return_value = None
        session = MagicMock()
        session.post.return_value = mgr

        with patch("os.path.exists", return_value=True), patch("builtins.open", MagicMock()):
            result = await send_upload_img(session, "upload1", "test.png")
            assert result == {"id": "upload1", "status": "complete"}


class TestAddPageToDb:
    @pytest.mark.asyncio
    async def test_raises_on_missing_file(self):
        session = MagicMock()
        with pytest.raises(FileNotFoundError):
            await add_page_to_db(session, "db1", "/nonexistent.png", "prompt")

    @pytest.mark.asyncio
    async def test_uploads_and_creates_page(self):
        response = _response_with_status(200, {"id": "upload1"})
        session = _session_with_post(response)

        with patch("chatgpt_to_notion.adapters.notion_api.create_upload_img") as mock_create, \
             patch("chatgpt_to_notion.adapters.notion_api.send_upload_img") as mock_send, \
             patch("chatgpt_to_notion.adapters.notion_api.get_notion_context") as mock_ctx, \
             patch("os.path.exists", return_value=True):

            mock_create.return_value = {"id": "upload1"}
            mock_send.return_value = {"id": "file1", "status": "complete"}
            mock_ctx.return_value.headers = {
                "Authorization": "Bearer x",
                "Notion-Version": "2022-06-28",
            }
            mock_ctx.return_value.account_name = "test@example.com"

            result = await add_page_to_db(session, "db1", "test.png", "my prompt")
            assert "id" in result
            mock_create.assert_awaited_once()
            mock_send.assert_awaited_once()