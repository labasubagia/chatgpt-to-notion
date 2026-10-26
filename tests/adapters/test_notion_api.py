"""Unit tests for Notion API adapter."""

from unittest.mock import MagicMock

import pytest

from chatgpt_to_notion.adapters.notion_api import (
    NotionClientImpl,
    _content_type_for_ext,
)
from tests.conftest import make_mock_response


class StubStore:
    """In-memory stand-in for SqliteStore to avoid touching a real DB."""

    def init_db(self) -> None:
        pass

    def get_cached_data_sources(self, db_id):
        return None

    async def async_set_cached_data_sources(self, db_id, sources):
        pass


def _make_ctx():
    ctx = MagicMock()
    ctx.headers = {
        "Authorization": "Bearer x",
        "Notion-Version": "2022-06-28",
    }
    ctx.account_name = "test@example.com"
    return ctx


class TestGetHeaders:
    def test_returns_headers_from_context(self):
        ctx = MagicMock()
        ctx.headers = {
            "Authorization": "Bearer x",
            "Notion-Version": "2022-06-28",
        }
        client = NotionClientImpl(ctx)
        assert client.get_headers() == {
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
class TestIsPageExistsInDb:
    @pytest.mark.asyncio
    async def test_false_on_empty_data_sources(self, mock_aiohttp_session):
        mock_aiohttp_session._responses = [make_mock_response({"data_sources": []})]
        client = NotionClientImpl(_make_ctx(), sqlite_store=StubStore())
        assert (
            await client.is_page_exists_in_db(mock_aiohttp_session, "db1", "query")
            is False
        )

    @pytest.mark.asyncio
    async def test_true_on_match(self, mock_aiohttp_session):
        mock_aiohttp_session._responses = [
            make_mock_response({"data_sources": [{"id": "ds1"}]}),
            make_mock_response(
                {
                    "results": [
                        {
                            "properties": {
                                "Name": {
                                    "title": [{"text": {"content": "query"}}]
                                }
                            }
                        }
                    ]
                }
            ),
        ]
        client = NotionClientImpl(_make_ctx(), sqlite_store=StubStore())
        assert (
            await client.is_page_exists_in_db(mock_aiohttp_session, "db1", "query")
            is True
        )

    @pytest.mark.asyncio
    async def test_false_on_no_match(self, mock_aiohttp_session):
        mock_aiohttp_session._responses = [
            make_mock_response({"data_sources": [{"id": "ds1"}]}),
            make_mock_response({"results": []}),
        ]
        client = NotionClientImpl(_make_ctx(), sqlite_store=StubStore())
        assert (
            await client.is_page_exists_in_db(
                mock_aiohttp_session, "db1", "nomatch"
            )
            is False
        )


class TestAddPageToDb:
    @pytest.mark.asyncio
    async def test_raises_on_missing_file(self, mock_aiohttp_session):
        client = NotionClientImpl(_make_ctx(), sqlite_store=StubStore())
        with pytest.raises(FileNotFoundError):
            await client.add_page_to_db(
                mock_aiohttp_session, "db1", "/nonexistent.png", "prompt"
            )

    @pytest.mark.asyncio
    async def test_uploads_and_creates_page(self, mock_aiohttp_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake-png")
        mock_aiohttp_session._responses = [
            make_mock_response({"id": "upload1"}),
            make_mock_response({"id": "file1", "status": "complete"}),
            make_mock_response({"id": "page1"}),
        ]
        client = NotionClientImpl(_make_ctx(), sqlite_store=StubStore())
        result = await client.add_page_to_db(
            mock_aiohttp_session, "db1", str(test_file), "my prompt"
        )
        assert result["id"] == "page1"

    @pytest.mark.asyncio
    async def test_send_400_treated_as_complete(self, mock_aiohttp_session, tmp_path):
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake-png")
        mock_aiohttp_session._responses = [
            make_mock_response({"id": "upload1"}),
            make_mock_response({}, status=400),
            make_mock_response({"id": "page1"}),
        ]
        client = NotionClientImpl(_make_ctx(), sqlite_store=StubStore())
        result = await client.add_page_to_db(
            mock_aiohttp_session, "db1", str(test_file), "prompt"
        )
        assert result["id"] == "page1"
