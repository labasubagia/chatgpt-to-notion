"""
Integration and unit tests for chatgpt_api.py using pure mocking.
"""

import pytest
from chatgpt_to_notion.adapters.chatgpt_api import (
    ChatGPTClientImpl,
    _parse_sse_events,
)
from chatgpt_to_notion.domain.models import RuntimeOptions
from chatgpt_to_notion.shared.http import DetailedHTTPError
from tests.conftest import make_mock_response


@pytest.mark.integration
class TestChatGPTHeaders:
    def test_get_headers_with_default(self, mock_config_toml):
        """Should return correct default headers from config.toml."""
        headers = ChatGPTClientImpl().get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_token_abc"
        assert headers["User-Agent"] == "TestAgent/1.0"

    def test_get_headers_with_options(self, mock_config_toml):
        """Should return headers corresponding to specified options/account."""
        options = RuntimeOptions(account="default@example.com")
        headers = ChatGPTClientImpl(options).get_headers()
        assert headers["Authorization"] == "Bearer test_token_abc"


@pytest.mark.integration
class TestChatGPTApiOperations:
    async def test_get_conversation_details_success(
        self, mock_aiohttp_session, mock_config_toml
    ):
        """Should fetch detailed conversation information successfully."""
        mock_response = make_mock_response({"id": "conv_123", "title": "Test Title"})
        mock_aiohttp_session._responses = [mock_response]

        client = ChatGPTClientImpl()
        res = await client.get_conversation_details(mock_aiohttp_session, "conv_123")
        assert res["id"] == "conv_123"

    async def test_delete_conversation_success(
        self, mock_aiohttp_session, mock_config_toml
    ):
        """Should delete conversation successfully (marks is_visible to False)."""
        mock_response = make_mock_response({"success": True})
        mock_aiohttp_session._responses = [mock_response]

        client = ChatGPTClientImpl()
        res = await client.delete_conversation(mock_aiohttp_session, "conv_123")
        assert res["success"] is True

    async def test_delete_conversation_already_deleted(
        self, mock_aiohttp_session, mock_config_toml
    ):
        """Should treat 404 'conversation_deleted' as success (already deleted)."""
        mock_response = make_mock_response(
            {},
            status=404,
            reason="Not Found",
            text_data=(
                '{"detail":{"message":"Conversation has been deleted.'
                'Start a new chat.","code":"conversation_deleted","can_retry":false}}'
            ),
        )
        mock_aiohttp_session._responses = [mock_response]

        client = ChatGPTClientImpl()
        res = await client.delete_conversation(mock_aiohttp_session, "conv_123")
        assert res == {"already_deleted": True}

    async def test_delete_conversation_404_other(
        self, mock_aiohttp_session, mock_config_toml
    ):
        """Should still raise on 404 without 'conversation_deleted' signal."""
        mock_response = make_mock_response(
            {},
            status=404,
            reason="Not Found",
            text_data="Conversation not found",
        )
        mock_aiohttp_session._responses = [mock_response]

        client = ChatGPTClientImpl()
        with pytest.raises(DetailedHTTPError) as exc_info:
            await client.delete_conversation(mock_aiohttp_session, "conv_123")
        assert exc_info.value.status == 404
        assert "Conversation not found" in str(exc_info.value)

    async def test_get_image_generations_success(
        self, mock_aiohttp_session, mock_config_toml
    ):
        """Should fetch image generation records successfully."""
        mock_response = make_mock_response({"items": [{"id": "gen_1"}]})
        mock_aiohttp_session._responses = [mock_response]

        client = ChatGPTClientImpl()
        res = await client.get_image_generations(mock_aiohttp_session)
        assert res == {"items": [{"id": "gen_1"}]}


@pytest.mark.integration
class TestChatGPTLibraryOperations:
    SAMPLE_ITEM = {
        "id": "libfile_ee6704e93ed08191b7780a970237626f",
        "file_id": "file_00000000a20481fa8d7f9a49f06f7841",
        "name": "Ocean sunset panorama.png",
        "parent_directory_id": "libdir_4596307cf16c819189c0ec4c6477618a",
        "thumbnail_url": (
            "https://chatgpt.com/backend-api/estuary/content"
            "?id=3f9f2216e9284b6%23file_00000000a20481fa8d7f9a49f06f7825"
            "%23thumbnail_fit_256x256"
            "&cd=inline&ts=495804&p=fs&cid=1"
            "&sig=df281804a4e145c023b9cf77c960a85206b9d6161ffefb2f1cd44b302d3e23b8"
        ),
    }

    async def test_get_library_images_success(
        self, mock_aiohttp_session, mock_config_toml
    ):
        """Should fetch library images with query and cursor."""
        mock_response = make_mock_response({
            "items": [self.SAMPLE_ITEM],
            "cursor": "next_cursor_value",
        })
        mock_aiohttp_session._responses = [mock_response]

        client = ChatGPTClientImpl()
        res = await client.get_library_images(
            mock_aiohttp_session, query="ocean", cursor="prev_cursor"
        )
        assert len(res["items"]) == 1
        assert res["cursor"] == "next_cursor_value"

    async def test_get_library_images_no_cursor(
        self, mock_aiohttp_session, mock_config_toml
    ):
        """Should work without cursor param."""
        mock_response = make_mock_response({"items": [self.SAMPLE_ITEM]})
        mock_aiohttp_session._responses = [mock_response]

        client = ChatGPTClientImpl()
        res = await client.get_library_images(mock_aiohttp_session, query="test")
        assert len(res["items"]) == 1

    async def test_delete_library_file_success(
        self, mock_aiohttp_session, mock_config_toml
    ):
        """Should delete library file via POST with query params."""
        mock_response = make_mock_response({"success": True})
        mock_aiohttp_session._responses = [mock_response]

        client = ChatGPTClientImpl()
        res = await client.delete_library_file(mock_aiohttp_session, self.SAMPLE_ITEM)
        assert res["success"] is True

    async def test_delete_library_file_already_deleted(
        self, mock_aiohttp_session, mock_config_toml
    ):
        """Should handle 404 as already_deleted."""
        mock_response = make_mock_response(
            {}, status=404, reason="Not Found", text_data="already deleted"
        )
        mock_aiohttp_session._responses = [mock_response]

        client = ChatGPTClientImpl()
        res = await client.delete_library_file(mock_aiohttp_session, self.SAMPLE_ITEM)
        assert res == {"already_deleted": True}

    async def test_delete_library_images_by_query(
        self, mock_aiohttp_session, mock_config_toml, mocker
    ):
        """Should loop through pages and delete all matched images."""
        item2 = {
            **self.SAMPLE_ITEM,
            "id": "libfile_another_id",
            "file_id": "file_another",
        }

        responses = [
            make_mock_response({"items": [self.SAMPLE_ITEM], "cursor": "c1"}),
            make_mock_response({"success": True}),
            make_mock_response({"items": [item2]}),
            make_mock_response({"success": True}),
            make_mock_response({"items": []}),
        ]
        mock_aiohttp_session._responses = responses

        client = ChatGPTClientImpl()
        total = await client.delete_library_images_by_query(
            mock_aiohttp_session, query="ocean", max_concurrent=2
        )
        assert total == 2

    async def test_count_library_images(
        self, mock_aiohttp_session, mock_config_toml
    ):
        """Should iterate all pages and sum item counts."""
        responses = [
            make_mock_response({
                "items": [self.SAMPLE_ITEM, {"id": "b"}],
                "cursor": "c1",
            }),
            make_mock_response({"items": [{"id": "c"}], "cursor": "c2"}),
            make_mock_response({"items": []}),
        ]
        mock_aiohttp_session._responses = responses

        client = ChatGPTClientImpl()
        total = await client.count_library_images(mock_aiohttp_session)
        assert total == 3


class TestParseSSEEvents:
    def test_parse_single_event(self):
        body = "data: {\"key\": \"value\"}\n\n"
        assert _parse_sse_events(body) == [{"key": "value"}]

    def test_parse_multiple_events(self):
        body = "data: {\"a\": 1}\n\ndata: {\"b\": 2}\n\n"
        assert _parse_sse_events(body) == [{"a": 1}, {"b": 2}]

    def test_parse_done_signal_skipped(self):
        body = "data: {\"a\": 1}\n\ndata: [DONE]\n\n"
        assert _parse_sse_events(body) == [{"a": 1}]

    def test_parse_empty_body(self):
        assert _parse_sse_events("") == []

    def test_parse_multiline_data(self):
        body = "data: {\"a\":\ndata: 1}\n\n"
        assert _parse_sse_events(body) == [{"a": 1}]

    def test_parse_ndjson_fallback(self):
        body = '{"a": 1}\n{"b": 2}\n'
        assert _parse_sse_events(body) == [{"a": 1}, {"b": 2}]

    def test_parse_ndjson_single_line(self):
        body = '{"a": 1}'
        assert _parse_sse_events(body) == [{"a": 1}]

    def test_parse_deletion_stream(self):
        body = (
            '{"event":"file.deletion.started","progress":0.0}\n'
            '{"event":"file.deletion.completed","progress":100.0}\n'
        )
        events = _parse_sse_events(body)
        assert len(events) == 2
        assert events[0]["event"] == "file.deletion.started"
        assert events[1]["event"] == "file.deletion.completed"
class TestConfigRateLimitFields:
    """Tests for rate limit config fields in domain models."""

    def test_runtime_options_defaults(self):
        """RuntimeOptions should default rate limit fields to None (not specified)."""
        options = RuntimeOptions()
        assert options.fetch_conversation_delay is None
        assert options.fetch_batch_size is None
