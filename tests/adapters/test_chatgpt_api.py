"""
Integration and unit tests for chatgpt_api.py using pure mocking.
"""

import pytest
from chatgpt_to_notion.adapters.chatgpt_api import (
    get_headers,
    get_conversations,
    get_conversation_details,
    delete_conversation,
    get_image_generations,
    get_library_images,
    delete_library_file,
    extract_file_id_from_thumbnail_url,
    remove_library_images_by_query,
    _parse_sse_events,
    get_conversation_mapping_key_by_asset_pointer,
    get_prompt_from_image_node_in_conversation,
)
from chatgpt_to_notion.domain.models import RuntimeOptions
from chatgpt_to_notion.shared.http import DetailedHTTPError
from tests.conftest import make_mock_response


@pytest.mark.integration
class TestChatGPTHeaders:
    def test_get_headers_with_default(self, mock_config_toml):
        """Should return correct default headers from config.toml."""
        headers = get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_token_abc"
        assert headers["User-Agent"] == "TestAgent/1.0"

    def test_get_headers_with_options(self, mock_config_toml):
        """Should return headers corresponding to specified options/account."""
        options = RuntimeOptions(account="default@example.com")
        headers = get_headers(options)
        assert headers["Authorization"] == "Bearer test_token_abc"


@pytest.mark.integration
class TestChatGPTApiOperations:
    async def test_get_conversations_success(self, mock_aiohttp_session):
        """Should fetch conversations dict successfully."""
        mock_response = make_mock_response({"items": [{"id": "conv_1"}]})
        mock_aiohttp_session._responses = [mock_response]

        res = await get_conversations(mock_aiohttp_session)
        assert res == {"items": [{"id": "conv_1"}]}

    async def test_get_conversations_failure(self, mock_aiohttp_session):
        """Should raise DetailedHTTPError with details on HTTP failure."""
        mock_response = make_mock_response(
            {}, status=400, reason="Bad Request", text_data="Error detail"
        )
        mock_aiohttp_session._responses = [mock_response]

        with pytest.raises(DetailedHTTPError) as exc_info:
            await get_conversations(mock_aiohttp_session)
        assert exc_info.value.status == 400
        assert "Bad Request" in str(exc_info.value)
        assert "Error detail" in str(exc_info.value)

    async def test_get_conversation_details_success(self, mock_aiohttp_session):
        """Should fetch detailed conversation information successfully."""
        mock_response = make_mock_response({"id": "conv_123", "title": "Test Title"})
        mock_aiohttp_session._responses = [mock_response]

        res = await get_conversation_details(mock_aiohttp_session, "conv_123")
        assert res["id"] == "conv_123"

    async def test_delete_conversation_success(self, mock_aiohttp_session):
        """Should delete conversation successfully (marks is_visible to False)."""
        mock_response = make_mock_response({"success": True})
        mock_aiohttp_session._responses = [mock_response]

        res = await delete_conversation(mock_aiohttp_session, "conv_123")
        assert res["success"] is True

    async def test_delete_conversation_already_deleted(self, mock_aiohttp_session):
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

        res = await delete_conversation(mock_aiohttp_session, "conv_123")
        assert res == {"already_deleted": True}

    async def test_delete_conversation_404_other(self, mock_aiohttp_session):
        """Should still raise on 404 without 'conversation_deleted' signal."""
        mock_response = make_mock_response(
            {},
            status=404,
            reason="Not Found",
            text_data="Conversation not found",
        )
        mock_aiohttp_session._responses = [mock_response]

        with pytest.raises(DetailedHTTPError) as exc_info:
            await delete_conversation(mock_aiohttp_session, "conv_123")
        assert exc_info.value.status == 404
        assert "Conversation not found" in str(exc_info.value)

    async def test_get_image_generations_success(self, mock_aiohttp_session):
        """Should fetch image generation records successfully."""
        mock_response = make_mock_response({"items": [{"id": "gen_1"}]})
        mock_aiohttp_session._responses = [mock_response]

        res = await get_image_generations(mock_aiohttp_session)
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

    async def test_get_library_images_success(self, mock_aiohttp_session):
        """Should fetch library images with query and cursor."""
        mock_response = make_mock_response({
            "items": [self.SAMPLE_ITEM],
            "cursor": "next_cursor_value",
        })
        mock_aiohttp_session._responses = [mock_response]

        res = await get_library_images(
            mock_aiohttp_session, query="ocean", cursor="prev_cursor"
        )
        assert len(res["items"]) == 1
        assert res["cursor"] == "next_cursor_value"

    async def test_get_library_images_no_cursor(self, mock_aiohttp_session):
        """Should work without cursor param."""
        mock_response = make_mock_response({"items": [self.SAMPLE_ITEM]})
        mock_aiohttp_session._responses = [mock_response]

        res = await get_library_images(mock_aiohttp_session, query="test")
        assert len(res["items"]) == 1

    async def test_delete_library_file_success(self, mock_aiohttp_session):
        """Should delete library file via POST with query params."""
        mock_response = make_mock_response({"success": True})
        mock_aiohttp_session._responses = [mock_response]

        res = await delete_library_file(mock_aiohttp_session, self.SAMPLE_ITEM)
        assert res["success"] is True

    async def test_delete_library_file_already_deleted(self, mock_aiohttp_session):
        """Should handle 404 as already_deleted."""
        mock_response = make_mock_response(
            {}, status=404, reason="Not Found", text_data="already deleted"
        )
        mock_aiohttp_session._responses = [mock_response]

        res = await delete_library_file(mock_aiohttp_session, self.SAMPLE_ITEM)
        assert res == {"already_deleted": True}

    async def test_remove_library_images_by_query(
        self, mock_aiohttp_session, mocker
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

        total = await remove_library_images_by_query(
            mock_aiohttp_session, query="ocean", max_concurrent=2
        )
        assert total == 2


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


class TestExtractFileIdFromThumbnailUrl:
    def test_extract_success(self):
        """Should extract file_id from thumbnail_url id param."""
        url = (
            "https://chatgpt.com/backend-api/estuary/content"
            "?id=3f9f1716e9284b6%23file_00000000a20481fa8d7f9a49f06f7841"
            "%23thumbnail_fit_256x256"
        )
        assert extract_file_id_from_thumbnail_url(url) == "file_00000000a20481fa8d7f9a49f06f7841"

    def test_extract_no_id_param(self):
        """Should return None when id param is missing."""
        assert extract_file_id_from_thumbnail_url("https://example.com") is None

    def test_extract_malformed_id(self):
        """Should return None when id has no # separators."""
        url = "https://example.com?id=just_single_value"
        assert extract_file_id_from_thumbnail_url(url) is None


class TestChatGPTMappingHelper:
    def test_get_conversation_mapping_key_by_asset_pointer(self):
        """Should locate node mapping key corresponding to asset pointer."""
        data = {
            "mapping": {
                "node_1": {
                    "message": {
                        "content": {
                            "parts": [{"asset_pointer": "target_asset"}]
                        }
                    }
                },
                "node_2": {
                    "message": {
                        "content": {
                            "parts": [{"asset_pointer": "other_asset"}]
                        }
                    }
                },
            }
        }
        key = get_conversation_mapping_key_by_asset_pointer(data, "target_asset")
        assert key == "node_1"

        # Test target not found
        assert (
            get_conversation_mapping_key_by_asset_pointer(data, "nonexistent")
            is None
        )

    def test_get_prompt_from_image_node_in_conversation(self):
        """Should climb the conversation tree and retrieve prompt from user message."""
        data = {
            "mapping": {
                "node_child": {
                    "parent": "node_parent",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": [{"asset_pointer": "my_asset"}]},
                    },
                },
                "node_parent": {
                    "parent": None,
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["This is the prompt!"]},
                    },
                },
            }
        }

        # Start search with start_node_id
        prompt = get_prompt_from_image_node_in_conversation(
            data, "node_child", "my_asset"
        )
        assert prompt == "This is the prompt!"

        # Fallback using asset_pointer mapping search if start_node_id not in mapping
        prompt_fallback = get_prompt_from_image_node_in_conversation(
            data, "node_unknown", "my_asset"
        )
        assert prompt_fallback == "This is the prompt!"

        # None if asset not found
        assert (
            get_prompt_from_image_node_in_conversation(
                data, "node_unknown", "nonexistent_asset"
            )
            is None
        )


@pytest.mark.integration
class TestConversationDetailsBatch:
    """Tests for get_conversation_details_batch and get_conversation_details_with_delay."""

    async def test_get_conversation_details_with_delay(self, mock_aiohttp_session, mock_config_toml):
        """Should fetch conversation details with optional delay."""
        from chatgpt_to_notion.adapters.chatgpt_api import get_conversation_details_with_delay
        from unittest.mock import AsyncMock, patch

        mock_response = make_mock_response({"id": "conv_123", "title": "Test"})
        mock_aiohttp_session._responses = [mock_response]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            res = await get_conversation_details_with_delay(
                mock_aiohttp_session, "conv_123", delay=1.5,
                headers={"Authorization": "Bearer test"}
            )
            mock_sleep.assert_awaited_once_with(1.5)
            assert res["id"] == "conv_123"

    async def test_get_conversation_details_batch_sequential(self, mock_aiohttp_session, mock_config_toml):
        """Should fetch multiple conversations sequentially when batch_size=0."""
        from chatgpt_to_notion.adapters.chatgpt_api import get_conversation_details_batch
        from unittest.mock import AsyncMock, patch

        mock_aiohttp_session._responses = [
            make_mock_response({"id": "conv_1"}),
            make_mock_response({"id": "conv_2"}),
            make_mock_response({"id": "conv_3"}),
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            res = await get_conversation_details_batch(
                mock_aiohttp_session,
                ["conv_1", "conv_2", "conv_3"],
                delay=0.5,
                batch_size=0,
                headers={"Authorization": "Bearer test"},
            )
            assert len(res) == 3
            assert res["conv_1"]["id"] == "conv_1"
            assert res["conv_2"]["id"] == "conv_2"
            assert res["conv_3"]["id"] == "conv_3"
            # Should sleep 3 times for 3 requests
            assert mock_sleep.await_count == 3

    async def test_get_conversation_details_batch_with_batching(self, mock_aiohttp_session, mock_config_toml):
        """Should fetch in batches with pause between batches."""
        from chatgpt_to_notion.adapters.chatgpt_api import get_conversation_details_batch
        from unittest.mock import AsyncMock, patch

        # 5 conversations, batch_size=2 -> 3 batches
        mock_aiohttp_session._responses = [
            make_mock_response({"id": "conv_1"}),
            make_mock_response({"id": "conv_2"}),
            make_mock_response({"id": "conv_3"}),
            make_mock_response({"id": "conv_4"}),
            make_mock_response({"id": "conv_5"}),
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            res = await get_conversation_details_batch(
                mock_aiohttp_session,
                ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"],
                delay=0.1,
                batch_size=2,
                batch_pause=1.0,
                headers={"Authorization": "Bearer test"},
            )
            assert len(res) == 5
            # Should sleep for each request (5) + between batches (2 pauses)
            assert mock_sleep.await_count >= 5  # at least 5 for delays

    async def test_get_conversation_details_batch_empty(self, mock_aiohttp_session, mock_config_toml):
        """Should return empty dict for empty conversation_ids."""
        from chatgpt_to_notion.adapters.chatgpt_api import get_conversation_details_batch

        res = await get_conversation_details_batch(mock_aiohttp_session, [], headers={"Authorization": "Bearer test"})
        assert res == {}

    async def test_get_conversation_details_batch_raises_on_error(self, mock_aiohttp_session, mock_config_toml):
        """Should raise exception when any request fails (uses non-retryable 400 error)."""
        from chatgpt_to_notion.adapters.chatgpt_api import get_conversation_details_batch
        from chatgpt_to_notion.shared.http import DetailedHTTPError

        mock_aiohttp_session._responses = [
            make_mock_response({"id": "conv_1"}),
            make_mock_response({}, status=400, reason="Bad Request", text_data="Error"),
        ]

        with pytest.raises(DetailedHTTPError):
            await get_conversation_details_batch(
                mock_aiohttp_session,
                ["conv_1", "conv_2"],
                delay=0.0,
                batch_size=0,
                headers={"Authorization": "Bearer test"},
            )




@pytest.mark.integration
class TestConfigRateLimitFields:
    """Tests for rate limit config fields in domain models."""

    def test_runtime_options_has_rate_limit_fields(self):
        """RuntimeOptions should have fetch_conversation_delay and fetch_batch_size."""
        from chatgpt_to_notion.domain.models import RuntimeOptions
        options = RuntimeOptions(
            fetch_conversation_delay=2.0,
            fetch_batch_size=10,
        )
        assert options.fetch_conversation_delay == 2.0
        assert options.fetch_batch_size == 10

    def test_runtime_options_defaults(self):
        """RuntimeOptions should default rate limit fields to None (not specified)."""
        from chatgpt_to_notion.domain.models import RuntimeOptions
        options = RuntimeOptions()
        assert options.fetch_conversation_delay is None
        assert options.fetch_batch_size is None
