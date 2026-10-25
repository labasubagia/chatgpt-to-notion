"""Tests for chatgpt_api module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chatgpt_to_notion.adapters import chatgpt_api
from chatgpt_to_notion.domain.models import RuntimeOptions


def _make_mock_response(json_data=None, status=200):
    """Create a mock HTTP response that works as async context manager."""
    mock_response = AsyncMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    mock_response.json = AsyncMock(return_value=json_data or {})
    mock_response.status = status
    mock_response.text = AsyncMock(return_value="")
    return mock_response


def _make_mock_session():
    """Create a mock session with get/patch/delete/post returning async context managers."""
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=_make_mock_response())
    mock_session.patch = MagicMock(return_value=_make_mock_response())
    mock_session.delete = MagicMock(return_value=_make_mock_response())
    mock_session.post = MagicMock(return_value=_make_mock_response())
    return mock_session


class TestGetHeaders:
    """Tests for get_headers function."""

    def test_returns_provider_context_headers(self):
        """Should return headers from provider context."""
        mock_context = MagicMock()
        mock_context.headers = {"Authorization": "Bearer test", "User-Agent": "Test"}

        with patch("chatgpt_to_notion.adapters.chatgpt_api.get_provider_context", return_value=mock_context):
            headers = chatgpt_api.get_headers()
            assert headers == {"Authorization": "Bearer test", "User-Agent": "Test"}


class TestGetConversationDetails:
    """Tests for get_conversation_details function."""

    @pytest.mark.asyncio
    async def test_calls_correct_endpoint(self):
        """Should call conversation detail endpoint."""
        mock_session = _make_mock_session()

        await chatgpt_api.get_conversation_details(mock_session, "conv_123", {"Authorization": "Bearer test"})

        mock_session.get.assert_called_once()
        args, _ = mock_session.get.call_args
        assert "conversation/conv_123" in args[0]

    @pytest.mark.asyncio
    async def test_returns_json_response(self):
        """Should return parsed JSON response."""
        mock_session = _make_mock_session()
        mock_session.get.return_value = _make_mock_response({"id": "conv_123", "mapping": {}})

        result = await chatgpt_api.get_conversation_details(mock_session, "conv_123", {"Authorization": "Bearer test"})

        assert result == {"id": "conv_123", "mapping": {}}


class TestDeleteConversation:
    """Tests for delete_conversation function."""

    @pytest.mark.asyncio
    async def test_calls_patch_endpoint(self):
        """Should call PATCH to delete conversation."""
        mock_session = _make_mock_session()
        mock_session.patch.return_value = _make_mock_response({"success": True})

        result = await chatgpt_api.delete_conversation(mock_session, "conv_123", {"Authorization": "Bearer test"})

        mock_session.patch.assert_called_once()
        args, _ = mock_session.patch.call_args
        assert "conversation/conv_123" in args[0]
        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_returns_already_deleted_on_404(self):
        """Should return already_deleted when conversation already deleted."""
        mock_session = _make_mock_session()
        mock_response = _make_mock_response({"success": True}, status=404)
        mock_response.text = AsyncMock(return_value="conversation_deleted")
        mock_session.patch.return_value = mock_response

        result = await chatgpt_api.delete_conversation(mock_session, "conv_123", {"Authorization": "Bearer test"})

        assert result == {"already_deleted": True}


class TestGetImageGenerations:
    """Tests for get_image_generations function."""

    @pytest.mark.asyncio
    async def test_calls_library_endpoint(self):
        """Should call image_gen endpoint."""
        mock_session = _make_mock_session()

        await chatgpt_api.get_image_generations(mock_session, {"Authorization": "Bearer test"}, limit=50)

        mock_session.get.assert_called_once()
        args, kwargs = mock_session.get.call_args
        assert "image_gen" in args[0]
        assert kwargs["params"]["limit"] == 50

    @pytest.mark.asyncio
    async def test_returns_json_response(self):
        """Should return parsed JSON response."""
        mock_session = _make_mock_session()
        mock_session.get.return_value = _make_mock_response({"items": [{"id": "gen_1"}]})

        result = await chatgpt_api.get_image_generations(mock_session, {"Authorization": "Bearer test"})

        assert result == {"items": [{"id": "gen_1"}]}


class TestGetConversationMappingKeyByAssetPointer:
    """Tests for get_conversation_mapping_key_by_asset_pointer function."""

    def test_finds_matching_asset_pointer(self):
        """Should return key for matching asset_pointer in parts."""
        data = {
            "mapping": {
                "node_1": {"message": {"author": {"role": "assistant"}, "content": {"parts": [{"asset_pointer": "asset_123"}]}}},
                "node_2": {"message": {"author": {"role": "user"}, "content": {"parts": []}}},
            }
        }
        result = chatgpt_api.get_conversation_mapping_key_by_asset_pointer(data, "asset_123")
        assert result == "node_1"

    def test_returns_none_when_not_found(self):
        """Should return None when asset_pointer not found."""
        data = {"mapping": {"node_1": {"message": {"author": {"role": "user"}, "content": {"parts": []}}}}}
        result = chatgpt_api.get_conversation_mapping_key_by_asset_pointer(data, "missing")
        assert result is None

    def test_skips_nodes_without_message(self):
        """Should skip nodes without message field."""
        data = {"mapping": {"node_1": {"something": "else"}}}
        result = chatgpt_api.get_conversation_mapping_key_by_asset_pointer(data, "asset_123")
        assert result is None

    def test_finds_in_user_message_too(self):
        """Should find asset_pointer regardless of message role."""
        data = {
            "mapping": {
                "node_1": {"message": {"author": {"role": "user"}, "content": {"parts": [{"asset_pointer": "asset_123"}]}}},
            }
        }
        result = chatgpt_api.get_conversation_mapping_key_by_asset_pointer(data, "asset_123")
        assert result == "node_1"

    def test_skips_nodes_without_asset_pointer_in_parts(self):
        """Should skip nodes without asset_pointer in parts."""
        data = {
            "mapping": {
                "node_1": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["just text"]}}},
            }
        }
        result = chatgpt_api.get_conversation_mapping_key_by_asset_pointer(data, "asset_123")
        assert result is None


class TestGetPromptFromImageNodeInConversation:
    """Tests for get_prompt_from_image_node_in_conversation function."""

    def test_extracts_prompt_from_previous_user_node(self):
        """Should extract prompt from previous user message."""
        data = {
            "mapping": {
                "node_1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Test prompt"]}, "metadata": {}}},
                "node_2": {"message": {"author": {"role": "assistant"}, "content": {"parts": []}, "metadata": {}}, "parent": "node_1"},
            }
        }
        result = chatgpt_api.get_prompt_from_image_node_in_conversation(data, "node_2", "asset_123")
        assert result == "Test prompt"

    def test_returns_none_when_no_parent(self):
        """Should return None when node has no parent."""
        data = {"mapping": {"node_1": {"message": {"author": {"role": "assistant"}, "content": {"parts": []}}}}}
        result = chatgpt_api.get_prompt_from_image_node_in_conversation(data, "node_1", "asset_123")
        assert result is None

    def test_returns_none_when_parent_not_user(self):
        """Should return None when parent is not a user message."""
        data = {
            "mapping": {
                "node_1": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["Previous"]}}},
                "node_2": {"message": {"author": {"role": "assistant"}, "content": {"parts": []}}, "parent": "node_1"},
            }
        }
        result = chatgpt_api.get_prompt_from_image_node_in_conversation(data, "node_2", "asset_123")
        assert result is None

    def test_returns_none_when_parent_missing_content(self):
        """Should return None when parent has no content parts."""
        data = {
            "mapping": {
                "node_1": {"message": {"author": {"role": "user"}, "content": {"parts": []}}},
                "node_2": {"message": {"author": {"role": "assistant"}, "content": {"parts": []}}, "parent": "node_1"},
            }
        }
        result = chatgpt_api.get_prompt_from_image_node_in_conversation(data, "node_2", "asset_123")
        assert result is None

    def test_skips_system_messages(self):
        """Should skip system messages when traversing parents."""
        data = {
            "mapping": {
                "node_1": {"message": {"author": {"role": "system"}, "content": {"parts": ["System"]}}},
                "node_2": {"message": {"author": {"role": "user"}, "content": {"parts": ["User prompt"]}}, "parent": "node_1"},
                "node_3": {"message": {"author": {"role": "assistant"}, "content": {"parts": []}}, "parent": "node_2"},
            }
        }
        result = chatgpt_api.get_prompt_from_image_node_in_conversation(data, "node_3", "asset_123")
        assert result == "User prompt"


class TestParseSSEEvents:
    """Tests for _parse_sse_events function."""

    def test_parses_valid_sse(self):
        """Should parse valid SSE events."""
        body = 'data: {"event": "test", "data": "value"}\n\n'
        events = chatgpt_api._parse_sse_events(body)
        assert len(events) == 1
        assert events[0] == {"event": "test", "data": "value"}

    def test_skips_invalid_json(self):
        """Should skip invalid JSON lines."""
        body = 'data: not json\n\ndata: {"valid": true}\n\n'
        events = chatgpt_api._parse_sse_events(body)
        assert len(events) == 1
        assert events[0] == {"valid": True}

    def test_skips_empty_data(self):
        """Should skip empty data lines."""
        body = 'data: \n\ndata: {"valid": true}\n\n'
        events = chatgpt_api._parse_sse_events(body)
        assert len(events) == 1
        assert events[0] == {"valid": True}

    def test_handles_multiple_events(self):
        """Should handle multiple SSE events."""
        body = 'data: {"a": 1}\n\ndata: {"b": 2}\n\n'
        events = chatgpt_api._parse_sse_events(body)
        assert len(events) == 2
        assert events[0] == {"a": 1}
        assert events[1] == {"b": 2}


class TestGetLibraryImages:
    """Tests for get_library_images function."""

    @pytest.mark.asyncio
    async def test_calls_correct_endpoint(self):
        """Should call library endpoint with image category."""
        mock_session = _make_mock_session()

        await chatgpt_api.get_library_images(mock_session, query="test", headers={"Authorization": "Bearer test"})

        mock_session.get.assert_called_once()
        args, kwargs = mock_session.get.call_args
        assert "library/nodes" in args[0]
        assert kwargs["params"]["categories"] == "image"
        assert kwargs["params"]["q"] == "test"

    @pytest.mark.asyncio
    async def test_returns_json_response(self):
        """Should return parsed JSON response."""
        mock_session = _make_mock_session()
        mock_session.get.return_value = _make_mock_response({"nodes": [{"id": "lib_1"}]})

        result = await chatgpt_api.get_library_images(mock_session, query="test", headers={"Authorization": "Bearer test"})

        assert result == {"nodes": [{"id": "lib_1"}]}


class TestDeleteLibraryFile:
    """Tests for delete_library_file function."""

    @pytest.mark.asyncio
    async def test_calls_delete_endpoint(self):
        """Should call DELETE for library file."""
        mock_session = _make_mock_session()
        mock_session.post.return_value = _make_mock_response({"success": True, "events": []})

        item = {"id": "lib_123", "file_id": "file_123", "parent_directory_id": "dir_123", "name": "test.png"}

        result = await chatgpt_api.delete_library_file(mock_session, item, {"Authorization": "Bearer test"})

        mock_session.post.assert_called_once()
        args, _ = mock_session.post.call_args
        assert "library/files/lib_123/delete_stream" in args[0]
        assert result == {"success": True, "events": []}

    @pytest.mark.asyncio
    async def test_returns_already_deleted_on_404(self):
        """Should return already_deleted when file already deleted."""
        mock_session = _make_mock_session()
        mock_response = _make_mock_response({"already_deleted": True}, status=404)
        mock_session.post.return_value = mock_response

        item = {"id": "lib_123", "file_id": "file_123", "parent_directory_id": "dir_123", "name": "test.png"}

        result = await chatgpt_api.delete_library_file(mock_session, item, {"Authorization": "Bearer test"})

        assert result == {"already_deleted": True}


class TestExtractFileIdFromThumbnailUrl:
    """Tests for extract_file_id_from_thumbnail_url function."""

    def test_extracts_file_id(self):
        """Should extract file ID from thumbnail URL query param."""
        url = "https://example.com/thumbnail?id=file_abc%23file_abc"
        result = chatgpt_api.extract_file_id_from_thumbnail_url(url)
        assert result == "file_abc"

    def test_returns_none_for_invalid_url(self):
        """Should return None for malformed URL."""
        result = chatgpt_api.extract_file_id_from_thumbnail_url("https://example.com/")
        assert result is None

    def test_returns_none_when_no_file_segment(self):
        """Should return None when URL has no id query param."""
        url = "https://example.com/thumbnail"
        result = chatgpt_api.extract_file_id_from_thumbnail_url(url)
        assert result is None


class TestCountLibraryImages:
    """Tests for count_library_images function."""

    @pytest.mark.asyncio
    async def test_returns_total_count(self):
        """Should return total count from paginated results."""
        mock_session = _make_mock_session()
        mock_session.get.side_effect = [
            _make_mock_response({"items": [{"id": "1"}, {"id": "2"}], "cursor": "cursor_1"}),
            _make_mock_response({"items": [{"id": "3"}], "cursor": None}),
        ]

        total = await chatgpt_api.count_library_images(mock_session, {"Authorization": "Bearer test"})

        assert total == 3

    @pytest.mark.asyncio
    async def test_empty_library_returns_zero(self):
        """Empty library should return 0."""
        mock_session = _make_mock_session()
        mock_session.get.return_value = _make_mock_response({"items": [], "cursor": None})

        total = await chatgpt_api.count_library_images(mock_session, {"Authorization": "Bearer test"})

        assert total == 0


class TestDeleteLibraryImagesByQuery:
    """Tests for delete_library_images_by_query function."""

    @pytest.mark.asyncio
    async def test_deletes_matching_images(self):
        """Should delete images matching query."""
        mock_session = _make_mock_session()

        # get_library_images returns items with required fields
        mock_session.get.return_value = _make_mock_response(
            {"items": [{"id": "lib_1", "file_id": "file_1", "parent_directory_id": "dir_1", "name": "test.png"}], "cursor": None}
        )
        mock_session.post.return_value = _make_mock_response({"success": True, "events": []})

        deleted = await chatgpt_api.delete_library_images_by_query(
            mock_session, "test", {"Authorization": "Bearer test"}
        )

        assert deleted == 1
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_matches(self):
        """Should return 0 when no images match query (API returns empty)."""
        mock_session = _make_mock_session()
        # API returns empty items for non-matching query
        mock_session.get.return_value = _make_mock_response({"items": [], "cursor": None})

        deleted = await chatgpt_api.delete_library_images_by_query(
            mock_session, "test", {"Authorization": "Bearer test"}
        )

        assert deleted == 0
        mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_paginates_through_results(self):
        """Should paginate through all library images."""
        mock_session = _make_mock_session()

        mock_session.get.side_effect = [
            _make_mock_response({"items": [{"id": "lib_1", "file_id": "file_1", "parent_directory_id": "dir_1", "name": "test1.png"}], "cursor": "cursor_1"}),
            _make_mock_response({"items": [{"id": "lib_2", "file_id": "file_2", "parent_directory_id": "dir_2", "name": "test2.png"}], "cursor": None}),
        ]
        mock_session.post.return_value = _make_mock_response({"success": True, "events": []})

        deleted = await chatgpt_api.delete_library_images_by_query(
            mock_session, "test", {"Authorization": "Bearer test"}
        )

        assert deleted == 2
        assert mock_session.get.call_count == 2
        assert mock_session.post.call_count == 2