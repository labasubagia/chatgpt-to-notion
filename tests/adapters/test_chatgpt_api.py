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
        options = RuntimeOptions(account="default")
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

    async def test_get_image_generations_success(self, mock_aiohttp_session):
        """Should fetch image generation records successfully."""
        mock_response = make_mock_response({"items": [{"id": "gen_1"}]})
        mock_aiohttp_session._responses = [mock_response]

        res = await get_image_generations(mock_aiohttp_session)
        assert res == {"items": [{"id": "gen_1"}]}


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
