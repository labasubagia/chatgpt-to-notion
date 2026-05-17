"""
Integration tests for chatgpt.py using pure mocking.

No real API calls are made. All external dependencies are mocked.
"""

import pytest

import db
import chatgpt
from chatgpt import (
    delete_conversation,
    delete_conversation_of_image_generation_uploaded_to_notion,
    download_all_images,
    get_conversation_details,
    get_conversation_mapping_key_by_asset_pointer,
    get_conversations,
    get_headers,
    get_image_generations,
    get_prompt_from_image_node_in_conversation,
    load_image_generations,
)
from tests.conftest import make_mock_response


@pytest.mark.integration
class TestChatGPTHeaders:
    """Tests for ChatGPT headers generation."""

    def test_headers_contain_auth(self, mock_config_toml):
        """Headers should contain Authorization."""
        headers = get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    def test_headers_contain_user_agent(self, mock_config_toml):
        """Headers should contain User-Agent."""
        headers = get_headers()
        assert "User-Agent" in headers

    def test_headers_contain_content_type(self, mock_config_toml):
        """Headers should contain Content-Type."""
        headers = get_headers()
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"

    def test_headers_do_not_require_cookie(self, mock_config_toml):
        """Headers should not include a Cookie entry."""
        headers = get_headers()
        assert isinstance(headers, dict)
        assert "Authorization" in headers
        assert "User-Agent" in headers
        assert "Content-Type" in headers
        assert "Cookie" not in headers


@pytest.mark.integration
class TestChatGPTFetchImageGenerations:
    """Tests for fetch_image_generations function."""

    async def test_fetch_image_generations_new_items(
        self, mock_aiohttp_session, monkeypatch, isolated_db
    ):
        """Should fetch details only for new items and upsert to DB."""
        from unittest.mock import AsyncMock, patch

        mock_data = {
            "items": [
                {
                    "id": "img_123",
                    "conversation_id": "conv_abc",
                    "message_id": "msg_def",
                    "asset_pointer": "asset_ghi",
                    "url": "https://example.com/image.png",
                    "created_at": 1705315800,
                }
            ]
        }

        with patch("chatgpt.get_image_generations", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_data

            with patch(
                "chatgpt.get_conversation_details", new_callable=AsyncMock
            ) as mock_detail:
                mock_detail.return_value = {
                    "mapping": {
                        "node1": {
                            "message": {
                                "author": {"role": "user"},
                                "content": {"parts": ["Test prompt"]},
                            },
                            "parent": None,
                        }
                    }
                }

                with patch(
                    "chatgpt.get_prompt_from_image_node_in_conversation"
                ) as mock_prompt:
                    mock_prompt.return_value = "Test prompt"

                    with patch("chatgpt.get_headers") as mock_headers:
                        mock_headers.return_value = {"Authorization": "Bearer test"}

                        result = await chatgpt.fetch_image_generations(
                            limit=5,
                            options=type("Options", (), {"account": "default"})(),
                        )

                        assert isinstance(result, list)
                        assert len(result) == 1
                        assert result[0].id == "img_123"
                        assert result[0].prompt == "Test prompt"

                        mock_get.assert_called_once()
                        mock_detail.assert_called_once()

    async def test_fetch_image_generations_existing_items_skip_details(
        self, mock_aiohttp_session, monkeypatch, isolated_db
    ):
        """Should skip fetching details for items already in DB."""
        from unittest.mock import AsyncMock, patch
        from models import ChatGPTImageGeneration

        gen = ChatGPTImageGeneration(
            created_at="2024-01-15T10:30:00.000000+00:00",
            id="img_123",
            conversation_id="conv_abc",
            message_id="msg_def",
            asset_pointer="asset_ghi",
            url="https://example.com/image.png",
            prompt="Existing prompt",
        )
        db.upsert_generations("default", [gen])

        mock_data = {
            "items": [
                {
                    "id": "img_123",
                    "conversation_id": "conv_abc",
                    "message_id": "msg_def",
                    "asset_pointer": "asset_ghi",
                    "url": "https://example.com/image.png",
                    "created_at": 1705315800,
                }
            ]
        }

        with patch("chatgpt.get_image_generations", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_data

            with patch(
                "chatgpt.get_conversation_details", new_callable=AsyncMock
            ) as mock_detail:
                mock_detail.return_value = {}

                with patch("chatgpt.get_headers") as mock_headers:
                    mock_headers.return_value = {"Authorization": "Bearer test"}

                    result = await chatgpt.fetch_image_generations(
                        limit=5,
                        options=type("Options", (), {"account": "default"})(),
                    )

                    assert len(result) == 1
                    assert result[0].id == "img_123"
                    assert result[0].prompt == "Existing prompt"

                    mock_get.assert_called_once()
                    mock_detail.assert_not_called()

    async def test_fetch_image_generations_empty(
        self, mock_aiohttp_session, monkeypatch, isolated_db
    ):
        """Should return empty list when no generations."""
        from unittest.mock import AsyncMock, patch

        with patch("chatgpt.get_image_generations", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"items": []}

            with patch("chatgpt.get_headers") as mock_headers:
                mock_headers.return_value = {"Authorization": "Bearer test"}

                result = await chatgpt.fetch_image_generations(
                    limit=5,
                    options=type("Options", (), {"account": "default"})(),
                )

                assert result == []


@pytest.mark.integration
class TestChatGPTUploadToNotion:
    """Tests for upload_to_notion function."""

    async def test_upload_to_notion_full_workflow(self, monkeypatch, tmp_path):
        """Should execute full upload workflow."""
        from unittest.mock import AsyncMock, patch

        image_folder = str(tmp_path / "images")

        with patch(
            "chatgpt.fetch_image_generations", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = [
                {
                    "id": "img_123",
                    "url": "https://example.com/image.png",
                    "prompt": "Test prompt",
                }
            ]

            with patch(
                "chatgpt.download_all_images", new_callable=AsyncMock
            ) as mock_download:
                mock_download.return_value = None

                with patch("chatgpt.add_prompt_to_images") as mock_add_prompt:
                    mock_add_prompt.return_value = None

                    with patch(
                        "chatgpt.upload_all_images_to_notion", new_callable=AsyncMock
                    ) as mock_upload:
                        mock_upload.return_value = None

                        await chatgpt.upload_to_notion(
                            image_folder=image_folder,
                            db_id="test_db",
                            upload_to_notion=True,
                            remove_in_chatgpt=False,
                            add_prompt_to_image=True,
                            limit=5,
                        )

                        mock_fetch.assert_called_once()
                        mock_download.assert_called_once()
                        mock_add_prompt.assert_called_once()
                        mock_upload.assert_called_once()

    async def test_upload_to_notion_skip_download(self, monkeypatch, tmp_path):
        """Should skip download when add_prompt_to_image=False."""
        from unittest.mock import AsyncMock, patch

        image_folder = str(tmp_path / "images")

        with patch(
            "chatgpt.fetch_image_generations", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = []

            with patch(
                "chatgpt.download_all_images", new_callable=AsyncMock
            ) as mock_download:
                mock_download.return_value = None

                await chatgpt.upload_to_notion(
                    image_folder=image_folder,
                    db_id="test_db",
                    upload_to_notion=False,
                    add_prompt_to_image=False,
                    limit=5,
                )

                mock_fetch.assert_called_once()


@pytest.mark.integration
class TestChatGPTConversations:
    """Tests for ChatGPT conversation operations."""

    async def test_get_conversations(self, mock_aiohttp_session):
        """Should fetch conversations."""
        mock_aiohttp_session._responses = [
            make_mock_response({"items": [{"id": "conv_1"}]})
        ]

        result = await get_conversations(mock_aiohttp_session, limit=5)
        assert "items" in result
        assert len(result["items"]) == 1

    async def test_get_conversations_with_params(self, mock_aiohttp_session):
        """Should fetch conversations with filters."""
        mock_aiohttp_session._responses = [make_mock_response({"items": []})]

        result = await get_conversations(
            mock_aiohttp_session, limit=5, is_archived=False, is_starred=False
        )
        assert "items" in result

    async def test_get_conversation_details(self, mock_aiohttp_session):
        """Should fetch conversation details."""
        mock_aiohttp_session._responses = [
            make_mock_response({"mapping": {"node1": {}}})
        ]

        result = await get_conversation_details(mock_aiohttp_session, "conv_abc")
        assert "mapping" in result


@pytest.mark.integration
class TestChatGPTImageGenerations:
    """Tests for ChatGPT image generation operations."""

    async def test_get_image_generations(self, mock_aiohttp_session):
        """Should fetch recent image generations."""
        mock_aiohttp_session._responses = [
            make_mock_response({"items": [{"id": "img_1"}]})
        ]

        result = await get_image_generations(mock_aiohttp_session, limit=5)
        assert "items" in result
        assert len(result["items"]) == 1

    async def test_delete_conversation(self, mock_aiohttp_session):
        """Should delete/hide conversation."""
        mock_aiohttp_session._responses = [make_mock_response({"success": True})]

        result = await delete_conversation(mock_aiohttp_session, "conv_test")
        assert result is not None


class TestChatGPTHistoryDataset:
    """Tests for loading ChatGPT generations from history."""

    def test_loads_unuploaded_generations(
        self, isolated_db, monkeypatch
    ):
        from models import ChatGPTImageGeneration

        gen_1 = ChatGPTImageGeneration(
            created_at="2026-05-14T00:00:00+00:00",
            id="img_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/1.png",
            prompt="Prompt 1",
        )
        gen_2 = ChatGPTImageGeneration(
            created_at="2026-05-14T00:00:00+00:00",
            id="img_2",
            conversation_id="conv_2",
            message_id="msg_2",
            asset_pointer="asset_2",
            url="https://example.com/2.png",
            prompt="Prompt 2",
        )
        db.upsert_generations("default", [gen_1, gen_2])
        db.mark_uploaded("default", {"img_2"})

        generations = load_image_generations(
            "default",
            options=type("Options", (), {"account": "default"})(),
        )

        assert [generation.id for generation in generations] == ["img_1"]

    def test_loads_all_generations_when_include_uploaded(
        self, isolated_db, monkeypatch
    ):
        from models import ChatGPTImageGeneration

        gen_1 = ChatGPTImageGeneration(
            created_at="2026-05-14T00:00:00+00:00",
            id="img_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/1.png",
            prompt="Prompt 1",
        )
        gen_2 = ChatGPTImageGeneration(
            created_at="2026-05-14T00:00:00+00:00",
            id="img_2",
            conversation_id="conv_2",
            message_id="msg_2",
            asset_pointer="asset_2",
            url="https://example.com/2.png",
            prompt="Prompt 2",
        )
        db.upsert_generations("default", [gen_1, gen_2])
        db.mark_uploaded("default", {"img_2"})

        generations = load_image_generations(
            "default",
            include_uploaded=True,
            options=type("Options", (), {"account": "default"})(),
        )

        assert [generation.id for generation in generations] == ["img_1", "img_2"]


class TestChatGPTPromptExtraction:
    """Tests for prompt extraction logic."""

    def test_get_prompt_from_user_message(self):
        """Should extract prompt from user message."""
        data = {
            "mapping": {
                "node1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["A beautiful sunset"]},
                    },
                    "parent": None,
                }
            }
        }

        prompt = get_prompt_from_image_node_in_conversation(data, "node1", "asset_ghi")
        assert prompt == "A beautiful sunset"

    def test_get_prompt_walks_to_parent(self):
        """Should walk to parent node if current has no prompt."""
        data = {
            "mapping": {
                "node1": {
                    "message": None,
                    "parent": "node2",
                },
                "node2": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Parent prompt"]},
                    },
                    "parent": None,
                },
            }
        }

        prompt = get_prompt_from_image_node_in_conversation(data, "node1", "asset_ghi")
        assert prompt == "Parent prompt"

    def test_get_prompt_returns_none_if_not_found(self):
        """Should return None if no prompt found."""
        data = {
            "mapping": {
                "node1": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Not a user message"]},
                    },
                    "parent": None,
                }
            }
        }

        prompt = get_prompt_from_image_node_in_conversation(data, "node1", "asset_ghi")
        assert prompt is None

    def test_get_prompt_returns_none_when_node_not_in_mapping(self):
        """Should return None when node and asset lookup both fail."""
        data = {
            "mapping": {
                "node1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Some prompt"]},
                    },
                    "parent": None,
                }
            }
        }

        # start_node_id="nonexistent" is not in mapping
        # and asset_pointer won't be found either
        prompt = get_prompt_from_image_node_in_conversation(
            data, "nonexistent", "missing_asset"
        )
        assert prompt is None

    def test_get_conversation_mapping_key_by_asset_pointer(self):
        """Should find node by asset_pointer."""
        data = {
            "mapping": {
                "node1": {
                    "message": {"content": {"parts": [{"asset_pointer": "asset_123"}]}}
                }
            }
        }

        key = get_conversation_mapping_key_by_asset_pointer(data, "asset_123")
        assert key == "node1"

    def test_get_conversation_mapping_key_returns_none(self):
        """Should return None if asset_pointer not found."""
        data = {"mapping": {"node1": {"message": None}}}

        key = get_conversation_mapping_key_by_asset_pointer(data, "asset_456")
        assert key is None

    def test_get_conversation_mapping_key_message_not_dict(self):
        """Should continue when message is not a dict."""
        data = {
            "mapping": {
                "node1": {"message": "not_a_dict"},
                "node2": {
                    "message": {"content": {"parts": [{"asset_pointer": "asset_123"}]}}
                },
            }
        }

        key = get_conversation_mapping_key_by_asset_pointer(data, "asset_123")
        assert key == "node2"

    def test_get_conversation_mapping_key_content_not_dict(self):
        """Should continue when content is not a dict."""
        data = {
            "mapping": {
                "node1": {"message": {"content": "not_a_dict"}},
                "node2": {
                    "message": {"content": {"parts": [{"asset_pointer": "asset_123"}]}}
                },
            }
        }

        key = get_conversation_mapping_key_by_asset_pointer(data, "asset_123")
        assert key == "node2"

    def test_get_conversation_mapping_key_parts_not_list(self):
        """Should continue when parts is not a list."""
        data = {
            "mapping": {
                "node1": {"message": {"content": {"parts": "not_a_list"}}},
                "node2": {
                    "message": {"content": {"parts": [{"asset_pointer": "asset_123"}]}}
                },
            }
        }

        key = get_conversation_mapping_key_by_asset_pointer(data, "asset_123")
        assert key == "node2"


@pytest.mark.integration
class TestChatGPTDownloadImages:
    """Tests for download_all_images function."""

    async def test_download_all_images_success(self, monkeypatch, tmp_path):
        """Should download all images."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))
        download_folder = "images"
        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-01T00:00:00",
                id="img_1",
                conversation_id="conv_1",
                message_id="msg_1",
                asset_pointer="asset_1",
                url="https://example.com/img1.png",
                prompt="test 1",
            ),
            ChatGPTImageGeneration(
                created_at="2024-01-01T00:00:00",
                id="img_2",
                conversation_id="conv_2",
                message_id="msg_2",
                asset_pointer="asset_2",
                url="https://example.com/img2.png",
                prompt="test 2",
            ),
        ]

        with patch("chatgpt.download_image", new_callable=AsyncMock) as mock_download:
            await download_all_images(
                generations=generations, download_folder=download_folder
            )

            assert mock_download.call_count == 2

    async def test_download_all_images_skip_existing(
        self, monkeypatch, tmp_path, capsys
    ):
        """Should skip already downloaded images."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))
        download_folder = "images"

        # Create existing image
        images_dir = tmp_path / download_folder
        images_dir.mkdir()
        (images_dir / "img_1.png").write_bytes(b"fake png")

        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-01T00:00:00",
                id="img_1",
                conversation_id="conv_1",
                message_id="msg_1",
                asset_pointer="asset_1",
                url="https://example.com/img1.png",
                prompt="test 1",
            ),
            ChatGPTImageGeneration(
                created_at="2024-01-01T00:00:00",
                id="img_2",
                conversation_id="conv_2",
                message_id="msg_2",
                asset_pointer="asset_2",
                url="https://example.com/img2.png",
                prompt="test 2",
            ),
        ]

        with patch("chatgpt.download_image", new_callable=AsyncMock) as mock_download:
            await download_all_images(
                generations=generations, download_folder=download_folder
            )

            # Only img_2 should be downloaded
            assert mock_download.call_count == 1
            captured = capsys.readouterr()
            assert "skipped, already exists" in captured.out


@pytest.mark.integration
class TestChatGPTDeleteConversation:
    """Tests for delete_conversation_of_image_generation_uploaded_to_notion function."""

    async def test_delete_conversations_success(self, monkeypatch, capsys):
        """Should delete conversations for images in Notion."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-01T00:00:00",
                id="img_1",
                conversation_id="conv_1",
                message_id="msg_1",
                asset_pointer="asset_1",
                url="https://example.com/img1.png",
                prompt="test 1",
            ),
            ChatGPTImageGeneration(
                created_at="2024-01-01T00:00:00",
                id="img_2",
                conversation_id="conv_2",
                message_id="msg_2",
                asset_pointer="asset_2",
                url="https://example.com/img2.png",
                prompt="test 2",
            ),
        ]

        with patch(
            "chatgpt.is_page_exists_in_db", new_callable=AsyncMock
        ) as mock_exists:
            mock_exists.return_value = True

            with patch(
                "chatgpt.delete_conversation", new_callable=AsyncMock
            ) as mock_delete:
                await delete_conversation_of_image_generation_uploaded_to_notion(
                    generations, "test_db"
                )

                assert mock_exists.call_count == 2
                assert mock_delete.call_count == 2

    async def test_delete_conversations_skip_not_in_notion(self, monkeypatch, capsys):
        """Should skip conversations for images not in Notion."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-01T00:00:00",
                id="img_1",
                conversation_id="conv_1",
                message_id="msg_1",
                asset_pointer="asset_1",
                url="https://example.com/img1.png",
                prompt="test 1",
            )
        ]

        with patch(
            "chatgpt.is_page_exists_in_db", new_callable=AsyncMock
        ) as mock_exists:
            mock_exists.return_value = False

            with patch(
                "chatgpt.delete_conversation", new_callable=AsyncMock
            ) as mock_delete:
                await delete_conversation_of_image_generation_uploaded_to_notion(
                    generations, "test_db"
                )

                mock_exists.assert_called_once()
                mock_delete.assert_not_called()
                captured = capsys.readouterr()
                assert "not found in Notion, skipped" in captured.out

    async def test_delete_conversations_deduplicates(self, monkeypatch):
        """Should deduplicate by conversation_id."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        # Two images from same conversation
        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-01T00:00:00",
                id="img_1",
                conversation_id="conv_1",
                message_id="msg_1",
                asset_pointer="asset_1",
                url="https://example.com/img1.png",
                prompt="test 1",
            ),
            ChatGPTImageGeneration(
                created_at="2024-01-01T00:00:00",
                id="img_2",
                conversation_id="conv_1",
                message_id="msg_2",
                asset_pointer="asset_2",
                url="https://example.com/img2.png",
                prompt="test 2",
            ),
        ]

        with patch(
            "chatgpt.is_page_exists_in_db", new_callable=AsyncMock
        ) as mock_exists:
            mock_exists.return_value = True

            with patch(
                "chatgpt.delete_conversation", new_callable=AsyncMock
            ) as mock_delete:
                await delete_conversation_of_image_generation_uploaded_to_notion(
                    generations, "test_db"
                )

                # Should only delete once per conversation
                assert mock_delete.call_count == 1


@pytest.mark.integration
class TestChatGPTUploadToNotionComprehensive:
    """Comprehensive tests for upload_to_notion function."""

    async def test_upload_to_notion_full_workflow(self, monkeypatch, tmp_path):
        """Should execute full upload workflow."""
        from unittest.mock import AsyncMock, patch

        image_folder = str(tmp_path / "images")

        with patch(
            "chatgpt.fetch_image_generations", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = [
                {
                    "id": "img_123",
                    "url": "https://example.com/image.png",
                    "prompt": "Test",
                }
            ]

            with patch("chatgpt.download_all_images", new_callable=AsyncMock):
                with patch("chatgpt.add_prompt_to_images"):
                    with patch(
                        "chatgpt.upload_all_images_to_notion", new_callable=AsyncMock
                    ):
                        with patch("chatgpt.save_generations") as mock_save:
                            await chatgpt.upload_to_notion(
                                image_folder=image_folder,
                                db_id="test_db",
                                upload_to_notion=True,
                                remove_in_chatgpt=False,
                                add_prompt_to_image=True,
                                account="test",
                                limit=5,
                            )

                            mock_fetch.assert_called_once()
                            mock_save.assert_called_once()

    async def test_upload_to_notion_from_history(self, monkeypatch, tmp_path):
        """Should use history data instead of fetching live generations."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        image_folder = str(tmp_path / "images")
        generations = [
            ChatGPTImageGeneration(
                created_at="2026-05-14T00:00:00+00:00",
                id="gen_123",
                conversation_id="conv_abc",
                message_id="msg_def",
                asset_pointer="asset_ghi",
                url="https://example.com/image.png",
                prompt="Test prompt",
            )
        ]

        with patch("chatgpt.load_image_generations") as mock_load:
            mock_load.return_value = generations
            with patch(
                "chatgpt.fetch_image_generations", new_callable=AsyncMock
            ) as mock_fetch:
                with patch("chatgpt.download_all_images", new_callable=AsyncMock):
                    with patch("chatgpt.add_prompt_to_images"):
                        with patch(
                            "chatgpt.upload_all_images_to_notion",
                            new_callable=AsyncMock,
                        ) as mock_upload:
                            with patch("chatgpt.save_generations") as mock_save:
                                await chatgpt.upload_to_notion(
                                    image_folder=image_folder,
                                    db_id="test_db",
                                    account="default",
                                    check_notion_api=True,
                                    from_history=True,
                                )

        mock_load.assert_called_once_with(
            account="default",
            include_uploaded=True,
            keep_days=None,
            timezone_name=None,
            options=None,
        )
        mock_fetch.assert_not_called()
        mock_save.assert_not_called()
        mock_upload.assert_called_once()

    async def test_upload_to_notion_with_remove(self, monkeypatch, tmp_path):
        """Should remove conversations when remove_in_chatgpt=True."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        image_folder = str(tmp_path / "images")
        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:30:00.000000+00:00",
                id="gen_123",
                conversation_id="conv_abc",
                message_id="msg_def",
                asset_pointer="asset_ghi",
                url="https://example.com/image.png",
                prompt="Test prompt",
            )
        ]

        with patch(
            "chatgpt.fetch_image_generations", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = generations

            with patch("chatgpt.download_all_images", new_callable=AsyncMock):
                with patch("chatgpt.add_prompt_to_images"):
                    with patch(
                        "chatgpt.upload_all_images_to_notion", new_callable=AsyncMock
                    ):
                        with patch(
                            "chatgpt.delete_conversation_of_image_generation_uploaded_to_notion",
                            new_callable=AsyncMock,
                        ) as mock_remove:
                            await chatgpt.upload_to_notion(
                                image_folder=image_folder,
                                db_id="test_db",
                                remove_in_chatgpt=True,
                            )

                            mock_remove.assert_called_once()

    async def test_upload_to_notion_skip_optional_steps(self, monkeypatch, tmp_path):
        """Should skip optional steps when flags are False."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        image_folder = str(tmp_path / "images")
        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:30:00.000000+00:00",
                id="gen_123",
                conversation_id="conv_abc",
                message_id="msg_def",
                asset_pointer="asset_ghi",
                url="https://example.com/image.png",
                prompt="Test prompt",
            )
        ]

        with patch(
            "chatgpt.fetch_image_generations", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = generations

            with patch(
                "chatgpt.download_all_images", new_callable=AsyncMock
            ) as mock_download:
                with patch("chatgpt.add_prompt_to_images") as mock_add_prompt:
                    with patch(
                        "chatgpt.upload_all_images_to_notion", new_callable=AsyncMock
                    ) as mock_upload:
                        await chatgpt.upload_to_notion(
                            image_folder=image_folder,
                            db_id="test_db",
                            upload_to_notion=False,
                            add_prompt_to_image=False,
                        )

                        # download_all_images is always called
                        mock_download.assert_called_once()
                        # But add_prompt_to_images and upload are skipped
                        mock_add_prompt.assert_not_called()
                        mock_upload.assert_not_called()


@pytest.mark.integration
class TestChatGPTUploadToNotionSingle:
    """Tests for upload_to_notion_single function."""

    async def test_upload_to_notion_single_full_workflow(
        self, monkeypatch, tmp_path
    ):
        """Should execute per-file upload workflow."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        image_folder = str(tmp_path / "images")
        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:30:00.000000+00:00",
                id="gen_123",
                conversation_id="conv_abc",
                message_id="msg_def",
                asset_pointer="asset_ghi",
                url="https://example.com/image.png",
                prompt="Test prompt",
            )
        ]

        with patch(
            "chatgpt.fetch_image_generations", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = generations

            with patch("chatgpt.get_uploaded_generation_ids") as mock_uploaded:
                mock_uploaded.return_value = set()

                with patch(
                    "chatgpt.is_page_exists_in_db", new_callable=AsyncMock
                ) as mock_exists:
                    mock_exists.return_value = False

                    with patch(
                        "chatgpt.download_image", new_callable=AsyncMock
                    ) as mock_download:
                        with patch("chatgpt.add_prompt_to_image_single") as mock_prompt:
                            with patch(
                                "chatgpt.add_page_to_db", new_callable=AsyncMock
                            ) as mock_upload:
                                with patch("chatgpt.mark_generations_uploaded"):
                                    await chatgpt.upload_to_notion_single(
                                        image_folder=image_folder,
                                        db_id="test_db",
                                        account="test_account",
                                        upload_to_notion=True,
                                        remove_in_chatgpt=False,
                                        add_prompt_to_image=True,
                                        limit=5,
                                    )

                                    mock_fetch.assert_called_once()
                                    mock_download.assert_called_once()
                                    mock_prompt.assert_called_once()
                                    mock_upload.assert_called_once()

    async def test_upload_to_notion_single_skips_uploaded(
        self, monkeypatch, tmp_path, capsys
    ):
        """Should skip files already marked uploaded in CSV."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        image_folder = str(tmp_path / "images")
        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:30:00.000000+00:00",
                id="gen_123",
                conversation_id="conv_abc",
                message_id="msg_def",
                asset_pointer="asset_ghi",
                url="https://example.com/image.png",
                prompt="Test prompt",
            )
        ]

        with patch(
            "chatgpt.fetch_image_generations", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = generations

            with patch("chatgpt.get_uploaded_generation_ids") as mock_uploaded:
                mock_uploaded.return_value = {"gen_123"}

                with patch(
                    "chatgpt.download_image", new_callable=AsyncMock
                ) as mock_download:
                    with patch("chatgpt.add_page_to_db", new_callable=AsyncMock):
                        await chatgpt.upload_to_notion_single(
                            image_folder=image_folder,
                            db_id="test_db",
                            account="test_account",
                            upload_to_notion=True,
                            remove_in_chatgpt=False,
                            add_prompt_to_image=True,
                            limit=5,
                        )

                        mock_download.assert_not_called()
                        captured = capsys.readouterr()
                        assert "skipped, already uploaded" in captured.out

    async def test_upload_to_notion_single_skips_in_notion(
        self, monkeypatch, tmp_path, capsys
    ):
        """Should skip files already in Notion."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        image_folder = str(tmp_path / "images")
        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:30:00.000000+00:00",
                id="gen_123",
                conversation_id="conv_abc",
                message_id="msg_def",
                asset_pointer="asset_ghi",
                url="https://example.com/image.png",
                prompt="Test prompt",
            )
        ]

        with patch(
            "chatgpt.fetch_image_generations", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = generations

            with patch("chatgpt.get_uploaded_generation_ids") as mock_uploaded:
                mock_uploaded.return_value = set()

                with patch(
                    "chatgpt.is_page_exists_in_db", new_callable=AsyncMock
                ) as mock_exists:
                    mock_exists.return_value = True

                    with patch(
                        "chatgpt.download_image", new_callable=AsyncMock
                    ) as mock_download:
                        with patch("chatgpt.add_page_to_db", new_callable=AsyncMock):
                            await chatgpt.upload_to_notion_single(
                                image_folder=image_folder,
                                db_id="test_db",
                                account="test_account",
                                upload_to_notion=True,
                                remove_in_chatgpt=False,
                                add_prompt_to_image=True,
                                limit=5,
                            )

                            mock_download.assert_not_called()
                            captured = capsys.readouterr()
                            assert "skipped, already in Notion" in captured.out

    async def test_upload_to_notion_single_deletes_only_when_all_uploaded(
        self, monkeypatch, tmp_path, capsys
    ):
        """Should not delete conversation if not all images uploaded."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        image_folder = str(tmp_path / "images")
        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:30:00.000000+00:00",
                id="gen_1",
                conversation_id="conv_abc",
                message_id="msg_1",
                asset_pointer="asset_1",
                url="https://example.com/1.png",
                prompt="Test 1",
            ),
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:31:00.000000+00:00",
                id="gen_2",
                conversation_id="conv_abc",
                message_id="msg_2",
                asset_pointer="asset_2",
                url="https://example.com/2.png",
                prompt="Test 2",
            ),
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:32:00.000000+00:00",
                id="gen_3",
                conversation_id="conv_abc",
                message_id="msg_3",
                asset_pointer="asset_3",
                url="https://example.com/3.png",
                prompt="Test 3",
            ),
        ]

        async def mock_upload_fail_for_gen3(*args, **kwargs):
            file_path = args[2] if len(args) > 2 else kwargs.get("file_path", "")
            if "gen_3" in file_path:
                raise Exception("Network error")
            return {"id": "page_123"}

        with patch(
            "chatgpt.fetch_image_generations", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = generations

            with patch("chatgpt.get_uploaded_generation_ids") as mock_uploaded:
                mock_uploaded.return_value = set()

                with patch(
                    "chatgpt.is_page_exists_in_db", new_callable=AsyncMock
                ) as mock_exists:
                    mock_exists.return_value = False

                    with patch("chatgpt.download_image", new_callable=AsyncMock):
                        with patch("chatgpt.add_prompt_to_image_single"):
                            with patch(
                                "chatgpt.add_page_to_db",
                                new_callable=AsyncMock,
                                side_effect=mock_upload_fail_for_gen3,
                            ):
                                with patch("chatgpt.mark_generations_uploaded"):
                                    with patch(
                                        "chatgpt.delete_conversation",
                                        new_callable=AsyncMock,
                                    ) as mock_delete:
                                        await chatgpt.upload_to_notion_single(
                                            image_folder=image_folder,
                                            db_id="test_db",
                                            account="test_account",
                                            upload_to_notion=True,
                                            remove_in_chatgpt=True,
                                            add_prompt_to_image=True,
                                            limit=5,
                                        )

                                        mock_delete.assert_not_called()
                                        captured = capsys.readouterr()
                                        assert "skipped" in captured.out

    async def test_upload_to_notion_single_resume_after_failure(
        self, monkeypatch, tmp_path
    ):
        """Should retry failed files on re-run."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        image_folder = str(tmp_path / "images")
        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:30:00.000000+00:00",
                id="gen_fail",
                conversation_id="conv_abc",
                message_id="msg_1",
                asset_pointer="asset_1",
                url="https://example.com/fail.png",
                prompt="Test",
            ),
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:31:00.000000+00:00",
                id="gen_ok",
                conversation_id="conv_def",
                message_id="msg_2",
                asset_pointer="asset_2",
                url="https://example.com/ok.png",
                prompt="Test",
            ),
        ]

        call_count = 0

        async def mock_upload(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if args[2] == str(tmp_path / "images" / "gen_fail.png"):
                raise Exception("Network error")
            return {"id": "page_123"}

        with patch(
            "chatgpt.fetch_image_generations", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = generations

            with patch("chatgpt.get_uploaded_generation_ids") as mock_uploaded:
                mock_uploaded.return_value = set()

                with patch(
                    "chatgpt.is_page_exists_in_db", new_callable=AsyncMock
                ) as mock_exists:
                    mock_exists.return_value = False

                    with patch("chatgpt.download_image", new_callable=AsyncMock):
                        with patch("chatgpt.add_prompt_to_image_single"):
                            with patch(
                                "chatgpt.add_page_to_db",
                                new_callable=AsyncMock,
                                side_effect=mock_upload,
                            ):
                                with patch("chatgpt.mark_generations_uploaded"):
                                    await chatgpt.upload_to_notion_single(
                                        image_folder=image_folder,
                                        db_id="test_db",
                                        account="test_account",
                                        upload_to_notion=True,
                                        remove_in_chatgpt=False,
                                        add_prompt_to_image=True,
                                        limit=5,
                                    )

                                    # gen_ok uploaded, gen_fail failed
                                    assert call_count == 2

    async def test_delete_conversation_no_dedup_bug(
        self, monkeypatch, capsys
    ):
        """Should delete conversation only when all images confirmed in Notion."""
        from unittest.mock import AsyncMock, patch

        from models import ChatGPTImageGeneration

        # Three images from same conversation
        generations = [
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:30:00.000000+00:00",
                id="img_1",
                conversation_id="conv_1",
                message_id="msg_1",
                asset_pointer="asset_1",
                url="https://example.com/1.png",
                prompt="test 1",
            ),
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:31:00.000000+00:00",
                id="img_2",
                conversation_id="conv_1",
                message_id="msg_2",
                asset_pointer="asset_2",
                url="https://example.com/2.png",
                prompt="test 2",
            ),
            ChatGPTImageGeneration(
                created_at="2024-01-15T10:32:00.000000+00:00",
                id="img_3",
                conversation_id="conv_1",
                message_id="msg_3",
                asset_pointer="asset_3",
                url="https://example.com/3.png",
                prompt="test 3",
            ),
        ]

        with patch(
            "chatgpt.is_page_exists_in_db", new_callable=AsyncMock
        ) as mock_exists:
            mock_exists.return_value = True

            with patch(
                "chatgpt.delete_conversation", new_callable=AsyncMock
            ) as mock_delete:
                await chatgpt.delete_conversation_of_image_generation_uploaded_to_notion(
                    generations, "test_db"
                )

                # Should delete once for the conversation
                assert mock_delete.call_count == 1
