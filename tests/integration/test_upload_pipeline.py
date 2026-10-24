"""
Integration tests for chatgpt.py using pure mocking.

No real API calls are made. All external dependencies are mocked.
"""

import tomllib

import pytest

from chatgpt_to_notion.adapters import sqlite_store as db
from chatgpt_to_notion.domain.models import AppConfig
from chatgpt_to_notion.services import upload_sequential as chatgpt_sequential
from chatgpt_to_notion.services import upload_pipeline as chatgpt
from chatgpt_to_notion.shared.verbosity import SIMPLE, VERBOSE, set_verbosity
from chatgpt_to_notion.services.upload_pipeline import (
    delete_conversation,
    delete_conversation_of_image_generation_uploaded_to_notion,
    delete_conversations_after_upload,
    download_all_images,
    get_conversation_details,
    get_conversation_mapping_key_by_asset_pointer,
    get_conversations,
    get_headers,
    get_image_generations,
    get_prompt_from_image_node_in_conversation,
    load_image_generations,
)
from chatgpt_to_notion.adapters.chatgpt_api import get_conversation_details_batch
from tests.conftest import make_mock_response


@pytest.fixture(autouse=True)
def _set_verbose_for_tests():
    """Set VERBOSE mode so per-file output assertions work."""
    set_verbosity(VERBOSE)
    yield
    set_verbosity(SIMPLE)


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

    def test_headers_do_not_require_cookie(self, tmp_path, monkeypatch):
        """Headers should not include a Cookie entry if not configured."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[shared]
user_agent = "TestAgent/1.0"

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."default@example.com"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
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

        with patch("chatgpt_to_notion.services.history_service.get_image_generations", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_data

            with patch(
                "chatgpt_to_notion.services.history_service.get_conversation_details", new_callable=AsyncMock
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
                    "chatgpt_to_notion.services.history_service.get_prompt_from_image_node_in_conversation"
                ) as mock_prompt:
                    mock_prompt.return_value = "Test prompt"

                    with patch("chatgpt_to_notion.services.history_service.get_headers") as mock_headers:
                        mock_headers.return_value = {"Authorization": "Bearer test"}

                        with patch(
                            "chatgpt_to_notion.adapters.chatgpt_api.get_conversation_details_batch", new_callable=AsyncMock
                        ) as mock_detail_batch:
                            mock_detail_batch.return_value = {
                                "conv_abc": {
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
                            }

                            remote_generations, result = await chatgpt.fetch_image_generations(
                                limit=5,
                                options=type("Options", (), {"account": "default@example.com"}),
                            )

                            assert isinstance(remote_generations, list)
                            assert isinstance(result, list)
                            assert len(result) == 1
                            assert result[0].id == "img_123"
                            assert result[0].prompt == "Test prompt"

                            mock_get.assert_called_once()
                            mock_detail_batch.assert_called_once()

    async def test_fetch_image_generations_existing_items_skip_details(
        self, mock_aiohttp_session, monkeypatch, isolated_db
    ):
        """Should skip fetching details for items already in DB."""
        from unittest.mock import AsyncMock, patch
        from chatgpt_to_notion.domain.models import ChatGPTImageGeneration

        gen = ChatGPTImageGeneration(
            created_at="2024-01-15T10:30:00.000000+00:00",
            id="img_123",
            conversation_id="conv_abc",
            message_id="msg_def",
            asset_pointer="asset_ghi",
            url="https://example.com/image.png",
            prompt="Existing prompt",
        )
        db.upsert_generations("default@example.com", [gen])

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

        with patch("chatgpt_to_notion.services.history_service.get_image_generations", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_data

            with patch(
                "chatgpt_to_notion.services.history_service.get_conversation_details", new_callable=AsyncMock
            ) as mock_detail:
                mock_detail.return_value = {}

                with patch("chatgpt_to_notion.services.history_service.get_headers") as mock_headers:
                    mock_headers.return_value = {"Authorization": "Bearer test"}

                    remote_generations, result = await chatgpt.fetch_image_generations(
                        limit=5,
                        options=type("Options", (), {"account": "default@example.com"})(),
                    )

                    assert len(remote_generations) == 1
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

        with patch("chatgpt_to_notion.services.history_service.get_image_generations", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"items": []}

            with patch("chatgpt_to_notion.services.history_service.get_headers") as mock_headers:
                mock_headers.return_value = {"Authorization": "Bearer test"}

                remote_generations, result = await chatgpt.fetch_image_generations(
                    limit=5,
                    options=type("Options", (), {"account": "default@example.com"})(),
                )

                assert remote_generations == []
                assert result == []


@pytest.mark.integration
class TestRemoveLibraryImages:
    """Tests for the _remove_library_images helper."""

    async def test_remove_library_images_uses_configured_queries(
        self, tmp_path, monkeypatch
    ):
        """Should delete per configured query when remove_all is not set."""
        from unittest.mock import AsyncMock, MagicMock, patch

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="
delete_library_queries = ["ocean", "city"]

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."default@example.com"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        remove_mock = AsyncMock()
        count_mock = AsyncMock(return_value=3)
        session_mock = MagicMock()
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "chatgpt_to_notion.services.upload_pipeline.aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.chatgpt_api.delete_library_images_by_query",
            remove_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.chatgpt_api.count_library_images",
            count_mock,
        ):
            await chatgpt._remove_library_images(None)

        assert remove_mock.call_count == 2
        assert remove_mock.await_args_list[0].kwargs["query"] == "ocean"
        assert remove_mock.await_args_list[1].kwargs["query"] == "city"
        count_mock.assert_awaited_once()

    async def test_remove_library_images_remove_all_flag(self, tmp_path, monkeypatch):
        """Should remove ALL images when remove_all=True, ignoring queries."""
        from unittest.mock import AsyncMock, MagicMock, patch

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="
delete_library_queries = ["ocean"]

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."default@example.com"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        remove_mock = AsyncMock()
        count_mock = AsyncMock(return_value=0)
        session_mock = MagicMock()
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "chatgpt_to_notion.services.upload_pipeline.aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.chatgpt_api.delete_library_images_by_query",
            remove_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.chatgpt_api.count_library_images",
            count_mock,
        ):
            await chatgpt._remove_library_images(None, remove_all=True)

        remove_mock.assert_awaited_once()
        assert remove_mock.await_args.kwargs["query"] is None
        count_mock.assert_awaited_once()

    async def test_remove_library_images_remove_all_from_config(
        self, tmp_path, monkeypatch
    ):
        """Should remove ALL images when config sets remove_all_library_images."""
        from unittest.mock import AsyncMock, MagicMock, patch

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="
remove_all_library_images = true

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."default@example.com"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        remove_mock = AsyncMock()
        count_mock = AsyncMock(return_value=0)
        session_mock = MagicMock()
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "chatgpt_to_notion.services.upload_pipeline.aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.chatgpt_api.delete_library_images_by_query",
            remove_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.chatgpt_api.count_library_images",
            count_mock,
        ):
            await chatgpt._remove_library_images(None)

        remove_mock.assert_awaited_once()
        assert remove_mock.await_args.kwargs["query"] is None
        count_mock.assert_awaited_once()

    async def test_remove_library_images_noop_when_nothing_configured(
        self, tmp_path, monkeypatch
    ):
        """Should do nothing when no queries and remove_all is not set."""
        from unittest.mock import AsyncMock, patch

        remove_mock = AsyncMock()
        count_mock = AsyncMock()

        with patch(
            "chatgpt_to_notion.services.upload_pipeline.aiohttp.ClientSession",
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.chatgpt_api.delete_library_images_by_query",
            remove_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.chatgpt_api.count_library_images",
            count_mock,
        ):
            await chatgpt._remove_library_images(None)

        remove_mock.assert_not_awaited()
        count_mock.assert_not_awaited()
@pytest.mark.integration
class TestDeleteConversation:
    """Tests for delete_conversation behavior in sequential upload."""

    async def test_delete_conversation_called_when_flag_true(
        self, tmp_path, monkeypatch, isolated_db
    ):
        """Should call delete_conversation when delete_conversation=True."""
        from unittest.mock import AsyncMock, MagicMock, patch

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."default@example.com"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        delete_conv_mock = AsyncMock()
        session_mock = MagicMock()
        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.__aenter__ = AsyncMock(return_value=response_mock)
        response_mock.__aexit__ = AsyncMock(return_value=False)
        response_mock.json = AsyncMock(return_value={})
        response_mock.text = AsyncMock(return_value="")
        session_mock.get = MagicMock(return_value=response_mock)
        session_mock.patch = MagicMock(return_value=response_mock)
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "chatgpt_to_notion.services.upload_pipeline.aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_image_generations",
            new_callable=AsyncMock,
        ) as mock_get_gen, patch(
            "chatgpt_to_notion.services.upload_pipeline.is_page_exists_in_db",
            new_callable=AsyncMock,
        ) as mock_notion, patch(
            "chatgpt_to_notion.services.upload_pipeline.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_page_to_db",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_uploaded_generation_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.adapters.sqlite_store.get_existing_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.services.history_service.mark_generations_uploaded",
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_conversation_details",
            new_callable=AsyncMock,
        ) as mock_conv_detail, patch(
            "chatgpt_to_notion.services.upload_pipeline.delete_conversation",
            delete_conv_mock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt, patch(
            "chatgpt_to_notion.services.upload_pipeline.get_headers"
        ) as mock_headers:
            mock_headers.return_value = {"Authorization": "Bearer test"}
            mock_get_gen.return_value = {
                "items": [
                    {
                        "id": "gen_1",
                        "message_id": "msg_1",
                        "asset_pointer": "asset_1",
                        "conversation_id": "conv_1",
                        "url": "https://example.com/img.png",
                        "prompt": "test",
                        "created_at": 1705315800,
                    }
                ]
            }
            mock_notion.return_value = False
            mock_conv_detail.return_value = {
                "mapping": {
                    "msg_1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Test prompt"]},
                        },
                        "parent": None,
                    }
                }
            }
            mock_prompt.return_value = "Test prompt"

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=True,
                account="default@example.com",
                options=None,
            )

        delete_conv_mock.assert_awaited_once()

    async def test_delete_conversation_not_called_when_flag_false(
        self, tmp_path, monkeypatch, isolated_db
    ):
        """Should NOT call delete_conversation when delete_conversation=False."""
        from unittest.mock import AsyncMock, MagicMock, patch

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."default@example.com"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        delete_conv_mock = AsyncMock()
        session_mock = MagicMock()
        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.__aenter__ = AsyncMock(return_value=response_mock)
        response_mock.__aexit__ = AsyncMock(return_value=False)
        response_mock.json = AsyncMock(return_value={})
        response_mock.text = AsyncMock(return_value="")
        session_mock.get = MagicMock(return_value=response_mock)
        session_mock.patch = MagicMock(return_value=response_mock)
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "chatgpt_to_notion.services.upload_pipeline.aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_image_generations",
            new_callable=AsyncMock,
        ) as mock_get_gen, patch(
            "chatgpt_to_notion.services.upload_pipeline.is_page_exists_in_db",
            new_callable=AsyncMock,
        ) as mock_notion, patch(
            "chatgpt_to_notion.services.upload_pipeline.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_page_to_db",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_uploaded_generation_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.adapters.sqlite_store.get_existing_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.services.history_service.mark_generations_uploaded",
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_conversation_details",
            new_callable=AsyncMock,
        ) as mock_conv_detail, patch(
            "chatgpt_to_notion.services.upload_pipeline.delete_conversation",
            delete_conv_mock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt, patch(
            "chatgpt_to_notion.services.upload_pipeline.get_headers"
        ) as mock_headers:
            mock_headers.return_value = {"Authorization": "Bearer test"}
            mock_get_gen.return_value = {
                "items": [
                    {
                        "id": "gen_1",
                        "message_id": "msg_1",
                        "asset_pointer": "asset_1",
                        "conversation_id": "conv_1",
                        "url": "https://example.com/img.png",
                        "prompt": "test",
                        "created_at": 1705315800,
                    }
                ]
            }
            mock_notion.return_value = False
            mock_conv_detail.return_value = {
                "mapping": {
                    "msg_1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Test prompt"]},
                        },
                        "parent": None,
                    }
                }
            }
            mock_prompt.return_value = "Test prompt"

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=False,
                account="default@example.com",
                options=None,
            )

        delete_conv_mock.assert_not_awaited()

    async def test_delete_conversation_not_called_in_from_history_mode(
        self, tmp_path, monkeypatch, isolated_db
    ):
        """Should NOT call delete_conversation when from_history=True even if flag is True."""
        from unittest.mock import AsyncMock, MagicMock, patch

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."default@example.com"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        delete_conv_mock = AsyncMock()
        session_mock = MagicMock()
        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.__aenter__ = AsyncMock(return_value=response_mock)
        response_mock.__aexit__ = AsyncMock(return_value=False)
        response_mock.json = AsyncMock(return_value={})
        response_mock.text = AsyncMock(return_value="")
        session_mock.get = MagicMock(return_value=response_mock)
        session_mock.patch = MagicMock(return_value=response_mock)
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=False)

        mock_gen = MagicMock()
        mock_gen.id = "gen_1"
        mock_gen.conversation_id = "conv_1"
        mock_gen.url = "https://example.com/img.png"
        mock_gen.prompt = "test"

        with patch(
            "chatgpt_to_notion.services.upload_pipeline.aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.is_page_exists_in_db",
            new_callable=AsyncMock,
        ) as mock_notion, patch(
            "chatgpt_to_notion.services.upload_pipeline.history_service.load_image_generations",
            new_callable=AsyncMock,
        ) as mock_load_gen, patch(
            "chatgpt_to_notion.services.upload_pipeline.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_page_to_db",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.delete_conversation",
            delete_conv_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_headers"
        ) as mock_headers:
            mock_headers.return_value = {"Authorization": "Bearer test"}
            mock_load_gen.return_value = [mock_gen]
            mock_notion.return_value = False

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=True,
                from_history=True,
                account="default@example.com",
                options=None,
            )

        delete_conv_mock.assert_not_awaited()


@pytest.mark.integration
class TestDeleteLibrary:
    """Tests for delete_library and delete_library_all behavior in sequential upload."""

    async def test_delete_library_called_when_flag_true(
        self, tmp_path, monkeypatch, isolated_db
    ):
        """Should call delete_library_images_by_query when delete_library=True."""
        from unittest.mock import AsyncMock, MagicMock, patch

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="
delete_library_queries = ["test query"]

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."default@example.com"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        remove_lib_mock = AsyncMock()
        count_lib_mock = AsyncMock(return_value=0)
        session_mock = MagicMock()
        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.__aenter__ = AsyncMock(return_value=response_mock)
        response_mock.__aexit__ = AsyncMock(return_value=False)
        response_mock.json = AsyncMock(return_value={})
        response_mock.text = AsyncMock(return_value="")
        session_mock.get = MagicMock(return_value=response_mock)
        session_mock.patch = MagicMock(return_value=response_mock)
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "chatgpt_to_notion.services.upload_pipeline.aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_image_generations",
            new_callable=AsyncMock,
        ) as mock_get_gen, patch(
            "chatgpt_to_notion.services.upload_pipeline.is_page_exists_in_db",
            new_callable=AsyncMock,
        ) as mock_notion, patch(
            "chatgpt_to_notion.services.upload_pipeline.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_page_to_db",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_uploaded_generation_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.adapters.sqlite_store.get_existing_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.services.history_service.mark_generations_uploaded",
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_conversation_details",
            new_callable=AsyncMock,
        ) as mock_conv_detail, patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt, patch(
            "chatgpt_to_notion.adapters.chatgpt_api.delete_library_images_by_query",
            remove_lib_mock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.count_library_images",
            count_lib_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_headers"
        ) as mock_headers:
            mock_headers.return_value = {"Authorization": "Bearer test"}
            mock_get_gen.return_value = {
                "items": [
                    {
                        "id": "gen_1",
                        "message_id": "msg_1",
                        "asset_pointer": "asset_1",
                        "conversation_id": "conv_1",
                        "url": "https://example.com/img.png",
                        "prompt": "test",
                        "created_at": 1705315800,
                    }
                ]
            }
            mock_notion.return_value = False
            mock_conv_detail.return_value = {
                "mapping": {
                    "msg_1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Test prompt"]},
                        },
                        "parent": None,
                    }
                }
            }
            mock_prompt.return_value = "Test prompt"

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_library=True,
                account="default@example.com",
                options=None,
            )

        remove_lib_mock.assert_awaited_once()
        count_lib_mock.assert_awaited_once()

    async def test_delete_library_not_called_when_flag_false(
        self, tmp_path, monkeypatch, isolated_db
    ):
        """Should NOT call delete_library_images_by_query when delete_library=False."""
        from unittest.mock import AsyncMock, MagicMock, patch

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="
delete_library_queries = ["test query"]

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."default@example.com"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        remove_lib_mock = AsyncMock()
        count_lib_mock = AsyncMock(return_value=0)
        session_mock = MagicMock()
        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.__aenter__ = AsyncMock(return_value=response_mock)
        response_mock.__aexit__ = AsyncMock(return_value=False)
        response_mock.json = AsyncMock(return_value={})
        response_mock.text = AsyncMock(return_value="")
        session_mock.get = MagicMock(return_value=response_mock)
        session_mock.patch = MagicMock(return_value=response_mock)
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "chatgpt_to_notion.services.upload_pipeline.aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_image_generations",
            new_callable=AsyncMock,
        ) as mock_get_gen, patch(
            "chatgpt_to_notion.services.upload_pipeline.is_page_exists_in_db",
            new_callable=AsyncMock,
        ) as mock_notion, patch(
            "chatgpt_to_notion.services.upload_pipeline.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_page_to_db",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_uploaded_generation_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.adapters.sqlite_store.get_existing_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.services.history_service.mark_generations_uploaded",
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_conversation_details",
            new_callable=AsyncMock,
        ) as mock_conv_detail, patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt, patch(
            "chatgpt_to_notion.adapters.chatgpt_api.delete_library_images_by_query",
            remove_lib_mock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.count_library_images",
            count_lib_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_headers"
        ) as mock_headers:
            mock_headers.return_value = {"Authorization": "Bearer test"}
            mock_get_gen.return_value = {
                "items": [
                    {
                        "id": "gen_1",
                        "message_id": "msg_1",
                        "asset_pointer": "asset_1",
                        "conversation_id": "conv_1",
                        "url": "https://example.com/img.png",
                        "prompt": "test",
                        "created_at": 1705315800,
                    }
                ]
            }
            mock_notion.return_value = False
            mock_conv_detail.return_value = {
                "mapping": {
                    "msg_1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Test prompt"]},
                        },
                        "parent": None,
                    }
                }
            }
            mock_prompt.return_value = "Test prompt"

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_library=False,
                account="default@example.com",
                options=None,
            )

        remove_lib_mock.assert_not_awaited()
        count_lib_mock.assert_not_awaited()

    async def test_delete_library_all_called_when_flag_true(
        self, tmp_path, monkeypatch, isolated_db
    ):
        """Should call delete_library_images_by_query with query=None when delete_library_all=True."""
        from unittest.mock import AsyncMock, MagicMock, patch

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."default@example.com"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        remove_lib_mock = AsyncMock()
        count_lib_mock = AsyncMock(return_value=5)
        session_mock = MagicMock()
        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.__aenter__ = AsyncMock(return_value=response_mock)
        response_mock.__aexit__ = AsyncMock(return_value=False)
        response_mock.json = AsyncMock(return_value={})
        response_mock.text = AsyncMock(return_value="")
        session_mock.get = MagicMock(return_value=response_mock)
        session_mock.patch = MagicMock(return_value=response_mock)
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "chatgpt_to_notion.services.upload_pipeline.aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_image_generations",
            new_callable=AsyncMock,
        ) as mock_get_gen, patch(
            "chatgpt_to_notion.services.upload_pipeline.is_page_exists_in_db",
            new_callable=AsyncMock,
        ) as mock_notion, patch(
            "chatgpt_to_notion.services.upload_pipeline.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_page_to_db",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_uploaded_generation_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.adapters.sqlite_store.get_existing_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.services.history_service.mark_generations_uploaded",
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_conversation_details",
            new_callable=AsyncMock,
        ) as mock_conv_detail, patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt, patch(
            "chatgpt_to_notion.adapters.chatgpt_api.delete_library_images_by_query",
            remove_lib_mock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.count_library_images",
            count_lib_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_headers"
        ) as mock_headers:
            mock_headers.return_value = {"Authorization": "Bearer test"}
            mock_get_gen.return_value = {
                "items": [
                    {
                        "id": "gen_1",
                        "message_id": "msg_1",
                        "asset_pointer": "asset_1",
                        "conversation_id": "conv_1",
                        "url": "https://example.com/img.png",
                        "prompt": "test",
                        "created_at": 1705315800,
                    }
                ]
            }
            mock_notion.return_value = False
            mock_conv_detail.return_value = {
                "mapping": {
                    "msg_1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Test prompt"]},
                        },
                        "parent": None,
                    }
                }
            }
            mock_prompt.return_value = "Test prompt"

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_library_all=True,
                account="default@example.com",
                options=None,
            )

        remove_lib_mock.assert_awaited_once()
        count_lib_mock.assert_awaited_once()

    async def test_delete_library_called_in_from_history_mode(
        self, tmp_path, monkeypatch, isolated_db
    ):
        """Should call delete_library_images_by_query when from_history=True and flag is True."""
        from unittest.mock import AsyncMock, MagicMock, patch

        config_file = tmp_path / "config.toml"
        config_file.write_text(
            """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="
delete_library_queries = ["test query"]

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."default@example.com"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        remove_lib_mock = AsyncMock()
        count_lib_mock = AsyncMock(return_value=0)
        session_mock = MagicMock()
        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.__aenter__ = AsyncMock(return_value=response_mock)
        response_mock.__aexit__ = AsyncMock(return_value=False)
        response_mock.json = AsyncMock(return_value={})
        response_mock.text = AsyncMock(return_value="")
        session_mock.get = MagicMock(return_value=response_mock)
        session_mock.patch = MagicMock(return_value=response_mock)
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=False)

        mock_gen = MagicMock()
        mock_gen.id = "gen_1"
        mock_gen.conversation_id = "conv_1"
        mock_gen.url = "https://example.com/img.png"
        mock_gen.prompt = "test"

        with patch(
            "chatgpt_to_notion.services.upload_pipeline.aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.is_page_exists_in_db",
            new_callable=AsyncMock,
        ) as mock_notion, patch(
            "chatgpt_to_notion.services.upload_pipeline.history_service.load_image_generations",
            new_callable=AsyncMock,
        ) as mock_load_gen, patch(
            "chatgpt_to_notion.services.upload_pipeline.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.add_page_to_db",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.delete_library_images_by_query",
            remove_lib_mock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.count_library_images",
            count_lib_mock,
        ), patch(
            "chatgpt_to_notion.services.upload_pipeline.get_headers"
        ) as mock_headers:
            mock_headers.return_value = {"Authorization": "Bearer test"}
            mock_load_gen.return_value = [mock_gen]
            mock_notion.return_value = False

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_library=True,
                from_history=True,
                account="default@example.com",
                options=None,
            )

        remove_lib_mock.assert_awaited_once()
        count_lib_mock.assert_awaited_once()