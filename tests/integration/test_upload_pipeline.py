"""
Integration tests for chatgpt.py using pure mocking.

No real API calls are made. All external dependencies are mocked.
"""
from unittest.mock import AsyncMock, MagicMock, patch
import tomllib

import pytest

from chatgpt_to_notion.adapters.sqlite_store import SqliteStore
from chatgpt_to_notion.adapters.chatgpt_api import ChatGPTClientImpl
from chatgpt_to_notion.adapters.config import Config, select_account
from chatgpt_to_notion.domain.models import AppConfig, RuntimeOptions
from chatgpt_to_notion.services import upload_sequential as chatgpt_sequential
from chatgpt_to_notion.shared.verbosity import SIMPLE, VERBOSE, set_verbosity
from chatgpt_to_notion.services.history_service import load_image_generations
from tests.conftest import make_mock_response

db = SqliteStore()


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
        headers = ChatGPTClientImpl(
            select_account(Config(), RuntimeOptions()).provider_context()
        ).get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    def test_headers_contain_user_agent(self, mock_config_toml):
        """Headers should contain User-Agent."""
        headers = ChatGPTClientImpl(
            select_account(Config(), RuntimeOptions()).provider_context()
        ).get_headers()
        assert "User-Agent" in headers

    def test_headers_contain_content_type(self, mock_config_toml):
        """Headers should contain Content-Type."""
        headers = ChatGPTClientImpl(
            select_account(Config(), RuntimeOptions()).provider_context()
        ).get_headers()
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
        headers = ChatGPTClientImpl(
            select_account(Config(), RuntimeOptions()).provider_context()
        ).get_headers()
        assert isinstance(headers, dict)
        assert "Authorization" in headers
        assert "User-Agent" in headers
        assert "Content-Type" in headers
        assert "Cookie" not in headers


@pytest.mark.integration
class TestRemoveLibraryImages:
    """Tests for library-image removal routed through the chatgpt client."""

    @pytest.mark.asyncio
    async def test_remove_library_images_uses_configured_queries(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service
    ):
        """Should call remove_library_func with configured queries when delete_library=True."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig
        from chatgpt_to_notion.adapters.config import _load_config

        account = "default@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="
delete_library_queries = ["ocean", "city"]

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."{account}"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config._load_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        mock_chatgpt_client.get_image_generations.return_value = {"items": []}

        from chatgpt_to_notion.services import upload_sequential

        # Pass delete_library=True to trigger the callback
        await upload_sequential.upload_to_notion_sequential(
            image_folder=None,
            db_id="test_db",
            delete_library=True,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
        )

        # Client method should be called once (with remove_all=False)
        # The queries are in options
        mock_chatgpt_client.remove_library_images.assert_awaited_once()
        _, kwargs = mock_chatgpt_client.remove_library_images.call_args
        assert kwargs.get("remove_all") is False

    @pytest.mark.asyncio
    async def test_remove_library_images_default_uses_client_method(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service
    ):
        """Default removal should route through the chatgpt client's method."""
        import tomllib
        from unittest.mock import AsyncMock
        from chatgpt_to_notion.domain.models import AppConfig
        from chatgpt_to_notion.adapters.config import _load_config

        account = "default@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."{account}"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config._load_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        mock_chatgpt_client.get_image_generations.return_value = {"items": []}
        mock_chatgpt_client.remove_library_images = AsyncMock(return_value=None)

        from chatgpt_to_notion.services import upload_sequential

        await upload_sequential.upload_to_notion_sequential(
            image_folder=None,
            db_id="test_db",
            delete_library=True,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
        )

        mock_chatgpt_client.remove_library_images.assert_called_once()
        _, kwargs = mock_chatgpt_client.remove_library_images.call_args
        assert kwargs.get("remove_all") is False

    @pytest.mark.asyncio
    async def test_remove_library_images_remove_all_flag(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service
    ):
        """Should call remove_library_func with remove_all=True when flag is set."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig
        from chatgpt_to_notion.adapters.config import _load_config

        account = "default@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="
delete_library_queries = ["ocean"]

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."{account}"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config._load_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        mock_chatgpt_client.get_image_generations.return_value = {"items": []}

        from chatgpt_to_notion.services import upload_sequential

        await upload_sequential.upload_to_notion_sequential(
            image_folder=None,
            db_id="test_db",
            delete_library_all=True,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
        )

        # Client method should be called once with remove_all=True
        mock_chatgpt_client.remove_library_images.assert_awaited_once()
        _, kwargs = mock_chatgpt_client.remove_library_images.call_args
        assert kwargs.get("remove_all") is True

    @pytest.mark.asyncio
    async def test_remove_library_images_remove_all_from_config(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service
    ):
        """Should call remove_library_func with remove_all=True when delete_library_all=True."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig
        from chatgpt_to_notion.adapters.config import _load_config

        account = "default@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."{account}"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config._load_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        mock_chatgpt_client.get_image_generations.return_value = {"items": []}

        from chatgpt_to_notion.services import upload_sequential

        # Pass delete_library_all=True to trigger the callback with remove_all=True
        await upload_sequential.upload_to_notion_sequential(
            image_folder=None,
            db_id="test_db",
            delete_library_all=True,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
        )

        # Client method should be called once with remove_all=True
        mock_chatgpt_client.remove_library_images.assert_awaited_once()
        _, kwargs = mock_chatgpt_client.remove_library_images.call_args
        assert kwargs.get("remove_all") is True

    @pytest.mark.asyncio
    async def test_remove_library_images_noop_when_nothing_configured(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service
    ):
        """Should not call client.remove_library_images when nothing is configured."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig
        from chatgpt_to_notion.adapters.config import _load_config

        account = "default@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."{account}"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config._load_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        mock_chatgpt_client.get_image_generations.return_value = {"items": []}

        from chatgpt_to_notion.services import upload_sequential

        await upload_sequential.upload_to_notion_sequential(
            image_folder=None,
            db_id="test_db",
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
        )

        # Client method should not be called
        mock_chatgpt_client.remove_library_images.assert_not_awaited()

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
            "chatgpt_to_notion.adapters.config._load_config",
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

        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_chat.get_image_generations = AsyncMock(
            return_value={
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
        )
        fake_chat.get_conversation_details = AsyncMock(
            return_value={
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
        )
        fake_chat.delete_conversation = delete_conv_mock
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock()

        with patch(
            "aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.image_service.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.history_service.get_uploaded_generation_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.services.history_service.mark_generations_uploaded",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt:
            mock_prompt.return_value = "Test prompt"

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=True,
                account="default@example.com",
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
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
            "chatgpt_to_notion.adapters.config._load_config",
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

        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_chat.get_image_generations = AsyncMock(
            return_value={
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
        )
        fake_chat.get_conversation_details = AsyncMock(
            return_value={
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
        )
        fake_chat.delete_conversation = delete_conv_mock
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock()

        with patch(
            "aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.image_service.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.history_service.get_uploaded_generation_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.services.history_service.mark_generations_uploaded",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt:
            mock_prompt.return_value = "Test prompt"

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=False,
                account="default@example.com",
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
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
            "chatgpt_to_notion.adapters.config._load_config",
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

        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_chat.delete_conversation = delete_conv_mock
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock()

        with patch(
            "aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.history_service.load_image_generations",
            new_callable=AsyncMock,
        ) as mock_load_gen, patch(
            "chatgpt_to_notion.services.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.image_service.add_prompt_to_image_single",
        ):
            mock_load_gen.return_value = [mock_gen]

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=True,
                from_history=True,
                account="default@example.com",
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
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
            "chatgpt_to_notion.adapters.config._load_config",
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

        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_chat.get_image_generations = AsyncMock(
            return_value={
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
        )
        fake_chat.get_conversation_details = AsyncMock(
            return_value={
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
        )
        fake_chat.remove_library_images = remove_lib_mock
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock()

        with patch(
            "aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.image_service.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.history_service.get_uploaded_generation_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.services.history_service.mark_generations_uploaded",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt:
            mock_prompt.return_value = "Test prompt"

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_library=True,
                account="default@example.com",
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
                options=None,
            )

        remove_lib_mock.assert_awaited_once()
        count_lib_mock.assert_not_awaited()

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
            "chatgpt_to_notion.adapters.config._load_config",
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

        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_chat.get_image_generations = AsyncMock(
            return_value={
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
        )
        fake_chat.get_conversation_details = AsyncMock(
            return_value={
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
        )
        fake_chat.remove_library_images = remove_lib_mock
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock()

        with patch(
            "aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.image_service.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.history_service.get_uploaded_generation_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.services.history_service.mark_generations_uploaded",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt:
            mock_prompt.return_value = "Test prompt"

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_library=False,
                account="default@example.com",
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
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
            "chatgpt_to_notion.adapters.config._load_config",
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

        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_chat.get_image_generations = AsyncMock(
            return_value={
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
        )
        fake_chat.get_conversation_details = AsyncMock(
            return_value={
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
        )
        fake_chat.remove_library_images = remove_lib_mock
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock()

        with patch(
            "aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.image_service.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.history_service.get_uploaded_generation_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.services.history_service.mark_generations_uploaded",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt:
            mock_prompt.return_value = "Test prompt"

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_library_all=True,
                account="default@example.com",
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
                options=None,
            )

        remove_lib_mock.assert_awaited_once()
        count_lib_mock.assert_not_awaited()

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
            "chatgpt_to_notion.adapters.config._load_config",
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

        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_chat.remove_library_images = remove_lib_mock
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock()

        with patch(
            "aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.history_service.load_image_generations",
            new_callable=AsyncMock,
        ) as mock_load_gen, patch(
            "chatgpt_to_notion.services.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.image_service.add_prompt_to_image_single",
        ):
            mock_load_gen.return_value = [mock_gen]

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_library=True,
                from_history=True,
                account="default@example.com",
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
                options=None,
            )

        remove_lib_mock.assert_awaited_once()
        count_lib_mock.assert_not_awaited()
@pytest.mark.integration
class TestUploadToAccountStatus:
    """End-to-end tests verifying upload-to-notion saves history to account-status."""

    async def test_from_history_upload_marks_generations_uploaded(
        self, tmp_path, monkeypatch, isolated_db
    ):
        """Full flow: load from history -> upload -> mark uploaded -> account-status sees it."""
        db = SqliteStore()
        from chatgpt_to_notion.domain.models import ChatGPTImageGeneration
        from datetime import datetime, timezone
        from chatgpt_to_notion.services import history_service
        from chatgpt_to_notion.services.account_status_service import get_account_activity_statuses
        from chatgpt_to_notion.services import upload_sequential

        account = "default@example.com"

        # Setup: insert generations directly into DB (simulating prior fetch)
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        gen1 = ChatGPTImageGeneration(
            created_at=now,
            id="gen_hist_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="history prompt 1",
        )
        gen2 = ChatGPTImageGeneration(
            created_at=now,
            id="gen_hist_2",
            conversation_id="conv_2",
            message_id="msg_2",
            asset_pointer="asset_2",
            url="https://example.com/img2.png",
            prompt="history prompt 2",
        )
        db.upsert_generations(account, [gen1, gen2])

        # Verify initially not uploaded
        uploaded_before = db.get_uploaded_ids(account)
        assert uploaded_before == set()

        # Mock external calls (Notion, image download)
        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock()

        with patch("chatgpt_to_notion.services.history_service.download_image", new_callable=AsyncMock) as mock_download, \
             patch("chatgpt_to_notion.services.image_service.add_prompt_to_image_single") as mock_add_prompt:


            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=False,
                from_history=True,
                account=account,
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
                options=None,
            )

        # Verify generations were marked as uploaded in DB
        uploaded_after = db.get_uploaded_ids(account)
        assert "gen_hist_1" in uploaded_after
        assert "gen_hist_2" in uploaded_after

        # Verify account-status can see today's activity (based on created_at)
        # Since both generations created at "now", they are active (within 24h)
        today_rows, yesterday_rows, total = get_account_activity_statuses(timezone_name=None)
        assert len(today_rows) == 1
        assert today_rows[0]["Account"] == account
        assert today_rows[0]["Service"] == "chatgpt"
        # Both are active (created now), so shows "2/2 to wait"
        assert "2/2 to wait" in today_rows[0]["Ready Generate?"]

    async def test_api_upload_creates_and_marks_generations(
        self, tmp_path, monkeypatch, isolated_db
    ):
        """Full flow: fetch from API -> save to DB -> upload -> mark uploaded -> account-status sees it."""
        db = SqliteStore()
        from chatgpt_to_notion.services.account_status_service import get_account_activity_statuses
        from chatgpt_to_notion.services import upload_sequential

        # Create config.toml with test account so account-status can find it
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

        account = "default@example.com"

        # Mock all external API calls
        from datetime import datetime, timezone
        recent_ts = int(datetime.now(timezone.utc).timestamp())
        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_chat.get_image_generations = AsyncMock(
            return_value={
                "items": [
                    {
                        "id": "gen_api_1",
                        "message_id": "msg_1",
                        "asset_pointer": "asset_1",
                        "conversation_id": "conv_1",
                        "url": "https://example.com/img.png",
                        "prompt": "api prompt",
                        "created_at": recent_ts,
                    }
                ]
            }
        )
        fake_chat.get_conversation_details = AsyncMock(
            return_value={
                "mapping": {
                    "msg_1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["api prompt"]},
                        },
                        "parent": None,
                    }
                }
            }
        )
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock()

        with patch("chatgpt_to_notion.services.history_service.download_image", new_callable=AsyncMock) as mock_download, \
             patch("chatgpt_to_notion.services.image_service.add_prompt_to_image_single") as mock_add_prompt:


            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=False,
                from_history=False,
                account=account,
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
                options=None,
            )

        # Verify generation was saved to DB and marked as uploaded
        uploaded_after = db.get_uploaded_ids(account)
        assert "gen_api_1" in uploaded_after

        # Verify account-status can see today's activity
        # Generation created_at is recent (within 24h), so it's active
        today_rows, yesterday_rows, total = get_account_activity_statuses(timezone_name=None)
        assert len(today_rows) == 1
        assert today_rows[0]["Account"] == account
        assert today_rows[0]["Service"] == "chatgpt"
        # Active (recent created_at), shows "1/1 to wait"
        assert "1/1 to wait" in today_rows[0]["Ready Generate?"]
    async def test_verify_history_checks_notion_and_updates_local(
        self, tmp_path, monkeypatch, isolated_db
    ):
        """verify_history flag loads all (including uploaded) and syncs with Notion."""
        db = SqliteStore()
        from chatgpt_to_notion.domain.models import ChatGPTImageGeneration
        from datetime import datetime, timezone
        from chatgpt_to_notion.services import history_service
        from chatgpt_to_notion.services import upload_sequential

        account = "default@example.com"

        # Setup: generation already marked uploaded locally
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        gen = ChatGPTImageGeneration(
            created_at=now,
            id="gen_verify_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img.png",
            prompt="verify prompt",
        )
        db.upsert_generations(account, [gen])
        await history_service.mark_generations_uploaded(account, {"gen_verify_1"})

        # Verify it's already uploaded locally
        uploaded_before = db.get_uploaded_ids(account)
        assert "gen_verify_1" in uploaded_before

        # Mock Notion to say page exists
        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=True)
        fake_notion.add_page_to_db = AsyncMock()

        with patch("chatgpt_to_notion.services.history_service.load_image_generations") as mock_load_gen:

            # Create mock generation for load_image_generations to return
            mock_gen = MagicMock()
            mock_gen.id = "gen_verify_1"
            mock_gen.conversation_id = "conv_1"
            mock_gen.url = "https://example.com/img.png"
            mock_gen.prompt = "verify prompt"
            mock_load_gen.return_value = [mock_gen]

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=False,
                from_history=True,
                check_notion_api=True,  # verify_history sets both
                account=account,
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
                options=None,
            )

        # Should still be marked uploaded (no change)
        uploaded_after = db.get_uploaded_ids(account)
        assert "gen_verify_1" in uploaded_after

    async def test_check_notion_api_false_skips_already_uploaded(
        self, tmp_path, monkeypatch, isolated_db
    ):
        """With check_notion_api=False (default), already-uploaded generations are skipped."""
        db = SqliteStore()
        from chatgpt_to_notion.domain.models import ChatGPTImageGeneration
        from datetime import datetime, timezone
        from chatgpt_to_notion.services import history_service
        from chatgpt_to_notion.services import upload_sequential

        account = "default@example.com"

        # Setup: generation already marked uploaded locally
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        gen = ChatGPTImageGeneration(
            created_at=now,
            id="gen_skip_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img.png",
            prompt="skip prompt",
        )
        db.upsert_generations(account, [gen])
        await history_service.mark_generations_uploaded(account, {"gen_skip_1"})

        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock()

        with patch("chatgpt_to_notion.services.history_service.load_image_generations") as mock_load_gen:

            # load_image_generations should NOT return already-uploaded when check_notion_api=False
            mock_load_gen.return_value = []  # Empty because gen_skip_1 is already uploaded

            # load_image_generations should NOT return already-uploaded when check_notion_api=False
            mock_load_gen.return_value = []  # Empty because gen_skip_1 is already uploaded

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=False,
                from_history=True,
                check_notion_api=False,  # default
                account=account,
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
                options=None,
            )

        # load_image_generations should have been called with include_uploaded=False
        mock_load_gen.assert_called_once()
        args, kwargs = mock_load_gen.call_args
        assert kwargs.get("include_uploaded") is False


@pytest.mark.integration
class TestImageProcessingOrder:
    """Tests for image processing order within a conversation."""

    async def test_api_mode_images_processed_oldest_first(
        self, tmp_path, monkeypatch, isolated_db
    ):
        """In API mode, images within a conversation should be processed oldest-first.

        The API returns items newest-first, but upload should process oldest-first.
        """
        from unittest.mock import AsyncMock, MagicMock, patch
        from chatgpt_to_notion.domain.models import AppConfig
        import tomllib

        account = "order@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

[accounts."{account}"]
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config._load_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        # API returns items newest-first (typical API behavior)
        # created_at timestamps: gen_3 (newest), gen_2, gen_1 (oldest)
        api_items = [
            {
                "id": "gen_3",
                "message_id": "msg_3",
                "asset_pointer": "asset_3",
                "conversation_id": "conv_1",
                "url": "https://example.com/img3.png",
                "prompt": "prompt 3",
                "created_at": 1705316000,  # newest
            },
            {
                "id": "gen_2",
                "message_id": "msg_2",
                "asset_pointer": "asset_2",
                "conversation_id": "conv_1",
                "url": "https://example.com/img2.png",
                "prompt": "prompt 2",
                "created_at": 1705315900,  # middle
            },
            {
                "id": "gen_1",
                "message_id": "msg_1",
                "asset_pointer": "asset_1",
                "conversation_id": "conv_1",
                "url": "https://example.com/img1.png",
                "prompt": "prompt 1",
                "created_at": 1705315800,  # oldest
            },
        ]

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

        mark_uploaded_mock = MagicMock()
        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_chat.get_image_generations = AsyncMock(return_value={"items": api_items})
        fake_chat.get_conversation_details = AsyncMock(
            return_value={
                "mapping": {
                    "msg_1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Prompt 1"]},
                        },
                        "parent": None,
                    },
                    "msg_2": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Prompt 2"]},
                        },
                        "parent": None,
                    },
                    "msg_3": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Prompt 3"]},
                        },
                        "parent": None,
                    },
                }
            }
        )
        fake_chat.delete_conversation = AsyncMock()
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock()

        with patch(
            "aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.image_service.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.history_service.get_uploaded_generation_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.services.history_service.mark_generations_uploaded",
            mark_uploaded_mock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt:
            mock_prompt.side_effect = ["Prompt 1", "Prompt 2", "Prompt 3"]

            from chatgpt_to_notion.services import upload_sequential

            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=False,
                account=account,
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
                options=None,
            )

            # Verify mark_generations_uploaded was called in oldest-first order: gen_1, gen_2, gen_3
            calls = mark_uploaded_mock.call_args_list
            assert len(calls) == 3
            processed_ids = []
            for call_args in calls:
                # mark_generations_uploaded(account, generation_ids_set)
                # generation_ids_set is the 2nd positional arg (index 1)
                gen_ids = call_args[0][1]
                processed_ids.extend(list(gen_ids))

            assert processed_ids == ["gen_1", "gen_2", "gen_3"], (
                f"Expected oldest-first order [gen_1, gen_2, gen_3], got {processed_ids}"
            )

@pytest.mark.integration
class TestDBInitialization:
    """Tests for database auto-initialization."""

    async def test_upload_sequential_auto_inits_db(self, tmp_path, monkeypatch):
        """upload_to_notion_sequential should auto-initialize DB if not exists.

        Previously this failed with OperationalError: no such table: image_generations
        because init_db() was not called before first DB access.
        """
        from unittest.mock import AsyncMock, MagicMock, patch
        from chatgpt_to_notion.domain.models import AppConfig
        import tomllib

        account = "initdb@example.com"
        config_file = tmp_path / "config.toml"
        account_section = f'[accounts."{account}"]'
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

{account_section}
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config._load_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        session_mock = MagicMock()
        response_mock = AsyncMock()
        response_mock.status = 200
        response_mock.__aenter__ = AsyncMock(return_value=response_mock)
        response_mock.__aexit__ = AsyncMock(return_value=False)
        # Return a valid image response for download
        response_mock.read = AsyncMock(return_value=b"fake_image_data")
        response_mock.json = AsyncMock(return_value={})
        response_mock.text = AsyncMock(return_value="")
        session_mock.get = MagicMock(return_value=response_mock)
        session_mock.patch = MagicMock(return_value=response_mock)
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=False)

        # No isolated_db fixture - this tests that init_db is called before first DB access
        # The default DB path will be tmp_path/output/chatgpt.db
        # Do NOT mock get_uploaded_generation_ids or get_existing_ids - let them hit real DB
        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_chat.get_image_generations = AsyncMock(
            return_value={
                "items": [
                    {
                        "id": "test_img_1",
                        "conversation_id": "test_conv_1",
                        "message_id": "test_msg_1",
                        "asset_pointer": "asset_1",
                        "url": "https://example.com/image.png",
                        "created_at": 1705315800,
                    }
                ]
            }
        )
        fake_chat.get_conversation_details = AsyncMock(return_value={"mapping": {}})
        fake_chat.delete_conversation = AsyncMock()
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock(return_value={"id": "notion_page_1"})

        with patch(
            "aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.history_service.download_image",
            new_callable=AsyncMock,
        ) as mock_download, patch(
            "chatgpt_to_notion.services.image_service.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt:
            mock_download.return_value = b"fake_image_data"
            mock_prompt.return_value = "Test prompt"

            from chatgpt_to_notion.services import upload_sequential

            # Should not raise OperationalError
            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=False,
                account=account,
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
                options=None,
            )

            # Verify it completed without error
            fake_chat.get_image_generations.assert_called_once()
            fake_notion.add_page_to_db.assert_called_once()
        """upload_to_notion_sequential should auto-initialize DB if not exists.

        Previously this failed with OperationalError: no such table: image_generations
        because init_db() was not called before first DB access.
        """
        from unittest.mock import AsyncMock, MagicMock, patch
        from chatgpt_to_notion.domain.models import AppConfig
        import tomllib

        account = "initdb@example.com"
        config_file = tmp_path / "config.toml"
        account_section = f'[accounts."{account}"]'
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test_database_123"

{account_section}
authorization_token = "test_token_abc"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config._load_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

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

        # No isolated_db fixture - this tests that init_db is called before first DB access
        # The default DB path will be tmp_path/output/chatgpt.db
        # The default DB path will be tmp_path/output/chatgpt.db
        # This tests that init_db is called before first DB access
        fake_chat = MagicMock()
        fake_chat.get_headers.return_value = {"Authorization": "Bearer test"}
        fake_chat.get_image_generations = AsyncMock(return_value={"items": []})
        fake_chat.get_conversation_details = AsyncMock(return_value={"mapping": {}})
        fake_chat.delete_conversation = AsyncMock()
        fake_notion = MagicMock()
        fake_notion.is_page_exists_in_db = AsyncMock(return_value=False)
        fake_notion.add_page_to_db = AsyncMock()
        with patch(
            "aiohttp.ClientSession",
            return_value=session_mock,
        ), patch(
            "chatgpt_to_notion.services.history_service.download_image",
            new_callable=AsyncMock,
        ), patch(
            "chatgpt_to_notion.services.image_service.add_prompt_to_image_single",
        ), patch(
            "chatgpt_to_notion.adapters.notion_api.os.path.exists",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.history_service.get_uploaded_generation_ids",
            return_value=set(),
        ), patch(
            "chatgpt_to_notion.services.history_service.mark_generations_uploaded",
            new_callable=MagicMock,
        ), patch(
            "chatgpt_to_notion.adapters.chatgpt_api.get_prompt_from_image_node_in_conversation",
            new_callable=AsyncMock,
        ) as mock_prompt:
            mock_prompt.return_value = ""

            from chatgpt_to_notion.services import upload_sequential

            # Should not raise OperationalError
            await upload_sequential.upload_to_notion_sequential(
                image_folder=None,
                db_id="test_db",
                delete_conversation=False,
                account=account,
                chatgpt_client=fake_chat,
                notion_client=fake_notion,
                options=None,
            )

            # Verify it completed without error
            fake_chat.get_image_generations.assert_called_once()

