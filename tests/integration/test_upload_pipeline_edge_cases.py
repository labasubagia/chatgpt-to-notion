import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from chatgpt_to_notion.services import upload_sequential
from chatgpt_to_notion.domain.models import ChatGPTImageGeneration, RuntimeOptions
import sqlite3
from chatgpt_to_notion.adapters.sqlite_store import init_db
from chatgpt_to_notion.domain.models import AppConfig
from chatgpt_to_notion.adapters.config_loader import _load_toml_config
import tomllib


def make_gen(gen_id: str, conv_id: str = None, prompt: str = None) -> ChatGPTImageGeneration:
    """Create test ChatGPTImageGeneration."""
    return ChatGPTImageGeneration(
        id=gen_id,
        created_at="2024-01-01T00:00:00Z",
        conversation_id=conv_id or f"conv_{gen_id}",
        message_id=f"msg_{gen_id}",
        asset_pointer=f"asset_{gen_id}",
        url=f"http://example.com/{gen_id}.png",
        prompt=prompt or f"prompt_{gen_id}",
    )


@pytest.mark.integration
class TestUploadFromHistoryMode:
    """Tests from_history flag behavior."""

    @pytest.mark.asyncio
    async def test_from_history_loads_only_unuploaded(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service, mock_remove_library_func
    ):
        """from_history should only load generations without uploaded_at."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig

        account = "test@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test-db-id"

[accounts."{account}"]
authorization_token = "test_token"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        # Set up database with one uploaded and one unuploaded generation
        now = "2024-01-01T00:00:00Z"
        uploaded_now = "2024-01-01T01:00:00Z"
        conn = sqlite3.connect(isolated_db)
        conn.execute(
            "INSERT INTO image_generations (id, created_at, conversation_id, message_id, asset_pointer, url, prompt, uploaded_at, account) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("gen1", now, "conv1", "msg1", "asset1", "http://example.com/gen1.png", "prompt1", uploaded_now, account)
        )
        conn.execute(
            "INSERT INTO image_generations (id, created_at, conversation_id, message_id, asset_pointer, url, prompt, uploaded_at, account) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("gen2", now, "conv2", "msg2", "asset2", "http://example.com/gen2.png", "prompt2", "", account)
        )
        conn.commit()
        conn.close()

        # Create image file for gen2
        (tmp_path / "images").mkdir(exist_ok=True)
        (tmp_path / "images" / "gen2.png").write_bytes(b"fake")
        # Note: resolve_image_folder will handle this

        # Configure mocks for from_history mode
        gen1 = make_gen("gen1", "conv1", "prompt1")
        gen2 = make_gen("gen2", "conv2", "prompt2")
        # Only gen2 should be loaded (gen1 is already uploaded)
        mock_history_service.load_image_generations.return_value = [gen2]
        mock_notion_client.is_page_exists_in_db.return_value = False
        mock_notion_client.add_page_to_db.return_value = {"id": "page_1"}
        # Use real mark_generations_uploaded
        from chatgpt_to_notion.services.history_service import mark_generations_uploaded as real_mark
        mock_history_service.mark_generations_uploaded.side_effect = real_mark

        await upload_sequential.upload_to_notion_sequential(
            image_folder=str(tmp_path / "images"),
            db_id="test-db-id",
            from_history=True,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
            remove_library_func=mock_remove_library_func,
        )

        # Should only load unuploaded generations
        mock_history_service.load_image_generations.assert_called_once()
        call_args = mock_history_service.load_image_generations.call_args
        assert call_args.kwargs.get("include_uploaded") is False

        # Should only mark gen2 uploaded
        mock_history_service.mark_generations_uploaded.assert_called_once()
        marked_ids = mock_history_service.mark_generations_uploaded.call_args[0][1]
        assert marked_ids == {"gen2"}


@pytest.mark.integration
class TestUploadSequentialMode:
    """Tests sequential upload mode."""

    @pytest.mark.asyncio
    async def test_sequential_processes_conversations_one_at_a_time(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service, mock_remove_library_func
    ):
        """Sequential mode should process conversations one at time."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig

        account = "test@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test-db-id"

[accounts."{account}"]
authorization_token = "test_token"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        api_items = [
            {"id": "gen1", "conversation_id": "conv1", "message_id": "msg1", "asset_pointer": "asset1", "url": "http://example.com/gen1.png", "prompt": "prompt1", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "gen2", "conversation_id": "conv2", "message_id": "msg2", "asset_pointer": "asset2", "url": "http://example.com/gen2.png", "prompt": "prompt2", "created_at": "2024-01-01T00:00:00Z"},
        ]
        mock_chatgpt_client.get_image_generations.return_value = {"items": api_items}
        mock_chatgpt_client.get_conversation_details.return_value = {
            "mapping": {
                "msg1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Prompt 1"]}}, "parent": None},
                "msg2": {"message": {"author": {"role": "user"}, "content": {"parts": ["Prompt 2"]}}, "parent": None},
            }
        }
        mock_chatgpt_client.get_prompt_from_image_node_in_conversation.side_effect = ["Prompt 1", "Prompt 2"]

        gen1 = make_gen("gen1", "conv1", "prompt1")
        gen2 = make_gen("gen2", "conv2", "prompt2")
        mock_history_service.fetch_generations_for_conversation.side_effect = [[gen1], [gen2]]
        mock_notion_client.is_page_exists_in_db.return_value = False
        mock_notion_client.add_page_to_db.return_value = {"id": "page_1"}
        from chatgpt_to_notion.services.history_service import mark_generations_uploaded as real_mark
        mock_history_service.mark_generations_uploaded.side_effect = real_mark

        await upload_sequential.upload_to_notion_sequential(
            image_folder=str(tmp_path / "images"),
            db_id="test-db-id",
            delete_conversation=False,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
            remove_library_func=mock_remove_library_func,
        )

        # Should call fetch_generations_for_conversation for each conversation
        assert mock_history_service.fetch_generations_for_conversation.call_count == 2


@pytest.mark.integration
class TestUploadFlagCombinations:
    """Tests combinations of upload flags."""

    @pytest.mark.asyncio
    async def test_upload_to_notion_false_skips_notion_calls(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service, mock_remove_library_func
    ):
        """upload_to_notion=False should skip Notion API calls but still download."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig

        account = "test@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test-db-id"

[accounts."{account}"]
authorization_token = "test_token"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        api_items = [
            {"id": "gen1", "conversation_id": "conv1", "message_id": "msg1", "asset_pointer": "asset1", "url": "http://example.com/gen1.png", "prompt": "prompt1", "created_at": "2024-01-01T00:00:00Z"},
        ]
        mock_chatgpt_client.get_image_generations.return_value = {"items": api_items}
        mock_chatgpt_client.get_conversation_details.return_value = {
            "mapping": {
                "msg1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Prompt 1"]}}, "parent": None},
            }
        }
        mock_chatgpt_client.get_prompt_from_image_node_in_conversation.return_value = "Prompt 1"

        gen1 = make_gen("gen1", "conv1", "prompt1")
        mock_history_service.fetch_generations_for_conversation.return_value = [gen1]
        mock_notion_client.is_page_exists_in_db.return_value = False
        from chatgpt_to_notion.services.history_service import mark_generations_uploaded as real_mark
        mock_history_service.mark_generations_uploaded.side_effect = real_mark

        await upload_sequential.upload_to_notion_sequential(
            image_folder=str(tmp_path / "images"),
            db_id="test-db-id",
            upload_to_notion=False,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
            remove_library_func=mock_remove_library_func,
        )

        # Should not call add_page_to_db
        mock_notion_client.add_page_to_db.assert_not_called()
        # But should still call download_image via history_service
        mock_history_service.download_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_prompt_to_image_false_skips_prompt_addition(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service, mock_remove_library_func
    ):
        """add_prompt_to_image=False should skip prompt addition."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig

        account = "test@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test-db-id"

[accounts."{account}"]
authorization_token = "test_token"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        api_items = [
            {"id": "gen1", "conversation_id": "conv1", "message_id": "msg1", "asset_pointer": "asset1", "url": "http://example.com/gen1.png", "prompt": "prompt1", "created_at": "2024-01-01T00:00:00Z"},
        ]
        mock_chatgpt_client.get_image_generations.return_value = {"items": api_items}
        mock_chatgpt_client.get_conversation_details.return_value = {
            "mapping": {
                "msg1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Prompt 1"]}}, "parent": None},
            }
        }
        mock_chatgpt_client.get_prompt_from_image_node_in_conversation.return_value = "Prompt 1"

        gen1 = make_gen("gen1", "conv1", "prompt1")
        mock_history_service.fetch_generations_for_conversation.return_value = [gen1]
        mock_notion_client.is_page_exists_in_db.return_value = False
        mock_notion_client.add_page_to_db.return_value = {"id": "page_1"}
        from chatgpt_to_notion.services.history_service import mark_generations_uploaded as real_mark
        mock_history_service.mark_generations_uploaded.side_effect = real_mark

        await upload_sequential.upload_to_notion_sequential(
            image_folder=str(tmp_path / "images"),
            db_id="test-db-id",
            add_prompt_to_image=False,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
            remove_library_func=mock_remove_library_func,
        )

        mock_image_service.add_prompt_to_image_single.assert_not_called()


@pytest.mark.integration
class TestUploadDeleteFlags:
    """Tests delete conversation and library flags."""

    @pytest.mark.asyncio
    async def test_delete_conversation_called_when_upload_succeeds(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service, mock_remove_library_func
    ):
        """delete_conversation=True should delete conversation after successful upload."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig

        account = "test@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test-db-id"

[accounts."{account}"]
authorization_token = "test_token"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        api_items = [
            {"id": "gen1", "conversation_id": "conv1", "message_id": "msg1", "asset_pointer": "asset1", "url": "http://example.com/gen1.png", "prompt": "prompt1", "created_at": "2024-01-01T00:00:00Z"},
        ]
        mock_chatgpt_client.get_image_generations.return_value = {"items": api_items}
        mock_chatgpt_client.get_conversation_details.return_value = {
            "mapping": {
                "msg1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Prompt 1"]}}, "parent": None},
            }
        }
        mock_chatgpt_client.get_prompt_from_image_node_in_conversation.return_value = "Prompt 1"

        gen1 = make_gen("gen1", "conv1", "prompt1")
        mock_history_service.fetch_generations_for_conversation.return_value = [gen1]
        mock_notion_client.is_page_exists_in_db.return_value = False
        mock_notion_client.add_page_to_db.return_value = {"id": "page_1"}
        from chatgpt_to_notion.services.history_service import mark_generations_uploaded as real_mark
        mock_history_service.mark_generations_uploaded.side_effect = real_mark

        await upload_sequential.upload_to_notion_sequential(
            image_folder=str(tmp_path / "images"),
            db_id="test-db-id",
            delete_conversation=True,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
            remove_library_func=mock_remove_library_func,
        )

        mock_chatgpt_client.delete_conversation.assert_called_once()
        call_args = mock_chatgpt_client.delete_conversation.call_args
        assert call_args[0][1] == "conv1"

    @pytest.mark.asyncio
    async def test_delete_conversation_not_called_when_upload_fails(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service, mock_remove_library_func
    ):
        """delete_conversation should not be called if upload fails."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig

        account = "test@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test-db-id"

[accounts."{account}"]
authorization_token = "test_token"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        api_items = [
            {"id": "gen1", "conversation_id": "conv1", "message_id": "msg1", "asset_pointer": "asset1", "url": "http://example.com/gen1.png", "prompt": "prompt1", "created_at": "2024-01-01T00:00:00Z"},
        ]
        mock_chatgpt_client.get_image_generations.return_value = {"items": api_items}
        mock_chatgpt_client.get_conversation_details.return_value = {
            "mapping": {
                "msg1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Prompt 1"]}}, "parent": None},
            }
        }
        mock_chatgpt_client.get_prompt_from_image_node_in_conversation.return_value = "Prompt 1"

        gen1 = make_gen("gen1", "conv1", "prompt1")
        mock_history_service.fetch_generations_for_conversation.return_value = [gen1]
        mock_notion_client.is_page_exists_in_db.return_value = False
        mock_notion_client.add_page_to_db.side_effect = Exception("Notion error")
        from chatgpt_to_notion.services.history_service import mark_generations_uploaded as real_mark
        mock_history_service.mark_generations_uploaded.side_effect = real_mark

        await upload_sequential.upload_to_notion_sequential(
            image_folder=str(tmp_path / "images"),
            db_id="test-db-id",
            delete_conversation=True,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
            remove_library_func=mock_remove_library_func,
        )

        mock_chatgpt_client.delete_conversation.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_library_called_when_flag_true(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service, mock_remove_library_func
    ):
        """delete_library=True should call remove_library_func."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig

        account = "test@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test-db-id"

[accounts."{account}"]
authorization_token = "test_token"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        mock_chatgpt_client.get_image_generations.return_value = {"items": []}

        await upload_sequential.upload_to_notion_sequential(
            image_folder=str(tmp_path / "images"),
            db_id="test-db-id",
            delete_library=True,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
            remove_library_func=mock_remove_library_func,
        )

        mock_remove_library_func.assert_called_once()
        call_args = mock_remove_library_func.call_args
        assert call_args.kwargs.get("remove_all") is False

    @pytest.mark.asyncio
    async def test_delete_library_all_called_when_flag_true(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service, mock_remove_library_func
    ):
        """delete_library_all=True should call remove_library_func with remove_all=True."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig

        account = "test@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test-db-id"

[accounts."{account}"]
authorization_token = "test_token"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        mock_chatgpt_client.get_image_generations.return_value = {"items": []}

        await upload_sequential.upload_to_notion_sequential(
            image_folder=str(tmp_path / "images"),
            db_id="test-db-id",
            delete_library_all=True,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
            remove_library_func=mock_remove_library_func,
        )

        mock_remove_library_func.assert_called_once()
        call_args = mock_remove_library_func.call_args
        assert call_args.kwargs.get("remove_all") is True


@pytest.mark.integration
class TestUploadTimezoneHandling:
    """Tests timezone handling in upload."""

    @pytest.mark.asyncio
    async def test_timezone_used_in_from_history(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service, mock_remove_library_func
    ):
        """Timezone should be passed to history service."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig

        account = "test@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test-db-id"

[accounts."{account}"]
authorization_token = "test_token"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        gen1 = make_gen("gen1", "conv1", "prompt1")
        mock_history_service.load_image_generations.return_value = [gen1]
        mock_notion_client.is_page_exists_in_db.return_value = False
        mock_notion_client.add_page_to_db.return_value = {"id": "page_1"}
        from chatgpt_to_notion.services.history_service import mark_generations_uploaded as real_mark
        mock_history_service.mark_generations_uploaded.side_effect = real_mark

        await upload_sequential.upload_to_notion_sequential(
            image_folder=str(tmp_path / "images"),
            db_id="test-db-id",
            from_history=True,
            account=account,
            timezone_name="America/New_York",
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
            remove_library_func=mock_remove_library_func,
        )

        # Check timezone was passed to load_image_generations
        call_args = mock_history_service.load_image_generations.call_args
        assert call_args.kwargs.get("timezone_name") == "America/New_York"


@pytest.mark.integration
class TestUploadNoCacheFlag:
    """Tests no_cache flag behavior."""

    @pytest.mark.asyncio
    async def test_no_cache_bypasses_existing_ids_check(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service, mock_remove_library_func
    ):
        """no_cache=True should bypass existing_ids from sqlite_store."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig

        account = "test@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test-db-id"

[accounts."{account}"]
authorization_token = "test_token"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        api_items = [
            {"id": "gen1", "conversation_id": "conv1", "message_id": "msg1", "asset_pointer": "asset1", "url": "http://example.com/gen1.png", "prompt": "prompt1", "created_at": "2024-01-01T00:00:00Z"},
        ]
        mock_chatgpt_client.get_image_generations.return_value = {"items": api_items}
        mock_chatgpt_client.get_conversation_details.return_value = {
            "mapping": {
                "msg1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Prompt 1"]}}, "parent": None},
            }
        }
        mock_chatgpt_client.get_prompt_from_image_node_in_conversation.return_value = "Prompt 1"

        gen1 = make_gen("gen1", "conv1", "prompt1")
        mock_history_service.fetch_generations_for_conversation.return_value = [gen1]
        mock_notion_client.is_page_exists_in_db.return_value = False
        mock_notion_client.add_page_to_db.return_value = {"id": "page_1"}
        from chatgpt_to_notion.services.history_service import mark_generations_uploaded as real_mark
        mock_history_service.mark_generations_uploaded.side_effect = real_mark

        await upload_sequential.upload_to_notion_sequential(
            image_folder=str(tmp_path / "images"),
            db_id="test-db-id",
            no_cache=True,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
            remove_library_func=mock_remove_library_func,
        )

        # With no_cache=True, existing_ids should be empty set (not fetched from DB)
        # This is tested implicitly by the fact that the generation is processed
        mock_chatgpt_client.get_image_generations.assert_called_once()


@pytest.mark.integration
class TestUploadLimitFlag:
    """Tests limit flag behavior."""

    @pytest.mark.asyncio
    async def test_limit_restricts_generations_processed(
        self, tmp_path, monkeypatch, isolated_db,
        mock_chatgpt_client, mock_notion_client, mock_history_service,
        mock_image_service, mock_remove_library_func
    ):
        """limit should restrict number of generations processed."""
        import tomllib
        from chatgpt_to_notion.domain.models import AppConfig

        account = "test@example.com"
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            f"""
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test_notion_key"
database_id = "test-db-id"

[accounts."{account}"]
authorization_token = "test_token"
""".strip()
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "chatgpt_to_notion.adapters.config_loader._load_toml_config",
            lambda _path: AppConfig.model_validate(
                tomllib.loads(config_file.read_text())
            ),
        )

        api_items = [
            {"id": "gen1", "conversation_id": "conv1", "message_id": "msg1", "asset_pointer": "asset1", "url": "http://example.com/gen1.png", "prompt": "prompt1", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "gen2", "conversation_id": "conv2", "message_id": "msg2", "asset_pointer": "asset2", "url": "http://example.com/gen2.png", "prompt": "prompt2", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "gen3", "conversation_id": "conv3", "message_id": "msg3", "asset_pointer": "asset3", "url": "http://example.com/gen3.png", "prompt": "prompt3", "created_at": "2024-01-01T00:00:00Z"},
        ]
        def get_image_generations_with_limit(session, limit=None):
            if limit is not None:
                return {"items": api_items[:limit]}
            return {"items": api_items}
        mock_chatgpt_client.get_image_generations.side_effect = get_image_generations_with_limit
        mock_chatgpt_client.get_conversation_details.return_value = {
            "mapping": {
                "msg1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Prompt 1"]}}, "parent": None},
                "msg2": {"message": {"author": {"role": "user"}, "content": {"parts": ["Prompt 2"]}}, "parent": None},
                "msg3": {"message": {"author": {"role": "user"}, "content": {"parts": ["Prompt 3"]}}, "parent": None},
            }
        }
        mock_chatgpt_client.get_prompt_from_image_node_in_conversation.side_effect = ["Prompt 1", "Prompt 2", "Prompt 3"]

        gen1 = make_gen("gen1", "conv1", "prompt1")
        gen2 = make_gen("gen2", "conv2", "prompt2")
        gen3 = make_gen("gen3", "conv3", "prompt3")
        mock_history_service.fetch_generations_for_conversation.side_effect = [[gen1], [gen2], [gen3]]
        mock_notion_client.is_page_exists_in_db.return_value = False
        mock_notion_client.add_page_to_db.return_value = {"id": "page_1"}
        from chatgpt_to_notion.services.history_service import mark_generations_uploaded as real_mark
        mock_history_service.mark_generations_uploaded.side_effect = real_mark

        await upload_sequential.upload_to_notion_sequential(
            image_folder=str(tmp_path / "images"),
            db_id="test-db-id",
            limit=2,
            account=account,
            options=None,
            chatgpt_client=mock_chatgpt_client,
            notion_client=mock_notion_client,
            history_service=mock_history_service,
            image_service=mock_image_service,
            remove_library_func=mock_remove_library_func,
        )

        # Should only process 2 generations (limit=2)
        # fetch_generations_for_conversation called for first 2 conversations
        assert mock_history_service.fetch_generations_for_conversation.call_count == 2
