"""Tests for maintenance CLI commands."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from chatgpt_to_notion.cli.app import app
from chatgpt_to_notion.shared import constants


class TestCleanOutputPath:
    """Tests for clean-output-path command."""

    def test_clean_output_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(constants, "OUTPUT_PATH", str(tmp_path))
        (tmp_path / "file1.txt").write_text("test")
        (tmp_path / "file2.txt").write_text("test")
        (tmp_path / ".gitkeep").write_text("")

        runner = CliRunner()
        result = runner.invoke(app, ["clean-output-path"])

        assert result.exit_code == 0
        assert "Cleaning output path" in result.output
        assert "Output path cleaned" in result.output
        assert not (tmp_path / "file1.txt").exists()
        assert not (tmp_path / "file2.txt").exists()
        assert (tmp_path / ".gitkeep").exists()

    def test_clean_output_path_nonexistent(self, tmp_path, monkeypatch):
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.setattr(constants, "OUTPUT_PATH", str(nonexistent))

        runner = CliRunner()
        result = runner.invoke(app, ["clean-output-path"])

        assert result.exit_code == 0
        assert "Cleaning output path" in result.output
        assert "Output path cleaned" in result.output


class TestBackupDb:
    """Tests for backup-db command."""

    def test_backup_db(self, isolated_db, tmp_path):
        backup_target = tmp_path / "backup.db"

        runner = CliRunner()
        result = runner.invoke(app, ["backup-db", str(backup_target)])

        assert result.exit_code == 0
        assert "Backing up database" in result.output
        assert "Backup saved to" in result.output
        assert backup_target.exists()

    def test_backup_db_creates_parent_dirs(self, isolated_db, tmp_path):
        backup_target = tmp_path / "subdir" / "backup.db"

        runner = CliRunner()
        result = runner.invoke(app, ["backup-db", str(backup_target)])

        assert result.exit_code == 0
        assert backup_target.exists()

class TestRestoreDb:
    """Tests for restore-db command."""

    def test_restore_db(self, tmp_path, monkeypatch):
        from chatgpt_to_notion.adapters import sqlite_store
        from chatgpt_to_notion.adapters.sqlite_store import SqliteStore
        from chatgpt_to_notion.domain.models import ChatGPTImageGeneration

        isolated_db = tmp_path / "isolated_test.db"
        store = SqliteStore(db_path=isolated_db)
        store.init_db()
        monkeypatch.setattr(sqlite_store, "_get_db_path", lambda: isolated_db)

        gen1 = ChatGPTImageGeneration(
            created_at="2024-01-01T00:00:00Z",
            id="gen1",
            conversation_id="conv1",
            message_id="msg1",
            asset_pointer="asset1",
            url="http://example.com",
            prompt="prompt1",
        )
        store.upsert_generations("test@example.com", [gen1])

        backup_path = tmp_path / "backup.db"
        backup_store = SqliteStore(db_path=backup_path)
        backup_store.init_db()
        gen2 = gen1.model_copy(
            update={
                "id": "gen2",
                "conversation_id": "conv2",
                "message_id": "msg2",
                "asset_pointer": "asset2",
                "prompt": "prompt2",
            }
        )
        backup_store.upsert_generations("test@example.com", [gen2])

        runner = CliRunner()
        result = runner.invoke(app, ["restore-db", str(backup_path)])

        assert result.exit_code == 0
        assert "Restoring database" in result.output
        assert "Database restored" in result.output

        restored = SqliteStore(db_path=isolated_db)
        rows = restored.get_generations(
            "test@example.com", include_uploaded=True, ids_filter={"gen2"}
        )
        assert [g.prompt for g in rows] == ["prompt2"]
    def test_restore_db_missing_backup_raises(self, tmp_path, monkeypatch):
        from chatgpt_to_notion.adapters import sqlite_store
        from chatgpt_to_notion.adapters.sqlite_store import SqliteStore

        isolated_db = tmp_path / "isolated_test.db"
        SqliteStore(db_path=isolated_db).init_db()

        monkeypatch.setattr(sqlite_store, "_get_db_path", lambda: isolated_db)

        missing_backup = tmp_path / "missing.db"

        runner = CliRunner()
        result = runner.invoke(app, ["restore-db", str(missing_backup)])

        assert result.exit_code != 0

class TestDeleteLibrary:
    """Tests for delete-library command."""

    def test_delete_library_with_query(self, mock_aiohttp_session):
        runner = CliRunner()

        call_count = {"delete": 0, "count": 0}

        async def mock_delete(*args, **kwargs):
            call_count["delete"] += 1
            return 5

        async def mock_count(*args, **kwargs):
            call_count["count"] += 1
            return 10

        class MockSessionContext:
            async def __aenter__(self):
                return mock_aiohttp_session
            async def __aexit__(self, *args):
                pass

        with patch("chatgpt_to_notion.adapters.chatgpt_api.ChatGPTClientImpl.delete_library_images_by_query", mock_delete):
            with patch("chatgpt_to_notion.adapters.chatgpt_api.ChatGPTClientImpl.count_library_images", mock_count):
                with patch("chatgpt_to_notion.cli.commands.maintenance.config_loader.account_names", return_value=["test@example.com"]), patch("chatgpt_to_notion.cli.commands.maintenance.config_loader.provider_context", return_value=MagicMock()):
                    with patch("chatgpt_to_notion.cli.commands.maintenance.aiohttp.ClientSession", return_value=MockSessionContext()):
                        result = runner.invoke(app, ["delete-library", "test query", "--max-concurrent", "5"])

        assert result.exit_code == 0
        assert "Deleted 5 images" in result.output
        assert "Total images remaining in library: 10" in result.output
        assert call_count["delete"] == 1
        assert call_count["count"] == 1

    def test_delete_library_all_flag(self, mock_aiohttp_session):
        runner = CliRunner()

        call_count = {"delete": 0, "count": 0}

        async def mock_delete(*args, **kwargs):
            call_count["delete"] += 1
            return 10

        async def mock_count(*args, **kwargs):
            call_count["count"] += 1
            return 0

        class MockSessionContext:
            async def __aenter__(self):
                return mock_aiohttp_session
            async def __aexit__(self, *args):
                pass

        with patch("chatgpt_to_notion.adapters.chatgpt_api.ChatGPTClientImpl.delete_library_images_by_query", mock_delete):
            with patch("chatgpt_to_notion.adapters.chatgpt_api.ChatGPTClientImpl.count_library_images", mock_count):
                with patch("chatgpt_to_notion.cli.commands.maintenance.config_loader.account_names", return_value=["test@example.com"]), patch("chatgpt_to_notion.cli.commands.maintenance.config_loader.provider_context", return_value=MagicMock()):
                    with patch("chatgpt_to_notion.cli.commands.maintenance.aiohttp.ClientSession", return_value=MockSessionContext()):
                        result = runner.invoke(app, ["delete-library", "--max-concurrent", "5"])

        assert result.exit_code == 0
        assert "Deleted 10 images" in result.output
        assert "Total images remaining in library: 0" in result.output
        assert call_count["delete"] == 1
        assert call_count["count"] == 1

    def test_delete_library_multiple_accounts(self, mock_aiohttp_session):
        runner = CliRunner()

        call_count = {"delete": 0, "count": 0}

        async def mock_delete(*args, **kwargs):
            call_count["delete"] += 1
            return 3

        async def mock_count(*args, **kwargs):
            call_count["count"] += 1
            return 7

        class MockSessionContext:
            async def __aenter__(self):
                return mock_aiohttp_session
            async def __aexit__(self, *args):
                pass

        with patch("chatgpt_to_notion.adapters.chatgpt_api.ChatGPTClientImpl.delete_library_images_by_query", mock_delete):
            with patch("chatgpt_to_notion.adapters.chatgpt_api.ChatGPTClientImpl.count_library_images", mock_count):
                with patch("chatgpt_to_notion.cli.commands.maintenance.config_loader.account_names", return_value=["a@test.com", "b@test.com"]), patch("chatgpt_to_notion.cli.commands.maintenance.config_loader.provider_context", return_value=MagicMock()):
                    with patch("chatgpt_to_notion.cli.commands.maintenance.aiohttp.ClientSession", return_value=MockSessionContext()):
                        result = runner.invoke(app, ["delete-library", "query"])

        assert result.exit_code == 0
        assert "a@test.com" in result.output
        assert "b@test.com" in result.output
        assert call_count["delete"] == 2
        assert call_count["count"] == 2
    def test_delete_library_session_still_open_for_count(self, mock_aiohttp_session):
        """Verify count_library_images is called while session context is still active."""
        runner = CliRunner()

        session_state = {"active": True}

        class TrackingSessionContext:
            async def __aenter__(self):
                session_state["active"] = True
                return mock_aiohttp_session
            async def __aexit__(self, *args):
                session_state["active"] = False

        count_called_while_active = {"value": False}

        async def mock_count(*args, **kwargs):
            count_called_while_active["value"] = session_state["active"]
            return 5

        with patch("chatgpt_to_notion.adapters.chatgpt_api.ChatGPTClientImpl.delete_library_images_by_query", return_value=3):
            with patch("chatgpt_to_notion.adapters.chatgpt_api.ChatGPTClientImpl.count_library_images", mock_count):
                with patch("chatgpt_to_notion.cli.commands.maintenance.config_loader.account_names", return_value=["test@example.com"]), patch("chatgpt_to_notion.cli.commands.maintenance.config_loader.provider_context", return_value=MagicMock()):
                    with patch("chatgpt_to_notion.cli.commands.maintenance.aiohttp.ClientSession", return_value=TrackingSessionContext()):
                        result = runner.invoke(app, ["delete-library"])

        assert result.exit_code == 0
        assert count_called_while_active["value"] is True, "count_library_images must be called while session is active"


class TestMaintenanceCommandRegistration:
    """Tests that maintenance commands are registered."""

    def test_commands_exist(self):
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "clean-output-path" in result.output
        assert "backup-db" in result.output
        assert "restore-db" in result.output
        assert "delete-library" in result.output