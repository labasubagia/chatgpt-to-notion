"""CLI tests for main.py using Typer's test utilities."""
import re
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from chatgpt_to_notion.cli.app import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
    return ansi_escape.sub("", text)


class TestCLIHelp:
    """Tests for CLI help messages."""

    def test_main_help(self):
        """Should show main help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "upload" in clean_output
        assert "account-status" in clean_output
        assert "clean-output-path" in clean_output
        # upload-to-notion is hidden

    def test_upload_history_help(self):
        """Should show upload history help."""
        result = runner.invoke(app, ["upload", "history", "--help"])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "--image-folder" in clean_output
        assert "--limit" in clean_output
        assert "--from-history" in clean_output
        assert "--no-cache" in clean_output

    def test_upload_verify_help(self):
        """Should show upload verify help."""
        result = runner.invoke(app, ["upload", "verify", "--help"])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "--check-notion-api" in clean_output

    def test_upload_library_help(self):
        """Should show upload library help."""
        result = runner.invoke(app, ["upload", "library", "--help"])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "--delete" in clean_output
        assert "--delete-all" in clean_output

    def test_upload_to_notion_deprecated_help(self):
        """upload-to-notion is deprecated but still available."""
        result = runner.invoke(app, ["upload-to-notion", "--help"])
        # hidden=True means it won't show in main help but can be invoked
        assert result.exit_code == 0


class TestCLIValidation:
    """Tests for CLI input validation."""

    def test_invalid_db_id_too_short(self):
        """Should reject database IDs that are too short."""
        result = runner.invoke(app, ["upload", "history", "--db-id", "short"])
        output = strip_ansi(result.stderr)
        assert (
            "Notion database ID must be a valid ID" in output
            or "Invalid value" in output
        )

    def test_invalid_db_id_empty(self):
        """Should reject empty database IDs."""
        result = runner.invoke(app, ["upload", "history", "--db-id", ""])
        output = strip_ansi(result.stderr)
        assert (
            "Notion database ID must be a valid ID" in output
            or "Invalid value" in output
        )


class TestCLICommands:
    """Tests for CLI command execution."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_config_toml, tmp_output_dir, monkeypatch):
        """Setup test environment."""
        monkeypatch.setattr(
            "chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_output_dir)
        )
        yield

    @patch(
        "chatgpt_to_notion.cli.commands.upload.upload_sequential.upload_to_notion_sequential",
        new_callable=AsyncMock,
    )
    def test_upload_history(self, mock_upload, mock_config_toml):
        """Should call upload_to_notion_sequential with from_history=True."""
        result = runner.invoke(
            app,
            [
                "upload",
                "history",
                "--image-folder",
                "test_images",
                "--db-id",
                "test_db_12345678901234567890",
                "--limit",
                "10",
            ],
        )
        mock_upload.assert_called_once()
        kwargs = mock_upload.call_args.kwargs
        assert kwargs["from_history"] is True
        assert kwargs["upload_to_notion"] is True
        assert kwargs["delete_conversation"] is False

    @patch(
        "chatgpt_to_notion.cli.commands.upload.upload_sequential.upload_to_notion_sequential",
        new_callable=AsyncMock,
    )
    def test_upload_history_all_flag(self, mock_upload, mock_config_toml):
        """Should pass all flag (keep_days=None)."""
        result = runner.invoke(
            app,
            [
                "upload",
                "history",
                "--db-id",
                "test_db_12345678901234567890",
                "--all",
            ],
        )
        assert result.exit_code == 0
        mock_upload.assert_called_once()
        assert mock_upload.call_args.kwargs["keep_days"] is None

    @patch(
        "chatgpt_to_notion.cli.commands.upload.upload_sequential.upload_to_notion_sequential",
        new_callable=AsyncMock,
    )
    def test_upload_history_defaults_to_all_accounts(
        self, mock_upload, mock_config_toml
    ):
        """Should run all configured accounts when --account is omitted."""
        result = runner.invoke(
            app,
            [
                "upload",
                "history",
                "--db-id",
                "test_db_12345678901234567890",
            ],
        )
        assert result.exit_code == 0
        mock_upload.assert_called_once()

    @patch(
        "chatgpt_to_notion.cli.commands.upload.upload_sequential.upload_to_notion_sequential",
        new_callable=AsyncMock,
    )
    def test_upload_history_uses_account(self, mock_upload, mock_config_toml):
        """Should pass account name directly."""
        result = runner.invoke(
            app,
            [
                "upload",
                "history",
                "--db-id",
                "test_db_12345678901234567890",
            ],
        )
        assert result.exit_code == 0
        mock_upload.assert_called_once()
        assert mock_upload.call_args.kwargs["account"] == "default@example.com"
        assert mock_upload.call_args.kwargs["check_notion_api"] is True

    @patch(
        "chatgpt_to_notion.cli.commands.upload.upload_sequential.upload_to_notion_sequential",
        new_callable=AsyncMock,
    )
    def test_upload_history_no_cache_flag(self, mock_upload, mock_config_toml):
        """Should pass no_cache flag through."""
        result = runner.invoke(
            app,
            [
                "upload",
                "history",
                "--db-id",
                "test_db_12345678901234567890",
                "--no-cache",
            ],
        )

        assert result.exit_code == 0
        assert mock_upload.call_args.kwargs["no_cache"] is True

    @patch(
        "chatgpt_to_notion.cli.commands.upload.upload_sequential.upload_to_notion_sequential",
        new_callable=AsyncMock,
    )
    def test_upload_history_delete_conversation_flag(
        self, mock_upload, mock_config_toml
    ):
        """Should pass delete_conversation flag through."""
        result = runner.invoke(
            app,
            [
                "upload",
                "history",
                "--db-id",
                "test_db_12345678901234567890",
                "--delete-conversation",
            ],
        )

        assert result.exit_code == 0
        assert mock_upload.call_args.kwargs["delete_conversation"] is True

    @patch(
        "chatgpt_to_notion.cli.commands.upload.upload_sequential.upload_to_notion_sequential",
        new_callable=AsyncMock,
    )
    def test_upload_verify(self, mock_upload, mock_config_toml):
        """Should call upload_to_notion_sequential with upload_to_notion=False."""
        result = runner.invoke(
            app,
            [
                "upload",
                "verify",
                "--db-id",
                "test_db_12345678901234567890",
            ],
        )
        assert result.exit_code == 0
        mock_upload.assert_called_once()
        kwargs = mock_upload.call_args.kwargs
        assert kwargs["upload_to_notion"] is False
        assert kwargs["check_notion_api"] is True
        assert kwargs["from_history"] is True

    @patch(
        "chatgpt_to_notion.cli.commands.upload.upload_sequential.upload_to_notion_sequential",
        new_callable=AsyncMock,
    )
    def test_upload_verify_no_check(self, mock_upload, mock_config_toml):
        """Should respect --no-check-notion-api."""
        result = runner.invoke(
            app,
            [
                "upload",
                "verify",
                "--db-id",
                "test_db_12345678901234567890",
                "--no-check-notion-api",
            ],
        )
        assert result.exit_code == 0
        mock_upload.assert_called_once()
        assert mock_upload.call_args.kwargs["check_notion_api"] is False

    @patch(
        "chatgpt_to_notion.cli.commands.upload.count_library_images",
        new_callable=AsyncMock,
    )
    @patch(
        "chatgpt_to_notion.cli.commands.upload.delete_library_images_by_query",
        new_callable=AsyncMock,
    )
    @patch(
        "chatgpt_to_notion.cli.commands.upload.get_headers"
    )
    def test_upload_library_delete(
        self, mock_get_headers, mock_delete, mock_count, mock_config_toml
    ):
        """Should call delete_library_images_by_query with query."""
        mock_get_headers.return_value = {"Authorization": "Bearer test"}
        mock_delete.return_value = 5
        mock_count.return_value = 10

        result = runner.invoke(
            app,
            [
                "upload",
                "library",
                "--delete",
                "test query",
            ],
        )
        assert result.exit_code == 0
        assert "Deleted 5 images matching 'test query'" in result.stdout
        assert "Total images remaining in library: 10" in result.stdout

    @patch(
        "chatgpt_to_notion.cli.commands.upload.count_library_images",
        new_callable=AsyncMock,
    )
    @patch(
        "chatgpt_to_notion.cli.commands.upload.delete_library_images_by_query",
        new_callable=AsyncMock,
    )
    @patch(
        "chatgpt_to_notion.cli.commands.upload.get_headers"
    )
    def test_upload_library_delete_all(
        self, mock_get_headers, mock_delete, mock_count, mock_config_toml
    ):
        """Should call delete_library_images_by_query with query=None for --delete-all."""
        mock_get_headers.return_value = {"Authorization": "Bearer test"}
        mock_delete.return_value = 100
        mock_count.return_value = 0

        result = runner.invoke(
            app,
            [
                "upload",
                "library",
                "--delete-all",
            ],
        )
        assert result.exit_code == 0
        assert "Deleted 100 images (all)" in result.stdout
        assert "Total images remaining in library: 0" in result.stdout


class TestCLIConfigValidation:
    """Tests for TOML configuration validation."""

    def test_missing_config_file(self, tmp_path, monkeypatch):
        """Should fail gracefully when config file doesn't exist."""
        monkeypatch.setattr(
            "chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_path)
        )
        result = runner.invoke(
            app,
            [
                "upload",
                "history",
                "--config",
                str(tmp_path / "nonexistent.toml"),
                "--db-id",
                "test_db_12345678901234567890",
            ],
        )
        assert result.exit_code != 0

    def test_empty_accounts_config(self, tmp_path, monkeypatch):
        """Should fail when config has no accounts."""
        monkeypatch.setattr(
            "chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_path)
        )
        config = tmp_path / "config.toml"
        config.write_text(
            '[notion]\napi_key = "test"\ndatabase_id = "test_db_12345678901234567890"\n'
        )
        result = runner.invoke(
            app,
            [
                "upload",
                "history",
                "--config",
                str(config),
                "--db-id",
                "test_db_12345678901234567890",
            ],
        )
        assert result.exit_code != 0


class TestCLIDefaults:
    """Tests for CLI default values."""

    @patch(
        "chatgpt_to_notion.cli.commands.upload.upload_sequential.upload_to_notion_sequential",
        new_callable=AsyncMock,
    )
    def test_chatgpt_default_image_folder(self, mock_upload, mock_config_toml):
        """Should default image_folder to None (resolved by upload function)."""
        result = runner.invoke(
            app,
            [
                "upload",
                "history",
                "--db-id",
                "test_db_12345678901234567890",
            ],
        )
        assert result.exit_code == 0
        mock_upload.assert_called_once()
        called_image_folder = mock_upload.call_args.kwargs["image_folder"]
        # image_folder is None when not specified; upload_sequential.resolve_image_folder handles default
        assert called_image_folder is None

    @patch(
        "chatgpt_to_notion.cli.commands.upload.upload_sequential.upload_to_notion_sequential",
        new_callable=AsyncMock,
    )
    def test_upload_to_notion_deprecated_alias(self, mock_upload, mock_config_toml):
        """Deprecated upload-to-notion still works."""
        result = runner.invoke(
            app,
            [
                "upload-to-notion",
                "--image-folder",
                "test_images",
                "--db-id",
                "test_db_12345678901234567890",
                "--limit",
                "10",
            ],
        )
        assert result.exit_code == 0
        mock_upload.assert_called_once()