"""
CLI tests for main.py using Typer's test utilities.
"""
import re
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from main import app

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
    return ansi_escape.sub('', text)


class TestCLIHelp:
    """Tests for CLI help messages."""

    def test_main_help(self):
        """Should show main help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "upload-to-notion" in result.stdout
        assert "account-status" in result.stdout
        assert "clean-output-path" in result.stdout

    def test_upload_help(self):
        """Should show upload-to-notion help."""
        result = runner.invoke(app, ["upload-to-notion", "--help"])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "--image-folder" in clean_output
        assert "--limit" in clean_output


class TestCLIValidation:
    """Tests for CLI input validation."""

    def test_invalid_db_id_too_short(self):
        """Should reject database IDs that are too short."""
        result = runner.invoke(
            app,
            ["upload-to-notion", "--db-id", "short"],
        )
        assert result.exit_code != 0
        # Error message may be in stdout or stderr
        output = result.stdout + str(result.stderr)
        assert (
            "Notion database ID must be a valid ID" in output
            or "Invalid value" in output
        )

    def test_invalid_db_id_empty(self):
        """Should reject empty database IDs."""
        result = runner.invoke(
            app,
            ["upload-to-notion", "--db-id", ""],
        )
        assert result.exit_code != 0


class TestCLICommands:
    """Tests for CLI command execution."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_config_toml, tmp_output_dir, monkeypatch):
        """Setup test environment."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        yield

    @patch("chatgpt.upload_to_notion", new_callable=AsyncMock)
    def test_upload_to_notion(self, mock_upload, mock_config_toml):
        """Should call chatgpt.upload_to_notion."""
        result = runner.invoke(
            app,
            [
                "upload-to-notion",
                "--image-folder", "test_images",
                "--db-id", "test_db_12345678901234567890",
                "--limit", "10",
                "--no-remove",
            ],
        )
        assert result.exit_code == 0
        mock_upload.assert_called_once()

    @patch("chatgpt.upload_to_notion", new_callable=AsyncMock)
    def test_upload_to_notion_defaults_to_all_accounts(
        self, mock_upload, mock_config_toml
    ):
        """Should run all configured accounts when --account is omitted."""
        result = runner.invoke(
            app,
            [
                "upload-to-notion",
                "--db-id", "test_db_12345678901234567890",
            ],
        )
        assert result.exit_code == 0
        mock_upload.assert_called_once()

    @patch("util.validate_runtime_config")
    @patch("util.resolve_config")
    @patch("util.get_account_names", return_value=["acc1", "acc2"])
    @patch("chatgpt.upload_to_notion", new_callable=AsyncMock)
    def test_upload_to_notion_multiple_accounts(
        self,
        mock_upload,
        mock_accounts,
        mock_resolve,
        mock_validate,
        mock_config_toml,
    ):
        """Should call chatgpt.upload_to_notion for every configured account."""
        from models import AccountConfig, NotionConfig, ResolvedConfig

        mock_validate.return_value = None
        mock_resolve.side_effect = [
            ResolvedConfig(
                account_name="acc1",
                account=AccountConfig(
                    authorization_token="token1",
                    user_agent="ua",
                ),
                notion=NotionConfig(api_key="key", database_id="db"),
            ),
            ResolvedConfig(
                account_name="acc2",
                account=AccountConfig(
                    authorization_token="token2",
                    user_agent="ua",
                ),
                notion=NotionConfig(api_key="key", database_id="db"),
            ),
        ]
        result = runner.invoke(
            app, ["upload-to-notion", "--db-id", "test_db_12345678901234567890"]
        )
        assert result.exit_code == 0
        assert mock_upload.call_count == 2

    @patch("chatgpt.upload_to_notion", new_callable=AsyncMock)
    def test_upload_to_notion_uses_account_csv(
        self, mock_upload, mock_config_toml
    ):
        """Should write the single per-account CSV."""
        result = runner.invoke(
            app,
            [
                "upload-to-notion",
                "--db-id", "test_db_12345678901234567890",
            ],
        )
        assert result.exit_code == 0
        mock_upload.assert_called_once()
        assert mock_upload.call_args.kwargs["dataset"] == "history/default_chatgpt.csv"
        assert mock_upload.call_args.kwargs["check_notion_api"] is False

    @patch("chatgpt.upload_to_notion", new_callable=AsyncMock)
    def test_upload_to_notion_check_notion_api_flag(
        self, mock_upload, mock_config_toml
    ):
        """Should pass check_notion_api flag through."""
        result = runner.invoke(
            app,
            [
                "upload-to-notion",
                "--db-id",
                "test_db_12345678901234567890",
                "--check-notion-api",
            ],
        )

        assert result.exit_code == 0
        assert mock_upload.call_args.kwargs["check_notion_api"] is True

    @patch("chatgpt.upload_to_notion", new_callable=AsyncMock)
    def test_upload_to_notion_from_history_flag(
        self, mock_upload, mock_config_toml
    ):
        """Should pass from_history flag through."""
        result = runner.invoke(
            app,
            [
                "upload-to-notion",
                "--db-id",
                "test_db_12345678901234567890",
                "--from-history",
            ],
        )

        assert result.exit_code == 0
        assert mock_upload.call_args.kwargs["from_history"] is True
        assert mock_upload.call_args.kwargs["check_notion_api"] is False

    @patch("chatgpt.upload_to_notion", new_callable=AsyncMock)
    def test_upload_to_notion_verify_history_flag(
        self, mock_upload, mock_config_toml
    ):
        """Should make verify_history imply history source and Notion verification."""
        result = runner.invoke(
            app,
            [
                "upload-to-notion",
                "--db-id",
                "test_db_12345678901234567890",
                "--verify-history",
            ],
        )

        assert result.exit_code == 0
        assert mock_upload.call_args.kwargs["from_history"] is True
        assert mock_upload.call_args.kwargs["check_notion_api"] is True

    def test_clean_output_path(self, tmp_output_dir, monkeypatch):
        """Should clean output path."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))

        # Create test files
        (tmp_output_dir / "test.txt").write_text("test")
        (tmp_output_dir / ".gitkeep").write_text("keep")

        result = runner.invoke(app, ["clean-output-path"])
        assert result.exit_code == 0
        assert "Cleaning output path" in result.stdout
        assert "Output path cleaned" in result.stdout

        # Verify files cleaned
        assert not (tmp_output_dir / "test.txt").exists()
        assert (tmp_output_dir / ".gitkeep").exists()

    def test_account_status(self):
        """Should show account readiness table."""
        result = runner.invoke(app, ["account-status", "--timezone", "UTC"])

        assert result.exit_code == 0
        assert "default" in result.stdout
        assert "Ready" in result.stdout
        assert "0s" in result.stdout

    def test_account_status_multiple_accounts_no_flag(self):
        """Should work without --account when multiple accounts exist."""
        with patch("util._load_toml_config") as mock_load:
            from models import AppConfig, AccountConfig, NotionConfig, SharedAccountConfig
            mock_load.return_value = AppConfig(
                notion=NotionConfig(api_key="x", database_id="x"),
                shared=SharedAccountConfig(user_agent="x"),
                accounts={
                    "account_a": AccountConfig(authorization_token="token_a"),
                    "account_b": AccountConfig(authorization_token="token_b"),
                },
            )
            result = runner.invoke(app, ["account-status"])
            assert result.exit_code == 0
            assert "account_a" in result.stdout
            assert "account_b" in result.stdout


class TestCLIConfigValidation:
    """Tests for TOML configuration validation."""

    def test_missing_config_values_chatgpt(self):
        """Should fail if required TOML config values are missing."""
        with patch("util.validate_runtime_config") as mock_validate:
            mock_validate.side_effect = ValueError("Missing CHATGPT_USER_AGENT")
            result = runner.invoke(
                app,
                [
                    "upload-to-notion",
                    "--db-id", "test_db_12345678901234567890",
                ],
            )
            assert result.exit_code != 0


class TestCLIDefaults:
    """Tests for CLI default values."""

    def test_chatgpt_default_image_folder(self):
        """Should use default image folder."""
        result = runner.invoke(app, ["upload-to-notion", "--help"])
        assert result.exit_code == 0
        assert "[default: images]" in result.stdout

    def test_chatgpt_default_limit(self):
        """Should use default limit."""
        result = runner.invoke(app, ["upload-to-notion", "--help"])
        assert result.exit_code == 0
        assert "[default: 100]" in result.stdout

    def test_upload_to_notion_timezone_option(self):
        """Should accept --timezone option."""
        result = runner.invoke(app, ["upload-to-notion", "--help"])
        assert result.exit_code == 0
        assert "--timezone" in strip_ansi(result.stdout)

    def test_account_status_timezone_option(self):
        """Should accept --timezone option."""
        result = runner.invoke(app, ["account-status", "--timezone", "Asia/Jakarta"])
        assert result.exit_code == 0
