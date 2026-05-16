"""
Unit tests for util.py - pure functions, no external dependencies.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import aiohttp
import pandas as pd
import pytest

from models import RuntimeOptions
from util import (
    HTTP_TIMEOUT_SECONDS,
    MAX_CONCURRENT_DOWNLOADS,
    MAX_CONCURRENT_REQUESTS,
    MAX_RETRIES,
    OUTPUT_PATH,
    clean_output_path,
    download_image,
    get_account_activity_statuses,
    get_account_names,
    get_http_timeout,
    get_notion_context,
    get_output_path,
    get_provider_context,
    get_uploaded_generation_ids,
    http_retryable,
    mark_generations_uploaded,
    resolve_config,
    retry_http,
    save_to_dataset,
    should_retry_http,
)


class TestGetOutputPath:
    """Tests for get_output_path function."""

    def test_relative_path_allowed(self, tmp_output_dir, monkeypatch):
        """Relative paths within output directory should work."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        path = get_output_path("images/test.png")
        assert path.name == "test.png"
        assert "images" in str(path)

    def test_absolute_path_rejected(self):
        """Absolute paths should be rejected."""
        with pytest.raises(ValueError, match="Absolute paths are not allowed"):
            get_output_path("/etc/passwd")

    def test_path_traversal_rejected(self, tmp_output_dir, monkeypatch):
        """Path traversal attempts should be rejected."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        with pytest.raises(ValueError, match="attempts to escape"):
            get_output_path("../../etc/passwd")

    def test_creates_parent_directories(self, tmp_output_dir, monkeypatch):
        """Parent directories should be created automatically."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        path = get_output_path("nested/deep/path/file.png")
        assert path.parent.exists()

    def test_is_dir_flag(self, tmp_output_dir, monkeypatch):
        """is_dir=True should create directory."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        path = get_output_path("test_dir", is_dir=True)
        assert path.is_dir()


class TestHttpRetryable:
    """Tests for http_retryable function."""

    def test_429_retryable(self):
        """Rate limit should be retryable."""
        assert http_retryable(429) is True

    def test_5xx_retryable(self):
        """Server errors should be retryable."""
        assert http_retryable(500) is True
        assert http_retryable(503) is True
        assert http_retryable(504) is True

    def test_4xx_not_retryable(self):
        """Client errors should not be retryable."""
        assert http_retryable(400) is False
        assert http_retryable(404) is False

    def test_2xx_not_retryable(self):
        """Success should not be retryable."""
        assert http_retryable(200) is False

    def test_none_not_retryable(self):
        """None status code should not be retryable."""
        assert http_retryable(None) is False


class TestShouldRetryHttp:
    """Tests for should_retry_http function."""

    def test_client_response_error_429(self):
        """429 status should retry."""
        err = aiohttp.ClientResponseError(None, None, status=429)
        assert should_retry_http(err) is True

    def test_client_response_error_503(self):
        """503 status should retry."""
        err = aiohttp.ClientResponseError(None, None, status=503)
        assert should_retry_http(err) is True

    def test_client_response_error_404(self):
        """404 status should not retry."""
        err = aiohttp.ClientResponseError(None, None, status=404)
        assert should_retry_http(err) is False

    def test_client_connector_error(self):
        """Connector errors should retry."""
        import errno
        os_error = OSError(errno.ECONNREFUSED, "Connection refused")
        err = aiohttp.ClientConnectorError(None, os_error)
        assert should_retry_http(err) is True

    def test_client_timeout(self):
        """Timeout errors should retry."""
        err = aiohttp.ClientTimeout()
        assert should_retry_http(err) is True

    def test_client_error_with_status(self):
        """ClientError with status should retry."""
        err = aiohttp.ClientError()
        err.status = 502
        assert should_retry_http(err) is True

    def test_client_error_with_non_retryable_status(self):
        """ClientError with non-retryable status should not retry."""
        err = aiohttp.ClientError()
        err.status = 404
        assert should_retry_http(err) is False

    def test_client_error_without_status(self):
        """ClientError without status should retry."""
        err = aiohttp.ClientError()
        assert should_retry_http(err) is True

    def test_other_exception(self):
        """Other exceptions should not retry by default."""
        err = ValueError("test")
        assert should_retry_http(err) is False


class TestGetHttpTimeout:
    """Tests for get_http_timeout function."""

    def test_returns_client_timeout(self):
        """Should return aiohttp.ClientTimeout."""
        timeout = get_http_timeout()
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total == 30


class TestTomlConfig:
    """Tests for TOML multi-account configuration."""

    def test_resolve_config_from_toml(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '\n'.join(
                [
                    "[notion]",
                    'api_key = "notion_key"',
                    'database_id = "db_default_1234567890"',
                    "",
                    "[accounts.work]",
                    'authorization_token = "token_work"',
                    'user_agent = "Agent/1.0"',
                    'notion_database_id = "db_work_1234567890"',
                ]
            )
        )
        monkeypatch.chdir(tmp_path)

        resolved = resolve_config(RuntimeOptions(account="work"))

        assert resolved.account_name == "work"
        assert resolved.account.authorization_token == "token_work"
        assert resolved.notion.database_id == "db_work_1234567890"

    def test_provider_and_notion_context_from_toml(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '\n'.join(
                [
                    "[notion]",
                    'api_key = "notion_key"',
                    'database_id = "db_default_1234567890"',
                    "",
                    "[accounts.personal]",
                    'authorization_token = "token_personal"',
                    'user_agent = "Agent/2.0"',
                ]
            )
        )
        monkeypatch.chdir(tmp_path)

        provider = get_provider_context("chatgpt", RuntimeOptions(account="personal"))
        notion = get_notion_context(RuntimeOptions(account="personal"))

        assert provider.headers["Authorization"] == "Bearer token_personal"
        assert notion.headers["Authorization"] == "Bearer notion_key"

    def test_shared_defaults_applied_to_account(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '\n'.join(
                [
                    "[shared]",
                    'user_agent = "SharedAgent/1.0"',
                    "",
                    "[accounts.personal]",
                    'authorization_token = "token_personal"',
                ]
            )
        )
        monkeypatch.chdir(tmp_path)

        provider = get_provider_context("chatgpt", RuntimeOptions(account="personal"))

        assert provider.headers["User-Agent"] == "SharedAgent/1.0"

    def test_get_account_names_from_toml(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '\n'.join(
                [
                    "[accounts.a]",
                    'authorization_token = "a"',
                    'user_agent = "ua"',
                    "",
                    "[accounts.b]",
                    'authorization_token = "b"',
                    'user_agent = "ua"',
                ]
            )
        )
        monkeypatch.chdir(tmp_path)

        assert get_account_names() == ["a", "b"]


class TestAccountActivityStatus:
    """Tests for account readiness status."""

    def test_csv_file_not_exists_is_ready(self):
        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC")

        assert today_rows[0]["Account"] == "default"
        assert today_rows[0]["Next Wait"] == "Ready"
        assert today_rows[0]["Ready Generate?"] == "✅"
        assert yesterday_rows[0]["Next Wait"] == "Ready"
        assert yesterday_rows[0]["Ready Generate?"] == "✅"

    def test_today_all_active_red(self, tmp_path, monkeypatch):
        """Today has data, all active (<24h) -> red"""
        history_dir = tmp_path / "output" / "history"
        history_dir.mkdir(parents=True)
        now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
        today_start = now.replace(hour=0, minute=0)
        pd.DataFrame(
            [
                {"id": "a", "created_at": (today_start + timedelta(hours=1)).isoformat()},
                {"id": "b", "created_at": (today_start + timedelta(hours=2)).isoformat()},
                {"id": "c", "created_at": (today_start + timedelta(hours=3)).isoformat()},
                {"id": "d", "created_at": (today_start + timedelta(hours=4)).isoformat()},
                {"id": "e", "created_at": (today_start + timedelta(hours=5)).isoformat()},
                {"id": "f", "created_at": (today_start + timedelta(hours=6)).isoformat()},
                {"id": "g", "created_at": (today_start + timedelta(hours=7)).isoformat()},
                {"id": "h", "created_at": (today_start + timedelta(hours=8)).isoformat()},
            ]
        ).to_csv(history_dir / "default_chatgpt.csv", index=False)
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "output"))

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC", now=now)

        assert today_rows[0]["Ready Generate?"] == "❌  (8/8 to wait)"
        assert yesterday_rows[0]["Ready Generate?"] == "✅"

    def test_fallback_to_yesterday_partial_active(self, tmp_path, monkeypatch):
        """Today empty, yesterday has partial active -> yellow"""
        history_dir = tmp_path / "output" / "history"
        history_dir.mkdir(parents=True)
        now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0)
        pd.DataFrame(
            [
                {"id": "a", "created_at": (yesterday_start + timedelta(hours=13)).isoformat()},
                {"id": "b", "created_at": (yesterday_start + timedelta(hours=14)).isoformat()},
                {"id": "c", "created_at": (yesterday_start + timedelta(hours=15)).isoformat()},
                {"id": "d", "created_at": (yesterday_start + timedelta(hours=16)).isoformat()},
                {"id": "e", "created_at": (yesterday_start + timedelta(hours=17)).isoformat()},
                {"id": "f", "created_at": (yesterday_start + timedelta(hours=1)).isoformat()},
                {"id": "g", "created_at": (yesterday_start + timedelta(hours=2)).isoformat()},
                {"id": "h", "created_at": (yesterday_start + timedelta(hours=3)).isoformat()},
            ]
        ).to_csv(history_dir / "default_chatgpt.csv", index=False)
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "output"))

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC", now=now)

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "⚠️  (5/8 to wait)"

    def test_fallback_to_yesterday_all_active_red(self, tmp_path, monkeypatch):
        """Today empty, yesterday has all active -> red"""
        history_dir = tmp_path / "output" / "history"
        history_dir.mkdir(parents=True)
        now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0)
        pd.DataFrame(
            [
                {"id": "a", "created_at": (yesterday_start + timedelta(hours=13)).isoformat()},
                {"id": "b", "created_at": (yesterday_start + timedelta(hours=14)).isoformat()},
                {"id": "c", "created_at": (yesterday_start + timedelta(hours=15)).isoformat()},
                {"id": "d", "created_at": (yesterday_start + timedelta(hours=16)).isoformat()},
                {"id": "e", "created_at": (yesterday_start + timedelta(hours=17)).isoformat()},
                {"id": "f", "created_at": (yesterday_start + timedelta(hours=18)).isoformat()},
                {"id": "g", "created_at": (yesterday_start + timedelta(hours=19)).isoformat()},
                {"id": "h", "created_at": (yesterday_start + timedelta(hours=20)).isoformat()},
            ]
        ).to_csv(history_dir / "default_chatgpt.csv", index=False)
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "output"))

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC", now=now)

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "❌  (8/8 to wait)"

    def test_both_empty_ready(self, tmp_path, monkeypatch):
        """Data from 2+ days ago -> today empty, yesterday ready"""
        history_dir = tmp_path / "output" / "history"
        history_dir.mkdir(parents=True)
        now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
        pd.DataFrame(
            [
                {"id": "a", "created_at": (now - timedelta(days=2)).isoformat()},
                {"id": "b", "created_at": (now - timedelta(days=3)).isoformat()},
            ]
        ).to_csv(history_dir / "default_chatgpt.csv", index=False)
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "output"))

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC")

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "✅"

    def test_selected_file_all_over_24h_ready(self, tmp_path, monkeypatch):
        """Yesterday data all > 24h old -> ready, today has no data"""
        history_dir = tmp_path / "output" / "history"
        history_dir.mkdir(parents=True)
        now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0)
        pd.DataFrame(
            [
                {"id": "a", "created_at": (yesterday_start + timedelta(hours=1)).isoformat()},
                {"id": "b", "created_at": (yesterday_start + timedelta(hours=2)).isoformat()},
                {"id": "c", "created_at": (yesterday_start + timedelta(hours=3)).isoformat()},
                {"id": "d", "created_at": (yesterday_start + timedelta(hours=4)).isoformat()},
                {"id": "e", "created_at": (yesterday_start + timedelta(hours=5)).isoformat()},
                {"id": "f", "created_at": (yesterday_start + timedelta(hours=6)).isoformat()},
                {"id": "g", "created_at": (yesterday_start + timedelta(hours=7)).isoformat()},
                {"id": "h", "created_at": (yesterday_start + timedelta(hours=8)).isoformat()},
            ]
        ).to_csv(history_dir / "default_chatgpt.csv", index=False)
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "output"))

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC")

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "✅"

    def test_yesterday_late_night_still_active(self, tmp_path, monkeypatch):
        """Yesterday late night entries (23:50) still active at 07:30 -> red"""
        history_dir = tmp_path / "output" / "history"
        history_dir.mkdir(parents=True)
        now = datetime.now(timezone.utc).replace(hour=7, minute=30, second=0, microsecond=0)
        yesterday_late = (now - timedelta(days=1)).replace(hour=23, minute=50, second=0)
        pd.DataFrame(
            [
                {"id": "a", "created_at": yesterday_late.isoformat()},
            ]
        ).to_csv(history_dir / "default_chatgpt.csv", index=False)
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "output"))

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC")

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "❌  (1/1 to wait)"

    def test_yesterday_partial_ready(self, tmp_path, monkeypatch):
        """Yesterday has 2 active (<24h), 6 ready (>24h) -> yellow"""
        history_dir = tmp_path / "output" / "history"
        history_dir.mkdir(parents=True)
        now = datetime.now(timezone.utc).replace(hour=23, minute=40, second=0, microsecond=0)
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0)
        pd.DataFrame(
            [
                {"id": "a", "created_at": (yesterday_start + timedelta(hours=1)).isoformat()},
                {"id": "b", "created_at": (yesterday_start + timedelta(hours=2)).isoformat()},
                {"id": "c", "created_at": (yesterday_start + timedelta(hours=3)).isoformat()},
                {"id": "d", "created_at": (yesterday_start + timedelta(hours=4)).isoformat()},
                {"id": "e", "created_at": (yesterday_start + timedelta(hours=5)).isoformat()},
                {"id": "f", "created_at": (yesterday_start + timedelta(hours=6)).isoformat()},
                {"id": "g", "created_at": (now - timedelta(hours=25)).isoformat()},
                {"id": "h", "created_at": (now - timedelta(hours=26)).isoformat()},
            ]
        ).to_csv(history_dir / "default_chatgpt.csv", index=False)
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "output"))

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC")

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "⚠️  (2/8 to wait)"

    def test_csv_file_empty_ready(self, tmp_path, monkeypatch):
        """CSV file is empty -> both ready"""
        history_dir = tmp_path / "output" / "history"
        history_dir.mkdir(parents=True)
        pd.DataFrame(columns=["id", "created_at"]).to_csv(
            history_dir / "default_chatgpt.csv", index=False
        )
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "output"))

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC")

        assert today_rows[0]["Ready Generate?"] == "✅"
        assert yesterday_rows[0]["Ready Generate?"] == "✅"

    def test_csv_no_valid_created_at_ready(self, tmp_path, monkeypatch):
        """CSV has no valid created_at column -> both ready"""
        history_dir = tmp_path / "output" / "history"
        history_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {"id": "a", "created_at": "invalid"},
                {"id": "b", "created_at": "also_invalid"},
            ]
        ).to_csv(history_dir / "default_chatgpt.csv", index=False)
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "output"))

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC")

        assert today_rows[0]["Ready Generate?"] == "✅"
        assert yesterday_rows[0]["Ready Generate?"] == "✅"

    def test_csv_corrupted_ready(self, tmp_path, monkeypatch):
        """CSV is corrupted -> both ready"""
        history_dir = tmp_path / "output" / "history"
        history_dir.mkdir(parents=True)
        (history_dir / "default_chatgpt.csv").write_text("not,a,valid,csv,file,at,all")
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "output"))

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC")

        assert today_rows[0]["Ready Generate?"] == "✅"
        assert yesterday_rows[0]["Ready Generate?"] == "✅"

    def test_timezone_filter_uses_user_timezone(self, tmp_path, monkeypatch):
        """Entry is yesterday in Asia/Jakarta but <24h old -> yellow"""
        history_dir = tmp_path / "output" / "history"
        history_dir.mkdir(parents=True)
        jakarta = timezone(timedelta(hours=7))
        now_utc = datetime.now(timezone.utc).replace(hour=14, minute=0, second=0, microsecond=0)
        now_jakarta = now_utc.astimezone(jakarta)
        yesterday_late_night = now_jakarta.replace(hour=23, minute=0, second=0) - timedelta(days=1)
        yesterday_early = now_jakarta.replace(hour=2, minute=0, second=0) - timedelta(days=1)
        entry_utc_new = yesterday_late_night.astimezone(timezone.utc)
        entry_utc_old = yesterday_early.astimezone(timezone.utc)
        pd.DataFrame(
            [
                {"id": "a", "created_at": entry_utc_new.isoformat()},
                {"id": "b", "created_at": entry_utc_old.isoformat()},
            ]
        ).to_csv(history_dir / "default_chatgpt.csv", index=False)
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "output"))

        today_rows, yesterday_rows = get_account_activity_statuses(
            timezone_name="Asia/Jakarta", now=now_jakarta
        )

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "⚠️  (1/2 to wait)"

    def test_both_days_have_data(self, tmp_path, monkeypatch):
        """Today and yesterday both have data -> both tables show status"""
        history_dir = tmp_path / "output" / "history"
        history_dir.mkdir(parents=True)
        now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
        today_start = now.replace(hour=0, minute=0)
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0)
        pd.DataFrame(
            [
                {"id": "a", "created_at": (today_start + timedelta(hours=2)).isoformat()},
                {"id": "b", "created_at": (today_start + timedelta(hours=4)).isoformat()},
                {"id": "c", "created_at": (yesterday_start + timedelta(hours=14)).isoformat()},
                {"id": "d", "created_at": (yesterday_start + timedelta(hours=16)).isoformat()},
            ]
        ).to_csv(history_dir / "default_chatgpt.csv", index=False)
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "output"))

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC", now=now)

        assert today_rows[0]["Ready Generate?"] == "❌  (2/2 to wait)"
        assert yesterday_rows[0]["Ready Generate?"] == "❌  (2/2 to wait)"


class TestSaveToDataset:
    """Tests for save_to_dataset function."""

    def test_none_dataset_skips(self, capsys):
        """None dataset should skip silently."""
        save_to_dataset(None, [{"id": "test"}])
        captured = capsys.readouterr()
        assert "Saved dataset" not in captured.out

    def test_empty_data_skips(self, capsys):
        """Empty data should skip."""
        save_to_dataset("test.csv", [])
        captured = capsys.readouterr()
        assert "No generations to save" in captured.out

    def test_saves_csv(self, tmp_output_dir, monkeypatch):
        """Should save to CSV file."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        data = [{"id": "test123", "prompt": "test prompt"}]
        save_to_dataset("test_dataset.csv", data)

        csv_path = tmp_output_dir / "test_dataset.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "id" in content
        assert "test123" in content

    def test_merges_existing_csv_by_id_and_keeps_recent_rows(
        self, tmp_output_dir, monkeypatch
    ):
        """Should merge new data with old dataset and keep only recent rows."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        now = datetime.now(timezone.utc).replace(microsecond=0)
        csv_path = tmp_output_dir / "chatgpt.csv"
        pd.DataFrame(
            [
                {"id": "old", "created_at": now.isoformat(), "prompt": "old"},
                {
                    "id": "dupe",
                    "created_at": (now - timedelta(hours=1)).isoformat(),
                    "prompt": "before",
                },
                {
                    "id": "expired",
                    "created_at": (now - timedelta(days=3)).isoformat(),
                    "prompt": "expired",
                },
            ]
        ).to_csv(csv_path, index=False)

        save_to_dataset(
            "chatgpt.csv",
            [
                {
                    "id": "dupe",
                    "created_at": now.isoformat(),
                    "prompt": "after",
                },
                {
                    "id": "new",
                    "created_at": now.isoformat(),
                    "prompt": "new",
                },
            ],
        )

        merged = pd.read_csv(csv_path)
        assert set(merged["id"]) == {"old", "dupe", "new"}
        assert merged.loc[merged["id"] == "dupe", "prompt"].item() == "after"

    def test_preserves_uploaded_at_when_merging_existing_rows(
        self, tmp_output_dir, monkeypatch
    ):
        """Should keep uploaded_at when an existing generation is fetched again."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        csv_path = tmp_output_dir / "history" / "default_chatgpt.csv"
        csv_path.parent.mkdir(parents=True)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        pd.DataFrame(
            [
                {
                    "id": "dupe",
                    "created_at": now.isoformat(),
                    "prompt": "before",
                    "uploaded_at": "2026-05-14T00:00:00+00:00",
                }
            ]
        ).to_csv(csv_path, index=False)

        save_to_dataset(
            "history/default_chatgpt.csv",
            [
                {
                    "id": "dupe",
                    "created_at": now.isoformat(),
                    "prompt": "after",
                }
            ],
        )

        merged = pd.read_csv(csv_path)
        assert merged.loc[0, "prompt"] == "after"
        assert merged.loc[0, "uploaded_at"] == "2026-05-14T00:00:00+00:00"

    def test_uploaded_generation_helpers(self, tmp_output_dir, monkeypatch):
        """Should read and update uploaded_at in dataset CSV."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))
        csv_path = tmp_output_dir / "history" / "default_chatgpt.csv"
        csv_path.parent.mkdir(parents=True)
        pd.DataFrame(
            [
                {"id": "a", "created_at": "2026-05-14T00:00:00+00:00"},
                {
                    "id": "b",
                    "created_at": "2026-05-14T00:00:00+00:00",
                    "uploaded_at": "2026-05-14T01:00:00+00:00",
                },
            ]
        ).to_csv(csv_path, index=False)

        assert get_uploaded_generation_ids("history/default_chatgpt.csv") == {"b"}

        mark_generations_uploaded("history/default_chatgpt.csv", {"a"})

        assert get_uploaded_generation_ids("history/default_chatgpt.csv") == {
            "a",
            "b",
        }


class TestCleanOutputPath:
    """Tests for clean_output_path function."""

    def test_removes_files(self, tmp_output_dir, monkeypatch):
        """Should remove all files except .gitkeep."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))

        # Create test files
        (tmp_output_dir / "test.txt").write_text("test")
        (tmp_output_dir / ".gitkeep").write_text("keep")

        clean_output_path()

        assert not (tmp_output_dir / "test.txt").exists()
        assert (tmp_output_dir / ".gitkeep").exists()

    def test_removes_directories(self, tmp_output_dir, monkeypatch):
        """Should remove directories."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_output_dir))

        # Create test directory
        test_dir = tmp_output_dir / "test_dir"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test")

        clean_output_path()

        assert not test_dir.exists()

    def test_nonexistent_path(self, tmp_path, monkeypatch):
        """Should handle non-existent output path."""
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path / "nonexistent"))
        # Should not raise
        clean_output_path()


class TestRetryHttp:
    """Tests for retry_http function."""

    def test_returns_retry_decorator(self):
        """Should return a retry decorator."""
        decorator = retry_http()
        assert decorator is not None

    @patch("util.MAX_RETRIES", 2)
    @pytest.mark.asyncio
    async def test_applies_retry_logic(self):
        """Should retry on failure."""
        import errno
        call_count = 0

        @retry_http()
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                os_error = OSError(errno.ECONNREFUSED, "Connection refused")
                raise aiohttp.ClientConnectorError(None, os_error)
            return "success"

        result = await failing_func()
        assert result == "success"
        assert call_count == 2


class TestDownloadImage:
    """Tests for download_image function."""

    class MockDownloadResponse:
        def __init__(self, content: bytes = b"data", error: Exception | None = None):
            self.content = content
            self.error = error

        def raise_for_status(self):
            if self.error:
                raise self.error

        async def read(self):
            return self.content

    class MockContext:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, *args):
            return None

    @pytest.mark.asyncio
    async def test_downloads_image(self, tmp_path):
        """Should download image to file."""
        mock_response = self.MockDownloadResponse(content=b"image_data")

        mock_session = MagicMock()
        mock_session.get.return_value = self.MockContext(mock_response)

        file_path = tmp_path / "test.png"
        await download_image(mock_session, "http://example.com/img.png", str(file_path))

        mock_session.get.assert_called_once_with(
            "http://example.com/img.png", headers={}
        )
        assert file_path.read_bytes() == b"image_data"

    @pytest.mark.asyncio
    async def test_with_custom_headers(self, tmp_path):
        """Should use custom headers."""
        mock_response = self.MockDownloadResponse(content=b"data")

        mock_session = MagicMock()
        mock_session.get.return_value = self.MockContext(mock_response)

        headers = {"Authorization": "Bearer token"}
        file_path = tmp_path / "test.png"
        await download_image(
            mock_session, "http://example.com/img.png", str(file_path), headers
        )

        mock_session.get.assert_called_once_with(
            "http://example.com/img.png", headers=headers
        )

    @pytest.mark.asyncio
    async def test_http_error_raises(self, tmp_path):
        """Should raise on HTTP error."""
        error = aiohttp.ClientResponseError(
            None, None, status=404, message="Not Found"
        )
        mock_response = self.MockDownloadResponse(error=error)

        mock_session = MagicMock()
        mock_session.get.return_value = self.MockContext(mock_response)

        file_path = tmp_path / "test.png"
        with pytest.raises(aiohttp.ClientResponseError):
            await download_image(
                mock_session, "http://example.com/img.png", str(file_path)
            )


class TestConstants:
    """Tests for module constants."""

    def test_max_retries(self):
        """MAX_RETRIES should be 5."""
        assert MAX_RETRIES == 5

    def test_max_concurrent_downloads(self):
        """MAX_CONCURRENT_DOWNLOADS should be 10."""
        assert MAX_CONCURRENT_DOWNLOADS == 10

    def test_max_concurrent_requests(self):
        """MAX_CONCURRENT_REQUESTS should be 10."""
        assert MAX_CONCURRENT_REQUESTS == 10

    def test_http_timeout_seconds(self):
        """HTTP_TIMEOUT_SECONDS should be 30."""
        assert HTTP_TIMEOUT_SECONDS == 30

    def test_output_path(self):
        """OUTPUT_PATH should be ./output."""
        assert OUTPUT_PATH == "./output"
