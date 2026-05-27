"""
Unit tests for util.py - pure functions, no external dependencies.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import asyncio

import aiohttp
import pytest

from chatgpt_to_notion.adapters import sqlite_store as db
from chatgpt_to_notion.adapters.config_loader import (
    get_account_names,
    get_notion_context,
    get_provider_context,
    resolve_config,
)
from chatgpt_to_notion.adapters.filesystem import (
    clean_output_path,
    get_output_path,
    resolve_image_folder,
)
from chatgpt_to_notion.domain.models import ChatGPTImageGeneration, RuntimeOptions
from chatgpt_to_notion.services.account_status_service import get_account_activity_statuses
from chatgpt_to_notion.services.history_service import (
    download_image,
    get_uploaded_generation_ids,
    mark_generations_uploaded,
    save_generations,
)
from chatgpt_to_notion.shared.constants import (
    HTTP_TIMEOUT_SECONDS,
    MAX_CONCURRENT_DOWNLOADS,
    MAX_CONCURRENT_REQUESTS,
    MAX_RETRIES,
    OUTPUT_PATH,
)
from chatgpt_to_notion.shared.http import (
    get_http_timeout,
    http_retryable,
    retry_http,
    should_retry_http,
    DetailedHTTPError,
)
from chatgpt_to_notion.shared.time import resolve_timezone


class TestGetOutputPath:
    """Tests for get_output_path function."""

    def test_relative_path_allowed(self, tmp_output_dir, monkeypatch):
        monkeypatch.setattr("chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_output_dir))
        path = get_output_path("images/test.png")
        assert path.name == "test.png"
        assert "images" in str(path)

    def test_absolute_path_rejected(self):
        with pytest.raises(ValueError, match="Absolute paths are not allowed"):
            get_output_path("/etc/passwd")

    def test_path_traversal_rejected(self, tmp_output_dir, monkeypatch):
        monkeypatch.setattr("chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_output_dir))
        with pytest.raises(ValueError, match="attempts to escape"):
            get_output_path("../../etc/passwd")

    def test_creates_parent_directories(self, tmp_output_dir, monkeypatch):
        monkeypatch.setattr("chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_output_dir))
        path = get_output_path("nested/deep/path/file.png")
        assert path.parent.exists()

    def test_is_dir_flag(self, tmp_output_dir, monkeypatch):
        monkeypatch.setattr("chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_output_dir))
        path = get_output_path("test_dir", is_dir=True)
        assert path.is_dir()


class TestResolveImageFolder:
    """Tests for resolve_image_folder function."""

    def test_none_uses_default(self, tmp_output_dir, monkeypatch):
        monkeypatch.setattr("chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_output_dir))
        path = resolve_image_folder(None)
        assert path == tmp_output_dir / "images"

    def test_absolute_path_returned_as_is(self):
        path = resolve_image_folder("/custom/absolute/path")
        assert path == Path("/custom/absolute/path")

    def test_relative_path_wrapped_in_output(self, tmp_output_dir, monkeypatch):
        monkeypatch.setattr("chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_output_dir))
        path = resolve_image_folder("my_images")
        assert path == tmp_output_dir / "my_images"

    def test_relative_path_creates_directory(self, tmp_output_dir, monkeypatch):
        monkeypatch.setattr("chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_output_dir))
        path = resolve_image_folder("new_folder/sub")
        assert path.parent.exists()


class TestResolveTimezone:
    """Tests for resolve_timezone function."""

    def test_returns_zoneinfo_for_valid_name(self):
        from zoneinfo import ZoneInfo

        tz = resolve_timezone("Asia/Jakarta")
        assert isinstance(tz, ZoneInfo)

    def test_returns_utc(self):
        from zoneinfo import ZoneInfo

        tz = resolve_timezone("UTC")
        assert isinstance(tz, ZoneInfo)

    def test_returns_system_tz_when_none(self):
        tz = resolve_timezone(None)
        assert tz is not None

    def test_returns_system_tz_when_empty(self):
        tz = resolve_timezone("")
        assert tz is not None


class TestHttpRetryable:
    """Tests for http_retryable function."""

    def test_429_retryable(self):
        assert http_retryable(429) is True

    def test_5xx_retryable(self):
        assert http_retryable(500) is True
        assert http_retryable(503) is True
        assert http_retryable(504) is True

    def test_4xx_not_retryable(self):
        assert http_retryable(400) is False
        assert http_retryable(404) is False

    def test_2xx_not_retryable(self):
        assert http_retryable(200) is False

    def test_none_not_retryable(self):
        assert http_retryable(None) is False


class TestShouldRetryHttp:
    """Tests for should_retry_http function."""

    def test_client_response_error_429(self):
        err = aiohttp.ClientResponseError(None, None, status=429)
        assert should_retry_http(err) is True

    def test_client_response_error_503(self):
        err = aiohttp.ClientResponseError(None, None, status=503)
        assert should_retry_http(err) is True

    def test_client_response_error_404(self):
        err = aiohttp.ClientResponseError(None, None, status=404)
        assert should_retry_http(err) is False

    def test_client_connector_error(self):
        import errno
        os_error = OSError(errno.ECONNREFUSED, "Connection refused")
        err = aiohttp.ClientConnectorError(None, os_error)
        assert should_retry_http(err) is True

    def test_client_timeout(self):
        err = aiohttp.ClientTimeout()
        assert should_retry_http(err) is True

    def test_client_error_with_status(self):
        err = aiohttp.ClientError()
        err.status = 502
        assert should_retry_http(err) is True

    def test_client_error_with_non_retryable_status(self):
        err = aiohttp.ClientError()
        err.status = 404
        assert should_retry_http(err) is False

    def test_client_error_without_status(self):
        err = aiohttp.ClientError()
        assert should_retry_http(err) is True

    def test_other_exception(self):
        err = ValueError("test")
        assert should_retry_http(err) is False


class TestGetHttpTimeout:
    """Tests for get_http_timeout function."""

    def test_returns_client_timeout(self):
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

    def test_cookie_string_resolves_and_decodes(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '\n'.join(
                [
                    "[shared]",
                    'cookie_string_base64 = "Y29va2llX3NoYXJlZA=="', # "cookie_shared"
                    "",
                    "[accounts.personal]",
                    'authorization_token = "token_personal"',
                    "",
                    "[accounts.override]",
                    'authorization_token = "token_override"',
                    'cookie_string_base64 = "Y29va2llX292ZXJyaWRl"', # "cookie_override"
                ]
            )
        )
        monkeypatch.chdir(tmp_path)

        # 1. Test shared default resolution & decoding
        provider_personal = get_provider_context("chatgpt", RuntimeOptions(account="personal"))
        assert provider_personal.headers["Cookie"] == "cookie_shared"

        # 2. Test account-level override resolution & decoding
        provider_override = get_provider_context("chatgpt", RuntimeOptions(account="override"))
        assert provider_override.headers["Cookie"] == "cookie_override"

    def test_cookie_string_invalid_base64(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            '\n'.join(
                [
                    "[accounts.personal]",
                    'authorization_token = "token_personal"',
                    'cookie_string_base64 = "not-valid-base64!"',
                ]
            )
        )
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="Invalid base64 in cookie_string_base64"):
            get_provider_context("chatgpt", RuntimeOptions(account="personal"))

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

    def test_no_data_is_ready(self, isolated_db, monkeypatch):
        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC")

        assert today_rows[0]["Account"] == "default"
        assert today_rows[0]["Next Wait"] == "Ready"
        assert today_rows[0]["Ready Generate?"] == "✅"
        assert yesterday_rows[0]["Next Wait"] == "Ready"
        assert yesterday_rows[0]["Ready Generate?"] == "✅"

    def test_today_all_active_red(self, isolated_db, monkeypatch):
        now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
        today_start = now.replace(hour=0, minute=0)
        generations = [
            ChatGPTImageGeneration(
                created_at=(today_start + timedelta(hours=h)).isoformat(),
                id=f"gen_{h}",
                conversation_id=f"conv_{h}",
                message_id=f"msg_{h}",
                asset_pointer=f"asset_{h}",
                url=f"https://example.com/{h}.png",
                prompt=f"Prompt {h}",
            )
            for h in range(1, 9)
        ]
        db.upsert_generations("default", generations)

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC", now=now)

        assert today_rows[0]["Ready Generate?"] == "❌  (8/8 to wait)"
        assert yesterday_rows[0]["Ready Generate?"] == "✅"

    def test_fallback_to_yesterday_partial_active(self, isolated_db, monkeypatch):
        now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0)
        generations = [
            ChatGPTImageGeneration(
                created_at=(yesterday_start + timedelta(hours=h)).isoformat(),
                id=f"gen_{h}",
                conversation_id=f"conv_{h}",
                message_id=f"msg_{h}",
                asset_pointer=f"asset_{h}",
                url=f"https://example.com/{h}.png",
                prompt=f"Prompt {h}",
            )
            for h in [13, 14, 15, 16, 17, 1, 2, 3]
        ]
        db.upsert_generations("default", generations)

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC", now=now)

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "⚠️  (5/8 to wait)"

    def test_fallback_to_yesterday_all_active_red(self, isolated_db, monkeypatch):
        now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0)
        generations = [
            ChatGPTImageGeneration(
                created_at=(yesterday_start + timedelta(hours=h)).isoformat(),
                id=f"gen_{h}",
                conversation_id=f"conv_{h}",
                message_id=f"msg_{h}",
                asset_pointer=f"asset_{h}",
                url=f"https://example.com/{h}.png",
                prompt=f"Prompt {h}",
            )
            for h in [13, 14, 15, 16, 17, 18, 19, 20]
        ]
        db.upsert_generations("default", generations)

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC", now=now)

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "❌  (8/8 to wait)"

    def test_both_empty_ready(self, isolated_db, monkeypatch):
        now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
        generations = [
            ChatGPTImageGeneration(
                created_at=(now - timedelta(days=d)).isoformat(),
                id=f"gen_{d}",
                conversation_id=f"conv_{d}",
                message_id=f"msg_{d}",
                asset_pointer=f"asset_{d}",
                url=f"https://example.com/{d}.png",
                prompt=f"Prompt {d}",
            )
            for d in [2, 3]
        ]
        db.upsert_generations("default", generations)

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC")

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "✅"

    def test_yesterday_late_night_still_active(self, isolated_db, monkeypatch):
        now = datetime.now(timezone.utc).replace(hour=7, minute=30, second=0, microsecond=0)
        yesterday_late = (now - timedelta(days=1)).replace(hour=23, minute=50, second=0)
        generations = [
            ChatGPTImageGeneration(
                created_at=yesterday_late.isoformat(),
                id="gen_1",
                conversation_id="conv_1",
                message_id="msg_1",
                asset_pointer="asset_1",
                url="https://example.com/1.png",
                prompt="Prompt 1",
            )
        ]
        db.upsert_generations("default", generations)

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC")

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "❌  (1/1 to wait)"

    def test_yesterday_partial_ready(self, isolated_db, monkeypatch):
        now = datetime.now(timezone.utc).replace(hour=23, minute=40, second=0, microsecond=0)
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0)
        generations = [
            ChatGPTImageGeneration(
                created_at=(yesterday_start + timedelta(hours=h)).isoformat(),
                id=f"gen_{h}",
                conversation_id=f"conv_{h}",
                message_id=f"msg_{h}",
                asset_pointer=f"asset_{h}",
                url=f"https://example.com/{h}.png",
                prompt=f"Prompt {h}",
            )
            for h in [1, 2, 3, 4, 5, 6]
        ] + [
            ChatGPTImageGeneration(
                created_at=(yesterday_start + timedelta(hours=23, minutes=m)).isoformat(),
                id=f"gen_late_{m}",
                conversation_id=f"conv_late_{m}",
                message_id=f"msg_late_{m}",
                asset_pointer=f"asset_late_{m}",
                url=f"https://example.com/late_{m}.png",
                prompt=f"Late Prompt {m}",
            )
            for m in [50, 55]
        ]
        db.upsert_generations("default", generations)

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC", now=now)

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "⚠️  (2/8 to wait)"

    def test_timezone_filter_uses_user_timezone(self, isolated_db, monkeypatch):
        jakarta = timezone(timedelta(hours=7))
        now_utc = datetime.now(timezone.utc).replace(hour=14, minute=0, second=0, microsecond=0)
        now_jakarta = now_utc.astimezone(jakarta)
        yesterday_late_night = now_jakarta.replace(hour=23, minute=0, second=0) - timedelta(days=1)
        yesterday_early = now_jakarta.replace(hour=2, minute=0, second=0) - timedelta(days=1)
        entry_utc_new = yesterday_late_night.astimezone(timezone.utc)
        entry_utc_old = yesterday_early.astimezone(timezone.utc)
        generations = [
            ChatGPTImageGeneration(
                created_at=entry_utc_new.isoformat(),
                id="gen_new",
                conversation_id="conv_new",
                message_id="msg_new",
                asset_pointer="asset_new",
                url="https://example.com/new.png",
                prompt="New Prompt",
            ),
            ChatGPTImageGeneration(
                created_at=entry_utc_old.isoformat(),
                id="gen_old",
                conversation_id="conv_old",
                message_id="msg_old",
                asset_pointer="asset_old",
                url="https://example.com/old.png",
                prompt="Old Prompt",
            ),
        ]
        db.upsert_generations("default", generations)

        today_rows, yesterday_rows = get_account_activity_statuses(
            timezone_name="Asia/Jakarta", now=now_jakarta
        )

        assert len(today_rows) == 0
        assert yesterday_rows[0]["Ready Generate?"] == "⚠️  (1/2 to wait)"

    def test_both_days_have_data(self, isolated_db, monkeypatch):
        now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
        today_start = now.replace(hour=0, minute=0)
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0)
        generations = [
            ChatGPTImageGeneration(
                created_at=(today_start + timedelta(hours=2)).isoformat(),
                id="gen_today_1",
                conversation_id="conv_today_1",
                message_id="msg_today_1",
                asset_pointer="asset_today_1",
                url="https://example.com/today1.png",
                prompt="Today 1",
            ),
            ChatGPTImageGeneration(
                created_at=(today_start + timedelta(hours=4)).isoformat(),
                id="gen_today_2",
                conversation_id="conv_today_2",
                message_id="msg_today_2",
                asset_pointer="asset_today_2",
                url="https://example.com/today2.png",
                prompt="Today 2",
            ),
            ChatGPTImageGeneration(
                created_at=(yesterday_start + timedelta(hours=14)).isoformat(),
                id="gen_yesterday_1",
                conversation_id="conv_yesterday_1",
                message_id="msg_yesterday_1",
                asset_pointer="asset_yesterday_1",
                url="https://example.com/yesterday1.png",
                prompt="Yesterday 1",
            ),
            ChatGPTImageGeneration(
                created_at=(yesterday_start + timedelta(hours=16)).isoformat(),
                id="gen_yesterday_2",
                conversation_id="conv_yesterday_2",
                message_id="msg_yesterday_2",
                asset_pointer="asset_yesterday_2",
                url="https://example.com/yesterday2.png",
                prompt="Yesterday 2",
            ),
        ]
        db.upsert_generations("default", generations)

        today_rows, yesterday_rows = get_account_activity_statuses(timezone_name="UTC", now=now)

        assert today_rows[0]["Ready Generate?"] == "❌  (2/2 to wait)"
        assert yesterday_rows[0]["Ready Generate?"] == "❌  (2/2 to wait)"


class TestSaveGenerations:
    """Tests for save_generations function."""

    def test_empty_data_skips(self, capsys):
        save_generations("test", [])
        captured = capsys.readouterr()
        assert "No generations to save" in captured.out

    def test_saves_to_db(self, isolated_db, monkeypatch):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        data = [
            ChatGPTImageGeneration(
                created_at=now.isoformat(),
                id="test123",
                conversation_id="conv_1",
                message_id="msg_1",
                asset_pointer="asset_1",
                url="https://example.com/1.png",
                prompt="test prompt",
            )
        ]
        save_generations("default", data)

        result = db.get_generations("default", include_uploaded=True)
        assert len(result) == 1
        assert result[0].id == "test123"


class TestUploadedGenerationHelpers:
    """Tests for uploaded_at helpers."""

    def test_get_and_mark_uploaded(self, isolated_db, monkeypatch):
        gen_a = ChatGPTImageGeneration(
            created_at="2026-05-14T00:00:00+00:00",
            id="a",
            conversation_id="conv_a",
            message_id="msg_a",
            asset_pointer="asset_a",
            url="https://example.com/a.png",
            prompt="A",
        )
        gen_b = ChatGPTImageGeneration(
            created_at="2026-05-14T00:00:00+00:00",
            id="b",
            conversation_id="conv_b",
            message_id="msg_b",
            asset_pointer="asset_b",
            url="https://example.com/b.png",
            prompt="B",
        )
        db.upsert_generations("default", [gen_a, gen_b])

        assert get_uploaded_generation_ids("default") == set()

        mark_generations_uploaded("default", {"b"})

        assert get_uploaded_generation_ids("default") == {"b"}


class TestCleanOutputPath:
    """Tests for clean_output_path function."""

    def test_removes_files(self, tmp_output_dir, monkeypatch):
        monkeypatch.setattr("chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_output_dir))

        (tmp_output_dir / "test.txt").write_text("test")
        (tmp_output_dir / ".gitkeep").write_text("keep")

        clean_output_path()

        assert not (tmp_output_dir / "test.txt").exists()
        assert (tmp_output_dir / ".gitkeep").exists()

    def test_removes_directories(self, tmp_output_dir, monkeypatch):
        monkeypatch.setattr("chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_output_dir))

        test_dir = tmp_output_dir / "test_dir"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test")

        clean_output_path()

        assert not test_dir.exists()

    def test_nonexistent_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("chatgpt_to_notion.shared.constants.OUTPUT_PATH", str(tmp_path / "nonexistent"))
        clean_output_path()


class TestRetryHttp:
    """Tests for retry_http function."""

    def test_returns_retry_decorator(self):
        decorator = retry_http()
        assert decorator is not None

    @patch("chatgpt_to_notion.shared.http.MAX_RETRIES", 2)
    @pytest.mark.asyncio
    async def test_applies_retry_logic(self, caplog, capsys):
        import errno
        import logging
        caplog.set_level(logging.WARNING, logger="chatgpt_to_notion.http")
        call_count = 0

        @retry_http()
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                os_error = OSError(errno.ECONNREFUSED, "Connection refused")

                class MockConnKey:
                    ssl = False
                    host = "localhost"
                    port = 80
                    is_ssl = False
                    proxy = None
                    proxy_auth = None
                    proxy_headers_hash = None

                raise aiohttp.ClientConnectorError(MockConnKey(), os_error)
            return "success"

        result = await failing_func()
        assert result == "success"
        assert call_count == 2

        # Verify exactly one retry attempt was logged
        captured = capsys.readouterr()
        retries_logged = caplog.text.count("Retrying") + captured.out.count("Retrying") + captured.err.count("Retrying")
        assert retries_logged == 1


class TestDownloadImage:
    """Tests for download_image function."""

    class MockDownloadResponse:
        def __init__(self, content: bytes = b"data", error: Exception | None = None):
            self._content = content
            self.error = error
            self.status = getattr(error, "status", 200) if error else 200
            self.reason = "OK" if not error else "Error"
            self.url = "https://mock.url"
            self.request_info = MagicMock()

        def raise_for_status(self):
            if self.error:
                raise self.error

        async def read(self):
            return self._content

        async def text(self):
            return "mock error body"

    class MockStream:
        def __init__(self, content: bytes):
            self._content = content

        async def iter_chunked(self, chunk_size):
            for i in range(0, len(self._content), chunk_size):
                yield self._content[i : i + chunk_size]

    class MockContext:
        def __init__(self, response):
            self.response = response
            if hasattr(response, '_content'):
                response.content = TestDownloadImage.MockStream(response._content)

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, *args):
            return None

    @pytest.mark.asyncio
    async def test_downloads_image(self, tmp_path):
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
        error = aiohttp.ClientResponseError(
            None, None, status=404, message="Not Found"
        )
        mock_response = self.MockDownloadResponse(error=error)

        mock_session = MagicMock()
        mock_session.get.return_value = self.MockContext(mock_response)

        file_path = tmp_path / "test.png"
        with pytest.raises(DetailedHTTPError):
            await download_image(
                mock_session, "http://example.com/img.png", str(file_path)
            )

    @patch("chatgpt_to_notion.shared.http.MAX_RETRIES", 2)
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, tmp_path):
        mock_response = self.MockDownloadResponse(content=b"retry_image_data")

        call_count = 0

        def get_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError("Connection timed out")
            return self.MockContext(mock_response)

        mock_session = MagicMock()
        mock_session.get.side_effect = get_side_effect

        file_path = tmp_path / "test.png"
        await download_image(mock_session, "http://example.com/img.png", str(file_path))

        assert call_count == 2
        assert file_path.read_bytes() == b"retry_image_data"


class TestConstants:
    """Tests for module constants."""

    def test_max_retries(self):
        assert MAX_RETRIES == 5

    def test_max_concurrent_downloads(self):
        assert MAX_CONCURRENT_DOWNLOADS == 10

    def test_max_concurrent_requests(self):
        assert MAX_CONCURRENT_REQUESTS == 10

    def test_http_timeout_seconds(self):
        assert HTTP_TIMEOUT_SECONDS == 30

    def test_output_path(self):
        assert OUTPUT_PATH == "./output"
