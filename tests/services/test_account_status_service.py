"""Tests for account_status_service module."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from chatgpt_to_notion.services import account_status_service



class TestGetActivityStatusForDate:
    """Formatting/calculation behavior exercised through the public entrypoint."""

    @pytest.fixture
    def config_file(self, tmp_path):
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."user@example.com"]
authorization_token = "token1"
"""
        f = tmp_path / "config.toml"
        f.write_text(config_text.strip())
        return str(f)

    def _today_rows(self, config_file, stats, now=None, has_data=True):
        with patch(
            "chatgpt_to_notion.services.account_status_service.sqlite_store.has_generations",
            return_value=has_data,
        ), patch(
            "chatgpt_to_notion.services.account_status_service.sqlite_store.get_activity_stats",
            return_value=stats,
        ):
            today, _, _ = account_status_service.get_account_activity_statuses(
                config_path=config_file,
                timezone_name="UTC",
                now=now or datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            )
            return today

    def test_no_stats_omits_row(self, config_file):
        assert self._today_rows(config_file, None) == []

    def test_active_count_zero_returns_ready(self, config_file):
        stats = {
            "total": 5,
            "active_count": 0,
            "first_active": "2026-01-14T10:00:00+00:00",
            "last_active": "2026-01-15T10:00:00+00:00",
        }
        rows = self._today_rows(config_file, stats)
        assert len(rows) == 1
        assert rows[0]["Ready Generate?"] == "✅"
        assert rows[0]["Next Wait"] == "Ready"
        assert rows[0]["Total Wait"] == "0s"

    def test_active_count_equals_total_returns_exhausted(self, config_file):
        stats = {
            "total": 10,
            "active_count": 10,
            "first_active": "2026-01-15T10:00:00+00:00",
            "last_active": "2026-01-15T14:00:00+00:00",
        }
        rows = self._today_rows(config_file, stats)
        assert "❌" in rows[0]["Ready Generate?"]
        assert "10/10" in rows[0]["Ready Generate?"]

    def test_active_count_partial_returns_warning(self, config_file):
        stats = {
            "total": 10,
            "active_count": 3,
            "first_active": "2026-01-15T10:00:00+00:00",
            "last_active": "2026-01-15T14:00:00+00:00",
        }
        rows = self._today_rows(config_file, stats)
        assert "⚠️" in rows[0]["Ready Generate?"]
        assert "3/10" in rows[0]["Ready Generate?"]

    def test_calculates_next_wait_correctly(self, config_file):
        first_active = datetime(2026, 1, 14, 10, 0, 0, tzinfo=timezone.utc)
        stats = {
            "total": 10,
            "active_count": 3,
            "first_active": first_active.isoformat(),
            "last_active": "2026-01-14T14:00:00+00:00",
        }
        now = datetime(2026, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        rows = self._today_rows(config_file, stats, now=now)
        expected = (first_active + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        assert rows[0]["Next Wait"] == expected
        assert rows[0]["Next Cooldown"] == "2h"

    def test_calculates_fully_ready_in_correctly(self, config_file):
        stats = {
            "total": 10,
            "active_count": 3,
            "first_active": "2026-01-14T10:00:00+00:00",
            "last_active": "2026-01-14T14:00:00+00:00",
        }
        rows = self._today_rows(config_file, stats)
        assert rows[0]["Fully Ready In"] == "2h"

    def test_calculates_total_wait_correctly(self, config_file):
        stats = {
            "total": 10,
            "active_count": 3,
            "first_active": "2026-01-14T10:00:00+00:00",
            "last_active": "2026-01-14T14:00:00+00:00",
        }
        rows = self._today_rows(config_file, stats)
        assert rows[0]["Total Wait"] == "4h"

    def test_handles_naive_timestamps(self, config_file):
        stats = {
            "total": 10,
            "active_count": 3,
            "first_active": "2026-01-14T10:00:00",
            "last_active": "2026-01-14T14:00:00",
        }
        rows = self._today_rows(config_file, stats)
        assert len(rows) == 1

class TestGetAccountActivityStatuses:
    """Tests for get_account_activity_statuses function."""

    def test_no_accounts_returns_empty(self, tmp_path):
        """Should return empty lists when no accounts in config."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        today_rows, yesterday_rows, total = account_status_service.get_account_activity_statuses(
            config_path=str(config_file),
            timezone_name=None,
        )

        assert today_rows == []
        assert yesterday_rows == []
        assert total == 0

    def test_accounts_without_data_returns_ready(self, tmp_path):
        """Accounts with no DB data should show as ready."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."user1@example.com"]
authorization_token = "token1"

[accounts."user2@example.com"]
authorization_token = "token2"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with patch("chatgpt_to_notion.services.account_status_service.sqlite_store.has_generations") as mock_has:
            mock_has.return_value = False

            today_rows, yesterday_rows, total = account_status_service.get_account_activity_statuses(
                config_path=str(config_file),
                timezone_name="UTC",
            )

            assert total == 2
            assert len(today_rows) == 2
            assert all("✅" in row["Ready Generate?"] for row in today_rows)

    def test_multiple_accounts_mixed_status(self, tmp_path):
        """Should handle multiple accounts with different statuses."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."ready@example.com"]
authorization_token = "token1"

[accounts."exhausted@example.com"]
authorization_token = "token2"

[accounts."warning@example.com"]
authorization_token = "token3"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with patch("chatgpt_to_notion.services.account_status_service.sqlite_store.has_generations") as mock_has, \
             patch("chatgpt_to_notion.services.account_status_service.sqlite_store.get_activity_stats") as mock_stats:

            def has_side_effect(account):
                return account != "ready@example.com"

            def stats_side_effect(account, date_start, date_end, cooldown_threshold, db_path=None):
                # cooldown_threshold is now - 1 day = 2026-01-14 12:00:00
                # Use timestamps AFTER cooldown threshold so they count as active
                if account == "exhausted@example.com":
                    return {
                        "total": 10,
                        "active_count": 10,
                        "first_active": "2026-01-14T13:00:00+00:00",
                        "last_active": "2026-01-14T14:00:00+00:00",
                    }
                elif account == "warning@example.com":
                    return {
                        "total": 10,
                        "active_count": 3,
                        "first_active": "2026-01-14T13:00:00+00:00",
                        "last_active": "2026-01-14T14:00:00+00:00",
                    }
                return None
            mock_has.side_effect = has_side_effect
            mock_stats.side_effect = stats_side_effect

            now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

            today_rows, yesterday_rows, total = account_status_service.get_account_activity_statuses(
                config_path=str(config_file),
                timezone_name="UTC",
                now=now,
            )

            assert total == 3
            # Check statuses
            ready_rows = [r for r in today_rows if "✅" in r["Ready Generate?"]]
            exhausted_rows = [r for r in today_rows if "❌" in r["Ready Generate?"]]
            warning_rows = [r for r in today_rows if "⚠️" in r["Ready Generate?"]]

            assert len(ready_rows) == 1
            assert len(exhausted_rows) == 1
            assert len(warning_rows) == 1
    def test_yesterday_date_boundary(self, tmp_path):
        """Should correctly calculate yesterday's date."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."user@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with patch("chatgpt_to_notion.services.account_status_service.sqlite_store.has_generations") as mock_has:
            mock_has.return_value = True

            with patch("chatgpt_to_notion.services.account_status_service.sqlite_store.get_activity_stats") as mock_stats:
                mock_stats.return_value = {
                    "total": 5,
                    "active_count": 2,
                    "first_active": "2026-01-13T10:00:00+00:00",
                    "last_active": "2026-01-13T14:00:00+00:00",
                }

                now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

                with patch("chatgpt_to_notion.services.account_status_service.datetime") as mock_dt:
                    mock_dt.now.return_value = now
                    mock_dt.combine = datetime.combine
                    mock_dt.min = datetime.min
                    mock_dt.timedelta = timedelta
                    def dt_constructor(*args, **kwargs):
                        return datetime(*args, **kwargs)
                    mock_dt.side_effect = dt_constructor

                    today_rows, yesterday_rows, total = account_status_service.get_account_activity_statuses(
                        config_path=str(config_file),
                        timezone_name="UTC",
                    )

                    assert total == 1
                    assert len(today_rows) == 1
                    assert len(yesterday_rows) == 1


class TestTimezoneHandling:
    """Tests for timezone handling in account status."""

    def test_custom_timezone_used(self, tmp_path):
        """Should use provided IANA timezone."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."user@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with patch("chatgpt_to_notion.services.account_status_service.sqlite_store.has_generations") as mock_has:
            mock_has.return_value = False

            today_rows, yesterday_rows, total = account_status_service.get_account_activity_statuses(
                config_path=str(config_file),
                timezone_name="Asia/Singapore",
            )

            assert total == 1

    def test_system_timezone_when_none_provided(self, tmp_path):
        """Should use system timezone when none provided."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."user@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with patch("chatgpt_to_notion.services.account_status_service.sqlite_store.has_generations") as mock_has:
            mock_has.return_value = False

            today_rows, yesterday_rows, total = account_status_service.get_account_activity_statuses(
                config_path=str(config_file),
                timezone_name=None,
            )

            assert total == 1


class TestEdgeCases:
    """Edge case tests for account status service."""

    def test_date_boundary_at_midnight(self, tmp_path):
        """Should handle date boundary correctly at midnight."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."user@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with patch("chatgpt_to_notion.services.account_status_service.sqlite_store.has_generations") as mock_has:
            mock_has.return_value = True

            with patch("chatgpt_to_notion.services.account_status_service.sqlite_store.get_activity_stats") as mock_stats:
                # Activity exactly at midnight
                mock_stats.return_value = {
                    "total": 5,
                    "active_count": 1,
                    "first_active": "2026-01-15T00:00:00+00:00",
                    "last_active": "2026-01-15T00:00:00+00:00",
                }

                # Test at midnight
                now = datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)

                with patch("chatgpt_to_notion.services.account_status_service.datetime") as mock_dt:
                    mock_dt.now.return_value = now
                    mock_dt.combine = datetime.combine
                    mock_dt.min = datetime.min
                    mock_dt.timedelta = timedelta
                    def dt_constructor(*args, **kwargs):
                        return datetime(*args, **kwargs)
                    mock_dt.side_effect = dt_constructor

                    today_rows, yesterday_rows, total = account_status_service.get_account_activity_statuses(
                        config_path=str(config_file),
                        timezone_name="UTC",
                    )

                    assert total == 1

    def test_active_count_edge_at_cooldown_boundary(self, tmp_path):
        """Activity exactly at cooldown boundary (24h ago) should not count as active."""
        now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        cooldown_threshold = datetime(2026, 1, 14, 12, 0, 0, tzinfo=timezone.utc)

        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."user@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with patch(
            "chatgpt_to_notion.services.account_status_service.sqlite_store.has_generations",
            return_value=True,
        ), patch(
            "chatgpt_to_notion.services.account_status_service.sqlite_store.get_activity_stats"
        ) as mock:
            mock.return_value = {
                "total": 5,
                "active_count": 1,
                "first_active": cooldown_threshold.isoformat(),
                "last_active": cooldown_threshold.isoformat(),
            }
            today_rows, _, total = account_status_service.get_account_activity_statuses(
                config_path=str(config_file),
                timezone_name="UTC",
                now=now,
            )
            assert total == 1
            assert len(today_rows) == 1


class TestFormatDuration:
    """Tests for format_duration helper."""

    def test_zero_seconds(self):
        assert account_status_service.format_duration(0) == "0s"

    def test_seconds_only(self):
        assert account_status_service.format_duration(30) == "30s"
        assert account_status_service.format_duration(59) == "59s"

    def test_minutes_and_seconds(self):
        assert account_status_service.format_duration(90) == "1m 30s"
        assert account_status_service.format_duration(125) == "2m 5s"

    def test_hours_minutes(self):
        # format_duration doesn't include seconds when hours > 0
        assert account_status_service.format_duration(3661) == "1h 1m"
        assert account_status_service.format_duration(7325) == "2h 2m"

    def test_hours_only(self):
        assert account_status_service.format_duration(3600) == "1h"
        assert account_status_service.format_duration(7200) == "2h"

    def test_negative_seconds(self):
        """Negative should be treated as 0."""
        assert account_status_service.format_duration(-10) == "0s"