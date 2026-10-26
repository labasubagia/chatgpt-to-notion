"""Tests for configuration validation - email keys, overrides, edge cases."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from chatgpt_to_notion.adapters.config_loader import _load_toml_config, config_loader
from chatgpt_to_notion.domain.models import AppConfig, AccountConfig


class TestAccountKeyEmailValidation:
    """Tests for account key email format validation."""

    def test_valid_email_keys_pass(self, tmp_path):
        """Valid email keys should pass validation."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."user@example.com"]
authorization_token = "token1"

[accounts."test.user@domain.org"]
authorization_token = "token2"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        app_config = _load_toml_config(str(config_file))
        assert app_config is not None
        assert "user@example.com" in app_config.accounts
        assert "test.user@domain.org" in app_config.accounts

    def test_invalid_email_keys_fail(self, tmp_path):
        """Non-email keys should raise ValidationError."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."not-an-email"]
authorization_token = "token1"

[accounts."also-invalid"]
authorization_token = "token2"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with pytest.raises(ValidationError, match="Account keys must be valid email addresses"):
            _load_toml_config(str(config_file))

    def test_mixed_valid_invalid_fails(self, tmp_path):
        """One invalid key should fail entire config."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."valid@example.com"]
authorization_token = "token1"

[accounts."invalid-key"]
authorization_token = "token2"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with pytest.raises(ValidationError, match="Account keys must be valid email addresses"):
            _load_toml_config(str(config_file))

    def test_empty_account_key_fails(self, tmp_path):
        """Empty string key should fail."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts.""]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with pytest.raises(ValidationError, match="Account keys must be valid email addresses"):
            _load_toml_config(str(config_file))


class TestPerAccountNotionDatabaseOverride:
    """Tests for per-account Notion database ID override."""

    def test_global_database_used_by_default(self, tmp_path):
        """Accounts without override use global database_id."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "global_db_id"

[accounts."user1@example.com"]
authorization_token = "token1"

[accounts."user2@example.com"]
authorization_token = "token2"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        from chatgpt_to_notion.domain.models import RuntimeOptions
        options = RuntimeOptions(account="user1@example.com")
        resolved = config_loader.resolve(options)
        assert resolved.notion.database_id == "global_db_id"

    def test_per_account_override_works(self, tmp_path):
        """Per-account notion_database_id should override global."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "global_db_id"

[accounts."user1@example.com"]
authorization_token = "token1"
notion_database_id = "account1_db"

[accounts."user2@example.com"]
authorization_token = "token2"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        from chatgpt_to_notion.domain.models import RuntimeOptions
        options1 = RuntimeOptions(account="user1@example.com")
        options2 = RuntimeOptions(account="user2@example.com")

        resolved1 = config_loader.resolve(options1)
        resolved2 = config_loader.resolve(options2)

        assert resolved1.notion.database_id == "account1_db"
        assert resolved2.notion.database_id == "global_db_id"

    def test_per_account_fetch_conversation_delay_override(self, tmp_path):
        """Per-account fetch_conversation_delay should override global."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="
fetch_conversation_delay = 5.0

[notion]
api_key = "secret_test"
database_id = "global_db"

[accounts."user1@example.com"]
authorization_token = "token1"
fetch_conversation_delay = 1.5

[accounts."user2@example.com"]
authorization_token = "token2"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        from chatgpt_to_notion.domain.models import RuntimeOptions
        options1 = RuntimeOptions(account="user1@example.com")
        options2 = RuntimeOptions(account="user2@example.com")

        resolved1 = config_loader.resolve(options1)
        resolved2 = config_loader.resolve(options2)

        assert resolved1.fetch_conversation_delay == 1.5
        assert resolved2.fetch_conversation_delay == 5.0

    def test_per_account_cookie_override(self, tmp_path):
        """Per-account cookie_string_base64 should override global."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Z2xvYmFsX2Nvb2tpZQ=="  # global_cookie

[notion]
api_key = "secret_test"
database_id = "global_db"

[accounts."user1@example.com"]
authorization_token = "token1"
cookie_string_base64 = "YWNjb3VudDFfY29va2ll"  # account1_cookie

[accounts."user2@example.com"]
authorization_token = "token2"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        from chatgpt_to_notion.domain.models import RuntimeOptions
        options1 = RuntimeOptions(account="user1@example.com")
        options2 = RuntimeOptions(account="user2@example.com")

        resolved1 = config_loader.resolve(options1)
        resolved2 = config_loader.resolve(options2)

        import base64
        assert base64.b64decode(resolved1.account.cookie_string_base64).decode() == "account1_cookie"
        assert base64.b64decode(resolved2.account.cookie_string_base64).decode() == "global_cookie"


class TestConfigLoadingEdgeCases:
    """Tests for config loading edge cases."""

    def test_missing_config_file_returns_none(self):
        """Non-existent config path should return None."""
        result = _load_toml_config("/nonexistent/path/config.toml")
        assert result is None

    def test_empty_accounts_dict_allowed(self, tmp_path):
        """Config with no accounts should be valid (but config_loader.resolve will fail)."""
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

        app_config = _load_toml_config(str(config_file))
        assert app_config is not None
        assert app_config.accounts == {}

    def test_get_account_names_returns_keys(self, tmp_path):
        """config_loader.account_names should return account keys in insertion order."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."zebra@example.com"]
authorization_token = "token1"

[accounts."alpha@example.com"]
authorization_token = "token2"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        names = config_loader.account_names(str(config_file))
        # Returns in insertion order (Python 3.7+ dict preserves insertion order)
        assert names == ["zebra@example.com", "alpha@example.com"]

    def test_missing_shared_section_has_defaults(self, tmp_path):
        """Missing [shared] section uses defaults (all fields optional)."""
        config_text = """
[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."user@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        # Shared fields are all optional with defaults, so this should pass
        app_config = _load_toml_config(str(config_file))
        assert app_config is not None
        assert app_config.shared.user_agent is None
        assert app_config.shared.cookie_string_base64 is None

    def test_missing_notion_section_has_defaults(self, tmp_path):
        """Missing [notion] section uses defaults (fields optional)."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[accounts."user@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        # Notion fields are optional with defaults
        app_config = _load_toml_config(str(config_file))
        assert app_config is not None
        assert app_config.notion.api_key is None
        assert app_config.notion.database_id is None

    def test_missing_authorization_token_fails(self, tmp_path):
        """Account without authorization_token should fail."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."user@example.com"]
# authorization_token missing
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with pytest.raises(ValidationError):
            _load_toml_config(str(config_file))

    def test_extra_fields_in_shared_forbidden(self, tmp_path):
        """Extra fields in [shared] section should be rejected."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="
extra_field = "not allowed"

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."user@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _load_toml_config(str(config_file))
            _load_toml_config(str(config_file))

    def test_extra_fields_in_account_forbidden(self, tmp_path):
        """Extra fields in [accounts.*] section should be rejected."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."user@example.com"]
authorization_token = "token1"
extra_field = "not allowed"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _load_toml_config(str(config_file))


class TestResolveConfigWithOptions:
    """Tests for config_loader.resolve with RuntimeOptions."""

    def test_account_option_selects_account(self, tmp_path):
        """--account option should select specific account."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "global_db"

[accounts."user1@example.com"]
authorization_token = "token1"

[accounts."user2@example.com"]
authorization_token = "token2"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        from chatgpt_to_notion.domain.models import RuntimeOptions
        options = RuntimeOptions(account="user2@example.com")
        resolved = config_loader.resolve(options)

        assert resolved.account_name == "user2@example.com"
        assert resolved.account.authorization_token == "token2"

    def test_account_option_invalid_raises(self, tmp_path):
        """Invalid --account should raise ValueError."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "global_db"

[accounts."user1@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        from chatgpt_to_notion.domain.models import RuntimeOptions
        options = RuntimeOptions(account="nonexistent@example.com")

        with pytest.raises(ValueError, match="Unknown account: nonexistent@example.com"):
            config_loader.resolve(options)

    def test_no_account_with_multiple_accounts_raises(self, tmp_path):
        """No --account with multiple accounts should raise ValueError."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "global_db"

[accounts."user1@example.com"]
authorization_token = "token1"

[accounts."user2@example.com"]
authorization_token = "token2"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        from chatgpt_to_notion.domain.models import RuntimeOptions
        options = RuntimeOptions()  # no account specified

        with pytest.raises(ValueError, match="No account selected"):
            config_loader.resolve(options)

    def test_single_account_auto_selected(self, tmp_path):
        """Single account should be auto-selected when no --account given."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="

[notion]
api_key = "secret_test"
database_id = "global_db"

[accounts."user1@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        from chatgpt_to_notion.domain.models import RuntimeOptions
        options = RuntimeOptions()  # no account specified
        resolved = config_loader.resolve(options)

        assert resolved.account_name == "user1@example.com"

    def test_cli_fetch_conversation_delay_overrides_config(self, tmp_path):
        """CLI fetch_conversation_delay should override config."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="
fetch_conversation_delay = 5.0

[notion]
api_key = "secret_test"
database_id = "global_db"

[accounts."user@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        from chatgpt_to_notion.domain.models import RuntimeOptions
        options = RuntimeOptions(account="user@example.com", fetch_conversation_delay=2.5)
        resolved = config_loader.resolve(options)

        assert resolved.fetch_conversation_delay == 2.5

    def test_cli_fetch_batch_size_overrides_config(self, tmp_path):
        """CLI fetch_batch_size should override config."""
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "Y29va2llX2RhdGE="
fetch_batch_size = 10

[notion]
api_key = "secret_test"
database_id = "global_db"

[accounts."user@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        from chatgpt_to_notion.domain.models import RuntimeOptions
        options = RuntimeOptions(account="user@example.com", fetch_batch_size=5)
        resolved = config_loader.resolve(options)

        assert resolved.fetch_batch_size == 5