"""Tests for history_service module."""

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatgpt_to_notion.domain.models import ChatGPTImageGeneration, RuntimeOptions, AccountConfig, ResolvedConfig, NotionConfig
from chatgpt_to_notion.services import history_service
from chatgpt_to_notion.adapters import sqlite_store


def _mock_resolved_config(account_name="test_account"):
    """Helper to create a mocked ResolvedConfig."""
    return ResolvedConfig(
        account_name=account_name,
        account=AccountConfig(
            authorization_token="token",
            user_agent="TestAgent/1.0",
            cookie_string_base64="dGVzdA==",
        ),
        notion=NotionConfig(
            api_key="secret_test",
            database_id="test_db",
        ),
        history_folder=None,
        image_folder=None,
        delete_library_queries=[],
        remove_all_library_images=False,
        fetch_conversation_delay=0.0,
        fetch_batch_size=0,
    )


@pytest.fixture(autouse=True)
def init_db(isolated_db):
    """Ensure database is initialized for each test."""
    sqlite_store.init_db()


class TestSaveGenerations:
    """Tests for save_generations function."""

    def test_save_generations_empty_list(self, isolated_db):
        """Empty list should not raise."""
        history_service.save_generations("test_account", [])
        # Should not raise

    def test_save_generations_normal(self, isolated_db):
        """Normal save should insert generations."""
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Test prompt",
        )
        history_service.save_generations("test_account", [gen])

        loaded = history_service.load_image_generations("test_account")
        assert len(loaded) == 1
        assert loaded[0].id == "gen_1"
        assert loaded[0].prompt == "Test prompt"

    def test_save_generations_duplicate_ids(self, isolated_db):
        """Duplicate IDs should update existing records (upsert)."""
        gen1 = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Original",
        )
        history_service.save_generations("test_account", [gen1])

        gen2 = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Updated",
        )
        history_service.save_generations("test_account", [gen2])

        loaded = history_service.load_image_generations("test_account")
        assert len(loaded) == 1
        assert loaded[0].prompt == "Updated"


class TestGetUploadedGenerationIds:
    """Tests for get_uploaded_generation_ids function."""

    def test_empty_account_returns_empty_set(self, isolated_db):
        """Account with no uploaded generations returns empty set."""
        result = history_service.get_uploaded_generation_ids("empty_account")
        assert result == set()

    def test_account_with_uploaded_returns_ids(self, isolated_db):
        """Should return IDs of uploaded generations."""
        gen1 = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Test",
        )
        gen2 = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_2",
            conversation_id="conv_2",
            message_id="msg_2",
            asset_pointer="asset_2",
            url="https://example.com/img2.png",
            prompt="Test",
        )
        history_service.save_generations("test_account", [gen1, gen2])
        history_service.mark_generations_uploaded("test_account", ["gen_1"])

        result = history_service.get_uploaded_generation_ids("test_account")
        assert result == {"gen_1"}


class TestMarkGenerationsUploaded:
    """Tests for mark_generations_uploaded function."""

    def test_empty_account_noop(self, isolated_db):
        """Empty account should not raise."""
        history_service.mark_generations_uploaded("empty_account", ["gen_1"])
        # Should not raise

    def test_empty_ids_noop(self, isolated_db):
        """Empty IDs list should not raise."""
        history_service.mark_generations_uploaded("test_account", [])
        # Should not raise

    def test_normal_mark(self, isolated_db):
        """Normal mark should update uploaded flag."""
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Test",
        )
        history_service.save_generations("test_account", [gen])
        history_service.mark_generations_uploaded("test_account", ["gen_1"])

        uploaded = history_service.get_uploaded_generation_ids("test_account")
        assert uploaded == {"gen_1"}

    def test_mark_multiple_ids(self, isolated_db):
        """Marking multiple IDs should update all."""
        gen1 = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Test",
        )
        gen2 = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_2",
            conversation_id="conv_2",
            message_id="msg_2",
            asset_pointer="asset_2",
            url="https://example.com/img2.png",
            prompt="Test",
        )
        history_service.save_generations("test_account", [gen1, gen2])
        history_service.mark_generations_uploaded("test_account", ["gen_1", "gen_2"])

        uploaded = history_service.get_uploaded_generation_ids("test_account")
        assert uploaded == {"gen_1", "gen_2"}


class TestLoadImageGenerations:
    """Tests for load_image_generations function."""

    def test_load_empty_db(self, isolated_db):
        """Empty database returns empty list."""
        result = history_service.load_image_generations("test_account")
        assert result == []

    def test_load_with_include_uploaded_false_excludes_uploaded(self, isolated_db):
        """include_uploaded=False should exclude uploaded generations."""
        gen1 = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Test",
        )
        gen2 = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_2",
            conversation_id="conv_2",
            message_id="msg_2",
            asset_pointer="asset_2",
            url="https://example.com/img2.png",
            prompt="Test",
        )
        history_service.save_generations("test_account", [gen1, gen2])
        history_service.mark_generations_uploaded("test_account", ["gen_1"])

        result = history_service.load_image_generations("test_account", include_uploaded=False)
        assert len(result) == 1
        assert result[0].id == "gen_2"

    def test_load_with_include_uploaded_true_includes_all(self, isolated_db):
        """include_uploaded=True should include all generations."""
        gen1 = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Test",
        )
        gen2 = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_2",
            conversation_id="conv_2",
            message_id="msg_2",
            asset_pointer="asset_2",
            url="https://example.com/img2.png",
            prompt="Test",
        )
        history_service.save_generations("test_account", [gen1, gen2])
        history_service.mark_generations_uploaded("test_account", ["gen_1"])

        result = history_service.load_image_generations("test_account", include_uploaded=True)
        assert len(result) == 2
        assert {g.id for g in result} == {"gen_1", "gen_2"}

    def test_load_with_keep_days_filters_old(self, isolated_db):
        """Old generations beyond keep_days should be excluded."""
        old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        recent_time = datetime.now(timezone.utc).isoformat()

        gen_old = ChatGPTImageGeneration(
            created_at=old_time,
            id="gen_old",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/old.png",
            prompt="Old",
        )
        gen_recent = ChatGPTImageGeneration(
            created_at=recent_time,
            id="gen_recent",
            conversation_id="conv_2",
            message_id="msg_2",
            asset_pointer="asset_2",
            url="https://example.com/recent.png",
            prompt="Recent",
        )
        history_service.save_generations("test_account", [gen_old, gen_recent], keep_days=3)

        result = history_service.load_image_generations("test_account", keep_days=3)
        assert len(result) == 1
        assert result[0].id == "gen_recent"

    def test_load_with_timezone_conversion(self, isolated_db):
        """timezone_name should convert timestamps - just verify it runs without error."""
        from zoneinfo import ZoneInfo

        utc_time = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        gen = ChatGPTImageGeneration(
            created_at=utc_time,
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Test",
        )
        history_service.save_generations("test_account", [gen])

        # Load with US/Eastern timezone - just verify it doesn't crash
        result = history_service.load_image_generations("test_account", timezone_name="US/Eastern")
        assert len(result) == 1

    def test_load_with_include_uploaded_filter(self, isolated_db):
        """include_uploaded=True should include uploaded, False should exclude."""
        gen1 = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Test",
        )
        gen2 = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_2",
            conversation_id="conv_2",
            message_id="msg_2",
            asset_pointer="asset_2",
            url="https://example.com/img2.png",
            prompt="Test",
        )
        history_service.save_generations("test_account", [gen1, gen2])
        history_service.mark_generations_uploaded("test_account", ["gen_1"])

        # include_uploaded=False (default behavior)
        result_exclude = history_service.load_image_generations("test_account", include_uploaded=False)
        assert len(result_exclude) == 1
        assert result_exclude[0].id == "gen_2"

        # include_uploaded=True
        result_include = history_service.load_image_generations("test_account", include_uploaded=True)
        assert len(result_include) == 2


class TestFetchGenerationsForConversation:
    """Tests for fetch_generations_for_conversation function."""

    @pytest.mark.asyncio
    async def test_empty_items_returns_empty_list(self):
        """Empty items list should return empty list."""
        mock_session = AsyncMock()
        result = await history_service.fetch_generations_for_conversation(
            mock_session, "conv_1", [], {"Authorization": "Bearer test"}, set()
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_all_items_existing_returns_empty(self, isolated_db):
        """If all items already in DB, should return empty without API call."""
        existing_gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="existing_gen",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/existing.png",
            prompt="Existing",
        )
        history_service.save_generations("test_account", [existing_gen])

        mock_items = [
            {"id": "existing_gen", "conversation_id": "conv_1", "message_id": "msg_1", "asset_pointer": "asset_1", "url": "https://example.com/existing.png", "created_at": 1704067200},
        ]

        mock_session = AsyncMock()

        with patch("chatgpt_to_notion.services.history_service.chatgpt_api.get_conversation_details") as mock_get_detail, \
             patch("chatgpt_to_notion.services.history_service.get_prompt_from_image_node_in_conversation") as mock_prompt, \
             patch("chatgpt_to_notion.services.history_service.resolve_config", return_value=_mock_resolved_config()):

            mock_get_detail.return_value = {"mapping": {}}
            mock_prompt.return_value = "Prompt"

            result = await history_service.fetch_generations_for_conversation(
                mock_session, "conv_1", mock_items, {"Authorization": "Bearer test"}, {"existing_gen"}
            )

            assert result == []
            mock_get_detail.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_items_fetched(self, isolated_db):
        """New items not in DB should be fetched from API."""
        mock_items = [
            {"id": "new_gen", "conversation_id": "conv_1", "message_id": "msg_1", "asset_pointer": "asset_1", "url": "https://example.com/new.png", "created_at": 1704067200},
        ]

        mock_session = AsyncMock()

        with patch("chatgpt_to_notion.services.history_service.chatgpt_api.get_conversation_details") as mock_get_detail, \
             patch("chatgpt_to_notion.services.history_service.get_prompt_from_image_node_in_conversation") as mock_prompt, \
             patch("chatgpt_to_notion.services.history_service.resolve_config", return_value=_mock_resolved_config()):

            mock_get_detail.return_value = {"mapping": {}}
            mock_prompt.return_value = "Test Prompt"

            result = await history_service.fetch_generations_for_conversation(
                mock_session, "conv_1", mock_items, {"Authorization": "Bearer test"}, set()
            )

            assert len(result) == 1
            assert result[0].id == "new_gen"
            assert result[0].prompt == "Test Prompt"
            mock_get_detail.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_new_items_all_processed(self, isolated_db):
        """Multiple new items should all be processed."""
        mock_items = [
            {"id": "gen_1", "conversation_id": "conv_1", "message_id": "msg_1", "asset_pointer": "asset_1", "url": "https://example.com/img1.png", "created_at": 1704067200},
            {"id": "gen_2", "conversation_id": "conv_1", "message_id": "msg_2", "asset_pointer": "asset_2", "url": "https://example.com/img2.png", "created_at": 1704067200},
            {"id": "gen_3", "conversation_id": "conv_1", "message_id": "msg_3", "asset_pointer": "asset_3", "url": "https://example.com/img3.png", "created_at": 1704067200},
        ]

        mock_session = AsyncMock()

        with patch("chatgpt_to_notion.services.history_service.chatgpt_api.get_conversation_details") as mock_get_detail, \
             patch("chatgpt_to_notion.services.history_service.get_prompt_from_image_node_in_conversation") as mock_prompt, \
             patch("chatgpt_to_notion.services.history_service.resolve_config", return_value=_mock_resolved_config()):

            mock_get_detail.return_value = {"mapping": {}}
            mock_prompt.side_effect = ["Prompt 1", "Prompt 2", "Prompt 3"]

            result = await history_service.fetch_generations_for_conversation(
                mock_session, "conv_1", mock_items, {"Authorization": "Bearer test"}, set()
            )

            assert len(result) == 3
            assert {g.id for g in result} == {"gen_1", "gen_2", "gen_3"}
            assert mock_get_detail.call_count == 1


class TestFetchImageGenerations:
    """Tests for fetch_image_generations async function."""

    @pytest.mark.asyncio
    async def test_no_new_items_returns_existing(self, isolated_db):
        """When API returns only already-saved items, should return existing from DB."""
        existing_gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="existing_gen",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/existing.png",
            prompt="Existing",
        )
        history_service.save_generations("test_account", [existing_gen])

        mock_items = [
            {"id": "existing_gen", "conversation_id": "conv_1", "message_id": "msg_1", "asset_pointer": "asset_1", "url": "https://example.com/existing.png", "created_at": 1704067200},
        ]

        with patch("chatgpt_to_notion.services.history_service.get_image_generations") as mock_get, \
             patch("chatgpt_to_notion.services.history_service.chatgpt_api.get_conversation_details_batch") as mock_batch, \
             patch("chatgpt_to_notion.services.history_service.get_prompt_from_image_node_in_conversation") as mock_prompt, \
             patch("chatgpt_to_notion.services.history_service.get_provider_context") as mock_provider, \
             patch("chatgpt_to_notion.services.history_service.resolve_config", return_value=_mock_resolved_config()):

            mock_get.return_value = {"items": mock_items}
            mock_batch.return_value = {"conv_1": {"mapping": {}}}
            mock_prompt.return_value = "Prompt"
            mock_provider.return_value.headers = {"Authorization": "Bearer test"}

            options = RuntimeOptions(account="test_account")
            remote, local = await history_service.fetch_image_generations(limit=100, options=options)

            assert len(remote) == 1
            assert remote[0].id == "existing_gen"
            assert len(local) == 1
            assert local[0].id == "existing_gen"
            mock_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_items_fetched_and_saved(self, isolated_db):
        """New items from API should be fetched and saved to DB."""
        mock_items = [
            {"id": "new_gen", "conversation_id": "conv_1", "message_id": "msg_1", "asset_pointer": "asset_1", "url": "https://example.com/new.png", "created_at": 1704067200},
        ]

        with patch("chatgpt_to_notion.services.history_service.get_image_generations") as mock_get, \
             patch("chatgpt_to_notion.services.history_service.chatgpt_api.get_conversation_details_batch") as mock_batch, \
             patch("chatgpt_to_notion.services.history_service.get_prompt_from_image_node_in_conversation") as mock_prompt, \
             patch("chatgpt_to_notion.services.history_service.get_provider_context") as mock_provider, \
             patch("chatgpt_to_notion.services.history_service.resolve_config", return_value=_mock_resolved_config()):

            mock_get.return_value = {"items": mock_items}
            mock_batch.return_value = {"conv_1": {"mapping": {}}}
            mock_prompt.return_value = "New Prompt"
            mock_provider.return_value.headers = {"Authorization": "Bearer test"}

            options = RuntimeOptions(account="test_account")
            remote, local = await history_service.fetch_image_generations(limit=100, options=options)

            assert len(remote) == 1
            assert remote[0].id == "new_gen"
            assert len(local) == 1
            assert local[0].id == "new_gen"
            assert local[0].prompt == "New Prompt"

            # Verify saved to DB
            loaded = history_service.load_image_generations("test_account")
            assert len(loaded) == 1
            assert loaded[0].id == "new_gen"

    @pytest.mark.asyncio
    async def test_existing_ids_filtered_out(self, isolated_db):
        """Items already in DB should be filtered out from API fetch (no re-fetch)."""
        existing_gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="existing_gen",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/existing.png",
            prompt="Existing",
        )
        history_service.save_generations("test_account", [existing_gen])

        mock_items = [
            {"id": "existing_gen", "conversation_id": "conv_1", "message_id": "msg_1", "asset_pointer": "asset_1", "url": "https://example.com/existing.png", "created_at": 1704067200},
            {"id": "new_gen", "conversation_id": "conv_2", "message_id": "msg_2", "asset_pointer": "asset_2", "url": "https://example.com/new.png", "created_at": 1704067200},
        ]

        with patch("chatgpt_to_notion.services.history_service.get_image_generations") as mock_get, \
             patch("chatgpt_to_notion.services.history_service.chatgpt_api.get_conversation_details_batch") as mock_batch, \
             patch("chatgpt_to_notion.services.history_service.get_prompt_from_image_node_in_conversation") as mock_prompt, \
             patch("chatgpt_to_notion.services.history_service.get_provider_context") as mock_provider, \
             patch("chatgpt_to_notion.services.history_service.resolve_config", return_value=_mock_resolved_config()):

            mock_get.return_value = {"items": mock_items}
            mock_batch.return_value = {"conv_1": {"mapping": {}}, "conv_2": {"mapping": {}}}
            mock_prompt.return_value = "Prompt"
            mock_provider.return_value.headers = {"Authorization": "Bearer test"}

            options = RuntimeOptions(account="test_account")
            remote, local = await history_service.fetch_image_generations(limit=100, options=options)

            # remote should include both (both in DB and in api_ids)
            assert len(remote) == 2
            assert {g.id for g in remote} == {"existing_gen", "new_gen"}

            # get_conversation_details_batch should only be called for conv_2 (existing_gen filtered out)
            called_conv_ids = mock_batch.call_args[0][1]  # second arg is conv_ids
            assert called_conv_ids == ["conv_2"]

    @pytest.mark.asyncio
    async def test_conversation_detail_missing_filtered(self, isolated_db):
        """Missing conversation detail should filter out that generation."""
        mock_items = [
            {"id": "gen_1", "conversation_id": "conv_1", "message_id": "msg_1", "asset_pointer": "asset_1", "url": "https://example.com/img1.png", "created_at": 1704067200},
        ]

        with patch("chatgpt_to_notion.services.history_service.get_image_generations") as mock_get, \
             patch("chatgpt_to_notion.services.history_service.chatgpt_api.get_conversation_details_batch") as mock_batch, \
             patch("chatgpt_to_notion.services.history_service.get_prompt_from_image_node_in_conversation") as mock_prompt, \
             patch("chatgpt_to_notion.services.history_service.get_provider_context") as mock_provider, \
             patch("chatgpt_to_notion.services.history_service.resolve_config", return_value=_mock_resolved_config()):

            mock_get.return_value = {"items": mock_items}
            mock_batch.return_value = {}  # Empty - conversation not found
            mock_prompt.return_value = "Prompt"
            mock_provider.return_value.headers = {"Authorization": "Bearer test"}

            options = RuntimeOptions(account="test_account")
            remote, local = await history_service.fetch_image_generations(limit=100, options=options)

            # Should filter out generation with missing conversation detail
            assert len(remote) == 0
            assert len(local) == 0

    @pytest.mark.asyncio
    async def test_rate_limiting_logs_info(self, isolated_db):
        """Rate limiting config should log info message."""
        mock_items = [{"id": "gen_1", "conversation_id": "conv_1", "message_id": "msg_1", "asset_pointer": "asset_1", "url": "https://example.com/img1.png", "created_at": 1704067200}]

        with patch("chatgpt_to_notion.services.history_service.get_image_generations") as mock_get, \
             patch("chatgpt_to_notion.services.history_service.chatgpt_api.get_conversation_details_batch") as mock_batch, \
             patch("chatgpt_to_notion.services.history_service.get_prompt_from_image_node_in_conversation") as mock_prompt, \
             patch("chatgpt_to_notion.services.history_service.logger") as mock_logger, \
             patch("chatgpt_to_notion.services.history_service.get_provider_context") as mock_provider, \
             patch("chatgpt_to_notion.services.history_service.resolve_config") as mock_resolve:

            mock_get.return_value = {"items": mock_items}
            mock_batch.return_value = {"conv_1": {"mapping": {}}}
            mock_prompt.return_value = "Prompt"
            mock_provider.return_value.headers = {"Authorization": "Bearer test"}
            mock_resolve.return_value = _mock_resolved_config(account_name="test_account")
            mock_resolve.return_value.fetch_conversation_delay = 1.0
            mock_resolve.return_value.fetch_batch_size = 5

            options = RuntimeOptions(account="test_account", fetch_conversation_delay=1.0, fetch_batch_size=5)
            await history_service.fetch_image_generations(limit=100, options=options)

            # Verify info log about rate limiting
            mock_logger.info.assert_called()
            format_str = mock_logger.info.call_args[0][0]
            args = mock_logger.info.call_args[0][1:]
            assert "Rate limiting enabled" in format_str
            assert "delay=%.1fs" in format_str
            assert args == (1.0, 5, 5.0)


def _make_mock_client_session(mock_session):
    """Create a mock ClientSession that returns mock_session from __aenter__."""
    mock_client_session = MagicMock()
    mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_session.__aexit__ = AsyncMock(return_value=None)
    return mock_client_session


def _make_mock_response(status=200, content_type="image/png", data=b"data", headers=None):
    """Create a mock HTTP response."""
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.content_type = content_type
    mock_response.headers = headers or {}

    async def mock_iter_chunked(_size):
        yield data

    mock_response.content = MagicMock()
    mock_response.content.iter_chunked = mock_iter_chunked
    return mock_response


def _make_mock_cm(response):
    """Create a mock async context manager for session.get()."""
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=response)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    return mock_cm


class TestDownloadAllImages:
    """Tests for download_all_images function."""

    @pytest.mark.asyncio
    async def test_empty_list_noop(self, tmp_path):
        """Empty generations list should be a no-op."""
        await history_service.download_all_images([], str(tmp_path))
        # Should not raise

    @pytest.mark.asyncio
    async def test_download_success(self, tmp_path, isolated_db):
        """Successful download should save file."""
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Test prompt",
        )

        mock_response = _make_mock_response(data=b"png data")
        mock_cm = _make_mock_cm(mock_response)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)

        mock_client_session = _make_mock_client_session(mock_session)

        with patch("chatgpt_to_notion.services.history_service.aiohttp.ClientSession", return_value=mock_client_session):
            await history_service.download_all_images([gen], str(tmp_path))

            mock_session.get.assert_called_once()
            args, _ = mock_session.get.call_args
            assert args[0] == "https://example.com/img1.png"
            files = list(tmp_path.glob("*.png"))
            assert len(files) == 1

    @pytest.mark.asyncio
    async def test_download_404_skips(self, tmp_path, caplog):
        """404 response should skip generation."""
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/notfound.png",
            prompt="Test",
        )

        mock_response = _make_mock_response(status=404)
        mock_cm = _make_mock_cm(mock_response)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)

        mock_client_session = _make_mock_client_session(mock_session)

        with patch("chatgpt_to_notion.services.history_service.aiohttp.ClientSession", return_value=mock_client_session):
            await history_service.download_all_images([gen], str(tmp_path))

            files = list(tmp_path.glob("*.png"))
            assert len(files) == 0

    @pytest.mark.asyncio
    async def test_download_retry_on_5xx(self, tmp_path, isolated_db):
        """5xx errors should be retried."""
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Test",
        )

        fail_response = _make_mock_response(status=500)
        success_response = _make_mock_response(data=b"png data")

        fail_cm = _make_mock_cm(fail_response)
        success_cm = _make_mock_cm(success_response)

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=[fail_cm, success_cm])

        mock_client_session = _make_mock_client_session(mock_session)

        with patch("chatgpt_to_notion.services.history_service.aiohttp.ClientSession", return_value=mock_client_session):
            try:
                await history_service.download_all_images([gen], str(tmp_path))
            except Exception:
                pass

            assert mock_session.get.call_count >= 2

    @pytest.mark.asyncio
    async def test_download_429_respects_retry_after(self, tmp_path):
        """429 should respect Retry-After header."""
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Test",
        )

        fail_response = _make_mock_response(status=429, headers={"Retry-After": "2"})
        success_response = _make_mock_response(data=b"png data")

        fail_cm = _make_mock_cm(fail_response)
        success_cm = _make_mock_cm(success_response)

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=[fail_cm, success_cm])

        mock_client_session = _make_mock_client_session(mock_session)

        with patch("chatgpt_to_notion.services.history_service.aiohttp.ClientSession", return_value=mock_client_session):
            try:
                await history_service.download_all_images([gen], str(tmp_path))
            except Exception:
                pass

            assert mock_session.get.call_count >= 2


class TestDownloadImage:
    """Tests for download_image function."""

    @pytest.mark.asyncio
    async def test_download_works(self, tmp_path):
        """Download should work with valid session and URL."""
        file_path = tmp_path / "image.png"

        mock_response = _make_mock_response(data=b"png data")
        mock_cm = _make_mock_cm(mock_response)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)

        await history_service.download_image(mock_session, "https://example.com/img.png", str(file_path))

        assert file_path.exists()
        assert file_path.read_bytes() == b"png data"

    @pytest.mark.asyncio
    async def test_non_png_download_works(self, tmp_path):
        """Non-PNG content-type should download without error."""
        file_path = tmp_path / "image.jpg"

        mock_response = _make_mock_response(content_type="image/jpeg", data=b"jpeg data")
        mock_cm = _make_mock_cm(mock_response)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_cm)

        await history_service.download_image(mock_session, "https://example.com/img.jpg", str(file_path))

        assert file_path.exists()
        assert file_path.read_bytes() == b"jpeg data"


class TestGetHeaders:
    """Tests for get_headers function."""

    def test_returns_headers_from_provider_context(self, tmp_path):
        """Should return headers from resolved provider context."""
        # Create a minimal config file
        config_text = """
[shared]
user_agent = "TestAgent/1.0"
cookie_string_base64 = "dGVzdA=="

[notion]
api_key = "secret_test"
database_id = "test_db"

[accounts."test@example.com"]
authorization_token = "token1"
"""
        config_file = tmp_path / "config.toml"
        config_file.write_text(config_text.strip())

        from chatgpt_to_notion.domain.models import RuntimeOptions
        options = RuntimeOptions(
            config_path=str(config_file),
            account="test@example.com",
        )

        headers = history_service.get_headers(options)

        assert "Authorization" in headers
        assert "User-Agent" in headers
        assert "Cookie" in headers
        assert headers["Cookie"] == "test"