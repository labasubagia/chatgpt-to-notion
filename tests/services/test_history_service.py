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
        gen2 = gen1.model_copy(update={"id": "gen_2"})
        sqlite_store.upsert_generations("test_account", [gen1, gen2], isolated_db)
        sqlite_store.mark_uploaded("test_account", {"gen_1"}, isolated_db)
        result = history_service.get_uploaded_generation_ids("test_account")
        assert result == {"gen_1"}


class TestMarkGenerationsUploaded:
    """Tests for mark_generations_uploaded function."""
    def test_empty_account_noop(self, isolated_db):
        """Empty account should not raise."""
        history_service.mark_generations_uploaded("empty_account", ["gen_1"])

    def test_empty_ids_noop(self, isolated_db):
        """Empty IDs list should not raise."""
        history_service.mark_generations_uploaded("test_account", [])

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
        sqlite_store.upsert_generations("test_account", [gen], isolated_db)
        history_service.mark_generations_uploaded("test_account", ["gen_1"])
        conn = sqlite_store._get_connection(isolated_db)
        row = conn.execute(
            "SELECT uploaded_at FROM image_generations WHERE id = ?",
            ("gen_1",),
        ).fetchone()
        conn.close()
        assert row["uploaded_at"] != ""

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
        gen2 = gen1.model_copy(update={"id": "gen_2"})
        sqlite_store.upsert_generations("test_account", [gen1, gen2], isolated_db)
        history_service.mark_generations_uploaded("test_account", ["gen_1", "gen_2"])
        conn = sqlite_store._get_connection(isolated_db)
        rows = conn.execute(
            "SELECT id, uploaded_at FROM image_generations WHERE id IN (?, ?)",
            ("gen_1", "gen_2"),
        ).fetchall()
        conn.close()
        assert all(row["uploaded_at"] != "" for row in rows)


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
        gen2 = gen1.model_copy(update={"id": "gen_2"})
        sqlite_store.upsert_generations("test_account", [gen1, gen2], isolated_db)
        sqlite_store.mark_uploaded("test_account", {"gen_1"}, isolated_db)
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
        gen2 = gen1.model_copy(update={"id": "gen_2"})
        sqlite_store.upsert_generations("test_account", [gen1, gen2], isolated_db)
        sqlite_store.mark_uploaded("test_account", {"gen_1"}, isolated_db)
        result = history_service.load_image_generations("test_account", include_uploaded=True)
        assert len(result) == 2

    def test_load_with_keep_days_filters_old(self, isolated_db):
        """Old generations beyond keep_days should be excluded."""
        old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        new_time = datetime.now(timezone.utc).isoformat()

        old_gen = ChatGPTImageGeneration(
            created_at=old_time,
            id="gen_old",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/old.png",
            prompt="Old",
        )
        new_gen = ChatGPTImageGeneration(
            created_at=new_time,
            id="gen_new",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/new.png",
            prompt="New",
        )
        sqlite_store.upsert_generations("test_account", [old_gen, new_gen], isolated_db)
        result = history_service.load_image_generations("test_account", keep_days=2)
        assert len(result) == 1
        assert result[0].id == "gen_new"

    def test_load_with_timezone_conversion(self, isolated_db):
        """timezone_name should convert timestamps - just verify it runs without error."""
        from zoneinfo import ZoneInfo
        gen = ChatGPTImageGeneration(
            created_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            id="gen_1",
            conversation_id="conv_1",
            message_id="msg_1",
            asset_pointer="asset_1",
            url="https://example.com/img1.png",
            prompt="Test",
        )
        sqlite_store.upsert_generations("test_account", [gen], isolated_db)
        result = history_service.load_image_generations("test_account", timezone_name="UTC")
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
        gen2 = gen1.model_copy(update={"id": "gen_2"})
        sqlite_store.upsert_generations("test_account", [gen1, gen2], isolated_db)
        sqlite_store.mark_uploaded("test_account", {"gen_1"}, isolated_db)
        result_true = history_service.load_image_generations("test_account", include_uploaded=True)
        result_false = history_service.load_image_generations("test_account", include_uploaded=False)
        assert len(result_true) == 2
        assert len(result_false) == 1


class TestFetchGenerationsForConversation:
    """Tests for fetch_generations_for_conversation function."""
    @pytest.mark.asyncio
    async def test_empty_items_returns_empty_list(self):
        """Empty items list should return empty list."""
        mock_session = AsyncMock()
        result = await history_service.fetch_generations_for_conversation(
            mock_session,
            "conv_1",
            [],
            {"Authorization": "Bearer token"},
            set(),
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
        sqlite_store.upsert_generations("test_account", [existing_gen], isolated_db)

        mock_session = AsyncMock()

        with patch("chatgpt_to_notion.services.history_service.chatgpt_api.get_conversation_details") as mock_get_detail, \
             patch("chatgpt_to_notion.services.history_service.get_prompt_from_image_node_in_conversation") as mock_prompt, \
             patch("chatgpt_to_notion.services.history_service.resolve_config", return_value=_mock_resolved_config()):

            mock_get_detail.return_value = {"mapping": {}}
            mock_prompt.return_value = "Test Prompt"

            result = await history_service.fetch_generations_for_conversation(
                mock_session,
                "conv_1",
                [
                    {"id": "existing_gen", "conversation_id": "conv_1", "message_id": "msg_1", "asset_pointer": "asset_1", "url": "https://example.com/existing.png", "created_at": 1704067200},
                ],
                {"Authorization": "Bearer test"},
                {"existing_gen"},
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
                mock_session,
                "conv_1",
                mock_items,
                {"Authorization": "Bearer test"},
                set(),
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
                mock_session,
                "conv_1",
                mock_items,
                {"Authorization": "Bearer test"},
                set(),
            )
            assert len(result) == 3
            assert {g.id for g in result} == {"gen_1", "gen_2", "gen_3"}
            assert mock_get_detail.call_count == 1


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
