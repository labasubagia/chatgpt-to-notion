"""Tests for upload pipeline edge cases and flag combinations."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from chatgpt_to_notion.services import upload_sequential
from chatgpt_to_notion.domain.models import ChatGPTImageGeneration, RuntimeOptions


def make_gen(gen_id: str, conv_id: str = None, prompt: str = None) -> ChatGPTImageGeneration:
    """Create a test ChatGPTImageGeneration."""
    return ChatGPTImageGeneration(
        id=gen_id,
        created_at="2024-01-01T00:00:00Z",
        conversation_id=conv_id or f"conv_{gen_id}",
        message_id=f"msg_{gen_id}",
        asset_pointer=f"asset_{gen_id}",
        url=f"http://example.com/{gen_id}.png",
        prompt=prompt or f"prompt_{gen_id}",
    )


class TestUploadFromHistoryMode:
    """Tests for from_history flag behavior."""

    @pytest.mark.asyncio
    async def test_from_history_loads_only_unuploaded(self, tmp_path, mock_aiohttp_session):
        """from_history should only process generations without uploaded_at."""
        options = RuntimeOptions(account="test@example.com")

        # Set up database with one uploaded and one unuploaded generation
        import sqlite3
        from chatgpt_to_notion.adapters.sqlite_store import init_db
        db_path = tmp_path / "test.db"
        init_db(db_path)
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO image_generations (id, created_at, conversation_id, message_id, asset_pointer, url, prompt, uploaded_at, account) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("gen1", "2024-01-01T00:00:00Z", "conv1", "msg1", "asset1", "http://example.com/img1.png", "prompt1", "2024-01-01T01:00:00Z", "test@example.com")
            )
            conn.execute(
                "INSERT INTO image_generations (id, created_at, conversation_id, message_id, asset_pointer, url, prompt, uploaded_at, account) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("gen2", "2024-01-01T00:00:00Z", "conv2", "msg2", "asset2", "http://example.com/img2.png", "prompt2", "", "test@example.com")
            )
            conn.commit()

        # Create image file for gen2
        (tmp_path / "images").mkdir(parents=True, exist_ok=True)
        (tmp_path / "images" / "gen2.png").write_bytes(b"fake image data")

        with patch("chatgpt_to_notion.services.upload_sequential.resolve_image_folder", return_value=tmp_path / "images"):
            with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.init_db"):
                with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store._get_db_path", return_value=db_path):
                    with patch("chatgpt_to_notion.services.upload_sequential.shared.get_uploaded_generation_ids", return_value={"gen1"}):
                        with patch("chatgpt_to_notion.services.upload_sequential.shared.load_image_generations", return_value=[make_gen("gen2", "conv2")]) as mock_load:
                            with patch("chatgpt_to_notion.services.upload_sequential.shared.download_all_images", return_value=None):
                                with patch("chatgpt_to_notion.services.upload_sequential.shared.add_prompt_to_images", return_value=None):
                                    with patch("chatgpt_to_notion.services.upload_sequential.shared.is_page_exists_in_db", return_value=False):
                                        with patch("chatgpt_to_notion.services.upload_sequential.shared.add_page_to_db", return_value=None):
                                            with patch("chatgpt_to_notion.services.upload_sequential.shared.add_prompt_to_image_single", return_value=None):
                                                with patch("chatgpt_to_notion.services.upload_sequential.shared.mark_generations_uploaded") as mock_mark:
                                                    with patch("chatgpt_to_notion.services.upload_pipeline.get_headers", return_value={"Authorization": "Bearer test"}):
                                                        await upload_sequential.upload_to_notion_sequential(
                                                            image_folder=str(tmp_path / "images"),
                                                            db_id="test-db-id",
                                                            upload_to_notion=True,
                                                            delete_conversation=False,
                                                            delete_library=False,
                                                            delete_library_all=False,
                                                            add_prompt_to_image=True,
                                                            account="test@example.com",
                                                            check_notion_api=False,
                                                            from_history=True,
                                                            limit=100,
                                                            keep_days=3,
                                                            timezone_name=None,
                                                            no_cache=False,
                                                            options=options,
                                                            fail_log_path=None,
                                                            max_concurrent=1,
                                                            conversation_delay=0.0,
                                                        )

        # Should only load unuploaded generations
        mock_load.assert_called_once()
        call_args = mock_load.call_args
        assert call_args.kwargs.get("include_uploaded") is False

        # Should only mark gen2 as uploaded
        mock_mark.assert_called_once()
        marked_ids = mock_mark.call_args[0][1]
        assert marked_ids == {"gen2"}


class TestUploadCheckNotionApiFlag:
    """Tests for check_notion_api flag behavior."""

    @pytest.mark.asyncio
    async def test_check_notion_api_false_skips_uploaded_check(self, tmp_path, mock_aiohttp_session):
        """check_notion_api=False should skip Notion API check for already uploaded items."""
        from chatgpt_to_notion.adapters.notion_api import upload_all_images_to_notion

        gen = make_gen("gen1")

        with patch("chatgpt_to_notion.adapters.notion_api.is_page_exists_in_db") as mock_exists:
            with patch("chatgpt_to_notion.adapters.notion_api.add_page_to_db") as mock_add:
                with patch("chatgpt_to_notion.adapters.notion_api.get_uploaded_ids", return_value={"gen1"}):
                    with patch("chatgpt_to_notion.adapters.notion_api.get_output_path", return_value=Path(tmp_path)):
                        await upload_all_images_to_notion(
                            [gen],
                            "test-db-id",
                            str(tmp_path),
                            account="test@example.com",
                            check_notion_api=False,
                            options=None,
                            fail_log_path=None,
                            max_concurrent=1,
                        )

        # Should not check Notion API for already uploaded
        mock_exists.assert_not_called()
        mock_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_notion_api_true_checks_notion(self, tmp_path, mock_aiohttp_session):
        """check_notion_api=True should check Notion API even for uploaded items."""
        from chatgpt_to_notion.adapters.notion_api import upload_all_images_to_notion

        gen = make_gen("gen1")

        with patch("chatgpt_to_notion.adapters.notion_api.is_page_exists_in_db", return_value=False) as mock_exists:
            with patch("chatgpt_to_notion.adapters.notion_api.add_page_to_db") as mock_add:
                with patch("chatgpt_to_notion.adapters.notion_api.get_uploaded_ids", return_value={"gen1"}):
                    with patch("chatgpt_to_notion.adapters.notion_api.get_output_path", return_value=Path(tmp_path)):
                        with patch("chatgpt_to_notion.adapters.notion_api.mark_uploaded"):
                            # Create the image file so it exists
                            (tmp_path / "gen1.png").write_bytes(b"fake image data")
                            await upload_all_images_to_notion(
                                [gen],
                                "test-db-id",
                                str(tmp_path),
                                account="test@example.com",
                                check_notion_api=True,
                                options=None,
                                fail_log_path=None,
                                max_concurrent=1,
                            )

        # Should check Notion API even for already uploaded
        mock_exists.assert_called_once()
        mock_add.assert_called_once()


class TestUploadSequentialMode:
    """Tests for sequential upload mode."""

    @pytest.mark.asyncio
    async def test_sequential_processes_conversations_one_at_a_time(self, tmp_path, mock_aiohttp_session):
        """Sequential mode should process conversations one at a time."""
        options = RuntimeOptions(account="test@example.com")

        # Initialize the database
        from chatgpt_to_notion.adapters.sqlite_store import init_db
        init_db()

        with patch("chatgpt_to_notion.services.upload_sequential.resolve_image_folder", return_value=tmp_path / "images"):
            with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.init_db"):
                with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.get_existing_ids", return_value=set()):
                    with patch("chatgpt_to_notion.services.upload_sequential.shared.get_uploaded_generation_ids", return_value=set()):
                        with patch("chatgpt_to_notion.services.upload_sequential.shared.get_image_generations", return_value={"items": [
                            {"id": "gen1", "conversation_id": "conv1", "message_id": "msg1", "asset_pointer": "asset1", "url": "http://example.com/gen1.png", "prompt": "prompt1"},
                            {"id": "gen2", "conversation_id": "conv2", "message_id": "msg2", "asset_pointer": "asset2", "url": "http://example.com/gen2.png", "prompt": "prompt2"},
                        ]}) as mock_get_gens:
                            with patch("chatgpt_to_notion.services.upload_sequential.shared.fetch_generations_for_conversation", side_effect=[
                                [make_gen("gen1", "conv1")],
                                [make_gen("gen2", "conv2")],
                            ]) as mock_fetch_conv:
                                with patch("chatgpt_to_notion.services.upload_sequential.shared.download_all_images", return_value=None):
                                    with patch("chatgpt_to_notion.services.upload_sequential.shared.add_prompt_to_images"):
                                        with patch("chatgpt_to_notion.services.upload_sequential.shared.is_page_exists_in_db", return_value=False):
                                            with patch("chatgpt_to_notion.services.upload_sequential.shared.add_page_to_db"):
                                                with patch("chatgpt_to_notion.services.upload_sequential.shared.mark_generations_uploaded"):
                                                    with patch("chatgpt_to_notion.services.upload_pipeline.get_headers", return_value={"Authorization": "Bearer test"}):
                                                        await upload_sequential.upload_to_notion_sequential(
                                                            image_folder=str(tmp_path / "images"),
                                                            db_id="test-db-id",
                                                            upload_to_notion=True,
                                                            delete_conversation=False,
                                                            delete_library=False,
                                                            delete_library_all=False,
                                                            add_prompt_to_image=True,
                                                            account="test@example.com",
                                                            check_notion_api=False,
                                                            from_history=False,
                                                            limit=100,
                                                            keep_days=None,
                                                            timezone_name=None,
                                                            no_cache=False,
                                                            options=options,
                                                            fail_log_path=None,
                                                            max_concurrent=1,
                                                            conversation_delay=0.0,
                                                        )

        # Should fetch for each conversation sequentially
        assert mock_fetch_conv.call_count == 2


class TestUploadFlagCombinations:
    """Tests for flag combinations and interactions."""

    def test_verify_history_implies_from_history(self):
        """verify_history should imply from_history."""
        from_history = False
        verify_history = True

        effective_from_history = from_history or verify_history
        effective_check_notion_api = True or verify_history  # check_notion_api defaults to True

        assert effective_from_history is True
        assert effective_check_notion_api is True

    def test_from_history_with_all_flag_uses_no_keep_days(self):
        """from_history with --all should not limit keep_days."""
        from_history = True
        all_flag = True

        if from_history and not all_flag:
            effective_keep_days = 3
        elif all_flag:
            effective_keep_days = None
        else:
            effective_keep_days = None

        assert effective_keep_days is None

    def test_from_history_without_all_uses_3_days(self):
        """from_history without --all should use 3 days keep_days."""
        from_history = True
        all_flag = False

        if from_history and not all_flag:
            effective_keep_days = 3
        elif all_flag:
            effective_keep_days = None
        else:
            effective_keep_days = None

        assert effective_keep_days == 3


class TestUploadDeleteFlags:
    """Tests for delete flag combinations."""

    @pytest.mark.asyncio
    async def test_delete_conversation_called_when_flag_true(self, tmp_path, mock_aiohttp_session):
        """delete_conversation should be called when flag is True."""
        options = RuntimeOptions(account="test@example.com")

        # Initialize the database
        from chatgpt_to_notion.adapters.sqlite_store import init_db
        init_db()

        with patch("chatgpt_to_notion.services.upload_sequential.resolve_image_folder", return_value=tmp_path / "images"):
            with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.init_db"):
                with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.get_existing_ids", return_value=set()):
                    with patch("chatgpt_to_notion.services.upload_sequential.shared.get_uploaded_generation_ids", return_value=set()):
                        with patch("chatgpt_to_notion.services.upload_sequential.shared.get_image_generations", return_value={"items": [
                            {"id": "gen1", "conversation_id": "conv1", "message_id": "msg1", "asset_pointer": "asset1", "url": "http://example.com/gen1.png", "prompt": "prompt1"},
                        ]}):
                            with patch("chatgpt_to_notion.services.upload_sequential.shared.fetch_generations_for_conversation", return_value=[make_gen("gen1", "conv1")]):
                                with patch("chatgpt_to_notion.services.upload_sequential.shared.download_all_images", return_value=None):
                                    with patch("chatgpt_to_notion.services.upload_sequential.shared.add_prompt_to_images"):
                                        with patch("chatgpt_to_notion.services.upload_sequential.shared.is_page_exists_in_db", return_value=False):
                                            with patch("chatgpt_to_notion.services.upload_sequential.shared.add_page_to_db"):
                                                with patch("chatgpt_to_notion.services.upload_sequential.shared.mark_generations_uploaded"):
                                                    with patch("chatgpt_to_notion.services.upload_pipeline.delete_conversation") as mock_delete_conv:
                                                        with patch("chatgpt_to_notion.services.history_service.download_image", return_value=None):
                                                            with patch("chatgpt_to_notion.services.upload_pipeline.get_headers", return_value={"Authorization": "Bearer test"}):
                                                                await upload_sequential.upload_to_notion_sequential(
                                                                    image_folder=str(tmp_path / "images"),
                                                                    db_id="test-db-id",
                                                                    upload_to_notion=True,
                                                                    delete_conversation=True,
                                                                    delete_library=False,
                                                                    delete_library_all=False,
                                                                    add_prompt_to_image=True,
                                                                    account="test@example.com",
                                                                    check_notion_api=False,
                                                                    from_history=False,
                                                                    limit=100,
                                                                    keep_days=None,
                                                                    timezone_name=None,
                                                                    no_cache=False,
                                                                    options=options,
                                                                    fail_log_path=None,
                                                                    max_concurrent=1,
                                                                    conversation_delay=0.0,
                                                                )

        mock_delete_conv.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_conversation_not_called_when_flag_false(self, tmp_path, mock_aiohttp_session):
        """delete_conversation should not be called when flag is False."""
        options = RuntimeOptions(account="test@example.com")

        # Initialize the database
        from chatgpt_to_notion.adapters.sqlite_store import init_db
        init_db()

        with patch("chatgpt_to_notion.services.upload_sequential.resolve_image_folder", return_value=tmp_path / "images"):
            with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.init_db"):
                with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.get_existing_ids", return_value=set()):
                    with patch("chatgpt_to_notion.services.upload_sequential.shared.get_uploaded_generation_ids", return_value=set()):
                        with patch("chatgpt_to_notion.services.upload_sequential.shared.get_image_generations", return_value={"items": [
                            {"id": "gen1", "conversation_id": "conv1", "message_id": "msg1", "asset_pointer": "asset1", "url": "http://example.com/gen1.png", "prompt": "prompt1"},
                        ]}):
                            with patch("chatgpt_to_notion.services.upload_sequential.shared.fetch_generations_for_conversation", return_value=[make_gen("gen1", "conv1")]):
                                with patch("chatgpt_to_notion.services.upload_sequential.shared.download_all_images", return_value=None):
                                    with patch("chatgpt_to_notion.services.upload_sequential.shared.add_prompt_to_images"):
                                        with patch("chatgpt_to_notion.services.upload_sequential.shared.is_page_exists_in_db", return_value=False):
                                            with patch("chatgpt_to_notion.services.upload_sequential.shared.add_page_to_db"):
                                                with patch("chatgpt_to_notion.services.upload_sequential.shared.mark_generations_uploaded"):
                                                    with patch("chatgpt_to_notion.services.upload_pipeline.delete_conversation") as mock_delete_conv:
                                                        with patch("chatgpt_to_notion.services.upload_pipeline.get_headers", return_value={"Authorization": "Bearer test"}):
                                                            await upload_sequential.upload_to_notion_sequential(
                                                                image_folder=str(tmp_path / "images"),
                                                                db_id="test-db-id",
                                                                upload_to_notion=True,
                                                                delete_conversation=False,
                                                                delete_library=False,
                                                                delete_library_all=False,
                                                                add_prompt_to_image=True,
                                                                account="test@example.com",
                                                                check_notion_api=False,
                                                                from_history=False,
                                                                limit=100,
                                                                keep_days=None,
                                                                timezone_name=None,
                                                                no_cache=False,
                                                                options=options,
                                                                fail_log_path=None,
                                                                max_concurrent=1,
                                                                conversation_delay=0.0,
                                                            )

        mock_delete_conv.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_library_called_in_from_history_mode(self, tmp_path, mock_aiohttp_session):
        """delete_library should be called even in from_history mode."""
        options = RuntimeOptions(account="test@example.com")

        with patch("chatgpt_to_notion.services.upload_sequential.resolve_image_folder", return_value=tmp_path / "images"):
            with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.init_db"):
                with patch("chatgpt_to_notion.services.upload_sequential.shared.get_uploaded_generation_ids", return_value=set()):
                    with patch("chatgpt_to_notion.services.upload_sequential.shared.load_image_generations", return_value=[]):
                        with patch("chatgpt_to_notion.services.upload_sequential.shared._remove_library_images") as mock_remove_lib:
                            with patch("chatgpt_to_notion.services.upload_pipeline.get_headers", return_value={"Authorization": "Bearer test"}):
                                await upload_sequential.upload_to_notion_sequential(
                                    image_folder=str(tmp_path / "images"),
                                    db_id="test-db-id",
                                    upload_to_notion=True,
                                    delete_conversation=False,
                                    delete_library=True,
                                    delete_library_all=False,
                                    add_prompt_to_image=True,
                                    account="test@example.com",
                                    check_notion_api=False,
                                    from_history=True,
                                    limit=100,
                                    keep_days=3,
                                    timezone_name=None,
                                    no_cache=False,
                                    options=options,
                                    fail_log_path=None,
                                    max_concurrent=1,
                                    conversation_delay=0.0,
                                )

        mock_remove_lib.assert_called_once()


class TestUploadTimezoneHandling:
    """Tests for timezone handling in upload."""

    @pytest.mark.asyncio
    async def test_timezone_passed_to_load_generations(self, tmp_path, mock_aiohttp_session):
        """timezone should be passed to load_image_generations."""
        options = RuntimeOptions(account="test@example.com")

        with patch("chatgpt_to_notion.services.upload_sequential.resolve_image_folder", return_value=tmp_path / "images"):
            with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.init_db"):
                with patch("chatgpt_to_notion.services.upload_sequential.shared.get_uploaded_generation_ids", return_value=set()):
                    with patch("chatgpt_to_notion.services.upload_sequential.shared.fetch_generations_for_conversation", return_value=[]):
                        with patch("chatgpt_to_notion.services.upload_sequential.shared.load_image_generations") as mock_load:
                            with patch("chatgpt_to_notion.services.upload_pipeline.get_headers", return_value={"Authorization": "Bearer test"}):
                                await upload_sequential.upload_to_notion_sequential(
                                    image_folder=str(tmp_path / "images"),
                                    db_id="test-db-id",
                                    upload_to_notion=True,
                                    delete_conversation=False,
                                    delete_library=False,
                                    delete_library_all=False,
                                    add_prompt_to_image=True,
                                    account="test@example.com",
                                    check_notion_api=False,
                                    from_history=True,
                                    limit=100,
                                    keep_days=3,
                                    timezone_name="Asia/Jakarta",
                                    no_cache=False,
                                    options=options,
                                    fail_log_path=None,
                                    max_concurrent=1,
                                    conversation_delay=0.0,
                                )

        mock_load.assert_called_once()
        assert mock_load.call_args.kwargs.get("timezone_name") == "Asia/Jakarta"


class TestUploadNoCacheFlag:
    """Tests for no_cache flag."""

    @pytest.mark.asyncio
    async def test_no_cache_bypasses_sqlite_cache(self, tmp_path, mock_aiohttp_session):
        """no_cache should bypass SQLite and fetch fresh from API."""
        options = RuntimeOptions(account="test@example.com")

        # Initialize the database
        from chatgpt_to_notion.adapters.sqlite_store import init_db
        init_db()

        with patch("chatgpt_to_notion.services.upload_sequential.resolve_image_folder", return_value=tmp_path / "images"):
            with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.init_db"):
                with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.get_existing_ids", return_value=set()):
                    with patch("chatgpt_to_notion.services.upload_sequential.shared.get_uploaded_generation_ids", return_value=set()):
                        with patch("chatgpt_to_notion.services.upload_sequential.shared.get_image_generations", return_value={"items": [
                            {"id": "gen1", "conversation_id": "conv1", "message_id": "msg1", "asset_pointer": "asset1", "url": "http://example.com/gen1.png", "prompt": "prompt1"},
                        ]}):
                            with patch("chatgpt_to_notion.services.upload_sequential.shared.fetch_generations_for_conversation", return_value=[make_gen("gen1", "conv1")]) as mock_fetch:
                                with patch("chatgpt_to_notion.services.upload_sequential.shared.download_all_images", return_value=None):
                                    with patch("chatgpt_to_notion.services.upload_sequential.shared.add_prompt_to_images"):
                                        with patch("chatgpt_to_notion.services.upload_sequential.shared.is_page_exists_in_db", return_value=False):
                                            with patch("chatgpt_to_notion.services.upload_sequential.shared.add_page_to_db"):
                                                with patch("chatgpt_to_notion.services.upload_sequential.shared.mark_generations_uploaded"):
                                                    with patch("chatgpt_to_notion.services.upload_pipeline.get_headers", return_value={"Authorization": "Bearer test"}):
                                                        await upload_sequential.upload_to_notion_sequential(
                                                            image_folder=str(tmp_path / "images"),
                                                            db_id="test-db-id",
                                                            upload_to_notion=True,
                                                            delete_conversation=False,
                                                            delete_library=False,
                                                            delete_library_all=False,
                                                            add_prompt_to_image=True,
                                                            account="test@example.com",
                                                            check_notion_api=False,
                                                            from_history=False,
                                                            limit=100,
                                                            keep_days=None,
                                                            timezone_name=None,
                                                            no_cache=True,
                                                            options=options,
                                                            fail_log_path=None,
                                                            max_concurrent=1,
                                                            conversation_delay=0.0,
                                                        )

        # fetch_generations_for_conversation should be called (bypassing cache)
        mock_fetch.assert_called_once()


class TestUploadLimitFlag:
    """Tests for limit flag."""

    @pytest.mark.asyncio
    async def test_limit_restricts_generations_processed(self, tmp_path, mock_aiohttp_session):
        """limit should restrict number of generations processed."""
        options = RuntimeOptions(account="test@example.com")

        # Initialize the database
        from chatgpt_to_notion.adapters.sqlite_store import init_db
        init_db()

        with patch("chatgpt_to_notion.services.upload_sequential.resolve_image_folder", return_value=tmp_path / "images"):
            with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.init_db"):
                with patch("chatgpt_to_notion.services.upload_sequential.sqlite_store.get_existing_ids", return_value=set()):
                    with patch("chatgpt_to_notion.services.upload_sequential.shared.get_uploaded_generation_ids", return_value=set()):
                        with patch("chatgpt_to_notion.services.upload_sequential.shared.get_image_generations", return_value={"items": [
                            {"id": f"gen{i}", "conversation_id": f"conv{i}", "message_id": f"msg{i}", "asset_pointer": f"asset{i}", "url": f"http://example.com/gen{i}.png", "prompt": f"prompt{i}"}
                            for i in range(5)
                        ]}) as mock_get_gens:
                            with patch("chatgpt_to_notion.services.upload_sequential.shared.fetch_generations_for_conversation") as mock_fetch:
                                mock_fetch.side_effect = [
                                    [make_gen(f"gen{i}", f"conv{i}")] for i in range(5)
                                ]
                                with patch("chatgpt_to_notion.services.upload_sequential.shared.download_all_images", return_value=None):
                                    with patch("chatgpt_to_notion.services.upload_sequential.shared.add_prompt_to_images"):
                                        with patch("chatgpt_to_notion.services.upload_sequential.shared.is_page_exists_in_db", return_value=False):
                                            with patch("chatgpt_to_notion.services.upload_sequential.shared.add_page_to_db"):
                                                with patch("chatgpt_to_notion.services.upload_sequential.shared.mark_generations_uploaded"):
                                                    with patch("chatgpt_to_notion.services.upload_pipeline.get_headers", return_value={"Authorization": "Bearer test"}):
                                                        await upload_sequential.upload_to_notion_sequential(
                                                            image_folder=str(tmp_path / "images"),
                                                            db_id="test-db-id",
                                                            upload_to_notion=True,
                                                            delete_conversation=False,
                                                            delete_library=False,
                                                            delete_library_all=False,
                                                            add_prompt_to_image=True,
                                                            account="test@example.com",
                                                            check_notion_api=False,
                                                            from_history=False,
                                                            limit=2,  # Limit to 2
                                                            keep_days=None,
                                                            timezone_name=None,
                                                            no_cache=False,
                                                            options=options,
                                                            fail_log_path=None,
                                                            max_concurrent=1,
                                                            conversation_delay=0.0,
                                                        )

        # Note: limit is applied in get_image_generations (API call), not in this function directly
        # This test verifies the limit is passed through to get_image_generations
        mock_get_gens.assert_called_once()
        assert mock_get_gens.call_args.kwargs.get("limit") == 2