"""Tests for sqlite_store.py - SQLite persistence layer."""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chatgpt_to_notion.adapters import sqlite_store
from chatgpt_to_notion.domain.models import ChatGPTImageGeneration


@pytest.fixture
def tmp_db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def sample_generation():
    return ChatGPTImageGeneration(
        created_at=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
        id="gen_test123",
        conversation_id="conv_abc",
        message_id="msg_def",
        asset_pointer="asset_ghi",
        url="https://example.com/image.png",
        prompt="A beautiful sunset",
    )


@pytest.fixture
def sample_generations(sample_generation):
    older = sample_generation.model_copy(
        update={
            "id": "gen_older",
            "created_at": (
                datetime.now(timezone.utc) - timedelta(days=5)
            ).isoformat(),
        }
    )
    newer = sample_generation.model_copy(
        update={
            "id": "gen_newer",
            "created_at": (
                datetime.now(timezone.utc) - timedelta(hours=1)
            ).isoformat(),
        }
    )
    return [older, sample_generation, newer]


class TestInitDb:
    def test_creates_tables(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        conn = sqlite3.connect(str(tmp_db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {t[0] for t in tables}
        assert "image_generations" in names
        assert "cache_data_sources" in names


class TestUpsertGenerations:
    def test_inserts_new(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)
        result = sqlite_store.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 1
        assert result[0].id == "gen_test123"

    def test_updates_existing(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)
        updated = sample_generation.model_copy(update={"prompt": "updated"})
        sqlite_store.upsert_generations("test_account", [updated], tmp_db_path)
        result = sqlite_store.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 1
        assert result[0].prompt == "updated"

    def test_multiple_generations(self, tmp_db_path, sample_generations):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", sample_generations, tmp_db_path)
        result = sqlite_store.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == len(sample_generations)


class TestGetGenerations:
    def test_filters_unuploaded_by_default(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)
        sqlite_store.mark_uploaded("test_account", {"gen_test123"}, tmp_db_path)
        result = sqlite_store.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 0

    def test_includes_uploaded_when_flagged(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)
        sqlite_store.mark_uploaded("test_account", {"gen_test123"}, tmp_db_path)
        result = sqlite_store.get_generations("test_account", include_uploaded=True, db_path=tmp_db_path)
        assert len(result) == 1

    def test_filters_by_keep_days(self, tmp_db_path, sample_generations):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", sample_generations, tmp_db_path)

        result = sqlite_store.get_generations(
            "test_account", keep_days=3, db_path=tmp_db_path
        )
        assert len(result) == 2

    def test_filters_by_ids_filter(self, tmp_db_path, sample_generations):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", sample_generations, tmp_db_path)

        ids = {sample_generations[0].id, sample_generations[2].id}
        result = sqlite_store.get_generations(
            "test_account", ids_filter=ids, db_path=tmp_db_path
        )
        assert len(result) == 2
        result_ids = {g.id for g in result}
        assert result_ids == ids

    def test_ids_filter_empty_account_returns_empty(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        result = sqlite_store.get_generations(
            "test_account", ids_filter={"nonexistent"}, db_path=tmp_db_path
        )
        assert result == []

    def test_empty_account_returns_empty_list(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        result = sqlite_store.get_generations("nonexistent", db_path=tmp_db_path)
        assert result == []


class TestGetUploadedIds:
    def test_returns_uploaded_ids(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)
        sqlite_store.mark_uploaded("test_account", {sample_generation.id}, tmp_db_path)

        result = sqlite_store.get_uploaded_ids("test_account", db_path=tmp_db_path)
        assert result == {sample_generation.id}

    def test_returns_empty_for_no_uploads(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)

        result = sqlite_store.get_uploaded_ids("test_account", db_path=tmp_db_path)
        assert result == set()


class TestMarkUploaded:
    def test_sets_uploaded_at(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)
        sqlite_store.mark_uploaded("test_account", {sample_generation.id}, tmp_db_path)

        conn = sqlite_store._get_connection(tmp_db_path)
        row = conn.execute(
            "SELECT uploaded_at FROM image_generations WHERE id = ?",
            (sample_generation.id,),
        ).fetchone()
        conn.close()
        assert row["uploaded_at"] != ""

    def test_empty_ids_does_nothing(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)
        sqlite_store.mark_uploaded("test_account", set(), tmp_db_path)

        result = sqlite_store.get_uploaded_ids("test_account", db_path=tmp_db_path)
        assert result == set()


class TestGetExistingIds:
    def test_returns_matching_ids(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        gen2 = sample_generation.model_copy(update={"id": "gen_2"})
        sqlite_store.upsert_generations("test_account", [sample_generation, gen2], tmp_db_path)
        result = sqlite_store.get_existing_ids("test_account", {"gen_test123", "gen_2", "gen_3"}, tmp_db_path)
        assert result == {"gen_test123", "gen_2"}

    def test_returns_empty_when_none_match(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)
        result = sqlite_store.get_existing_ids("test_account", {"gen_99"}, tmp_db_path)
        assert result == set()

    def test_returns_empty_for_empty_input(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        result = sqlite_store.get_existing_ids("test_account", set(), tmp_db_path)
        assert result == set()

    def test_respects_account_boundary(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("account_a", [sample_generation], tmp_db_path)
        result = sqlite_store.get_existing_ids("account_b", {"gen_test123"}, tmp_db_path)
        assert result == set()




class TestDataSourcesCache:
    @pytest.mark.asyncio
    async def test_set_and_get(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        sources = [{"id": "ds_1"}, {"id": "ds_2"}]
        await sqlite_store.async_set_cached_data_sources("test_db", sources, db_path=tmp_db_path)

        result = sqlite_store.get_cached_data_sources("test_db", db_path=tmp_db_path)
        assert result == sources

    def test_returns_none_when_missing(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        result = sqlite_store.get_cached_data_sources("nonexistent", db_path=tmp_db_path)
        assert result is None

    @pytest.mark.asyncio
    async def test_expires_after_ttl(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        sources = [{"id": "ds_1"}]
        await sqlite_store.async_set_cached_data_sources("test_db", sources, ttl_days=0, db_path=tmp_db_path)

        import time
        time.sleep(0.01)

        result = sqlite_store.get_cached_data_sources("test_db", db_path=tmp_db_path)
        assert result is None


class TestHasGenerations:
    def test_returns_true_when_exists(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)
        assert sqlite_store.has_generations("test_account", tmp_db_path) is True

    def test_returns_false_when_empty(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        assert sqlite_store.has_generations("empty_account", tmp_db_path) is False

    def test_returns_false_for_other_account(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("account_a", [sample_generation], tmp_db_path)
        assert sqlite_store.has_generations("account_b", tmp_db_path) is False


class TestGetActivityStats:
    def test_returns_stats_for_date(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        cooldown = today_start - timedelta(hours=1)

        gens = [
            ChatGPTImageGeneration(
                created_at=(today_start + timedelta(hours=h)).isoformat(),
                id=f"g{h}",
                conversation_id="c1",
                message_id=f"m{h}",
                asset_pointer=f"a{h}",
                url=f"https://example.com/{h}.png",
                prompt="test",
            )
            for h in range(1, 5)
        ]
        sqlite_store.upsert_generations("acc", gens, db_path=tmp_db_path)

        stats = sqlite_store.get_activity_stats(
            "acc",
            date_start=today_start.isoformat(),
            date_end=today_end.isoformat(),
            cooldown_threshold=cooldown.isoformat(),
            db_path=tmp_db_path,
        )
        assert stats is not None
        assert stats["total"] == 4
        assert stats["active_count"] == 4

    def test_returns_none_for_no_data(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        now = datetime.now(timezone.utc)
        stats = sqlite_store.get_activity_stats(
            "acc",
            date_start=now.isoformat(),
            date_end=(now + timedelta(days=1)).isoformat(),
            cooldown_threshold=(now - timedelta(days=1)).isoformat(),
            db_path=tmp_db_path,
        )
        assert stats is None

    def test_active_count_filters_by_cooldown(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        gens = [
            ChatGPTImageGeneration(
                created_at=(now - timedelta(hours=25)).isoformat(),
                id="g_old",
                conversation_id="c1",
                message_id="m1",
                asset_pointer="a1",
                url="https://example.com/old.png",
                prompt="old",
            ),
            ChatGPTImageGeneration(
                created_at=(now - timedelta(minutes=30)).isoformat(),
                id="g_new",
                conversation_id="c1",
                message_id="m2",
                asset_pointer="a2",
                url="https://example.com/new.png",
                prompt="new",
            ),
        ]
        sqlite_store.upsert_generations("acc", gens, db_path=tmp_db_path)

        # Use a wide date range that includes both entries
        date_start = (now - timedelta(days=2)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        date_end = now + timedelta(days=1)
        cooldown = now - timedelta(days=1)

        stats = sqlite_store.get_activity_stats(
            "acc",
            date_start=date_start.isoformat(),
            date_end=date_end.isoformat(),
            cooldown_threshold=cooldown.isoformat(),
            db_path=tmp_db_path,
        )
        assert stats is not None
        assert stats["total"] == 2
        assert stats["active_count"] == 1




class TestBackupDb:
    def test_backup_creates_copy(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)
        backup = tmp_db_path.with_name("backup.db")
        sqlite_store.backup_db(backup, tmp_db_path)
        assert backup.exists()
        conn = sqlite3.connect(str(backup))
        row = conn.execute("SELECT id FROM image_generations").fetchone()
        assert row[0] == "gen_test123"

    def test_backup_creates_parent_dirs(self, tmp_path):
        db = tmp_path / "db" / "test.db"
        sqlite_store.init_db(db)
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        sqlite_store.upsert_generations("test_account", [gen], db)
        backup = tmp_path / "backup" / "backup.db"
        sqlite_store.backup_db(backup, db)
        assert backup.exists()

    def test_backup_with_absolute_path(self, tmp_path):
        db = tmp_path / "test.db"
        sqlite_store.init_db(db)
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        sqlite_store.upsert_generations("test_account", [gen], db)
        backup = tmp_path / "backup.db"
        sqlite_store.backup_db(backup.resolve(), db)
        assert backup.exists()


class TestRestoreDb:
    def test_restore_replaces_database(self, tmp_path):
        src = tmp_path / "src.db"
        dst = tmp_path / "dst.db"
        sqlite_store.init_db(src)
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        sqlite_store.upsert_generations("test_account", [gen], src)
        sqlite_store.init_db(dst)
        sqlite_store.restore_db(src, dst)
        result = sqlite_store.get_generations("test_account", db_path=dst)
        assert len(result) == 1
        assert result[0].id == "g1"

    def test_restore_overwrites_existing_database(self, tmp_path):
        src = tmp_path / "src.db"
        dst = tmp_path / "dst.db"
        sqlite_store.init_db(src)
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        sqlite_store.upsert_generations("test_account", [gen], src)
        sqlite_store.init_db(dst)
        other = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="other",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="other",
        )
        sqlite_store.upsert_generations("test_account", [other], dst)
        sqlite_store.restore_db(src, dst)
        result = sqlite_store.get_generations("test_account", db_path=dst)
        assert len(result) == 1
        assert result[0].id == "g1"

    def test_restore_missing_backup_raises(self, tmp_path):
        dst = tmp_path / "dst.db"
        sqlite_store.init_db(dst)
        with pytest.raises(FileNotFoundError):
            sqlite_store.restore_db(tmp_path / "missing.db", dst)

    def test_restore_creates_parent_dirs(self, tmp_path):
        src = tmp_path / "src.db"
        dst = tmp_path / "nested" / "dst.db"
        sqlite_store.init_db(src)
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        sqlite_store.upsert_generations("test_account", [gen], src)
        sqlite_store.restore_db(src, dst)
        assert dst.exists()
        result = sqlite_store.get_generations("test_account", db_path=dst)
        assert len(result) == 1
