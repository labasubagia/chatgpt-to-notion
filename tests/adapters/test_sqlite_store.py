"""Tests for sqlite_store.py - SQLite persistence layer."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chatgpt_to_notion.adapters.sqlite_store import SqliteStore
from chatgpt_to_notion.domain.models import ChatGPTImageGeneration

@pytest.fixture
def tmp_db_path(tmp_path):
    return tmp_path / "test.db"

@pytest.fixture
def store(tmp_db_path):
    return SqliteStore(db_path=tmp_db_path)

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
    @pytest.mark.asyncio
    async def test_creates_tables(self, store, sample_generation):
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])
        assert len(store.get_generations("test_account")) == 1
        await store.async_set_cached_data_sources("db1", [{"id": "ds1"}])
        assert store.get_cached_data_sources("db1") == [{"id": "ds1"}]

class TestUpsertGenerations:
    def test_inserts_new(self, store, sample_generation):
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])
        result = store.get_generations("test_account")
        assert len(result) == 1
        assert result[0].id == "gen_test123"

    def test_updates_existing(self, store, sample_generation):
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])
        updated = sample_generation.model_copy(update={"prompt": "updated"})
        store.upsert_generations("test_account", [updated])
        result = store.get_generations("test_account")
        assert len(result) == 1
        assert result[0].prompt == "updated"

    def test_multiple_generations(self, store, sample_generations):
        store.init_db()
        store.upsert_generations("test_account", sample_generations)
        result = store.get_generations("test_account")
        assert len(result) == len(sample_generations)

class TestGetGenerations:
    def test_filters_unuploaded_by_default(self, store, sample_generation):
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])
        store.mark_uploaded("test_account", {"gen_test123"})
        result = store.get_generations("test_account")
        assert len(result) == 0

    def test_includes_uploaded_when_flagged(self, store, sample_generation):
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])
        store.mark_uploaded("test_account", {"gen_test123"})
        result = store.get_generations("test_account", include_uploaded=True)
        assert len(result) == 1

    def test_filters_by_keep_days(self, store, sample_generations):
        store.init_db()
        store.upsert_generations("test_account", sample_generations)

        result = store.get_generations("test_account", keep_days=3)
        assert len(result) == 2

    def test_filters_by_ids_filter(self, store, sample_generations):
        store.init_db()
        store.upsert_generations("test_account", sample_generations)

        ids = {sample_generations[0].id, sample_generations[2].id}
        result = store.get_generations("test_account", ids_filter=ids)
        assert len(result) == 2
        result_ids = {g.id for g in result}
        assert result_ids == ids

    def test_ids_filter_empty_account_returns_empty(self, store):
        store.init_db()
        result = store.get_generations("test_account", ids_filter={"nonexistent"})
        assert result == []

    def test_empty_account_returns_empty_list(self, store):
        store.init_db()
        result = store.get_generations("nonexistent")
        assert result == []

class TestGetUploadedIds:
    def test_returns_uploaded_ids(self, store, sample_generation):
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])
        store.mark_uploaded("test_account", {sample_generation.id})

        result = store.get_uploaded_ids("test_account")
        assert result == {sample_generation.id}

    def test_returns_empty_for_no_uploads(self, store, sample_generation):
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])

        result = store.get_uploaded_ids("test_account")
        assert result == set()

class TestMarkUploaded:
    def test_sets_uploaded_at(self, store, tmp_db_path, sample_generation):
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])
        store.mark_uploaded("test_account", {sample_generation.id})
        # Persisted on disk: a fresh store on the same file sees the upload.
        fresh = SqliteStore(db_path=tmp_db_path)
        assert fresh.get_uploaded_ids("test_account") == {sample_generation.id}

    def test_empty_ids_does_nothing(self, store, sample_generation):
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])
        store.mark_uploaded("test_account", set())

        result = store.get_uploaded_ids("test_account")
        assert result == set()

class TestGetExistingIds:
    def test_returns_matching_ids(self, store, sample_generation):
        store.init_db()
        gen2 = sample_generation.model_copy(update={"id": "gen_2"})
        store.upsert_generations("test_account", [sample_generation, gen2])
        result = store.get_existing_ids("test_account", {"gen_test123", "gen_2", "gen_3"})
        assert result == {"gen_test123", "gen_2"}

    def test_returns_empty_when_none_match(self, store, sample_generation):
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])
        result = store.get_existing_ids("test_account", {"gen_99"})
        assert result == set()

    def test_returns_empty_for_empty_input(self, store):
        store.init_db()
        result = store.get_existing_ids("test_account", set())
        assert result == set()

    def test_respects_account_boundary(self, store, sample_generation):
        store.init_db()
        store.upsert_generations("account_a", [sample_generation])
        result = store.get_existing_ids("account_b", {"gen_test123"})
        assert result == set()

class TestDataSourcesCache:
    @pytest.mark.asyncio
    async def test_set_and_get(self, store):
        store.init_db()
        sources = [{"id": "ds_1"}, {"id": "ds_2"}]
        await store.async_set_cached_data_sources("test_db", sources)

        result = store.get_cached_data_sources("test_db")
        assert result == sources

    def test_returns_none_when_missing(self, store):
        store.init_db()
        result = store.get_cached_data_sources("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_expires_after_ttl(self, store):
        store.init_db()
        sources = [{"id": "ds_1"}]
        await store.async_set_cached_data_sources("test_db", sources, ttl_days=0)

        import time
        time.sleep(0.01)

        result = store.get_cached_data_sources("test_db")
        assert result is None

class TestHasGenerations:
    def test_returns_true_when_exists(self, store, sample_generation):
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])
        assert store.has_generations("test_account") is True

    def test_returns_false_when_empty(self, store):
        store.init_db()
        assert store.has_generations("empty_account") is False

    def test_returns_false_for_other_account(self, store, sample_generation):
        store.init_db()
        store.upsert_generations("account_a", [sample_generation])
        assert store.has_generations("account_b") is False

class TestGetActivityStats:
    def test_returns_stats_for_date(self, store):
        store.init_db()
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
        store.upsert_generations("acc", gens)

        stats = store.get_activity_stats(
            "acc",
            date_start=today_start.isoformat(),
            date_end=today_end.isoformat(),
            cooldown_threshold=cooldown.isoformat(),
        )
        assert stats is not None
        assert stats["total"] == 4
        assert stats["active_count"] == 4
    def test_returns_none_for_no_data(self, store):
        store.init_db()
        now = datetime.now(timezone.utc)
        stats = store.get_activity_stats(
            "acc",
            date_start=now.isoformat(),
            date_end=(now + timedelta(days=1)).isoformat(),
            cooldown_threshold=(now - timedelta(days=1)).isoformat(),
        )
        assert stats is None

    def test_active_count_filters_by_cooldown(self, store):
        store.init_db()

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
        store.upsert_generations("acc", gens)

        # Use a wide date range that includes both entries
        date_start = (now - timedelta(days=2)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        date_end = now + timedelta(days=1)
        cooldown = now - timedelta(days=1)

        stats = store.get_activity_stats(
            "acc",
            date_start=date_start.isoformat(),
            date_end=date_end.isoformat(),
            cooldown_threshold=cooldown.isoformat(),
        )
        assert stats["total"] == 2
        assert stats["active_count"] == 1

class TestBackupDb:
    def test_backup_creates_copy(self, store, tmp_db_path, sample_generation):
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])
        backup = tmp_db_path.with_name("backup.db")
        store.backup(backup)
        assert backup.exists()
        backup_store = SqliteStore(db_path=backup)
        ids = [g.id for g in backup_store.get_generations("test_account", include_uploaded=True)]
        assert ids == [sample_generation.id]
        store.init_db()
        store.upsert_generations("test_account", [sample_generation])
        backup = Path("backup.db")
        store.backup(backup)
        assert backup.exists()
        backup_store = SqliteStore(db_path=backup)
        ids = [g.id for g in backup_store.get_generations("test_account", include_uploaded=True)]
        assert ids == [sample_generation.id]

    def test_backup_creates_parent_dirs(self, tmp_path):
        db = tmp_path / "db" / "test.db"
        store = SqliteStore(db_path=db)
        store.init_db()
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        store.upsert_generations("test_account", [gen])
        backup = tmp_path / "backup" / "backup.db"
        store.backup(backup)
        assert backup.exists()

    def test_backup_with_absolute_path(self, tmp_path):
        db = tmp_path / "test.db"
        store = SqliteStore(db_path=db)
        store.init_db()
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        store.upsert_generations("test_account", [gen])
        backup = tmp_path / "backup.db"
        store.backup(backup.resolve())
        assert backup.exists()

class TestRestoreDb:
    def test_restore_replaces_database(self, tmp_path):
        src = tmp_path / "src.db"
        dst = tmp_path / "dst.db"
        src_store = SqliteStore(db_path=src)
        src_store.init_db()
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        src_store.upsert_generations("test_account", [gen])
        dst_store = SqliteStore(db_path=dst)
        dst_store.init_db()
        dst_store.restore(src)
        result = dst_store.get_generations("test_account")
        assert len(result) == 1
        assert result[0].id == "g1"

    def test_restore_overwrites_existing_database(self, tmp_path):
        src = tmp_path / "src.db"
        dst = tmp_path / "dst.db"
        src_store = SqliteStore(db_path=src)
        src_store.init_db()
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        src_store.upsert_generations("test_account", [gen])
        dst_store = SqliteStore(db_path=dst)
        dst_store.init_db()
        other = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="other",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="other",
        )
        dst_store.upsert_generations("test_account", [other])
        dst_store.restore(src)
        result = dst_store.get_generations("test_account")
        assert len(result) == 1
        assert result[0].id == "g1"

    def test_restore_missing_backup_raises(self, tmp_path):
        dst = tmp_path / "dst.db"
        SqliteStore(db_path=dst).init_db()
        with pytest.raises(FileNotFoundError):
            SqliteStore(db_path=dst).restore(tmp_path / "missing.db")

    def test_restore_creates_parent_dirs(self, tmp_path):
        src = tmp_path / "src.db"
        dst = tmp_path / "nested" / "dst.db"
        src_store = SqliteStore(db_path=src)
        src_store.init_db()
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        src_store.upsert_generations("test_account", [gen])
        dst_store = SqliteStore(db_path=dst)
        dst_store.restore(src)
        assert dst.exists()
        result = dst_store.get_generations("test_account")
        assert len(result) == 1
