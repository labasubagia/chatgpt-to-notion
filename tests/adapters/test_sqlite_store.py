"""Tests for sqlite_store.py - SQLite persistence layer."""

import sqlite3
import tempfile
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
    return [sample_generation, older, newer]


class TestInitDb:
    def test_creates_tables(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        conn = sqlite3.connect(str(tmp_db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = {t[0] for t in tables}
        assert "image_generations" in table_names
        assert "cache_data_sources" in table_names


class TestUpsertGenerations:
    def test_inserts_new_rows(self, tmp_db_path, sample_generations):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", sample_generations, tmp_db_path)

        result = sqlite_store.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 3

    def test_upserts_existing_rows(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)

        updated = sample_generation.model_copy(update={"prompt": "Updated prompt"})
        sqlite_store.upsert_generations("test_account", [updated], tmp_db_path)

        result = sqlite_store.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 1
        assert result[0].prompt == "Updated prompt"

    def test_empty_list_does_nothing(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [], tmp_db_path)
        result = sqlite_store.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 0


class TestGetGenerations:
    def test_returns_all_for_account(self, tmp_db_path, sample_generations):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", sample_generations, tmp_db_path)
        sqlite_store.upsert_generations(
            "other_account",
            [sample_generations[0].model_copy(update={"id": "other_id"})],
            tmp_db_path,
        )

        result = sqlite_store.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 3

    def test_filters_unuploaded_by_default(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)
        sqlite_store.mark_uploaded("test_account", {sample_generation.id}, tmp_db_path)

        result = sqlite_store.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 0

    def test_includes_uploaded_when_flagged(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", [sample_generation], tmp_db_path)
        sqlite_store.mark_uploaded("test_account", {sample_generation.id}, tmp_db_path)

        result = sqlite_store.get_generations(
            "test_account", include_uploaded=True, db_path=tmp_db_path
        )
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

    def test_empty_ids_does_nothing(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.mark_uploaded("test_account", set(), tmp_db_path)


class TestGetExistingIds:
    def test_returns_matching_ids(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        now = datetime.now(timezone.utc)
        gens = [
            ChatGPTImageGeneration(
                created_at=now.isoformat(),
                id=f"g{i}",
                conversation_id="c1",
                message_id=f"m{i}",
                asset_pointer=f"a{i}",
                url=f"https://example.com/{i}.png",
                prompt="test",
            )
            for i in range(1, 4)
        ]
        sqlite_store.upsert_generations("acc", gens, db_path=tmp_db_path)

        result = sqlite_store.get_existing_ids("acc", {"g1", "g3", "g99"}, db_path=tmp_db_path)
        assert result == {"g1", "g3"}

    def test_returns_empty_when_none_match(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        sqlite_store.upsert_generations("acc", [gen], db_path=tmp_db_path)

        result = sqlite_store.get_existing_ids("acc", {"g99"}, db_path=tmp_db_path)
        assert result == set()

    def test_returns_empty_for_empty_input(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        result = sqlite_store.get_existing_ids("acc", set(), db_path=tmp_db_path)
        assert result == set()

    def test_respects_account_boundary(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        sqlite_store.upsert_generations("acc_a", [gen], db_path=tmp_db_path)

        result = sqlite_store.get_existing_ids("acc_b", {"g1"}, db_path=tmp_db_path)
        assert result == set()


class TestDeleteOldGenerations:
    def test_deletes_rows_older_than_keep_days(self, tmp_db_path, sample_generations):
        sqlite_store.init_db(tmp_db_path)
        sqlite_store.upsert_generations("test_account", sample_generations, tmp_db_path)

        deleted = sqlite_store.delete_old_generations("test_account", keep_days=3, db_path=tmp_db_path)
        assert deleted == 1

        remaining = sqlite_store.get_generations(
            "test_account", include_uploaded=True, db_path=tmp_db_path
        )
        assert len(remaining) == 2

    def test_does_not_delete_other_accounts(self, tmp_db_path, sample_generation):
        sqlite_store.init_db(tmp_db_path)
        old = sample_generation.model_copy(
            update={
                "id": "old_gen",
                "created_at": (
                    datetime.now(timezone.utc) - timedelta(days=10)
                ).isoformat(),
            }
        )
        sqlite_store.upsert_generations("test_account", [old], tmp_db_path)
        sqlite_store.upsert_generations("other_account", [old.model_copy(update={"id": "other_old"})], tmp_db_path)

        sqlite_store.delete_old_generations("test_account", keep_days=2, db_path=tmp_db_path)

        other = sqlite_store.get_generations("other_account", include_uploaded=True, db_path=tmp_db_path)
        assert len(other) == 1


class TestDataSourcesCache:
    def test_set_and_get(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        sources = [{"id": "ds_1"}, {"id": "ds_2"}]
        sqlite_store.set_cached_data_sources("test_db", sources, db_path=tmp_db_path)

        result = sqlite_store.get_cached_data_sources("test_db", db_path=tmp_db_path)
        assert result == sources

    def test_returns_none_when_missing(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        result = sqlite_store.get_cached_data_sources("nonexistent", db_path=tmp_db_path)
        assert result is None

    def test_expires_after_ttl(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        sources = [{"id": "ds_1"}]
        sqlite_store.set_cached_data_sources("test_db", sources, ttl_days=0, db_path=tmp_db_path)

        import time
        time.sleep(0.01)

        result = sqlite_store.get_cached_data_sources("test_db", db_path=tmp_db_path)
        assert result is None


class TestHasGenerations:
    def test_returns_true_when_exists(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        sqlite_store.upsert_generations("acc", [gen], db_path=tmp_db_path)
        assert sqlite_store.has_generations("acc", db_path=tmp_db_path) is True

    def test_returns_false_when_empty(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        assert sqlite_store.has_generations("acc", db_path=tmp_db_path) is False

    def test_returns_false_for_other_account(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        gen = ChatGPTImageGeneration(
            created_at=datetime.now(timezone.utc).isoformat(),
            id="g1",
            conversation_id="c1",
            message_id="m1",
            asset_pointer="a1",
            url="https://example.com/1.png",
            prompt="test",
        )
        sqlite_store.upsert_generations("acc", [gen], db_path=tmp_db_path)
        assert sqlite_store.has_generations("other", db_path=tmp_db_path) is False


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
        # first_active and last_active should only reflect active items
        first_active = datetime.fromisoformat(stats["first_active"])
        last_active = datetime.fromisoformat(stats["last_active"])
        assert first_active > cooldown
        assert last_active > cooldown


class TestCountRecentGenerations:
    def test_counts_within_window(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        now = datetime.now(timezone.utc)
        gens = [
            ChatGPTImageGeneration(
                created_at=(now - timedelta(hours=h)).isoformat(),
                id=f"g{h}",
                conversation_id="c1",
                message_id=f"m{h}",
                asset_pointer=f"a{h}",
                url=f"https://example.com/{h}.png",
                prompt="test",
            )
            for h in [1, 2, 3, 25, 26]
        ]
        sqlite_store.upsert_generations("acc", gens, db_path=tmp_db_path)

        since = (now - timedelta(days=1)).isoformat()
        count = sqlite_store.count_recent_generations("acc", since, db_path=tmp_db_path)
        assert count == 3

    def test_returns_zero_when_none(self, tmp_db_path):
        sqlite_store.init_db(tmp_db_path)
        since = datetime.now(timezone.utc).isoformat()
        count = sqlite_store.count_recent_generations("acc", since, db_path=tmp_db_path)
        assert count == 0

