"""Tests for db.py - SQLite persistence layer."""

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import db
from models import ChatGPTImageGeneration


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
        db.init_db(tmp_db_path)
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
        db.init_db(tmp_db_path)
        db.upsert_generations("test_account", sample_generations, tmp_db_path)

        result = db.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 3

    def test_upserts_existing_rows(self, tmp_db_path, sample_generation):
        db.init_db(tmp_db_path)
        db.upsert_generations("test_account", [sample_generation], tmp_db_path)

        updated = sample_generation.model_copy(update={"prompt": "Updated prompt"})
        db.upsert_generations("test_account", [updated], tmp_db_path)

        result = db.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 1
        assert result[0].prompt == "Updated prompt"

    def test_empty_list_does_nothing(self, tmp_db_path):
        db.init_db(tmp_db_path)
        db.upsert_generations("test_account", [], tmp_db_path)
        result = db.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 0


class TestGetGenerations:
    def test_returns_all_for_account(self, tmp_db_path, sample_generations):
        db.init_db(tmp_db_path)
        db.upsert_generations("test_account", sample_generations, tmp_db_path)
        db.upsert_generations(
            "other_account",
            [sample_generations[0].model_copy(update={"id": "other_id"})],
            tmp_db_path,
        )

        result = db.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 3

    def test_filters_unuploaded_by_default(self, tmp_db_path, sample_generation):
        db.init_db(tmp_db_path)
        db.upsert_generations("test_account", [sample_generation], tmp_db_path)
        db.mark_uploaded("test_account", {sample_generation.id}, tmp_db_path)

        result = db.get_generations("test_account", db_path=tmp_db_path)
        assert len(result) == 0

    def test_includes_uploaded_when_flagged(self, tmp_db_path, sample_generation):
        db.init_db(tmp_db_path)
        db.upsert_generations("test_account", [sample_generation], tmp_db_path)
        db.mark_uploaded("test_account", {sample_generation.id}, tmp_db_path)

        result = db.get_generations(
            "test_account", include_uploaded=True, db_path=tmp_db_path
        )
        assert len(result) == 1

    def test_filters_by_keep_days(self, tmp_db_path, sample_generations):
        db.init_db(tmp_db_path)
        db.upsert_generations("test_account", sample_generations, tmp_db_path)

        result = db.get_generations(
            "test_account", keep_days=3, db_path=tmp_db_path
        )
        assert len(result) == 2

    def test_filters_by_ids_filter(self, tmp_db_path, sample_generations):
        db.init_db(tmp_db_path)
        db.upsert_generations("test_account", sample_generations, tmp_db_path)

        ids = {sample_generations[0].id, sample_generations[2].id}
        result = db.get_generations(
            "test_account", ids_filter=ids, db_path=tmp_db_path
        )
        assert len(result) == 2
        result_ids = {g.id for g in result}
        assert result_ids == ids

    def test_ids_filter_empty_account_returns_empty(self, tmp_db_path):
        db.init_db(tmp_db_path)
        result = db.get_generations(
            "test_account", ids_filter={"nonexistent"}, db_path=tmp_db_path
        )
        assert result == []

    def test_empty_account_returns_empty_list(self, tmp_db_path):
        db.init_db(tmp_db_path)
        result = db.get_generations("nonexistent", db_path=tmp_db_path)
        assert result == []


class TestGetUploadedIds:
    def test_returns_uploaded_ids(self, tmp_db_path, sample_generation):
        db.init_db(tmp_db_path)
        db.upsert_generations("test_account", [sample_generation], tmp_db_path)
        db.mark_uploaded("test_account", {sample_generation.id}, tmp_db_path)

        result = db.get_uploaded_ids("test_account", db_path=tmp_db_path)
        assert result == {sample_generation.id}

    def test_returns_empty_for_no_uploads(self, tmp_db_path, sample_generation):
        db.init_db(tmp_db_path)
        db.upsert_generations("test_account", [sample_generation], tmp_db_path)

        result = db.get_uploaded_ids("test_account", db_path=tmp_db_path)
        assert result == set()


class TestMarkUploaded:
    def test_sets_uploaded_at(self, tmp_db_path, sample_generation):
        db.init_db(tmp_db_path)
        db.upsert_generations("test_account", [sample_generation], tmp_db_path)
        db.mark_uploaded("test_account", {sample_generation.id}, tmp_db_path)

        conn = db._get_connection(tmp_db_path)
        row = conn.execute(
            "SELECT uploaded_at FROM image_generations WHERE id = ?",
            (sample_generation.id,),
        ).fetchone()
        conn.close()
        assert row["uploaded_at"] != ""

    def test_empty_ids_does_nothing(self, tmp_db_path):
        db.init_db(tmp_db_path)
        db.mark_uploaded("test_account", set(), tmp_db_path)


class TestDeleteOldGenerations:
    def test_deletes_rows_older_than_keep_days(self, tmp_db_path, sample_generations):
        db.init_db(tmp_db_path)
        db.upsert_generations("test_account", sample_generations, tmp_db_path)

        deleted = db.delete_old_generations("test_account", keep_days=3, db_path=tmp_db_path)
        assert deleted == 1

        remaining = db.get_generations(
            "test_account", include_uploaded=True, db_path=tmp_db_path
        )
        assert len(remaining) == 2

    def test_does_not_delete_other_accounts(self, tmp_db_path, sample_generation):
        db.init_db(tmp_db_path)
        old = sample_generation.model_copy(
            update={
                "id": "old_gen",
                "created_at": (
                    datetime.now(timezone.utc) - timedelta(days=10)
                ).isoformat(),
            }
        )
        db.upsert_generations("test_account", [old], tmp_db_path)
        db.upsert_generations("other_account", [old.model_copy(update={"id": "other_old"})], tmp_db_path)

        db.delete_old_generations("test_account", keep_days=2, db_path=tmp_db_path)

        other = db.get_generations("other_account", include_uploaded=True, db_path=tmp_db_path)
        assert len(other) == 1


class TestDataSourcesCache:
    def test_set_and_get(self, tmp_db_path):
        db.init_db(tmp_db_path)
        sources = [{"id": "ds_1"}, {"id": "ds_2"}]
        db.set_cached_data_sources("test_db", sources, db_path=tmp_db_path)

        result = db.get_cached_data_sources("test_db", db_path=tmp_db_path)
        assert result == sources

    def test_returns_none_when_missing(self, tmp_db_path):
        db.init_db(tmp_db_path)
        result = db.get_cached_data_sources("nonexistent", db_path=tmp_db_path)
        assert result is None

    def test_expires_after_ttl(self, tmp_db_path):
        db.init_db(tmp_db_path)
        sources = [{"id": "ds_1"}]
        db.set_cached_data_sources("test_db", sources, ttl_days=0, db_path=tmp_db_path)

        import time
        time.sleep(0.01)

        result = db.get_cached_data_sources("test_db", db_path=tmp_db_path)
        assert result is None


