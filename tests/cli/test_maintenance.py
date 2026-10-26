"""Tests for maintenance CLI commands."""

import shutil
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chatgpt_to_notion.cli.app import app
from chatgpt_to_notion.shared import constants


runner = CliRunner()


class TestCleanOutputPath:
    """Tests for clean-output-path command."""

    def test_clean_output_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(constants, "OUTPUT_PATH", str(tmp_path))
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / ".gitkeep").write_text("keep")
        result = runner.invoke(app, ["clean-output-path"])
        assert result.exit_code == 0
        assert not (tmp_path / "file.txt").exists()
        assert (tmp_path / ".gitkeep").exists()
        assert "Output path cleaned" in result.output

    def test_clean_output_path_nonexistent(self, tmp_path, monkeypatch):
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.setattr(constants, "OUTPUT_PATH", str(nonexistent))
        result = runner.invoke(app, ["clean-output-path"])
        assert result.exit_code == 0
        assert "Output path cleaned" in result.output


class TestBackupDb:
    """Tests for backup-db command."""

    def test_backup_db(self, isolated_db, tmp_path, monkeypatch):
        from chatgpt_to_notion.adapters import sqlite_store

        monkeypatch.setattr(sqlite_store, "_get_db_path", lambda: isolated_db)
        backup_target = tmp_path / "backup.sqlite"
        result = runner.invoke(app, ["backup-db", str(backup_target)])
        assert result.exit_code == 0
        assert backup_target.exists()
        assert backup_target.stat().st_size > 0
        assert "Backup saved to" in result.output

    def test_backup_db_creates_parent_dirs(self, isolated_db, tmp_path, monkeypatch):
        from chatgpt_to_notion.adapters import sqlite_store

        monkeypatch.setattr(sqlite_store, "_get_db_path", lambda: isolated_db)
        backup_target = tmp_path / "subdir" / "backup.sqlite"
        result = runner.invoke(app, ["backup-db", str(backup_target)])
        assert result.exit_code == 0
        assert backup_target.exists()
        assert backup_target.parent.exists()


class TestRestoreDb:
    """Tests for restore-db command."""

    def test_restore_db(self, tmp_path, monkeypatch):
        from chatgpt_to_notion.adapters import sqlite_store

        original_db = tmp_path / "original.sqlite"
        sqlite_store.init_db(original_db)
        conn = sqlite3.connect(original_db)
        conn.execute(
            "INSERT INTO image_generations (id, created_at, conversation_id, message_id, asset_pointer, url, prompt, account) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("gen1", "2024-01-01T00:00:00Z", "conv1", "msg1", "asset1", "http://example.com", "prompt1", "test@example.com"),
        )
        conn.execute(
            "INSERT INTO image_generations (id, created_at, conversation_id, message_id, asset_pointer, url, prompt, account) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("gen2", "2024-01-02T00:00:00Z", "conv2", "msg2", "asset2", "http://example.com", "prompt2", "test@example.com"),
        )
        conn.commit()
        conn.close()

        backup_db = tmp_path / "backup.sqlite"
        shutil.copy2(original_db, backup_db)

        conn = sqlite3.connect(original_db)
        conn.execute("DELETE FROM image_generations")
        conn.commit()
        conn.close()

        monkeypatch.setattr(sqlite_store, "_get_db_path", lambda: original_db)
        result = runner.invoke(app, ["restore-db", str(backup_db)])
        assert result.exit_code == 0
        assert "Database restored from" in result.output

        conn = sqlite3.connect(original_db)
        rows = conn.execute("SELECT prompt FROM image_generations ORDER BY prompt").fetchall()
        conn.close()
        assert len(rows) == 2
        assert rows[0][0] == "prompt1"
        assert rows[1][0] == "prompt2"

    def test_restore_db_missing_backup_raises(self, tmp_path, monkeypatch):
        from chatgpt_to_notion.adapters import sqlite_store

        monkeypatch.setattr(sqlite_store, "_get_db_path", lambda: tmp_path / "original.sqlite")
        missing_backup = tmp_path / "missing.sqlite"
        result = runner.invoke(app, ["restore-db", str(missing_backup)])
        assert result.exit_code != 0
        assert isinstance(result.exception, FileNotFoundError)
        assert "Backup file not found" in str(result.exception)


class TestMaintenanceCommandRegistration:
    """Tests that maintenance commands are registered."""

    def test_commands_exist(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "clean-output-path" in result.output
        assert "backup-db" in result.output
        assert "restore-db" in result.output
        # delete-library moved to upload library
        assert "delete-library" not in result.output