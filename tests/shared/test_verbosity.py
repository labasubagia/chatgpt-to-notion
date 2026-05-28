"""Tests for verbosity module."""

import json

from chatgpt_to_notion.shared.verbosity import (
    QUIET,
    SIMPLE,
    VERBOSE,
    StageCounter,
    get_verbosity,
    is_quiet,
    is_verbose,
    set_verbosity,
    write_fail_log,
)


class TestVerbosityLevel:
    def test_default_is_simple(self):
        set_verbosity(SIMPLE)
        assert get_verbosity() == SIMPLE

    def test_set_verbose(self):
        set_verbosity(VERBOSE)
        assert get_verbosity() == VERBOSE
        assert is_verbose() is True
        assert is_quiet() is False

    def test_set_quiet(self):
        set_verbosity(QUIET)
        assert get_verbosity() == QUIET
        assert is_verbose() is False
        assert is_quiet() is True

    def test_simple_not_verbose_not_quiet(self):
        set_verbosity(SIMPLE)
        assert is_verbose() is False
        assert is_quiet() is False


class TestStageCounter:
    def test_single_category(self):
        c = StageCounter("Downloaded")
        c.add("success", 3)
        assert c.summary_line() == "Downloaded: 3 success"

    def test_multiple_categories(self):
        c = StageCounter("Uploaded")
        c.add("success", 2)
        c.add("skipped", 1)
        c.add("failed", 0)
        line = c.summary_line()
        assert "2 success" in line
        assert "1 skipped" in line
        assert "0 failed" not in line

    def test_nothing_to_do(self):
        c = StageCounter("Processed")
        assert c.summary_line() == "Processed: nothing to do"

    def test_default_count_is_one(self):
        c = StageCounter("Test")
        c.add("success")
        assert c.summary_line() == "Test: 1 success"


class TestWriteFailLog:
    def test_creates_file(self, tmp_path):
        log_path = tmp_path / "fails.jsonl"
        write_fail_log(log_path, {"stage": "upload", "file": "test.png", "error": "timeout"})

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["stage"] == "upload"
        assert entry["file"] == "test.png"
        assert entry["error"] == "timeout"
        assert "timestamp" in entry

    def test_appends_multiple_entries(self, tmp_path):
        log_path = tmp_path / "fails.jsonl"
        write_fail_log(log_path, {"error": "first"})
        write_fail_log(log_path, {"error": "second"})

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["error"] == "first"
        assert json.loads(lines[1])["error"] == "second"

    def test_preserves_existing_fields(self, tmp_path):
        log_path = tmp_path / "fails.jsonl"
        entry = {"stage": "download", "file": "a.png", "extra": 42}
        write_fail_log(log_path, entry)

        loaded = json.loads(log_path.read_text().strip())
        assert loaded["extra"] == 42
        assert loaded["stage"] == "download"
