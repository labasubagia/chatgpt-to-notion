"""
Unit tests for img.py - image processing functions.
"""

from unittest.mock import patch

import pytest
from PIL import Image

from img import add_prompt_to_images, edit_png_info, get_png_prompt
from util import MAX_RETRIES


class TestEditPngInfo:
    """Tests for edit_png_info function."""

    def test_file_not_found(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            edit_png_info("/nonexistent/file.png", {"Prompt": "test"})

    def test_adds_metadata(self, tmp_path, sample_image_bytes):
        """Should add metadata to PNG file."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        edit_png_info(str(img_path), {"Prompt": "Test prompt"})

        # Verify metadata was added
        with Image.open(img_path) as img:
            assert img.info.get("Prompt") == "Test prompt"

    def test_overwrite_existing(self, tmp_path, sample_image_bytes):
        """Should overwrite existing metadata by default."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        # Add initial metadata
        edit_png_info(str(img_path), {"Prompt": "Original"})

        # Overwrite
        edit_png_info(str(img_path), {"Prompt": "Updated"})

        with Image.open(img_path) as img:
            assert img.info.get("Prompt") == "Updated"

    def test_no_overwrite_flag(self, tmp_path, sample_image_bytes):
        """Should not overwrite when overwrite=False."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        # Add initial metadata
        edit_png_info(str(img_path), {"Prompt": "Original"})

        # Try to overwrite with overwrite=False
        edit_png_info(str(img_path), {"Prompt": "New"}, overwrite=False)

        with Image.open(img_path) as img:
            assert img.info.get("Prompt") == "Original"

    def test_preserves_existing_metadata(self, tmp_path, sample_image_bytes):
        """Should preserve existing metadata when adding new."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        # Add initial metadata
        edit_png_info(str(img_path), {"Prompt": "Test", "Author": "John"})

        # Add new metadata
        edit_png_info(str(img_path), {"Title": "Sunset"})

        with Image.open(img_path) as img:
            assert img.info.get("Prompt") == "Test"
            assert img.info.get("Author") == "John"
            assert img.info.get("Title") == "Sunset"

    def test_get_png_prompt(self, tmp_path, sample_image_bytes):
        """Should read existing prompt metadata."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)
        edit_png_info(str(img_path), {"Prompt": "Test prompt"})

        assert get_png_prompt(str(img_path)) == "Test prompt"


class TestAddPromptToImages:
    """Tests for add_prompt_to_images function."""

    def test_missing_image_skipped(
        self, tmp_path, capsys, sample_chatgpt_generations, monkeypatch
    ):
        """Missing images should be skipped with warning."""
        # Use relative path within tmp_path
        folder = tmp_path / "images"
        folder.mkdir()
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))

        add_prompt_to_images(sample_chatgpt_generations, "images")

        captured = capsys.readouterr()
        assert "not found, skipped" in captured.out

    def test_adds_prompts(
        self,
        tmp_path,
        capsys,
        sample_chatgpt_generations,
        sample_image_bytes,
        monkeypatch,
    ):
        """Should add prompts to existing images."""
        folder = tmp_path / "images"
        folder.mkdir()
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))

        # Create test images
        for gen in sample_chatgpt_generations:
            img_path = folder / f"{gen.id}.png"
            img_path.write_bytes(sample_image_bytes)

        add_prompt_to_images(sample_chatgpt_generations, "images")

        # Verify prompts were added
        for gen in sample_chatgpt_generations:
            img_path = folder / f"{gen.id}.png"
            with Image.open(img_path) as img:
                assert img.info.get("Prompt") == gen.prompt

        captured = capsys.readouterr()
        assert "✅" in captured.out

    def test_skips_when_prompt_unchanged(
        self,
        tmp_path,
        capsys,
        sample_chatgpt_generation,
        sample_image_bytes,
        monkeypatch,
    ):
        """Should not rewrite image metadata when prompt already matches."""
        folder = tmp_path / "images"
        folder.mkdir()
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))
        img_path = folder / f"{sample_chatgpt_generation.id}.png"
        img_path.write_bytes(sample_image_bytes)
        edit_png_info(str(img_path), {"Prompt": sample_chatgpt_generation.prompt})

        with patch("img.edit_png_info") as mock_edit:
            add_prompt_to_images([sample_chatgpt_generation], "images")

        mock_edit.assert_not_called()
        captured = capsys.readouterr()
        assert "prompt unchanged" in captured.out

    def test_retry_on_failure(
        self,
        tmp_path,
        capsys,
        sample_chatgpt_generations,
        sample_image_bytes,
        monkeypatch,
    ):
        """Should retry on failure up to MAX_RETRIES."""
        folder = tmp_path / "images"
        folder.mkdir()
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))

        # Create test image
        for gen in sample_chatgpt_generations:
            img_path = folder / f"{gen.id}.png"
            img_path.write_bytes(sample_image_bytes)

        # Mock edit_png_info to fail twice then succeed
        call_count = [0]

        def flaky_edit(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise OSError("Simulated failure")

        with patch("img.edit_png_info", side_effect=flaky_edit):
            add_prompt_to_images(sample_chatgpt_generations[:1], "images")

        captured = capsys.readouterr()
        assert "retrying" in captured.out
        assert "✅" in captured.out

    def test_fails_after_max_retries(
        self,
        tmp_path,
        capsys,
        sample_chatgpt_generations,
        sample_image_bytes,
        monkeypatch,
    ):
        """Should fail after MAX_RETRIES attempts."""
        folder = tmp_path / "images"
        folder.mkdir()
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))

        # Create test image
        for gen in sample_chatgpt_generations:
            img_path = folder / f"{gen.id}.png"
            img_path.write_bytes(sample_image_bytes)

        # Mock edit_png_info to always fail
        with patch("img.edit_png_info", side_effect=OSError("Always fails")):
            add_prompt_to_images(sample_chatgpt_generations[:1], "images")

        captured = capsys.readouterr()
        assert f"failed after {MAX_RETRIES} retries" in captured.out
