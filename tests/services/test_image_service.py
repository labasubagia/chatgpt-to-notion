"""
Unit tests for img.py - image processing functions.
"""

from unittest.mock import patch

import pytest
from PIL import Image

from chatgpt_to_notion.services.image_service import edit_png_info, get_png_prompt
from chatgpt_to_notion.shared.constants import MAX_RETRIES
from chatgpt_to_notion.shared.verbosity import SIMPLE, VERBOSE, set_verbosity


@pytest.fixture(autouse=True)
def _set_verbose_for_tests():
    """Set VERBOSE mode so per-file output assertions work."""
    set_verbosity(VERBOSE)
    yield
    set_verbosity(SIMPLE)


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


