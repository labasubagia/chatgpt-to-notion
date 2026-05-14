"""
Integration tests for notion.py using pure mocking.

No real API calls are made. All external dependencies are mocked.
"""

from unittest.mock import AsyncMock, patch

import pytest

from models import SoraImageGeneration
from notion import (
    _db_data_sources_cache,
    _db_page_cache,
    add_page_to_db,
    create_upload_img,
    get_db_data_sources,
    get_headers,
    is_page_exists_in_db,
    query_data_source,
    send_upload_img,
    upload_all_images_to_notion,
)
from tests.conftest import make_mock_response


@pytest.mark.integration
class TestNotionHeaders:
    """Tests for Notion headers generation."""

    def test_headers_contain_auth(self, mock_config_toml):
        """Headers should contain Authorization."""
        headers = get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    def test_headers_contain_notion_version(self, mock_config_toml):
        """Headers should contain Notion-Version."""
        headers = get_headers()
        assert "Notion-Version" in headers
        assert headers["Notion-Version"] == "2025-09-03"

    def test_headers_contain_content_type(self, mock_config_toml):
        """Headers should contain Content-Type."""
        headers = get_headers()
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"


@pytest.mark.integration
class TestNotionDatabase:
    """Tests for Notion database operations."""

    async def test_get_db_data_sources(self, mock_aiohttp_session):
        """Should fetch database data sources."""
        mock_aiohttp_session._responses = [
            make_mock_response({"data_sources": [{"id": "ds_123"}]})
        ]

        sources = await get_db_data_sources(mock_aiohttp_session, "test_db_123")
        assert isinstance(sources, list)
        assert len(sources) == 1
        assert sources[0]["id"] == "ds_123"

    async def test_get_db_data_sources_uses_cache(self, mock_aiohttp_session):
        """Should use cache for repeated calls."""
        _db_data_sources_cache.clear()
        _db_data_sources_cache["test_db"] = [{"id": "cached_ds"}]

        sources = await get_db_data_sources(mock_aiohttp_session, "test_db")

        assert sources == [{"id": "cached_ds"}]

    async def test_query_data_source(self, mock_aiohttp_session):
        """Should query data source."""
        mock_aiohttp_session._responses = [
            make_mock_response({"results": [{"id": "page_1"}]})
        ]

        result = await query_data_source(mock_aiohttp_session, "ds_123", "test.png")
        assert "results" in result
        assert len(result["results"]) == 1

    async def test_is_page_exists_in_db(self, mock_aiohttp_session):
        """Should check if page exists in database."""
        with patch("notion.get_db_data_sources", new_callable=AsyncMock) as mock_get_ds:
            mock_get_ds.return_value = [{"id": "ds_123"}]

            with patch(
                "notion.query_data_source", new_callable=AsyncMock
            ) as mock_query:
                mock_query.return_value = {
                    "results": [
                        {
                            "properties": {
                                "Name": {"title": [{"text": {"content": "test.png"}}]}
                            }
                        }
                    ]
                }

                exists = await is_page_exists_in_db(
                    mock_aiohttp_session, "test_db_123", "test.png"
                )
                assert exists is True

    async def test_is_page_exists_not_found(self, mock_aiohttp_session):
        """Should return False if page not found."""
        with patch("notion.get_db_data_sources", new_callable=AsyncMock) as mock_get_ds:
            mock_get_ds.return_value = [{"id": "ds_123"}]

            with patch(
                "notion.query_data_source", new_callable=AsyncMock
            ) as mock_query:
                mock_query.return_value = {"results": []}

                exists = await is_page_exists_in_db(
                    mock_aiohttp_session, "test_db_123", "nonexistent.png"
                )
                assert exists is False

    async def test_is_page_exists_uses_cache(self, mock_aiohttp_session):
        """Should use cache for repeated checks."""
        _db_page_cache.clear()
        _db_page_cache.add("cached_image.png")

        exists = await is_page_exists_in_db(
            mock_aiohttp_session, "test_db_123", "cached_image.png"
        )
        assert exists is True


@pytest.mark.integration
class TestNotionUpload:
    """Tests for Notion upload operations."""

    async def test_create_upload_img(
        self, mock_aiohttp_session, sample_image_bytes, tmp_path
    ):
        """Should create file upload."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        mock_aiohttp_session._responses = [
            make_mock_response({"id": "upload_123", "filename": "test.png"})
        ]

        result = await create_upload_img(mock_aiohttp_session, str(img_path))
        assert "id" in result
        assert result["id"] == "upload_123"

    async def test_create_upload_img_file_not_found(self, mock_aiohttp_session):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            await create_upload_img(mock_aiohttp_session, "/nonexistent/file.png")

    async def test_send_upload_img(
        self, mock_aiohttp_session, sample_image_bytes, tmp_path
    ):
        """Should send file upload."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        mock_aiohttp_session._responses = [
            make_mock_response({"id": "upload_123", "status": "complete"})
        ]

        result = await send_upload_img(
            mock_aiohttp_session, "upload_123", str(img_path)
        )
        assert result is not None

    async def test_send_upload_img_file_not_found(self, mock_aiohttp_session):
        """Should raise FileNotFoundError when file does not exist."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            await send_upload_img(
                mock_aiohttp_session, "upload_123", "/nonexistent/file.png"
            )

    async def test_add_page_to_db(
        self, mock_aiohttp_session, sample_image_bytes, tmp_path
    ):
        """Should add page to database."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_image_bytes)

        mock_aiohttp_session._responses = [
            make_mock_response({"id": "upload_123"}),  # create_upload_img
            make_mock_response(
                {"id": "upload_123", "status": "complete"}
            ),  # send_upload_img
            make_mock_response(
                {  # add_page_to_db
                    "id": "page_123",
                    "properties": {
                        "Prompt": {"rich_text": [{"text": {"content": "Test prompt"}}]}
                    },
                }
            ),
        ]

        result = await add_page_to_db(
            mock_aiohttp_session,
            "test_db_123",
            str(img_path),
            "Test prompt",
            model="Sora",
        )
        assert "id" in result
        assert result["id"] == "page_123"

    async def test_add_page_to_db_file_not_found(self, mock_aiohttp_session):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            await add_page_to_db(
                mock_aiohttp_session,
                "test_db_123",
                "/nonexistent/file.png",
                "Test prompt",
            )


class TestNotionCaching:
    """Tests for Notion caching behavior."""

    def test_data_sources_cache_populated(self, mock_config_toml):
        """Data sources cache should be populated."""
        _db_data_sources_cache.clear()
        _db_data_sources_cache["test_db"] = [{"id": "ds_123"}]

        assert "test_db" in _db_data_sources_cache

    def test_page_cache_populated(self, mock_config_toml):
        """Page cache should be populated."""
        _db_page_cache.clear()
        _db_page_cache.add("test_image.png")

        assert "test_image.png" in _db_page_cache

    def test_cache_cleared_by_fixture(self):
        """Caches should be cleared by reset_caches fixture."""
        assert len(_db_data_sources_cache) == 0
        assert len(_db_page_cache) == 0


@pytest.mark.integration
class TestNotionUploadAllImages:
    """Tests for upload_all_images_to_notion function."""

    async def test_upload_all_images_success(self, monkeypatch, tmp_path):
        """Should upload all images to Notion."""
        from unittest.mock import AsyncMock, patch

        from models import SoraImageGeneration

        # Use relative path within tmp_path
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))
        image_folder = "images"

        # Create test image files
        images_dir = tmp_path / image_folder
        images_dir.mkdir()
        (images_dir / "img_123.png").write_bytes(b"fake png")
        (images_dir / "img_456.png").write_bytes(b"fake png")

        generations = [
            SoraImageGeneration(
                created_at="2024-01-01T00:00:00",
                id="img_123",
                task_id="task_1",
                url="https://example.com/img1.png",
                prompt="Test prompt 1",
            ),
            SoraImageGeneration(
                created_at="2024-01-01T00:00:00",
                id="img_456",
                task_id="task_2",
                url="https://example.com/img2.png",
                prompt="Test prompt 2",
            ),
        ]

        with patch(
            "notion.is_page_exists_in_db", new_callable=AsyncMock
        ) as mock_exists:
            mock_exists.return_value = False

            with patch("notion.add_page_to_db", new_callable=AsyncMock) as mock_add:
                mock_add.return_value = {"id": "page_123"}

                await upload_all_images_to_notion(
                    generations=generations,
                    db_id="test_db",
                    image_folder=image_folder,
                )

                assert mock_exists.call_count == 2
                assert mock_add.call_count == 2

    async def test_upload_all_images_skip_existing(self, monkeypatch, tmp_path):
        """Should skip images that already exist in Notion."""
        from unittest.mock import AsyncMock, patch

        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))
        image_folder = "images"

        # Create test image file
        images_dir = tmp_path / image_folder
        images_dir.mkdir()
        (images_dir / "img_123.png").write_bytes(b"fake png")

        generations = [
            SoraImageGeneration(
                created_at="2024-01-01T00:00:00",
                id="img_123",
                task_id="task_1",
                url="https://example.com/img.png",
                prompt="Test prompt",
            )
        ]

        with patch(
            "notion.is_page_exists_in_db", new_callable=AsyncMock
        ) as mock_exists:
            mock_exists.return_value = True

            with patch("notion.add_page_to_db", new_callable=AsyncMock) as mock_add:
                await upload_all_images_to_notion(
                    generations=generations,
                    db_id="test_db",
                    image_folder=image_folder,
                )

                mock_exists.assert_called_once()
                mock_add.assert_not_called()

    async def test_upload_all_images_skips_uploaded_at_csv(self, monkeypatch, tmp_path):
        """Should skip Notion API when uploaded_at is already set."""
        from unittest.mock import AsyncMock, patch

        import pandas as pd

        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))
        image_folder = "images"
        images_dir = tmp_path / image_folder
        images_dir.mkdir()
        (images_dir / "img_123.png").write_bytes(b"fake png")
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "id": "img_123",
                    "created_at": "2026-05-14T00:00:00+00:00",
                    "uploaded_at": "2026-05-14T01:00:00+00:00",
                }
            ]
        ).to_csv(history_dir / "default_chatgpt.csv", index=False)

        generations = [
            SoraImageGeneration(
                created_at="2026-05-14T00:00:00+00:00",
                id="img_123",
                task_id="task_1",
                url="https://example.com/img.png",
                prompt="Test prompt",
            )
        ]

        with patch(
            "notion.is_page_exists_in_db", new_callable=AsyncMock
        ) as mock_exists:
            with patch("notion.add_page_to_db", new_callable=AsyncMock) as mock_add:
                await upload_all_images_to_notion(
                    generations=generations,
                    db_id="test_db",
                    image_folder=image_folder,
                    dataset="history/default_chatgpt.csv",
                )

                mock_exists.assert_not_called()
                mock_add.assert_not_called()

    async def test_upload_all_images_check_notion_api_bypasses_uploaded_at_csv(
        self, monkeypatch, tmp_path
    ):
        """Should check Notion API when check_notion_api is set."""
        from unittest.mock import AsyncMock, patch

        import pandas as pd

        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))
        image_folder = "images"
        images_dir = tmp_path / image_folder
        images_dir.mkdir()
        (images_dir / "img_123.png").write_bytes(b"fake png")
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "id": "img_123",
                    "created_at": "2026-05-14T00:00:00+00:00",
                    "uploaded_at": "2026-05-14T01:00:00+00:00",
                }
            ]
        ).to_csv(history_dir / "default_chatgpt.csv", index=False)

        generations = [
            SoraImageGeneration(
                created_at="2026-05-14T00:00:00+00:00",
                id="img_123",
                task_id="task_1",
                url="https://example.com/img.png",
                prompt="Test prompt",
            )
        ]

        with patch(
            "notion.is_page_exists_in_db", new_callable=AsyncMock
        ) as mock_exists:
            mock_exists.return_value = True
            with patch("notion.add_page_to_db", new_callable=AsyncMock) as mock_add:
                await upload_all_images_to_notion(
                    generations=generations,
                    db_id="test_db",
                    image_folder=image_folder,
                    dataset="history/default_chatgpt.csv",
                    check_notion_api=True,
                )

                mock_exists.assert_called_once()
                mock_add.assert_not_called()

    async def test_upload_all_images_marks_uploaded_at_after_upload(
        self, monkeypatch, tmp_path
    ):
        """Should mark uploaded_at after successful upload."""
        from unittest.mock import AsyncMock, patch

        import pandas as pd

        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))
        image_folder = "images"
        images_dir = tmp_path / image_folder
        images_dir.mkdir()
        (images_dir / "img_123.png").write_bytes(b"fake png")
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        dataset_path = history_dir / "default_chatgpt.csv"
        pd.DataFrame(
            [
                {
                    "id": "img_123",
                    "created_at": "2026-05-14T00:00:00+00:00",
                    "uploaded_at": "",
                }
            ]
        ).to_csv(dataset_path, index=False)

        generations = [
            SoraImageGeneration(
                created_at="2026-05-14T00:00:00+00:00",
                id="img_123",
                task_id="task_1",
                url="https://example.com/img.png",
                prompt="Test prompt",
            )
        ]

        with patch(
            "notion.is_page_exists_in_db", new_callable=AsyncMock
        ) as mock_exists:
            mock_exists.return_value = False
            with patch("notion.add_page_to_db", new_callable=AsyncMock) as mock_add:
                mock_add.return_value = {"id": "page_123"}
                await upload_all_images_to_notion(
                    generations=generations,
                    db_id="test_db",
                    image_folder=image_folder,
                    dataset="history/default_chatgpt.csv",
                )

        history = pd.read_csv(dataset_path)
        assert history.loc[0, "uploaded_at"]
