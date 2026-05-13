"""
Integration tests for sora.py using pure mocking.

No real API calls are made. All external dependencies are mocked.
"""
from unittest.mock import AsyncMock

import pytest

from tests.conftest import make_mock_response

from sora import (
    archive_generation,
    archive_task,
    cleanup_tasks,
    cleanup_trash,
    delete_empty_tasks,
    delete_generation,
    delete_generations,
    delete_generations_already_uploaded_to_notion,
    delete_task,
    download_all_images,
    fetch_all_lists_tasks,
    fetch_list_tasks,
    fetch_recent_tasks,
    get_generation_download_url,
    get_generations_from_tasks,
    get_headers,
    trash_generations_already_uploaded_to_notion,
    upload_to_notion,
)


@pytest.mark.integration
class TestSoraHeaders:
    """Tests for Sora headers generation."""

    def test_headers_contain_auth(self, mock_env_vars):
        """Headers should contain Authorization."""
        headers = get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")

    def test_headers_contain_user_agent(self, mock_env_vars):
        """Headers should contain User-Agent."""
        headers = get_headers()
        assert "User-Agent" in headers

    def test_headers_contain_content_type(self, mock_env_vars):
        """Headers should contain Content-Type."""
        headers = get_headers()
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"


@pytest.mark.integration
class TestSoraTasks:
    """Tests for Sora task operations."""

    async def test_fetch_recent_tasks(self, mock_aiohttp_session):
        """Should fetch recent tasks."""
        mock_aiohttp_session._responses = [
            make_mock_response({"task_responses": [{"id": "task_1"}]})
        ]

        result = await fetch_recent_tasks(mock_aiohttp_session, limit=5)
        assert "task_responses" in result
        assert len(result["task_responses"]) == 1

    async def test_fetch_list_tasks(self, mock_aiohttp_session):
        """Should fetch list of tasks."""
        mock_aiohttp_session._responses = [
            make_mock_response({"task_responses": [{"id": "task_1"}]})
        ]

        result = await fetch_list_tasks(mock_aiohttp_session, limit=5)
        assert "task_responses" in result

    async def test_fetch_all_lists_tasks(self, monkeypatch):
        """Should fetch all tasks with pagination."""
        from unittest.mock import AsyncMock, patch
        
        # Mock the entire function to avoid real API calls
        mock_tasks = [
            {"id": "task_1", "generations": []},
            {"id": "task_2", "generations": []},
        ]
        
        with patch("sora.fetch_list_tasks", new_callable=AsyncMock) as mock_fetch:
            # First call returns has_more=True
            # Second call returns has_more=False
            mock_fetch.side_effect = [
                {"task_responses": [mock_tasks[0]], "has_more": True, "last_id": "task_1"},
                {"task_responses": [mock_tasks[1]], "has_more": False, "last_id": None},
            ]
            
            tasks = await fetch_all_lists_tasks(limit=5)
            assert isinstance(tasks, list)
            assert len(tasks) == 2
            assert mock_fetch.call_count == 2

    async def test_archive_task(self, mock_aiohttp_session):
        """Should archive/trash a task."""
        mock_aiohttp_session._responses = [
            make_mock_response({"success": True})
        ]

        result = await archive_task(mock_aiohttp_session, "task_test")
        assert result is not None

    async def test_delete_task(self, mock_aiohttp_session):
        """Should delete a task."""
        mock_aiohttp_session._responses = [
            make_mock_response({"success": True})
        ]

        result = await delete_task(mock_aiohttp_session, "task_test")
        assert result is not None


@pytest.mark.integration
class TestSoraGenerations:
    """Tests for Sora generation operations."""

    async def test_archive_generation(self, mock_aiohttp_session):
        """Should archive/trash a generation."""
        mock_aiohttp_session._responses = [
            make_mock_response({"success": True})
        ]

        result = await archive_generation(mock_aiohttp_session, "gen_test")
        assert result is not None

    async def test_delete_generation(self, mock_aiohttp_session):
        """Should delete a generation."""
        mock_aiohttp_session._responses = [
            make_mock_response({"success": True})
        ]

        result = await delete_generation(mock_aiohttp_session, "gen_test")
        assert result is not None


class TestSoraDataProcessing:
    """Tests for Sora data processing functions."""

    def test_get_generations_from_tasks(self):
        """Should extract generations from tasks."""
        tasks = [
            {
                "id": "task_123",
                "created_at": "2024-01-15T10:30:00Z",
                "generations": [
                    {
                        "id": "gen_abc",
                        "task_id": "task_123",
                        "url": "https://example.com/img1.png",
                        "prompt": "First prompt",
                    },
                    {
                        "id": "gen_def",
                        "task_id": "task_123",
                        "url": "https://example.com/img2.png",
                        "prompt": "Second prompt",
                    },
                ],
            }
        ]

        generations = get_generations_from_tasks(tasks)

        assert len(generations) == 2
        assert generations[0].id == "gen_abc"
        assert generations[0].prompt == "First prompt"
        assert generations[1].id == "gen_def"

    def test_get_generations_sorted_by_created_at(self):
        """Should sort generations by created_at."""
        tasks = [
            {
                "id": "task_1",
                "created_at": "2024-01-15T12:00:00Z",
                "generations": [
                    {"id": "gen_later", "task_id": "task_1", "url": "url2", "prompt": "p2"}
                ],
            },
            {
                "id": "task_2",
                "created_at": "2024-01-15T10:00:00Z",
                "generations": [
                    {"id": "gen_earlier", "task_id": "task_2", "url": "url1", "prompt": "p1"}
                ],
            },
        ]

        generations = get_generations_from_tasks(tasks)

        assert len(generations) == 2
        assert generations[0].id == "gen_earlier"
        assert generations[1].id == "gen_later"

    def test_get_generations_empty_tasks(self):
        """Should handle tasks with no generations."""
        tasks = [
            {"id": "task_1", "created_at": "2024-01-15T10:00:00Z", "generations": []},
            {"id": "task_2", "created_at": "2024-01-15T11:00:00Z", "generations": []},
        ]

        generations = get_generations_from_tasks(tasks)
        assert len(generations) == 0

    def test_get_generations_missing_fields(self):
        """Should handle missing fields gracefully."""
        tasks = [
            {
                "id": "task_1",
                "created_at": "2024-01-15T10:00:00Z",
                "generations": [
                    {"id": "gen_1"},
                ],
            }
        ]

        generations = get_generations_from_tasks(tasks)

        assert len(generations) == 1
        assert generations[0].task_id is None
        assert generations[0].url is None
        assert generations[0].prompt == ""


@pytest.mark.integration
class TestSoraUploadToNotion:
    """Tests for upload_to_notion function."""

    async def test_upload_to_notion_full_workflow(self, monkeypatch, tmp_path):
        """Should execute full upload workflow."""
        from unittest.mock import patch, AsyncMock
        
        image_folder = str(tmp_path / "images")
        
        with patch("sora.fetch_recent_tasks", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {"task_responses": [{"id": "task_1", "generations": []}]}
            
            with patch("sora.get_generations_from_tasks") as mock_gen:
                mock_gen.return_value = [{"id": "gen_1", "prompt": "Test"}]
                
                with patch("sora.download_all_images", new_callable=AsyncMock) as mock_download:
                    mock_download.return_value = None
                    
                    with patch("sora.add_prompt_to_images") as mock_add:
                        mock_add.return_value = None
                        
                        with patch("sora.upload_all_images_to_notion", new_callable=AsyncMock) as mock_upload:
                            mock_upload.return_value = None
                            
                            await upload_to_notion(
                                image_folder=image_folder,
                                db_id="test_db",
                                upload_to_notion=True,
                                trash_in_sora=False,
                                remove_in_sora=False,
                                add_prompt_to_image=True,
                                limit=5,
                            )
                            
                            mock_fetch.assert_called_once()
                            mock_gen.assert_called_once()
                            mock_download.assert_called_once()
                            mock_add.assert_called_once()
                            mock_upload.assert_called_once()

    async def test_upload_to_notion_trash_in_sora(self, monkeypatch, tmp_path):
        """Should trash generations in Sora after upload."""
        from unittest.mock import patch, AsyncMock
        from models import SoraImageGeneration
        
        image_folder = str(tmp_path / "images")
        generations = [
            SoraImageGeneration(
                created_at="2024-01-15T10:30:00.000000+00:00",
                id="gen_123",
                task_id="task_xyz",
                url="https://example.com/image.png",
                prompt="Test prompt",
            )
        ]
        
        with patch("sora.fetch_recent_tasks", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {"task_responses": []}
            
            with patch("sora.get_generations_from_tasks") as mock_gen:
                mock_gen.return_value = generations
                
                with patch("sora.download_all_images", new_callable=AsyncMock):
                    with patch("sora.add_prompt_to_images"):
                        with patch("sora.upload_all_images_to_notion", new_callable=AsyncMock):
                            with patch("sora.trash_generations_already_uploaded_to_notion", new_callable=AsyncMock) as mock_trash:
                                await upload_to_notion(
                                    image_folder=image_folder,
                                    db_id="test_db",
                                    trash_in_sora=True,
                                )
                                
                                mock_trash.assert_called_once()


@pytest.mark.integration
class TestSoraCleanup:
    """Tests for cleanup functions."""

    async def test_cleanup_trash(self, monkeypatch):
        """Should cleanup trashed generations."""
        from unittest.mock import patch, AsyncMock
        
        with patch("sora.fetch_all_lists_tasks", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [{"id": "task_1", "generations": []}]
            
            with patch("sora.get_generations_from_tasks") as mock_gen:
                mock_gen.return_value = [{"id": "gen_1"}]
                
                with patch("sora.delete_generations", new_callable=AsyncMock) as mock_delete:
                    await cleanup_trash(task_limit=100)
                    
                    mock_fetch.assert_called_once()
                    mock_gen.assert_called_once()
                    mock_delete.assert_called_once()

    async def test_cleanup_tasks(self, monkeypatch):
        """Should cleanup empty tasks."""
        from unittest.mock import patch
        
        with patch("sora.delete_empty_tasks", new_callable=AsyncMock) as mock_delete:
            await cleanup_tasks()
            mock_delete.assert_called_once()


@pytest.mark.integration
class TestSoraFetchTasks:
    """Tests for task fetching functions."""

    async def test_fetch_recent_tasks_with_params(self, mock_aiohttp_session):
        """Should fetch recent tasks with pagination params."""
        mock_aiohttp_session._responses = [
            make_mock_response({"task_responses": [{"id": "task_1"}]})
        ]

        result = await fetch_recent_tasks(
            mock_aiohttp_session, limit=10, before_task_id="task_0", archived=True
        )
        
        assert "task_responses" in result
        assert len(result["task_responses"]) == 1

    async def test_fetch_list_tasks_with_params(self, mock_aiohttp_session):
        """Should fetch list tasks with pagination params."""
        mock_aiohttp_session._responses = [
            make_mock_response({"task_responses": [{"id": "task_1"}]})
        ]

        result = await fetch_list_tasks(
            mock_aiohttp_session, limit=10, after_task_id="task_0", archived=True
        )
        
        assert "task_responses" in result

    async def test_fetch_all_lists_tasks_pagination(self, monkeypatch):
        """Should fetch all tasks with pagination."""
        from unittest.mock import patch, AsyncMock
        
        with patch("sora.fetch_list_tasks", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = [
                {"task_responses": [{"id": "task_1"}], "has_more": True, "last_id": "task_1"},
                {"task_responses": [{"id": "task_2"}], "has_more": False, "last_id": None},
            ]
            
            tasks = await fetch_all_lists_tasks(limit=5)
            
            assert len(tasks) == 2
            assert mock_fetch.call_count == 2

    async def test_fetch_all_lists_tasks_error_handling(self, monkeypatch, capsys):
        """Should handle fetch errors gracefully."""
        from unittest.mock import patch, AsyncMock
        
        with patch("sora.fetch_list_tasks", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("API Error")
            
            tasks = await fetch_all_lists_tasks(limit=5)
            
            assert tasks == []
            captured = capsys.readouterr()
            assert "fetch error" in captured.out

    async def test_get_generation_download_url(self, mock_aiohttp_session):
        """Should get download URL for generation."""
        mock_aiohttp_session._responses = [
            make_mock_response({"url": "https://example.com/download.png"})
        ]

        url = await get_generation_download_url(mock_aiohttp_session, "gen_123")
        
        assert url == "https://example.com/download.png"


@pytest.mark.integration
class TestSoraDownloadImages:
    """Tests for download_all_images function."""

    async def test_download_all_images_success(self, monkeypatch, tmp_path):
        """Should download all images."""
        from unittest.mock import patch, AsyncMock
        from models import SoraImageGeneration
        
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))
        download_folder = "images"
        generations = [
            SoraImageGeneration(id="gen_1", url="https://example.com/img1.png"),
            SoraImageGeneration(id="gen_2", url="https://example.com/img2.png"),
        ]
        
        with patch("sora.download_image", new_callable=AsyncMock) as mock_download:
            await download_all_images(generations=generations, download_folder=download_folder)
            
            assert mock_download.call_count == 2

    async def test_download_all_images_skip_existing(self, monkeypatch, tmp_path, capsys):
        """Should skip already downloaded images."""
        from unittest.mock import patch, AsyncMock
        from models import SoraImageGeneration
        
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))
        download_folder = "images"
        
        # Create existing image
        images_dir = tmp_path / download_folder
        images_dir.mkdir()
        (images_dir / "gen_1.png").write_bytes(b"fake png")
        
        generations = [
            SoraImageGeneration(id="gen_1", url="https://example.com/img1.png"),
            SoraImageGeneration(id="gen_2", url="https://example.com/img2.png"),
        ]
        
        with patch("sora.download_image", new_callable=AsyncMock) as mock_download:
            await download_all_images(generations=generations, download_folder=download_folder)
            
            # Only gen_2 should be downloaded
            assert mock_download.call_count == 1
            captured = capsys.readouterr()
            assert "skipped, already exists" in captured.out


@pytest.mark.integration
class TestSoraDeleteGenerations:
    """Tests for deletion functions."""

    async def test_delete_generation(self, mock_aiohttp_session):
        """Should delete a generation."""
        mock_aiohttp_session._responses = [
            make_mock_response({"success": True})
        ]

        result = await delete_generation(mock_aiohttp_session, "gen_123")
        
        assert result is not None

    async def test_delete_generations(self, monkeypatch):
        """Should delete multiple generations."""
        from unittest.mock import patch, AsyncMock
        from models import SoraImageGeneration
        
        generations = [SoraImageGeneration(id="gen_1"), SoraImageGeneration(id="gen_2")]
        
        with patch("sora.delete_generation", new_callable=AsyncMock) as mock_delete:
            await delete_generations(generations)
            
            assert mock_delete.call_count == 2

    async def test_delete_generations_already_uploaded_to_notion(self, monkeypatch, capsys):
        """Should delete generations that exist in Notion."""
        from unittest.mock import patch, AsyncMock
        from models import SoraImageGeneration
        
        generations = [SoraImageGeneration(id="gen_1")]
        
        with patch("sora.is_page_exists_in_db", new_callable=AsyncMock) as mock_exists:
            mock_exists.return_value = True
            
            with patch("sora.delete_generation", new_callable=AsyncMock) as mock_delete:
                await delete_generations_already_uploaded_to_notion(
                    generations, "test_db"
                )
                
                mock_exists.assert_called_once()
                mock_delete.assert_called_once()

    async def test_delete_generations_skip_not_uploaded(self, monkeypatch, capsys):
        """Should skip generations not in Notion."""
        from unittest.mock import patch, AsyncMock
        from models import SoraImageGeneration
        
        generations = [SoraImageGeneration(id="gen_1")]
        
        with patch("sora.is_page_exists_in_db", new_callable=AsyncMock) as mock_exists:
            mock_exists.return_value = False
            
            with patch("sora.delete_generation", new_callable=AsyncMock) as mock_delete:
                await delete_generations_already_uploaded_to_notion(
                    generations, "test_db"
                )
                
                mock_exists.assert_called_once()
                mock_delete.assert_not_called()
                captured = capsys.readouterr()
                assert "skipped, not uploaded to notion yet" in captured.out


@pytest.mark.integration
class TestSoraTrashGenerations:
    """Tests for trash functions."""

    async def test_trash_generations_already_uploaded_to_notion(self, monkeypatch):
        """Should trash generations that exist in Notion."""
        from unittest.mock import patch, AsyncMock
        from models import SoraImageGeneration
        
        generations = [SoraImageGeneration(id="gen_1")]
        
        with patch("sora.is_page_exists_in_db", new_callable=AsyncMock) as mock_exists:
            mock_exists.return_value = True
            
            with patch("sora.archive_generation", new_callable=AsyncMock) as mock_archive:
                await trash_generations_already_uploaded_to_notion(
                    generations, "test_db"
                )
                
                mock_exists.assert_called_once()
                mock_archive.assert_called_once()

    async def test_trash_generations_skip_not_uploaded(self, monkeypatch, capsys):
        """Should skip generations not in Notion."""
        from unittest.mock import patch, AsyncMock
        from models import SoraImageGeneration
        
        generations = [SoraImageGeneration(id="gen_1")]
        
        with patch("sora.is_page_exists_in_db", new_callable=AsyncMock) as mock_exists:
            mock_exists.return_value = False
            
            with patch("sora.archive_generation", new_callable=AsyncMock) as mock_archive:
                await trash_generations_already_uploaded_to_notion(
                    generations, "test_db"
                )
                
                mock_exists.assert_called_once()
                mock_archive.assert_not_called()
                captured = capsys.readouterr()
                assert "skipped, not uploaded to notion yet" in captured.out


@pytest.mark.integration
class TestSoraUploadToNotionValidation:
    """Tests for upload_to_notion validation."""

    async def test_upload_to_notion_validation_error(self, tmp_path):
        """Should raise error when both trash and remove are True."""
        from unittest.mock import patch
        
        with pytest.raises(ValueError, match="cannot be both True"):
            with patch("sora.fetch_recent_tasks"):
                await upload_to_notion(
                    image_folder=str(tmp_path),
                    db_id="test_db",
                    trash_in_sora=True,
                    remove_in_sora=True,
                )

    async def test_upload_to_notion_with_dataset(self, monkeypatch, tmp_path):
        """Should save to dataset when provided."""
        from unittest.mock import patch, AsyncMock
        from models import SoraImageGeneration
        
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))
        generations = [
            SoraImageGeneration(
                created_at="2024-01-15T10:30:00.000000+00:00",
                id="gen_123",
                task_id="task_xyz",
                url="https://example.com/image.png",
                prompt="Test prompt",
            )
        ]
        
        with patch("sora.fetch_recent_tasks", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = {"task_responses": []}
            
            with patch("sora.get_generations_from_tasks") as mock_gen:
                mock_gen.return_value = generations
                
                with patch("sora.download_all_images", new_callable=AsyncMock):
                    with patch("sora.add_prompt_to_images"):
                        with patch("sora.upload_all_images_to_notion", new_callable=AsyncMock):
                            with patch("sora.save_to_dataset") as mock_save:
                                await upload_to_notion(
                                    image_folder="images",
                                    db_id="test_db",
                                    dataset="test_dataset.csv",
                                )
                                
                                mock_save.assert_called_once()


@pytest.mark.integration
class TestSoraCleanupExtended:
    """Tests for cleanup functions with more coverage."""

    async def test_cleanup_trash_with_dataset(self, monkeypatch, tmp_path):
        """Should save to dataset during cleanup."""
        from unittest.mock import patch, AsyncMock
        
        monkeypatch.setattr("util.OUTPUT_PATH", str(tmp_path))
        
        with patch("sora.fetch_all_lists_tasks", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []
            
            with patch("sora.get_generations_from_tasks") as mock_gen:
                mock_gen.return_value = []
                
                with patch("sora.delete_generations", new_callable=AsyncMock):
                    with patch("sora.save_to_dataset") as mock_save:
                        await cleanup_trash(task_limit=50, dataset="cleanup.csv")
                        
                        mock_fetch.assert_called_once()
                        mock_save.assert_called_once()


@pytest.mark.integration
class TestSoraDeleteEmptyTasks:
    """Tests for delete_empty_tasks function."""

    async def test_delete_empty_tasks_success(self, monkeypatch, capsys):
        """Should delete tasks with no generations."""
        from unittest.mock import patch, AsyncMock
        
        with patch("sora.fetch_all_lists_tasks", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [
                {"id": "task_1", "generations": []},
                {"id": "task_2", "generations": []},
            ]
            
            with patch("sora.archive_task", new_callable=AsyncMock) as mock_archive:
                with patch("sora.delete_task", new_callable=AsyncMock) as mock_delete:
                    await delete_empty_tasks()
                    
                    mock_fetch.assert_called_once()
                    assert mock_archive.call_count == 2
                    assert mock_delete.call_count == 2
                    captured = capsys.readouterr()
                    assert "Deleting empty tasks" in captured.err or "✅" in captured.out

    async def test_delete_empty_tasks_skip_non_empty(self, monkeypatch, capsys):
        """Should skip tasks that have generations."""
        from unittest.mock import patch, AsyncMock
        
        with patch("sora.fetch_all_lists_tasks", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [
                {"id": "task_1", "generations": [{"id": "gen_1"}]},
                {"id": "task_2", "generations": []},
            ]
            
            with patch("sora.archive_task", new_callable=AsyncMock) as mock_archive:
                with patch("sora.delete_task", new_callable=AsyncMock) as mock_delete:
                    await delete_empty_tasks()
                    
                    # Only task_2 should be deleted
                    assert mock_archive.call_count == 1
                    assert mock_delete.call_count == 1

    async def test_delete_empty_tasks_no_empty_tasks(self, monkeypatch, capsys):
        """Should handle case with no empty tasks."""
        from unittest.mock import patch, AsyncMock
        
        with patch("sora.fetch_all_lists_tasks", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [
                {"id": "task_1", "generations": [{"id": "gen_1"}]},
                {"id": "task_2", "generations": [{"id": "gen_2"}]},
            ]
            
            with patch("sora.archive_task", new_callable=AsyncMock) as mock_archive:
                with patch("sora.delete_task", new_callable=AsyncMock) as mock_delete:
                    await delete_empty_tasks()
                    
                    assert mock_archive.call_count == 0
                    assert mock_delete.call_count == 0

    async def test_delete_empty_tasks_error_handling(self, monkeypatch, capsys):
        """Should handle errors during deletion gracefully."""
        from unittest.mock import patch, AsyncMock
        
        with patch("sora.fetch_all_lists_tasks", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [
                {"id": "task_1", "generations": []},
            ]
            
            with patch("sora.archive_task", new_callable=AsyncMock) as mock_archive:
                with patch("sora.delete_task", new_callable=AsyncMock) as mock_delete:
                    mock_delete.side_effect = Exception("Delete failed")
                    
                    await delete_empty_tasks()
                    
                    captured = capsys.readouterr()
                    assert "failed" in captured.out
