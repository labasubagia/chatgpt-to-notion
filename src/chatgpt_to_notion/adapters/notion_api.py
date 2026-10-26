"""Notion API adapter."""

import asyncio
import mimetypes
import os
from typing import Any

import aiohttp

from ..domain.models import RuntimeOptions
from ..shared.http import (
    DetailedHTTPError,
    raise_for_status_with_detail,
    retry_http,
)
from ..shared.logging import get_logger
from .config_loader import ConfigLoader
from .config_loader import config_loader as default_config_loader
from .sqlite_store import SqliteStore
from .sqlite_store import sqlite_store as default_store

logger = get_logger("notion")

BASE_URL = "https://api.notion.com"


def get_headers(options: RuntimeOptions | None = None) -> dict[str, str]:
    return default_config_loader.notion_context(options).headers


def _content_type_for_ext(ext: str) -> str:
    """Return the MIME type for an image extension, defaulting to image/png."""
    guessed, _ = mimetypes.guess_type(f"file{ext}")
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/png"


class NotionClientImpl:
    """Concrete implementation of NotionClient protocol."""

    def __init__(
        self,
        options: RuntimeOptions | None = None,
        sqlite_store: SqliteStore | None = None,
        config_loader: ConfigLoader | None = None,
    ):
        self._options = options
        self._sqlite_store = sqlite_store or default_store
        self._config_loader = config_loader or default_config_loader

    def get_headers(self) -> dict[str, str]:
        return self._config_loader.notion_context(self._options).headers

    @retry_http()
    async def _get_db_data_sources(
        self,
        session: aiohttp.ClientSession,
        db_id: str,
    ) -> list[dict[str, Any]]:
        self._sqlite_store.init_db()
        cached = self._sqlite_store.get_cached_data_sources(db_id)
        if cached is not None:
            return cached

        async with session.get(
            f"{BASE_URL}/v1/databases/{db_id}", headers=self.get_headers()
        ) as response:
            await raise_for_status_with_detail(response)
            data = await response.json()
            data_sources = data.get("data_sources", [])
            await self._sqlite_store.async_set_cached_data_sources(db_id, data_sources)
            return data_sources

    @retry_http()
    async def _query_data_source(
        self,
        session: aiohttp.ClientSession,
        data_source_id: str,
        query: str,
    ) -> dict[str, Any]:
        async with session.post(
            f"{BASE_URL}/v1/data_sources/{data_source_id}/query",
            headers=self.get_headers(),
            json={
                "filter": {
                    "and": [{"property": "Name", "rich_text": {"equals": query}}]
                }
            },
        ) as response:
            await raise_for_status_with_detail(response)
            return await response.json()

    async def _query_data_source_for_page(
        self,
        session: aiohttp.ClientSession,
        data_source_id: str,
        query: str,
    ) -> bool:
        data = await self._query_data_source(session, data_source_id, query)
        for page in data.get("results", []):
            try:
                title_parts = page["properties"]["Name"]["title"]
                name = title_parts[0]["text"]["content"] if title_parts else ""
            except (KeyError, IndexError):
                logger.warning(
                    "Unexpected page structure for page %s, skipping",
                    page.get("id", "unknown"),
                )
                continue
            if name == query:
                return True
        return False

    async def is_page_exists_in_db(
        self,
        session: aiohttp.ClientSession,
        db_id: str,
        query: str,
    ) -> bool:
        data_sources = await self._get_db_data_sources(session, db_id)
        if not data_sources:
            return False

        results = await asyncio.gather(
            *[
                self._query_data_source_for_page(session, ds["id"], query)
                for ds in data_sources
            ],
            return_exceptions=True,
        )
        return any(r for r in results if isinstance(r, bool) and r)

    @retry_http()
    async def _create_upload_img(
        self,
        session: aiohttp.ClientSession,
        file_path: str,
        content_type: str = "image/png",
    ) -> dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        file_name = os.path.basename(file_path)

        async with session.post(
            f"{BASE_URL}/v1/file_uploads",
            headers=self.get_headers(),
            json={
                "mode": "single_part",
                "filename": file_name,
                "content_type": content_type,
            },
        ) as response:
            await raise_for_status_with_detail(response)
            return await response.json()

    @retry_http()
    async def _send_upload_img(
        self,
        session: aiohttp.ClientSession,
        file_upload_id: str,
        file_path: str,
        content_type: str = "image/png",
    ) -> dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        file_name = os.path.basename(file_path)
        headers = self.get_headers()
        with open(file_path, "rb") as file_obj:
            data = aiohttp.FormData()
            data.add_field(
                "file", file_obj, filename=file_name, content_type=content_type
            )
            try:
                async with session.post(
                    f"{BASE_URL}/v1/file_uploads/{file_upload_id}/send",
                    headers={
                        "Authorization": headers["Authorization"],
                        "Notion-Version": headers["Notion-Version"],
                    },
                    data=data,
                ) as response:
                    await raise_for_status_with_detail(response)
                    return await response.json()
            except DetailedHTTPError as exc:
                if exc.status == 400:
                    logger.info(
                        "File upload %s returned 400 (%s), assuming already uploaded "
                        "and proceeding.",
                        file_upload_id,
                        exc.message_text,
                    )
                    return {"id": file_upload_id, "status": "complete"}
                raise

    @retry_http()
    async def add_page_to_db(
        self,
        session: aiohttp.ClientSession,
        db_id: str,
        file_path: str,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower() or ".png"
        content_type = _content_type_for_ext(ext)

        notion_ctx = self._config_loader.notion_context(self._options)
        headers = notion_ctx.headers
        account_name = notion_ctx.account_name
        create_upload_res = await self._create_upload_img(
            session, file_path, content_type=content_type
        )
        send_upload_res = await self._send_upload_img(
            session,
            create_upload_res["id"],
            file_path,
            content_type=content_type,
        )

        normalized_prompt = (prompt or "").strip()
        normalized_prompt = normalized_prompt.replace("\r\n", "\n")
        normalized_prompt = normalized_prompt.replace("\r", "\n")

        body = f"**Prompt:**\n\n```\n{normalized_prompt}\n```"
        if account_name:
            body += f"\n\n**Email:** {account_name}"

        payload: dict[str, Any] = {
            "parent": {"database_id": db_id},
            "properties": {
                "Name": {"title": [{"text": {"content": file_name}}]},
                "Image": {
                    "files": [
                        {
                            "type": "file_upload",
                            "file_upload": {"id": send_upload_res["id"]},
                        }
                    ]
                },
            },
            "markdown": body,
        }

        async with session.post(
            f"{BASE_URL}/v1/pages",
            headers=headers,
            json=payload,
        ) as response:
            await raise_for_status_with_detail(response)
            return await response.json()
