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
from .config_loader import get_notion_context
from .sqlite_store import (
    async_set_cached_data_sources,
    get_cached_data_sources,
)

logger = get_logger("notion")

BASE_URL = "https://api.notion.com"


def get_headers(options: RuntimeOptions | None = None) -> dict[str, str]:
    return get_notion_context(options).headers


def _content_type_for_ext(ext: str) -> str:
    """Return the MIME type for an image extension, defaulting to image/png."""
    guessed, _ = mimetypes.guess_type(f"file{ext}")
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/png"


@retry_http()
async def get_db_data_sources(
    session: aiohttp.ClientSession,
    db_id: str,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    from . import sqlite_store

    sqlite_store.init_db()
    cached = get_cached_data_sources(db_id)
    if cached is not None:
        return cached

    async with session.get(
        f"{BASE_URL}/v1/databases/{db_id}", headers=headers or get_headers()
    ) as response:
        await raise_for_status_with_detail(response)
        data = await response.json()
        data_sources = data.get("data_sources", [])
        await async_set_cached_data_sources(db_id, data_sources)
        return data_sources


@retry_http()
async def query_data_source(
    session: aiohttp.ClientSession,
    data_source_id: str,
    query: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    async with session.post(
        f"{BASE_URL}/v1/data_sources/{data_source_id}/query",
        headers=headers or get_headers(),
        json={
            "filter": {"and": [{"property": "Name", "rich_text": {"equals": query}}]}
        },
    ) as response:
        await raise_for_status_with_detail(response)
        return await response.json()


async def _query_data_source_for_page(
    session: aiohttp.ClientSession,
    data_source_id: str,
    query: str,
    headers: dict[str, str],
) -> bool:
    data = await query_data_source(session, data_source_id, query, headers=headers)
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
    session: aiohttp.ClientSession,
    db_id: str,
    query: str,
    options: RuntimeOptions | None = None,
) -> bool:
    headers = get_headers(options)
    data_sources = await get_db_data_sources(session, db_id, headers=headers)
    if not data_sources:
        return False

    results = await asyncio.gather(
        *[
            _query_data_source_for_page(session, ds["id"], query, headers)
            for ds in data_sources
        ],
        return_exceptions=True,
    )
    return any(r for r in results if isinstance(r, bool) and r)


@retry_http()
async def create_upload_img(
    session: aiohttp.ClientSession,
    file_path: str,
    headers: dict[str, str] | None = None,
    content_type: str = "image/png",
) -> dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)

    async with session.post(
        f"{BASE_URL}/v1/file_uploads",
        headers=headers or get_headers(),
        json={
            "mode": "single_part",
            "filename": file_name,
            "content_type": content_type,
        },
    ) as response:
        await raise_for_status_with_detail(response)
        return await response.json()


@retry_http()
async def send_upload_img(
    session: aiohttp.ClientSession,
    file_upload_id: str,
    file_path: str,
    headers: dict[str, str] | None = None,
    content_type: str = "image/png",
) -> dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)
    notion_headers = headers or get_headers()
    with open(file_path, "rb") as file_obj:
        data = aiohttp.FormData()
        data.add_field("file", file_obj, filename=file_name, content_type=content_type)
        try:
            async with session.post(
                f"{BASE_URL}/v1/file_uploads/{file_upload_id}/send",
                headers={
                    "Authorization": notion_headers["Authorization"],
                    "Notion-Version": notion_headers["Notion-Version"],
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
    session: aiohttp.ClientSession,
    db_id: str,
    file_path: str,
    prompt: str | None,
    model: str = "ChatGPT",
    face: str = "_original_",
    options: RuntimeOptions | None = None,
) -> dict[str, Any]:
    del model, face
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower() or ".png"
    content_type = _content_type_for_ext(ext)

    notion_ctx = get_notion_context(options)
    headers = notion_ctx.headers
    account_name = notion_ctx.account_name
    create_upload_res = await create_upload_img(
        session, file_path, headers=headers, content_type=content_type
    )
    send_upload_res = await send_upload_img(
        session,
        create_upload_res["id"],
        file_path,
        headers=headers,
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


class NotionClientImpl:
    """Concrete implementation of NotionClient protocol."""

    def __init__(self, options: RuntimeOptions | None = None):
        self._options = options

    def get_headers(self) -> dict[str, str]:
        return get_headers(self._options)

    async def is_page_exists_in_db(
        self,
        session: aiohttp.ClientSession,
        db_id: str,
        query: str,
    ) -> bool:
        return await is_page_exists_in_db(session, db_id, query, options=self._options)

    async def add_page_to_db(
        self,
        session: aiohttp.ClientSession,
        db_id: str,
        file_path: str,
        prompt: str | None,
    ) -> dict[str, Any]:
        return await add_page_to_db(
            session, db_id, file_path, prompt, options=self._options
        )
