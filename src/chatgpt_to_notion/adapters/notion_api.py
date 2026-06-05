"""Notion API adapter."""

import asyncio
import mimetypes
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

from ..domain.models import ImageGeneration, RuntimeOptions
from ..shared.constants import MAX_CONCURRENT_REQUESTS, image_ext_from_url
from ..shared.http import (
    DetailedHTTPError,
    exc_detail,
    get_http_timeout,
    raise_for_status_with_detail,
    retry_http,
)
from ..shared.logging import get_logger
from ..shared.verbosity import StageCounter, is_verbose, write_fail_log
from .config_loader import get_notion_context
from .filesystem import get_output_path
from .sqlite_store import (
    async_set_cached_data_sources,
    get_cached_data_sources,
    get_uploaded_ids,
    mark_uploaded,
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

    headers = get_headers(options)
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
        "markdown": f"**Prompt:**\n\n```\n{normalized_prompt}\n```",
    }

    async with session.post(
        f"{BASE_URL}/v1/pages",
        headers=headers,
        json=payload,
    ) as response:
        await raise_for_status_with_detail(response)
        return await response.json()


async def upload_all_images_to_notion(
    generations: Sequence[ImageGeneration],
    db_id: str,
    image_folder: str,
    account: str | None = None,
    check_notion_api: bool = False,
    options: RuntimeOptions | None = None,
    fail_log_path: Path | None = None,
) -> None:
    total = len(generations)
    pbar = tqdm(total=total, desc="Uploading to Notion")
    counter = StageCounter("Uploaded")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    uploaded_generation_ids = get_uploaded_ids(account) if account else set()

    image_folder_path = Path(image_folder)
    if not image_folder_path.is_absolute():
        image_folder_path = get_output_path(image_folder)

    async with aiohttp.ClientSession(timeout=get_http_timeout()) as session:

        async def upload(
            generation_id: str,
            prompt: str | None,
            generation_url: str = "",
        ) -> str | None:
            async with semaphore:
                ext = image_ext_from_url(generation_url)
                file_name = f"{generation_id}{ext}"
                if generation_id in uploaded_generation_ids and not check_notion_api:
                    counter.add("skipped")
                    if is_verbose():
                        pbar.write(f"⏭️  {file_name} skipped, marked uploaded")
                    pbar.update(1)
                    return None

                file_path = image_folder_path / file_name
                if not os.path.exists(file_path):
                    counter.add("skipped")
                    if is_verbose():
                        pbar.write(f"⚠️  {file_name} not found, skipped")
                    pbar.update(1)
                    return None

                try:
                    if await is_page_exists_in_db(
                        session, db_id, file_name, options=options
                    ):
                        counter.add("skipped")
                        if is_verbose():
                            pbar.write(f"⏭️  {file_name} skipped, already exists")
                        return generation_id
                    await add_page_to_db(
                        session,
                        db_id,
                        str(file_path),
                        prompt,
                        model="ChatGPT",
                        options=options,
                    )
                    counter.add("success")
                    if is_verbose():
                        pbar.write(f"✅ {file_name} uploaded")
                    return generation_id
                except Exception as exc:
                    counter.add("failed")
                    if is_verbose():
                        pbar.write(f"❌ {file_name} failed: {exc}")
                    detail = exc_detail(exc)
                    if is_verbose():
                        logger.exception(
                            "Failed to upload %s\n%s",
                            file_name,
                            detail,
                        )
                    else:
                        logger.error("Failed to upload %s: %s", file_name, detail)
                    if fail_log_path:
                        write_fail_log(
                            fail_log_path,
                            {"stage": "upload", "file": file_name, "error": str(exc)},
                        )
                    return None
                finally:
                    pbar.update(1)

        uploaded_results: list[str | None] = []
        try:
            uploaded_results = await asyncio.gather(
                *[
                    upload(row.id, row.prompt, getattr(row, "url", ""))
                    for row in generations
                ]
            )
        finally:
            pbar.close()
            if not is_verbose():
                print(counter.summary_line())
            if account:
                successful_ids = {
                    generation_id for generation_id in uploaded_results if generation_id
                }
                if successful_ids:
                    mark_uploaded(account, successful_ids)
    print()
