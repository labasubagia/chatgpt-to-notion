import asyncio
import os
from collections.abc import Sequence
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

from models import ImageGeneration, RuntimeOptions
from util import (
    MAX_CONCURRENT_REQUESTS,
    get_http_timeout,
    get_notion_context,
    get_output_path,
    retry_http,
)

BASE_URL = "https://api.notion.com"

DB_ID: str | None = None

# Cache for database data sources
_db_data_sources_cache: dict[str, Any] = {}
_db_page_cache: set[str] = set()


def get_headers(options: RuntimeOptions | None = None) -> dict[str, str]:
    """Get headers for Notion API requests"""
    return get_notion_context(options).headers


@retry_http()
async def get_db_data_sources(
    session: aiohttp.ClientSession,
    db_id: str,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if db_id in _db_data_sources_cache:
        return _db_data_sources_cache[db_id]

    async with session.get(
        f"{BASE_URL}/v1/databases/{db_id}", headers=headers or get_headers()
    ) as response:
        response.raise_for_status()
        data = await response.json()
        data_sources = data.get("data_sources", [])
        _db_data_sources_cache[db_id] = data_sources
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
        response.raise_for_status()
        return await response.json()


async def is_page_exists_in_db(
    session: aiohttp.ClientSession,
    db_id: str,
    query: str,
    options: RuntimeOptions | None = None,
) -> bool:
    headers = get_headers(options)
    if query in _db_page_cache:
        return True
    data_sources = await get_db_data_sources(session, db_id, headers=headers)
    for data_source in data_sources:
        data = await query_data_source(
            session, data_source["id"], query, headers=headers
        )
        for page in data.get("results", []):
            name = page["properties"]["Name"]["title"][0]["text"]["content"]
            if name == query:
                _db_page_cache.add(query)
                return True
    return False


@retry_http()
async def create_upload_img(
    session: aiohttp.ClientSession,
    file_path: str,
    headers: dict[str, str] | None = None,
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
            "content_type": "image/png",
        },
    ) as response:
        response.raise_for_status()
        return await response.json()


@retry_http()
async def send_upload_img(
    session: aiohttp.ClientSession,
    file_upload_id: str,
    file_path: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)

    notion_headers = headers or get_headers()
    with open(file_path, "rb") as f:
        data = aiohttp.FormData()
        data.add_field("file", f, filename=file_name, content_type="image/png")
        async with session.post(
            f"{BASE_URL}/v1/file_uploads/{file_upload_id}/send",
            headers={
                "Authorization": notion_headers["Authorization"],
                "Notion-Version": notion_headers["Notion-Version"],
            },
            data=data,
        ) as response:
            response.raise_for_status()
            return await response.json()


@retry_http()
async def add_page_to_db(
    session: aiohttp.ClientSession,
    db_id: str,
    file_path: str,
    prompt: str | None,
    model: str = "Sora",
    face: str = "_original_",
    options: RuntimeOptions | None = None,
) -> dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)

    headers = get_headers(options)
    create_upload_res = await create_upload_img(session, file_path, headers=headers)
    send_upload_res = await send_upload_img(
        session, create_upload_res["id"], file_path, headers=headers
    )

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
    }
    prompt = str(prompt).strip()
    payload["markdown"] = f"""
**Prompt:**

```
{prompt}
```

""".strip()

    async with session.post(
        f"{BASE_URL}/v1/pages",
        headers=headers,
        json=payload,
    ) as response:
        response.raise_for_status()
        _db_page_cache.add(file_name)
        return await response.json()


async def upload_all_images_to_notion(
    generations: Sequence[ImageGeneration],
    db_id: str,
    image_folder: str,
    options: RuntimeOptions | None = None,
) -> None:
    total = len(generations)
    pbar = tqdm(total=total, desc="Uploading to Notion")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(timeout=get_http_timeout()) as session:

        async def upload(generation_id: str, prompt: str | None):
            async with semaphore:
                file_name = f"{generation_id}.png"
                file_path = get_output_path(os.path.join(image_folder, file_name))
                if not os.path.exists(file_path):
                    pbar.write(f"⚠️  {file_name} not found, skipped")
                    pbar.update(1)
                    return

                try:
                    if await is_page_exists_in_db(
                        session, db_id, file_name, options=options
                    ):
                        pbar.write(f"⏭️  {file_name} skipped, already exists")
                    else:
                        await add_page_to_db(
                            session,
                            db_id,
                            file_path,
                            prompt,
                            model="Sora",
                            options=options,
                        )
                        pbar.write(f"✅ {file_name} uploaded")
                except Exception as e:
                    pbar.write(f"❌ {file_name} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(*[upload(row.id, row.prompt) for row in generations])

    pbar.close()
    print()  # Add spacing after progress bar
