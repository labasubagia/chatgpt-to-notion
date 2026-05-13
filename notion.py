import asyncio
import os
from collections.abc import Sequence
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

from models import ImageGeneration
from util import (
    MAX_CONCURRENT_REQUESTS,
    get_config,
    get_http_timeout,
    get_output_path,
    retry_http,
)

BASE_URL = "https://api.notion.com"

# Default database ID from config (can be overridden via CLI)
_config = get_config()
DB_ID = _config.get("NOTION_DATABASE_ID")

# Cache for database data sources
_db_data_sources_cache: dict[str, Any] = {}
_db_page_cache: set[str] = set()


def get_headers() -> dict[str, str]:
    """Get headers for Notion API requests"""
    config = get_config()
    return {
        "Authorization": f"Bearer {config.get('NOTION_API_KEY')}",
        "Notion-Version": "2025-09-03",
        "Content-Type": "application/json",
    }


@retry_http()
async def get_db_data_sources(
    session: aiohttp.ClientSession, db_id: str
) -> list[dict[str, Any]]:
    if db_id in _db_data_sources_cache:
        return _db_data_sources_cache[db_id]

    async with session.get(
        f"{BASE_URL}/v1/databases/{db_id}", headers=get_headers()
    ) as response:
        response.raise_for_status()
        data = await response.json()
        data_sources = data.get("data_sources", [])
        _db_data_sources_cache[db_id] = data_sources
        return data_sources


@retry_http()
async def query_data_source(
    session: aiohttp.ClientSession, data_source_id: str, query: str
) -> dict[str, Any]:
    async with session.post(
        f"{BASE_URL}/v1/data_sources/{data_source_id}/query",
        headers=get_headers(),
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
) -> bool:
    if query in _db_page_cache:
        return True
    data_sources = await get_db_data_sources(session, db_id)
    for data_source in data_sources:
        data = await query_data_source(session, data_source["id"], query)
        for page in data.get("results", []):
            name = page["properties"]["Name"]["title"][0]["text"]["content"]
            if name == query:
                _db_page_cache.add(query)
                return True
    return False


@retry_http()
async def create_upload_img(
    session: aiohttp.ClientSession, file_path: str
) -> dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)

    async with session.post(
        f"{BASE_URL}/v1/file_uploads",
        headers=get_headers(),
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
    session: aiohttp.ClientSession, file_upload_id: str, file_path: str
) -> dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        data = aiohttp.FormData()
        data.add_field("file", f, filename=file_name, content_type="image/png")
        async with session.post(
            f"{BASE_URL}/v1/file_uploads/{file_upload_id}/send",
            headers={
                "Authorization": get_headers()["Authorization"],
                "Notion-Version": get_headers()["Notion-Version"],
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
) -> dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    file_name = os.path.basename(file_path)

    create_upload_res = await create_upload_img(session, file_path)
    send_upload_res = await send_upload_img(session, create_upload_res["id"], file_path)

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
            "Prompt": {"rich_text": []},
            "Model": {"select": {"name": model}},
            "Face": {"select": {"name": face}},
        },
    }
    prompt = str(prompt).strip()

    if 0 > len(prompt) < 2000:
        payload["properties"]["Prompt"]["rich_text"].append(
            {"text": {"content": prompt}}
        )
    elif len(prompt) >= 2000:
        payload["properties"]["Prompt"]["rich_text"].append(
            {"text": {"content": "refer body"}}
        )
        payload["markdown"] = f"""
**Prompt:**

```
{prompt}
```

""".strip()
    else:
        payload["properties"]["Prompt"]["rich_text"].append(
            {"text": {"content": "N/A"}}
        )

    async with session.post(
        f"{BASE_URL}/v1/pages",
        headers=get_headers(),
        json=payload,
    ) as response:
        response.raise_for_status()
        _db_page_cache.add(file_name)
        return await response.json()


async def upload_all_images_to_notion(
    generations: Sequence[ImageGeneration], db_id: str, image_folder: str
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
                    if await is_page_exists_in_db(session, db_id, file_name):
                        pbar.write(f"⏭️  {file_name} skipped, already exists")
                    else:
                        await add_page_to_db(
                            session, db_id, file_path, prompt, model="Sora"
                        )
                        pbar.write(f"✅ {file_name} uploaded")
                except Exception as e:
                    pbar.write(f"❌ {file_name} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(*[upload(row.id, row.prompt) for row in generations])

    pbar.close()
    print()  # Add spacing after progress bar
