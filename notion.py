import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

from models import ImageGeneration, RuntimeOptions
from util import (
    MAX_CONCURRENT_REQUESTS,
    get_http_timeout,
    get_notion_context,
    get_output_path,
    get_uploaded_generation_ids,
    mark_generations_uploaded,
    retry_http,
)

BASE_URL = "https://api.notion.com"

DB_ID: str | None = None


def get_headers(options: RuntimeOptions | None = None) -> dict[str, str]:
    """Get headers for Notion API requests"""
    return get_notion_context(options).headers


@retry_http()
async def get_db_data_sources(
    session: aiohttp.ClientSession,
    db_id: str,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    from db import async_set_cached_data_sources, get_cached_data_sources

    cached = get_cached_data_sources(db_id)
    if cached is not None:
        return cached

    async with session.get(
        f"{BASE_URL}/v1/databases/{db_id}", headers=headers or get_headers()
    ) as response:
        response.raise_for_status()
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
        response.raise_for_status()
        return await response.json()


async def is_page_exists_in_db(
    session: aiohttp.ClientSession,
    db_id: str,
    query: str,
    options: RuntimeOptions | None = None,
) -> bool:
    headers = get_headers(options)
    data_sources = await get_db_data_sources(session, db_id, headers=headers)
    for data_source in data_sources:
        data = await query_data_source(
            session, data_source["id"], query, headers=headers
        )
        for page in data.get("results", []):
            name = page["properties"]["Name"]["title"][0]["text"]["content"]
            if name == query:
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
    model: str = "ChatGPT",
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

    # Normalize broken / mixed line endings
    prompt = prompt.replace("\r \n", "\n")
    prompt = prompt.replace("\r\n", "\n")
    prompt = prompt.replace("\r", "\n")

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
        code = response.status
        if code >= 400:
            text = await response.text()
            try:
                data = json.loads(text)
                print(json.dumps(data, indent=2))
            except Exception:
                pass
            print(f"Failed to add page for {file_name}: {code} - {text}")
            print(json.dumps(payload, indent=2))
        response.raise_for_status()
        return await response.json()


async def upload_all_images_to_notion(
    generations: Sequence[ImageGeneration],
    db_id: str,
    image_folder: str,
    account: str | None = None,
    check_notion_api: bool = False,
    options: RuntimeOptions | None = None,
) -> None:
    total = len(generations)
    pbar = tqdm(total=total, desc="Uploading to Notion")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    uploaded_generation_ids = get_uploaded_generation_ids(account, options)

    image_folder_path = Path(image_folder)
    if not image_folder_path.is_absolute():
        image_folder_path = get_output_path(image_folder)

    async with aiohttp.ClientSession(timeout=get_http_timeout()) as session:

        async def upload(generation_id: str, prompt: str | None) -> str | None:
            async with semaphore:
                file_name = f"{generation_id}.png"
                if generation_id in uploaded_generation_ids and not check_notion_api:
                    pbar.write(f"⏭️  {file_name} skipped, marked uploaded")
                    pbar.update(1)
                    return None

                file_path = image_folder_path / file_name
                if not os.path.exists(file_path):
                    pbar.write(f"⚠️  {file_name} not found, skipped")
                    pbar.update(1)
                    return None

                try:
                    if await is_page_exists_in_db(
                        session, db_id, file_name, options=options
                    ):
                        pbar.write(f"⏭️  {file_name} skipped, already exists")
                        return generation_id
                    else:
                        await add_page_to_db(
                            session,
                            db_id,
                            file_path,
                            prompt,
                            model="ChatGPT",
                            options=options,
                        )
                        pbar.write(f"✅ {file_name} uploaded")
                        return generation_id
                except Exception as e:
                    pbar.write(f"❌ {file_name} failed: {e}")
                    return None
                finally:
                    pbar.update(1)

        uploaded_results = await asyncio.gather(
            *[upload(row.id, row.prompt) for row in generations]
        )

    pbar.close()
    mark_generations_uploaded(
        account,
        {generation_id for generation_id in uploaded_results if generation_id},
        options,
    )
    print()  # Add spacing after progress bar
