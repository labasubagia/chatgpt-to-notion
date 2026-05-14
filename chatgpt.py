import asyncio
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

from img import add_prompt_to_images
from models import ChatGPTImageGeneration, RuntimeOptions
from notion import is_page_exists_in_db, upload_all_images_to_notion
from util import (
    MAX_CONCURRENT_DOWNLOADS,
    MAX_CONCURRENT_REQUESTS,
    download_image,
    get_http_timeout,
    get_output_path,
    get_provider_context,
    retry_http,
    save_to_dataset,
)

BASE_URL = "https://chatgpt.com/backend-api"


def get_headers(options: RuntimeOptions | None = None) -> dict[str, str]:
    """Get headers for ChatGPT API requests"""
    return get_provider_context("chatgpt", options).headers


@retry_http()
async def get_conversations(
    session: aiohttp.ClientSession,
    headers: dict[str, str] | None = None,
    offset: int = 0,
    limit: int = 100,
    is_archived: bool = False,
    is_starred: bool = False,
    order: str = "updated",
) -> dict[str, Any]:
    async with session.get(
        f"{BASE_URL}/conversations",
        headers=headers or get_headers(),
        params={
            "offset": offset,
            "limit": limit,
            "order": order,
            "is_archived": str(is_archived).lower(),
            "is_starred": str(is_starred).lower(),
        },
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return data


@retry_http()
async def get_conversation_details(
    session: aiohttp.ClientSession,
    conversation_id: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    async with session.get(
        f"{BASE_URL}/conversation/{conversation_id}",
        headers=headers or get_headers(),
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return data


@retry_http()
async def delete_conversation(
    session: aiohttp.ClientSession,
    conversation_id: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    async with session.patch(
        f"{BASE_URL}/conversation/{conversation_id}",
        headers=headers or get_headers(),
        json={"is_visible": False},
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return data


@retry_http()
async def get_image_generations(
    session: aiohttp.ClientSession,
    headers: dict[str, str] | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    async with session.get(
        f"{BASE_URL}/my/recent/image_gen",
        headers=headers or get_headers(),
        params={"limit": limit},
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return data


def get_conversation_mapping_key_by_asset_pointer(
    data: dict[str, Any], asset_pointer: str
) -> str | None:
    for key, node in data.get("mapping", {}).items():
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict) and part.get("asset_pointer") == asset_pointer:
                return key
    return None


def get_prompt_from_image_node_in_conversation(
    data: dict[str, Any], start_node_id: str, asset_pointer: str
) -> str | None:
    mapping = data["mapping"]
    current_id: str | None = start_node_id
    if current_id not in mapping:
        current_id = get_conversation_mapping_key_by_asset_pointer(data, asset_pointer)
        if current_id is None:
            return None

    while current_id:
        node = mapping[current_id]
        msg = node.get("message")

        if msg:
            author = msg.get("author", {}).get("role")
            if author == "user":
                inputs = msg.get("content", {}).get("parts", [])
                for part in inputs:
                    if isinstance(part, str):
                        return part
        current_id = node.get("parent")

    return None


async def fetch_image_generations(
    limit: int = 100,
    options: RuntimeOptions | None = None,
) -> list[ChatGPTImageGeneration]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    headers = get_headers(options)

    async with aiohttp.ClientSession(timeout=get_http_timeout()) as session:
        data = await get_image_generations(session, headers=headers, limit=limit)

        total = len(data.get("items", []))
        pbar = tqdm(total=total, desc="Fetching generation details")

        async def fetch_generation_details(
            img_gen: dict[str, Any],
        ) -> ChatGPTImageGeneration | None:
            async with semaphore:
                try:
                    detail = await get_conversation_details(
                        session, img_gen["conversation_id"], headers=headers
                    )
                    prompt = get_prompt_from_image_node_in_conversation(
                        detail, img_gen["message_id"], img_gen["asset_pointer"]
                    )
                    created_at = datetime.fromtimestamp(
                        img_gen["created_at"], tz=timezone.utc
                    ).isoformat(timespec="microseconds")
                    pbar.write(f"✅ img ID {img_gen['id']}")
                    return ChatGPTImageGeneration(
                        created_at=created_at,
                        id=img_gen["id"],
                        conversation_id=img_gen["conversation_id"],
                        message_id=img_gen["message_id"],
                        asset_pointer=img_gen["asset_pointer"],
                        url=img_gen["url"],
                        prompt=prompt or "",
                    )
                except Exception as e:
                    pbar.write(f"❌ img ID {img_gen['id']} failed: {e}")
                    return None
                finally:
                    pbar.update(1)

        results = await asyncio.gather(
            *[fetch_generation_details(img_gen) for img_gen in data.get("items", [])],
            return_exceptions=True,
        )

        pbar.close()
        print()

        # Filter out None and exceptions
        valid_generations = [
            g
            for g in results
            if g is not None
            and not isinstance(g, Exception)
            and isinstance(g, ChatGPTImageGeneration)
        ]
        return sorted(valid_generations, key=lambda x: x.created_at)


async def download_all_images(
    generations: list[ChatGPTImageGeneration],
    download_folder: str,
    options: RuntimeOptions | None = None,
) -> None:
    total = len(generations)
    pbar = tqdm(total=total, desc="Downloading images")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async with aiohttp.ClientSession(
        headers=get_headers(options), timeout=get_http_timeout()
    ) as session:

        async def download(row: ChatGPTImageGeneration):
            async with semaphore:
                file_name = f"{row.id}.png"
                file_path = get_output_path(os.path.join(download_folder, file_name))

                if os.path.exists(file_path):
                    pbar.write(f"⏭️  {file_name} skipped, already exists")
                    pbar.update(1)
                    return

                try:
                    await download_image(
                        session, row.url, str(file_path), headers=get_headers(options)
                    )
                    pbar.write(f"✅ {file_name}")
                except Exception as e:
                    pbar.write(f"❌ {file_name} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(*[download(row) for row in generations])

    pbar.close()
    print()


async def delete_conversation_of_image_generation_uploaded_to_notion(
    generations: list[ChatGPTImageGeneration],
    db_id: str,
    options: RuntimeOptions | None = None,
) -> None:
    generations = list({gen.conversation_id: gen for gen in generations}.values())
    total = len(generations)
    pbar = tqdm(total=total, desc="Deleting conversations")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    conversation_map = defaultdict(set)
    for gen in generations:
        conversation_map[gen.conversation_id].add(gen.id)

    async with aiohttp.ClientSession(
        headers=get_headers(options), timeout=get_http_timeout()
    ) as session:

        async def delete(row: ChatGPTImageGeneration):
            async with semaphore:
                file_name = f"{row.id}.png"
                conversation_id = row.conversation_id

                try:
                    exists = await is_page_exists_in_db(
                        session, db_id, file_name, options=options
                    )
                    if not exists:
                        pbar.write(f"⏭️  {file_name} not found in Notion, skipped")
                        pbar.update(1)
                        return

                    conversation_map[conversation_id].discard(row.id)
                    if conversation_map[conversation_id]:
                        pbar.write(
                            f"⏭️  Conversation ID {conversation_id} skipped, still has pending images"  # noqa: E501
                        )
                        pbar.update(1)
                        return

                    await delete_conversation(
                        session, conversation_id, headers=get_headers(options)
                    )
                    pbar.write(f"✅ Conversation ID {conversation_id}")
                except Exception as e:
                    pbar.write(f"❌ Conversation ID {conversation_id} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(*[delete(row) for row in generations])

    pbar.close()
    print()


async def upload_to_notion(
    image_folder: str,
    db_id: str,
    upload_to_notion: bool = True,
    remove_in_chatgpt: bool = False,
    add_prompt_to_image: bool = True,
    dataset: str | None = None,
    check_notion_api: bool = False,
    limit: int = 100,
    options: RuntimeOptions | None = None,
) -> None:
    generations = await fetch_image_generations(limit=limit, options=options)

    if not generations:
        print("No generations found.")
        return

    if dataset:
        save_to_dataset(dataset=dataset, data=generations)

    await download_all_images(
        generations=generations, download_folder=image_folder, options=options
    )

    if add_prompt_to_image:
        add_prompt_to_images(generations=generations, folder=image_folder)

    if upload_to_notion:
        await upload_all_images_to_notion(
            generations=generations,
            db_id=db_id,
            image_folder=image_folder,
            dataset=dataset,
            check_notion_api=check_notion_api,
            options=options,
        )

    if remove_in_chatgpt:
        await delete_conversation_of_image_generation_uploaded_to_notion(
            generations=generations, db_id=db_id, options=options
        )
