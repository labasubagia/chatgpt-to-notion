import asyncio
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

import db
from img import add_prompt_to_image_single, add_prompt_to_images
from models import ChatGPTImageGeneration, RuntimeOptions
from notion import add_page_to_db, is_page_exists_in_db, upload_all_images_to_notion
from util import (
    MAX_CONCURRENT_DOWNLOADS,
    MAX_CONCURRENT_REQUESTS,
    download_image,
    get_http_timeout,
    get_provider_context,
    get_uploaded_generation_ids,
    mark_generations_uploaded,
    resolve_image_folder,
    retry_http,
    save_generations,
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
    no_cache: bool = False,
) -> list[ChatGPTImageGeneration]:
    account = options.account if options else None
    if not account:
        account = "default"

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    headers = get_headers(options)

    async with aiohttp.ClientSession(timeout=get_http_timeout()) as session:
        data = await get_image_generations(session, headers=headers, limit=limit)

        items = data.get("items", [])
        api_ids = {item["id"] for item in items}

        existing = db.get_generations(
            account, include_uploaded=True, ids_filter=api_ids
        )
        existing_map = {g.id: g for g in existing}

        new_items = [item for item in items if item["id"] not in existing_map]

        if new_items:
            pbar = tqdm(total=len(new_items), desc="Fetching new generation details")

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
                *[fetch_generation_details(item) for item in new_items],
                return_exceptions=True,
            )

            pbar.close()
            print()

            new_generations = [
                g
                for g in results
                if g is not None
                and not isinstance(g, Exception)
                and isinstance(g, ChatGPTImageGeneration)
            ]

            await db.async_upsert_generations(account, new_generations)
            for g in new_generations:
                existing_map[g.id] = g

        result = sorted(existing_map.values(), key=lambda x: x.created_at)
        return result


def load_image_generations(
    account: str,
    include_uploaded: bool = False,
    keep_days: int | None = None,
    timezone_name: str | None = None,
    options: RuntimeOptions | None = None,
) -> list[ChatGPTImageGeneration]:
    return db.get_generations(
        account,
        include_uploaded=include_uploaded,
        keep_days=keep_days,
        timezone_name=timezone_name,
    )


async def download_all_images(
    generations: list[ChatGPTImageGeneration],
    download_folder: str,
    options: RuntimeOptions | None = None,
) -> None:
    total = len(generations)
    pbar = tqdm(total=total, desc="Downloading images")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    download_path = Path(download_folder)

    async with aiohttp.ClientSession(
        headers=get_headers(options), timeout=get_http_timeout()
    ) as session:

        async def download(row: ChatGPTImageGeneration):
            async with semaphore:
                file_name = f"{row.id}.png"
                file_path = download_path / file_name

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
    conversation_map: dict[str, set[str]] = defaultdict(set)
    for gen in generations:
        conversation_map[gen.conversation_id].add(gen.id)

    total = len(conversation_map)
    pbar = tqdm(total=total, desc="Deleting conversations")

    async with aiohttp.ClientSession(
        headers=get_headers(options), timeout=get_http_timeout()
    ) as session:

        async def delete_conversation_by_id(conversation_id: str, image_ids: set[str]):
            try:
                for img_id in image_ids:
                    file_name = f"{img_id}.png"
                    exists = await is_page_exists_in_db(
                        session, db_id, file_name, options=options
                    )
                    if not exists:
                        pbar.write(f"⏭️  {file_name} not found in Notion, skipped")
                        return

                await delete_conversation(
                    session, conversation_id, headers=get_headers(options)
                )
                pbar.write(f"✅ Conversation ID {conversation_id}")
            except Exception as e:
                pbar.write(f"❌ Conversation ID {conversation_id} failed: {e}")
            finally:
                pbar.update(1)

        await asyncio.gather(
            *[
                delete_conversation_by_id(cid, ids)
                for cid, ids in conversation_map.items()
            ]
        )

    pbar.close()
    print()


async def upload_to_notion(
    image_folder: str | None,
    db_id: str,
    upload_to_notion: bool = True,
    remove_in_chatgpt: bool = False,
    add_prompt_to_image: bool = True,
    account: str | None = None,
    check_notion_api: bool = False,
    from_history: bool = False,
    limit: int = 100,
    keep_days: int | None = None,
    timezone_name: str | None = None,
    no_cache: bool = False,
    options: RuntimeOptions | None = None,
) -> None:
    resolved_image_folder = resolve_image_folder(image_folder, options)

    db.init_db()

    if from_history:
        if not account:
            raise ValueError("account is required when from_history=True")
        generations = load_image_generations(
            account=account,
            include_uploaded=check_notion_api,
            keep_days=keep_days,
            timezone_name=timezone_name,
            options=options,
        )
    else:
        generations = await fetch_image_generations(
            limit=limit, options=options, no_cache=no_cache
        )

    if not generations:
        print("No generations found.")
        return

    if account and not from_history:
        save_generations(account=account, data=generations, options=options)

    await download_all_images(
        generations=generations,
        download_folder=str(resolved_image_folder),
        options=options,
    )

    if add_prompt_to_image:
        add_prompt_to_images(generations=generations, folder=str(resolved_image_folder))

    if upload_to_notion:
        await upload_all_images_to_notion(
            generations=generations,
            db_id=db_id,
            image_folder=str(resolved_image_folder),
            account=account,
            check_notion_api=check_notion_api,
            options=options,
        )

    if remove_in_chatgpt:
        await delete_conversation_of_image_generation_uploaded_to_notion(
            generations=generations, db_id=db_id, options=options
        )


async def delete_conversations_after_upload(
    generations: list[ChatGPTImageGeneration],
    db_id: str,
    account: str | None,
    uploaded_ids: set[str],
    options: RuntimeOptions | None = None,
) -> None:
    conversation_map: dict[str, set[str]] = defaultdict(set)
    for gen in generations:
        conversation_map[gen.conversation_id].add(gen.id)

    total = len(conversation_map)
    if total == 0:
        return
    pbar = tqdm(total=total, desc="Deleting conversations")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(
        headers=get_headers(options), timeout=get_http_timeout()
    ) as session:

        async def delete_conv(conv_id: str, image_ids: set[str]):
            async with semaphore:
                remaining = image_ids - uploaded_ids
                if remaining:
                    pbar.write(
                        f"⏭️  Conversation {conv_id} skipped, "
                        f"{len(remaining)} image(s) not uploaded"
                    )
                    pbar.update(1)
                    return

                try:
                    await delete_conversation(
                        session, conv_id, headers=get_headers(options)
                    )
                    pbar.write(f"✅ Conversation {conv_id}")
                except Exception as e:
                    pbar.write(f"❌ Conversation {conv_id} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(
            *[delete_conv(cid, ids) for cid, ids in conversation_map.items()]
        )

    pbar.close()
    print()


async def upload_to_notion_single(
    image_folder: str | None,
    db_id: str,
    upload_to_notion: bool = True,
    remove_in_chatgpt: bool = False,
    add_prompt_to_image: bool = True,
    account: str | None = None,
    check_notion_api: bool = False,
    from_history: bool = False,
    limit: int = 100,
    keep_days: int | None = None,
    timezone_name: str | None = None,
    no_cache: bool = False,
    options: RuntimeOptions | None = None,
) -> None:
    resolved_image_folder = resolve_image_folder(image_folder, options)

    db.init_db()

    if from_history:
        if not account:
            raise ValueError("account is required when from_history=True")
        generations = load_image_generations(
            account=account,
            include_uploaded=check_notion_api,
            keep_days=keep_days,
            timezone_name=timezone_name,
            options=options,
        )
    else:
        generations = await fetch_image_generations(
            limit=limit, options=options, no_cache=no_cache
        )

    if not generations:
        print("No generations found.")
        return

    if account and not from_history:
        save_generations(account=account, data=generations, options=options)

    uploaded_ids = get_uploaded_generation_ids(account, options)
    db_lock = asyncio.Lock()

    total = len(generations)
    pbar = tqdm(total=total, desc="Processing files")
    semaphore = asyncio.Semaphore(5)

    async with aiohttp.ClientSession(
        headers=get_headers(options), timeout=get_http_timeout()
    ) as chatgpt_session:
        async with aiohttp.ClientSession(timeout=get_http_timeout()) as notion_session:

            async def process_one(gen: ChatGPTImageGeneration):
                async with semaphore:
                    file_name = f"{gen.id}.png"
                    try:
                        if gen.id in uploaded_ids and not check_notion_api:
                            pbar.write(f"⏭️  {file_name} skipped, already uploaded")
                            return

                        if await is_page_exists_in_db(
                            notion_session, db_id, file_name, options=options
                        ):
                            uploaded_ids.add(gen.id)
                            pbar.write(f"⏭️  {file_name} skipped, already in Notion")
                            return

                        file_path = resolved_image_folder / file_name
                        if not file_path.exists():
                            await download_image(
                                chatgpt_session,
                                gen.url,
                                str(file_path),
                                headers=get_headers(options),
                            )

                        if add_prompt_to_image:
                            add_prompt_to_image_single(gen, str(resolved_image_folder))

                        if upload_to_notion:
                            await add_page_to_db(
                                notion_session,
                                db_id,
                                str(file_path),
                                gen.prompt,
                                model="ChatGPT",
                                options=options,
                            )

                            if account:
                                async with db_lock:
                                    mark_generations_uploaded(
                                        account, {gen.id}, options
                                    )
                            uploaded_ids.add(gen.id)

                        pbar.write(f"✅ {file_name}")
                    except Exception as e:
                        pbar.write(f"❌ {file_name} failed: {e}")
                    finally:
                        pbar.update(1)

            await asyncio.gather(*[process_one(gen) for gen in generations])

    pbar.close()
    print()

    if remove_in_chatgpt:
        await delete_conversations_after_upload(
            generations=generations,
            db_id=db_id,
            account=account,
            uploaded_ids=uploaded_ids,
            options=options,
        )
