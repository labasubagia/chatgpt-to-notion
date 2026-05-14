import asyncio
import os
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

from img import add_prompt_to_images
from models import RuntimeOptions, SoraImageGeneration
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

BASE_URL = "https://sora.chatgpt.com/backend"


def get_headers(options: RuntimeOptions | None = None) -> dict[str, str]:
    """Get headers for Sora API requests.

    Sora uses the same authentication as ChatGPT, so we delegate to chatgpt.get_headers.
    """
    return get_provider_context("sora", options).headers


@retry_http()
async def archive_generation(
    session: aiohttp.ClientSession,
    generation_id: str,
    headers: dict[str, str] | None = None,
    is_archived: bool = True,
) -> dict[str, Any]:
    """
    Trash a generation in Sora.

    is_archived=True means trash, is_archived=False means untrash/restore
    """
    async with session.post(
        f"{BASE_URL}/generations/{generation_id}",
        json={"is_archived": is_archived},
        headers=headers or get_headers(),
    ) as response:
        response.raise_for_status()
        json_data = await response.json()
        return json_data


@retry_http()
async def archive_task(
    session: aiohttp.ClientSession,
    task_id: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Trash a task in Sora
    """
    async with session.post(
        f"{BASE_URL}/video_gen/{task_id}/archive", headers=headers or get_headers()
    ) as response:
        response.raise_for_status()
        json_data = await response.json()
        return json_data


@retry_http()
async def delete_task(
    session: aiohttp.ClientSession,
    task_id: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    async with session.delete(
        f"{BASE_URL}/video_gen/{task_id}", headers=headers or get_headers()
    ) as response:
        response.raise_for_status()
        json_data = await response.json()
        return json_data


@retry_http()
async def fetch_recent_tasks(
    session: aiohttp.ClientSession,
    headers: dict[str, str] | None = None,
    limit: int = 100,
    before_task_id: str | None = None,
    archived: bool = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if before_task_id:
        params["before"] = before_task_id
    if archived:
        params["archived"] = "true"

    async with session.get(
        f"{BASE_URL}/v2/recent_tasks", params=params, headers=headers or get_headers()
    ) as response:
        response.raise_for_status()
        data_json = await response.json()
        return data_json


@retry_http()
async def fetch_list_tasks(
    session: aiohttp.ClientSession,
    headers: dict[str, str] | None = None,
    limit: int = 20,
    after_task_id: str | None = None,
    archived: bool = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if archived:
        params["archived"] = "true"
    if after_task_id:
        params["after"] = after_task_id

    async with session.get(
        f"{BASE_URL}/v2/list_tasks", params=params, headers=headers or get_headers()
    ) as response:
        response.raise_for_status()
        json_data = await response.json()
        return json_data


async def fetch_all_lists_tasks(
    limit: int = 20,
    archived: bool = False,
    options: RuntimeOptions | None = None,
) -> list[dict[str, Any]]:
    batch_count = 1
    all_tasks: list[dict[str, Any]] = []
    has_more = True
    last_id: str | None = None

    headers = get_headers(options)
    async with aiohttp.ClientSession(
        headers=headers, timeout=get_http_timeout()
    ) as session:
        while has_more:
            try:
                data = await fetch_list_tasks(
                    session,
                    headers=headers,
                    limit=limit,
                    after_task_id=last_id,
                    archived=archived,
                )
                last_id = data.get("last_id", None)
                has_more = data.get("has_more", False)
                tasks = data.get("task_responses", [])
                all_tasks.extend(tasks)
                print(
                    f"Batch {batch_count} fetched, "
                    f"last_id {last_id}, has_more: {has_more}"
                )
                batch_count += 1
            except Exception as e:
                print(f"Batch {batch_count} fetch error: {e}")
                break

    return all_tasks


async def delete_empty_tasks(options: RuntimeOptions | None = None) -> None:
    empty_tasks: list[str] = []
    tasks = await fetch_all_lists_tasks(limit=100, archived=False, options=options)
    for task in tasks:
        if len(task.get("generations", [])) == 0:
            task_id = task.get("id")
            if task_id:
                empty_tasks.append(task_id)

    total = len(empty_tasks)
    pbar = tqdm(total=total, desc="Deleting empty tasks")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(
        headers=get_headers(options), timeout=get_http_timeout()
    ) as session:

        async def delete(task_id: str):
            async with semaphore:
                try:
                    headers = get_headers(options)
                    await archive_task(session, task_id, headers=headers)
                    await delete_task(session, task_id, headers=headers)
                    pbar.write(f"✅ {task_id}")
                except Exception as e:
                    pbar.write(f"❌ task {task_id} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(
            *[delete(task_id) for task_id in empty_tasks], return_exceptions=True
        )

    pbar.close()
    print()


def get_generations_from_tasks(
    tasks: list[dict[str, Any]],
) -> list[SoraImageGeneration]:
    generations: list[SoraImageGeneration] = []
    for task in tasks:
        for generation in task.get("generations", []):
            generations.append(
                SoraImageGeneration(
                    created_at=task.get("created_at"),
                    id=generation.get("id"),
                    task_id=generation.get("task_id"),
                    url=generation.get("url"),
                    prompt=generation.get("prompt") or "",
                )
            )
    return sorted(generations, key=lambda x: x.created_at or "")


@retry_http()
async def get_generation_download_url(
    session: aiohttp.ClientSession,
    generation_id: str,
    headers: dict[str, str] | None = None,
) -> str | None:
    async with session.get(
        f"{BASE_URL}/generations/{generation_id}/download",
        headers=headers or get_headers(),
    ) as response:
        response.raise_for_status()
        json_data = await response.json()
        return json_data.get("url", None)


async def download_all_images(
    generations: list[SoraImageGeneration],
    download_folder: str = "sora_images",
    options: RuntimeOptions | None = None,
) -> None:
    total = len(generations)
    pbar = tqdm(total=total, desc="Downloading images")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async with aiohttp.ClientSession(timeout=get_http_timeout()) as session:

        async def download(generation: SoraImageGeneration):
            async with semaphore:
                generation_id = generation.id
                url = generation.url
                file_name = f"{generation_id}.png"
                file_path = get_output_path(os.path.join(download_folder, file_name))

                if os.path.exists(file_path):
                    pbar.write(f"⏭️  {file_name} skipped, already exists")
                    pbar.update(1)
                    return

                try:
                    await download_image(session, url or "", str(file_path))
                    pbar.write(f"✅ {file_name}")
                except Exception as e:
                    pbar.write(f"❌ {file_name} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(*[download(gen) for gen in generations])

    pbar.close()
    print()


@retry_http()
async def delete_generation(
    session: aiohttp.ClientSession,
    id: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    async with session.delete(
        f"{BASE_URL}/generations/{id}", headers=headers or get_headers()
    ) as response:
        response.raise_for_status()
        json_data = await response.json()
        return json_data


async def delete_generations(
    generations: list[SoraImageGeneration], options: RuntimeOptions | None = None
) -> None:
    total = len(generations)
    pbar = tqdm(total=total, desc="Deleting generations")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    headers = get_headers(options)
    async with aiohttp.ClientSession(
        headers=headers, timeout=get_http_timeout()
    ) as session:

        async def delete(generation: SoraImageGeneration):
            async with semaphore:
                generation_id = generation.id
                try:
                    await delete_generation(session, generation_id, headers=headers)
                    pbar.write(f"✅ {generation_id}")
                except Exception as e:
                    pbar.write(f"❌ {generation_id} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(*[delete(gen) for gen in generations])

    pbar.close()
    print()


async def delete_generations_already_uploaded_to_notion(
    generations: list[SoraImageGeneration],
    db_id: str,
    options: RuntimeOptions | None = None,
) -> None:
    total = len(generations)
    pbar = tqdm(total=total, desc="Deleting uploaded generations")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    headers = get_headers(options)
    async with aiohttp.ClientSession(timeout=get_http_timeout()) as session:

        async def delete(generation: SoraImageGeneration):
            async with semaphore:
                generation_id = generation.id
                file_name = f"{generation_id}.png"

                try:
                    if await is_page_exists_in_db(
                        session, db_id, file_name, options=options
                    ):
                        await delete_generation(session, generation_id, headers=headers)
                        pbar.write(f"✅ {generation_id} deleted")
                    else:
                        pbar.write(
                            f"⏭️  {generation_id} skipped, not uploaded to notion yet"
                        )
                except Exception as e:
                    pbar.write(f"❌ {generation_id} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(*[delete(gen) for gen in generations])

    pbar.close()
    print()


async def trash_generations_already_uploaded_to_notion(
    generations: list[SoraImageGeneration],
    db_id: str,
    options: RuntimeOptions | None = None,
) -> None:
    total = len(generations)
    pbar = tqdm(total=total, desc="Trashing uploaded generations")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    headers = get_headers(options)
    async with aiohttp.ClientSession(timeout=get_http_timeout()) as session:

        async def trash(generation: SoraImageGeneration):
            async with semaphore:
                generation_id = generation.id
                file_name = f"{generation_id}.png"

                try:
                    if await is_page_exists_in_db(
                        session, db_id, file_name, options=options
                    ):
                        await archive_generation(
                            session, generation_id, headers=headers
                        )
                        pbar.write(f"✅ {generation_id}")
                    else:
                        pbar.write(
                            f"⏭️  {generation_id} skipped, not uploaded to notion yet"
                        )
                except Exception as e:
                    pbar.write(f"❌ {generation_id} failed: {e}")
                finally:
                    pbar.update(1)

        await asyncio.gather(*[trash(gen) for gen in generations])

    pbar.close()
    print()


async def upload_to_notion(
    image_folder: str,
    db_id: str,
    upload_to_notion: bool = True,
    trash_in_sora: bool = False,
    remove_in_sora: bool = False,
    add_prompt_to_image: bool = True,
    limit: int = 100,
    dataset: str | None = None,
    check_notion_api: bool = False,
    options: RuntimeOptions | None = None,
) -> None:
    if trash_in_sora and remove_in_sora:
        raise ValueError("trash_in_sora and remove_in_sora cannot be both True.")

    async with aiohttp.ClientSession(timeout=get_http_timeout()) as session:
        data = await fetch_recent_tasks(
            session, headers=get_headers(options), limit=limit, archived=False
        )

    tasks = data.get("task_responses", [])
    generations = get_generations_from_tasks(tasks)

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

    if trash_in_sora:
        await trash_generations_already_uploaded_to_notion(
            generations=generations, db_id=db_id, options=options
        )

    if remove_in_sora:
        await delete_generations_already_uploaded_to_notion(
            generations=generations, db_id=db_id, options=options
        )


async def cleanup_trash(
    task_limit: int = 100,
    dataset: str | None = None,
    options: RuntimeOptions | None = None,
) -> None:
    tasks = await fetch_all_lists_tasks(
        limit=task_limit, archived=True, options=options
    )
    generations = get_generations_from_tasks(tasks)

    if dataset:
        save_to_dataset(dataset=dataset, data=generations)

    await delete_generations(generations=generations, options=options)


async def cleanup_tasks(options: RuntimeOptions | None = None) -> None:
    await delete_empty_tasks(options=options)
